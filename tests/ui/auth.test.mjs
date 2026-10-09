import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source = readFileSync(new URL("../../src/soul/static/auth.js", import.meta.url), "utf8");
function setup(fetch, saved = "") {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      hidden: true, inert: true, open: false, value: "", textContent: "", disabled: false,
      classList: {add() {}, remove() {}}, listeners: {},
      addEventListener(type, callback) { this.listeners[type] = callback; },
      setAttribute() {}, removeAttribute() {}, focus() {},
      showModal() { this.open = true; }, close() { this.open = false; },
      querySelector() { return element("heading"); },
    });
    return elements.get(id);
  };
  const storage = new Map(saved ? [["soul-admin-key/team/soul", saved]] : []);
  const sandbox = {
    fetch, AbortController, DOMException, setTimeout, clearTimeout,
    sessionStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key)},
    document: {getElementById: element, querySelectorAll: () => [], body: element("body")},
  };
  const Auth = runInNewContext(`${source}\nControllerAuth`, sandbox);
  const received = [];
  const auth = new Auth({path: "/team/soul", validatePath: "/admin/state", onConnect: data => received.push(data)});
  return {auth, element, storage, received};
}
const response = (status, data = {}) => ({status, ok: status < 400, json: async () => data});
function deferred() { let resolve; const promise = new Promise(r => resolve = r); return {promise, resolve}; }

test("first visit stays locked without making anonymous admin requests; Escape cannot dismiss sign-in", () => {
  let calls = 0;
  const {auth, element} = setup(() => { calls++; });
  auth.start();
  assert.equal(calls, 0);
  assert.equal(element("console").hidden, true);
  assert.equal(element("console").inert, true);
  assert.equal(element("auth-dialog").open, true);
  let prevented = false;
  element("auth-dialog").listeners.cancel({preventDefault: () => prevented = true});
  assert.equal(prevented, true);
});

test("saved tokens are revalidated before revealing content and remain scoped to the controller path", async () => {
  const pending = deferred();
  const {auth, element, storage} = setup((url, options) => {
    assert.equal(url, "/team/soul/admin/state");
    assert.equal(options.headers["X-Soul-Admin-Key"], "saved-token");
    return pending.promise;
  }, "saved-token");
  const login = auth.start();
  assert.equal(element("console").hidden, true);
  assert.equal(element("auth-submit").disabled, true);
  pending.resolve(response(200, {nodes: []}));
  await login;
  assert.equal(element("console").hidden, false);
  assert.equal(element("console").inert, false);
  assert.equal(element("auth-dialog").open, false);
  assert.equal(storage.get("soul-admin-key/team/soul"), "saved-token");
  assert.equal(element("admin-key").value, "");
});

for (const status of [401, 403]) test(`rejected tokens (${status}) stay hidden and can be retried`, async () => {
  let valid = false;
  const {auth, element, storage} = setup(async () => response(valid ? 200 : status));
  auth.start();
  await auth.signIn("wrong");
  assert.equal(element("console").hidden, true);
  assert.equal(element("auth-submit").disabled, false);
  assert.match(element("auth-message").textContent, /Token not accepted/);
  assert.equal(storage.size, 0);
  valid = true;
  await auth.signIn("right");
  assert.equal(auth.authenticated, true);
});

test("network failure while validating never reveals the console", async () => {
  const {auth, element} = setup(async () => { throw new TypeError("Failed to fetch"); }, "saved");
  await auth.start();
  assert.equal(element("console").hidden, true);
  assert.equal(element("auth-submit").disabled, false);
  assert.match(element("auth-message").textContent, /Unable to connect/);
});

test("an API rejection after login locks the UI and aborts other requests", async () => {
  let calls = 0, signal;
  const stale = deferred();
  const {auth, element, storage} = setup(async (url, options) => {
    if (++calls === 1) return response(200);
    if (calls === 2) { signal = options.signal; return stale.promise; }
    return response(401);
  });
  await auth.signIn("valid");
  const inflight = auth.request("/admin/state");
  await assert.rejects(auth.request("/admin/nodes/node/logs"));
  assert.equal(signal.aborted, true);
  assert.equal(element("console").hidden, true);
  assert.match(element("auth-message").textContent, /Sign in again/);
  assert.equal(storage.size, 0);
  stale.resolve(response(200, {privateData: true}));
  await assert.rejects(inflight, {name: "AbortError"});
});

test("a late successful login cannot reopen the dashboard after sign-out", async () => {
  const pending = deferred();
  const {auth, element, received} = setup(() => pending.promise);
  const login = auth.signIn("valid");
  auth.lock("You have signed out.");
  pending.resolve(response(200));
  await login;
  assert.equal(element("console").hidden, true);
  assert.equal(auth.authenticated, false);
  assert.deepEqual(received, []);
});

test("repeated submissions perform one verification request", async () => {
  const pending = deferred(); let calls = 0;
  const {auth} = setup(() => { calls++; return pending.promise; });
  const login = auth.signIn("valid");
  await auth.signIn("another");
  assert.equal(calls, 1);
  pending.resolve(response(200));
  await login;
});
