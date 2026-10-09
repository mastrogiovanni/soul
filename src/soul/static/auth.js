"use strict";

// Shared sign-in gate for both admin pages. API authorization remains server-side.
class ControllerAuth {
  constructor({path, validatePath, onConnect, onLock = () => {}}) {
    this.path = path;
    this.validatePath = validatePath;
    this.onConnect = onConnect;
    this.onLock = onLock;
    this.storageKey = `soul-admin-key${path}`;
    this.token = "";
    this.authenticated = false;
    this.pending = false;
    this.generation = 0;
    this.requests = new Set();
    this.panel = document.getElementById("console");
    this.dialog = document.getElementById("auth-dialog");
    this.form = document.getElementById("auth-form");
    this.input = document.getElementById("admin-key");
    this.submit = document.getElementById("auth-submit");
    this.message = document.getElementById("auth-message");
    this.dialog.addEventListener("cancel", event => event.preventDefault());
    this.form.onsubmit = event => { event.preventDefault(); this.signIn(this.input.value.trim()); };
    document.getElementById("disconnect").onclick = () => this.lock("You have signed out.");
  }

  storage(value) {
    // A blocked storage API must not prevent signing in for the current page.
    try {
      if (value === undefined) return sessionStorage.getItem(this.storageKey) || "";
      if (value) sessionStorage.setItem(this.storageKey, value);
      else sessionStorage.removeItem(this.storageKey);
    } catch { /* Keep the token in memory when session storage is unavailable. */ }
    return "";
  }

  start() {
    const saved = this.storage();
    this.lock();
    if (saved) return this.signIn(saved);
  }

  lock(message = "") {
    this.authenticated = false;
    this.token = "";
    this.generation++;
    this.storage("");
    for (const controller of this.requests) controller.abort();
    this.panel.hidden = true;
    this.panel.inert = true;
    document.body.classList.add("auth-locked");
    document.querySelectorAll("#console dialog[open]").forEach(dialog => dialog.close());
    this.onLock();
    this.input.value = "";
    this.message.textContent = message;
    this.message.hidden = !message;
    if (!this.dialog.open) this.dialog.showModal();
    if (!this.pending) this.input.focus();
  }

  async signIn(token) {
    if (this.pending || !token) return;
    this.pending = true;
    this.token = token;
    const generation = this.generation;
    this.input.disabled = this.submit.disabled = true;
    this.submit.textContent = "Signing in…";
    this.form.setAttribute("aria-busy", "true");
    this.message.textContent = "Verifying your token…";
    this.message.hidden = false;
    const timeout = new AbortController();
    const timer = setTimeout(() => timeout.abort(), 15000);
    try {
      const data = await this.request(this.validatePath, {signal: timeout.signal});
      if (generation !== this.generation) return;
      this.onConnect(data);
      this.authenticated = true;
      this.storage(token);
      this.input.value = "";
      this.panel.hidden = false;
      this.panel.inert = false;
      document.body.classList.remove("auth-locked");
      this.dialog.close();
      const heading = this.panel.querySelector("h1");
      heading.setAttribute("tabindex", "-1");
      heading.focus({preventScroll: true});
    } catch {
      // An authorization failure already locked the UI with a specific message.
      if (generation === this.generation) this.lock("Unable to connect to the controller. Check your connection and try again.");
    } finally {
      clearTimeout(timer);
      this.pending = false;
      this.input.disabled = this.submit.disabled = false;
      this.submit.textContent = "Sign in";
      this.form.removeAttribute("aria-busy");
      if (!this.authenticated) this.input.focus();
    }
  }

  async request(path, options = {}) {
    if (!this.token) throw new Error("Sign in with your admin token to continue.");
    const generation = this.generation;
    const controller = new AbortController();
    const abort = () => controller.abort();
    options.signal?.addEventListener("abort", abort, {once: true});
    if (options.signal?.aborted) abort();
    this.requests.add(controller);
    try {
      const response = await fetch(`${this.path}${path}`, {
        credentials: "same-origin", ...options, signal: controller.signal,
        headers: {"Content-Type": "application/json", ...options.headers, "X-Soul-Admin-Key": this.token},
      });
      if (generation !== this.generation) throw new DOMException("Session changed", "AbortError");
      if (response.status === 401 || response.status === 403) {
        const message = this.authenticated ? "Your token is no longer accepted. Sign in again." : "Token not accepted. Check your admin token and try again.";
        this.lock(message);
        throw new Error(message);
      }
      const data = await response.json();
      if (generation !== this.generation) throw new DOMException("Session changed", "AbortError");
      if (!response.ok) {
        const error = new Error(data.error?.message || data.detail || `Request failed (${response.status})`);
        error.code = data.error?.code;
        throw error;
      }
      return data;
    } finally {
      this.requests.delete(controller);
      options.signal?.removeEventListener("abort", abort);
    }
  }
}
