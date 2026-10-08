import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { getMessages } from "../../site/i18n.mjs";

const italianURL = new URL("../../site/index.html", import.meta.url);
const englishURL = new URL("../../site/en/index.html", import.meta.url);
const italian = readFileSync(italianURL, "utf8");
const english = readFileSync(englishURL, "utf8");

test("Italian is the static default; English has its own address and language switch", () => {
  assert.match(italian, /<html lang="it">/);
  assert.match(italian, /<title>Soul — Grandi idee\. Risorse condivise\.<\/title>/);
  assert.match(english, /<html lang="en">/);
  assert.match(italian, /href="en\/" data-language-link lang="en"/);
  assert.match(english, /href="\.\.\/" data-language-link lang="it"/);
  for (const [html, language] of [[italian, "it"], [english, "en"]]) {
    const activeLinks = html.match(/<a\b[^>]*aria-current="page"[^>]*>/g);
    assert.equal(activeLinks.length, 1);
    assert.match(activeLinks[0], new RegExp(`lang="${language}"`));
    for (const alternate of ["it", "en", "x-default"]) {
      assert.match(html, new RegExp(`rel="alternate" hreflang="${alternate}"`));
    }
  }
});

test("shared assets and fragment links resolve from both language directories", () => {
  for (const [html, pageURL] of [[italian, italianURL], [english, englishURL]]) {
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
    assert.equal(ids.length, new Set(ids).size, "Duplicate element IDs");
    for (const [, fragment] of html.matchAll(/href="#([^"]+)"/g)) {
      assert.ok(ids.includes(fragment), `Missing section: ${fragment}`);
    }
    for (const [, path] of html.matchAll(/<(?:script|img|link)\b[^>]*\b(?:src|href)="([^"]+)"/g)) {
      if (/^https?:/.test(path)) continue;
      const asset = new URL(path, pageURL);
      asset.search = "";
      assert.ok(existsSync(asset), `Missing asset: ${asset}`);
    }
  }
});

test("translation preserves model facts, IDs, and executable installation commands", () => {
  const models = (html) => html.match(/<tr\b[^>]*data-model="[^"]+"[^>]*>/g);
  assert.equal(models(italian).length, 9);
  assert.deepEqual(models(italian), models(english));
  for (const id of ["install-command", "limited-command", "restart-command"]) {
    const pattern = new RegExp(`<code id="${id}">([^<]+)<\\/code>`);
    assert.equal(italian.match(pattern)[1], english.match(pattern)[1]);
  }
  const sections = (html) => [...html.matchAll(/<section\b[^>]*\bid="([^"]+)"/g)].map((match) => match[1]);
  assert.deepEqual(sections(italian), sections(english));
});

test("dynamic messages localize number formats, singulars, and empty states", () => {
  const it = getMessages();
  const en = getMessages("en-GB");
  assert.equal(it.locale, "it");
  assert.equal(getMessages("fr").locale, "it");
  assert.equal(it.format.format(10240.25), "10.240,25");
  assert.equal(en.format.format(10240.25), "10,240.25");
  assert.equal(it.resultCount(0, 9), "0 modelli mostrati su 9");
  assert.equal(it.resultCount(1, 9), "1 modello mostrato su 9");
  assert.equal(en.resultCount(1, 9), "1 of 9 models shown");
  assert.equal(it.moreMachines(1, 8.25), "Circa 1 macchina in più da 8,25 GiB");
  assert.equal(it.moreMachines(2, 8), "Circa 2 macchine in più da 8 GiB");
  assert.equal(it.labels["more-memory"], "Serve più memoria");
  assert.equal(en.labels["more-memory"], "More memory needed");
  assert.deepEqual(Object.keys(it).sort(), Object.keys(en).sort());
  assert.deepEqual(Object.keys(it.labels).sort(), Object.keys(en.labels).sort());
});
