"use strict";
const $ = id => document.getElementById(id);
const base = new URL(".", document.baseURI);
const controllerPath = base.pathname.replace(/\/$/, "");
const storageKey = `soul-admin-key${controllerPath}`;
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
const gib = bytes => `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
const labels = {candidate:"Supported format · check layout", supported:"Supported · layout verified", unsupported:"Unsupported", unknown:"Not checked"};
let key = sessionStorage.getItem(storageKey) || "";
let rows = [], pages = [""], page = 0, nextCursor = null, searchRequest = null, selected = null, detailRequest = 0, preparing = false;
let filters = {q:"", family:"", sort:"downloads"};
const checked = new Map();
function notice(message, success = false) { $("notice").textContent = message; $("notice").hidden = !message; $("notice").classList.toggle("success", success); }
async function api(path, options = {}) {
  const response = await fetch(`${controllerPath}${path}`, {credentials:"same-origin", ...options, headers:{"X-Soul-Admin-Key":key, "Content-Type":"application/json"}});
  const data = await response.json();
  if (!response.ok) { const error = new Error(data.error?.message || "Request failed"); error.code = data.error?.code; if (response.status === 401) $("auth-panel").hidden = false; throw error; }
  return data;
}
function render() {
  const filter = $("support-filter").value;
  const visible = rows.filter(row => filter === "all" || (filter === "compatible" ? ["candidate", "supported"].includes(row.status) : row.status === filter));
  $("results").innerHTML = visible.length ? visible.map(row => `<article class="panel model-card"><span class="badge ${esc(row.status)}">${esc(labels[row.status])}</span><h3><a href="${esc(row.url)}" target="_blank" rel="noopener">${esc(row.model)}</a></h3><div class="model-meta">${esc(row.architecture)} · ${esc(row.quantization === "none" ? "Unquantized" : row.quantization)} · ${Number(row.downloads || 0).toLocaleString()} downloads${row.gated ? " · Gated access" : ""}</div><p>${esc(row.reason)}</p><div class="card-actions">${row.status !== "unsupported" ? `<button class="button secondary" data-check="${esc(row.model)}">${row.status === "supported" ? "Review & prepare" : "Check compatibility"}</button>` : ""}<a class="text-button" href="${esc(row.url)}" target="_blank" rel="noopener">Model card ↗</a></div></article>`).join("") : '<div class="catalog-empty">No models match these filters on this page. Change the search or try the next page.</div>';
  $("results-status").textContent = `${visible.length} of ${rows.length} results on page ${page + 1} · live Hugging Face catalog`;
  $("page-number").textContent = `Page ${page + 1}`;
  $("previous").disabled = page === 0;
  $("next").disabled = !nextCursor;
}
async function search(targetPage = 0, reset = false) {
  if (!key) { $("admin-key").focus(); return; }
  searchRequest?.abort();
  const request = new AbortController(); searchRequest = request;
  if (reset) { filters = {q:$("search-query").value.trim(), family:$("family").value.trim(), sort:$("sort").value}; pages = [""]; page = 0; nextCursor = null; rows = []; $("results").innerHTML = '<div class="catalog-empty">Searching model configurations…</div>'; }
  $("results").setAttribute("aria-busy", "true"); $("results-status").textContent = "Searching the Hub and reading model configurations…";
  $("previous").disabled = $("next").disabled = true;
  notice("");
  try {
    const params = new URLSearchParams({...filters, cursor:pages[targetPage] || ""});
    const data = await api(`/admin/catalog/search?${params}`, {signal:request.signal});
    if (request !== searchRequest) return;
    page = targetPage; nextCursor = data.next_cursor;
    rows = data.models.map(row => ({...row, ...(checked.get(`${row.model}@${row.revision}`) || {})}));
    render();
    document.querySelectorAll("[data-family]").forEach(button => button.setAttribute("aria-pressed", button.dataset.family === filters.family));
  } catch (error) {
    if (error.name !== "AbortError") { notice(error.message); $("results-status").textContent = "Search unavailable. You can retry or use the control room to add a model by ID."; $("previous").disabled = page === 0; $("next").disabled = !nextCursor; }
  } finally { if (request === searchRequest) $("results").setAttribute("aria-busy", "false"); }
}
async function connect() {
  if (!key) return;
  try {
    const data = await api("/admin/catalog/support");
    sessionStorage.setItem(storageKey, key); $("auth-panel").hidden = true;
    $("families").innerHTML = data.families.map(f => `<option value="${esc(f.id)}">${esc(f.name)} · ${f.supported ? "Supported" : "Adapter needed"}</option>`).join("");
    $("family-shortcuts").innerHTML = [{id:"", name:"All families"}, ...data.families.slice(0, 5)].map(f => `<button type="button" data-family="${esc(f.id)}" aria-pressed="false">${esc(f.name)}</button>`).join("");
    await search(0, true);
  } catch (error) { notice(error.message); $("auth-panel").hidden = false; }
}
function showDetails(data) {
  $("detail-body").innerHTML = `<span class="badge supported">Supported · layout verified</span><dl><dt>Architecture / format</dt><dd>${esc(data.architecture)} / ${esc(data.quantization)}</dd><dt>Revision</dt><dd>${esc(data.revision.slice(0, 12))}</dd><dt>Layers / configured context</dt><dd>${data.layers} / ${data.context.toLocaleString()} tokens</dd><dt>Weight memory · FP16 nodes</dt><dd>${gib(data.weight_bytes_fp16)}</dd><dt>Weight memory · FP32 nodes</dt><dd>${gib(data.weight_bytes_fp32)}</dd><dt>Checkpoint tensor download</dt><dd>${gib(data.download_bytes)}</dd><dt>Compatible connected nodes</dt><dd>${data.compatible_nodes}</dd></dl><p>${esc(data.reason)}</p><p>Prepare now to download the shards that fit. If capacity is insufficient, Soul keeps the model with a warning and distributes missing pieces automatically when compatible nodes join.</p>`;
  $("prepare-form").hidden = false;
}
async function check(model) {
  if (preparing) return;
  const request = ++detailRequest;
  const row = rows.find(item => item.model === model);
  selected = null;
  $("detail-model").textContent = model;
  $("detail-body").textContent = "Reading configuration and tensor headers. No model weights are downloaded during this check.";
  $("prepare-form").hidden = true;
  $("model-dialog").showModal();
  try {
    const data = await api("/admin/catalog/check", {method:"POST", body:JSON.stringify({model, revision:row?.revision || undefined})});
    checked.set(`${model}@${data.revision}`, {status:"supported", reason:"Checkpoint layout verified. Review memory estimates and prepare it on your network."});
    if (row && row.revision === data.revision) Object.assign(row, checked.get(`${model}@${data.revision}`));
    render();
    if (request !== detailRequest) return;
    selected = data; showDetails(data);
  } catch (error) {
    if (request === detailRequest) $("detail-body").textContent = error.message;
    if (row && error.code === "unsupported_model") { row.status = "unsupported"; row.reason = error.message; render(); }
  }
}
$("auth-form").onsubmit = event => { event.preventDefault(); key = $("admin-key").value.trim(); connect(); };
$("disconnect").onclick = () => { sessionStorage.removeItem(storageKey); location.reload(); };
$("search-form").onsubmit = event => { event.preventDefault(); search(0, true); };
$("support-filter").onchange = render;
$("previous").onclick = () => search(page - 1);
$("next").onclick = () => { if (nextCursor) { pages[page + 1] = nextCursor; search(page + 1); } };
$("family-shortcuts").onclick = event => { const button = event.target.closest("[data-family]"); if (button) { $("family").value = button.dataset.family; search(0, true); } };
$("results").onclick = event => { const button = event.target.closest("[data-check]"); if (button) check(button.dataset.check); };
document.querySelectorAll("[data-example]").forEach(button => button.onclick = () => { $("search-query").value = button.dataset.example; $("family").value = ""; $("support-filter").value = "all"; search(0, true); $("search-form").scrollIntoView({behavior:"smooth"}); });
$("close-detail").onclick = () => $("model-dialog").close();
$("model-dialog").addEventListener("close", () => { ++detailRequest; });
$("prepare-form").onsubmit = async event => {
  event.preventDefault(); if (!selected || preparing) return;
  const model = selected.model; preparing = true; $("prepare-submit").disabled = true;
  $("detail-body").textContent = `Preparing ${model}. Follow download progress in the control room.`;
  try {
    const result = await api(`/admin/models/${$("prepare-action").value}`, {method:"POST", body:JSON.stringify({model, revision:selected.revision})});
    $("detail-body").textContent = result.executable || result.status === "ready" ? `${model} is prepared and available for inference. Open the control room and choose “Use model” for connection instructions.` : `${model}: ${result.distribution?.cached_layers || 0} / ${result.layers} layers cached. ${result.warning} Open the control room to track distribution or choose “Distribute missing” after freeing capacity.`;
    $("prepare-form").hidden = true;
  } catch (error) { $("detail-body").textContent = error.message; }
  finally { preparing = false; $("prepare-submit").disabled = false; }
};
connect();
