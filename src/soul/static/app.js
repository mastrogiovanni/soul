"use strict";
const controllerURL = new URL(".", document.baseURI);
const controllerPath = controllerURL.pathname.replace(/\/$/, "");
const $ = (id) => document.getElementById(id);
const renderedMarkup = new Map();
function renderHTML(id, markup) {
  // Keep warning icons mounted during polling so hover and keyboard focus persist.
  if (renderedMarkup.get(id) === markup) return;
  $(id).innerHTML = markup;
  renderedMarkup.set(id, markup);
}
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const modelLabel = (id) => id.startsWith("/") ? `local · ${id.split("/").pop()}` : id;
const number = (n) => Number(n || 0).toLocaleString();
const gib = (n) => `${(Number(n || 0) / 1024 ** 3).toFixed(1)} GiB`;
const bytes = (n) => { const unit = Math.min(3, Math.max(0, Math.floor(Math.log2(n || 1) / 10))); return `${(n / 1024 ** unit).toFixed(unit ? 1 : 0)} ${['B', 'KiB', 'MiB', 'GiB'][unit]}`; };
const badge = (status) => `<span class="badge ${escapeHTML(status)}">${escapeHTML(status)}</span>`;
const warningIcon = (model) => {
  if (!model.warning) return '';
  const message = escapeHTML([model.warning, model.error].filter(Boolean).join('\n\n'));
  return `<span class="model-warning-icon" role="img" tabindex="0" aria-label="Model not yet available: ${message}" title="${message}">!</span>`;
};
const distributionText = (model) => model.distribution ? `${model.distribution.cached_layers} / ${model.layers} layers cached · ${bytes(model.distribution.cached_bytes)} / ${bytes(model.distribution.total_bytes)}` : 'Waiting for node inventories';
function preparedNotice(result, model) {
  const available = result.executable || result.status === "ready";
  notice(available ? `${model} is prepared and available for inference.` : `${model}: ${distributionText(result)}. See the model warning for details.`, available);
  pendingNoticeModel = available ? null : model;
}
const layerRanges = (layers) => {
  const ranges = []; let start = null, end = null;
  for (const layer of [...layers].sort((a, b) => a - b)) {
    if (start === null) start = end = layer;
    else if (layer === end + 1) end = layer;
    else { ranges.push(start === end ? `${start + 1}` : `${start + 1}–${end + 1}`); start = end = layer; }
  }
  if (start !== null) ranges.push(start === end ? `${start + 1}` : `${start + 1}–${end + 1}`);
  return ranges.join(", ") || "partial tensors";
};
let snapshot = null, selected = "", busy = false, logsNode = null, pendingNoticeModel = null;
function notice(text, success = false) { pendingNoticeModel = null; $("notice").textContent = text; $("notice").classList.toggle("success", success); $("notice").hidden = !text; }
const auth = new ControllerAuth({
  path: controllerPath, validatePath: "/admin/state",
  onConnect: data => { snapshot = data; notice(""); renderConnected(); },
  onLock: () => { snapshot = null; renderedMarkup.clear(); },
});
const api = (path, options) => auth.request(path, options);
function setTab(tab) {
  if (!["overview", "models", "nodes", "requests"].includes(tab)) tab = "overview";
  document.body.dataset.view = tab;
  document.querySelectorAll(".view").forEach((el) => el.hidden = el.id !== `${tab}-view`);
  document.querySelectorAll(".nav-item").forEach((el) => {
    const active = el.dataset.tab === tab;
    el.classList.toggle("active", active);
    if (active) el.setAttribute("aria-current", "page");
    else el.removeAttribute("aria-current");
  });
  const titles = {overview:["Network overview","Check your nodes, models and recent requests."], models:["Models","Check availability, distribute layers and connect a client."], nodes:["Nodes","Check hardware, downloads and node status."], requests:["Requests","Recent inference activity. Prompt and response content are never stored."]};
  $("page-title").textContent = titles[tab][0]; $("page-description").textContent = titles[tab][1]; $("crumb").textContent = tab[0].toUpperCase() + tab.slice(1);
}
function navigate(tab) {
  setTab(tab);
  history.replaceState(null, "", `#${document.body.dataset.view}`);
  window.scrollTo({top: 0, behavior: "instant"});
}
document.querySelectorAll("[data-tab]").forEach((el) => el.onclick = () => navigate(el.dataset.tab));
document.querySelectorAll("[data-go]").forEach((el) => el.onclick = () => navigate(el.dataset.go));
window.addEventListener("hashchange", () => setTab(location.hash.slice(1)));
setTab(location.hash.slice(1));
document.querySelectorAll("[data-close]").forEach((el) => el.onclick = () => $(el.dataset.close).close());
$("open-load").onclick = () => { if (!auth.authenticated) return; $("load-dialog").showModal(); };
$("load-form").onsubmit = async (event) => {
  event.preventDefault(); const model = $("model-id").value.trim(); if (!model) return;
  $("load-submit").disabled = true; $("load-dialog").close(); notice(`Inspecting and distributing ${model}. Download progress will appear on the nodes.`);
  selected = model;
  try { const result = await api(`/admin/models/${$("model-action").value}`, {method:"POST", body:JSON.stringify({model})}); preparedNotice(result, model); }
  catch (error) { notice(error.message); } finally { $("load-submit").disabled = false; await refresh(); }
};
$("pipeline-model").onchange = () => { selected = $("pipeline-model").value; renderTopology(); };
function renderJoinCommand() {
  const profile = $("join-profile").value;
  const device = profile === "node" ? "" : " --device cuda:0";
  $("join-command").textContent = `uv sync --locked --extra ${profile}\n\nexport SOUL_NODE_TOKEN='<node-token>'\nuv run --locked --extra ${profile} soul node --controller ${controllerURL.href.replace(/\/$/, "")}${device}`;
}
$("join-profile").onchange = renderJoinCommand;
$("join-button").onclick = () => { renderJoinCommand(); $("join-dialog").showModal(); };
$("logs-dialog").addEventListener("close", () => logsNode = null);
async function refreshLogs() {
  if (!logsNode) return;
  try { const data = await api(`/admin/nodes/${encodeURIComponent(logsNode)}/logs`); $("logs-output").textContent = data.logs.map((log) => `${new Date(log.time * 1000).toLocaleTimeString()}  ${log.level.padEnd(5)}  ${log.message}`).join("\n") || "No events received yet."; }
  catch (error) { $("logs-output").textContent = error.message; }
}

let usageModel = null;
const shellQuote = (value) => "'" + String(value).replace(/'/g, "'\"'\"'") + "'";
function renderUsage() {
  if (!usageModel) return;
  const model = snapshot.models.find((m) => m.model === usageModel.model) || usageModel;
  const base = new URL("v1", controllerURL).href;
  const format = $("use-format").value, basic = $("use-basic").checked;
  $("use-model-id").textContent = model.model;
  $("use-base-url").textContent = base;
  $("use-status").textContent = model.executable ? `Available for inference · ${number(model.context)} token context (prompt + output).` : model.warning || "Not currently executable. Connect more nodes or choose Distribute missing before sending requests.";
  $("use-request-options").hidden = format !== "curl";
  $("use-copy-status").textContent = "";
  if (format === "curl") {
    const stream = $("use-stream").checked;
    const payload = {model:model.model, messages:[{role:"user", content:"Hello!"}], max_tokens:Math.max(1, Math.min(64, Math.floor(model.context / 4))), stream};
    const auth = basic ? '  --user "$SOUL_INGRESS_USERNAME:$SOUL_INGRESS_PASSWORD" \\\n  -H "X-Soul-API-Key: $SOUL_API_KEY"' : '  -H "Authorization: Bearer $SOUL_API_KEY"';
    $("use-content").textContent = `# Set SOUL_API_KEY to your inference key first.${basic ? "\n# Also set SOUL_INGRESS_USERNAME and SOUL_INGRESS_PASSWORD." : ""}\ncurl --fail-with-body${stream ? " --no-buffer" : ""} ${shellQuote(base + "/chat/completions")} \\\n${auth} \\\n  -H 'Content-Type: application/json' \\\n  --data-raw ${shellQuote(JSON.stringify(payload, null, 2))}`;
    $("use-docs").href = new URL("docs", controllerURL).href;
    $("use-docs").textContent = "API reference ↗";
  } else if (format === "webui") {
    $("use-content").textContent = `1. Open Settings → Admin → Connections in Open WebUI.\n2. Add an OpenAI API connection (Provider: Default).\n3. URL: ${base}\n4. API key: your SOUL_API_KEY inference key.\n5. Save, verify the connection, then refresh the model list.\n6. Select this model: ${model.model}\n\nOnly executable models appear in Soul's /models list.\nUse text chat; disable tools and unsupported request options.${basic ? "\n\nThis site also requires Basic Auth. The default Bearer-only connection is insufficient. Use a connection/gateway that sends both:\nAuthorization: Basic <base64(site-username:site-password)>\nX-Soul-API-Key: <your inference key>\n\nIf your Open WebUI version cannot set both headers, use an approved gateway or private controller endpoint that enforces Soul's API key. Keep the public ingress authentication enabled." : ""}`;
    $("use-docs").href = "https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/";
    $("use-docs").textContent = "Open WebUI connection guide ↗";
  } else {
    $("use-content").textContent = `Soul does not implement Ollama's native /api/chat, /api/generate or /api/tags endpoints. Pointing ollama run or an Ollama-only connection at Soul will not work.\n\nTo use this distributed model in a chat application that also supports Ollama, add a separate OpenAI-compatible connection:\n\nBase URL: ${base}\nAPI key: your SOUL_API_KEY inference key\nModel ID: ${model.model}\n\nOpen WebUI can keep your Ollama connection alongside this Soul connection. Select the Open WebUI guide for setup.${basic ? "\n\nFor this site's Basic Auth, see the Open WebUI guide or the curl example with the Basic Auth option enabled." : ""}`;
    $("use-docs").href = "https://docs.ollama.com/api/openai-compatibility";
    $("use-docs").textContent = "Ollama API compatibility reference ↗";
  }
}
function openUsage(id) {
  usageModel = snapshot.models.find((m) => m.model === id);
  if (!usageModel) return;
  renderUsage();
  $("use-dialog").showModal();
}
for (const id of ["use-format", "use-stream", "use-basic"]) $(id).onchange = renderUsage;
$("use-dialog").addEventListener("close", () => usageModel = null);
async function copyUsage(id) {
  const element = $(id), value = element.textContent;
  try {
    await navigator.clipboard.writeText(value);
    $("use-copy-status").textContent = "Copied.";
  } catch {
    const range = document.createRange(); range.selectNodeContents(element);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    $("use-copy-status").textContent = "Clipboard unavailable. Text selected; press Ctrl+C or ⌘C to copy.";
  }
}

let pendingDeletion = null, destructiveBusy = false;
function confirmDeletion(kind, id) {
  if (destructiveBusy) return;
  const node = snapshot.nodes.find(n => n.node_id === id);
  pendingDeletion = {kind, id};
  $("delete-title").textContent = kind === "model" ? "Delete model?" : "Clear every cached layer?";
  $("delete-target").textContent = kind === "model" ? id : `${node?.name || id} (${id})`;
  $("delete-description").textContent = kind === "model" ? "Remove this model from the catalog, unload it, and delete all its cached revisions, tensors and tokenizer files across the nodes. Offline nodes will be cleaned when they reconnect." : "Unload this node and permanently remove every model layer and tokenizer in its Soul cache. Other nodes keep their files. Downloads to this node will stay paused until you choose Resume downloads. Offline nodes will be cleaned when they reconnect.";
  $("delete-submit").textContent = kind === "model" ? "Delete model" : "Clear all cached layers";
  $("delete-dialog").showModal();
}
$("delete-dialog").addEventListener("close", () => pendingDeletion = null);
$("delete-form").onsubmit = async event => {
  event.preventDefault();
  if (!pendingDeletion || destructiveBusy) return;
  const {kind, id} = pendingDeletion;
  destructiveBusy = true; $("delete-dialog").close();
  notice("Waiting for active work to finish, then removing cached files…");
  try {
    const result = kind === "model" ? await api("/admin/models/delete", {method:"POST", body:JSON.stringify({model:id})}) : await api(`/admin/nodes/${encodeURIComponent(id)}/cache`, {method:"DELETE"});
    const pending = kind === "model" ? result.pending_nodes.length > 0 : result.pending;
    notice(kind === "model" ? `${id} removed from the catalog. ${pending ? 'Some node files still await cleanup; see the pending list.' : `${bytes(result.freed_bytes)} of cached files removed.`}` : pending ? "Node cache clearing is queued. Downloads are paused; see the pending list." : `${bytes(result.freed_bytes)} of cached files removed from this node. Downloads are paused until you choose Resume downloads.`, !pending);
  } catch (error) { notice(error.message); }
  finally { destructiveBusy = false; await refresh(); }
};

document.addEventListener("click", async (event) => {
  const deleteModel = event.target.closest("[data-delete-model]");
  if (deleteModel) confirmDeletion("model", deleteModel.dataset.deleteModel);
  const clearCache = event.target.closest("[data-clear-cache]");
  if (clearCache) confirmDeletion("node", clearCache.dataset.clearCache);
  const resumeCache = event.target.closest("[data-resume-cache]");
  if (resumeCache) {
    resumeCache.disabled = true;
    try { await api(`/admin/nodes/${encodeURIComponent(resumeCache.dataset.resumeCache)}/cache/resume`, {method:"POST"}); notice("Downloads resumed. Registered models will be distributed as capacity allows.", true); }
    catch (error) { notice(error.message); } finally { resumeCache.disabled = false; await refresh(); }
  }
  const retryCleanup = event.target.closest("[data-retry-cleanup]");
  if (retryCleanup) {
    retryCleanup.disabled = true;
    try { const result = await api("/admin/cache/retry", {method:"POST"}); const pending = result.pending_models.length || result.nodes.some(n => n.pending); notice(pending ? "Cleanup is still pending. Reconnect or update the listed nodes, then retry." : "Cache cleanup completed.", !pending); }
    catch (error) { notice(error.message); } finally { retryCleanup.disabled = false; await refresh(); }
  }
  const use = event.target.closest("[data-use-model]");
  if (use) openUsage(use.dataset.useModel);
  const copy = event.target.closest("[data-copy-use]");
  if (copy) await copyUsage(copy.dataset.copyUse);
  const remove = event.target.closest("[data-remove-node]");
  if (remove) {
    remove.disabled = true;
    try { await api(`/admin/nodes/${encodeURIComponent(remove.dataset.removeNode)}`, {method:"DELETE"}); notice("Offline node removed. Missing model shards will be prepared on connected nodes when capacity allows.", true); await refresh(); }
    catch (error) { notice(error.message); } finally { remove.disabled = false; }
  }
  const prepare = event.target.closest("[data-prepare-model]");
  if (prepare) {
    prepare.disabled = true;
    try { const result = await api(`/admin/models/${prepare.dataset.action}`, {method:"POST", body:JSON.stringify({model:prepare.dataset.prepareModel})}); preparedNotice(result, prepare.dataset.prepareModel); }
    catch (error) { notice(error.message); } finally { prepare.disabled = false; await refresh(); }
  }
  const logs = event.target.closest("[data-logs]");
  if (logs) { logsNode = logs.dataset.logs; $("logs-title").textContent = `Logs · ${snapshot.nodes.find((n) => n.node_id === logsNode)?.name || logsNode}`; $("logs-output").textContent = "Loading…"; $("logs-dialog").showModal(); await refreshLogs(); }
  const unload = event.target.closest("[data-unload]");
  if (unload) { unload.disabled = true; try { await api("/admin/models/unload", {method:"POST",body:JSON.stringify({model:unload.dataset.unload})}); notice("Model unloaded. Its downloaded weights remain in the nodes’ disk cache.", true); await refresh(); } catch(error) { notice(error.message); } finally { unload.disabled = false; } }
});
function renderTopology() {
  const model = snapshot.models.find((m) => m.model === selected);
  const topology = $("topology");
  if (!model) {
    topology.classList.add("empty-topology");
    renderHTML("topology", '<div class="controller-orb">✳</div><h3>No models registered</h3><p>Preload a model to distribute its layers across your nodes.</p>');
    return;
  }
  topology.classList.remove("empty-topology");
  const stages = model.assignments.map((assignment, index) => {
    const node = snapshot.nodes.find((n) => n.node_id === assignment.node_id);
    const status = !node?.connected ? "offline" : node.download?.model === model.model && model.status === "downloading" ? "downloading" : ["available", "partial"].includes(model.status) ? "cached" : node.status;
    return `${index ? '<span class="stage-arrow" aria-hidden="true">→</span>' : ''}<article class="stage"><div class="stage-top"><span class="machine-icon">▣</span><div><div class="stage-name">${escapeHTML(node?.name || assignment.node_id)}</div><div class="stage-device">${escapeHTML(node?.capabilities?.device_name || "Disconnected")}</div></div></div><div class="stage-range"><span>Layers ${assignment.start + 1}–${assignment.end}</span><span>${assignment.end - assignment.start} / ${model.layers}</span></div><div class="layer-strip">${Array.from({length:Math.min(assignment.end-assignment.start,32)},()=>"<i></i>").join("")}</div><div class="stage-footer">${badge(status)}<span>${gib(assignment.estimated_bytes)} reserved</span></div></article>`;
  }).join("");
  renderHTML("topology", `<div class="controller-row"><span class="controller-orb">✳</span><div><strong>Controller ${badge(model.status)} ${warningIcon(model)}</strong><small>${escapeHTML(distributionText(model))}</small></div></div>${model.warning ? `<p class="model-warning-text">${escapeHTML([model.warning, model.error].filter(Boolean).join(" "))}</p>` : ""}<div class="stage-list">${stages || '<p class="empty-small">No complete shard fits yet. Connect a compatible node to start distributing.</p>'}</div>`);
}
function render() {
  const {nodes, models, metrics} = snapshot, connected = nodes.filter((n) => n.connected), ready = models.filter((m) => m.executable);
  const cleanup = (snapshot.cache_cleanup || []).map(item => `<li><strong>${escapeHTML(item.model)}</strong>: waiting for ${escapeHTML(item.pending_nodes.join(', '))}${Object.values(item.errors || {}).length ? `<small>${escapeHTML(Object.values(item.errors).join('; '))}</small>` : ''}</li>`);
  cleanup.push(...nodes.filter(n => n.cache_clear_pending).map(n => `<li><strong>${escapeHTML(n.name)}</strong>: cache clearing pending${n.cache_error ? `<small>${escapeHTML(n.cache_error)}</small>` : ' · reconnect the node to finish'}</li>`));
  $("cleanup-status").hidden = !cleanup.length;
  renderHTML("cleanup-status", cleanup.length ? `<strong>Cache cleanup pending</strong><ul>${cleanup.join('')}</ul><button class="button secondary" data-retry-cleanup>Retry cleanup</button>` : '');
  const pendingModel = models.find(m => m.model === pendingNoticeModel);
  if (pendingModel) preparedNotice(pendingModel, pendingModel.model);
  $("stat-nodes").textContent = connected.length; $("stat-models").textContent = ready.length; $("stat-requests").textContent = number(metrics.counts.completed); $("stat-tokens").textContent = number(metrics.output_tokens);
  $("node-count").textContent = connected.length; $("model-count").textContent = models.length; $("node-detail").textContent = `${gib(connected.reduce((sum,n)=>sum+(n.capabilities.budget_bytes||0),0))} combined memory budget`;
  $("request-detail").textContent = `${number(metrics.counts.failed)} failed · ${number(metrics.counts.cancelled)} cancelled`;
  $("queue-detail").textContent = `${snapshot.queue_depth} waiting · ${snapshot.active_request ? "1 running" : "none running"}`;
  $("context-label").textContent = `${number(snapshot.max_context)} token context`;
  if (!models.some((m) => m.model === selected)) selected = ready[0]?.model || models[0]?.model || "";
  $("pipeline-model").innerHTML = models.length ? models.map((m) => `<option value="${escapeHTML(m.model)}" ${m.model === selected ? "selected" : ""}>${m.warning ? '⚠ ' : ''}${escapeHTML(modelLabel(m.model))}</option>`).join("") : '<option value="">No models preloaded</option>';
  renderTopology();
  $("node-preview").innerHTML = nodes.length ? nodes.slice(0,4).map((n) => `<div class="node-row"><div class="machine-icon">▣</div><div class="node-info"><strong>${escapeHTML(n.name)}</strong><small>${escapeHTML(n.capabilities.device_name)} · ${gib(n.capabilities.budget_bytes)}</small></div>${badge(n.status)}<button class="text-button" data-logs="${escapeHTML(n.node_id)}">Logs ↗</button></div>`).join("") : '<div class="empty-small">Your nodes will appear here when they connect.</div>';
  $("activity").innerHTML = metrics.recent.length ? metrics.recent.slice(0,4).map((r) => `<div class="activity-row"><span class="activity-mark">${r.status === "completed" ? "✓" : "↗"}</span><div><strong>${escapeHTML(r.model.split('/').pop())}</strong><small>${escapeHTML(r.status)} · ${r.output_tokens} tokens · ${r.duration.toFixed(1)}s</small></div><time class="activity-time">${new Date(r.created*1000).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})}</time></div>`).join("") : '<div class="empty-small">No requests yet.<br><span>Ready when you are.</span></div>';
  renderHTML("models-table", models.length ? `<table><thead><tr><th scope="col">MODEL</th><th scope="col">ARCHITECTURE</th><th scope="col">DISTRIBUTION</th><th scope="col">STATUS</th><th scope="col"><span class="sr-only">Actions</span></th></tr></thead><tbody>${models.map((m)=>`<tr><td data-label="Model">${warningIcon(m)} ${escapeHTML(m.model)}<small>Revision ${escapeHTML(m.revision.slice(0,12))}</small></td><td data-label="Architecture">${escapeHTML(m.architecture)}<small>${escapeHTML(m.quantization === "none" ? "Unquantized" : m.quantization || "Unquantized")} · ${m.context} token context</small></td><td data-label="Distribution">${escapeHTML(distributionText(m))}${m.distribution ? `<small>${m.distribution.cached_tensors} / ${m.distribution.total_tensors} tensors · ${m.distribution.nodes} connected nodes</small><progress aria-label="Cached model tensor bytes" value="${m.distribution.cached_bytes}" max="${m.distribution.total_bytes || 1}"></progress>` : ''}</td><td data-label="Status">${badge(m.status)}${m.warning ? `<p class="model-warning-text">${escapeHTML([m.warning, m.error].filter(Boolean).join(" "))}</p>` : ""}</td><td data-label="Actions"><button data-use-model="${escapeHTML(m.model)}">Use model</button>${m.status === "ready" ? `<button data-unload="${escapeHTML(m.model)}">Unload GPU</button>` : `<button data-prepare-model="${escapeHTML(m.model)}" data-action="${m.executable ? 'load' : 'preload'}" ${m.status === 'downloading' ? 'disabled' : ''}>${m.executable ? 'Load GPU' : m.status === 'downloading' ? 'Distributing…' : 'Distribute missing'}</button>`}<button class="danger-button" data-delete-model="${escapeHTML(m.model)}">Delete model</button></td></tr>`).join("")}</tbody></table>` : '<div class="empty-small">No models yet. Preload a model to get started.</div>');
  $("nodes-table").innerHTML = nodes.length ? `<table><thead><tr><th scope="col">NODE</th><th scope="col">HARDWARE</th><th scope="col">MEMORY BUDGET</th><th scope="col">CACHED LAYERS</th><th scope="col">STATE</th><th scope="col"><span class="sr-only">Actions</span></th></tr></thead><tbody>${nodes.map((n)=>`<tr><td data-label="Node">${escapeHTML(n.name)}<small><code>${escapeHTML(n.node_id)}</code></small></td><td data-label="Hardware">${escapeHTML(n.capabilities.device_name)}<small>${escapeHTML(n.capabilities.dtype)} · ${escapeHTML(n.capabilities.weight_mode)} weights</small></td><td data-label="Memory budget">${gib(n.capabilities.budget_bytes)}</td><td data-label="Cached layers">${(n.cached_models || []).map((m)=>`${escapeHTML(modelLabel(m.model))}<small>${m.layers.length ? `Layers ${escapeHTML(layerRanges(m.layers))}` : "Partial cache"} · ${m.tensor_count} tensors</small>`).join('<br>') || '—'}</td><td data-label="State">${badge(n.status)}${n.cache_paused ? `<small>${n.cache_clear_pending ? 'Cache clear pending' : 'Downloads paused'}</small>` : ''}${n.download ? `<small>Downloading ${escapeHTML(modelLabel(n.download.model))}</small>` : ''}${n.total_bytes ? `<small>${gib(n.downloaded_bytes)} / ${gib(n.total_bytes)}</small><progress value="${Number(n.downloaded_bytes)}" max="${Number(n.total_bytes)}"></progress>`:''}${n.tensor ? `<small>${escapeHTML(n.tensor)}</small>`:''}${n.error ? `<small>${escapeHTML(n.error)}</small>`:''}</td><td data-label="Actions"><button data-logs="${escapeHTML(n.node_id)}">View logs</button><button class="danger-button" data-clear-cache="${escapeHTML(n.node_id)}" ${n.cache_clear_pending ? 'disabled' : ''}>Clear all cached layers</button>${n.cache_paused && !n.cache_clear_pending ? `<button data-resume-cache="${escapeHTML(n.node_id)}">Resume downloads</button>` : ''}${!n.connected ? `<button data-remove-node="${escapeHTML(n.node_id)}">Remove node</button>` : ''}</td></tr>`).join("")}</tbody></table>` : '<div class="empty-small">No nodes registered. Connect a node to contribute compute.</div>';
  $("requests-table").innerHTML = metrics.recent.length ? `<table><thead><tr><th scope="col">REQUEST</th><th scope="col">MODEL</th><th scope="col">STATUS</th><th scope="col">TOKENS IN / OUT</th><th scope="col">DURATION</th></tr></thead><tbody>${metrics.recent.map((r)=>`<tr><td data-label="Request"><code>${escapeHTML(r.id.slice(0,20))}…</code><small>${new Date(r.created*1000).toLocaleString()}</small></td><td data-label="Model">${escapeHTML(r.model)}</td><td data-label="Status">${badge(r.status)}${r.error ? `<small>${escapeHTML(r.error)}</small>`:''}</td><td data-label="Tokens in / out">${r.input_tokens} / ${r.output_tokens}</td><td data-label="Duration">${r.duration.toFixed(2)}s</td></tr>`).join("")}</tbody></table>` : '<div class="empty-small">Requests appear here when you use an inference endpoint.</div>';
}
function renderConnected() {
  $("live-dot").classList.add("online");
  $("live-label").textContent = "Controller connected";
  $("updated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
  render();
}
async function refresh() {
  if (!auth.authenticated || busy) return; busy = true;
  try {
    snapshot = await api("/admin/state"); renderConnected(); await refreshLogs();
  } catch (error) {
    $("live-dot").classList.remove("online"); $("live-label").textContent = "Connection interrupted";
  } finally { busy = false; }
}
auth.start(); setInterval(refresh, 3000);
