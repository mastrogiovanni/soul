"use strict";
const $ = (id) => document.getElementById(id);
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const modelLabel = (id) => id.startsWith("/") ? `local · ${id.split("/").pop()}` : id;
const number = (n) => Number(n || 0).toLocaleString();
const gib = (n) => `${(Number(n || 0) / 1024 ** 3).toFixed(1)} GiB`;
const badge = (status) => `<span class="badge ${escapeHTML(status)}">${escapeHTML(status)}</span>`;
let key = sessionStorage.getItem("soul-admin-key") || "", snapshot = null, selected = "", busy = false, logsNode = null;
function notice(text, success = false) { $("notice").textContent = text; $("notice").classList.toggle("success", success); $("notice").hidden = !text; }
async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {"Authorization": `Bearer ${key}`, "Content-Type": "application/json", ...options.headers}});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || data.detail || `Request failed (${response.status})`);
  return data;
}
function setTab(tab) {
  document.querySelectorAll(".view").forEach((el) => el.hidden = el.id !== `${tab}-view`);
  document.querySelectorAll(".nav-item").forEach((el) => el.classList.toggle("active", el.dataset.tab === tab));
  const titles = {overview:["Your network, in sync.","A single view of the machines bringing your models to life."], models:["A place for every layer.","Inspect model placement, memory reservations, and availability."], nodes:["Meet your network.","Connected hardware, download progress, and the work happening on each node."], requests:["Every request, accounted for.","A durable record of inference activity, without storing conversation content."]};
  $("page-title").textContent = titles[tab][0]; $("page-description").textContent = titles[tab][1]; $("crumb").textContent = tab[0].toUpperCase() + tab.slice(1);
}
document.querySelectorAll("[data-tab]").forEach((el) => el.onclick = () => setTab(el.dataset.tab));
document.querySelectorAll("[data-go]").forEach((el) => el.onclick = () => setTab(el.dataset.go));
document.querySelectorAll("[data-close]").forEach((el) => el.onclick = () => $(el.dataset.close).close());
$("auth-form").onsubmit = async (event) => { event.preventDefault(); key = $("admin-key").value.trim(); await refresh(true); };
$("disconnect").onclick = () => { key = ""; sessionStorage.removeItem("soul-admin-key"); location.reload(); };
$("open-load").onclick = () => { if (!key) { $("admin-key").focus(); return; } $("load-dialog").showModal(); };
$("load-form").onsubmit = async (event) => {
  event.preventDefault(); const model = $("model-id").value.trim(); if (!model) return;
  $("load-submit").disabled = true; $("load-dialog").close(); notice(`Inspecting and distributing ${model}. Download progress will appear on the nodes.`);
  try { await api("/admin/models/load", {method:"POST", body:JSON.stringify({model})}); selected = model; notice(`${model} is ready for inference.`, true); await refresh(); }
  catch (error) { notice(error.message); } finally { $("load-submit").disabled = false; }
};
$("pipeline-model").onchange = () => { selected = $("pipeline-model").value; renderTopology(); };
function renderJoinCommand() {
  const profile = $("join-profile").value;
  const device = profile === "node" ? "" : " --device cuda:0";
  $("join-command").textContent = `uv sync --locked --extra ${profile}\n\nexport SOUL_NODE_TOKEN='<node-token>'\nuv run --locked --extra ${profile} soul node --controller ${location.origin}${device}`;
}
$("join-profile").onchange = renderJoinCommand;
$("join-button").onclick = () => { renderJoinCommand(); $("join-dialog").showModal(); };
$("logs-dialog").addEventListener("close", () => logsNode = null);
async function refreshLogs() {
  if (!logsNode) return;
  try { const data = await api(`/admin/nodes/${encodeURIComponent(logsNode)}/logs`); $("logs-output").textContent = data.logs.map((log) => `${new Date(log.time * 1000).toLocaleTimeString()}  ${log.level.padEnd(5)}  ${log.message}`).join("\n") || "No events received yet."; }
  catch (error) { $("logs-output").textContent = error.message; }
}
document.addEventListener("click", async (event) => {
  const logs = event.target.closest("[data-logs]");
  if (logs) { logsNode = logs.dataset.logs; $("logs-title").textContent = `Logs · ${snapshot.nodes.find((n) => n.node_id === logsNode)?.name || logsNode}`; $("logs-output").textContent = "Loading…"; $("logs-dialog").showModal(); await refreshLogs(); }
  const unload = event.target.closest("[data-unload]");
  if (unload) { unload.disabled = true; try { await api("/admin/models/unload", {method:"POST",body:JSON.stringify({model:unload.dataset.unload})}); notice("Model unloaded. Its downloaded weights remain in the nodes’ disk cache.", true); await refresh(); } catch(error) { notice(error.message); } finally { unload.disabled = false; } }
});
function renderTopology() {
  const model = snapshot.models.find((m) => m.model === selected);
  if (!model) return;
  const topology = $("topology"); topology.classList.remove("empty-topology");
  const stages = model.assignments.map((assignment, index) => {
    const node = snapshot.nodes.find((n) => n.node_id === assignment.node_id);
    const status = node?.status || "offline";
    return `${index ? '<span class="stage-arrow" aria-hidden="true">→</span>' : ''}<article class="stage"><div class="stage-top"><span class="machine-icon">▣</span><div><div class="stage-name">${escapeHTML(node?.name || assignment.node_id)}</div><div class="stage-device">${escapeHTML(node?.capabilities?.device_name || "Disconnected")}</div></div></div><div class="stage-range"><span>Layers ${assignment.start + 1}–${assignment.end}</span><span>${assignment.end - assignment.start} / ${model.layers}</span></div><div class="layer-strip">${Array.from({length:Math.min(assignment.end-assignment.start,32)},()=>"<i></i>").join("")}</div><div class="stage-footer">${badge(status)}<span>${gib(assignment.estimated_bytes)} reserved</span></div></article>`;
  }).join("");
  topology.innerHTML = `<div class="controller-row"><span class="controller-orb">✳</span><div><strong>Controller ${badge(model.status)}</strong><small>Orchestration · token routing · API</small></div></div><div class="stage-list">${stages}</div>`;
}
function render() {
  const {nodes, models, metrics} = snapshot, connected = nodes.filter((n) => n.connected), ready = models.filter((m) => m.status === "ready");
  $("stat-nodes").textContent = connected.length; $("stat-models").textContent = ready.length; $("stat-requests").textContent = number(metrics.counts.completed); $("stat-tokens").textContent = number(metrics.output_tokens);
  $("node-count").textContent = connected.length; $("model-count").textContent = ready.length; $("node-detail").textContent = `${gib(connected.reduce((sum,n)=>sum+(n.capabilities.budget_bytes||0),0))} combined memory budget`;
  $("request-detail").textContent = `${number(metrics.counts.failed)} failed · ${number(metrics.counts.cancelled)} cancelled`;
  $("queue-detail").textContent = `${snapshot.queue_depth} waiting · ${snapshot.active_request ? "1 running" : "none running"}`;
  $("context-label").textContent = `${number(snapshot.max_context)} token context`;
  if (!models.some((m) => m.model === selected)) selected = ready[0]?.model || models[0]?.model || "";
  $("pipeline-model").innerHTML = models.length ? models.map((m) => `<option value="${escapeHTML(m.model)}" ${m.model === selected ? "selected" : ""}>${escapeHTML(modelLabel(m.model))}</option>`).join("") : '<option value="">No models loaded</option>';
  renderTopology();
  $("node-preview").innerHTML = nodes.length ? nodes.slice(0,4).map((n) => `<div class="node-row"><div class="machine-icon">▣</div><div class="node-info"><strong>${escapeHTML(n.name)}</strong><small>${escapeHTML(n.capabilities.device_name)} · ${gib(n.capabilities.budget_bytes)}</small></div>${badge(n.status)}<button class="text-button" data-logs="${escapeHTML(n.node_id)}">Logs ↗</button></div>`).join("") : '<div class="empty-small">Your nodes will appear here when they connect.</div>';
  $("activity").innerHTML = metrics.recent.length ? metrics.recent.slice(0,4).map((r) => `<div class="activity-row"><span class="activity-mark">${r.status === "completed" ? "✓" : "↗"}</span><div><strong>${escapeHTML(r.model.split('/').pop())}</strong><small>${escapeHTML(r.status)} · ${r.output_tokens} tokens · ${r.duration.toFixed(1)}s</small></div><time class="activity-time">${new Date(r.created*1000).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})}</time></div>`).join("") : '<div class="empty-small">No requests yet.<br><span>Ready when you are.</span></div>';
  $("models-table").innerHTML = models.length ? `<table><thead><tr><th>MODEL</th><th>ARCHITECTURE</th><th>DISTRIBUTION</th><th>STATUS</th><th></th></tr></thead><tbody>${models.map((m)=>`<tr><td>${escapeHTML(m.model)}<small>Revision ${escapeHTML(m.revision.slice(0,12))}</small></td><td>${escapeHTML(m.architecture)}<small>${m.context} token context</small></td><td>${m.layers} layers · ${m.assignments.length} nodes</td><td>${badge(m.status)}</td><td>${m.status === "ready" ? `<button data-unload="${escapeHTML(m.model)}">Unload</button>` : ''}</td></tr>`).join("")}</tbody></table>` : '<div class="empty-small">No model deployments yet. Load a model to get started.</div>';
  $("nodes-table").innerHTML = nodes.length ? `<table><thead><tr><th>NODE</th><th>HARDWARE</th><th>MEMORY BUDGET</th><th>STATE</th><th></th></tr></thead><tbody>${nodes.map((n)=>`<tr><td>${escapeHTML(n.name)}<small><code>${escapeHTML(n.node_id)}</code></small></td><td>${escapeHTML(n.capabilities.device_name)}<small>${escapeHTML(n.capabilities.dtype)} · ${escapeHTML(n.capabilities.weight_mode)} weights</small></td><td>${gib(n.capabilities.budget_bytes)}</td><td>${badge(n.status)}${n.total_bytes ? `<small>${gib(n.downloaded_bytes)} / ${gib(n.total_bytes)}</small><progress value="${Number(n.downloaded_bytes)}" max="${Number(n.total_bytes)}"></progress>`:''}${n.tensor ? `<small>${escapeHTML(n.tensor)}</small>`:''}${n.error ? `<small>${escapeHTML(n.error)}</small>`:''}</td><td><button data-logs="${escapeHTML(n.node_id)}">View logs</button></td></tr>`).join("")}</tbody></table>` : '<div class="empty-small">No nodes registered. Connect a node to contribute compute.</div>';
  $("requests-table").innerHTML = metrics.recent.length ? `<table><thead><tr><th>REQUEST</th><th>MODEL</th><th>STATUS</th><th>TOKENS IN / OUT</th><th>DURATION</th></tr></thead><tbody>${metrics.recent.map((r)=>`<tr><td><code>${escapeHTML(r.id.slice(0,20))}…</code><small>${new Date(r.created*1000).toLocaleString()}</small></td><td>${escapeHTML(r.model)}</td><td>${badge(r.status)}${r.error ? `<small>${escapeHTML(r.error)}</small>`:''}</td><td>${r.input_tokens} / ${r.output_tokens}</td><td>${r.duration.toFixed(2)}s</td></tr>`).join("")}</tbody></table>` : '<div class="empty-small">Requests appear here when you use an inference endpoint.</div>';
}
async function refresh(explicit=false) {
  if (!key || busy) return; busy = true;
  try {
    snapshot = await api("/admin/state"); sessionStorage.setItem("soul-admin-key", key); $("auth-panel").hidden = true; $("live-dot").classList.add("online"); $("live-label").textContent = "Controller connected"; $("updated").textContent = `Updated ${new Date().toLocaleTimeString()}`; render(); if (explicit) notice(""); await refreshLogs();
  } catch (error) { $("live-dot").classList.remove("online"); $("live-label").textContent = "Connection interrupted"; if (explicit || error.message.includes("API key")) { $("auth-panel").hidden = false; notice(error.message); } }
  finally { busy = false; }
}
refresh(); setInterval(refresh, 3000);
