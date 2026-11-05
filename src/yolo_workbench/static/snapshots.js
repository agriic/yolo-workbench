// The Dataset tab's Snapshots panel: take a snapshot (a job), list them, compare one with the dataset, download,
// keep, rename, delete, restore (in place, or as a new project in projects mode).
import { $, api, apiUrl, esc, formatBytes } from "./api.js";
import { state } from "./state.js";
import { personName, startedBy } from "./collab.js";
let deps;

export function configureSnapshots(dependencies) {
  deps = dependencies;
}

const SPLITS = ["train", "val", "test"];
const plural = (count, noun) => `${count.toLocaleString()} ${noun}${count === 1 ? "" : "s"}`;
const when = iso => new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
let items = [];
let retentionDays = 30;
let snapshotJob = null;
let restoreJob = null;
let comparing = null;  // the id whose comparison is shown

export function bindSnapshots() {
  $("snapshot-form").onsubmit = e => { e.preventDefault(); takeSnapshot(); };
  $("refresh-snapshots").onclick = () => loadSnapshots(true);
  $("snapshot-cancel").onclick = cancelJob;
  $("snapshot-list").addEventListener("click", e => {
    const button = e.target.closest("[data-snapshot-action]");
    if (!button) return;
    const item = items.find(entry => entry.id === button.dataset.snapshot);
    if (!item) return;
    const action = button.dataset.snapshotAction;
    if (action === "compare") return compare(item);
    if (action === "download") return download(item, button.dataset.kind);
    if (action === "keep") return update(item, { keep: true }, "Kept until you delete it");
    if (action === "rename") return rename(item);
    if (action === "delete") return remove(item);
    if (action === "recount") return recount(item);
    if (action === "project") return toProject(item);
    if (action === "restore") return restore(item);
    if (action === "apply-split") return deps.applySplit(item.rebalance.rules);
  });
  $("snapshot-compare").addEventListener("click", e => {
    const subset = e.target.closest("[data-subset]");
    if (subset) deps.showImages(subset.dataset.subset, subset.dataset.label);
    if (e.target.closest("[data-close-compare]")) closeCompare();
  });
}

export async function loadSnapshots(checkLinks = false) {
  try {
    const data = await api("/snapshots");
    items = data.items;
    retentionDays = data.retention_days;
    $("snapshots-total").textContent = items.length ? `${items.length} · ${formatBytes(data.total_bytes)} own` : "";
    renderList();
    // an open comparison follows the dataset (a rebalance, a restore, someone's edits)
    if (comparing) {
      const item = items.find(entry => entry.id === comparing);
      if (item) compare(item); else closeCompare();
    }
    if (checkLinks || !$("snapshot-link-note").dataset.checked) loadLinkNote();
    resumeJobs();
  } catch (error) { deps.toast(error.message, "error"); }
}

async function loadLinkNote() {
  const note = $("snapshot-link-note");
  try {
    const check = await api("/snapshots/link-check");
    note.dataset.checked = "1";
    note.classList.toggle("warn-text", check.mode === "copy" || check.mode === "mixed");
    note.textContent = check.mode === "copy"
      ? `This drive has no hardlinks: a snapshot copies all ${plural(check.images, "image")} (${formatBytes(check.copy_bytes)}).`
      : check.mode === "mixed"
        ? `${plural(check.copy_images, "image")} (${formatBytes(check.copy_bytes)}) are on a drive without hardlinks and are copied; the others take no space.`
        : "";
    note.hidden = !note.textContent;
  } catch { note.hidden = true; }
}

function kindNote(item) {
  if (item.kind === "auto") {
    if (item.used_by?.length) return `<span class="pill" title="Taken automatically; kept while training runs use it">auto</span>`;
    const days = retentionDays ? Math.max(0, Math.ceil(retentionDays - (Date.now() - new Date(item.created)) / 86400000)) : null;
    return `<span class="pill" title="Taken automatically${days !== null ? `; deleted in ${plural(days, "day")} unless kept` : ""}">auto${days !== null ? ` · ${days} d` : ""}</span>`;
  }
  if (item.kind === "run") return `<span class="pill" title="Taken for a training run; deleted with the last run that uses it">run</span>`;
  return "";
}

function renderList() {
  const projects = state.mode === "projects";
  $("snapshot-list").innerHTML = items.length ? items.map(item => {
    const splits = SPLITS.filter(split => item.splits?.[split]).map(split => `${split} ${item.splits[split].toLocaleString()}`).join(" · ");
    const size = item.own_bytes != null ? `${formatBytes(item.own_bytes)} own` : "";
    const runs = item.used_by?.length ? ` · ${plural(item.used_by.length, "training run")}` : "";
    const proposed = item.rebalance ? ` · <span title="Laid out by a rebalance plan; nothing moved in the dataset">proposed split</span>` : "";
    return `<div class="storage-row snapshot-row${comparing === item.id ? " active" : ""}">
      <div class="snapshot-main"><strong>${esc(item.name)}</strong> ${kindNote(item)}
        <div class="muted">${esc(when(item.created))}${item.by ? ` · ${esc(personName(item.by))}` : ""} · ${plural(item.images, "image")}${splits ? ` (${splits})` : ""}${size ? ` · <span title="Space only this snapshot takes; images it shares with the dataset or other snapshots take none">${size}</span>` : ""}${runs}${proposed}</div>
        ${item.note ? `<div class="snapshot-note">${esc(item.note)}</div>` : ""}</div>
      <span class="row-actions">
        <button class="secondary mini" data-snapshot-action="compare" data-snapshot="${esc(item.id)}" title="What changed in the dataset since this snapshot">Compare</button>
        <details class="snapshot-more"><summary class="secondary mini" title="More">⋯</summary><div class="snapshot-menu">
          <button class="link-btn" data-snapshot-action="download" data-kind="full" data-snapshot="${esc(item.id)}">Download images &amp; labels</button>
          ${item.category === "classification" ? "" : `<button class="link-btn" data-snapshot-action="download" data-kind="labels" data-snapshot="${esc(item.id)}">Download labels only</button>`}
          <button class="link-btn" data-snapshot-action="rename" data-snapshot="${esc(item.id)}">Rename / note…</button>
          ${item.kind !== "manual" ? `<button class="link-btn" data-snapshot-action="keep" data-snapshot="${esc(item.id)}" title="Keep it until you delete it">Keep</button>` : ""}
          <button class="link-btn" data-snapshot-action="recount" data-snapshot="${esc(item.id)}" title="Count the space only this snapshot takes">Recount size</button>
          ${projects && !item.rebalance ? `<button class="link-btn" data-snapshot-action="project" data-snapshot="${esc(item.id)}" title="A new project with this snapshot's images, labels and classes">New project from it…</button>` : ""}
          ${!item.rebalance ? `<button class="link-btn" data-snapshot-action="restore" data-snapshot="${esc(item.id)}" title="Make the dataset as it was in this snapshot">Restore…</button>` : `<button class="link-btn" data-snapshot-action="apply-split" data-snapshot="${esc(item.id)}" title="The rebalance dialog with this split's rules: apply it to the dataset">Apply this split…</button>`}
          <button class="link-btn danger-btn" data-snapshot-action="delete" data-snapshot="${esc(item.id)}">Delete…</button>
        </div></details>
      </span>
    </div>`;
  }).join("") : `<p class="panel-note">No snapshots yet.</p>`;
}

/* ---------- taking one (and other jobs of this panel) ---------- */

async function takeSnapshot() {
  const name = $("snapshot-name").value.trim();
  try {
    trackJob(await api("/snapshots", { method: "POST", body: JSON.stringify({ name, note: $("snapshot-note").value.trim() }) }));
    $("snapshot-name").value = "";
    $("snapshot-note").value = "";
  } catch (error) { deps.toast(error.message, "error"); }
}

export function trackJob(job, onDone = null) {
  const restoring = job.kind === "restore";
  if (restoring) restoreJob = job; else snapshotJob = job;
  showProgress(job);
  if (job.state === "running") {
    setTimeout(async () => {
      try { trackJob(await api(`/api/v1/jobs/${job.id}`), onDone); }
      catch (error) { finish(job); deps.toast(error.message, "error"); }
    }, 600);
    return;
  }
  finish(job);
  if (job.state === "done" && job.result) {
    if (onDone) onDone(job.result);
    else if (restoring) {
      deps.toast(`Restored “${job.result.name}” · ${plural(job.result.images, "image")}${job.result.before ? " — the state before is a snapshot too" : ""}`);
      deps.datasetChanged({ schema: true });
    } else if (job.kind === "snapshot") {
      const copied = job.result.copied ? ` · ${plural(job.result.copied, "image")} copied (${formatBytes(job.result.copied_bytes)})` : "";
      deps.toast(`Snapshot “${job.result.name}”: ${plural(job.result.images, "image")}${copied}`, job.result.missing ? "error" : "info");
    }
  } else if (job.state === "cancelled") {
    deps.toast(restoring ? "Restore cancelled — nothing changed" : "Snapshot cancelled", "error");
  } else if (job.state === "failed") {
    deps.toast(`${restoring ? "Restore" : "Snapshot"} failed: ${job.error}`, "error");
  }
  loadSnapshots();
}

function finish(job) {
  if (job.kind === "restore") restoreJob = null; else snapshotJob = null;
  showProgress(snapshotJob || restoreJob);
}

function showProgress(job) {
  $("snapshot-form").hidden = !!job;
  $("snapshot-progress").hidden = !job;
  if (!job) return;
  $("snapshot-progress-label").textContent = `${job.message || job.title}${startedBy(job.started_by)}`;
  $("snapshot-progress-count").textContent = job.total ? `${job.done.toLocaleString()} / ${job.total.toLocaleString()}` : "";
  $("snapshot-progress-bar").max = Math.max(1, job.total);
  $("snapshot-progress-bar").value = job.done;
  $("snapshot-cancel").disabled = job.cancel_requested;
}

async function cancelJob() {
  const job = snapshotJob || restoreJob;
  if (!job) return;
  try { showProgress(await api(`/api/v1/jobs/${job.id}/cancel`, { method: "POST" })); }
  catch (error) { deps.toast(error.message, "error"); }
}

// a snapshot or restore started earlier (another tab, a reload) shows its progress here too
async function resumeJobs() {
  if (snapshotJob || restoreJob) return;
  try {
    const query = state.project ? `?project=${encodeURIComponent(state.project)}` : "";
    const { items: jobs } = await api(`/api/v1/jobs${query}`);
    const running = jobs.find(job => (job.kind === "snapshot" || job.kind === "restore") && job.state === "running");
    if (running) trackJob(running);
  } catch { /* job listing is best effort */ }
}

/* ---------- comparing ---------- */

async function compare(item) {
  comparing = item.id;
  renderList();
  const panel = $("snapshot-compare");
  panel.hidden = false;
  panel.innerHTML = `<p class="panel-note">Comparing with “${esc(item.name)}”…</p>`;
  try {
    renderCompare(item, await api(`/snapshots/${encodeURIComponent(item.id)}/diff`));
  } catch (error) {
    panel.innerHTML = `<p class="panel-note warn-text">${esc(error.message)}</p>`;
  }
}

function closeCompare() {
  comparing = null;
  $("snapshot-compare").hidden = true;
  $("snapshot-compare").innerHTML = "";
  renderList();
}

const change = (then, now) => then === now ? `${now.toLocaleString()}` : `<span class="muted">${then.toLocaleString()} →</span> <strong>${now.toLocaleString()}</strong>`;

export function splitTable(then, now, names) {
  const splits = SPLITS.filter(split => then[split] || now[split]);
  const cell = (table, split, pick) => pick(table[split] ?? { images: 0, background: 0, objects: 0, classes: {} });
  const ids = [...new Set(splits.flatMap(split => [...Object.keys(then[split]?.classes ?? {}), ...Object.keys(now[split]?.classes ?? {})]))].sort((a, b) => a - b);
  const row = (label, pick) => `<tr><th>${label}</th>${splits.map(split => `<td>${change(cell(then, split, pick), cell(now, split, pick))}</td>`).join("")}</tr>`;
  return `<table class="split-table"><thead><tr><th></th>${splits.map(split => `<th><span class="split-badge">${split}</span></th>`).join("")}</tr></thead><tbody>
    ${row("Images", row => row.images)}${row("Objects", row => row.objects)}${row("Background", row => row.background)}
    ${ids.map(id => row(`<span class="chip" style="--c:var(--text-2)">${esc(names[id] ?? `class ${id}`)}</span>`, row => row.classes?.[id]?.images ?? 0)).join("")}
  </tbody></table>`;
}

function renderCompare(item, diff) {
  const { counts } = diff;
  const link = (kind, noun) => counts[kind] ? (diff.subsets[kind]
    ? `<button class="link-btn" data-subset="${esc(diff.subsets[kind])}" data-label="${esc(`${noun} since “${item.name}”`)}" title="Show them in the Annotation tab">${plural(counts[kind], "image")} ${noun}</button>`
    : `${plural(counts[kind], "image")} ${noun}`) : "";
  const parts = [link("added", "added"), counts.removed ? `${plural(counts.removed, "image")} removed` : "", link("moved", "moved to another split"), link("changed", state.meta.category === "classification" ? "with another class" : "relabelled")].filter(Boolean);
  const names = Object.fromEntries((diff.names.now.length >= diff.names.then.length ? diff.names.now : diff.names.then).map((name, id) => [id, name]));
  const classNote = diff.names.same ? "" : `<p class="panel-note warn-text">The classes changed: ${esc(diff.names.then.join(", ") || "none")} → ${esc(diff.names.now.join(", ") || "none")}.</p>`;
  $("snapshot-compare").innerHTML = `
    <div class="panel-head"><h3>Since “${esc(item.name)}”</h3><button class="link-btn" data-close-compare>Close</button></div>
    <p class="panel-note">${parts.length ? parts.join(" · ") : "No image was added, removed, moved or relabelled."}${counts.unchanged ? ` · ${plural(counts.unchanged, "image")} as they were` : ""}</p>
    ${classNote}
    ${splitTable(diff.table.then, diff.table.now, names)}
    ${diff.removed.length ? `<details class="snapshot-removed"><summary>Removed images</summary><ul>${diff.removed.slice(0, 200).map(path => `<li><code>${esc(path)}</code></li>`).join("")}${diff.removed.length > 200 ? `<li class="muted">… ${plural(diff.removed.length - 200, "more")}</li>` : ""}</ul></details>` : ""}`;
}

/* ---------- the menu ---------- */

function download(item, kind) {
  const link = document.createElement("a");
  link.href = apiUrl(`/snapshots/${encodeURIComponent(item.id)}/download?kind=${kind}`);
  link.download = "";
  link.click();
}

async function update(item, body, message) {
  try {
    await api(`/snapshots/${encodeURIComponent(item.id)}`, { method: "PATCH", body: JSON.stringify(body) });
    if (message) deps.toast(message);
    loadSnapshots();
  } catch (error) { deps.toast(error.message, "error"); }
}

function rename(item) {
  const name = prompt("Snapshot name:", item.name);
  if (name === null) return;
  const note = prompt("Note (optional):", item.note || "");
  update(item, { name: name.trim() || item.name, note: note === null ? item.note : note });
}

async function remove(item) {
  const runs = item.used_by?.length ? ` ${plural(item.used_by.length, "training run")} trained on it; their data lists will point at files that are gone.` : "";
  if (!confirm(`Delete snapshot “${item.name}”? Images it shares with the dataset stay in the dataset.${runs} This cannot be undone.`)) return;
  try {
    await api(`/snapshots/${encodeURIComponent(item.id)}`, { method: "DELETE" });
    if (comparing === item.id) closeCompare();
    deps.toast(`Deleted snapshot “${item.name}”`);
    loadSnapshots();
  } catch (error) { deps.toast(error.message, "error"); }
}

async function recount(item) {
  try {
    trackJob(await api(`/snapshots/${encodeURIComponent(item.id)}/recount`, { method: "POST" }), result => deps.toast(`“${result.name}”: ${formatBytes(result.own_bytes)} of its own, ${formatBytes(result.shared_bytes)} shared`));
  } catch (error) { deps.toast(error.message, "error"); }
}

async function toProject(item) {
  const name = prompt("Name of the new project:", `${state.meta.project?.name ?? "Project"} — ${item.name}`);
  if (!name?.trim()) return;
  try {
    const project = await api(`/snapshots/${encodeURIComponent(item.id)}/project`, { method: "POST", body: JSON.stringify({ name: name.trim() }) });
    deps.toast(`Created project “${project.name}”`);
    if (confirm(`Open “${project.name}” now?`)) location.assign(`/?project=${encodeURIComponent(project.slug)}`);
  } catch (error) { deps.toast(error.message, "error"); }
}

async function restore(item) {
  if (!confirm(`Make the dataset as it was in “${item.name}” (${when(item.created)})?\n\nImages added since go to the trash, removed ones come back, moved ones go back to their split, and labels and classes are put back. The state right now is kept as a snapshot first. The undo history starts over.`)) return;
  try {
    trackJob(await api(`/snapshots/${encodeURIComponent(item.id)}/restore`, { method: "POST" }));
  } catch (error) { deps.toast(error.message, "error"); }
}
