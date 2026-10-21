// Training tab: run history, the live view of the active run, the header chip, and the
// read-only banner while this dataset is training. The queue itself lives on the server
// (training.py); this module only polls it.
import { $, api, apiUrl, esc, formatBytes } from "./api.js";
import { state } from "./state.js";
import { personName } from "./collab.js";
import { LineChart, formatValue } from "./charts.js";
import { configureRunForm, loadTrainingOptions, openRunForm, trainingOptions } from "./trainform.js";
import { MAX_COMPARE, bestEpochRow, destroyCompare, renderCompare, shortModel } from "./compare.js";

let deps;
export function configureTraining(dependencies) {
  deps = dependencies;
  configureRunForm({
    toast: dependencies.toast,
    closeForm: () => { formOpen = false; selectedId = null; detail = null; loadRuns(); },
    started: runId => { formOpen = false; selectedId = runId; detail = null; loadRuns(); },
  });
}

const LIVE_POLL = 1000;       // selected run while it trains (tab visible)
const LIST_POLL = 3000;       // runs list while something is active or queued
const IDLE_LIST_POLL = 15000;
const QUEUE_POLL = 5000;      // header chip / lock, any tab
const LOG_POLL = 2000;
const EXPORT_POLL = 1000;     // a running ONNX export of the selected run

const ACTIVE = new Set(["starting", "preparing", "training", "validating", "testing"]);
const STATUS_LABEL = {
  queued: "Queued", starting: "Starting", preparing: "Preparing", training: "Training", validating: "Final validation",
  testing: "Testing", completed: "Completed", paused: "Paused", stopped: "Stopped", interrupted: "Interrupted",
  failed: "Failed", external: "External",
};
// fixed categorical slots (validated in both themes): color follows the loss component / metric, never its rank.
// RF-DETR's terms: bbox (L1) and giou are box losses, ce the classification loss, mask_* the mask head; loss is their weighted total
const LOSS_SLOTS = { box_loss: 1, seg_loss: 4, cls_loss: 2, dfl_loss: 3, loss: 1 };
const RFDETR_LOSS_SLOTS = { bbox_loss: 1, ce_loss: 2, giou_loss: 3, mask_ce_loss: 4, mask_dice_loss: 5, loss: 6 };
const LOSS_ORDER = ["box_loss", "seg_loss", "cls_loss", "dfl_loss", "bbox_loss", "giou_loss", "ce_loss", "mask_ce_loss", "mask_dice_loss", "loss"];
const METRIC_GROUPS = [
  { key: "precision", label: "P", slot: 1 },
  { key: "recall", label: "R", slot: 2 },
  { key: "mAP50", label: "mAP50", slot: 3 },
  { key: "mAP50-95", label: "mAP50-95", slot: 4 },
];
const slotColor = slot => `var(--series-${slot})`;

let runs = [];
let category = "detection";
let selectedId = null;
let detail = null;        // GET /training/runs/{id} of the selected run
let epochRows = [];
let liveState = null;     // runner state.json while the selected run is active
let charts = null;
let logOffset = null;
let listTimer = null;
let liveTimer = null;
let logTimer = null;
let queueTimer = null;
let exportTimer = null;
let lastLock = undefined;
let selectToken = 0;
let formOpen = false;     // the new-run form owns the detail pane
let compareOpen = false;  // so does the comparison
const compareIds = [];    // runs ticked for comparison, in tick order
const compareSlots = new Map();  // run id -> color slot, kept while ticked (color follows the run)
let queueState = { active: null, queued: [] };

const visible = () => $("training").classList.contains("active");
const pill = status => `<span class="run-status status-${esc(status)}">${ACTIVE.has(status) ? '<span class="live-dot" aria-hidden="true"></span>' : ""}${esc(STATUS_LABEL[status] ?? status)}</span>`;
const primaryMetric = () => category === "classification" ? "metrics/accuracy_top1" : category === "segmentation" ? "metrics/mAP50-95(M)" : "metrics/mAP50-95(B)";
const primaryLabel = () => category === "classification" ? "top-1" : category === "segmentation" ? "mask mAP50-95" : "mAP50-95";
const isRfdetr = run => run?.framework === "rfdetr";
// what the best weights are called in this run (Ultralytics' best.pt; RF-DETR's checkpoint_best_*.pth)
const bestName = run => isRfdetr(run) ? "best weights" : "best.pt";

function duration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "–";
  if (seconds < 10) return `${Math.max(0, seconds).toFixed(1)}s`;
  seconds = Math.round(seconds);
  const h = Math.floor(seconds / 3600), m = Math.floor(seconds % 3600 / 60), s = seconds % 60;
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : m ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`;
}

// a resumed or fine-tuned run's model is a weights path; the file name and its run folder say enough
const modelName = model => model ? shortModel(model) : "?";

function relativeTime(iso) {
  const then = Date.parse(iso);
  if (!then) return "";
  const minutes = Math.round((Date.now() - then) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)} h ago`;
  return new Date(then).toLocaleDateString();
}

// ---------- tab lifecycle ----------

export function bindTraining() {
  $("refresh-runs").onclick = () => loadRuns();
  $("new-run").onclick = () => openForm();
  $("run-list").addEventListener("click", e => {
    const item = e.target.closest("[data-run]");
    if (item) selectRun(item.dataset.run);
  });
  $("run-list").addEventListener("change", e => {
    const box = e.target.closest("[data-compare]");
    if (!box) return;
    toggleCompare(box.dataset.compare, box.checked);
    renderRunList();
    if (compareOpen) compareIds.length >= 2 ? openCompare() : (compareOpen = false, destroyCompare(), loadRuns());
  });
  $("compare-bar").addEventListener("click", e => {
    if (e.target.closest("[data-compare-open]")) openCompare();
    if (e.target.closest("[data-compare-clear]")) {
      [...compareIds].forEach(id => toggleCompare(id, false));
      renderRunList();
      if (compareOpen) { compareOpen = false; destroyCompare(); loadRuns(); }
    }
  });
  $("queue-panel").addEventListener("click", onQueueClick);
  // the submit event fires synchronously with the click (the dialog's close event is queued as a task)
  $("run-edit-dialog").querySelector("form").addEventListener("submit", e => { if (e.submitter?.value === "save") saveEdit(); });
  $("run-export-dialog").querySelector("form").addEventListener("submit", e => {
    if (e.submitter?.value !== "export") return;
    if (!exportOptions()) { e.preventDefault(); $("run-export-imgsz").focus(); return; }
    startExport();
  });
  $("run-detail").addEventListener("click", onDetailClick);
  $("training-chip").onclick = openActiveRun;
  $("training-lock-banner").addEventListener("click", e => { if (e.target.closest("[data-open-active]")) openActiveRun(); });
  $("plot-dialog").addEventListener("click", e => { if (e.target === $("plot-dialog") || e.target.closest("[data-close]")) $("plot-dialog").close(); });
  applyLock(state.meta.training_lock);
  pollQueue();
}

export function showTraining(runId = null) {
  if (runId) selectedId = runId;
  loadRuns();
}

export function hideTraining() {
  clearTimeout(listTimer);
  clearTimeout(liveTimer);
  clearTimeout(logTimer);
}

async function loadRuns() {
  clearTimeout(listTimer);
  try {
    const data = await api("/training/runs");
    runs = data.items;
    category = data.category;
    $("runs-root").textContent = data.runs_root;
    $("runs-root").title = "Runs are read from this folder, including runs made with the yolo CLI";
  } catch (error) {
    deps.toast(error.message, "error");
  }
  if (!visible()) return;
  runs.sort((a, b) => rank(a) - rank(b) || (b.created || "").localeCompare(a.created || ""));
  for (const id of [...compareIds]) if (!runs.some(run => run.id === id)) toggleCompare(id, false);
  renderRunList();
  renderQueuePanel();
  const busy = runs.some(run => run.active || run.status === "queued");
  if (formOpen || compareOpen) { listTimer = setTimeout(loadRuns, busy ? LIST_POLL : IDLE_LIST_POLL); return; }
  if (selectedId && !runs.some(run => run.id === selectedId)) selectedId = null;  // deleted on disk
  if (!selectedId && runs.length) selectedId = (runs.find(run => run.active) ?? runs[0]).id;
  if (selectedId && selectedId !== detail?.id) await selectRun(selectedId);
  else if (!selectedId) renderNoRun();
  else refreshSelectedSummary();
  listTimer = setTimeout(loadRuns, busy ? LIST_POLL : IDLE_LIST_POLL);
}

const rank = run => run.active ? 0 : run.status === "queued" ? 1 + (run.queue_position ?? 0) / 1000 : 2;

function renderRunList() {
  $("run-count").textContent = runs.length ? runs.length : "";
  if (!runs.length) {
    $("run-list").innerHTML = `<p class="empty-note">No training runs yet. Runs started here, and runs made with <code>yolo train</code> or RF-DETR in this folder, show up in this list.</p>`;
    return;
  }
  $("run-list").innerHTML = runs.map(run => {
    const metric = run.best_metrics?.[primaryMetric()];
    const test = run.test_metrics?.[primaryMetric()];
    const progress = run.active && run.epochs ? Math.min(1, (run.epoch || 0) / run.epochs) : null;
    const ticked = compareIds.includes(run.id);
    return `<div class="run-row ${ticked ? "ticked" : ""}">
      <input type="checkbox" class="run-compare" data-compare="${esc(run.id)}" ${ticked ? "checked" : ""} ${!ticked && compareIds.length >= MAX_COMPARE ? "disabled" : ""} aria-label="Compare ${esc(run.name)}" title="Select for comparison"${ticked ? ` style="--c: var(--series-${compareSlots.get(run.id)})"` : ""}>
      <button type="button" class="run-item ${run.id === selectedId ? "selected" : ""}" data-run="${esc(run.id)}" aria-current="${run.id === selectedId}">
      <span class="run-item-head"><strong>${esc(run.name)}</strong>${pill(run.status)}</span>
      <span class="run-item-meta">${esc(modelName(run.model))}${run.imgsz ? ` · ${esc(run.imgsz)}px` : ""} · ${run.status === "queued" ? `#${run.queue_position} in queue` : `${run.epochs_done}/${esc(run.epochs ?? "?")} ep`}</span>
      ${metric !== undefined ? `<span class="run-item-metric"><strong>${formatValue(metric)}</strong> ${esc(primaryLabel())} <span class="muted">★${run.best_epoch}</span>${test !== undefined ? ` · test <strong>${formatValue(test)}</strong>` : ""}</span>` : ""}
      ${progress !== null ? `<span class="run-item-progress"><span style="width:${(progress * 100).toFixed(1)}%"></span></span>` : ""}
      <span class="run-item-time">${esc(relativeTime(run.created))}${run.created_by && run.created_by.name !== state.user?.name ? ` · ${esc(personName(run.created_by))}` : ""}</span>
    </button></div>`;
  }).join("");
  renderCompareBar();
}

// ---------- comparison ----------

function toggleCompare(id, on) {
  const index = compareIds.indexOf(id);
  if (on && index < 0 && compareIds.length < MAX_COMPARE) {
    compareIds.push(id);
    const used = new Set(compareSlots.values());
    compareSlots.set(id, [1, 2, 3, 4, 5, 6, 7, 8].find(slot => !used.has(slot)));
  } else if (!on && index >= 0) {
    compareIds.splice(index, 1);
    compareSlots.delete(id);
  }
}

function renderCompareBar() {
  const bar = $("compare-bar");
  bar.hidden = compareIds.length === 0;
  bar.innerHTML = `<span>${compareIds.length} selected</span>
    <button type="button" class="secondary mini" data-compare-clear>Clear</button>
    <button type="button" class="primary-btn mini" data-compare-open ${compareIds.length < 2 ? "disabled title=\"Select at least two runs\"" : ""}>Compare</button>`;
}

async function openCompare() {
  if (compareIds.length < 2) return;
  compareOpen = true;
  formOpen = false;
  selectedId = null;
  detail = null;
  ++selectToken;
  clearTimeout(liveTimer);
  clearTimeout(logTimer);
  destroyCharts();
  $("run-list").querySelectorAll(".run-item").forEach(item => { item.classList.remove("selected"); item.setAttribute("aria-current", "false"); });
  const frameworks = trainingOptions()?.frameworks ?? (await loadTrainingOptions().catch(() => null))?.frameworks ?? null;
  await renderCompare($("run-detail"), {
    ids: [...compareIds], slots: compareSlots, category, frameworks,
    onClose: () => { compareOpen = false; destroyCompare(); loadRuns(); },
    onOpen: id => selectRun(id),
  });
}

// ---------- queue panel (all projects share one queue) ----------

function renderQueuePanel() {
  const panel = $("queue-panel");
  if (!panel) return;
  const { active, queued, held, held_reason: heldReason, pausing } = queueState;
  panel.hidden = !queued.length && !held && !active;
  if (panel.hidden) return;
  const mine = item => state.mode !== "projects" || item.project === state.project;
  const row = (item, index) => `<li class="queue-item ${mine(item) ? "" : "other-project"}">
      <span class="queue-pos">${index + 1}</span>
      <span class="queue-name" title="${esc(item.name)}${mine(item) ? "" : ` (project ${esc(item.project_name)})`}">${esc(item.name)}${mine(item) ? "" : ` <span class="muted">· ${esc(item.project_name)}</span>`}</span>
      <span class="queue-actions">
        <button type="button" class="icon-btn mini" data-queue-move="${index}" data-delta="-1" ${index === 0 ? "disabled" : ""} title="Move up" aria-label="Move ${esc(item.name)} up">↑</button>
        <button type="button" class="icon-btn mini" data-queue-move="${index}" data-delta="1" ${index === queued.length - 1 ? "disabled" : ""} title="Move down" aria-label="Move ${esc(item.name)} down">↓</button>
        ${mine(item) ? `<button type="button" class="icon-btn mini" data-queue-remove="${esc(item.id)}" title="Remove from queue" aria-label="Remove ${esc(item.name)} from the queue">×</button>` : ""}
      </span>
    </li>`;
  // a held queue starts nothing new (also after a restart); "Hold & pause" also pauses the active run at its
  // next checkpoint, and it resumes by itself once the queue is released (maintenance)
  const holdButtons = held
    ? `<button type="button" class="link-btn" data-queue-hold="release" title="Let queued runs start again">Release</button>`
    : `<button type="button" class="link-btn" data-queue-hold="hold" title="Start nothing new until released, also after a restart">Hold</button>${active && !pausing ? ` · <button type="button" class="link-btn" data-queue-hold="pause" title="Hold the queue and pause the active run at its next checkpoint; it resumes when the queue is released">Hold &amp; pause</button>` : ""}`;
  const notes = [
    held ? `<div class="queue-note">${esc(heldReason || "Held")}: nothing new starts until it is released.</div>` : "",
    pausing && active ? `<div class="queue-note">${esc(active.name)} pauses at its next checkpoint and resumes once the queue is released.</div>` : "",
  ].join("");
  panel.innerHTML = `<div class="panel-head"><h3>Queue${held ? " · held" : ""}</h3><span class="muted">${active && queued.length ? `after ${esc(active.name)}` : ""}</span><span class="queue-hold">${holdButtons}</span></div>
    ${notes}${queued.length ? `<ol class="queue-list">${queued.map(row).join("")}</ol>` : ""}`;
}

async function onQueueClick(e) {
  const move = e.target.closest("[data-queue-move]");
  const remove = e.target.closest("[data-queue-remove]");
  const hold = e.target.closest("[data-queue-hold]");
  try {
    if (hold) {
      const action = hold.dataset.queueHold;
      if (action === "pause" && !confirm("Hold the queue and pause the active run at its next checkpoint? It resumes when the queue is released.")) return;
      queueState = await api("/api/v1/training/queue/hold", { method: "POST", body: JSON.stringify({ held: action !== "release", pause_active: action === "pause" }) });
      renderQueuePanel();
      renderChip(queueState.active, queueState.queued.length, queueState.held);
      deps.toast(action === "release" ? "Queue released" : action === "pause" ? "Queue held; the active run pauses at its next checkpoint" : "Queue held");
      loadRuns();
    } else if (move) {
      const index = +move.dataset.queueMove;
      const target = index + +move.dataset.delta;
      const keys = queueState.queued.map(item => item.key);
      [keys[index], keys[target]] = [keys[target], keys[index]];
      queueState = await api("/api/v1/training/queue/reorder", { method: "POST", body: JSON.stringify({ order: keys }) });
      renderQueuePanel();
      loadRuns();
    } else if (remove) {
      await api(`/training/runs/${encodeURIComponent(remove.dataset.queueRemove)}/cancel`, { method: "POST" });
      deps.toast("Removed from the queue");
      if (selectedId === remove.dataset.queueRemove) { selectedId = null; detail = null; }
      await pollQueue();
      loadRuns();
    }
  } catch (error) {
    deps.toast(error.message, "error");
  }
}

// ---------- rename / notes ----------

function openEditDialog() {
  if (!detail) return;
  $("run-edit-name").value = detail.name;
  $("run-edit-notes").value = detail.notes || "";
  $("run-edit-dialog").showModal();
}

async function saveEdit() {
  if (!detail) return;
  const id = detail.id;
  try {
    await api(`/training/runs/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify({ name: $("run-edit-name").value, notes: $("run-edit-notes").value }) });
    deps.toast("Run updated");
    await reloadDetail();
    loadRuns();
  } catch (error) {
    deps.toast(error.message, "error");
  }
}

// ---------- ONNX export of the best weights ----------

function onnxButtons(run) {
  if (!run.onnx) return [];
  const { job, available, stale } = run.onnx;
  if (job?.state === "running") {
    return [`<button class="secondary" data-export-cancel title="${esc(job.message || "Starting the exporter…")}
Click to cancel"><span class="live-dot" aria-hidden="true"></span> Exporting ONNX…</button>`];
  }
  const buttons = [`<button class="secondary" data-export-onnx title="Export ${run.active ? "the best weights so far" : "the best weights"} to ONNX (${esc(run.onnx.file)}, next to them)">${available ? "Re-export ONNX…" : "Export ONNX…"}</button>`];
  if (available) {
    const facts = [formatBytes(run.onnx.bytes), `exported ${new Date(run.onnx.exported * 1000).toLocaleString()}`, onnxOptionsText(run.onnx.options)].filter(Boolean).join(" · ");
    const note = stale ? `
The best weights changed after this export; export again for the current ones.` : "";
    buttons.unshift(`<button class="secondary" data-download-onnx title="Download ${esc(run.onnx.file)} (${esc(facts)}), named dataset-run-imgsz-model-date.onnx${note}">Download ONNX${stale ? " (outdated)" : ""}</button>`);
  }
  return buttons;
}

function onnxOptionsText(options) {
  if (!options) return "";
  const imgsz = Array.isArray(options.imgsz) ? options.imgsz.join("×") : options.imgsz;
  return [imgsz && `${isRfdetr(detail) ? "resolution" : "imgsz"} ${imgsz}`, options.dynamic && "dynamic", options.simplify && "simplified", options.nms && "with NMS"].filter(Boolean).join(", ");
}

async function openExportDialog() {
  const run = detail;
  const rfdetr = isRfdetr(run);
  if (rfdetr && !trainingOptions()) await loadTrainingOptions().catch(() => null);  // the variants' resolution steps
  const previous = run.onnx?.options || {};
  $("run-export-intro").innerHTML = `Writes <code>${esc(run.onnx.file)}</code> next to ${run.active ? "the best weights so far" : "the best weights"}.`;
  $("run-export-ultralytics").hidden = rfdetr;
  const step = rfdetrStep(run);
  $("run-export-imgsz-label").textContent = rfdetr ? "Resolution" : "Image size";
  $("run-export-imgsz-hint").textContent = rfdetr
    ? `The model's fixed input size${run.imgsz ? `; trained at ${run.imgsz} px, which is what it does best` : ""}${step ? `. A multiple of ${step}` : ""}`
    : "The model's fixed input size, unless the shape is dynamic";
  $("run-export-imgsz").placeholder = rfdetr ? `${run.imgsz ?? 640}, or height,width` : "640, or height,width";
  $("run-export-nms-row").hidden = category === "classification";
  $("run-export-dynamic-label").textContent = rfdetr ? "Dynamic batch size" : "Dynamic batch and image size";
  const imgsz = previous.imgsz ?? run.imgsz;
  $("run-export-imgsz").value = Array.isArray(imgsz) ? imgsz.join(",") : imgsz ?? "";
  $("run-export-simplify").checked = previous.simplify ?? true;
  $("run-export-nms").checked = previous.nms ?? false;
  $("run-export-dynamic").checked = previous.dynamic ?? false;
  $("run-export-imgsz").setCustomValidity("");
  $("run-export-dialog").showModal();
}

// the dialog's values as the export request, or null when the image size is not one or two whole numbers
function exportOptions() {
  const options = { dynamic: $("run-export-dynamic").checked };
  const rfdetr = isRfdetr(detail);
  const step = rfdetr ? rfdetrStep(detail) : null;
  const text = $("run-export-imgsz").value.trim();
  const sides = text ? text.split(/[\s,x×]+/).filter(Boolean).map(Number) : [];
  const whole = sides.length <= 2 && sides.every(side => Number.isInteger(side) && side >= 32 && side <= 8192);
  const valid = whole && (!step || sides.every(side => side % step === 0));
  $("run-export-imgsz").setCustomValidity(valid ? "" : whole ? `A multiple of ${step} for this model (e.g. ${Math.floor(sides[0] / step) * step || step})` : "One size, or height,width: whole pixels from 32 to 8192");
  $("run-export-imgsz").reportValidity();
  if (!valid) return null;
  const imgsz = sides.length ? (sides.length === 1 ? sides[0] : sides) : null;
  if (rfdetr) return { ...options, imgsz };
  return { ...options, imgsz, simplify: $("run-export-simplify").checked, nms: $("run-export-nms").checked && category !== "classification" };
}

// RF-DETR's resolution rule for the run's variant (patch size × windows), when the run names one
const rfdetrStep = run => trainingOptions()?.frameworks?.find(item => item.id === "rfdetr")?.pretrained?.find(item => item.name === run.variant)?.step ?? null;

async function startExport() {
  if (!detail) return;
  const id = detail.id;
  try {
    const job = await api(`/training/runs/${encodeURIComponent(id)}/export/onnx`, { method: "POST", body: JSON.stringify(exportOptions()) });
    if (detail?.id !== id) return;
    detail.onnx = { ...detail.onnx, job };
    renderActions();
    renderAlerts();
    pollExport(id, job.id);
  } catch (error) {
    deps.toast(error.message, "error");
  }
}

async function pollExport(runId, jobId) {
  clearTimeout(exportTimer);
  let job;
  try {
    job = await api(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
  } catch (error) {
    return;  // the server restarted: the run's summary no longer lists the job
  }
  if (job.state === "running") {
    if (detail?.id === runId && detail.onnx) { detail.onnx.job = job; renderActions(); }
    exportTimer = setTimeout(() => pollExport(runId, jobId), EXPORT_POLL);
    return;
  }
  if (job.state === "done") deps.toast(`Exported to ONNX (${formatBytes(job.result.bytes)})`);
  else if (job.state === "failed") deps.toast(job.error, "error");
  if (detail?.id !== runId) return;
  try {
    const data = await api(`/training/runs/${encodeURIComponent(runId)}`);
    if (detail?.id !== runId) return;
    detail.onnx = data.onnx;
    renderActions();
    renderAlerts();
  } catch (error) { /* the next reload shows it */ }
}

function renderNoRun() {
  destroyCharts();
  detail = null;
  $("run-detail").innerHTML = runs.length
    ? `<div class="empty">Select a run to see its progress and results.</div>`
    : `<div class="empty">No training runs yet.<br><br><button type="button" class="primary-btn" data-new-run>New run</button></div>`;
}

// ---------- the selected run ----------

function openForm(source = {}) {
  formOpen = true;
  compareOpen = false;
  destroyCompare();
  selectedId = null;
  detail = null;
  ++selectToken;
  clearTimeout(liveTimer);
  clearTimeout(logTimer);
  destroyCharts();
  $("run-list").querySelectorAll(".run-item").forEach(item => {
    item.classList.remove("selected");
    item.setAttribute("aria-current", "false");
  });
  const url = new URL(location.href);
  url.searchParams.delete("run");
  history.replaceState(null, "", url);
  openRunForm(source);
}

async function selectRun(id) {
  formOpen = false;
  if (compareOpen) { compareOpen = false; destroyCompare(); }
  const token = ++selectToken;
  selectedId = id;
  clearTimeout(liveTimer);
  clearTimeout(logTimer);
  setUrlRun(id);
  $("run-list").querySelectorAll(".run-item").forEach(item => {
    item.classList.toggle("selected", item.dataset.run === id);
    item.setAttribute("aria-current", String(item.dataset.run === id));
  });
  let data;
  try {
    data = await api(`/training/runs/${encodeURIComponent(id)}`);
  } catch (error) {
    if (token !== selectToken) return;
    deps.toast(error.message, "error");
    selectedId = null;
    renderNoRun();
    return;
  }
  if (token !== selectToken) return;
  detail = data;
  logOffset = null;
  epochRows = [];
  liveState = data.active ? data.state : null;
  renderDetailSkeleton();
  const epochs = await api(`/training/runs/${encodeURIComponent(id)}/epochs`).catch(() => ({ items: [], state: null }));
  if (token !== selectToken) return;
  epochRows = epochs.items;
  if (epochs.state) liveState = epochs.state;
  renderLive();
  if (detail.active) liveTimer = setTimeout(pollLive, LIVE_POLL);
}

function setUrlRun(id) {
  const url = new URL(location.href);
  url.searchParams.set("tab", "training");
  url.searchParams.set("run", id);
  history.replaceState(null, "", url);
}

async function pollLive() {
  clearTimeout(liveTimer);
  if (!detail || !visible()) return;
  const id = detail.id;
  const since = epochRows.length ? epochRows[epochRows.length - 1].epoch : 0;
  let data;
  try {
    data = await api(`/training/runs/${encodeURIComponent(id)}/epochs?since=${since}`);
  } catch (error) {
    if (detail?.id === id) liveTimer = setTimeout(pollLive, LIVE_POLL * 3);
    return;
  }
  if (detail?.id !== id) return;
  for (const row of data.items) {
    const index = epochRows.findIndex(item => item.epoch === row.epoch);
    if (index >= 0) epochRows[index] = row; else epochRows.push(row);
  }
  if (data.state) {
    liveState = data.state;
    renderLive();
    liveTimer = setTimeout(pollLive, LIVE_POLL);
  } else {
    await reloadDetail();  // the run left the active slot: final results, plots, status
  }
}

async function reloadDetail() {
  if (!detail) return;
  const id = detail.id;
  try {
    const data = await api(`/training/runs/${encodeURIComponent(id)}`);
    if (detail?.id !== id) return;
    detail = data;
    liveState = data.active ? data.state : null;
    const epochs = await api(`/training/runs/${encodeURIComponent(id)}/epochs`);
    if (detail?.id !== id) return;
    epochRows = epochs.items;
    renderDetailSkeleton();
    renderLive();
    if (detail.active) liveTimer = setTimeout(pollLive, LIVE_POLL);
  } catch (error) {
    if (/not found/i.test(error.message)) { selectedId = null; loadRuns(); }
  }
}

// the list poll noticed a status change of the selected run (e.g. queued → starting)
function refreshSelectedSummary() {
  const run = runs.find(item => item.id === selectedId);
  if (!run || !detail) return;
  if (run.status !== detail.status || run.active !== detail.active || run.queue_position !== detail.queue_position) reloadDetail();
}

function destroyCharts() {
  charts?.loss.destroy();
  charts?.metrics.destroy();
  charts = null;
}

function renderDetailSkeleton() {
  destroyCharts();
  const run = detail;
  const created = run.created ? new Date(run.created).toLocaleString() : "";
  // a fine-tuned RF-DETR run's model is the previous run's weights: its variant says which architecture
  const variant = isRfdetr(run) && run.variant && modelName(run.model) !== run.variant ? run.variant : null;
  const facts = [modelName(run.model), variant, run.imgsz && `${isRfdetr(run) ? "resolution" : "imgsz"} ${run.imgsz}`, run.batch !== undefined && run.batch !== null && `batch ${run.batch}`, created, run.created_by && `by ${personName(run.created_by)}`].filter(Boolean);
  $("run-detail").innerHTML = `
    <div class="run-head">
      <div class="run-title">
        <h1>${esc(run.name)} ${pill(run.status)} ${run.source === "external" ? `<span class="split-badge" title="Made outside the workbench">external</span>` : ""}</h1>
        <p class="section-subtitle">${facts.map(esc).join(" · ")}</p>
        <p class="section-subtitle"><code title="Run folder">${esc(run.path)}</code></p>
      </div>
      <div class="run-actions" id="run-actions"></div>
    </div>
    ${run.notes ? `<p class="run-notes">${esc(run.notes)}</p>` : ""}
    <div id="run-alerts"></div>
    <div id="run-progress"></div>
    <div id="run-kpis" class="stats-kpis run-kpis"></div>
    <div class="stats-grid run-grid">
      <article class="stats-panel"><h2>Losses</h2><p class="panel-note">Train solid, val dashed</p><div id="chart-loss"></div></article>
      <article class="stats-panel"><h2>${category === "classification" ? "Accuracy" : "Validation metrics"}</h2><p class="panel-note">${category === "segmentation" ? "Box solid, mask dashed" : "Per epoch on the val split"}</p><div id="chart-metrics"></div></article>
      <article class="stats-panel stats-wide"><h2>Epochs <span id="epoch-count" class="muted"></span></h2><div id="epoch-table" class="table-scroll epoch-table"></div></article>
      <article class="stats-panel" id="val-panel"><h2>Validation <span class="muted">${esc(bestName(run))} · val split</span></h2><div id="val-results"></div></article>
      <article class="stats-panel" id="test-panel"><h2>Test <span class="muted">${esc(bestName(run))} · test split</span></h2><div id="test-results"></div></article>
      <article class="stats-panel stats-wide" id="plots-panel"><h2>Plots</h2><div id="run-plots" class="plot-grid"></div></article>
      <article class="stats-panel"><h2>Arguments</h2><div id="run-args">${argsHtml(run.args)}</div></article>
      <article class="stats-panel"><details id="log-details" ${run.status === "failed" || run.status === "interrupted" ? "open" : ""}><summary><h2>Log</h2></summary><pre id="run-log" class="run-log">Loading…</pre></details></article>
    </div>`;
  charts = {
    loss: new LineChart($("chart-loss"), { label: "Training and validation losses per epoch" }),
    metrics: new LineChart($("chart-metrics"), { label: "Validation metrics per epoch", yMin: 0 }),
  };
  $("log-details").addEventListener("toggle", () => { if ($("log-details").open) loadLog(true); });
  if ($("log-details").open) loadLog(true);
  renderActions();
  renderResults();
  renderPlots();
  if (run.onnx?.job?.state === "running") pollExport(run.id, run.onnx.job.id);
}

const argValue = value => value === null || value === undefined ? "null" : typeof value === "object" ? JSON.stringify(value) : String(value);

// what was chosen for this run (differs from its framework's default), then everything the trainer recorded
function argsHtml(args) {
  const entries = Object.entries(args || {});
  if (!entries.length) return `<p class="empty-note">No arguments recorded.</p>`;
  const framework = trainingOptions()?.frameworks?.find(item => item.id === (detail?.framework || "ultralytics"));
  const defaults = framework?.defaults;
  const managed = new Set(framework?.managed ?? []);
  if (!trainingOptions()) {
    loadTrainingOptions().then(() => { if ($("run-args") && detail) $("run-args").innerHTML = argsHtml(detail.args); }).catch(() => {});
  }
  if (!defaults) return `<pre class="run-args">${esc(entries.map(([key, value]) => `${key}: ${argValue(value)}`).join("\n"))}</pre>`;
  // Ultralytics records an unset option as '' in args.yaml (e.g. device for auto)
  const isDefault = (key, value) => JSON.stringify(defaults[key]) === JSON.stringify(value) || (defaults[key] === null && value === "");
  const chosen = entries.filter(([key, value]) => !managed.has(key) && (key === "model" || (key in defaults && !isDefault(key, value))));
  return `<pre class="run-args">${esc(chosen.map(([key, value]) => `${key}: ${argValue(value)}`).join("\n")) || "(all defaults)"}</pre>
    ${entries.length > chosen.length ? `<details class="all-args"><summary>All ${entries.length} arguments</summary><pre class="run-args">${entries.map(([key, value]) => {
      const changed = chosen.some(([name]) => name === key);
      return `<span class="${changed ? "arg-changed" : ""}">${esc(`${key}: ${argValue(value)}`)}</span>`;
    }).join("\n")}</pre></details>` : ""}`;
}

function renderActions() {
  const run = detail;
  const buttons = [];
  if (run.active) {
    buttons.push(`<button class="secondary" data-stop="pause" title="Stop after the current epoch; the run can be resumed later">Pause</button>`);
    buttons.push(`<button class="secondary" data-stop="finish" title="Stop after the current epoch and run the final validation and test evaluation (cannot be resumed)">Finish early</button>`);
    buttons.push(`<button class="danger-btn" data-stop="kill" title="Stop immediately; resumable from the last completed epoch">Kill</button>`);
  }
  if (run.resumable) buttons.push(`<button class="primary-btn" data-resume>Resume</button>`);
  if (run.status === "queued") buttons.push(`<button class="secondary" data-cancel>Remove from queue</button>`);
  const weightsFile = kind => run.weights?.[kind]?.split("/").pop() ?? `${kind}.pt`;
  const extension = isRfdetr(run) ? "pth" : "pt";
  if (run.has_weights?.best) buttons.push(`<button class="secondary" data-download-best title="Download ${run.active ? "the best weights so far" : "the best weights"} (${esc(weightsFile("best"))}), named dataset-run-imgsz-model-date.${extension}">Download ${esc(bestName(run))}</button>`);
  buttons.push(...onnxButtons(run));
  if (run.has_weights?.best || run.has_weights?.last) buttons.push(`<button class="secondary" data-use-model title="Load this run's ${esc(weightsFile(run.has_weights.best ? "best" : "last"))} as the prediction model in the Annotation tab">Use for labeling</button>`);
  buttons.push(`<button class="secondary" data-duplicate title="Start a new run with the same arguments">Duplicate</button>`);
  buttons.push(`<button class="icon-btn" data-edit-run title="Rename / notes" aria-label="Rename or add notes"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg></button>`);
  if (run.has_weights?.best || run.has_weights?.last) buttons.push(`<button class="secondary" data-finetune title="Start a new run from this run's ${esc(weightsFile(run.has_weights.best ? "best" : "last"))}">Fine-tune</button>`);
  if (!run.active && run.status !== "queued") buttons.push(`<button class="icon-btn danger-btn" data-delete-run title="Delete run" aria-label="Delete this run"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/></svg></button>`);
  $("run-actions").innerHTML = buttons.join("");
}

async function onDetailClick(e) {
  if (e.target.closest("[data-new-run]")) return openForm();
  if (!detail) return;
  const id = detail.id;
  const stop = e.target.closest("[data-stop]");
  const plot = e.target.closest("[data-plot]");
  const remove = e.target.closest("[data-delete-run]");
  try {
    if (stop) {
      const mode = stop.dataset.stop;
      if (mode === "kill" && !confirm(`Kill "${detail.name}" now? The current epoch is lost; the run can be resumed from the last completed one.`)) return;
      stop.disabled = true;
      await api(`/training/runs/${encodeURIComponent(id)}/stop`, { method: "POST", body: JSON.stringify({ mode }) });
      deps.toast(mode === "kill" ? "Stopping the run…" : mode === "pause" ? "The run pauses after this epoch" : "The run finishes after this epoch");
    } else if (e.target.closest("[data-resume]")) {
      await api(`/training/runs/${encodeURIComponent(id)}/resume`, { method: "POST" });
      deps.toast("Run resumed");
      await loadRuns();
      await reloadDetail();
    } else if (e.target.closest("[data-cancel]")) {
      await api(`/training/runs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
      deps.toast("Removed from the queue");
      selectedId = null;
      detail = null;
      await loadRuns();
    } else if (remove) {
      const name = detail.name;
      if (!confirm(`Delete run "${name}" for good?\n\nIts folder is removed from disk with the weights, plots and log (${formatBytes(detail.bytes ?? 0)}):\n${detail.path}`)) return;
      remove.disabled = true;
      const result = await api(`/training/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
      deps.toast(`Deleted "${name}" (${formatBytes(result.bytes)} freed)`);
      if (selectedId === id) { selectedId = null; detail = null; }
      await loadRuns();
    } else if (e.target.closest("[data-edit-run]")) {
      openEditDialog();
    } else if (e.target.closest("[data-download-best]")) {
      // the server names the file (Content-Disposition); a link keeps the browser's own download progress
      const link = document.createElement("a");
      link.href = apiUrl(`/training/runs/${encodeURIComponent(id)}/weights/best`);
      link.download = "";
      link.click();
    } else if (e.target.closest("[data-download-onnx]")) {
      const link = document.createElement("a");
      link.href = apiUrl(`/training/runs/${encodeURIComponent(id)}/weights/onnx`);
      link.download = "";
      link.click();
    } else if (e.target.closest("[data-export-onnx]")) {
      openExportDialog();
    } else if (e.target.closest("[data-export-cancel]")) {
      if (!confirm("Cancel the ONNX export?")) return;
      await api(`/api/v1/jobs/${encodeURIComponent(detail.onnx.job.id)}/cancel`, { method: "POST" });
    } else if (e.target.closest("[data-use-model]")) {
      await deps.useModel(detail.weights.best ?? detail.weights.last);
    } else if (e.target.closest("[data-duplicate]")) {
      openForm({ from: "duplicate", runId: id });
    } else if (e.target.closest("[data-finetune]")) {
      openForm({ from: "finetune", runId: id });
    } else if (plot) {
      e.preventDefault();
      openPlot(plot.dataset.plot);
    }
  } catch (error) {
    deps.toast(error.message, "error");
    if (stop) stop.disabled = false;
    if (remove) { remove.disabled = false; loadRuns(); }  // a partly failed delete still takes the run off the list
  }
}

// ---------- live parts: progress, KPIs, charts, epoch table ----------

function renderLive() {
  if (!detail) return;
  const best = bestRow();
  renderProgress(best);
  renderKpis(best);
  renderAlerts();
  renderCharts(best);
  renderEpochTable(best);
}

const bestRow = () => bestEpochRow(epochRows, detail?.framework);

function renderProgress(best) {
  const el = $("run-progress");
  const s = liveState;
  if (!detail.active || !s) { el.innerHTML = ""; return; }
  const epochs = s.epochs || detail.epochs || 0;
  const epoch = s.epoch || 0;
  const batchFraction = s.batches ? Math.min(1, (s.batch || 0) / s.batches) : 0;
  const phase = s.status === "training" ? (s.phase === "validating-epoch" ? "Validating epoch" : "Training")
    : STATUS_LABEL[s.status] ?? s.status;
  const elapsed = s.started ? (Date.now() / 1000 - s.started) : null;
  const pending = s.status === "starting" || s.status === "preparing";
  el.innerHTML = `<div class="stats-panel run-progress">
    <div class="run-progress-row">
      <span class="run-progress-epoch"><strong>${pending ? "–" : epoch}</strong><span class="muted">/ ${esc(epochs || "?")} epochs</span></span>
      <span class="run-progress-phase">${pill(s.status)} ${esc(pending ? pendingText(s) : phase)}${s.resumed ? ' <span class="muted">(resumed)</span>' : ""}</span>
      <span class="run-progress-facts">
        <span><span class="muted">Elapsed</span> ${duration(elapsed)}</span>
        <span><span class="muted">ETA</span> ${s.status === "training" ? duration(s.eta) : "–"}</span>
        ${s.gpu_mem ? `<span><span class="muted">GPU</span> ${s.gpu_mem.toFixed(1)} GB</span>` : ""}
        ${best ? `<span><span class="muted">Best</span> ★ ${best.epoch}</span>` : ""}
      </span>
    </div>
    ${s.status === "training" ? `<div class="run-bars">
      <label>Epoch<progress max="1" value="${epochs ? (Math.max(0, epoch - 1) + batchFraction) / epochs : 0}"></progress></label>
      <label>Batch ${esc(s.batch ?? 0)}/${esc(s.batches ?? "?")}<progress max="1" value="${batchFraction}"></progress></label>
    </div>` : ""}
    ${detail.state?.stopped_early || s.stopped_early ? `<p class="panel-note">Finishing early: final validation and test evaluation follow.</p>` : ""}
  </div>`;
}

function pendingText(s) {
  if (s.phase === "preparing-data") return s.prepare_total ? `Preparing RF-DETR's dataset folder (${s.prepared ?? 0}/${s.prepare_total} images)…` : "Preparing RF-DETR's dataset folder…";
  return isRfdetr(detail) ? "Loading RF-DETR and its weights…" : "Loading the model and scanning the dataset…";
}

function renderKpis(best) {
  const metric = primaryMetric();
  const val = detail.val_metrics?.[metric];
  const test = detail.test_metrics?.[metric];
  const epochsDone = epochRows.length ? epochRows[epochRows.length - 1].epoch : 0;
  const tiles = [
    [`${epochsDone}<span class="kpi-of">/${esc(liveState?.epochs || detail.epochs || "?")}</span>`, detail.stopped_early ? "epochs (finished early)" : "epochs"],
    [best ? `★ ${best.epoch}` : "–", "best epoch"],
    [best ? formatValue(best[metric]) : "–", `best ${primaryLabel()} (val)`],
    [val !== undefined ? formatValue(val) : "–", `final val ${primaryLabel()}`],
    [test !== undefined ? formatValue(test) : "–", `test ${primaryLabel()}`],
  ];
  $("run-kpis").innerHTML = tiles.map(([value, label]) => `<div class="stats-kpi"><strong>${value}</strong><span>${esc(label)}</span></div>`).join("");
}

function renderAlerts() {
  const notes = [];
  if (detail.error && ["failed", "interrupted", "stopped"].includes(detail.status)) notes.push(`<div class="banner run-error"><strong>${esc(STATUS_LABEL[detail.status])}:</strong> ${esc(detail.error)}${detail.resumable ? " — it can be resumed from the last completed epoch." : ""}</div>`);
  if (detail.test_error) notes.push(`<div class="banner">Test evaluation failed: ${esc(detail.test_error)}</div>`);
  if (detail.onnx?.job?.state === "failed") notes.push(`<div class="banner run-error">${esc(detail.onnx.job.error)} <span class="muted">(log: <code>${esc(detail.path)}/workbench/export.log</code>)</span></div>`);
  if (detail.waiting) notes.push(`<div class="banner">Waiting: ${esc(detail.waiting)}</div>`);
  if (detail.status === "queued") notes.push(`<div class="banner">Queued as #${esc(detail.queue_position)}; it starts when the runs ahead of it finish.</div>`);
  $("run-alerts").innerHTML = notes.join("");
}

function seriesFor(prefixes, key) {
  return epochRows.map(row => {
    for (const prefix of prefixes) if (row[prefix + key] !== undefined) return row[prefix + key];
    return null;
  });
}

function renderCharts(best) {
  if (!charts) return;
  const x = epochRows.map(row => row.epoch);
  const keys = new Set(epochRows.flatMap(row => Object.keys(row)));
  const xMax = liveState?.epochs || detail.epochs || 0;
  const marker = best?.epoch ?? null;

  const lossSeries = [];
  const slots = isRfdetr(detail) ? RFDETR_LOSS_SLOTS : LOSS_SLOTS;
  for (const component of LOSS_ORDER) {
    const name = component.replace(/_loss$/, "") || component;
    for (const [split, dashed] of [["train", false], ["val", true]]) {
      if (!keys.has(`${split}/${component}`)) continue;
      lossSeries.push({
        key: `${split}/${component}`, label: `${split} ${name}`, group: component, groupLabel: name,
        color: slotColor(slots[component]), dashed, values: seriesFor([`${split}/`], component),
      });
    }
  }
  charts.loss.update({ x, series: lossSeries, marker, xMax, styles: lossSeries.some(s => s.dashed) ? [{ label: "train", dashed: false }, { label: "val", dashed: true }] : [] });

  const metricSeries = [];
  if (category === "classification") {
    [["metrics/accuracy_top1", "top-1", 1], ["metrics/accuracy_top5", "top-5", 2]].forEach(([key, label, slot]) => {
      if (keys.has(key)) metricSeries.push({ key, label, group: key, groupLabel: label, color: slotColor(slot), values: seriesFor([""], key) });
    });
  } else {
    for (const metric of METRIC_GROUPS) {
      for (const [suffix, dashed, kind] of [["(B)", false, "box"], ["(M)", true, "mask"]]) {
        const key = `metrics/${metric.key}${suffix}`;
        if (!keys.has(key)) continue;
        metricSeries.push({
          key, label: category === "segmentation" ? `${metric.label} ${kind}` : metric.label, group: metric.key, groupLabel: metric.label,
          color: slotColor(metric.slot), dashed, values: seriesFor([""], key),
        });
      }
    }
  }
  charts.metrics.update({
    x, series: metricSeries, marker, xMax,
    styles: category === "segmentation" ? [{ label: "box", dashed: false }, { label: "mask", dashed: true }] : [],
    empty: detail.active ? "Waiting for the first epoch…" : "No epochs recorded",
  });
}

function columnLabel(key) {
  let label = key.replace(/^metrics\//, "").replace(/^precision/, "P").replace(/^recall/, "R").replace(/^accuracy_/, "");
  if (category === "detection") label = label.replace("(B)", "");
  label = label.replace("(B)", " box").replace("(M)", " mask");
  if (key.startsWith("train/") || key.startsWith("val/")) label = key.replace(/_loss$/, "").replace("/", " ");
  if (key.startsWith("lr/")) label = key === "lr/pg0" ? "lr" : key.replace("lr/", "lr ");
  if (key === "epoch_time") label = "time";
  return label;
}

function epochColumns() {
  const keys = new Set(epochRows.flatMap(row => Object.keys(row)));
  const ordered = [...keys];
  const pick = test => ordered.filter(test).sort((a, b) => {
    const component = key => LOSS_ORDER.indexOf(key.split("/")[1]);
    return component(a) - component(b) || a.localeCompare(b);
  });
  const metricOrder = key => {
    const group = METRIC_GROUPS.findIndex(metric => key.startsWith(`metrics/${metric.key}(`));
    return (key.endsWith("(M)") ? 10 : 0) + (group < 0 ? 5 : group);
  };
  return [
    ...pick(key => key.startsWith("train/")),
    ...pick(key => key.startsWith("val/")),
    ...ordered.filter(key => key.startsWith("metrics/")).sort((a, b) => metricOrder(a) - metricOrder(b) || a.localeCompare(b)),
    ...(keys.has("lr/pg0") ? ["lr/pg0"] : []),
    ...(keys.has("epoch_time") ? ["epoch_time"] : []),
  ];
}

function renderEpochTable(best) {
  const columns = epochColumns();
  $("epoch-count").textContent = epochRows.length ? `${epochRows.length}` : "";
  if (!epochRows.length && !(detail.active && liveState?.running_losses)) {
    $("epoch-table").innerHTML = `<p class="empty-note">${detail.active ? "The first epoch is still running." : "No epochs recorded."}</p>`;
    return;
  }
  const cell = (row, key) => key === "epoch_time" ? duration(row[key]) : formatValue(row[key]);
  const running = detail.active && liveState?.status === "training" && liveState.running_losses
    ? `<tr class="running-row"><td title="Running averages of the epoch in progress">${esc(liveState.epoch)} <span class="muted">running</span></td>${columns.map(key => `<td>${key.startsWith("train/") ? formatValue(liveState.running_losses[key]) : ""}</td>`).join("")}</tr>`
    : "";
  const rows = [...epochRows].reverse().map(row => {
    const isBest = best && row.epoch === best.epoch;
    return `<tr class="${isBest ? "best-row" : ""}"><td>${isBest ? `<span class="best-star" title="Best epoch (${esc(bestName(detail))})">★</span> ` : ""}${row.epoch}</td>${columns.map(key => `<td>${cell(row, key)}</td>`).join("")}</tr>`;
  }).join("");
  $("epoch-table").innerHTML = `<table class="stats-table"><thead><tr><th>Epoch</th>${columns.map(key => `<th title="${esc(key)}">${esc(columnLabel(key))}</th>`).join("")}</tr></thead><tbody>${running}${rows}</tbody></table>`;
}

// ---------- final results and plots ----------

function resultsTable(result) {
  if (!result) return null;
  const metrics = Object.entries(result.metrics || {}).filter(([key]) => key.startsWith("metrics/"));
  const classes = result.classes || [];
  const overall = metrics.length ? `<div class="result-overall">${metrics.map(([key, value]) => `<span><strong>${formatValue(value)}</strong> ${esc(columnLabel(key))}</span>`).join("")}</div>` : "";
  if (!classes.length || classes.length === 1 && !classes[0].Class) return overall;
  // F1 and image counts are derivable or secondary; dropping them keeps the table inside a half-width panel
  const columns = Object.keys(classes[0]).filter(key => !/(^|-)F1$/.test(key) && key !== "Images");
  const format = (key, value) => typeof value === "string" ? esc(value) : /^(Images|Instances)$/.test(key) ? Math.round(value) : formatValue(value);
  return `${overall}<div class="table-scroll"><table class="stats-table"><thead><tr>${columns.map(key => `<th>${esc(key)}</th>`).join("")}</tr></thead><tbody>${classes.map(row => `<tr>${columns.map(key => `<td>${format(key, row[key])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function renderResults() {
  const val = resultsTable(detail.val);
  const test = resultsTable(detail.test);
  const done = !detail.active && detail.status !== "queued";
  $("val-results").innerHTML = val ?? `<p class="empty-note">${done ? "No final validation (the run did not finish)." : `Runs after training on ${esc(bestName(detail))}.`}</p>`;
  $("test-results").innerHTML = test ?? `<p class="empty-note">${detail.source === "external" ? "Not recorded for runs made outside the workbench." : done ? "No test results (the dataset has no test split, or the run did not finish)." : "Runs after training when the dataset has a test split."}</p>`;
}

function plotUrl(name) {
  return apiUrl(`/training/runs/${encodeURIComponent(detail.id)}/files/${name.split("/").map(encodeURIComponent).join("/")}`);
}

// the plots people open first; the rest (train batches, label stats) follow in name order
const PLOT_ORDER = ["results", "confusion_matrix_normalized", "PR_curve", "F1_curve", "P_curve", "R_curve", "confusion_matrix", "val_batch", "labels", "train_batch"];
const plotRank = name => {
  const base = name.replace(/^test\//, "");
  const index = PLOT_ORDER.findIndex(key => base.startsWith(key) || base.includes(`${key}.`) || base.replace(/^(Box|Mask)/, "").startsWith(key));
  return index < 0 ? PLOT_ORDER.length : index;
};

function renderPlots() {
  const plots = [...(detail.plots || [])].sort((a, b) => plotRank(a) - plotRank(b) || a.localeCompare(b));
  $("plots-panel").hidden = !plots.length;
  const stamp = detail.active ? `?t=${Date.now()}` : "";
  const groups = [["", plots.filter(name => !name.startsWith("test/"))], ["Test split", plots.filter(name => name.startsWith("test/"))]];
  $("run-plots").innerHTML = groups.filter(([, names]) => names.length).map(([title, names]) => `
    ${title ? `<h3 class="plot-group-title">${esc(title)}</h3>` : ""}
    ${names.map(name => `<a class="plot-thumb" href="${esc(plotUrl(name))}" data-plot="${esc(name)}" target="_blank" rel="noopener">
      <img src="${esc(plotUrl(name))}${stamp}" alt="${esc(name)}" loading="lazy"><span>${esc(name.replace(/^test\//, "").replace(/\.\w+$/, "").replace(/_/g, " "))}</span>
    </a>`).join("")}`).join("");
}

function openPlot(name) {
  $("plot-dialog-image").src = plotUrl(name);
  $("plot-dialog-image").alt = name;
  $("plot-dialog-title").textContent = name;
  $("plot-dialog-open").href = plotUrl(name);
  $("plot-dialog").showModal();
}

// ---------- log ----------

async function loadLog(reset = false) {
  clearTimeout(logTimer);
  if (!detail || !$("log-details")?.open) return;
  const id = detail.id;
  if (reset) logOffset = null;
  try {
    const data = await api(`/training/runs/${encodeURIComponent(id)}/log${logOffset !== null ? `?offset=${logOffset}` : ""}`);
    if (detail?.id !== id) return;
    const pre = $("run-log");
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
    if (logOffset === null) pre.textContent = (data.truncated ? "…\n" : "") + (data.text || "(empty log)");
    else if (data.text) pre.textContent += data.text;
    logOffset = data.offset;
    if (atBottom || reset) pre.scrollTop = pre.scrollHeight;
  } catch (error) {
    $("run-log").textContent = error.message;
  }
  if (detail?.active) logTimer = setTimeout(loadLog, LOG_POLL);
}

// ---------- header chip and dataset lock (all tabs) ----------

async function pollQueue() {
  clearTimeout(queueTimer);
  try {
    const queue = await api("/api/v1/training/queue");
    queueState = queue;
    if (visible()) renderQueuePanel();
    renderChip(queue.active, queue.queued.length, queue.held);
    const here = queue.active && (state.mode !== "projects" || queue.active.project === state.project);
    applyLock(here ? { reason: `training run '${queue.active.name}' is in progress`, run_id: queue.active.id, run: queue.active.name } : null);
  } catch { /* the chip is informational; try again later */ }
  queueTimer = setTimeout(pollQueue, QUEUE_POLL);
}

let chipRun = null;
function renderChip(active, queued, held = false) {
  chipRun = active;
  const chip = $("training-chip");
  if (!active && held) {
    chip.hidden = false;
    chip.innerHTML = `<span class="chip-text">Training queue held</span>${queued ? `<span class="chip-queue">+${queued}</span>` : ""}`;
    chip.title = "The training queue is held: nothing new starts until it is released (Training tab)";
    return;
  }
  if (!active) { chip.hidden = true; return; }
  chip.hidden = false;
  const metric = active.best_metrics?.[primaryMetric()] ?? null;
  const elsewhere = state.mode === "projects" && active.project !== state.project;
  const progress = active.status === "training" ? `${active.epoch ?? 0}/${active.epochs ?? "?"}` : STATUS_LABEL[active.status] ?? active.status;
  chip.innerHTML = `<span class="live-dot" aria-hidden="true"></span><span class="chip-text">${elsewhere ? `${esc(active.project_name)} · ` : ""}${esc(active.name)} · ${esc(progress)}${metric !== null ? ` · ★ ${formatValue(metric)}` : ""}</span>${queued ? `<span class="chip-queue">+${queued}</span>` : ""}`;
  chip.title = `Training ${active.name}${elsewhere ? ` (project ${active.project_name})` : ""}${queued ? `; ${queued} queued` : ""}${held ? "; queue held" : ""} — open the run`;
}

function openActiveRun() {
  if (!chipRun) return;
  if (state.mode === "projects" && chipRun.project !== state.project) {
    location.href = `/?project=${encodeURIComponent(chipRun.project)}&tab=training&run=${encodeURIComponent(chipRun.id)}`;
    return;
  }
  selectedId = chipRun.id;
  deps.showTab("training");
}

function applyLock(lock) {
  const key = lock ? lock.run_id : null;
  if (key === lastLock) return;
  const wasLocked = !!lastLock;
  lastLock = key;
  state.meta.training_lock = lock;
  document.body.classList.toggle("training-locked", !!lock);
  const banner = $("training-lock-banner");
  banner.hidden = !lock;
  if (lock) banner.innerHTML = `<strong>Read-only:</strong> ${esc(lock.reason)}. Changes to labels, classes, splits and imports are refused until it ends. <button type="button" class="link-btn" data-open-active>View run</button>`;
  if (wasLocked && !lock) deps.toast("Training finished — the dataset can be edited again");
  if (visible() && wasLocked !== !!lock) loadRuns();
}
