import { $, api, classColor, esc, pointInPolygon } from "./api.js";
import { state } from "./state.js";
let deps;

export function configurePredictor(dependencies) {
  deps = dependencies;
}

/* ---------- model-assisted labeling ---------- */

const UNMAPPED_COLOR = "#8b8f98";

export async function refreshPredictor() {
  try {
    applyPredictorState(await api("/api/v1/predictor"));
    if (state.pred.status !== "unavailable") await loadModelOptions();
  } catch (error) { deps.toast(error.message, "error"); }
}

export async function loadModelOptions() {
  try {
    state.pred.models = (await api("/api/v1/predictor/models")).items;
  } catch { state.pred.models = []; }
  // the most recently used model is the best default for a returning session
  const recent = state.pred.models.find(item => item.source === "recent");
  if (recent && !$("model-path").value) $("model-path").value = recent.path;
  $("model-browse").hidden = !state.pred.models.length;
  renderModelMenu();
}

export function renderModelMenu() {
  const items = state.pred.models || [];
  $("model-menu").innerHTML = items.map(item => `
    <button type="button" class="model-option ${item.path === state.pred.modelPath ? "active" : ""}" data-path="${esc(item.path)}" title="${esc(item.path)}">
      <span class="model-option-name">${esc(item.name)}${item.source === "recent" ? ' <span class="model-recent">recent</span>' : ""}${item.path === state.pred.modelPath ? ' <span class="model-recent">loaded</span>' : ""}</span>
      <span class="model-option-meta">${esc(shortDir(item.path))} · ${formatSize(item.size)}</span>
    </button>`).join("") || `<p class="empty-note">No .pt files found near the dataset — type a path instead.</p>`;
}

const shortDir = path => {
  const parts = path.split("/").slice(0, -1);
  return parts.length > 3 ? `…/${parts.slice(-2).join("/")}` : parts.join("/") || "/";
};

const formatSize = bytes => bytes >= 1 << 30 ? `${(bytes / (1 << 30)).toFixed(1)} GB`
  : bytes >= 1 << 20 ? `${(bytes / (1 << 20)).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;

export function applyPredictorState(data) {
  const pred = state.pred;
  const wasRunning = pred.job.state === "running";
  pred.status = data.status;
  pred.error = data.error;
  pred.names = data.model.names;
  pred.modelPath = data.model.path;
  pred.mapping = data.mapping;
  pred.pending = data.pending;
  pred.job = data.job;
  const ready = pred.status === "ready";
  const running = pred.job.state === "running";
  const canCancel = running && pred.job.done < pred.job.total && !pred.job.cancel_requested;
  const canRetry = !running && Number(pred.job.failed) > 0;
  $("load-model").disabled = pred.status === "unavailable" || pred.status === "loading" || running;
  $("run-predict").hidden = !ready;
  $("run-predict").disabled = running;
  $("cancel-predict").hidden = !canCancel;
  $("cancel-predict").disabled = !canCancel;
  $("retry-predict").hidden = !canRetry;
  $("predict-unlabeled-wrap").hidden = !ready;
  $("predict-image").hidden = !ready;
  $("predict-image").disabled = running;
  $("mapping-panel").hidden = !ready;
  if (ready && data.model.path && !$("model-path").value) $("model-path").value = data.model.path;
  const unmapped = Object.values(pred.mapping).filter(value => value === null).length;
  $("predictor-status").textContent = {
    unavailable: "Assisted labeling needs the [predict] extra (ultralytics)",
    idle: "",
    loading: "Loading model…",
    error: pred.error || "Failed to load model",
    ready: running
      ? `Predicting ${pred.job.done}/${pred.job.total} · ${pred.job.completed || 0} completed · ${pred.job.failed || 0} failed${pred.job.cancel_requested ? " · cancelling…" : ""}`
      : `${Object.keys(pred.names).length} model classes${unmapped ? ` · ${unmapped} unmapped` : ""}${pred.job.failed ? ` · ${pred.job.failed} failed` : ""}`,
  }[pred.status] || "";
  $("predictor-status").title = $("predictor-status").textContent;
  renderMapping();
  renderModelMenu();
  clearTimeout(pred.poll);
  if (running) {
    pred.poll = setTimeout(refreshPredictor, 800);
  } else if (wasRunning) {
    const count = Object.values(pred.pending).reduce((sum, n) => sum + n, 0);
    if (pred.job.state === "done") deps.toast(`Predicted ${pred.job.completed} image${pred.job.completed === 1 ? "" : "s"}${pred.job.failed ? ` · ${pred.job.failed} failed` : ""} · ${count} pending prediction${count === 1 ? "" : "s"}`, pred.job.failed ? "error" : undefined);
    if (pred.job.state === "cancelled") deps.toast(`Prediction cancelled · ${pred.job.completed} completed · ${pred.job.cancelled} skipped`);
    deps.loadImages();
    if (state.detail) loadEditorPredictions(state.detail.id);
  }
}

export function renderMapping() {
  const pred = state.pred;
  const options = value => `<option value="">— unmapped</option>` + Object.entries(state.meta.names)
    .map(([id, name]) => `<option value="${id}" ${value !== null && +id === value ? "selected" : ""}>${esc(id)} · ${esc(name)}</option>`).join("");
  $("mapping-list").innerHTML = Object.entries(pred.names).map(([id, name]) => `
    <label class="mapping-row"><span title="${esc(name)}">${esc(id)} · ${esc(name)}</span>
      <select data-model-class="${esc(id)}">${options(pred.mapping[id] ?? null)}</select>
    </label>`).join("");
}

export async function loadModel() {
  const path = $("model-path").value.trim();
  if (!path) return deps.toast("Enter a model path first", "error");
  $("load-model").disabled = true;
  $("predictor-status").textContent = "Loading model…";
  $("predictor-status").title = "Loading model…";
  try {
    applyPredictorState(await api("/api/v1/predictor/load", { method: "POST", body: JSON.stringify({ path }) }));
    deps.toast("Model loaded");
    loadModelOptions(); // recents changed
  } catch (error) {
    $("load-model").disabled = false;
    $("predictor-status").textContent = error.message;
    $("predictor-status").title = error.message;
    deps.toast(error.message, "error");
  }
}

export async function runPredict() {
  try {
    applyPredictorState(await api("/api/v1/predictor/run", {
      method: "POST",
      body: JSON.stringify({ split: $("image-split").value, only_unlabeled: $("predict-unlabeled").checked }),
    }));
  } catch (error) { deps.toast(error.message, "error"); }
}

export async function cancelPredict() {
  try {
    applyPredictorState(await api("/api/v1/predictor/cancel", { method: "POST" }));
  } catch (error) { deps.toast(error.message, "error"); }
}

export async function retryPredict() {
  try {
    applyPredictorState(await api("/api/v1/predictor/retry", { method: "POST" }));
  } catch (error) { deps.toast(error.message, "error"); }
}

export async function predictImage() {
  if (!state.detail) return;
  if (state.pred.status !== "ready") return deps.toast("Load a model first", "error");
  const id = state.detail.id;
  const button = $("predict-image");
  button.disabled = true;
  try {
    const data = await api(`/api/v1/images/${id}/predictions/compute`, { method: "POST" });
    if (state.detail?.id === id) {
      state.pred.items = data.items;
      renderPredList();
      deps.render();
    }
    state.pred.pending[id] = data.items.length;
    deps.toast(data.items.length ? `${data.items.length} prediction${data.items.length === 1 ? "" : "s"} — click a ghost or use the panel to accept` : "The model found nothing on this image");
    deps.loadImages();
  } catch (error) {
    deps.toast(error.message, "error");
  } finally {
    button.disabled = state.pred.job.state === "running";
  }
}

export async function loadEditorPredictions(id) {
  if (state.pred.status === "unavailable") return;
  try {
    const data = await api(`/api/v1/images/${id}/predictions`);
    if (!state.detail || state.detail.id !== id) return;
    state.pred.items = data.items;
  } catch { state.pred.items = []; }
  renderPredList();
  deps.render();
}

export const visiblePredictions = () => state.pred.items.filter(p => p.confidence >= state.pred.minConf);

export function drawPredictions(ctx) {
  if (!state.detail || !state.img) return;
  for (const prediction of visiblePredictions()) {
    const color = prediction.class_id === null ? UNMAPPED_COLOR : classColor(prediction.class_id);
    const path = deps.shapePath(prediction);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.setLineDash([7, 5]);
    ctx.stroke(path);
    ctx.restore();
    drawPredictionBadge(ctx, prediction, color);
  }
}

export function drawPredictionBadge(ctx, prediction, color) {
  const [l, t, r] = deps.bounds(prediction);
  if ((r - l) * state.img.width * state.view.scale < 26) return;
  const [sx, sy] = deps.toScreen(l, t);
  const name = prediction.class_id === null
    ? `${prediction.model_class_name}?`
    : state.meta.names[prediction.class_id] ?? `class ${prediction.class_id}`;
  const text = `${name} ${Math.round(prediction.confidence * 100)}%`;
  ctx.font = "600 11px Inter, system-ui, sans-serif";
  const width = ctx.measureText(text).width + 10, height = 17;
  const y = sy - height - 1 < 2 ? sy + 1 : sy - height - 1;
  ctx.save();
  ctx.globalAlpha = 0.85;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(sx, y, width, height, 3);
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#fff";
  ctx.fillText(text, sx + 5, y + 12);
  ctx.restore();
}

export function hitPrediction([x, y]) {
  for (const prediction of [...visiblePredictions()].reverse()) {
    if (state.meta.category === "detection") {
      const [l, t, r, b] = deps.bounds(prediction);
      if (x >= l && x <= r && y >= t && y <= b) return prediction;
    } else if (pointInPolygon(x, y, prediction.points)) {
      return prediction;
    }
  }
  return null;
}

export function renderPredList() {
  const items = state.pred.items;
  const section = $("prediction-section");
  section.hidden = !items.length;
  if (!items.length) return;
  const visible = visiblePredictions();
  $("prediction-count").textContent = items.length;
  $("accept-all-preds").disabled = !visible.some(p => p.class_id !== null);
  $("prediction-list").innerHTML = items.map(p => {
    const unmapped = p.class_id === null;
    const name = unmapped ? `${p.model_class_name} (unmapped)` : state.meta.names[p.class_id] ?? `class ${p.class_id}`;
    return `<div class="annotation-row pred-row ${p.confidence < state.pred.minConf ? "pred-dim" : ""}" data-prediction="${esc(p.id)}">
      <span class="swatch" style="background:${unmapped ? UNMAPPED_COLOR : classColor(p.class_id)}"></span>
      <span class="pred-name" title="${esc(name)}">${esc(name)}</span>
      <span class="row-meta">${Math.round(p.confidence * 100)}%</span>
      <button class="row-accept" title="${unmapped ? "Unmapped classes cannot be accepted" : "Accept prediction"}" ${unmapped ? "disabled" : ""}>✓</button>
      <button class="row-delete" title="Reject prediction">×</button>
    </div>`;
  }).join("");
}

export async function acceptPredictions(predictionIds, minConfidence = null) {
  if (!state.detail) return;
  const imageId = state.detail.id;
  const request = deps.editorRequest();
  const body = { prediction_ids: predictionIds };
  if (predictionIds === null && minConfidence !== null) body.min_confidence = minConfidence;
  try {
    await deps.saveQueue();
    const result = await api(`/api/v1/images/${imageId}/predictions/accept`, { method: "POST", body: JSON.stringify(body) });
    state.pred.pending[imageId] = result.predictions.length;
    state.canUndo = true;
    state.canRedo = false;
    deps.updateHistoryButtons();
    deps.statisticsChanged();
    deps.toast(`Accepted ${result.accepted} prediction${result.accepted === 1 ? "" : "s"} — Ctrl+Z undoes`);
    if (request === deps.editorRequest() && state.detail?.id === imageId) {
      state.detail = result.detail;
      deps.savedRevisionByImage.set(imageId, result.detail.revision);
      state.pred.items = result.predictions;
      deps.renderList(); renderPredList(); deps.render();
    }
    await Promise.all([deps.loadImages(), deps.loadObjects(), deps.loadIssues()]);
  } catch (error) { deps.toast(error.message, "error"); }
}

export async function rejectPredictions(predictionIds) {
  if (!state.detail) return;
  const imageId = state.detail.id;
  const request = deps.editorRequest();
  try {
    const result = await api(`/api/v1/images/${imageId}/predictions/reject`, { method: "POST", body: JSON.stringify({ prediction_ids: predictionIds }) });
    state.pred.pending[imageId] = result.predictions.length;
    if (request === deps.editorRequest() && state.detail?.id === imageId) {
      state.pred.items = result.predictions;
      renderPredList(); deps.render();
    }
    deps.loadImages();
  } catch (error) { deps.toast(error.message, "error"); }
}

