import { $, api, apiUrl, esc } from "./api.js";
import { state } from "./state.js";
import { applyPredictorState } from "./predictor.js";
let deps;

// Label QA tab: run the loaded model over the dataset, list where it disagrees with the labels.
export function configureLabelQA(dependencies) {
  deps = dependencies;
}

const PAGE = 60;
const KINDS = {
  missing_label: { title: "Missing label", hint: "The model is confident about an object that has no label" },
  wrong_class: { title: "Wrong class", hint: "The label and the model's box match, but the classes differ" },
  box_mismatch: { title: "Loose box", hint: "The model draws this object's box quite differently" },
  unmatched_label: { title: "Unmatched label", hint: "Nothing the model predicted overlaps this label" },
};
const qa = { status: null, kind: null, items: [], total: 0, counts: {}, request: 0, poll: null, wasRunning: false };

export function bindLabelQA() {
  $("qa-load").onclick = loadQAModel;
  $("qa-run").onclick = runQA;
  $("qa-cancel").onclick = () => qaAction("/qa/cancel", "POST");
  $("qa-clear").onclick = async () => {
    if (!confirm("Forget this analysis and its dismissed findings?")) return;
    await qaAction("/qa", "DELETE");
    qa.items = [];
    renderFindings();
  };
  $("qa-conf").oninput = () => {
    $("qa-conf-value").textContent = `${Math.round($("qa-conf").value * 100)}%`;
    reloadFindingsSoon();
  };
  $("qa-tolerance").oninput = () => {
    $("qa-tolerance-value").textContent = `${$("qa-tolerance").value} px`;
    reloadFindingsSoon();
  };
  $("qa-filter-split").onchange = () => loadFindings();
  $("qa-kinds").addEventListener("click", e => {
    const chip = e.target.closest("[data-qa-kind]");
    if (!chip) return;
    qa.kind = chip.dataset.qaKind || null;
    loadFindings();
  });
  $("qa-more").onclick = () => loadFindings({ append: true });
  $("qa-grid").addEventListener("click", e => {
    const card = e.target.closest("[data-qa-key]");
    if (!card) return;
    const item = qa.items.find(entry => entry.key === card.dataset.qaKey && entry.image_id === card.dataset.qaImage);
    if (!item) return;
    if (e.target.closest("[data-qa-apply]")) return applyFinding(item, card);
    if (e.target.closest("[data-qa-dismiss]")) return dismissFinding(item, card);
    if (e.target.closest("[data-qa-open]") || e.target.closest(".thumb")) return openFinding(item);
  });
}

export async function showLabelQA() {
  const splits = Object.keys(state.meta.split_counts);
  for (const id of ["qa-split", "qa-filter-split"]) {
    const select = $(id), previous = select.value;
    select.innerHTML = `<option value="all">All splits</option>` + splits.map(split => `<option>${esc(split)}</option>`).join("");
    select.value = [...select.options].some(option => option.value === previous) ? previous : "all";
  }
  await Promise.all([refreshQA(), loadModelChoices()]);
  await loadFindings();
}

// The labels may have changed elsewhere (the editor): findings are recomputed from the current labels.
export function labelsChangedForQA() {
  if ($("quality").classList.contains("active") && qa.status?.result) loadFindings({ keep: true });
}

/* ---------- model and analysis ---------- */

async function loadModelChoices() {
  let items = [];
  try { items = (await api("/predictor/models")).items; } catch { /* the select falls back to the loaded model */ }
  // this dataset's trained runs are the natural reviewers of its labels
  items.sort((a, b) => (b.trained - a.trained) || (b.loaded - a.loaded));
  const loaded = qa.status?.predictor.model;
  if (loaded && !items.some(item => item.path === loaded)) items.unshift({ path: loaded, name: loaded.split("/").pop(), loaded: true });
  $("qa-model").innerHTML = items.length
    ? items.map(item => `<option value="${esc(item.path)}" ${item.loaded ? "selected" : ""} title="${esc(item.path)}">${esc(item.name)}${item.trained ? " · trained here" : ""}${item.loaded ? " · loaded" : ""}</option>`).join("")
    : `<option value="">No models found — train one in the Training tab, or load one in the Annotation tab</option>`;
  renderSetup();
}

async function loadQAModel() {
  const path = $("qa-model").value;
  if (!path) return;
  $("qa-load").disabled = true;
  $("qa-status").textContent = "Loading model…";
  try {
    const predictor = await api("/predictor/load", { method: "POST", body: JSON.stringify({ path }) });
    applyPredictorState(predictor);  // the Annotation tab predicts with the same model
    $("model-path").value = path;
    deps.toast("Model loaded");
    await refreshQA();
    await loadModelChoices();
  } catch (error) {
    deps.toast(error.message, "error");
  } finally {
    $("qa-load").disabled = false;
    renderSetup();
  }
}

async function runQA() {
  try {
    qa.status = await api("/qa/run", { method: "POST", body: JSON.stringify({ split: $("qa-split").value }) });
    qa.items = [];
    renderFindings();
    renderSetup();
    pollQA();
  } catch (error) { deps.toast(error.message, "error"); }
}

async function qaAction(path, method) {
  try {
    qa.status = await api(path, { method });
    renderSetup();
  } catch (error) { deps.toast(error.message, "error"); }
}

async function refreshQA() {
  try {
    qa.status = await api("/qa");
  } catch (error) {
    deps.toast(error.message, "error");
    return;
  }
  renderSetup();
  pollQA();
}

function pollQA() {
  clearTimeout(qa.poll);
  const running = qa.status?.job.state === "running";
  if (running) {
    qa.wasRunning = true;
    qa.poll = setTimeout(refreshQA, 1000);
  } else if (qa.wasRunning) {
    qa.wasRunning = false;
    const job = qa.status.job;
    if (job.state === "error") deps.toast(`Label QA failed: ${job.error}`, "error");
    else deps.toast(`Analyzed ${job.done - job.failed} image${job.done - job.failed === 1 ? "" : "s"}${job.failed ? ` · ${job.failed} failed` : ""}${job.state === "cancelled" ? " · cancelled" : ""}`, job.failed ? "error" : "info");
    loadFindings();
  }
}

function renderSetup() {
  const status = qa.status;
  if (!status) return;
  const job = status.job, result = status.result;
  const running = job.state === "running";
  const predictorReady = status.predictor.status === "ready";
  const choice = $("qa-model").value;
  const loadedChosen = predictorReady && choice && choice === status.predictor.model;
  $("qa-load").hidden = !choice || loadedChosen;
  $("qa-load").disabled = running || status.predictor.status === "unavailable";
  $("qa-run").disabled = running || !loadedChosen;
  $("qa-run").textContent = result ? "Analyze again" : "Analyze";
  $("qa-cancel").hidden = !running;
  $("qa-clear").hidden = !result || running;
  $("qa-progress").hidden = !running;
  $("qa-progress").max = Math.max(1, job.total);
  $("qa-progress").value = job.done;
  let text;
  if (status.predictor.status === "unavailable") text = "Label QA needs ultralytics (run 'uv sync').";
  else if (running) text = `Analyzing ${job.done.toLocaleString()} / ${job.total.toLocaleString()} images…`;
  else if (!loadedChosen && choice) text = "Load the model, then analyze. The best weights of a training run on this dataset work well.";
  else if (result) {
    const when = new Date(result.created).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
    text = `Results of ${result.model_name} on ${result.images.toLocaleString()} image${result.images === 1 ? "" : "s"} (${result.split}) · ${when}`;
    if (result.unmapped.length) text += ` · model classes not in this dataset are ignored: ${result.unmapped.join(", ")}`;
  } else text = "Analyze runs the model over the images and compares its predictions with the labels.";
  $("qa-status").textContent = text;
  $("qa-status").title = text;
  $("qa-filters").hidden = !result;
}

/* ---------- findings ---------- */

let reloadTimer = null;
function reloadFindingsSoon() {
  clearTimeout(reloadTimer);
  reloadTimer = setTimeout(() => loadFindings(), 250);
}

async function loadFindings({ append = false, keep = false } = {}) {
  if (!qa.status?.result) {
    qa.items = [];
    $("qa-count").textContent = "";
    return renderFindings();
  }
  const request = ++qa.request;
  const offset = append ? qa.items.length : 0;
  const limit = keep ? Math.max(PAGE, qa.items.length) : PAGE;
  const params = new URLSearchParams({ split: $("qa-filter-split").value, min_conf: $("qa-conf").value, tolerance: $("qa-tolerance").value, offset: String(offset), limit: String(limit) });
  if (qa.kind) params.set("kind", qa.kind);
  try {
    const data = await api(`/qa/findings?${params}`);
    if (request !== qa.request) return;
    qa.items = append ? [...qa.items, ...data.items] : data.items;
    qa.total = data.total;
    qa.counts = data.counts;
    qa.stale = data.stale;
    qa.images = data.images;
    const all = Object.values(data.counts).reduce((sum, count) => sum + count, 0);
    $("qa-count").textContent = all ? all.toLocaleString() : "";
    renderFindings();
  } catch (error) {
    if (request === qa.request) deps.toast(error.message, "error");
  }
}

const className = id => state.meta.names[id] ?? `class ${id}`;
const percent = value => `${Math.round(value * 100)}%`;

function describe(item) {
  const label = item.label, prediction = item.prediction;
  switch (item.kind) {
    case "missing_label": return `Model sees <strong>${esc(className(prediction.class_id))}</strong> ${percent(prediction.confidence)} where nothing is labelled`;
    case "wrong_class": return `Labelled <strong>${esc(className(label.class_id))}</strong>, model says <strong>${esc(className(prediction.class_id))}</strong> ${percent(prediction.confidence)}`;
    case "box_mismatch": return item.deviation_px != null
      ? `<strong>${esc(className(label.class_id))}</strong> box edge ${Math.round(item.deviation_px)} px from the model's (model ${percent(prediction.confidence)})`
      : `<strong>${esc(className(label.class_id))}</strong> box overlaps the model's ${percent(item.iou)} (model ${percent(prediction.confidence)})`;
    default: return `<strong>${esc(className(label.class_id))}</strong> label the model does not find`;
  }
}

function fixLabel(item) {
  switch (item.kind) {
    case "missing_label": return ["Add label", "Add the model's object as a label"];
    case "wrong_class": return [`Make it ${className(item.prediction.class_id)}`, "Change the label's class to the model's"];
    case "box_mismatch": return ["Use model box", "Replace the label's geometry with the model's"];
    default: return ["Delete label", "Delete this label"];
  }
}

function renderFindings() {
  const result = qa.status?.result;
  const all = Object.values(qa.counts || {}).reduce((sum, count) => sum + count, 0);
  $("qa-kinds").innerHTML = result ? [["", "All", all], ...Object.entries(KINDS).map(([kind, info]) => [kind, info.title, qa.counts?.[kind] || 0])]
    .map(([kind, title, count]) => `<button class="issue-chip ${(qa.kind || "") === kind ? "active" : ""}" data-qa-kind="${kind}" title="${esc(KINDS[kind]?.hint ?? "Every finding")}">${esc(title)} <strong>${count.toLocaleString()}</strong></button>`).join("") : "";
  const stale = qa.stale ? ` · ${qa.stale.toLocaleString()} image${qa.stale === 1 ? "" : "s"} changed since the analysis (analyze again to include ${qa.stale === 1 ? "it" : "them"})` : "";
  $("qa-total").textContent = result ? `${qa.total.toLocaleString()} finding${qa.total === 1 ? "" : "s"} in ${(qa.images || 0).toLocaleString()} image${qa.images === 1 ? "" : "s"}${stale}` : "";
  $("qa-more").hidden = qa.items.length >= qa.total;
  if (!result) {
    $("qa-grid").innerHTML = `<div class="empty">No analysis yet. Load a model trained on similar data (a training run's best weights are ideal) and press Analyze.</div>`;
    return;
  }
  $("qa-grid").innerHTML = qa.items.map(item => {
    const [fix, fixHint] = fixLabel(item);
    return `<article class="card qa-card" data-qa-key="${esc(item.key)}" data-qa-image="${esc(item.image_id)}">
      <div class="thumb" title="Solid: the label · dashed: the model's prediction · click to open in the editor">
        <img loading="lazy" src="${apiUrl(`/qa/images/${item.image_id}/findings/${item.key}/crop`)}" alt="">
        <span class="qa-kind ${item.kind}">${esc(KINDS[item.kind].title)}</span>
        <span class="qa-score" title="How suspicious this looks (higher first)">${percent(item.score)}</span>
      </div>
      <div class="card-body">
        <div class="name" title="${esc(item.image_name)}">${esc(item.image_name)}</div>
        <div class="qa-detail">${describe(item)}</div>
        <div class="meta"><span class="split-badge">${esc(item.split)}</span></div>
        <div class="object-actions">
          <button class="qa-fix" data-qa-apply title="${esc(fixHint)} (Ctrl+Z undoes)">${esc(fix)}</button>
          <button data-qa-dismiss title="The label is right: hide this finding">Dismiss</button>
          <button data-qa-open title="Open in the editor">Open</button>
        </div>
      </div>
    </article>`;
  }).join("") || `<div class="empty">${all ? "No findings of this kind at this confidence." : "No disagreements at this confidence. The labels and the model agree."}</div>`;
}

async function applyFinding(item, card) {
  card.classList.add("busy");
  try {
    const result = await api(`/qa/images/${item.image_id}/findings/${item.key}/apply`, { method: "POST" });
    deps.toast(`${fixLabel(item)[0]}: done — Ctrl+Z undoes`);
    await deps.labelsChanged(item.image_id, result.detail);
    await loadFindings({ keep: true });
  } catch (error) {
    card.classList.remove("busy");
    deps.toast(error.message, "error");
    loadFindings({ keep: true });
  }
}

async function dismissFinding(item, card) {
  card.classList.add("busy");
  try {
    await api(`/qa/images/${item.image_id}/findings/${item.key}/dismiss`, { method: "POST" });
    await loadFindings({ keep: true });
  } catch (error) {
    card.classList.remove("busy");
    deps.toast(error.message, "error");
  }
}

const boxOf = points => state.meta.category === "detection"
  ? [points[0] - points[2] / 2, points[1] - points[3] / 2, points[0] + points[2] / 2, points[1] + points[3] / 2]
  : [Math.min(...points.filter((_, i) => i % 2 === 0)), Math.min(...points.filter((_, i) => i % 2)), Math.max(...points.filter((_, i) => i % 2 === 0)), Math.max(...points.filter((_, i) => i % 2))];

// The editor selects the label (if any), shows both shapes, and outlines the model's shape dashed.
function openFinding(item) {
  if (!item.prediction) return deps.openEditor(item.image_id, item.label.annotation_id);
  const model = boxOf(item.prediction.points);
  const label = item.label ? boxOf(item.label.points) : model;
  const box = [Math.min(label[0], model[0]), Math.min(label[1], model[1]), Math.max(label[2], model[2]), Math.max(label[3], model[3])];
  deps.openEditor(item.image_id, { annotation: item.label?.annotation_id ?? null, box, hint: item.prediction.points });
}
