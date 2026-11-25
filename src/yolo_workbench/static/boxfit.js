import { $, api, apiUrl, esc } from "./api.js";
import { state } from "./state.js";
import { startedBy } from "./collab.js";
let deps;

/* Label QA's SAM box fit: SAM fits every box of a class (or the boxes chosen in Exploration) tightly around its
   object; the boxes it would move are listed here, biggest move first, to apply one by one or all at once (one
   Ctrl+Z). Nothing changes until applied. */
export function configureBoxFit(dependencies) {
  deps = dependencies;
}

const PAGE = 60;
const SOURCE_KEY = "qaSource";
const KINDS = {
  fitted: { title: "Fitted", hint: "SAM's box fits the object and nothing about it looks wrong" },
  uncertain: { title: "Uncertain", hint: "SAM moved the box, but something may be off: look before applying" },
  failed: { title: "No outline", hint: "SAM found no object in the box: the label may be wrong" },
};
const REASONS = {
  low_score: "SAM is unsure of the outline",
  different: "very different from the label",
  edge: "the outline reaches the search area's edge (the object may be larger)",
  overlap: "overlaps another label (SAM outlines the visible part)",
  unstable: "SAM outlines something else when given more room (a tray, a hand, a neighbour)",
};
const SUBTITLES = {
  model: "Where a trained model disagrees with the labels: objects nobody labelled, wrong classes, loose boxes, and labels the model cannot find. Most suspicious first.",
  sam: "SAM fits every box of a class tightly around its object and lists the boxes it would move, biggest move first. Nothing changes until you apply.",
};
const fit = { status: null, kind: "fitted", items: [], total: 0, counts: {}, request: 0, poll: null, wasRunning: false };

export function qaSource() {
  if (state.meta?.category !== "detection") return "model";
  try { return localStorage.getItem(SOURCE_KEY) === "sam" ? "sam" : "model"; } catch { return "model"; }
}

/** Show the pane of the chosen source; the Label QA tab calls the model pane's own show. */
export function applyQaSource() {
  const source = qaSource();
  for (const input of document.querySelectorAll("input[name=qa-source]")) input.checked = input.value === source;
  $("qa-model-pane").hidden = source !== "model";
  $("qa-fit-pane").hidden = source !== "sam";
  $("qa-subtitle").textContent = SUBTITLES[source];
  if (source === "sam") $("qa-clear").hidden = true; else $("fit-clear").hidden = true;
  return source;
}

function setSource(source) {
  try { localStorage.setItem(SOURCE_KEY, source); } catch { /* the choice lasts while the page does */ }
  if (applyQaSource() === "sam") showBoxFit(); else deps.showLabelQA();
}

export function bindBoxFit() {
  for (const input of document.querySelectorAll("input[name=qa-source]")) input.onchange = () => setSource(input.value);
  $("fit-run").onclick = runFit;
  $("fit-cancel").onclick = () => fitAction("/boxfit/cancel", "POST");
  $("fit-clear").onclick = async () => {
    if (!confirm("Forget every SAM fit and the boxes you dismissed?")) return;
    await fitAction("/boxfit", "DELETE");
    fit.items = [];
    renderFindings();
  };
  $("fit-tolerance").oninput = () => {
    $("fit-tolerance-value").textContent = `${$("fit-tolerance").value} px`;
    reloadSoon();
  };
  $("fit-filter-class").onchange = () => loadFindings();
  $("fit-filter-split").onchange = () => loadFindings();
  $("fit-kinds").addEventListener("click", e => {
    const chip = e.target.closest("[data-fit-kind]");
    if (!chip) return;
    fit.kind = chip.dataset.fitKind || null;
    loadFindings();
  });
  $("fit-more").onclick = () => loadFindings({ append: true });
  $("fit-apply-all").onclick = applyAll;
  $("fit-grid").addEventListener("click", e => {
    const card = e.target.closest("[data-fit-key]");
    if (!card) return;
    const item = fit.items.find(entry => entry.key === card.dataset.fitKey && entry.image_id === card.dataset.fitImage);
    if (!item) return;
    if (e.target.closest("[data-fit-apply]")) return applyOne(item, card);
    if (e.target.closest("[data-fit-dismiss]")) return dismiss(item, card);
    if (e.target.closest("[data-fit-open]") || e.target.closest(".thumb")) return openFit(item);
  });
}

const classOptions = (all, selected) => (all ? `<option value="">${esc(all)}</option>` : "")
  + Object.entries(state.meta.names).map(([id, name]) => `<option value="${id}" ${String(selected) === id ? "selected" : ""}>${esc(name)}</option>`).join("");

export async function showBoxFit() {
  const splits = Object.keys(state.meta.split_counts);
  for (const id of ["fit-split", "fit-filter-split"]) {
    const select = $(id), previous = select.value;
    select.innerHTML = `<option value="all">All splits</option>` + splits.map(split => `<option>${esc(split)}</option>`).join("");
    select.value = [...select.options].some(option => option.value === previous) ? previous : "all";
  }
  // the class Exploration shows is the likely one to fit
  const runClass = $("fit-class").value || $("object-class").value;
  $("fit-class").innerHTML = classOptions("All classes", runClass);
  const filterClass = $("fit-filter-class").value;
  $("fit-filter-class").innerHTML = classOptions("All classes", filterClass);
  await refresh();
  await loadFindings();
}

// The labels may have changed elsewhere (the editor, an undo): findings are recomputed from the current labels.
export function labelsChangedForFit() {
  if ($("quality").classList.contains("active") && qaSource() === "sam" && fit.status?.result) loadFindings({ keep: true });
}

/** Exploration's Fit with SAM: fit the chosen objects, then show the result here. */
export async function fitObjects(objects) {
  try {
    fit.status = await api("/boxfit/run", { method: "POST", body: JSON.stringify({ objects }) });
  } catch (error) {
    deps.toast(error.message, "error");
    return false;
  }
  try { localStorage.setItem(SOURCE_KEY, "sam"); } catch { /* shown once */ }
  deps.showTab("quality");
  return true;
}

/* ---------- running ---------- */

async function runFit() {
  const classId = $("fit-class").value;
  try {
    fit.status = await api("/boxfit/run", {
      method: "POST",
      body: JSON.stringify({ class_ids: classId === "" ? null : [+classId], split: $("fit-split").value }),
    });
    if (classId !== "") $("fit-filter-class").value = classId;  // show what this run fits
    renderSetup();
    poll();
  } catch (error) { deps.toast(error.message, "error"); }
}

async function fitAction(path, method) {
  try {
    fit.status = { ...fit.status, ...(await api(path, { method })) };
    renderSetup();
  } catch (error) { deps.toast(error.message, "error"); }
}

async function refresh() {
  try {
    fit.status = await api("/boxfit");
  } catch (error) {
    deps.toast(error.message, "error");
    return;
  }
  renderSetup();
  poll();
}

function poll() {
  clearTimeout(fit.poll);
  const running = fit.status?.job.state === "running";
  if (running) {
    if (!fit.wasRunning) loadFindings();
    fit.wasRunning = true;
    fit.poll = setTimeout(async () => {
      await refresh();
      if (fit.status?.job.state === "running") loadFindings({ keep: true });  // fits show as they come
    }, 1500);
  } else if (fit.wasRunning) {
    fit.wasRunning = false;
    const job = fit.status.job;
    if (job.state === "error") deps.toast(`Box fit failed: ${job.error}`, "error");
    else deps.toast(`Fitted ${plural(job.objects, "box", "boxes")} in ${plural(job.done - job.failed, "image")}${job.failed ? ` · ${job.failed} failed` : ""}${job.state === "cancelled" ? " · cancelled" : ""}`, job.failed ? "error" : "info");
    deps.refreshSam();  // the run may have loaded SAM: the editor's button shows it
    loadFindings();
  }
}

const plural = (count, one, many = `${one}s`) => `${(count ?? 0).toLocaleString()} ${count === 1 ? one : many}`;

function renderSetup() {
  const status = fit.status;
  if (!status) return;
  const job = status.job, result = status.result, samState = status.sam ?? {};
  const running = job.state === "running";
  $("fit-run").disabled = running || !status.available;
  $("fit-run").textContent = result ? "Fit again" : "Fit boxes";
  $("fit-cancel").hidden = !running;
  $("fit-clear").hidden = !result || running || qaSource() !== "sam";
  $("fit-progress").hidden = !running;
  $("fit-progress").max = Math.max(1, job.total);
  $("fit-progress").value = job.done;
  let text;
  if (!status.available) text = "Box fit works on detection datasets.";
  else if (samState.status === "unavailable") text = "Box fit needs SAM: run 'uv sync' to install ultralytics.";
  else if (running && samState.status === "loading") text = samState.message || "Loading SAM…";
  else if (running) text = `Fitting boxes: ${job.done.toLocaleString()} / ${job.total.toLocaleString()} images · ${plural(job.objects, "box", "boxes")}${startedBy(result?.started_by)}`;
  else if (result) {
    const when = result.updated ? new Date(result.updated).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "";
    text = `${plural(result.objects, "box", "boxes")} fitted in ${plural(result.images, "image")}${result.model ? ` with ${result.model}` : ""}${when ? ` · last run ${when}` : ""}${startedBy(result.started_by)}. Fitting again adds to these.`;
  } else text = "Fit boxes runs SAM around every box of the class and proposes a better-fitting box where the label is off. Each box is encoded at a resolution that suits its size, so a GPU helps on large images.";
  $("fit-status").textContent = text;
  $("fit-status").title = text;
  $("fit-filters").hidden = !result;
}

/* ---------- findings ---------- */

let reloadTimer = null;
function reloadSoon() {
  clearTimeout(reloadTimer);
  reloadTimer = setTimeout(() => loadFindings(), 250);
}

function filterParams() {
  const params = new URLSearchParams({ split: $("fit-filter-split").value, tolerance: $("fit-tolerance").value });
  if ($("fit-filter-class").value !== "") params.set("class_id", $("fit-filter-class").value);
  return params;
}

async function loadFindings({ append = false, keep = false } = {}) {
  if (!fit.status?.result) {
    fit.items = [];
    return renderFindings();
  }
  const request = ++fit.request;
  const offset = append ? fit.items.length : 0;
  const limit = keep ? Math.max(PAGE, fit.items.length) : PAGE;
  const params = filterParams();
  params.set("offset", String(offset));
  params.set("limit", String(limit));
  if (fit.kind) params.set("kind", fit.kind);
  try {
    const data = await api(`/boxfit/findings?${params}`);
    if (request !== fit.request) return;
    fit.items = append ? [...fit.items, ...data.items] : data.items;
    fit.total = data.total;
    fit.counts = data.counts;
    fit.stale = data.stale;
    fit.images = data.images;
    renderFindings();
  } catch (error) {
    if (request === fit.request) deps.toast(error.message, "error");
  }
}

const className = id => state.meta.names[id] ?? `class ${id}`;
const SIDES = ["Left", "Top", "Right", "Bottom"];

// "Left 8 in · Right 10 in · Top 6 out": which way each edge moves (in: the box gets smaller)
function edgeText(edges) {
  if (!edges) return "";
  return edges.map((delta, side) => {
    if (Math.abs(delta) < 0.5) return null;
    const inward = side < 2 ? delta > 0 : delta < 0;
    return `${SIDES[side]} ${Math.round(Math.abs(delta))} ${inward ? "in" : "out"}`;
  }).filter(Boolean).join(" · ") + " px";
}

function describe(item) {
  const name = `<strong>${esc(className(item.class_id))}</strong>`;
  if (item.kind === "failed") return `${name} · SAM found no object in this box`;
  const reasons = item.reasons.length ? `<div class="fit-reasons">${esc(item.reasons.map(reason => REASONS[reason] ?? reason).join("; "))}</div>` : "";
  return `${name} · ${esc(edgeText(item.edges))}${reasons}`;
}

function renderFindings() {
  const result = fit.status?.result;
  const all = Object.values(fit.counts || {}).reduce((sum, count) => sum + count, 0);
  $("fit-kinds").innerHTML = result ? [["", "All", all], ...Object.entries(KINDS).map(([kind, info]) => [kind, info.title, fit.counts?.[kind] || 0])]
    .map(([kind, title, count]) => `<button class="issue-chip ${(fit.kind || "") === kind ? "active" : ""}" data-fit-kind="${kind}" title="${esc(KINDS[kind]?.hint ?? "Every box SAM would move, and the ones it found nothing in")}">${esc(title)} <strong>${count.toLocaleString()}</strong></button>`).join("") : "";
  const stale = fit.stale ? ` · ${plural(fit.stale, "image")} changed since ${fit.stale === 1 ? "its" : "their"} fit (fit again to include ${fit.stale === 1 ? "it" : "them"})` : "";
  $("fit-total").textContent = result ? `${plural(fit.total, "box", "boxes")} in ${plural(fit.images || 0, "image")}${stale}` : "";
  $("fit-more").hidden = fit.items.length >= fit.total;
  // Apply all: the fitted boxes ("All" too), or the uncertain ones once that chip is chosen
  const applyKind = fit.kind === "uncertain" ? "uncertain" : "fitted";
  const applyCount = fit.counts?.[applyKind] || 0;
  $("fit-apply-all").hidden = !result || fit.kind === "failed" || !applyCount;
  $("fit-apply-all").textContent = `Apply ${plural(applyCount, `${applyKind} box`, `${applyKind} boxes`)}`;
  $("fit-apply-all").title = `Replace ${applyCount === 1 ? "this box" : `these ${applyCount.toLocaleString()} boxes`} with SAM's (one Ctrl+Z undoes it)`;
  if (!result) {
    $("fit-grid").innerHTML = `<div class="empty">No fits yet. Pick a class and press Fit boxes, or select boxes in Exploration and press Fit with SAM.</div>`;
    return;
  }
  $("fit-grid").innerHTML = fit.items.map(item => `<article class="card qa-card" data-fit-key="${esc(item.key)}" data-fit-image="${esc(item.image_id)}">
      <div class="thumb" title="Solid: the label · dashed: SAM's box · green: what SAM's box adds · red: what it gives up · click to open in the editor">
        <img loading="lazy" src="${apiUrl(`/boxfit/images/${item.image_id}/findings/${item.key}/crop`)}" alt="">
        <span class="qa-kind ${item.kind}">${esc(KINDS[item.kind].title)}</span>
        ${item.deviation_px != null ? `<span class="qa-score" title="The largest edge move">${Math.round(item.deviation_px)} px</span>` : ""}
      </div>
      <div class="card-body">
        <div class="name" title="${esc(item.image_name)}">${esc(item.image_name)}</div>
        <div class="qa-detail">${describe(item)}</div>
        <div class="meta"><span class="split-badge">${esc(item.split)}</span>${item.kind !== "failed" ? `<span title="SAM's own estimate of its outline's quality">SAM ${Math.round(item.score * 100)}%</span>` : ""}</div>
        <div class="object-actions">
          ${item.prediction ? `<button class="qa-fix" data-fit-apply title="Replace the label's box with SAM's (Ctrl+Z undoes)">Use SAM box</button>` : ""}
          <button data-fit-dismiss title="The label is right: hide this box">Dismiss</button>
          <button data-fit-open title="Open in the editor">Open</button>
        </div>
      </div>
    </article>`).join("") || `<div class="empty">${fit.status.job.state === "running" ? "Fitting… boxes SAM would move show here as they are fitted."
      : all ? "No boxes of this kind." : "SAM would not move any box by more than this. The boxes are tight."}</div>`;
}

async function applyOne(item, card) {
  card.classList.add("busy");
  try {
    const result = await api(`/boxfit/images/${item.image_id}/findings/${item.key}/apply`, { method: "POST" });
    deps.toast("Box fitted — Ctrl+Z undoes");
    await deps.labelsChanged(item.image_id, result.detail);
    await loadFindings({ keep: true });
  } catch (error) {
    card.classList.remove("busy");
    deps.toast(error.message, "error");
    loadFindings({ keep: true });
  }
}

async function applyAll() {
  const kind = fit.kind === "uncertain" ? "uncertain" : "fitted";
  const count = fit.counts?.[kind] || 0;
  if (!count || !confirm(`Replace ${plural(count, `${kind} box`, `${kind} boxes`)} with SAM's? One Ctrl+Z undoes it.`)) return;
  const params = filterParams();
  $("fit-apply-all").disabled = true;
  try {
    const result = await api("/boxfit/apply-all", {
      method: "POST",
      body: JSON.stringify({ kind, class_id: params.has("class_id") ? +params.get("class_id") : null, split: params.get("split"), tolerance: +params.get("tolerance") }),
    });
    const skipped = result.skipped.length ? ` · ${result.skipped.length} skipped (changed meanwhile)` : "";
    deps.toast(`Fitted ${plural(result.applied, "box", "boxes")} in ${plural(result.files, "image")}${skipped} — Ctrl+Z undoes`);
    await deps.labelsChangedInBulk();
    await loadFindings();
  } catch (error) {
    deps.toast(error.message, "error");
  } finally {
    $("fit-apply-all").disabled = false;
  }
}

async function dismiss(item, card) {
  card.classList.add("busy");
  try {
    await api(`/boxfit/images/${item.image_id}/findings/${item.key}/dismiss`, { method: "POST" });
    await loadFindings({ keep: true });
  } catch (error) {
    card.classList.remove("busy");
    deps.toast(error.message, "error");
  }
}

const boxOf = points => [points[0] - points[2] / 2, points[1] - points[3] / 2, points[0] + points[2] / 2, points[1] + points[3] / 2];

// The editor selects the label, zooms to both boxes and outlines SAM's dashed.
function openFit(item) {
  if (!item.prediction) return deps.openEditor(item.image_id, item.label.annotation_id);
  const label = boxOf(item.label.points), sam = boxOf(item.prediction.points);
  const box = [Math.min(label[0], sam[0]), Math.min(label[1], sam[1]), Math.max(label[2], sam[2]), Math.max(label[3], sam[3])];
  deps.openEditor(item.image_id, { annotation: item.label.annotation_id, box, hint: item.prediction.points });
}
