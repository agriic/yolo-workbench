import { $, api, classColor } from "./api.js";
import { state } from "./state.js";
let deps;

export function configureSam(dependencies) {
  deps = dependencies;
}

/* ---------- smart annotation (SAM) ----------
   Left click marks the object, right click / Shift+click marks background, a left drag gives a
   rough box. Every prompt change asks the server for a new mask; ⏎ adds the result as an
   annotation through the normal save path. */

// set by the first click on "Smart annotation": from then on images are encoded as soon as they open
const USED_KEY = "samUsed";
const MAX_POINTS_KEY = "samMaxPoints";  // segmentation: the polygon point limit, kept across images and sessions
const DRAG_THRESHOLD = 4;   // px of pointer travel that turns a click into a box drag
const POINT_HIT = 8;        // px: clicking an existing point removes it

const used = () => { try { return localStorage.getItem(USED_KEY) === "1"; } catch { return false; } };
const sam = () => state.sam;

function maxPoints() {
  try {
    const value = Number(localStorage.getItem(MAX_POINTS_KEY));
    return Number.isInteger(value) && value >= 3 && value <= 10000 ? value : null;
  } catch { return null; }
}

let maxPointsTimer;

/** The Max points field changed: remember it and re-run the current prompts with the new limit. */
export function samMaxPointsChanged() {
  clearTimeout(maxPointsTimer);
  const text = $("sam-max-points").value.trim();
  const value = text === "" ? null : Math.round(+text);
  if (value !== null && !(value >= 3 && value <= 10000)) return;
  try {
    if (value === null) localStorage.removeItem(MAX_POINTS_KEY); else localStorage.setItem(MAX_POINTS_KEY, String(value));
  } catch { /* the limit then lasts only while the field keeps it */ }
  maxPointsTimer = setTimeout(() => { if (sam().result) segment(); }, 250);
}

export async function refreshSam() {
  try { applySamState(await api("/api/v1/sam")); } catch { /* the button stays disabled */ }
}

function applySamState(data) {
  const s = sam();
  const wasReady = s.status === "ready";
  s.status = data.status;
  s.model = data.model;
  s.message = data.message;
  s.error = data.error;
  s.unloaded = data.unloaded ?? null;  // a training run took the GPU: not loaded again by itself (an idle unload is)
  const button = $("smart-annotate");
  button.disabled = s.status === "unavailable";
  button.title = {
    unavailable: "Smart annotation needs ultralytics",
    loading: data.message || "Loading SAM…",
    error: data.error || "SAM failed to load — click to retry",
    ready: `Smart annotation with ${data.model}${data.device ? ` on ${data.device}` : ""} (S)`,
  }[s.status] || s.unloaded || "Smart annotation: click objects, SAM outlines them (S)";
  if (data.gpu_warning) button.title += ` — ${data.gpu_warning}`;
  if (data.gpu_warning && s.status === "ready" && !wasReady) deps.toast(data.gpu_warning);
  button.classList.toggle("loading", s.status === "loading");
  clearTimeout(s.poll);
  if (s.status === "loading") s.poll = setTimeout(refreshSam, 1000);
  if (s.status === "ready" && !wasReady) {
    encodeCurrent();
    if (s.points.length || s.box) segment();
    if (s.fitPending != null) { const all = s.fitPending; s.fitPending = null; fitBoxes(all); }
  }
  if (s.status === "error" && s.active) deps.toast(s.error, "error");
  if (s.status === "error" && s.fitPending != null) { s.fitPending = null; deps.toast(s.error, "error"); }
  deps.updateStatus();
}

async function startModel() {
  if (sam().status === "ready" || sam().status === "loading") return;
  try { applySamState(await api("/api/v1/sam/start", { method: "POST" })); } catch (error) { deps.toast(error.message, "error"); }
}

export async function toggleSam(force = null) {
  const s = sam();
  if (state.meta.category === "classification" || s.status === "unavailable") return;
  const active = force ?? !s.active;
  if (active === s.active) return;
  s.active = active;
  clearPrompts();
  $("smart-annotate").classList.toggle("active", active);
  $("smart-annotate").setAttribute("aria-pressed", String(active));
  $("sam-points-field").hidden = !active || state.meta.category !== "segmentation";
  if (active && document.activeElement !== $("sam-max-points")) $("sam-max-points").value = maxPoints() ?? "";
  if (active) {
    state.drawing = null;
    try { localStorage.setItem(USED_KEY, "1"); } catch { /* only skips encoding ahead */ }
    if (s.status === "ready") encodeCurrent(); else await startModel();
  }
  deps.render();
}

/** The editor opened an image: forget the old prompts and, once SAM was ever used, encode right away. */
export function samImageOpened() {
  clearPrompts();
  clearFit();
  if (!used() || state.meta.category === "classification" || sam().status === "unavailable") return;
  if (sam().status === "ready") encodeCurrent(); else if (!sam().unloaded) startModel();
}

export function samEditorClosed() {
  clearPrompts();
  clearFit();
  sam().fitPending = null;
  sam().press = null;
}

async function encodeCurrent() {
  const s = sam();
  const id = state.detail?.id;
  if (!id || s.status !== "ready" || s.encoded === id || s.encoding === id) return;
  s.encoding = id;
  deps.updateStatus();
  try {
    await api(`/images/${id}/sam/encode`, { method: "POST" });
    if (s.encoding === id) s.encoded = id;
  } catch (error) {
    if (!(await modelLost(error)) && state.detail?.id === id) deps.toast(error.message, "error");
  } finally {
    if (s.encoding === id) s.encoding = null;
    deps.updateStatus();
  }
}

/** After a failed request: true when the server no longer has the model. Unloaded because nobody used it: it
 * loads again and the prompt is answered once it is ready. Unloaded for a training run: smart annotation
 * turns off, so turning it on loads the model again. */
async function modelLost(error) {
  await refreshSam();
  const s = sam();
  if (s.status === "ready" || s.status === "loading") return false;
  if (s.status === "idle" && !s.unloaded) {
    await startModel();
    return true;
  }
  if (s.active) {
    await toggleSam(false);
    deps.toast(s.unloaded || error.message, "error");
  }
  return true;
}

export function clearPrompts() {
  const s = sam();
  s.points = [];
  s.box = null;
  s.drag = null;
  s.result = null;
  s.request++;
  s.busy = false;
}

export const hasPrompts = () => sam().points.length > 0 || sam().box !== null;

/** Undo the last prompt (Backspace). */
export function removeLastPrompt() {
  const s = sam();
  if (s.points.length) s.points.pop(); else s.box = null;
  promptsChanged();
}

function promptsChanged() {
  const s = sam();
  if (!hasPrompts() || (!s.box && !s.points.some(p => p.positive))) {
    s.request++;
    s.result = null;
    s.busy = false;
    deps.render();
    return;
  }
  deps.render();
  segment();
}

async function segment() {
  const s = sam();
  const id = state.detail?.id;
  if (!id) return;
  if (s.status !== "ready") { deps.updateStatus(); return; }  // runs once the model is ready
  const request = ++s.request;
  s.busy = true;
  deps.updateStatus();
  try {
    const result = await api(`/images/${id}/sam/segment`, {
      method: "POST",
      body: JSON.stringify({ points: s.points, box: s.box, max_points: currentMaxPoints() }),
    });
    if (request !== s.request || state.detail?.id !== id) return;
    s.result = result;
    s.encoded = id;
  } catch (error) {
    if (request === s.request && !(await modelLost(error))) deps.toast(error.message, "error");
  } finally {
    if (request === s.request) { s.busy = false; deps.render(); }
  }
}

/** ⏎: add the current mask as an annotation of the draw class. */
export async function acceptSam() {
  const s = sam();
  if (!state.detail) return;
  if (s.busy) return deps.toast("SAM is still working on the last click");
  if (!s.result?.found) return deps.toast(hasPrompts() ? "SAM found no object for these prompts" : "Click an object first", "error");
  state.detail.annotations.push({ id: null, class_id: +$("draw-class").value, points: [...s.result.points] });
  clearPrompts();
  await deps.save(state.detail.annotations.length - 1);
}

/* ---------- pointer handling (only while smart annotation is on) ---------- */

export function samPointerDown(e, { screen, norm }) {
  sam().press = { screen, norm, shift: e.shiftKey };
}

export function samPointerMove({ screen, norm }) {
  const s = sam();
  if (!s.press) return false;
  if (!s.drag && Math.hypot(screen[0] - s.press.screen[0], screen[1] - s.press.screen[1]) < DRAG_THRESHOLD) return true;
  s.drag = { start: clampPoint(s.press.norm), end: clampPoint(norm) };
  deps.render();
  return true;
}

export function samPointerUp() {
  const s = sam();
  const press = s.press;
  s.press = null;
  if (!press) return;
  if (s.drag) {
    const { start, end } = s.drag;
    s.drag = null;
    if (Math.abs(end[0] - start[0]) > 0.002 && Math.abs(end[1] - start[1]) > 0.002)
      s.box = [Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[0], end[0]), Math.max(start[1], end[1])];
    return promptsChanged();
  }
  addPoint(press.norm, press.screen, !press.shift);
}

/** A right click that did not pan: a background point. */
export function samRightClick(norm, screen) {
  addPoint(norm, screen, false);
}

function addPoint(norm, screen, positive) {
  const s = sam();
  if (norm[0] < 0 || norm[0] > 1 || norm[1] < 0 || norm[1] > 1) return;
  const hit = s.points.findIndex(p => Math.hypot(...diff(deps.toScreen(p.x, p.y), screen)) <= POINT_HIT);
  if (hit >= 0) s.points.splice(hit, 1);
  else s.points.push({ x: norm[0], y: norm[1], positive });
  promptsChanged();
}

// the field's value while it is valid (it is saved as it changes), else the remembered one
function currentMaxPoints() {
  const text = $("sam-max-points").value.trim();
  const value = Math.round(+text);
  return text !== "" && value >= 3 && value <= 10000 ? value : text === "" ? null : maxPoints();
}

const diff = (a, b) => [a[0] - b[0], a[1] - b[1]];
const clampPoint = ([x, y]) => [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];

/* ---------- drawing ---------- */

// the class color mixed with white: the line of what ⏎ adds, told apart from the mask's own edge
const lighten = (hex, amount) => `#${[1, 3, 5].map(i =>
  Math.round(parseInt(hex.slice(i, i + 2), 16) * (1 - amount) + 255 * amount).toString(16).padStart(2, "0")).join("")}`;

export function drawSam(ctx) {
  const s = sam();
  drawFit(ctx);
  if (!s.active || !state.img || !state.detail) return;
  const color = classColor(+$("draw-class").value);
  if (s.result?.found) {
    // SAM's mask as it is, filled: the shape further clicks steer
    const mask = new Path2D();
    for (const part of s.result.mask) {
      for (let i = 0; i < part.length; i += 2) {
        const [sx, sy] = deps.toScreen(part[i], part[i + 1]);
        i === 0 ? mask.moveTo(sx, sy) : mask.lineTo(sx, sy);
      }
      mask.closePath();
    }
    ctx.save();
    ctx.fillStyle = `${color}40`;
    ctx.fill(mask);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke(mask);
    // what ⏎ adds, unfilled and lighter: the box, or the simplified polygon
    const added = lighten(color, 0.6);
    ctx.strokeStyle = added;
    ctx.lineWidth = 1.5;
    if (state.meta.category === "detection") ctx.setLineDash([6, 4]);
    ctx.stroke(deps.shapePath({ points: s.result.points }));
    if (state.meta.category === "segmentation" && currentMaxPoints()) {
      // with a point limit the vertices are what is being tuned: show them
      ctx.fillStyle = added;
      for (let i = 0; i < s.result.points.length; i += 2) {
        const [sx, sy] = deps.toScreen(s.result.points[i], s.result.points[i + 1]);
        ctx.beginPath();
        ctx.arc(sx, sy, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }
  const box = s.drag ? [...s.drag.start, ...s.drag.end] : s.box;
  if (box) {
    const [x1, y1] = deps.toScreen(box[0], box[1]), [x2, y2] = deps.toScreen(box[2], box[3]);
    ctx.save();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
    ctx.restore();
  }
  for (const point of s.points) {
    const [sx, sy] = deps.toScreen(point.x, point.y);
    ctx.beginPath();
    ctx.arc(sx, sy, 5.5, 0, Math.PI * 2);
    ctx.fillStyle = point.positive ? "#10b981" : "#ef4444";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(sx - 2.5, sy);
    ctx.lineTo(sx + 2.5, sy);
    if (point.positive) { ctx.moveTo(sx, sy - 2.5); ctx.lineTo(sx, sy + 2.5); }
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

export function samHint() {
  const s = sam();
  if (s.status === "loading") return s.message || "Loading SAM…";
  if (s.status === "error") return `SAM failed to load: ${s.error} · press S to retry`;
  if (s.encoding === state.detail?.id) return "Encoding the image for SAM…";
  if (s.busy) return "SAM is segmenting…";
  if (s.result && !s.result.found) return "SAM found nothing here · add a point on the object or press Esc to start over";
  if (s.result?.found) return "⏎ adds the light outline · click more points to refine the filled mask · right click / Shift+click marks background · ⌫ undoes a point · Esc clears";
  return "Smart annotation: click the object · right click / Shift+click marks background · drag a box around it · S turns it off";
}

/* ---------- fit boxes (T): SAM tightens boxes that are already there ----------
   T fits the selected box, Shift+T every box of its class (of the draw class when none is selected). SAM's boxes
   are drawn dashed over them: green where it is sure, amber where the fit may be wrong (the hint says why).
   ⏎ applies the green ones, Shift+⏎ all of them, Esc drops them; one save, so one Ctrl+Z undoes it. */

const FIT_REASONS = {
  low_score: "SAM is unsure of the outline",
  different: "very different from the box",
  edge: "the outline reaches the search area's edge",
  overlap: "overlaps another box",
};
const FIT_COLORS = { fitted: "#22c55e", uncertain: "#f59e0b" };
const MIN_MOVE_PX = 0.5;  // a box SAM would move less than this on every edge is already tight

const corners = points => [points[0] - points[2] / 2, points[1] - points[3] / 2, points[0] + points[2] / 2, points[1] + points[3] / 2];
const samePoints = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);

// the largest edge move in image pixels
function movePx(before, after) {
  const a = corners(before), b = corners(after);
  return Math.max(...a.map((value, i) => Math.abs(b[i] - value) * (i % 2 ? state.img.height : state.img.width)));
}

export function clearFit() {
  const s = sam();
  s.fitRequest = (s.fitRequest ?? 0) + 1;
  if (!s.fit) return;
  s.fit = null;
  deps.render();
}

export async function fitBoxes(all = false) {
  if (!state.detail || !state.img) return;
  if (state.meta.category !== "detection") return deps.toast("Fitting boxes works on detection datasets", "error");
  const s = sam();
  if (s.status === "unavailable") return deps.toast("Fitting boxes needs SAM: run 'uv sync' to install ultralytics", "error");
  const annotations = state.detail.annotations;
  const selected = annotations.find(a => a.id === state.selected);
  const classId = selected ? selected.class_id : +$("draw-class").value;
  const targets = all ? annotations.filter(a => a.class_id === classId) : selected ? [selected] : [];
  if (!targets.length) {
    const name = state.meta.names[classId] ?? `class ${classId}`;
    return deps.toast(all ? `No ${name} boxes in this image` : "Select a box to fit · Shift+T fits every box of the draw class", "error");
  }
  if (s.status !== "ready") {
    s.fitPending = all;  // applySamState fits once the model is ready
    if (s.status !== "loading") deps.toast("Loading SAM — the boxes are fitted once it is ready");
    return startModel();
  }
  const imageId = state.detail.id;
  const request = s.fitRequest = (s.fitRequest ?? 0) + 1;
  s.fit = { imageId, busy: true, items: [], failed: 0 };
  deps.render();
  try {
    const result = await api(`/images/${imageId}/sam/fit`, {
      method: "POST",
      body: JSON.stringify({ boxes: annotations.map(a => a.points), targets: targets.map(a => annotations.indexOf(a)) }),
    });
    if (request !== s.fitRequest || state.detail?.id !== imageId) return;
    const failed = result.items.filter(item => !item.points).length;
    const items = result.items
      .map((item, i) => ({ ...item, id: targets[i].id, before: [...targets[i].points] }))
      .filter(item => item.points && movePx(item.before, item.points) >= MIN_MOVE_PX);
    if (!items.length) {
      s.fit = null;
      const tight = targets.length - failed;
      deps.toast([tight ? `${tight === 1 ? "The box is" : `${tight} boxes are`} already tight` : "", failed ? `SAM found no outline for ${failed === 1 ? "the box" : `${failed} boxes`}` : ""].filter(Boolean).join(" · "), failed ? "error" : "info");
    } else {
      s.fit = { imageId, busy: false, items, failed };
    }
  } catch (error) {
    if (request !== s.fitRequest) return;
    s.fit = null;
    if (!(await modelLost(error))) deps.toast(error.message, "error");
    else if (sam().status === "loading") s.fitPending = all;
  } finally {
    if (request === s.fitRequest) deps.render();
  }
}

/** ⏎ applies SAM's sure boxes, Shift+⏎ the uncertain ones too: one save. */
export async function applyFit(all = false) {
  const s = sam(), fit = s.fit;
  if (!fit || fit.busy || !state.detail || fit.imageId !== state.detail.id) return;
  const chosen = fit.items.filter(item => all || item.status === "fitted");
  if (!chosen.length) return deps.toast("SAM is unsure of every box here — Shift+⏎ applies them anyway, Esc drops them", "error");
  let applied = 0;
  for (const item of chosen) {
    const annotation = state.detail.annotations.find(a => a.id === item.id && samePoints(a.points, item.before));
    if (annotation) { annotation.points = [...item.points]; applied++; }
  }
  clearFit();
  if (!applied) return deps.toast("The boxes changed meanwhile; press T again", "error");
  await deps.save();
  deps.toast(`Fitted ${applied} box${applied === 1 ? "" : "es"} — Ctrl+Z undoes`);
}

function drawFit(ctx) {
  const fit = sam().fit;
  if (!fit || fit.busy || !state.img || !state.detail || fit.imageId !== state.detail.id) return;
  ctx.save();
  for (const item of fit.items) {
    const [l, t, r, b] = corners(item.points);
    const [x1, y1] = deps.toScreen(l, t), [x2, y2] = deps.toScreen(r, b);
    ctx.setLineDash([]);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgb(0 0 0 / 65%)";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = FIT_COLORS[item.status] ?? FIT_COLORS.uncertain;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  ctx.restore();
}

export function fitHint() {
  const fit = sam().fit;
  if (!fit) return null;
  if (fit.busy) return "SAM is fitting the boxes…";
  const sure = fit.items.filter(item => item.status === "fitted").length;
  const unsure = fit.items.filter(item => item.status !== "fitted");
  const reasons = [...new Set(unsure.flatMap(item => item.reasons))].map(reason => FIT_REASONS[reason] ?? reason);
  const parts = [];
  if (sure) parts.push(`${sure} tighter box${sure === 1 ? "" : "es"} (green)`);
  if (unsure.length) parts.push(`${unsure.length} uncertain (amber: ${reasons.join(", ")})`);
  if (fit.failed) parts.push(`no outline for ${fit.failed}`);
  parts.push(sure ? "⏎ applies the green ones" : "", unsure.length ? "Shift+⏎ applies all" : "", "Esc drops them");
  return parts.filter(Boolean).join(" · ");
}
