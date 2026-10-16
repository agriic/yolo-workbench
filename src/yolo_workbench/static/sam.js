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
const DRAG_THRESHOLD = 4;   // px of pointer travel that turns a click into a box drag
const POINT_HIT = 8;        // px: clicking an existing point removes it

const used = () => { try { return localStorage.getItem(USED_KEY) === "1"; } catch { return false; } };
const sam = () => state.sam;

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
  const button = $("smart-annotate");
  button.disabled = s.status === "unavailable";
  button.title = {
    unavailable: "Smart annotation needs ultralytics",
    loading: data.message || "Loading SAM…",
    error: data.error || "SAM failed to load — click to retry",
    ready: `Smart annotation with ${data.model}${data.device ? ` on ${data.device}` : ""} (S)`,
  }[s.status] || "Smart annotation: click objects, SAM outlines them (S)";
  button.classList.toggle("loading", s.status === "loading");
  clearTimeout(s.poll);
  if (s.status === "loading") s.poll = setTimeout(refreshSam, 1000);
  if (s.status === "ready" && !wasReady) {
    encodeCurrent();
    if (s.points.length || s.box) segment();
  }
  if (s.status === "error" && s.active) deps.toast(s.error, "error");
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
  if (!used() || state.meta.category === "classification" || sam().status === "unavailable") return;
  if (sam().status === "ready") encodeCurrent(); else startModel();
}

export function samEditorClosed() {
  clearPrompts();
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
    if (state.detail?.id === id) deps.toast(error.message, "error");
  } finally {
    if (s.encoding === id) s.encoding = null;
    deps.updateStatus();
  }
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
      body: JSON.stringify({ points: s.points, box: s.box }),
    });
    if (request !== s.request || state.detail?.id !== id) return;
    s.result = result;
    s.encoded = id;
  } catch (error) {
    if (request === s.request) deps.toast(error.message, "error");
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

const diff = (a, b) => [a[0] - b[0], a[1] - b[1]];
const clampPoint = ([x, y]) => [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];

/* ---------- drawing ---------- */

export function drawSam(ctx) {
  const s = sam();
  if (!s.active || !state.img || !state.detail) return;
  const color = classColor(+$("draw-class").value);
  if (s.result?.found) {
    const outline = new Path2D();
    for (let i = 0; i < s.result.outline.length; i += 2) {
      const [sx, sy] = deps.toScreen(s.result.outline[i], s.result.outline[i + 1]);
      i === 0 ? outline.moveTo(sx, sy) : outline.lineTo(sx, sy);
    }
    outline.closePath();
    ctx.save();
    ctx.fillStyle = `${color}40`;
    ctx.fill(outline);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke(outline);
    if (state.meta.category === "detection") {
      ctx.setLineDash([6, 4]);
      ctx.stroke(deps.shapePath({ points: s.result.points }));
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
  if (s.result?.found) return "⏎ adds the annotation · click more points to refine · right click / Shift+click marks background · ⌫ undoes a point · Esc clears";
  return "Smart annotation: click the object · right click / Shift+click marks background · drag a box around it · S turns it off";
}
