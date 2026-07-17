import { $, api, clamp, classColor, esc, HANDLE_CURSORS, pointInPolygon } from "./api.js";
import { state } from "./state.js";
let deps;

export function configureCanvas(dependencies) {
  deps = dependencies;
}

let dpr = 1;
let cssW = 0;
let cssH = 0;
export let editorRequest = 0;
export let saveQueue = Promise.resolve();
let saveSequence = 0;
const latestSaveByImage = new Map();
export const savedRevisionByImage = new Map();

/* ---------- editor ---------- */

export async function openEditor(id, focus = null, gridIndex = null) {
  const request = ++editorRequest;
  state.detail = null;
  state.editorGridIndex = null;
  state.img = null;
  state.drawing = null;
  state.drag = null;
  state.selected = null;
  render();
  await saveQueue;
  if (request !== editorRequest) return;
  let detail;
  try {
    detail = await api(`/api/v1/images/${id}`);
  } catch (error) {
    if (request === editorRequest) deps.toast(error.message, "error");
    return;
  }
  if (request !== editorRequest) return;
  state.detail = detail;
  state.editorGridIndex = Number.isInteger(gridIndex) ? gridIndex : null;
  savedRevisionByImage.set(id, detail.revision);
  state.pred.items = [];
  deps.loadEditorPredictions(id);
  state.selected = focus;
  state.drawing = null;
  state.drag = null;
  state.hovered = null;
  state.img = null;
  $("editor-title").textContent = state.detail.name;
  $("editor-subtitle").textContent = `${state.detail.split} · ${state.detail.width}×${state.detail.height}px`;
  if (!$("editor").open) $("editor").showModal();
  updateNavButtons();
  renderList();
  render();
  const img = new Image();
  img.onload = () => {
    if (!state.detail || state.detail.id !== id) return;
    state.img = img;
    resizeCanvas();
    fit();
    if (focus) {
      const annotation = state.detail.annotations.find(a => a.id === focus);
      if (annotation) zoomTo(annotation);
    }
  };
  img.onerror = () => deps.toast("Failed to load image file", "error");
  img.src = `/api/v1/images/${id}/file?v=${Date.now()}`;
}

export async function navigate(step) {
  if (!state.detail || state.editorGridIndex == null || !deps.imageGallery()?.total) return;
  const target = state.editorGridIndex + step;
  if (target < 0 || target >= deps.imageGallery().total) return;
  try {
    const item = await deps.imageGallery().itemAt(target);
    if (item) openEditor(item.id, null, target);
  } catch (error) {
    if (error.name !== "AbortError") deps.toast(error.message, "error");
  }
}

export function updateNavButtons() {
  if (!$("editor").open || !state.detail) return;
  const index = state.editorGridIndex;
  $("prev-image").disabled = index == null || index <= 0;
  $("next-image").disabled = index == null || deps.imageGallery()?.total == null || index >= deps.imageGallery().total - 1;
}

export function updateHistoryButtons() {
  $("undo").disabled = !state.canUndo;
  $("redo").disabled = !state.canRedo;
}

export function resizeCanvas() {
  const wrap = $("canvas-wrap"), canvas = $("canvas");
  dpr = window.devicePixelRatio || 1;
  cssW = wrap.clientWidth;
  cssH = wrap.clientHeight;
  canvas.width = Math.max(1, Math.round(cssW * dpr));
  canvas.height = Math.max(1, Math.round(cssH * dpr));
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  render();
}

export const toScreen = (nx, ny) => [state.view.x + nx * state.img.width * state.view.scale, state.view.y + ny * state.img.height * state.view.scale];

export function eventPoint(e) {
  const rect = $("canvas").getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  return {
    screen: [sx, sy],
    norm: [(sx - state.view.x) / (state.img.width * state.view.scale), (sy - state.view.y) / (state.img.height * state.view.scale)],
  };
}

export function fit() {
  if (!state.img) return;
  state.view.scale = clamp(Math.min((cssW - 48) / state.img.width, (cssH - 48) / state.img.height), 0.02, 40);
  state.view.x = (cssW - state.img.width * state.view.scale) / 2;
  state.view.y = (cssH - state.img.height * state.view.scale) / 2;
  render();
}

export function zoomAt(factor, sx, sy) {
  if (!state.img) return;
  const next = clamp(state.view.scale * factor, 0.02, 40);
  state.view.x = sx - (sx - state.view.x) * (next / state.view.scale);
  state.view.y = sy - (sy - state.view.y) * (next / state.view.scale);
  state.view.scale = next;
  render();
}

export function zoomCentered(factor) {
  zoomAt(factor, cssW / 2, cssH / 2);
}

export function closeEditorSession() {
  editorRequest += 1;
}

export function zoomTo(annotation) {
  if (!state.img) return;
  if (!annotation.points.length) return fit();
  const [l, t, r, b] = bounds(annotation);
  const width = Math.max((r - l) * state.img.width, 8), height = Math.max((b - t) * state.img.height, 8);
  state.view.scale = clamp(Math.min(cssW / width, cssH / height) * 0.5, 0.02, 12);
  state.view.x = cssW / 2 - (l + r) / 2 * state.img.width * state.view.scale;
  state.view.y = cssH / 2 - (t + b) / 2 * state.img.height * state.view.scale;
  render();
}

export function bounds(annotation) {
  if (state.meta.category === "detection") {
    const [cx, cy, w, h] = annotation.points;
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
  }
  const xs = annotation.points.filter((_, i) => i % 2 === 0), ys = annotation.points.filter((_, i) => i % 2 === 1);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

export function boxHandles(annotation) {
  const [l, t, r, b] = bounds(annotation), mx = (l + r) / 2, my = (t + b) / 2;
  return [[l, t], [mx, t], [r, t], [r, my], [r, b], [mx, b], [l, b], [l, my]];
}

export const selectedAnnotation = () => state.detail?.annotations.find(a => a.id === state.selected) ?? null;

/* ---------- rendering ---------- */

export function render() {
  const canvas = $("canvas"), ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  if (!state.img || !state.detail) { updateStatus(); return; }
  ctx.imageSmoothingEnabled = state.view.scale < 3;
  ctx.drawImage(state.img, state.view.x, state.view.y, state.img.width * state.view.scale, state.img.height * state.view.scale);
  if (state.meta.category !== "classification") {
    for (const annotation of state.detail.annotations)
      if (annotation.id !== state.selected) drawShape(ctx, annotation, false);
    const selected = selectedAnnotation();
    if (selected) drawShape(ctx, selected, true);
  }
  deps.drawPredictions(ctx);
  drawPreview(ctx);
  updateStatus();
}

export function shapePath(annotation) {
  const path = new Path2D();
  if (state.meta.category === "detection") {
    const [l, t, r, b] = bounds(annotation);
    const [x1, y1] = toScreen(l, t), [x2, y2] = toScreen(r, b);
    path.rect(x1, y1, x2 - x1, y2 - y1);
  } else {
    const points = annotation.points;
    for (let i = 0; i < points.length; i += 2) {
      const [sx, sy] = toScreen(points[i], points[i + 1]);
      i === 0 ? path.moveTo(sx, sy) : path.lineTo(sx, sy);
    }
    path.closePath();
  }
  return path;
}

export function drawShape(ctx, annotation, selected) {
  const color = classColor(annotation.class_id);
  const hovered = annotation.id === state.hovered;
  const path = shapePath(annotation);
  if (selected || hovered) { ctx.fillStyle = `${color}${selected ? "3a" : "26"}`; ctx.fill(path); }
  ctx.strokeStyle = color;
  ctx.lineWidth = selected || hovered ? 2.5 : 1.6;
  ctx.stroke(path);
  drawLabel(ctx, annotation, color);
  if (!selected) return;
  if (state.meta.category === "detection") {
    for (const [hx, hy] of boxHandles(annotation)) {
      const [sx, sy] = toScreen(hx, hy);
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.fillRect(sx - 4, sy - 4, 8, 8);
      ctx.strokeRect(sx - 4, sy - 4, 8, 8);
    }
  } else {
    for (let i = 0; i < annotation.points.length; i += 2) {
      const [sx, sy] = toScreen(annotation.points[i], annotation.points[i + 1]);
      ctx.beginPath();
      ctx.arc(sx, sy, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}

export function drawLabel(ctx, annotation, color) {
  const [l, t, r] = bounds(annotation);
  if ((r - l) * state.img.width * state.view.scale < 26) return;
  const [sx, sy] = toScreen(l, t);
  const name = state.meta.names[annotation.class_id] ?? `class ${annotation.class_id}`;
  ctx.font = "600 11px Inter, system-ui, sans-serif";
  const width = ctx.measureText(name).width + 10, height = 17;
  const y = sy - height - 1 < 2 ? sy + 1 : sy - height - 1;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(sx, y, width, height, 3);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.fillText(name, sx + 5, y + 12);
}

export function drawPreview(ctx) {
  if (!state.drawing) return;
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 4]);
  if (state.meta.category === "detection") {
    const [x1, y1] = toScreen(...state.drawing.start), [x2, y2] = toScreen(...state.drawing.end);
    ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
    ctx.setLineDash([]);
    const width = Math.round(Math.abs(state.drawing.end[0] - state.drawing.start[0]) * state.img.width);
    const height = Math.round(Math.abs(state.drawing.end[1] - state.drawing.start[1]) * state.img.height);
    if (width || height) {
      const text = `${width}×${height}`;
      ctx.font = "600 11px Inter, system-ui, sans-serif";
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = "rgb(0 0 0 / 65%)";
      ctx.fillRect(Math.max(x1, x2) + 6, Math.max(y1, y2) - 8, tw + 10, 17);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, Math.max(x1, x2) + 11, Math.max(y1, y2) + 4);
    }
  } else {
    const points = state.drawing.points.map(p => toScreen(...p));
    ctx.beginPath();
    points.forEach(([sx, sy], i) => i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy));
    if (state.drawing.cursor) ctx.lineTo(...toScreen(...state.drawing.cursor));
    ctx.stroke();
    ctx.setLineDash([]);
    const closable = state.drawing.points.length >= 3 && state.drawing.cursor
      && screenDistance(points[0], toScreen(...state.drawing.cursor)) < 12;
    points.forEach(([sx, sy], i) => {
      ctx.beginPath();
      ctx.arc(sx, sy, i === 0 && closable ? 7 : 4, 0, Math.PI * 2);
      ctx.fillStyle = i === 0 && closable ? "#10b981" : "#fff";
      ctx.fill();
    });
  }
}

export const screenDistance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function updateStatus() {
  $("zoom-level").textContent = `${Math.round(state.view.scale * 100)}%`;
  $("status-image").textContent = state.detail
    ? `${state.detail.width}×${state.detail.height}px · ${state.detail.annotations.length} annotation${state.detail.annotations.length === 1 ? "" : "s"}`
    : "";
  let hint;
  if (state.meta.category === "classification")
    hint = "Pick the class in the panel or press 1–9 · drag to pan · scroll to zoom";
  else if (state.drawing && state.meta.category === "segmentation")
    hint = "Click to add points · click the first point or press ⏎ to close · Esc cancels";
  else if (state.drawing) hint = "Release to create the box";
  else if (state.meta.category === "detection") hint = "Drag on empty space to draw a box · scroll to zoom · Space+drag to pan";
  else hint = "Click on empty space to start a polygon · scroll to zoom · Space+drag to pan";
  $("status-hint").textContent = hint;
}

/* ---------- hit testing & pointer interaction ---------- */

export function hitHandle(screen) {
  const annotation = selectedAnnotation();
  if (!annotation) return null;
  if (state.meta.category === "detection") {
    const handles = boxHandles(annotation);
    for (let i = 0; i < handles.length; i++)
      if (screenDistance(toScreen(...handles[i]), screen) <= 9) return { mode: "resize", index: i, a: annotation };
  } else {
    for (let i = 0; i * 2 < annotation.points.length; i++)
      if (screenDistance(toScreen(annotation.points[i * 2], annotation.points[i * 2 + 1]), screen) <= 9)
        return { mode: "vertex", index: i, a: annotation };
  }
  return null;
}

export function hitShape([x, y]) {
  for (const annotation of [...state.detail.annotations].reverse()) {
    if (state.meta.category === "detection") {
      const [l, t, r, b] = bounds(annotation);
      if (x >= l && x <= r && y >= t && y <= b) return annotation;
    } else if (pointInPolygon(x, y, annotation.points)) {
      return annotation;
    }
  }
  return null;
}

export function pointerDown(e) {
  if (!state.img) return;
  $("canvas").setPointerCapture(e.pointerId);
  if (e.button === 1 || e.button === 2 || state.spaceDown) {
    state.pan = { sx: e.clientX, sy: e.clientY, ox: state.view.x, oy: state.view.y };
    $("canvas").style.cursor = "grabbing";
    return;
  }
  if (e.button !== 0) return;
  if (state.meta.category === "classification") {
    // no geometry to draw or select; left-drag pans instead
    state.pan = { sx: e.clientX, sy: e.clientY, ox: state.view.x, oy: state.view.y };
    $("canvas").style.cursor = "grabbing";
    return;
  }
  const { screen, norm } = eventPoint(e);
  const clamped = [clamp(norm[0], 0, 1), clamp(norm[1], 0, 1)];
  if (state.meta.category === "segmentation" && state.drawing) {
    if (state.drawing.points.length >= 3 && screenDistance(toScreen(...state.drawing.points[0]), screen) < 12)
      return finishPolygon();
    state.drawing.points.push(clamped);
    return render();
  }
  const handle = hitHandle(screen);
  if (handle) {
    if (handle.mode === "vertex" && e.altKey) return removeVertex(handle.a, handle.index);
    state.drag = { ...handle, start: norm, original: [...handle.a.points] };
    return;
  }
  const found = hitShape(norm);
  if (found) {
    state.selected = found.id;
    state.drag = { mode: "move", a: found, start: norm, original: [...found.points] };
    renderList();
    return render();
  }
  const prediction = deps.hitPrediction(norm);
  if (prediction) {
    if (prediction.class_id === null) return deps.toast("Prediction class is unmapped — adjust the class mapping first", "error");
    return deps.acceptPredictions([prediction.id]);
  }
  state.selected = null;
  state.drawing = state.meta.category === "detection" ? { start: clamped, end: clamped } : { points: [clamped] };
  renderList();
  render();
}

export function pointerMove(e) {
  if (!state.img) return;
  const { screen, norm } = eventPoint(e);
  const inside = norm[0] >= 0 && norm[0] <= 1 && norm[1] >= 0 && norm[1] <= 1;
  $("status-cursor").textContent = inside
    ? `${Math.round(norm[0] * state.img.width)}, ${Math.round(norm[1] * state.img.height)}px` : "";
  if (state.pan) {
    state.view.x = state.pan.ox + e.clientX - state.pan.sx;
    state.view.y = state.pan.oy + e.clientY - state.pan.sy;
    return render();
  }
  if (state.drag) { applyDrag(norm); return render(); }
  if (state.drawing) {
    if (state.meta.category === "detection") state.drawing.end = [clamp(norm[0], 0, 1), clamp(norm[1], 0, 1)];
    else state.drawing.cursor = [clamp(norm[0], 0, 1), clamp(norm[1], 0, 1)];
    return render();
  }
  // hover feedback
  let cursor = "crosshair", hovered = null;
  if (state.meta.category === "classification") {
    cursor = "grab";
  } else if (state.spaceDown) {
    cursor = "grab";
  } else {
    const handle = hitHandle(screen);
    if (handle) {
      cursor = handle.mode === "resize" ? HANDLE_CURSORS[handle.index] : "move";
    } else {
      const found = hitShape(norm);
      if (found) { cursor = "move"; hovered = found.id; }
      else if (deps.hitPrediction(norm)) cursor = "pointer";
    }
  }
  $("canvas").style.cursor = cursor;
  if (hovered !== state.hovered) { state.hovered = hovered; render(); }
}

export function applyDrag(norm) {
  const { a, mode, index, start, original } = state.drag;
  const px = clamp(norm[0], 0, 1), py = clamp(norm[1], 0, 1);
  if (mode === "vertex") {
    a.points[index * 2] = px;
    a.points[index * 2 + 1] = py;
  } else if (mode === "resize") {
    let [l, t, r, b] = [original[0] - original[2] / 2, original[1] - original[3] / 2, original[0] + original[2] / 2, original[1] + original[3] / 2];
    if ([0, 6, 7].includes(index)) l = px;
    if ([2, 3, 4].includes(index)) r = px;
    if ([0, 1, 2].includes(index)) t = py;
    if ([4, 5, 6].includes(index)) b = py;
    const nl = Math.min(l, r), nr = Math.max(l, r), nt = Math.min(t, b), nb = Math.max(t, b);
    a.points = [(nl + nr) / 2, (nt + nb) / 2, Math.max(0.001, nr - nl), Math.max(0.001, nb - nt)];
  } else {
    let dx = norm[0] - start[0], dy = norm[1] - start[1];
    const detection = state.meta.category === "detection";
    const xs = detection ? [original[0] - original[2] / 2, original[0] + original[2] / 2] : original.filter((_, i) => i % 2 === 0);
    const ys = detection ? [original[1] - original[3] / 2, original[1] + original[3] / 2] : original.filter((_, i) => i % 2 === 1);
    dx = clamp(dx, -Math.min(...xs), 1 - Math.max(...xs));
    dy = clamp(dy, -Math.min(...ys), 1 - Math.max(...ys));
    a.points = detection
      ? [original[0] + dx, original[1] + dy, original[2], original[3]]
      : original.map((value, i) => i % 2 === 0 ? value + dx : value + dy);
  }
}

export async function pointerUp() {
  if (state.pan) {
    state.pan = null;
    $("canvas").style.cursor = state.spaceDown ? "grab" : "crosshair";
    return;
  }
  if (state.drag) {
    const changed = JSON.stringify(state.drag.a.points) !== JSON.stringify(state.drag.original);
    state.drag = null;
    if (changed) await save(); else render();
    return;
  }
  if (state.drawing && state.meta.category === "detection") {
    const { start, end } = state.drawing;
    state.drawing = null;
    const width = Math.abs(end[0] - start[0]), height = Math.abs(end[1] - start[1]);
    if (width > 0.004 && height > 0.004) {
      state.detail.annotations.push({
        id: null,
        class_id: +$("draw-class").value,
        points: [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, width, height],
      });
      await save(state.detail.annotations.length - 1);
    } else {
      render();
    }
  }
}

export async function finishPolygon() {
  const points = state.drawing.points.flat();
  state.drawing = null;
  state.detail.annotations.push({ id: null, class_id: +$("draw-class").value, points });
  await save(state.detail.annotations.length - 1);
}

export function removeVertex(annotation, index) {
  if (annotation.points.length <= 6) return deps.toast("A polygon needs at least 3 points", "error");
  annotation.points.splice(index * 2, 2);
  save();
}

export async function removeAnnotation(id) {
  state.detail.annotations = state.detail.annotations.filter(a => a.id !== id);
  if (state.selected === id) state.selected = null;
  await save();
}

export async function deleteSelected() {
  if (state.meta.category === "classification") return;
  if (state.selected && state.detail) await removeAnnotation(state.selected);
}

/* ---------- persistence ---------- */

export async function save(selectIndex = null) {
  if (!state.detail) return;
  const image = state.detail;
  const index = selectIndex ?? image.annotations.findIndex(a => a.id === state.selected);
  const request = editorRequest;
  const sequence = ++saveSequence;
  const annotations = image.annotations.map(annotation => ({
    id: annotation.id,
    class_id: annotation.class_id,
    points: [...annotation.points],
  }));
  latestSaveByImage.set(image.id, sequence);

  const persist = async () => {
    let detail;
    try {
      detail = await api(`/api/v1/images/${image.id}/annotations`, {
        method: "PUT",
        body: JSON.stringify({
          revision: savedRevisionByImage.get(image.id) ?? image.revision,
          annotations,
        }),
      });
      savedRevisionByImage.set(image.id, detail.revision);
      state.canUndo = true;
      state.canRedo = false;
      updateHistoryButtons();
      deps.statisticsChanged();
    } catch (error) {
      deps.toast(error.message, "error");
      if (request === editorRequest && state.detail?.id === image.id && latestSaveByImage.get(image.id) === sequence) {
        try {
          detail = await api(`/api/v1/images/${image.id}`);
          if (request !== editorRequest || state.detail?.id !== image.id || latestSaveByImage.get(image.id) !== sequence) return;
          savedRevisionByImage.set(image.id, detail.revision);
          state.detail = detail;
          state.selected = null;
          renderList();
          render();
        } catch (reloadError) {
          deps.toast(reloadError.message, "error");
        }
      }
      return;
    }

    const isLatest = latestSaveByImage.get(image.id) === sequence;
    if (request === editorRequest && state.detail?.id === image.id && isLatest) {
      state.detail = detail;
      // the server re-derives annotation ids from line numbers, so re-select by position
      state.selected = index >= 0 ? detail.annotations[index]?.id ?? null : null;
      renderList();
      render();
    }
    if (isLatest) {
      deps.loadImages();
      deps.loadObjects();
      deps.loadIssues();
    }
  };

  const queued = saveQueue.then(persist, persist);
  saveQueue = queued.catch(() => {});
  return queued;
}

export async function applyHistory(direction) {
  if (direction === "undo" ? !state.canUndo : !state.canRedo) return;
  await saveQueue;
  try {
    const result = await api(`/api/v1/history/${direction}`, { method: "POST" });
    state.canUndo = result.can_undo;
    state.canRedo = result.can_redo;
    updateHistoryButtons();
    deps.statisticsChanged();
    const request = editorRequest;
    const imageId = state.detail?.id;
    if (imageId && result.image_ids.includes(imageId)) {
      const detail = await api(`/api/v1/images/${imageId}`);
      if (request === editorRequest && state.detail?.id === imageId) {
        state.detail = detail;
        savedRevisionByImage.set(imageId, detail.revision);
        if (!detail.annotations.some(a => a.id === state.selected)) state.selected = null;
        renderList();
        render();
      }
    }
    await Promise.all([deps.loadImages(), deps.loadObjects(), deps.loadIssues()]);
  } catch (error) {
    deps.toast(error.message, "error");
  }
}

/* ---------- annotation list (inspector) ---------- */

export function renderList() {
  if (!state.detail) return;
  const annotations = state.detail.annotations;
  if (state.meta.category === "classification") {
    $("annotation-count").textContent = "";
    $("annotation-list").innerHTML = annotations.map(a => `
      <div class="annotation-row" data-annotation="${esc(a.id)}">
        <span class="swatch" style="background:${classColor(a.class_id)}"></span>
        <select class="row-class">${deps.classOptions(a.class_id)}</select>
        <span class="row-meta">whole image</span>
      </div>`).join("");
    return;
  }
  $("annotation-count").textContent = annotations.length || "";
  $("annotation-list").innerHTML = annotations.map(a => `
    <div class="annotation-row ${a.id === state.selected ? "selected" : ""}" data-annotation="${esc(a.id)}" title="Double-click to zoom">
      <span class="swatch" style="background:${classColor(a.class_id)}"></span>
      <select class="row-class">${deps.classOptions(a.class_id)}</select>
      <span class="row-meta">${sizeLabel(a)}</span>
      <button class="row-delete" title="Delete annotation">×</button>
    </div>`).join("")
    || `<p class="empty-note">No annotations yet. ${state.meta.category === "detection" ? "Drag on the image to draw a box." : "Click on the image to start a polygon."}</p>`;
}

export function sizeLabel(annotation) {
  if (state.meta.category === "detection")
    return `${Math.round(annotation.points[2] * state.detail.width)}×${Math.round(annotation.points[3] * state.detail.height)}`;
  return `${annotation.points.length / 2} pts`;
}
