import { $, api, apiUrl, clamp, classColor, esc } from "./api.js";
import { state } from "./state.js";
import { startedBy } from "./collab.js";
let deps;

export function configureEmbeddings(dependencies) {
  deps = dependencies;
}

/* ---------- embeddings (Voxel51 gt_viz) ---------- */

export async function refreshEmbeddings() {
  try {
    applyEmbedState(await api("/embeddings"));
  } catch (error) { deps.toast(error.message, "error"); }
}

export async function computeEmbeddings() {
  try {
    applyEmbedState(await api("/embeddings/compute", { method: "POST" }));
  } catch (error) { deps.toast(error.message, "error"); }
}

export function applyEmbedState(data) {
  const embed = state.embed;
  const wasReady = embed.status === "ready";
  embed.status = data.status;
  embed.error = data.error;
  embed.items = data.items;
  if (!wasReady || embed.status !== "ready") embed.selection = [];
  else {
    // the same objects with the server's ids and classes now (deleting a line renumbers the ones after it)
    const current = new Map(embed.items.map(point => [point.uid, point]));
    embed.selection = embed.selection.map(point => current.get(point.uid)).filter(Boolean);
  }
  embed.hovered = null;
  $("embed-tooltip").hidden = true;
  renderEmbedSelection();
  clearTimeout(embed.poll);
  if (embed.status === "computing")
    embed.poll = setTimeout(refreshEmbeddings, 2000);
  $("compute-embeddings").disabled = embed.status === "computing";
  $("compute-embeddings").textContent = embed.status === "ready" ? "Recompute gt_viz" : "Compute gt_viz";
  const statusText = {
    idle: "Not computed yet — click “Compute gt_viz”",
    computing: `Computing embeddings… this can take a while${startedBy(data.started_by)}`,
    ready: `${embed.items.length} objects`,
    error: embed.error || "Failed",
    unavailable: embed.error || "fiftyone is not installed",
  }[embed.status] || "";
  $("embed-status").textContent = statusText;
  if (embed.status === "error") deps.toast(embed.error, "error");
  renderEmbeddings();
}

// The points shown, and (`dimmed`) those of the split whose class is hidden: drawn faintly behind the others so
// the blobs stay visible around them, but not hovered or selected.
export function embedFiltered({ dimmed = false } = {}) {
  const classes = state.embed.classes, split = $("embed-split").value;
  if (dimmed && classes.size === 0) return [];
  return state.embed.items.filter(p =>
    (classes.size === 0 || classes.has(p.class_id) !== dimmed) && (split === "all" || p.split === split));
}

// Zoom limits: deep enough to pull apart a tight cluster of near-duplicates.
const ZOOM_2D = [0.5, 200], ZOOM_3D = [0.45, 200];
const WHEEL_SPEED = 0.0015;
const DIMMED_ALPHA = 0.2;

function axisRotation([x, y, z], angle) {
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return [
    t * x * x + c, t * x * y - s * z, t * x * z + s * y,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c,
  ];
}

function multiply(a, b) {
  const out = new Array(9);
  for (let row = 0; row < 3; row++) for (let col = 0; col < 3; col++)
    out[row * 3 + col] = a[row * 3] * b[col] + a[row * 3 + 1] * b[3 + col] + a[row * 3 + 2] * b[6 + col];
  return out;
}

// Rows of a rotation stay orthonormal however many drags are multiplied in.
function orthonormalize(m) {
  const unit = v => { const n = Math.hypot(...v); return v.map(c => c / n); };
  const r0 = unit(m.slice(0, 3));
  const r1 = m.slice(3, 6), d = r1[0] * r0[0] + r1[1] * r0[1] + r1[2] * r0[2];
  const u1 = unit(r1.map((c, i) => c - d * r0[i]));
  return [...r0, ...u1, r0[1] * u1[2] - r0[2] * u1[1], r0[2] * u1[0] - r0[0] * u1[2], r0[0] * u1[1] - r0[1] * u1[0]];
}

const DEFAULT_ROTATION = multiply(axisRotation([1, 0, 0], -0.35), axisRotation([0, 1, 0], -0.65));
const viewRotation = () => state.embed.rotation ?? DEFAULT_ROTATION;

// Pixels per cube unit at the target's depth.
const scale3d = (w, h) => Math.min(w, h) * 0.34 * state.embed.zoom;

// A screen offset at the target's depth, in cube coordinates (the view rotation undone).
function screenToCube(dx, dy, w, h) {
  const r = viewRotation(), scale = scale3d(w, h), u = dx / scale, v = -dy / scale;
  return [r[0] * u + r[3] * v, r[1] * u + r[4] * v, r[2] * u + r[5] * v];
}

export function embedLayout() {
  const wrap = $("embed-wrap"), pad = 24;
  const w = wrap.clientWidth, h = wrap.clientHeight;
  const { x: panX, y: panY } = state.embed.pan2d;
  const zoom = state.embed.zoom2d;
  return {
    w, h,
    sx: x => w / 2 + panX + (x - 0.5) * (w - 2 * pad) * zoom,
    sy: y => h / 2 + panY - (y - 0.5) * (h - 2 * pad) * zoom,
  };
}

export function setEmbedMode(mode) {
  const embed = state.embed;
  embed.mode = mode;
  embed.band = null;
  embed.rotate = null;
  embed.panDrag = null;
  embed.hovered = null;
  $("embed-tooltip").hidden = true;
  $("embed-mode-2d").classList.toggle("active", mode === "2d");
  $("embed-mode-3d").classList.toggle("active", mode === "3d");
  $("embed-reset-view").hidden = false;
  $("embed-view-hint").hidden = false;
  $("embed-view-hint").textContent = mode === "3d"
    ? "Drag to select · Ctrl + drag to rotate · Right-drag to pan · Scroll to zoom"
    : "Drag to select · Ctrl + drag or right-drag to pan · Scroll to zoom";
  $("embed-canvas").style.cursor = "crosshair";
  renderEmbedSelection();
  renderEmbeddings();
}

export function resetEmbedView() {
  if (state.embed.mode === "3d") {
    state.embed.rotation = null;
    state.embed.target = [0, 0, 0];
    state.embed.zoom = 1;
  } else {
    state.embed.pan2d = { x: 0, y: 0 };
    state.embed.zoom2d = 1;
  }
  renderEmbeddings();
}

// Projects a point (0..1 per axis) to the canvas: rotated around the target, which sits at the canvas center.
// The camera stays farther from the target than any cube point, so nothing comes to lie behind it.
export function embedProjector(w, h) {
  const r = viewRotation(), [tx, ty, tz] = state.embed.target;
  const distance = 3.2 + Math.hypot(tx, ty, tz), scale = scale3d(w, h);
  return point => {
    const x = (point.x - 0.5) * 2 - tx;
    const y = (point.y - 0.5) * 2 - ty;
    const z = ((point.z ?? 0.5) - 0.5) * 2 - tz;
    const rx = r[0] * x + r[1] * y + r[2] * z;
    const ry = r[3] * x + r[4] * y + r[5] * z;
    const depth = r[6] * x + r[7] * y + r[8] * z;
    const perspective = distance / (distance - depth);
    return { x: w / 2 + rx * scale * perspective, y: h / 2 - ry * scale * perspective, depth, perspective };
  };
}

export function embedScreenPoints({ dimmed = false } = {}) {
  const points = embedFiltered({ dimmed });
  if (state.embed.mode === "2d") {
    const { sx, sy } = embedLayout();
    return points.map(point => ({ point, x: sx(point.x), y: sy(point.y), depth: 0, perspective: 1 }));
  }
  const wrap = $("embed-wrap"), project = embedProjector(wrap.clientWidth, wrap.clientHeight);
  return points.map(point => ({ point, ...project(point) }));
}

export function renderEmbedAxes(ctx, w, h) {
  const project = embedProjector(w, h), corners = [];
  for (const x of [0, 1]) for (const y of [0, 1]) for (const z of [0, 1])
    corners.push({ point: { x, y, z }, ...project({ x, y, z }) });
  const corner = (x, y, z) => corners[(x * 4) + (y * 2) + z];
  ctx.save();
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--embed-grid").trim();
  ctx.lineWidth = 1;
  for (const x of [0, 1]) for (const y of [0, 1]) for (const z of [0, 1]) {
    for (const [dx, dy, dz] of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
      if (x + dx > 1 || y + dy > 1 || z + dz > 1) continue;
      const a = corner(x, y, z), b = corner(x + dx, y + dy, z + dz);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
  }
  const origin = corner(0, 0, 0);
  for (const [label, target, color] of [["X", corner(1, 0, 0), "#dc6b35"], ["Y", corner(0, 1, 0), "#0c7a63"], ["Z", corner(0, 0, 1), "#526fd3"]]) {
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(target.x, target.y); ctx.stroke();
    ctx.font = "600 11px system-ui"; ctx.fillText(label, target.x + 5, target.y - 5);
  }
  ctx.restore();
}

export function renderEmbeddings() {
  const canvas = $("embed-canvas"), { w, h } = embedLayout();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(w * ratio));
  canvas.height = Math.max(1, Math.round(h * ratio));
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (state.embed.status !== "ready") return;
  if (state.embed.mode === "3d") renderEmbedAxes(ctx, w, h);
  const hovered = state.embed.hovered;
  const selected = new Set(state.embed.selection.map(p => p.annotation_id));
  const depthScaleOf = screen => state.embed.mode === "3d" ? clamp((screen.depth + 1.75) / 3.5, 0, 1) : 0.5;
  // hidden classes: one faint path per class, so overlapping dots read as their blob's outline
  const dimmed = new Map();
  for (const screen of embedScreenPoints({ dimmed: true })) {
    if (screen.x < -8 || screen.y < -8 || screen.x > w + 8 || screen.y > h + 8) continue;
    let path = dimmed.get(screen.point.class_id);
    if (!path) dimmed.set(screen.point.class_id, path = new Path2D());
    const radius = 2.2 + depthScaleOf(screen) * 1.4;
    path.moveTo(screen.x + radius, screen.y);
    path.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
  }
  ctx.globalAlpha = DIMMED_ALPHA;
  for (const [classId, path] of dimmed) {
    ctx.fillStyle = classColor(classId);
    ctx.fill(path);
  }
  ctx.globalAlpha = 1;
  const projected = embedScreenPoints();
  state.embed.projected = projected;
  if (state.embed.mode === "3d") projected.sort((a, b) => a.depth - b.depth);
  for (const screen of projected) {
    const point = screen.point;
    const isHover = point === hovered, isSelected = selected.has(point.annotation_id);
    const depthScale = depthScaleOf(screen);
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, isHover || isSelected ? 6 : 3 + depthScale * 1.8, 0, Math.PI * 2);
    ctx.globalAlpha = isHover || isSelected ? 1 : 0.4 + depthScale * 0.55;
    ctx.fillStyle = classColor(point.class_id);
    ctx.fill();
    ctx.globalAlpha = 1;
    if (isHover || isSelected) {
      ctx.strokeStyle = isSelected ? "#0c7a63" : "#fff";
      ctx.lineWidth = isSelected ? 2 : 1.5;
      ctx.stroke();
    }
  }
  const band = state.embed.band;
  if (band) {
    ctx.strokeStyle = "#0c7a63";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 4]);
    ctx.fillStyle = "rgb(12 122 99 / 8%)";
    const bx = Math.min(band.x0, band.x1), by = Math.min(band.y0, band.y1);
    const bw = Math.abs(band.x1 - band.x0), bh = Math.abs(band.y1 - band.y0);
    ctx.fillRect(bx, by, bw, bh);
    ctx.strokeRect(bx, by, bw, bh);
    ctx.setLineDash([]);
  }
}

export function embedMouse(e) {
  const rect = $("embed-canvas").getBoundingClientRect();
  return [e.clientX - rect.left, e.clientY - rect.top];
}

export function embedNearest(mx, my) {
  let best = null, bestDistance = 10;
  const points = embedScreenPoints();
  if (state.embed.mode === "3d") points.sort((a, b) => b.depth - a.depth);
  for (const screen of points) {
    const distance = Math.hypot(screen.x - mx, screen.y - my);
    if (distance < bestDistance) { best = screen.point; bestDistance = distance; }
  }
  return best;
}

// Left drag selects (Ctrl: rotates in 3D, pans in 2D); right or middle drag pans.
export function embedPointerDown(e) {
  if (state.embed.status !== "ready" || e.button > 2) return;
  $("embed-canvas").setPointerCapture(e.pointerId);
  const [mx, my] = embedMouse(e);
  const embed = state.embed, pan = e.button !== 0 || (embed.mode === "2d" && e.ctrlKey);
  if (embed.mode === "3d" && e.ctrlKey && !pan) {
    embed.rotate = { x: mx, y: my, lastX: mx, lastY: my, moved: false, additive: e.shiftKey };
  } else if (pan) {
    embed.panDrag = {
      x: mx, y: my, panX: embed.pan2d.x, panY: embed.pan2d.y, target: [...embed.target],
      moved: false, click: e.button === 0, additive: e.shiftKey,
    };
  } else {
    embed.band = { x0: mx, y0: my, x1: mx, y1: my, moved: false, additive: e.shiftKey };
    return;
  }
  $("embed-canvas").style.cursor = "grabbing";
  $("embed-tooltip").hidden = true;
}

export function embedPointerMove(e) {
  if (state.embed.status !== "ready") return;
  const [mx, my] = embedMouse(e);
  const rotate = state.embed.rotate;
  if (rotate) {
    // trackball: each move turns the view about the screen axis across the drag, so it never stops at a pole
    if (Math.hypot(mx - rotate.x, my - rotate.y) > 4) rotate.moved = true;
    const dx = mx - rotate.lastX, dy = my - rotate.lastY, length = Math.hypot(dx, dy);
    if (!length) return;
    state.embed.rotation = orthonormalize(multiply(axisRotation([dy / length, dx / length, 0], length * 0.01), viewRotation()));
    rotate.lastX = mx; rotate.lastY = my;
    return renderEmbeddings();
  }
  const panDrag = state.embed.panDrag;
  if (panDrag) {
    const dx = mx - panDrag.x, dy = my - panDrag.y;
    if (Math.hypot(dx, dy) > 4) panDrag.moved = true;
    if (state.embed.mode === "2d") state.embed.pan2d = { x: panDrag.panX + dx, y: panDrag.panY + dy };
    else {
      const { w, h } = embedLayout(), shift = screenToCube(dx, dy, w, h);
      state.embed.target = panDrag.target.map((c, i) => c - shift[i]);
    }
    return renderEmbeddings();
  }
  const band = state.embed.band;
  if (band) {
    band.x1 = mx; band.y1 = my;
    if (Math.hypot(band.x1 - band.x0, band.y1 - band.y0) > 4) band.moved = true;
    $("embed-tooltip").hidden = true;
    return renderEmbeddings();
  }
  const best = embedNearest(mx, my);
  if (best !== state.embed.hovered) {
    state.embed.hovered = best;
    renderEmbeddings();
    const tooltip = $("embed-tooltip");
    if (best) {
      tooltip.innerHTML = `
        <img src="${apiUrl(`/objects/${best.image_id}/${encodeURIComponent(best.annotation_id)}/crop?padding=0.15`)}" alt="">
        <div><span class="class-tag" style="background:${classColor(best.class_id)}">${esc(state.meta.names[best.class_id] ?? `class ${best.class_id}`)}</span>
        <strong>${esc(best.image_name)}</strong> · ${esc(best.split)}</div>`;
      tooltip.hidden = false;
    } else {
      tooltip.hidden = true;
    }
  }
  const tooltip = $("embed-tooltip");
  if (best && !tooltip.hidden) {
    const rect = $("embed-canvas").getBoundingClientRect();
    tooltip.style.left = `${Math.min(mx + 14, rect.width - 240)}px`;
    tooltip.style.top = `${Math.min(my + 14, rect.height - 200)}px`;
  }
  $("embed-canvas").style.cursor = e.ctrlKey ? "grab" : (best ? "pointer" : "crosshair");
}

export function embedPointerUp(e) {
  const rotate = state.embed.rotate;
  if (rotate) {
    state.embed.rotate = null;
    $("embed-canvas").style.cursor = "crosshair";
    if (!rotate.moved) selectEmbedPoint(embedNearest(...embedMouse(e)), rotate.additive);
    renderEmbeddings();
    return;
  }
  const panDrag = state.embed.panDrag;
  if (panDrag) {
    state.embed.panDrag = null;
    $("embed-canvas").style.cursor = "crosshair";
    if (!panDrag.moved && panDrag.click) selectEmbedPoint(embedNearest(...embedMouse(e)), panDrag.additive);
    renderEmbeddings();
    return;
  }
  const band = state.embed.band;
  if (!band) return;
  state.embed.band = null;
  const embed = state.embed;
  const previous = band.additive ? embed.selection : [];
  if (band.moved) {
    const [xl, xr] = [Math.min(band.x0, band.x1), Math.max(band.x0, band.x1)];
    const [yt, yb] = [Math.min(band.y0, band.y1), Math.max(band.y0, band.y1)];
    // Screen-space selection makes the 3D box match exactly what the user
    // sees at the current rotation, depth projection, and zoom level.
    const inBand = embedScreenPoints()
      .filter(screen => screen.x >= xl && screen.x <= xr && screen.y >= yt && screen.y <= yb)
      .map(screen => screen.point);
    const known = new Set(previous.map(p => p.annotation_id));
    embed.selection = [...previous, ...inBand.filter(p => !known.has(p.annotation_id))];
  } else {
    const point = embedNearest(...embedMouse(e));
    if (point) {
      const already = previous.some(p => p.annotation_id === point.annotation_id);
      if (band.additive) embed.selection = already ? previous.filter(p => p.annotation_id !== point.annotation_id) : [...previous, point];
      else embed.selection = [point];
    } else if (!band.additive) {
      embed.selection = [];
    }
  }
  renderEmbedSelection();
  renderEmbeddings();
}

export function selectEmbedPoint(point, additive) {
  const embed = state.embed;
  const previous = additive ? embed.selection : [];
  if (point) {
    const already = previous.some(p => p.annotation_id === point.annotation_id);
    if (additive) embed.selection = already ? previous.filter(p => p.annotation_id !== point.annotation_id) : [...previous, point];
    else embed.selection = [point];
  } else if (!additive) embed.selection = [];
  renderEmbedSelection();
}

export function embedPointerCancel() {
  state.embed.band = null;
  state.embed.rotate = null;
  state.embed.panDrag = null;
  $("embed-canvas").style.cursor = "crosshair";
  renderEmbeddings();
}

export function embedWheel(e) {
  if (state.embed.status !== "ready") return;
  e.preventDefault();
  // zooms toward the cursor: what is under it stays there (in 3D: at the target's depth)
  const [mx, my] = embedMouse(e);
  const { w, h } = embedLayout();
  const factor = Math.exp(-e.deltaY * WHEEL_SPEED);
  if (state.embed.mode === "3d") {
    const oldZoom = state.embed.zoom;
    const zoom = clamp(oldZoom * factor, ...ZOOM_3D);
    const toward = 1 - oldZoom / zoom;
    const shift = screenToCube((mx - w / 2) * toward, (my - h / 2) * toward, w, h);
    state.embed.target = state.embed.target.map((c, i) => c + shift[i]);
    state.embed.zoom = zoom;
  } else {
    const oldZoom = state.embed.zoom2d;
    const zoom = clamp(oldZoom * factor, ...ZOOM_2D);
    const ratio = zoom / oldZoom;
    const pan = state.embed.pan2d;
    state.embed.pan2d = {
      x: mx - w / 2 - (mx - w / 2 - pan.x) * ratio,
      y: my - h / 2 - (my - h / 2 - pan.y) * ratio,
    };
    state.embed.zoom2d = zoom;
  }
  renderEmbeddings();
}

export function renderEmbedSelection() {
  const selection = state.embed.selection;
  $("embed-selection-count").textContent = selection.length || "";
  $("embed-clear").hidden = !selection.length;
  $("embed-delete").hidden = !selection.length;
  const emptyHint = state.embed.mode === "3d"
    ? "Drag a box over visible points to select them. Shift adds to the selection. Ctrl + drag rotates, right-drag pans."
    : "Drag a box to select points. Shift adds to the selection. Ctrl + drag or right-drag pans and scrolling zooms.";
  $("embed-selection").innerHTML = selection.map(point => `
    <div class="embed-object">
      <img loading="lazy" src="${apiUrl(`/objects/${point.image_id}/${encodeURIComponent(point.annotation_id)}/crop?padding=0.15&v=${point.key}.${point.version || 0}`)}" alt="">
      <div class="embed-object-body">
        <div class="name" title="${esc(point.image_name)}">${esc(point.image_name)}</div>
        <div class="meta"><span class="split-badge">${esc(point.split)}</span></div>
        <div class="object-actions">
          <select data-object-class="${esc(point.annotation_id)}" data-owner="${point.image_id}" data-key="${point.key}" title="Reassign class">${deps.classOptions(point.class_id)}</select>
          <button data-open-object="${esc(point.annotation_id)}" data-owner="${point.image_id}" title="Open in editor">Open</button>
        </div>
      </div>
    </div>`).join("") || `<p class="empty-note">${emptyHint}</p>`;
}

export async function reclassifyEmbedObject(imageId, annotationId, classId, key) {
  await deps.editObject(imageId, annotationId, { classId, key });
  if (state.meta.category !== "classification") return refreshEmbeddings();  // the class and key the server has now
  // a classification image moves to its new class's folder, and the server no longer knows it by this id
  for (const point of state.embed.items)
    if (point.annotation_id === annotationId && point.image_id === imageId) {
      point.class_id = classId;
      point.version = (point.version || 0) + 1;
    }
  renderEmbedSelection();
  renderEmbeddings();
}

// Deletes the selected objects' annotations in one transaction (Ctrl+Z undoes it). The key names each one even
// when an edit elsewhere has moved it to another line.
export async function deleteEmbedSelection() {
  const operations = state.embed.selection.map(point =>
    ({ image_id: point.image_id, annotation_id: point.annotation_id, action: "delete", class_id: null, key: point.key }));
  if (!operations.length) return;
  const button = $("embed-delete");
  button.disabled = true;  // a second click would reach the server after the first delete
  try {
    await deps.bulkEditObjects(operations, "delete", () => {
      state.embed.selection = [];
      renderEmbedSelection();
      renderEmbeddings();
      refreshEmbeddings();  // the deleted objects leave the plot
    });
  } finally {
    button.disabled = false;
  }
}
