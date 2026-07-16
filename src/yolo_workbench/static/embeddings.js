import { $, api, clamp, classColor, esc } from "./api.js";
import { state } from "./state.js";
let deps;

export function configureEmbeddings(dependencies) {
  deps = dependencies;
}

/* ---------- embeddings (Voxel51 gt_viz) ---------- */

export async function refreshEmbeddings() {
  try {
    applyEmbedState(await api("/api/v1/embeddings"));
  } catch (error) { deps.toast(error.message, "error"); }
}

export async function computeEmbeddings() {
  try {
    applyEmbedState(await api("/api/v1/embeddings/compute", { method: "POST" }));
  } catch (error) { deps.toast(error.message, "error"); }
}

export function applyEmbedState(data) {
  const embed = state.embed;
  const wasReady = embed.status === "ready";
  embed.status = data.status;
  embed.error = data.error;
  embed.items = data.items;
  if (!wasReady || embed.status !== "ready") { embed.selection = []; renderEmbedSelection(); }
  clearTimeout(embed.poll);
  if (embed.status === "computing")
    embed.poll = setTimeout(refreshEmbeddings, 2000);
  $("compute-embeddings").disabled = embed.status === "computing";
  $("compute-embeddings").textContent = embed.status === "ready" ? "Recompute gt_viz" : "Compute gt_viz";
  const statusText = {
    idle: "Not computed yet — click “Compute gt_viz”",
    computing: "Computing embeddings… this can take a while",
    ready: `${embed.items.length} objects`,
    error: embed.error || "Failed",
    unavailable: embed.error || "fiftyone is not installed",
  }[embed.status] || "";
  $("embed-status").textContent = statusText;
  if (embed.status === "error") deps.toast(embed.error, "error");
  renderEmbeddings();
}

export function embedFiltered() {
  const classes = state.embed.classes, split = $("embed-split").value;
  return state.embed.items.filter(p =>
    (classes.size === 0 || classes.has(p.class_id)) && (split === "all" || p.split === split));
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
    ? "Drag to select · Ctrl + drag to rotate · Scroll to zoom"
    : "Drag to select · Ctrl + drag to pan · Scroll to zoom";
  $("embed-canvas").style.cursor = "crosshair";
  renderEmbedSelection();
  renderEmbeddings();
}

export function resetEmbedView() {
  if (state.embed.mode === "3d") {
    state.embed.rotation = { yaw: -0.65, pitch: -0.35 };
    state.embed.zoom = 1;
  } else {
    state.embed.pan2d = { x: 0, y: 0 };
    state.embed.zoom2d = 1;
  }
  renderEmbeddings();
}

export function projectEmbedPoint(point, w, h) {
  const x = (point.x - 0.5) * 2;
  const y = (point.y - 0.5) * 2;
  const z = ((point.z ?? 0.5) - 0.5) * 2;
  const { yaw, pitch } = state.embed.rotation;
  const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
  const cosX = Math.cos(pitch), sinX = Math.sin(pitch);
  const rx = x * cosY + z * sinY;
  const rz = -x * sinY + z * cosY;
  const ry = y * cosX - rz * sinX;
  const depth = y * sinX + rz * cosX;
  const perspective = 3.2 / (3.2 - depth);
  const scale = Math.min(w, h) * 0.34 * state.embed.zoom * perspective;
  return { x: w / 2 + rx * scale, y: h / 2 - ry * scale, depth, perspective };
}

export function embedScreenPoints() {
  const wrap = $("embed-wrap"), w = wrap.clientWidth, h = wrap.clientHeight;
  if (state.embed.mode === "2d") {
    const { sx, sy } = embedLayout();
    return embedFiltered().map(point => ({ point, x: sx(point.x), y: sy(point.y), depth: 0, perspective: 1 }));
  }
  return embedFiltered().map(point => ({ point, ...projectEmbedPoint(point, w, h) }));
}

export function renderEmbedAxes(ctx, w, h) {
  const corners = [];
  for (const x of [0, 1]) for (const y of [0, 1]) for (const z of [0, 1])
    corners.push({ point: { x, y, z }, ...projectEmbedPoint({ x, y, z }, w, h) });
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
  const projected = embedScreenPoints();
  state.embed.projected = projected;
  if (state.embed.mode === "3d") projected.sort((a, b) => a.depth - b.depth);
  for (const screen of projected) {
    const point = screen.point;
    const isHover = point === hovered, isSelected = selected.has(point.annotation_id);
    const depthScale = state.embed.mode === "3d" ? clamp((screen.depth + 1.75) / 3.5, 0, 1) : 0.5;
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

export function embedPointerDown(e) {
  if (state.embed.status !== "ready" || e.button !== 0) return;
  $("embed-canvas").setPointerCapture(e.pointerId);
  const [mx, my] = embedMouse(e);
  if (state.embed.mode === "3d" && e.ctrlKey) {
    const { yaw, pitch } = state.embed.rotation;
    state.embed.rotate = { x: mx, y: my, lastX: mx, lastY: my, yaw, pitch, moved: false, additive: e.shiftKey };
    $("embed-canvas").style.cursor = "grabbing";
    $("embed-tooltip").hidden = true;
    return;
  }
  if (state.embed.mode === "2d" && e.ctrlKey) {
    const { x, y } = state.embed.pan2d;
    state.embed.panDrag = { x: mx, y: my, panX: x, panY: y, moved: false, additive: e.shiftKey };
    $("embed-canvas").style.cursor = "grabbing";
    $("embed-tooltip").hidden = true;
    return;
  }
  state.embed.band = { x0: mx, y0: my, x1: mx, y1: my, moved: false, additive: e.shiftKey };
}

export function embedPointerMove(e) {
  if (state.embed.status !== "ready") return;
  const [mx, my] = embedMouse(e);
  const rotate = state.embed.rotate;
  if (rotate) {
    if (Math.hypot(mx - rotate.x, my - rotate.y) > 4) rotate.moved = true;
    state.embed.rotation.yaw = rotate.yaw + (mx - rotate.x) * 0.01;
    state.embed.rotation.pitch = clamp(rotate.pitch + (my - rotate.y) * 0.01, -Math.PI / 2, Math.PI / 2);
    rotate.lastX = mx; rotate.lastY = my;
    return renderEmbeddings();
  }
  const panDrag = state.embed.panDrag;
  if (panDrag) {
    if (Math.hypot(mx - panDrag.x, my - panDrag.y) > 4) panDrag.moved = true;
    state.embed.pan2d = { x: panDrag.panX + mx - panDrag.x, y: panDrag.panY + my - panDrag.y };
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
        <img src="/api/v1/objects/${best.image_id}/${encodeURIComponent(best.annotation_id)}/crop?padding=0.15" alt="">
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
    if (!panDrag.moved) selectEmbedPoint(embedNearest(...embedMouse(e)), panDrag.additive);
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
  if (state.embed.mode === "3d") {
    state.embed.zoom = clamp(state.embed.zoom * Math.exp(-e.deltaY * 0.001), 0.45, 2.5);
  } else {
    const [mx, my] = embedMouse(e);
    const { w, h } = embedLayout();
    const oldZoom = state.embed.zoom2d;
    const zoom = clamp(oldZoom * Math.exp(-e.deltaY * 0.001), 0.5, 8);
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
  const emptyHint = state.embed.mode === "3d"
    ? "Drag a box over visible points to select them. Shift adds to the selection. Ctrl + drag rotates."
    : "Drag a box to select points. Shift adds to the selection. Ctrl + drag pans and scrolling zooms.";
  $("embed-selection").innerHTML = selection.map(point => `
    <div class="embed-object">
      <img loading="lazy" src="/api/v1/objects/${point.image_id}/${encodeURIComponent(point.annotation_id)}/crop?padding=0.15&v=${point.version || 0}" alt="">
      <div class="embed-object-body">
        <div class="name" title="${esc(point.image_name)}">${esc(point.image_name)}</div>
        <div class="meta"><span class="split-badge">${esc(point.split)}</span></div>
        <div class="object-actions">
          <select data-object-class="${esc(point.annotation_id)}" data-owner="${point.image_id}" title="Reassign class">${deps.classOptions(point.class_id)}</select>
          <button data-open-object="${esc(point.annotation_id)}" data-owner="${point.image_id}" title="Open in editor">Open</button>
        </div>
      </div>
    </div>`).join("") || `<p class="empty-note">${emptyHint}</p>`;
}

export async function reclassifyEmbedObject(imageId, annotationId, classId) {
  await deps.editObject(imageId, annotationId, a => a.class_id = classId);
  for (const point of state.embed.items)
    if (point.annotation_id === annotationId && point.image_id === imageId) {
      point.class_id = classId;
      point.version = (point.version || 0) + 1;
    }
  renderEmbedSelection();
  renderEmbeddings();
}

