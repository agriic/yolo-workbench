const $ = id => document.getElementById(id);
const api = async (path, options = {}) => {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail || response.statusText);
  }
  return response.json();
};
const esc = value => String(value).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
const debounce = (fn, wait) => { let timer; return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); }; };
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

// Keep in sync with PALETTE in web.py so grid overlays and crops match the editor.
const PALETTE = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#06b6d4", "#84cc16", "#a855f7", "#64748b"];
const classColor = id => PALETTE[((id % PALETTE.length) + PALETTE.length) % PALETTE.length];
const HANDLE_CURSORS = ["nwse-resize", "ns-resize", "nesw-resize", "ew-resize", "nwse-resize", "ns-resize", "nesw-resize", "ew-resize"];

const state = {
  meta: null,
  gridImages: [],
  gridObjects: [],
  objectSelection: new Map(),  // "imageId|annotationId" -> object item
  lastObjectIndex: null,
  issues: [],
  issueFilter: null,
  detail: null,        // image detail currently in the editor
  img: null,           // loaded HTMLImageElement
  selected: null,      // annotation id
  hovered: null,
  drawing: null,       // {start,end} box or {points,cursor} polygon in progress
  drag: null,          // {mode:"move"|"resize"|"vertex", a, index, start, original}
  pan: null,
  spaceDown: false,
  view: { scale: 1, x: 0, y: 0 },
  canUndo: false,
  canRedo: false,
  pred: {
    status: "unavailable", error: null, names: {}, mapping: {}, pending: {},
    job: { state: "idle", done: 0, total: 0 }, items: [], minConf: 0, poll: null,
  },
  embed: {
    status: "idle", error: null, items: [], hovered: null, poll: null,
    classes: new Set(), selection: [], band: null, mode: "3d",
    rotation: { yaw: -0.65, pitch: -0.35 }, zoom: 1, rotate: null, projected: [],
    pan2d: { x: 0, y: 0 }, zoom2d: 1, panDrag: null,
  },
};

let dpr = 1, cssW = 0, cssH = 0;

function toast(message, kind = "info") {
  const el = $("toast");
  el.textContent = message;
  el.className = `${kind} show`;
  clearTimeout(el.timer);
  el.timer = setTimeout(() => el.classList.remove("show"), 3200);
}

function classOptions(selectedId) {
  return Object.entries(state.meta.names)
    .map(([id, name]) => `<option value="${id}" ${+id === selectedId ? "selected" : ""}>${esc(name)}</option>`)
    .join("");
}

async function init() {
  state.meta = await api("/api/v1/dataset");
  const yamlName = state.meta.yaml.split("/").slice(-2).join("/");
  $("dataset-name").textContent = `${yamlName} · ${state.meta.category}`;
  $("dataset-name").title = state.meta.yaml;
  for (const select of [$("image-class"), $("object-class"), $("draw-class"), $("bulk-class")])
    for (const [id, name] of Object.entries(state.meta.names))
      select.insertAdjacentHTML("beforeend", `<option value="${esc(id)}">${esc(id)} · ${esc(name)}</option>`);
  for (const select of [$("image-split"), $("object-split"), $("embed-split")])
    for (const split of Object.keys(state.meta.split_counts))
      select.insertAdjacentHTML("beforeend", `<option>${esc(split)}</option>`);
  $("embed-classes").innerHTML = Object.entries(state.meta.names).map(([id, name]) =>
    `<button class="class-chip" data-class="${esc(id)}" style="--c:${classColor(+id)}">${esc(name)}</button>`).join("");
  renderShortcuts();
  bind();
  updateHistoryButtons();
  watchIndexing();
  await Promise.all([loadImages(), loadObjects(), loadIssues(), refreshPredictor()]);
}

function watchIndexing() {
  const el = $("indexing-status");
  const idx = state.meta.indexing;
  if (!idx || idx.done >= idx.total) { el.hidden = true; return; }
  el.hidden = false;
  el.textContent = `Indexing ${idx.done}/${idx.total}…`;
  setTimeout(async () => {
    try { state.meta = { ...state.meta, ...(await api("/api/v1/dataset")) }; } catch { /* retry on next tick */ }
    watchIndexing();
  }, 1000);
}

function renderShortcuts() {
  const detection = state.meta.category === "detection";
  const rows = [
    [detection ? "Drag" : "Click", detection ? "Draw a box" : "Add polygon points"],
    ...(detection ? [] : [["⏎", "Close the polygon"], ["Alt+Click", "Remove a vertex"]]),
    ["Scroll", "Zoom at cursor"],
    ["Space+Drag", "Pan the image"],
    ["1–9", "Set class of selection"],
    ["⌫", "Delete selection"],
    ["Ctrl+Z / Ctrl+⇧+Z", "Undo / redo"],
    ["← →", "Previous / next image"],
    ["P", "Predict with the loaded model"],
    ["F", "Fit image"],
    ["Esc", "Cancel drawing / close"],
  ];
  $("shortcut-list").innerHTML = rows.map(([key, hint]) => `<dt><kbd>${key}</kbd></dt><dd>${hint}</dd>`).join("");
}

function bind() {
  document.querySelectorAll(".tab").forEach(button => button.onclick = () => {
    document.querySelectorAll(".tab,.view").forEach(el => el.classList.remove("active"));
    button.classList.add("active");
    $(button.dataset.tab).classList.add("active");
    if (button.dataset.tab === "embeddings") refreshEmbeddings();
  });
  [$("image-split"), $("image-class"), $("show-overlays"), $("filter-predictions")].forEach(el => el.onchange = loadImages);
  $("image-search").oninput = debounce(loadImages, 250);
  [$("object-class"), $("object-split")].forEach(el => el.onchange = () => { clearObjectSelection(); loadObjects(); });
  $("bulk-relabel").onclick = () => runObjectBulk("relabel", +$("bulk-class").value);
  $("bulk-delete").onclick = () => {
    if (confirm(`Delete ${state.objectSelection.size} annotation${state.objectSelection.size === 1 ? "" : "s"}?`)) runObjectBulk("delete");
  };
  $("bulk-clear").onclick = () => { clearObjectSelection(); loadObjects(); };
  $("crop-padding").oninput = () => {
    $("padding-value").textContent = `${Math.round($("crop-padding").value * 100)}%`;
    reloadObjectsDebounced();
  };
  $("refresh-issues").onclick = loadIssues;

  // model-assisted labeling
  $("load-model").onclick = loadModel;
  $("model-path").addEventListener("keydown", e => { if (e.key === "Enter") loadModel(); });
  $("run-predict").onclick = runPredict;
  $("mapping-list").addEventListener("change", async e => {
    const select = e.target.closest("[data-model-class]");
    if (!select) return;
    try {
      applyPredictorState(await api("/api/v1/predictor/mapping", {
        method: "PUT",
        body: JSON.stringify({ mapping: { [select.dataset.modelClass]: select.value === "" ? null : +select.value } }),
      }));
      if (state.detail) await loadEditorPredictions(state.detail.id);
    } catch (error) { toast(error.message, "error"); }
  });
  $("pred-conf").oninput = () => {
    state.pred.minConf = +$("pred-conf").value;
    $("pred-conf-value").textContent = `${Math.round(state.pred.minConf * 100)}%`;
    renderPredList();
    render();
  };
  $("accept-all-preds").onclick = () => acceptPredictions(null, state.pred.minConf);
  $("reject-all-preds").onclick = () => rejectPredictions(null);
  $("prediction-list").addEventListener("click", e => {
    const row = e.target.closest("[data-prediction]");
    if (!row) return;
    if (e.target.closest(".row-delete")) return rejectPredictions([row.dataset.prediction]);
    if (e.target.closest(".row-accept")) return acceptPredictions([row.dataset.prediction]);
  });

  $("theme-toggle").onclick = () => {
    const root = document.documentElement;
    const dark = root.dataset.theme !== "dark";
    if (dark) root.dataset.theme = "dark"; else delete root.dataset.theme;
    localStorage.setItem("theme", dark ? "dark" : "light");
    if ($("embeddings").classList.contains("active")) renderEmbeddings();
  };

  $("compute-embeddings").onclick = computeEmbeddings;
  $("embed-mode-2d").onclick = () => setEmbedMode("2d");
  $("embed-mode-3d").onclick = () => setEmbedMode("3d");
  $("embed-reset-view").onclick = resetEmbedView;
  $("embed-split").onchange = () => { state.embed.selection = []; renderEmbedSelection(); renderEmbeddings(); };
  $("embed-classes").addEventListener("click", e => {
    const chip = e.target.closest("[data-class]");
    if (!chip) return;
    const id = +chip.dataset.class;
    state.embed.classes.has(id) ? state.embed.classes.delete(id) : state.embed.classes.add(id);
    chip.classList.toggle("active", state.embed.classes.has(id));
    renderEmbeddings();
  });
  const embedCanvas = $("embed-canvas");
  embedCanvas.onpointerdown = embedPointerDown;
  embedCanvas.onpointermove = embedPointerMove;
  embedCanvas.onpointerup = embedPointerUp;
  embedCanvas.onpointercancel = embedPointerCancel;
  embedCanvas.onwheel = embedWheel;
  embedCanvas.onpointerleave = () => { state.embed.hovered = null; $("embed-tooltip").hidden = true; renderEmbeddings(); };
  $("embed-clear").onclick = () => { state.embed.selection = []; renderEmbedSelection(); renderEmbeddings(); };
  $("embed-selection").addEventListener("click", e => {
    const open = e.target.closest("[data-open-object]");
    if (open) openEditor(open.dataset.owner, open.dataset.openObject);
  });
  $("embed-selection").addEventListener("change", e => {
    const select = e.target.closest("[data-object-class]");
    if (select) reclassifyEmbedObject(select.dataset.owner, select.dataset.objectClass, +select.value);
  });
  new ResizeObserver(() => { if ($("embeddings").classList.contains("active")) renderEmbeddings(); }).observe($("embed-wrap"));

  $("image-grid").addEventListener("click", e => {
    const card = e.target.closest("[data-image]");
    if (card) openEditor(card.dataset.image);
  });
  $("image-grid").addEventListener("keydown", e => {
    const card = e.target.closest("[data-image]");
    if (card && e.key === "Enter") openEditor(card.dataset.image);
  });
  $("object-grid").addEventListener("click", e => {
    const check = e.target.closest(".card-check");
    if (check) return handleObjectCheck(check, e.shiftKey);
    const open = e.target.closest("[data-open-object]");
    if (open) return openEditor(open.dataset.owner, open.dataset.openObject);
    const del = e.target.closest("[data-delete-object]");
    if (del && confirm("Delete this annotation?")) editObject(del.dataset.owner, del.dataset.deleteObject, null);
  });
  $("object-grid").addEventListener("change", e => {
    const select = e.target.closest("[data-object-class]");
    if (select) editObject(select.dataset.owner, select.dataset.objectClass, a => a.class_id = +select.value);
  });
  $("issue-summary").addEventListener("click", e => {
    const fixAll = e.target.closest("[data-fix-kind]");
    if (fixAll) return fixAllIssues(fixAll.dataset.fixKind, +fixAll.dataset.count);
    const chip = e.target.closest("[data-kind]");
    if (!chip) return;
    state.issueFilter = state.issueFilter === chip.dataset.kind ? null : chip.dataset.kind;
    renderIssues();
  });
  $("issues").addEventListener("click", async e => {
    const open = e.target.closest("[data-issue-open]");
    if (open) return openEditor(open.dataset.issueOpen, open.dataset.object || null);
    const fix = e.target.closest("[data-fix]");
    if (!fix) return;
    try {
      await api(`/api/v1/issues/${fix.dataset.fix}/fix`, { method: "POST" });
      state.canUndo = true; state.canRedo = false; updateHistoryButtons();
      toast("Issue fixed");
      await Promise.all([loadImages(), loadObjects(), loadIssues()]);
    } catch (error) { toast(error.message, "error"); }
  });

  // editor chrome
  $("close-editor").onclick = () => $("editor").close();
  $("prev-image").onclick = () => navigate(-1);
  $("next-image").onclick = () => navigate(1);
  $("undo").onclick = () => applyHistory("undo");
  $("redo").onclick = () => applyHistory("redo");
  $("delete").onclick = deleteSelected;
  $("predict-image").onclick = predictImage;
  $("fit").onclick = fit;
  $("zoom-in").onclick = () => zoomAt(1.25, cssW / 2, cssH / 2);
  $("zoom-out").onclick = () => zoomAt(0.8, cssW / 2, cssH / 2);
  $("zoom-level").onclick = () => zoomAt(1 / state.view.scale, cssW / 2, cssH / 2);

  const canvas = $("canvas");
  canvas.onpointerdown = pointerDown;
  canvas.onpointermove = pointerMove;
  canvas.onpointerup = pointerUp;
  canvas.onpointercancel = pointerUp;
  canvas.oncontextmenu = e => e.preventDefault();
  $("canvas-wrap").addEventListener("wheel", e => {
    if (!state.img) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    zoomAt(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX - rect.left, e.clientY - rect.top);
  }, { passive: false });
  new ResizeObserver(resizeCanvas).observe($("canvas-wrap"));

  $("editor").addEventListener("cancel", e => {
    if (state.drawing) { e.preventDefault(); state.drawing = null; render(); }
  });
  $("editor").addEventListener("close", () => {
    state.detail = null; state.img = null; state.drawing = null; state.drag = null; state.selected = null;
    state.pred.items = [];
    renderPredList();
  });

  $("annotation-list").addEventListener("click", e => {
    const row = e.target.closest("[data-annotation]");
    if (!row) return;
    if (e.target.closest(".row-delete")) return removeAnnotation(row.dataset.annotation);
    if (e.target.closest("select")) return;
    state.selected = row.dataset.annotation;
    renderList(); render();
  });
  $("annotation-list").addEventListener("dblclick", e => {
    const row = e.target.closest("[data-annotation]");
    const annotation = row && state.detail?.annotations.find(a => a.id === row.dataset.annotation);
    if (annotation) zoomTo(annotation);
  });
  $("annotation-list").addEventListener("change", e => {
    const row = e.target.closest("[data-annotation]");
    if (!row || !e.target.classList.contains("row-class")) return;
    const annotation = state.detail.annotations.find(a => a.id === row.dataset.annotation);
    if (annotation) { annotation.class_id = +e.target.value; save(); }
  });

  window.addEventListener("keydown", keyDown);
  window.addEventListener("keyup", e => {
    if (e.key === " ") { state.spaceDown = false; if (state.img) $("canvas").style.cursor = "crosshair"; }
  });
}

function keyDown(e) {
  if (/^(input|select|textarea)$/i.test(e.target.tagName)) return;
  const key = e.key;
  // undo/redo works everywhere so bulk fixes outside the editor can be reverted
  if ((e.ctrlKey || e.metaKey) && key.toLowerCase() === "z") {
    e.preventDefault();
    return applyHistory(e.shiftKey ? "redo" : "undo");
  }
  if ((e.ctrlKey || e.metaKey) && key.toLowerCase() === "y") { e.preventDefault(); return applyHistory("redo"); }
  if (!$("editor").open) return;
  if (key === " ") { e.preventDefault(); state.spaceDown = true; $("canvas").style.cursor = "grab"; return; }
  if (key === "Delete" || key === "Backspace") return deleteSelected();
  if (key === "ArrowLeft") return navigate(-1);
  if (key === "ArrowRight") return navigate(1);
  if (key.toLowerCase() === "f") return fit();
  if (key.toLowerCase() === "p") return predictImage();
  if (key === "+" || key === "=") return zoomAt(1.25, cssW / 2, cssH / 2);
  if (key === "-") return zoomAt(0.8, cssW / 2, cssH / 2);
  if (key === "Enter" && state.drawing?.points?.length >= 3) return finishPolygon();
  if (key >= "1" && key <= "9") {
    const classId = Object.keys(state.meta.names)[+key - 1];
    if (classId === undefined) return;
    if (state.selected && state.detail) {
      const annotation = state.detail.annotations.find(a => a.id === state.selected);
      if (annotation && annotation.class_id !== +classId) { annotation.class_id = +classId; save(); }
    } else {
      $("draw-class").value = classId;
    }
  }
}

/* ---------- annotation grid ---------- */

async function loadImages() {
  const params = new URLSearchParams({ split: $("image-split").value, search: $("image-search").value, limit: "500" });
  if ($("image-class").value !== "") params.set("class_id", $("image-class").value);
  const data = await api(`/api/v1/images?${params}`);
  if ($("filter-predictions").checked) {
    data.items = data.items.filter(item => state.pred.pending[item.id]);
    data.total = data.items.length;
  }
  state.gridImages = data.items;
  $("image-total").textContent = `${data.items.length < data.total ? `${data.items.length} of ` : ""}${data.total} images`;
  const overlay = $("show-overlays").checked ? "&annotated=1" : "";
  $("image-grid").innerHTML = data.items.map(item => {
    const chips = item.classes.slice(0, 3).map(id =>
      `<span class="chip" style="--c:${classColor(id)}">${esc(state.meta.names[id] ?? `class ${id}`)}</span>`).join("");
    const more = item.classes.length > 3 ? `<span class="chip">+${item.classes.length - 3}</span>` : "";
    return `<article class="card" data-image="${item.id}" tabindex="0">
      <div class="thumb"><img loading="lazy" src="/api/v1/images/${item.id}/thumbnail?size=320${overlay}" alt="">
        ${item.issue_count ? `<span class="issue-flag">${item.issue_count} ⚠</span>` : ""}
        ${state.pred.pending[item.id] ? `<span class="pred-flag" title="Pending predictions">${state.pred.pending[item.id]} ✨</span>` : ""}</div>
      <div class="card-body">
        <div class="name" title="${esc(item.name)}">${esc(item.name)}</div>
        <div class="meta"><span class="split-badge">${esc(item.split)}</span><span>${item.annotation_count} object${item.annotation_count === 1 ? "" : "s"}</span></div>
        <div class="chips">${chips}${more}</div>
      </div>
    </article>`;
  }).join("") || `<p class="empty">No images match the current filters.</p>`;
  updateNavButtons();
}

/* ---------- editor ---------- */

async function openEditor(id, focus = null) {
  state.detail = await api(`/api/v1/images/${id}`);
  state.pred.items = [];
  loadEditorPredictions(id);
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
  img.onerror = () => toast("Failed to load image file", "error");
  img.src = `/api/v1/images/${id}/file?v=${Date.now()}`;
}

function navigate(step) {
  if (!state.detail) return;
  const ids = state.gridImages.map(item => item.id);
  const index = ids.indexOf(state.detail.id);
  if (index === -1 || !ids[index + step]) return;
  openEditor(ids[index + step]);
}

function updateNavButtons() {
  if (!$("editor").open || !state.detail) return;
  const ids = state.gridImages.map(item => item.id);
  const index = ids.indexOf(state.detail.id);
  $("prev-image").disabled = index <= 0;
  $("next-image").disabled = index === -1 || index >= ids.length - 1;
}

function updateHistoryButtons() {
  $("undo").disabled = !state.canUndo;
  $("redo").disabled = !state.canRedo;
}

function resizeCanvas() {
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

const toScreen = (nx, ny) => [state.view.x + nx * state.img.width * state.view.scale, state.view.y + ny * state.img.height * state.view.scale];

function eventPoint(e) {
  const rect = $("canvas").getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  return {
    screen: [sx, sy],
    norm: [(sx - state.view.x) / (state.img.width * state.view.scale), (sy - state.view.y) / (state.img.height * state.view.scale)],
  };
}

function fit() {
  if (!state.img) return;
  state.view.scale = clamp(Math.min((cssW - 48) / state.img.width, (cssH - 48) / state.img.height), 0.02, 40);
  state.view.x = (cssW - state.img.width * state.view.scale) / 2;
  state.view.y = (cssH - state.img.height * state.view.scale) / 2;
  render();
}

function zoomAt(factor, sx, sy) {
  if (!state.img) return;
  const next = clamp(state.view.scale * factor, 0.02, 40);
  state.view.x = sx - (sx - state.view.x) * (next / state.view.scale);
  state.view.y = sy - (sy - state.view.y) * (next / state.view.scale);
  state.view.scale = next;
  render();
}

function zoomTo(annotation) {
  if (!state.img) return;
  const [l, t, r, b] = bounds(annotation);
  const width = Math.max((r - l) * state.img.width, 8), height = Math.max((b - t) * state.img.height, 8);
  state.view.scale = clamp(Math.min(cssW / width, cssH / height) * 0.5, 0.02, 12);
  state.view.x = cssW / 2 - (l + r) / 2 * state.img.width * state.view.scale;
  state.view.y = cssH / 2 - (t + b) / 2 * state.img.height * state.view.scale;
  render();
}

function bounds(annotation) {
  if (state.meta.category === "detection") {
    const [cx, cy, w, h] = annotation.points;
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
  }
  const xs = annotation.points.filter((_, i) => i % 2 === 0), ys = annotation.points.filter((_, i) => i % 2 === 1);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function boxHandles(annotation) {
  const [l, t, r, b] = bounds(annotation), mx = (l + r) / 2, my = (t + b) / 2;
  return [[l, t], [mx, t], [r, t], [r, my], [r, b], [mx, b], [l, b], [l, my]];
}

const selectedAnnotation = () => state.detail?.annotations.find(a => a.id === state.selected) ?? null;

/* ---------- rendering ---------- */

function render() {
  const canvas = $("canvas"), ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  if (!state.img || !state.detail) { updateStatus(); return; }
  ctx.imageSmoothingEnabled = state.view.scale < 3;
  ctx.drawImage(state.img, state.view.x, state.view.y, state.img.width * state.view.scale, state.img.height * state.view.scale);
  for (const annotation of state.detail.annotations)
    if (annotation.id !== state.selected) drawShape(ctx, annotation, false);
  const selected = selectedAnnotation();
  if (selected) drawShape(ctx, selected, true);
  drawPredictions(ctx);
  drawPreview(ctx);
  updateStatus();
}

function shapePath(annotation) {
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

function drawShape(ctx, annotation, selected) {
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

function drawLabel(ctx, annotation, color) {
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

function drawPreview(ctx) {
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

const screenDistance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function updateStatus() {
  $("zoom-level").textContent = `${Math.round(state.view.scale * 100)}%`;
  $("status-image").textContent = state.detail
    ? `${state.detail.width}×${state.detail.height}px · ${state.detail.annotations.length} annotation${state.detail.annotations.length === 1 ? "" : "s"}`
    : "";
  let hint;
  if (state.drawing && state.meta.category === "segmentation")
    hint = "Click to add points · click the first point or press ⏎ to close · Esc cancels";
  else if (state.drawing) hint = "Release to create the box";
  else if (state.meta.category === "detection") hint = "Drag on empty space to draw a box · scroll to zoom · Space+drag to pan";
  else hint = "Click on empty space to start a polygon · scroll to zoom · Space+drag to pan";
  $("status-hint").textContent = hint;
}

/* ---------- hit testing & pointer interaction ---------- */

function hitHandle(screen) {
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

function hitShape([x, y]) {
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

function pointerDown(e) {
  if (!state.img) return;
  $("canvas").setPointerCapture(e.pointerId);
  if (e.button === 1 || e.button === 2 || state.spaceDown) {
    state.pan = { sx: e.clientX, sy: e.clientY, ox: state.view.x, oy: state.view.y };
    $("canvas").style.cursor = "grabbing";
    return;
  }
  if (e.button !== 0) return;
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
  const prediction = hitPrediction(norm);
  if (prediction) {
    if (prediction.class_id === null) return toast("Prediction class is unmapped — adjust the class mapping first", "error");
    return acceptPredictions([prediction.id]);
  }
  state.selected = null;
  state.drawing = state.meta.category === "detection" ? { start: clamped, end: clamped } : { points: [clamped] };
  renderList();
  render();
}

function pointerMove(e) {
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
  if (state.spaceDown) {
    cursor = "grab";
  } else {
    const handle = hitHandle(screen);
    if (handle) {
      cursor = handle.mode === "resize" ? HANDLE_CURSORS[handle.index] : "move";
    } else {
      const found = hitShape(norm);
      if (found) { cursor = "move"; hovered = found.id; }
      else if (hitPrediction(norm)) cursor = "pointer";
    }
  }
  $("canvas").style.cursor = cursor;
  if (hovered !== state.hovered) { state.hovered = hovered; render(); }
}

function applyDrag(norm) {
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

async function pointerUp() {
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

async function finishPolygon() {
  const points = state.drawing.points.flat();
  state.drawing = null;
  state.detail.annotations.push({ id: null, class_id: +$("draw-class").value, points });
  await save(state.detail.annotations.length - 1);
}

function removeVertex(annotation, index) {
  if (annotation.points.length <= 6) return toast("A polygon needs at least 3 points", "error");
  annotation.points.splice(index * 2, 2);
  save();
}

async function removeAnnotation(id) {
  state.detail.annotations = state.detail.annotations.filter(a => a.id !== id);
  if (state.selected === id) state.selected = null;
  await save();
}

async function deleteSelected() {
  if (state.selected && state.detail) await removeAnnotation(state.selected);
}

/* ---------- persistence ---------- */

async function save(selectIndex = null) {
  const image = state.detail;
  const index = selectIndex ?? image.annotations.findIndex(a => a.id === state.selected);
  try {
    state.detail = await api(`/api/v1/images/${image.id}/annotations`, {
      method: "PUT",
      body: JSON.stringify({ annotations: image.annotations }),
    });
    state.canUndo = true;
    state.canRedo = false;
    // the server re-derives annotation ids from line numbers, so re-select by position
    state.selected = index >= 0 ? state.detail.annotations[index]?.id ?? null : null;
  } catch (error) {
    toast(error.message, "error");
    state.detail = await api(`/api/v1/images/${image.id}`);
    state.selected = null;
  }
  updateHistoryButtons();
  renderList();
  render();
  loadImages();
  loadObjects();
  loadIssues();
}

async function applyHistory(direction) {
  if (direction === "undo" ? !state.canUndo : !state.canRedo) return;
  try {
    const result = await api(`/api/v1/history/${direction}`, { method: "POST" });
    state.canUndo = result.can_undo;
    state.canRedo = result.can_redo;
    updateHistoryButtons();
    if (state.detail && result.image_ids.includes(state.detail.id)) {
      state.detail = await api(`/api/v1/images/${state.detail.id}`);
      if (!state.detail.annotations.some(a => a.id === state.selected)) state.selected = null;
      renderList();
      render();
    }
    await Promise.all([loadImages(), loadObjects(), loadIssues()]);
  } catch (error) {
    toast(error.message, "error");
  }
}

/* ---------- annotation list (inspector) ---------- */

function renderList() {
  if (!state.detail) return;
  const annotations = state.detail.annotations;
  $("annotation-count").textContent = annotations.length || "";
  $("annotation-list").innerHTML = annotations.map(a => `
    <div class="annotation-row ${a.id === state.selected ? "selected" : ""}" data-annotation="${esc(a.id)}" title="Double-click to zoom">
      <span class="swatch" style="background:${classColor(a.class_id)}"></span>
      <select class="row-class">${classOptions(a.class_id)}</select>
      <span class="row-meta">${sizeLabel(a)}</span>
      <button class="row-delete" title="Delete annotation">×</button>
    </div>`).join("")
    || `<p class="empty-note">No annotations yet. ${state.meta.category === "detection" ? "Drag on the image to draw a box." : "Click on the image to start a polygon."}</p>`;
}

function sizeLabel(annotation) {
  if (state.meta.category === "detection")
    return `${Math.round(annotation.points[2] * state.detail.width)}×${Math.round(annotation.points[3] * state.detail.height)}`;
  return `${annotation.points.length / 2} pts`;
}

/* ---------- model-assisted labeling ---------- */

const UNMAPPED_COLOR = "#8b8f98";

async function refreshPredictor() {
  try {
    applyPredictorState(await api("/api/v1/predictor"));
  } catch (error) { toast(error.message, "error"); }
}

function applyPredictorState(data) {
  const pred = state.pred;
  const wasRunning = pred.job.state === "running";
  pred.status = data.status;
  pred.error = data.error;
  pred.names = data.model.names;
  pred.mapping = data.mapping;
  pred.pending = data.pending;
  pred.job = data.job;
  const ready = pred.status === "ready";
  const running = pred.job.state === "running";
  $("load-model").disabled = pred.status === "unavailable" || pred.status === "loading" || running;
  $("run-predict").hidden = !ready;
  $("run-predict").disabled = running;
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
      ? `Predicting ${pred.job.done}/${pred.job.total}…`
      : `${Object.keys(pred.names).length} model classes${unmapped ? ` · ${unmapped} unmapped` : ""}`,
  }[pred.status] || "";
  if (pred.job.state === "error") toast(pred.job.error, "error");
  renderMapping();
  clearTimeout(pred.poll);
  if (running) {
    pred.poll = setTimeout(refreshPredictor, 800);
  } else if (wasRunning) {
    const count = Object.values(pred.pending).reduce((sum, n) => sum + n, 0);
    if (pred.job.state === "done") toast(`Predicted ${pred.job.total} image${pred.job.total === 1 ? "" : "s"} · ${count} pending prediction${count === 1 ? "" : "s"}`);
    loadImages();
    if (state.detail) loadEditorPredictions(state.detail.id);
  }
}

function renderMapping() {
  const pred = state.pred;
  const options = value => `<option value="">— unmapped</option>` + Object.entries(state.meta.names)
    .map(([id, name]) => `<option value="${id}" ${value !== null && +id === value ? "selected" : ""}>${esc(id)} · ${esc(name)}</option>`).join("");
  $("mapping-list").innerHTML = Object.entries(pred.names).map(([id, name]) => `
    <label class="mapping-row"><span title="${esc(name)}">${esc(id)} · ${esc(name)}</span>
      <select data-model-class="${esc(id)}">${options(pred.mapping[id] ?? null)}</select>
    </label>`).join("");
}

async function loadModel() {
  const path = $("model-path").value.trim();
  if (!path) return toast("Enter a model path first", "error");
  $("load-model").disabled = true;
  $("predictor-status").textContent = "Loading model…";
  try {
    applyPredictorState(await api("/api/v1/predictor/load", { method: "POST", body: JSON.stringify({ path }) }));
    toast("Model loaded");
  } catch (error) {
    $("load-model").disabled = false;
    $("predictor-status").textContent = error.message;
    toast(error.message, "error");
  }
}

async function runPredict() {
  try {
    applyPredictorState(await api("/api/v1/predictor/run", {
      method: "POST",
      body: JSON.stringify({ split: $("image-split").value, only_unlabeled: $("predict-unlabeled").checked }),
    }));
  } catch (error) { toast(error.message, "error"); }
}

async function predictImage() {
  if (!state.detail) return;
  if (state.pred.status !== "ready") return toast("Load a model first", "error");
  const id = state.detail.id;
  const button = $("predict-image");
  button.disabled = true;
  try {
    const data = await api(`/api/v1/images/${id}/predictions/compute`, { method: "POST" });
    if (state.detail?.id === id) {
      state.pred.items = data.items;
      renderPredList();
      render();
    }
    state.pred.pending[id] = data.items.length;
    toast(data.items.length ? `${data.items.length} prediction${data.items.length === 1 ? "" : "s"} — click a ghost or use the panel to accept` : "The model found nothing on this image");
    loadImages();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = state.pred.job.state === "running";
  }
}

async function loadEditorPredictions(id) {
  if (state.pred.status === "unavailable") return;
  try {
    const data = await api(`/api/v1/images/${id}/predictions`);
    if (!state.detail || state.detail.id !== id) return;
    state.pred.items = data.items;
  } catch { state.pred.items = []; }
  renderPredList();
  render();
}

const visiblePredictions = () => state.pred.items.filter(p => p.confidence >= state.pred.minConf);

function drawPredictions(ctx) {
  if (!state.detail || !state.img) return;
  for (const prediction of visiblePredictions()) {
    const color = prediction.class_id === null ? UNMAPPED_COLOR : classColor(prediction.class_id);
    const path = shapePath(prediction);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.setLineDash([7, 5]);
    ctx.stroke(path);
    ctx.restore();
    drawPredictionBadge(ctx, prediction, color);
  }
}

function drawPredictionBadge(ctx, prediction, color) {
  const [l, t, r] = bounds(prediction);
  if ((r - l) * state.img.width * state.view.scale < 26) return;
  const [sx, sy] = toScreen(l, t);
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

function hitPrediction([x, y]) {
  for (const prediction of [...visiblePredictions()].reverse()) {
    if (state.meta.category === "detection") {
      const [l, t, r, b] = bounds(prediction);
      if (x >= l && x <= r && y >= t && y <= b) return prediction;
    } else if (pointInPolygon(x, y, prediction.points)) {
      return prediction;
    }
  }
  return null;
}

function renderPredList() {
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

async function acceptPredictions(predictionIds, minConfidence = null) {
  if (!state.detail) return;
  const body = { prediction_ids: predictionIds };
  if (predictionIds === null && minConfidence !== null) body.min_confidence = minConfidence;
  try {
    const result = await api(`/api/v1/images/${state.detail.id}/predictions/accept`, { method: "POST", body: JSON.stringify(body) });
    state.detail = result.detail;
    state.pred.items = result.predictions;
    state.pred.pending[state.detail.id] = result.predictions.length;
    state.canUndo = true;
    state.canRedo = false;
    updateHistoryButtons();
    toast(`Accepted ${result.accepted} prediction${result.accepted === 1 ? "" : "s"} — Ctrl+Z undoes`);
    renderList(); renderPredList(); render();
    await Promise.all([loadImages(), loadObjects(), loadIssues()]);
  } catch (error) { toast(error.message, "error"); }
}

async function rejectPredictions(predictionIds) {
  if (!state.detail) return;
  try {
    const result = await api(`/api/v1/images/${state.detail.id}/predictions/reject`, { method: "POST", body: JSON.stringify({ prediction_ids: predictionIds }) });
    state.pred.items = result.predictions;
    state.pred.pending[state.detail.id] = result.predictions.length;
    renderPredList(); render();
    loadImages();
  } catch (error) { toast(error.message, "error"); }
}

/* ---------- exploration ---------- */

const reloadObjectsDebounced = debounce(loadObjects, 300);

async function loadObjects() {
  if (!state.meta) return;
  const classId = $("object-class").value || Object.keys(state.meta.names)[0];
  const params = new URLSearchParams({ class_id: classId, split: $("object-split").value, limit: "500" });
  const data = await api(`/api/v1/objects?${params}`);
  state.gridObjects = data.items;
  const valid = new Set(data.items.map(objectKey));
  for (const key of [...state.objectSelection.keys()]) if (!valid.has(key)) state.objectSelection.delete(key);
  renderObjectBulkBar();
  $("object-total").textContent = `${data.items.length < data.total ? `${data.items.length} of ` : ""}${data.total} objects`;
  const padding = $("crop-padding").value;
  $("object-grid").innerHTML = data.items.map((item, index) => `
    <article class="card object-card">
      <div class="thumb">
        <img loading="lazy" src="/api/v1/objects/${item.image_id}/${encodeURIComponent(item.id)}/crop?padding=${padding}" alt="">
        <input type="checkbox" class="card-check" data-index="${index}" title="Select (Shift-click for range)" ${state.objectSelection.has(objectKey(item)) ? "checked" : ""}>
        <span class="class-tag" style="background:${classColor(item.class_id)}">${esc(state.meta.names[item.class_id] ?? `class ${item.class_id}`)}</span>
      </div>
      <div class="card-body">
        <div class="name" title="${esc(item.image_name)}">${esc(item.image_name)}</div>
        <div class="meta"><span class="split-badge">${esc(item.split)}</span></div>
        <div class="object-actions">
          <select data-object-class="${esc(item.id)}" data-owner="${item.image_id}" title="Reassign class">${classOptions(item.class_id)}</select>
          <button data-open-object="${esc(item.id)}" data-owner="${item.image_id}" title="Open in editor">Open</button>
          <button data-delete-object="${esc(item.id)}" data-owner="${item.image_id}" title="Delete annotation">×</button>
        </div>
      </div>
    </article>`).join("") || `<p class="empty">No objects of this class in the selected split.</p>`;
}

const objectKey = item => `${item.image_id}|${item.id}`;

function handleObjectCheck(check, shiftKey) {
  const index = +check.dataset.index;
  const on = check.checked;
  const indices = shiftKey && state.lastObjectIndex != null
    ? Array.from({ length: Math.abs(index - state.lastObjectIndex) + 1 }, (_, i) => Math.min(index, state.lastObjectIndex) + i)
    : [index];
  for (const i of indices) {
    const item = state.gridObjects[i];
    if (!item) continue;
    if (on) state.objectSelection.set(objectKey(item), item);
    else state.objectSelection.delete(objectKey(item));
    const box = $("object-grid").querySelector(`.card-check[data-index="${i}"]`);
    if (box) box.checked = on;
  }
  state.lastObjectIndex = index;
  renderObjectBulkBar();
}

function renderObjectBulkBar() {
  const count = state.objectSelection.size;
  $("object-bulk").hidden = !count;
  $("object-selected-count").textContent = `${count} selected`;
}

function clearObjectSelection() {
  state.objectSelection.clear();
  state.lastObjectIndex = null;
  renderObjectBulkBar();
}

async function runObjectBulk(action, classId = null) {
  const operations = [...state.objectSelection.values()].map(item =>
    ({ image_id: item.image_id, annotation_id: item.id, action, class_id: classId }));
  if (!operations.length) return;
  try {
    const result = await api("/api/v1/objects/bulk", { method: "POST", body: JSON.stringify({ operations }) });
    state.canUndo = true;
    state.canRedo = false;
    updateHistoryButtons();
    const skipped = result.skipped.length ? ` · ${result.skipped.length} skipped` : "";
    toast(`${action === "delete" ? "Deleted" : "Relabeled"} ${result.applied} annotation${result.applied === 1 ? "" : "s"} in ${result.files} file${result.files === 1 ? "" : "s"}${skipped} — Ctrl+Z undoes`);
    clearObjectSelection();
    await Promise.all([loadImages(), loadObjects(), loadIssues()]);
  } catch (error) {
    toast(error.message, "error");
  }
}

async function editObject(imageId, id, change) {
  try {
    const detail = await api(`/api/v1/images/${imageId}`);
    if (change) change(detail.annotations.find(a => a.id === id));
    else detail.annotations = detail.annotations.filter(a => a.id !== id);
    await api(`/api/v1/images/${imageId}/annotations`, { method: "PUT", body: JSON.stringify({ annotations: detail.annotations }) });
    state.canUndo = true;
    state.canRedo = false;
    updateHistoryButtons();
    await Promise.all([loadImages(), loadObjects(), loadIssues()]);
  } catch (error) {
    toast(error.message, "error");
  }
}

/* ---------- embeddings (Voxel51 gt_viz) ---------- */

async function refreshEmbeddings() {
  try {
    applyEmbedState(await api("/api/v1/embeddings"));
  } catch (error) { toast(error.message, "error"); }
}

async function computeEmbeddings() {
  try {
    applyEmbedState(await api("/api/v1/embeddings/compute", { method: "POST" }));
  } catch (error) { toast(error.message, "error"); }
}

function applyEmbedState(data) {
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
  if (embed.status === "error") toast(embed.error, "error");
  renderEmbeddings();
}

function embedFiltered() {
  const classes = state.embed.classes, split = $("embed-split").value;
  return state.embed.items.filter(p =>
    (classes.size === 0 || classes.has(p.class_id)) && (split === "all" || p.split === split));
}

function embedLayout() {
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

function setEmbedMode(mode) {
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

function resetEmbedView() {
  if (state.embed.mode === "3d") {
    state.embed.rotation = { yaw: -0.65, pitch: -0.35 };
    state.embed.zoom = 1;
  } else {
    state.embed.pan2d = { x: 0, y: 0 };
    state.embed.zoom2d = 1;
  }
  renderEmbeddings();
}

function projectEmbedPoint(point, w, h) {
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

function embedScreenPoints() {
  const wrap = $("embed-wrap"), w = wrap.clientWidth, h = wrap.clientHeight;
  if (state.embed.mode === "2d") {
    const { sx, sy } = embedLayout();
    return embedFiltered().map(point => ({ point, x: sx(point.x), y: sy(point.y), depth: 0, perspective: 1 }));
  }
  return embedFiltered().map(point => ({ point, ...projectEmbedPoint(point, w, h) }));
}

function renderEmbedAxes(ctx, w, h) {
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

function renderEmbeddings() {
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

function embedMouse(e) {
  const rect = $("embed-canvas").getBoundingClientRect();
  return [e.clientX - rect.left, e.clientY - rect.top];
}

function embedNearest(mx, my) {
  let best = null, bestDistance = 10;
  const points = embedScreenPoints();
  if (state.embed.mode === "3d") points.sort((a, b) => b.depth - a.depth);
  for (const screen of points) {
    const distance = Math.hypot(screen.x - mx, screen.y - my);
    if (distance < bestDistance) { best = screen.point; bestDistance = distance; }
  }
  return best;
}

function embedPointerDown(e) {
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

function embedPointerMove(e) {
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

function embedPointerUp(e) {
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

function selectEmbedPoint(point, additive) {
  const embed = state.embed;
  const previous = additive ? embed.selection : [];
  if (point) {
    const already = previous.some(p => p.annotation_id === point.annotation_id);
    if (additive) embed.selection = already ? previous.filter(p => p.annotation_id !== point.annotation_id) : [...previous, point];
    else embed.selection = [point];
  } else if (!additive) embed.selection = [];
  renderEmbedSelection();
}

function embedPointerCancel() {
  state.embed.band = null;
  state.embed.rotate = null;
  state.embed.panDrag = null;
  $("embed-canvas").style.cursor = "crosshair";
  renderEmbeddings();
}

function embedWheel(e) {
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

function renderEmbedSelection() {
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
          <select data-object-class="${esc(point.annotation_id)}" data-owner="${point.image_id}" title="Reassign class">${classOptions(point.class_id)}</select>
          <button data-open-object="${esc(point.annotation_id)}" data-owner="${point.image_id}" title="Open in editor">Open</button>
        </div>
      </div>
    </div>`).join("") || `<p class="empty-note">${emptyHint}</p>`;
}

async function reclassifyEmbedObject(imageId, annotationId, classId) {
  await editObject(imageId, annotationId, a => a.class_id = classId);
  for (const point of state.embed.items)
    if (point.annotation_id === annotationId && point.image_id === imageId) {
      point.class_id = classId;
      point.version = (point.version || 0) + 1;
    }
  renderEmbedSelection();
  renderEmbeddings();
}

/* ---------- validation ---------- */

async function loadIssues() {
  if (!state.meta) return;
  const data = await api("/api/v1/issues");
  state.issues = data.items;
  $("issue-count").textContent = data.items.length || "";
  renderIssues();
}

function renderIssues() {
  const items = state.issues;
  const counts = {}, fixable = {};
  for (const issue of items) {
    counts[issue.kind] = (counts[issue.kind] || 0) + 1;
    if (issue.fixable) fixable[issue.kind] = (fixable[issue.kind] || 0) + 1;
  }
  if (state.issueFilter && !counts[state.issueFilter]) state.issueFilter = null;
  $("issue-summary").innerHTML = Object.entries(counts).map(([kind, count]) => `
    <span class="issue-chip-group">
      <button class="issue-chip ${state.issueFilter === kind ? "active" : ""}" data-kind="${esc(kind)}">${esc(kind.replaceAll("_", " "))} <strong>${count}</strong></button>
      ${fixable[kind] ? `<button class="fix-all" data-fix-kind="${esc(kind)}" data-count="${fixable[kind]}" title="Fix all ${fixable[kind]} automatically">Fix all</button>` : ""}
    </span>`).join("");
  const visible = state.issueFilter ? items.filter(issue => issue.kind === state.issueFilter) : items;
  $("issues").innerHTML = visible.map(issue => `
    <div class="issue">
      <span class="severity-dot ${esc(issue.severity)}" title="${esc(issue.severity)}"></span>
      <span class="pill">${esc(issue.kind.replaceAll("_", " "))}</span>
      <div><strong>${esc(issue.image_name)}</strong><div class="message">${esc(issue.message)}</div></div>
      <span class="split-badge">${esc(issue.split || "—")}</span>
      <span class="issue-actions">
        ${issue.image_id ? `<button data-issue-open="${issue.image_id}" data-object="${esc(issue.annotation_id || "")}">Open</button>` : ""}
        ${issue.fixable ? `<button data-fix="${issue.id}">Fix</button>` : ""}
      </span>
    </div>`).join("") || `<div class="empty">No issues found. The dataset looks clean.</div>`;
}

async function fixAllIssues(kind, count) {
  if (!confirm(`Fix all ${count} ${kind.replaceAll("_", " ")} issue${count === 1 ? "" : "s"}? Ctrl+Z undoes the whole batch.`)) return;
  try {
    const result = await api("/api/v1/issues/fix-bulk", { method: "POST", body: JSON.stringify({ kind }) });
    state.canUndo = true;
    state.canRedo = false;
    updateHistoryButtons();
    const skipped = result.skipped.length ? ` · ${result.skipped.length} image${result.skipped.length === 1 ? "" : "s"} skipped (${result.skipped[0].reason})` : "";
    toast(`Fixed ${result.fixed} issue${result.fixed === 1 ? "" : "s"} in ${result.files} file${result.files === 1 ? "" : "s"}${skipped}`, result.skipped.length ? "error" : "info");
    await Promise.all([loadImages(), loadObjects(), loadIssues()]);
  } catch (error) {
    toast(error.message, "error");
  }
}

/* ---------- helpers ---------- */

function pointInPolygon(x, y, points) {
  let inside = false;
  for (let i = 0, j = points.length - 2; i < points.length; j = i, i += 2)
    if ((points[i + 1] > y) !== (points[j + 1] > y) && x < (points[j] - points[i]) * (y - points[i + 1]) / (points[j + 1] - points[i + 1]) + points[i])
      inside = !inside;
  return inside;
}

init().catch(error => toast(error.message, "error"));
