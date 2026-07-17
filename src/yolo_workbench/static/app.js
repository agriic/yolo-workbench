import { $, api, esc, debounce, classColor } from "./api.js";
import { state } from "./state.js";
import { VirtualGallery } from "./grid.js";
import {
  applyHistory, bounds, closeEditorSession, configureCanvas, deleteSelected,
  editorRequest, finishPolygon, fit, navigate, openEditor, pointerDown, pointerMove,
  pointerUp, removeAnnotation, render, renderList, resizeCanvas, save, saveQueue,
  savedRevisionByImage, shapePath, toScreen, updateHistoryButtons, zoomCentered, updateNavButtons,
  zoomTo,
} from "./canvas.js";
import {
  acceptPredictions, applyPredictorState, cancelPredict, configurePredictor,
  drawPredictions, hitPrediction, loadEditorPredictions, loadModel, predictImage,
  refreshPredictor, rejectPredictions, renderModelMenu, renderPredList,
  retryPredict, runPredict,
} from "./predictor.js";
import {
  computeEmbeddings, configureEmbeddings, embedPointerCancel, embedPointerDown, embedPointerMove,
  embedPointerUp, embedWheel, refreshEmbeddings, reclassifyEmbedObject,
  renderEmbeddings, renderEmbedSelection, resetEmbedView, setEmbedMode,
} from "./embeddings.js";

export let imageGallery = null;
let objectGallery = null;
let statisticsRequest = 0;

export function toast(message, kind = "info") {
  const el = $("toast");
  el.textContent = message;
  el.className = `${kind} show`;
  clearTimeout(el.timer);
  el.timer = setTimeout(() => el.classList.remove("show"), 3200);
}

export function classOptions(selectedId) {
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
  setupVirtualGalleries();
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
    if (button.dataset.tab === "annotation") imageGallery.refreshLayout();
    if (button.dataset.tab === "exploration") objectGallery.refreshLayout();
    if (button.dataset.tab === "embeddings") refreshEmbeddings();
    if (button.dataset.tab === "statistics") loadStatistics();
  });
  [$("image-split"), $("image-class"), $("filter-predictions")].forEach(el => el.onchange = () => loadImages({ reset: true }));
  $("show-overlays").onchange = () => loadImages();
  $("image-search").oninput = debounce(() => loadImages({ reset: true }), 250);
  [$("object-class"), $("object-split")].forEach(el => el.onchange = () => { clearObjectSelection(); loadObjects({ reset: true }); });
  $("bulk-relabel").onclick = () => runObjectBulk("relabel", +$("bulk-class").value);
  $("bulk-delete").onclick = () => {
    if (confirm(`Delete ${state.objectSelection.size} annotation${state.objectSelection.size === 1 ? "" : "s"}?`)) runObjectBulk("delete");
  };
  $("bulk-clear").onclick = () => { clearObjectSelection(); objectGallery.render(); };
  $("crop-padding").oninput = () => {
    $("padding-value").textContent = `${Math.round($("crop-padding").value * 100)}%`;
    reloadObjectsDebounced();
  };
  $("refresh-issues").onclick = loadIssues;
  $("refresh-statistics").onclick = loadStatistics;
  $("stats-outliers").addEventListener("click", e => {
    const open = e.target.closest("[data-stat-open]");
    if (open) openEditor(open.dataset.statOpen, open.dataset.object || null);
  });

  // model-assisted labeling
  $("load-model").onclick = loadModel;
  $("model-path").addEventListener("keydown", e => { if (e.key === "Enter") loadModel(); });
  $("model-browse").onclick = () => {
    const menu = $("model-menu");
    menu.hidden = !menu.hidden;
    if (!menu.hidden) renderModelMenu();
  };
  $("model-menu").addEventListener("click", e => {
    const option = e.target.closest("[data-path]");
    if (!option) return;
    $("model-path").value = option.dataset.path;
    $("model-menu").hidden = true;
    loadModel();
  });
  document.addEventListener("pointerdown", e => {
    if (!e.target.closest(".model-picker")) $("model-menu").hidden = true;
  });
  $("run-predict").onclick = runPredict;
  $("cancel-predict").onclick = cancelPredict;
  $("retry-predict").onclick = retryPredict;
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
    if (card) openEditor(card.dataset.image, null, +card.dataset.index);
  });
  $("image-grid").addEventListener("keydown", e => {
    const card = e.target.closest("[data-image]");
    if (card && e.key === "Enter") openEditor(card.dataset.image, null, +card.dataset.index);
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
      statisticsChanged();
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
  $("zoom-in").onclick = () => zoomCentered(1.25);
  $("zoom-out").onclick = () => zoomCentered(0.8);
  $("zoom-level").onclick = () => zoomCentered(1 / state.view.scale);

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
    closeEditorSession();
    state.detail = null; state.img = null; state.drawing = null; state.drag = null; state.selected = null;
    state.editorGridIndex = null;
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
  if (key === "+" || key === "=") return zoomCentered(1.25);
  if (key === "-") return zoomCentered(0.8);
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

function setupVirtualGalleries() {
  imageGallery = new VirtualGallery({
    gridId: "image-grid",
    scrollId: "image-scroll",
    railId: "image-scroll-labels",
    errorId: "image-load-error",
    noun: "Images",
    emptyMessage: "No images match the current filters.",
    fetchPage: fetchImagePage,
    renderCard: renderImageCard,
    onUpdate: total => {
      $("image-total").textContent = `${total.toLocaleString()} image${total === 1 ? "" : "s"}`;
      updateNavButtons();
    },
  });
  objectGallery = new VirtualGallery({
    gridId: "object-grid",
    scrollId: "object-scroll",
    railId: "object-scroll-labels",
    errorId: "object-load-error",
    noun: "Objects",
    emptyMessage: "No objects of this class in the selected split.",
    fetchPage: fetchObjectPage,
    renderCard: renderObjectCard,
    isPinned: item => state.objectSelection.has(objectKey(item)),
    onUpdate: total => $("object-total").textContent = `${total.toLocaleString()} object${total === 1 ? "" : "s"}`,
  });
}

function fetchImagePage(offset, limit, signal) {
  const params = new URLSearchParams({ split: $("image-split").value, search: $("image-search").value, limit: String(limit), offset: String(offset) });
  if ($("image-class").value !== "") params.set("class_id", $("image-class").value);
  if ($("filter-predictions").checked) params.set("has_predictions", "true");
  return api(`/api/v1/images?${params}`, { signal });
}

function fetchObjectPage(offset, limit, signal) {
  const classId = $("object-class").value || Object.keys(state.meta.names)[0];
  const params = new URLSearchParams({ class_id: classId, split: $("object-split").value, limit: String(limit), offset: String(offset) });
  return api(`/api/v1/objects?${params}`, { signal });
}

export async function loadImages(options = {}) {
  if (!imageGallery) return;
  await imageGallery.reload(options);
  if (state.detail && state.editorGridIndex != null) {
    const item = imageGallery.cache.get(state.editorGridIndex);
    if (item && item.id !== state.detail.id) state.editorGridIndex = null;
    updateNavButtons();
  }
}

function renderImageCard(item, index) {
  const overlay = $("show-overlays").checked ? "&annotated=1" : "";
  const chips = item.classes.slice(0, 3).map(id =>
    `<span class="chip" style="--c:${classColor(id)}">${esc(state.meta.names[id] ?? `class ${id}`)}</span>`).join("");
  const more = item.classes.length > 3 ? `<span class="chip">+${item.classes.length - 3}</span>` : "";
  return `<article class="card" data-image="${item.id}" data-index="${index}" tabindex="0">
      <div class="thumb"><img loading="lazy" src="/api/v1/images/${item.id}/thumbnail?size=320${overlay}" alt="">
        ${item.issue_count ? `<span class="issue-flag">${item.issue_count} ⚠</span>` : ""}
        ${state.pred.pending[item.id] ? `<span class="pred-flag" title="Pending predictions">${state.pred.pending[item.id]} ✨</span>` : ""}</div>
      <div class="card-body">
        <div class="name" title="${esc(item.name)}">${esc(item.name)}</div>
        <div class="meta"><span class="split-badge">${esc(item.split)}</span><span>${item.annotation_count} object${item.annotation_count === 1 ? "" : "s"}</span></div>
        <div class="chips">${chips}${more}</div>
      </div>
    </article>`;
}

/* ---------- exploration ---------- */

const reloadObjectsDebounced = debounce(loadObjects, 300);

export async function loadObjects(options = {}) {
  if (!state.meta || !objectGallery) return;
  await objectGallery.reload(options);
  renderObjectBulkBar();
}

function renderObjectCard(item, index) {
  const padding = $("crop-padding").value;
  return `<article class="card object-card">
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
    </article>`;
}

const objectKey = item => `${item.image_id}|${item.id}`;

async function handleObjectCheck(check, shiftKey) {
  if (state.objectSelectionBusy) {
    objectGallery.render();
    return;
  }
  const index = +check.dataset.index;
  const on = check.checked;
  const start = shiftKey && state.lastObjectIndex != null ? Math.min(index, state.lastObjectIndex) : index;
  const end = shiftKey && state.lastObjectIndex != null ? Math.max(index, state.lastObjectIndex) + 1 : index + 1;
  state.objectSelectionBusy = end - start > 1;
  renderObjectBulkBar();
  try {
    const items = await objectGallery.ensureRange(start, end);
    if (items.length !== end - start) throw new Error("Could not load the complete selection range");
    for (const item of items) {
      if (on) state.objectSelection.set(objectKey(item), item);
      else state.objectSelection.delete(objectKey(item));
    }
    state.lastObjectIndex = index;
  } catch (error) {
    if (error.name !== "AbortError") toast(error.message, "error");
  } finally {
    state.objectSelectionBusy = false;
    objectGallery.render();
    renderObjectBulkBar();
  }
}

function renderObjectBulkBar() {
  const count = state.objectSelection.size;
  $("object-bulk").hidden = !count && !state.objectSelectionBusy;
  $("object-selected-count").textContent = state.objectSelectionBusy ? "Selecting range…" : `${count} selected`;
  $("object-bulk").querySelectorAll("button,select").forEach(control => control.disabled = state.objectSelectionBusy);
}

function clearObjectSelection() {
  state.objectSelection.clear();
  state.lastObjectIndex = null;
  renderObjectBulkBar();
  if (objectGallery) objectGallery.render();
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
    statisticsChanged();
    const skipped = result.skipped.length ? ` · ${result.skipped.length} skipped` : "";
    toast(`${action === "delete" ? "Deleted" : "Relabeled"} ${result.applied} annotation${result.applied === 1 ? "" : "s"} in ${result.files} file${result.files === 1 ? "" : "s"}${skipped} — Ctrl+Z undoes`);
    clearObjectSelection();
    await Promise.all([loadImages(), loadObjects(), loadIssues()]);
  } catch (error) {
    toast(error.message, "error");
  }
}

export async function editObject(imageId, id, change) {
  try {
    const detail = await api(`/api/v1/images/${imageId}`);
    if (change) change(detail.annotations.find(a => a.id === id));
    else detail.annotations = detail.annotations.filter(a => a.id !== id);
    await api(`/api/v1/images/${imageId}/annotations`, {
      method: "PUT",
      body: JSON.stringify({ revision: detail.revision, annotations: detail.annotations }),
    });
    state.canUndo = true;
    state.canRedo = false;
    updateHistoryButtons();
    statisticsChanged();
    await Promise.all([loadImages(), loadObjects(), loadIssues()]);
  } catch (error) {
    toast(error.message, "error");
  }
}

/* ---------- statistics ---------- */

async function loadStatistics() {
  const request = ++statisticsRequest;
  $("statistics-loading").textContent = "Loading statistics…";
  $("statistics-loading").hidden = false;
  $("statistics-content").hidden = true;
  try {
    const data = await api("/api/v1/statistics");
    if (request !== statisticsRequest) return;
    state.statistics = data;
    renderStatistics();
    $("statistics-loading").hidden = true;
    $("statistics-content").hidden = false;
  } catch (error) {
    if (request !== statisticsRequest) return;
    $("statistics-loading").textContent = `Could not load statistics: ${error.message}`;
    toast(error.message, "error");
  }
}

function renderStatistics() {
  const stats = state.statistics;
  if (!stats) return;
  const summary = stats.summary;
  const kpis = [
    [summary.images, "Images"],
    [summary.annotations, "Annotations"],
    [summary.annotated_images, "Annotated images"],
    [summary.unlabeled_images, "Unlabeled images"],
    [formatNumber(summary.average_annotations, 2), "Average per image"],
  ];
  $("stats-kpis").innerHTML = kpis.map(([value, label]) =>
    `<div class="stats-kpi"><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`).join("");

  const classes = [...stats.class_balance].sort((a, b) => b.annotations - a.annotations || a.class_id - b.class_id);
  const classMax = Math.max(1, ...classes.map(item => item.annotations));
  $("stats-class-balance").innerHTML = `<div class="stats-bars">${classes.map(item => `
    <div class="stats-bar-row" title="${esc(item.images)} image${item.images === 1 ? "" : "s"} · ${esc(item.percent)}% of annotations">
      <span class="stats-bar-label"><span class="chip" style="--c:${classColor(item.class_id)}">${esc(item.name)}</span></span>
      <span class="stats-bar-track"><span class="stats-bar-fill" style="--c:${classColor(item.class_id)};width:${item.annotations / classMax * 100}%"></span></span>
      <span class="stats-bar-value">${item.annotations} · ${item.images} img</span>
    </div>`).join("")}</div>`;

  renderDistribution("stats-density", stats.annotations_per_image, value => formatNumber(value, 1));
  renderDistribution("stats-box-size", stats.box_size, value => `${formatNumber(value * 100, 2)}%`);
  renderDistribution("stats-aspect", stats.aspect_ratio, value => formatNumber(value, 2));

  const classHeaders = classes.map(item => item.class_id);
  $("stats-splits").innerHTML = `<table class="stats-table"><thead><tr>
    <th>Split</th><th>Images</th><th>Annotated</th><th>Annotations</th><th>Avg/image</th>
    ${classHeaders.map(id => `<th title="${esc(state.meta.names[id] ?? `class ${id}`)}">${esc(state.meta.names[id] ?? id)}</th>`).join("")}
  </tr></thead><tbody>${stats.split_comparison.map(row => `<tr>
    <td><span class="split-badge">${esc(row.split)}</span></td><td>${row.images}</td><td>${row.annotated_images}</td><td>${row.annotations}</td><td>${formatNumber(row.average_annotations, 2)}</td>
    ${classHeaders.map(id => `<td>${row.class_annotations[id] || 0}</td>`).join("")}
  </tr>`).join("")}</tbody></table>`;

  renderCooccurrence(stats.cooccurrence);
  $("stats-outlier-count").textContent = stats.outliers.length ? `(${stats.outliers.length})` : "";
  $("stats-outliers").innerHTML = stats.outliers.map(item => `
    <div class="stats-outlier">
      <div><div class="name" title="${esc(item.image_name)}">${esc(item.image_name)}</div><span class="split-badge">${esc(item.split)}</span></div>
      <span class="pill">${esc(item.kind.replaceAll("_", " "))}</span>
      <span class="stats-outlier-reason">${esc(item.reason)}</span>
      <button data-stat-open="${item.image_id}" data-object="${esc(item.annotation_id || "")}">Open</button>
    </div>`).join("") || `<div class="empty">No strong statistical outliers detected.</div>`;
}

function renderDistribution(id, distribution, formatter) {
  const maximum = Math.max(1, ...distribution.bins.map(bin => bin.count));
  $(id).innerHTML = `<div class="stats-bars">${distribution.bins.map(bin => `
    <div class="stats-bar-row">
      <span class="stats-bar-label">${esc(bin.label)}</span>
      <span class="stats-bar-track"><span class="stats-bar-fill" style="width:${bin.count / maximum * 100}%"></span></span>
      <span class="stats-bar-value">${bin.count}</span>
    </div>`).join("")}</div>
    <div class="stats-summary"><span>Median <strong>${formatter(distribution.median)}</strong></span><span>Q1 <strong>${formatter(distribution.q1)}</strong></span><span>Q3 <strong>${formatter(distribution.q3)}</strong></span><span>P95 <strong>${formatter(distribution.p95)}</strong></span><span>Max <strong>${formatter(distribution.max)}</strong></span></div>`;
}

function renderCooccurrence(data) {
  const present = data.class_ids.map((id, index) => ({ id, index, count: data.matrix[index][index] }))
    .filter(item => item.count > 0).sort((a, b) => b.count - a.count || a.id - b.id);
  const ranked = present.slice(0, 12);
  const maximum = Math.max(1, ...ranked.flatMap(row => ranked.map(column => data.matrix[row.index][column.index])));
  $("stats-cooccurrence").innerHTML = ranked.length ? `<table class="cooccurrence"><thead><tr><th></th>${ranked.map(item => `<th title="${esc(data.names[item.index])}">${esc(data.names[item.index])}</th>`).join("")}</tr></thead><tbody>
    ${ranked.map(row => `<tr><th title="${esc(data.names[row.index])}">${esc(data.names[row.index])}</th>${ranked.map(column => {
      const count = data.matrix[row.index][column.index];
      return `<td style="--heat:${count / maximum}" title="${esc(data.names[row.index])} + ${esc(data.names[column.index])}: ${count} image${count === 1 ? "" : "s"}">${count || ""}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table>${present.length > ranked.length ? `<p class="panel-note">Showing the 12 classes present in the most images.</p>` : ""}` : `<p class="empty-note">No annotated classes to compare.</p>`;
}

function formatNumber(value, digits = 0) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function statisticsChanged() {
  state.statistics = null;
  if ($("statistics").classList.contains("active")) loadStatistics();
}

/* ---------- validation ---------- */

export async function loadIssues() {
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
    statisticsChanged();
    const skipped = result.skipped.length ? ` · ${result.skipped.length} image${result.skipped.length === 1 ? "" : "s"} skipped (${result.skipped[0].reason})` : "";
    toast(`Fixed ${result.fixed} issue${result.fixed === 1 ? "" : "s"} in ${result.files} file${result.files === 1 ? "" : "s"}${skipped}`, result.skipped.length ? "error" : "info");
    await Promise.all([loadImages(), loadObjects(), loadIssues()]);
  } catch (error) {
    toast(error.message, "error");
  }
}

configureCanvas({
  acceptPredictions,
  classOptions,
  drawPredictions,
  hitPrediction,
  imageGallery: () => imageGallery,
  loadEditorPredictions,
  loadImages,
  loadIssues,
  loadObjects,
  predictImage,
  renderPredList,
  statisticsChanged,
  toast,
});
configurePredictor({
  bounds,
  editorRequest: () => editorRequest,
  loadImages,
  loadIssues,
  loadObjects,
  render,
  renderList,
  savedRevisionByImage,
  saveQueue: () => saveQueue,
  shapePath,
  statisticsChanged,
  toast,
  toScreen,
  updateHistoryButtons,
});
configureEmbeddings({ classOptions, editObject, toast });

init().catch(error => toast(error.message, "error"));
