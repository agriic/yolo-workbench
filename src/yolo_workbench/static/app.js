import { $, api, apiUrl, setApiBase, esc, debounce, classColor } from "./api.js";
import { state } from "./state.js";
import { VirtualGallery } from "./grid.js";
import {
  applyHistory, bounds, closeEditorSession, configureCanvas, deleteSelected,
  editorRequest, finishPolygon, fit, navigate, openEditor, pointerDown, pointerMove,
  pointerUp, removeAnnotation, render, renderList, resizeCanvas, save, saveQueue,
  savedRevisionByImage, shapePath, toScreen, updateHistoryButtons, zoomCentered, updateNavButtons,
  updateReviewButtons, updateStatus, zoomAt, zoomTo,
} from "./canvas.js";
import {
  acceptPredictions, applyPredictorState, cancelPredict, configurePredictor,
  drawPredictions, hitPrediction, loadEditorPredictions, loadModel, maxPointsChanged, predictImage,
  refreshPredictor, rejectPredictions, renderModelMenu, renderPredList,
  retryPredict, runPredict,
} from "./predictor.js";
import {
  computeEmbeddings, configureEmbeddings, embedPointerCancel, embedPointerDown, embedPointerMove,
  embedPointerUp, embedWheel, refreshEmbeddings, reclassifyEmbedObject,
  renderEmbeddings, renderEmbedSelection, resetEmbedView, setEmbedMode,
} from "./embeddings.js";
import {
  acceptSam, configureSam, drawSam, hasPrompts, clearPrompts, refreshSam, removeLastPrompt, samEditorClosed,
  samHint, samImageOpened, samMaxPointsChanged, samPointerDown, samPointerMove, samPointerUp, samRightClick, toggleSam,
} from "./sam.js";
import { bindManage, configureManage, loadClasses, loadStorage, renderManageMeta } from "./manage.js";
import { bindRepair, configureRepair, openLabelText, renderUnknownClasses } from "./repair.js";
import { bindLabelQA, configureLabelQA, labelsChangedForQA, showLabelQA } from "./labelqa.js";
import { bindTraining, configureTraining, hideTraining, showTraining } from "./training.js";

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
  const info = await api("/api/v1/app");
  state.mode = info.mode;
  if (info.mode === "projects") {
    const slug = new URLSearchParams(location.search).get("project");
    if (!slug) { location.replace("/projects.html"); return; }
    state.project = slug;
    setApiBase(`/api/v1/projects/${encodeURIComponent(slug)}`);
  }
  try {
    state.meta = await api("/dataset");
  } catch (error) {
    if (state.mode === "projects") {
      // a deleted or busy project: go back to the list, which explains what is open
      sessionStorage.setItem("projects-message", error.message);
      location.replace("/projects.html");
      return;
    }
    throw error;
  }
  state.canUndo = state.meta.can_undo;
  state.canRedo = state.meta.can_redo;
  renderMetaControls();
  renderShortcuts();
  setupVirtualGalleries();
  bind();
  bindManage();
  bindRepair();
  bindLabelQA();
  bindTraining();
  updateHistoryButtons();
  watchIndexing();
  if (state.mode === "projects") renderProjectSwitch();
  const params = new URLSearchParams(location.search);
  if (params.get("tab") === "training") showTab("training", params.get("run"));
  else if (!state.meta.image_count && state.meta.capabilities.image_ops) showTab("dataset");
  await Promise.all([loadImages(), loadObjects(), loadIssues(), refreshPredictor(), refreshSam()]);
}

// Fill every class/split control from state.meta; safe to call again after a class change.
function renderMetaControls() {
  const meta = state.meta;
  document.body.classList.toggle("classification", meta.category === "classification");
  document.body.classList.toggle("image-ops", !!meta.capabilities.image_ops);
  if (state.mode !== "projects") {
    const yamlName = meta.yaml.split("/").slice(-2).join("/");
    $("dataset-name").textContent = `${yamlName} · ${meta.category}`;
    $("dataset-name").title = meta.yaml;
  } else {
    $("dataset-name").hidden = true;
    $("project-category").textContent = meta.category;
    document.title = `${meta.project?.name ?? "Project"} · YOLO Workbench`;
  }
  const names = Object.entries(meta.names);
  for (const [id, keepAll] of [["image-class", true], ["object-class", false], ["draw-class", false], ["bulk-class", false]]) {
    const select = $(id);
    const previous = select.selectedOptions[0]?.dataset.name;
    select.innerHTML = (keepAll ? `<option value="">All classes</option>` : "") + names.map(([classId, name]) =>
      `<option value="${esc(classId)}" data-name="${esc(name)}">${esc(classId)} · ${esc(name)}</option>`).join("");
    // keep the same class selected by name: ids may have been renumbered
    const match = [...select.options].find(option => option.dataset.name === previous);
    if (match) select.value = match.value;
  }
  const splits = [...new Set([...Object.keys(meta.split_counts), ...meta.capabilities.splits])];
  for (const id of ["image-split", "object-split", "embed-split"]) {
    const select = $(id);
    const previous = select.value;
    select.innerHTML = `<option value="all">All splits</option>` + Object.keys(meta.split_counts).map(split => `<option>${esc(split)}</option>`).join("");
    select.value = [...select.options].some(option => option.value === previous) ? previous : "all";
  }
  // move targets are split folders; the folder is only spelled out when a split has several
  const folders = meta.capabilities.folders || {};
  $("image-move-split").innerHTML = meta.capabilities.splits.map(split => (folders[split] || []).map(folder =>
    `<option value="${esc(`${split}|${folder}`)}" title="${esc(folder)}">${esc(folders[split].length > 1 ? `${split} · ${folder}` : split)}</option>`).join("")).join("");
  $("embed-classes").innerHTML = names.map(([id, name]) =>
    `<button class="class-chip ${state.embed.classes.has(+id) ? "active" : ""}" data-class="${esc(id)}" style="--c:${classColor(+id)}">${esc(name)}</button>`).join("");
  if (imageGallery) imageGallery.emptyMessage = imageEmptyMessage();
  renderManageMeta();
}

const imageEmptyMessage = () => state.meta.image_count
  ? "No images match the current filters."
  : state.meta.capabilities.image_ops ? "No images yet — import some in the Dataset tab." : "This dataset has no images.";

async function renderProjectSwitch() {
  $("project-switch").hidden = false;
  const select = $("project-select");
  const current = state.meta.project;
  select.innerHTML = `<option value="${esc(state.project)}">${esc(current?.name ?? state.project)}</option>`;
  select.onchange = () => { location.href = `/?project=${encodeURIComponent(select.value)}`; };
  try {
    const { projects } = await api("/api/v1/projects");
    select.innerHTML = projects.map(project =>
      `<option value="${esc(project.slug)}" ${project.slug === state.project ? "selected" : ""}>${esc(project.name)}</option>`).join("");
  } catch { /* the switcher is a convenience; the projects link still works */ }
}

async function refreshMeta() {
  state.meta = { ...state.meta, ...(await api("/dataset")) };
  renderMetaControls();
}

// Something changed beyond one image (class schema, import, image delete/move, undo of those):
// refresh metadata and every view that caches classes, splits, or images.
export async function datasetChanged({ schema = false } = {}) {
  if (schema && $("editor").open) $("editor").close();
  if (schema) {
    state.embed.classes.clear();
    state.embed.selection = [];
    clearObjectSelection();
  }
  clearImageSelection();
  try { await refreshMeta(); } catch (error) { toast(error.message, "error"); }
  statisticsChanged();
  updateHistoryButtons();
  const tasks = [loadImages({ reset: schema }), loadObjects({ reset: schema }), loadIssues(), refreshPredictor()];
  if ($("dataset").classList.contains("active")) tasks.push(loadClasses(), loadStorage());
  if ($("embeddings").classList.contains("active")) tasks.push(refreshEmbeddings());
  labelsChangedForQA();  // findings are recomputed from the labels (undo of a Label QA fix, imports, …)
  await Promise.all(tasks.map(task => Promise.resolve(task).catch(error => toast(error.message, "error"))));
}

function showTab(tab, runId = null) {
  const leaving = document.querySelector(".view.active")?.id;
  document.querySelectorAll(".tab,.view").forEach(el => el.classList.remove("active"));
  document.querySelector(`.tab[data-tab="${tab}"]`).classList.add("active");
  $(tab).classList.add("active");
  document.body.classList.toggle("on-training", tab === "training");
  if (leaving === "training" && tab !== "training") {
    hideTraining();
    refreshPredictor();  // a run that started unloaded the models
    refreshSam();
    const url = new URL(location.href);
    url.searchParams.delete("tab");
    url.searchParams.delete("run");
    history.replaceState(null, "", url);
  }
  if (tab === "training") showTraining(runId);
  if (tab === "annotation") imageGallery?.refreshLayout();
  if (tab === "exploration") objectGallery?.refreshLayout();
  if (tab === "embeddings") refreshEmbeddings();
  if (tab === "statistics") loadStatistics();
  if (tab === "quality") showLabelQA();
  if (tab === "dataset") { loadClasses(); loadStorage(); }
}

function watchIndexing() {
  const el = $("indexing-status");
  const idx = state.meta.indexing;
  if (!idx || idx.done >= idx.total) { el.hidden = true; return; }
  el.hidden = false;
  el.textContent = `Indexing ${idx.done}/${idx.total}…`;
  clearTimeout(watchIndexing.timer);
  watchIndexing.timer = setTimeout(async () => {
    try { state.meta = { ...state.meta, ...(await api("/dataset")) }; } catch { /* retry on next tick */ }
    watchIndexing();
  }, 1000);
}

function renderShortcuts() {
  if (state.meta.category === "classification") {
    $("shortcut-list").innerHTML = [
      ["1–9", "Set the image class"],
      ["Scroll", "Zoom at cursor"],
      ["Drag", "Pan the image"],
      ["Ctrl+Z / Ctrl+⇧+Z", "Undo / redo"],
      ["← →", "Previous / next image"],
      ["D / W", "Review: done / needs work, then next"],
      ["U", "Next unreviewed image"],
      ["F", "Fit image"],
      ["Esc", "Close"],
    ].map(([key, hint]) => `<dt><kbd>${key}</kbd></dt><dd>${hint}</dd>`).join("");
    return;
  }
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
    ["D / W", "Review: done / needs work, then next"],
    ["U", "Next unreviewed image"],
    ["P", "Predict with the loaded model"],
    ["S", "Smart annotation (SAM): click the object"],
    ["Right click", "SAM: mark background (or Shift+Click)"],
    ["⏎", "SAM: add the outlined object"],
    ["F", "Fit image"],
    ["Esc", "Cancel drawing / close"],
  ];
  $("shortcut-list").innerHTML = rows.map(([key, hint]) => `<dt><kbd>${key}</kbd></dt><dd>${hint}</dd>`).join("");
}

function bind() {
  document.querySelectorAll(".tab").forEach(button => button.onclick = () => showTab(button.dataset.tab));
  [$("image-split"), $("image-class"), $("filter-predictions"), $("image-review"), $("image-issues")].forEach(el => el.onchange = () => loadImages({ reset: true }));
  $("start-review").onclick = () => openNextUnreviewed(0);
  $("image-review-done").onclick = () => reviewSelection("done");
  $("image-review-work").onclick = () => reviewSelection("needs_work");
  $("image-review-clear").onclick = () => reviewSelection(null);
  $("show-overlays").onchange = () => loadImages();
  $("image-search").oninput = debounce(() => loadImages({ reset: true }), 250);
  [$("label-min"), $("label-max")].forEach(el => el.oninput = debounce(() => loadImages({ reset: true }), 250));
  $("image-filters-reset").onclick = () => {
    for (const [id, value] of [["image-split", "all"], ["image-class", ""], ["image-review", ""], ["image-issues", ""], ["label-min", ""], ["label-max", ""]]) $(id).value = value;
    $("filter-predictions").checked = false;
    loadImages({ reset: true });
  };
  // the filters and the prediction controls share one popup; the model chip opens it too
  $("image-options-toggle").onclick = () => showImageOptions($("image-options").hidden);
  $("model-chip").onclick = () => showImageOptions(true);
  document.addEventListener("pointerdown", e => {
    if (!$("image-options").hidden && !e.target.closest("#image-options-anchor, #model-chip")) showImageOptions(false);
  });
  $("image-options").addEventListener("keydown", e => {
    if (e.key !== "Escape") return;
    e.stopPropagation();
    if (!$("model-menu").hidden) return void ($("model-menu").hidden = true);
    showImageOptions(false);
    $("image-options-toggle").focus();
  });
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
      applyPredictorState(await api("/predictor/mapping", {
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
  $("pred-max-points").oninput = maxPointsChanged;
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
    const check = e.target.closest(".card-check");
    if (check) return handleImageCheck(check, e.shiftKey);
    const card = e.target.closest("[data-image]");
    if (card) openEditor(card.dataset.image, null, +card.dataset.index);
  });
  $("image-move").onclick = () => runImageBulk("move");
  $("image-delete").onclick = () => runImageBulk("delete");
  $("image-select-clear").onclick = clearImageSelection;
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
    const show = e.target.closest("[data-show-kind]");
    if (show) return showIssueImages(show.dataset.showKind);
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
    const raw = e.target.closest("[data-edit-label]");
    if (raw) return openLabelText(raw.dataset.editLabel);
    const fix = e.target.closest("[data-fix]");
    if (!fix) return;
    try {
      await api(`/issues/${fix.dataset.fix}/fix`, { method: "POST" });
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
  // a click on the active status unmarks it; otherwise mark and move on, like D / W
  $("review-done").onclick = e => { e.currentTarget.blur(); reviewCurrent("done", state.detail?.review !== "done"); };
  $("review-work").onclick = e => { e.currentTarget.blur(); reviewCurrent("needs_work", state.detail?.review !== "needs_work"); };
  $("review-next").onclick = e => { e.currentTarget.blur(); openNextUnreviewed(); };
  $("edit-label-text").onclick = () => state.detail && openLabelText(state.detail.id);
  $("label-warning").addEventListener("click", e => {
    if (e.target.closest("[data-edit-label]") && state.detail) openLabelText(state.detail.id);
  });
  // blurred so ⏎ (accept) and Space (pan) don't press the button again
  $("smart-annotate").onclick = e => { e.currentTarget.blur(); toggleSam(); };
  $("sam-max-points").oninput = samMaxPointsChanged;
  // ⏎ in a point-limit field hands the keyboard back to the editor (the next ⏎ accepts)
  for (const id of ["sam-max-points", "pred-max-points"]) $(id).addEventListener("keydown", e => { if (e.key === "Enter") e.currentTarget.blur(); });
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
    // Esc in smart annotation clears the prompts, then leaves the mode, before it closes the editor
    else if (state.sam.active) {
      e.preventDefault();
      if (hasPrompts()) { clearPrompts(); render(); } else toggleSam(false);
    }
  });
  $("editor").addEventListener("close", () => {
    closeEditorSession();
    state.detail = null; state.img = null; state.drawing = null; state.drag = null; state.selected = null;
    state.editorGridIndex = null;
    state.editorDetached = false;
    state.pred.items = [];
    renderPredList();
    samEditorClosed();
    labelsChangedForQA();
  });

  $("annotation-list").addEventListener("click", e => {
    const row = e.target.closest("[data-annotation]");
    if (!row) return;
    if (e.target.closest(".row-delete")) return removeAnnotation(row.dataset.annotation);
    if (e.target.closest("select")) {
      // select the row too, without re-rendering the list (that would close the open dropdown)
      if (state.selected === row.dataset.annotation) return;
      state.selected = row.dataset.annotation;
      for (const other of $("annotation-list").children) other.classList.toggle("selected", other === row);
      return render();
    }
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
  if (!$("editor").open) {
    if (key === "Escape" && state.imageSelection.size && $("annotation").classList.contains("active")) clearImageSelection();
    return;
  }
  if (key === " ") { e.preventDefault(); state.spaceDown = true; $("canvas").style.cursor = "grab"; return; }
  if ((key === "Delete" || key === "Backspace") && state.sam.active && hasPrompts()) return removeLastPrompt();
  if (key === "Delete" || key === "Backspace") return deleteSelected();
  if (key === "ArrowLeft") return navigate(-1);
  if (key === "ArrowRight") return navigate(1);
  if (key.toLowerCase() === "f") return fit();
  if (key.toLowerCase() === "p") return predictImage();
  if (!e.ctrlKey && !e.metaKey && !e.altKey) {
    if (key.toLowerCase() === "d") return reviewCurrent("done", true);
    if (key.toLowerCase() === "w") return reviewCurrent("needs_work", true);
    if (key.toLowerCase() === "u") return openNextUnreviewed();
  }
  if (key.toLowerCase() === "s" && !e.ctrlKey && !e.metaKey && !e.altKey) return toggleSam();
  if (key === "Enter" && state.sam.active) { e.preventDefault(); return acceptSam(); }
  if (key === "+" || key === "=") return zoomCentered(1.25);
  if (key === "-") return zoomCentered(0.8);
  if (key === "Enter" && state.drawing?.points?.length >= 3) return finishPolygon();
  if (key >= "1" && key <= "9") {
    const classId = Object.keys(state.meta.names)[+key - 1];
    if (classId === undefined) return;
    if (state.meta.category === "classification") {
      const annotation = state.detail?.annotations[0];
      if (annotation && annotation.class_id !== +classId) { annotation.class_id = +classId; save(); }
      return;
    }
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
    emptyMessage: imageEmptyMessage(),
    fetchPage: fetchImagePage,
    renderCard: renderImageCard,
    isPinned: item => state.imageSelection.has(item.id),
    onUpdate: total => {
      $("image-total").textContent = `${total.toLocaleString()} image${total === 1 ? "" : "s"}`;
      updateNavButtons();
    },
  });
  imageGallery.enableMarquee({
    canStart: () => !state.imageSelectionBusy,
    onSelect: marqueeSelect,
    onClear: additive => { if (!additive && state.imageSelection.size) clearImageSelection(); },
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

// The Annotation tab's filters as query parameters (the image list and "next unreviewed" share them).
function imageFilterParams() {
  const params = new URLSearchParams({ split: $("image-split").value, search: $("image-search").value });
  if ($("image-class").value !== "") params.set("class_id", $("image-class").value);
  if ($("filter-predictions").checked) params.set("has_predictions", "true");
  if ($("image-review").value) params.set("review", $("image-review").value);
  if ($("image-issues").value) params.set("issue", $("image-issues").value);
  if (state.meta.category !== "classification") {
    for (const [id, key] of [["label-min", "min_labels"], ["label-max", "max_labels"]]) {
      const value = $(id).value.trim();
      if (/^\d+$/.test(value)) params.set(key, value);
    }
  }
  return params;
}

function showImageOptions(open) {
  $("image-options").hidden = !open;
  $("image-options-toggle").setAttribute("aria-expanded", String(open));
}

// the toolbar button counts the popup's active filters (the search box is in plain sight)
function renderFilterCount() {
  const params = imageFilterParams();
  params.delete("search");
  if (params.get("split") === "all") params.delete("split");
  const count = new Set([...params.keys()].map(key => key.endsWith("_labels") ? "labels" : key)).size;
  $("image-filter-count").textContent = count || "";
}

function fetchImagePage(offset, limit, signal) {
  const params = imageFilterParams();
  params.set("limit", String(limit));
  params.set("offset", String(offset));
  return api(`/images?${params}`, { signal });
}

function fetchObjectPage(offset, limit, signal) {
  const classId = $("object-class").value || Object.keys(state.meta.names)[0];
  if (classId === undefined) return Promise.resolve({ total: 0, items: [] });
  const params = new URLSearchParams({ class_id: classId, split: $("object-split").value, limit: String(limit), offset: String(offset) });
  return api(`/objects?${params}`, { signal });
}

export async function loadImages(options = {}) {
  if (!imageGallery) return;
  renderFilterCount();
  await imageGallery.reload(options);
  await syncEditorGridIndex();
}

// Keep prev/next working when an edit (e.g. accepting predictions under "Has predictions")
// drops the open image out of the filtered grid: remember the gap instead of losing the position.
async function syncEditorGridIndex() {
  if (!state.detail || state.editorGridIndex == null) return;
  const id = state.detail.id, index = state.editorGridIndex;
  let item;
  try {
    item = await imageGallery.itemAt(index);
  } catch (error) {
    if (error.name !== "AbortError") toast(error.message, "error");
    return;
  }
  if (state.detail?.id !== id || state.editorGridIndex !== index) return;
  if (item?.id === id) {
    state.editorDetached = false;
  } else if (!state.editorDetached) {
    const moved = [...imageGallery.cache].find(([, cached]) => cached.id === id);
    if (moved) state.editorGridIndex = moved[0];
    else state.editorDetached = true;
  }
  updateNavButtons();
}

function renderImageCard(item, index) {
  const overlay = $("show-overlays").checked ? "&annotated=1" : "";
  const chips = item.classes.slice(0, 3).map(id =>
    `<span class="chip" style="--c:${classColor(id)}">${esc(state.meta.names[id] ?? `class ${id}`)}</span>`).join("");
  const more = item.classes.length > 3 ? `<span class="chip">+${item.classes.length - 3}</span>` : "";
  return `<article class="card ${state.imageSelection.has(item.id) ? "selected" : ""}" data-image="${item.id}" data-index="${index}" tabindex="0">
      <div class="thumb"><img loading="lazy" src="${apiUrl(`/images/${item.id}/thumbnail?size=320${overlay}`)}" alt="">
        <input type="checkbox" class="card-check" data-index="${index}" title="Select (Shift-click for a range, or drag a rectangle from the space between images)" aria-label="Select ${esc(item.name)}" ${state.imageSelection.has(item.id) ? "checked" : ""}>
        ${item.review ? `<span class="review-flag ${item.review}" title="${item.review === "done" ? "Reviewed: done" : "Review: needs work"}">${item.review === "done" ? "✓" : "⚑"}</span>` : ""}
        ${item.issue_count ? `<span class="issue-flag">${item.issue_count} ⚠</span>` : ""}
        ${state.pred.pending[item.id] ? `<span class="pred-flag" title="Pending predictions">${state.pred.pending[item.id]} ✨</span>` : ""}</div>
      <div class="card-body">
        <div class="name" title="${esc(item.name)}">${esc(item.name)}</div>
        <div class="meta"><span class="split-badge">${esc(item.split)}</span><span>${item.annotation_count} object${item.annotation_count === 1 ? "" : "s"}</span></div>
        <div class="chips">${chips}${more}</div>
      </div>
    </article>`;
}

async function handleImageCheck(check, shiftKey) {
  if (state.imageSelectionBusy) return imageGallery.render();
  const index = +check.dataset.index;
  const on = check.checked;
  const start = shiftKey && state.lastImageIndex != null ? Math.min(index, state.lastImageIndex) : index;
  const end = shiftKey && state.lastImageIndex != null ? Math.max(index, state.lastImageIndex) + 1 : index + 1;
  state.imageSelectionBusy = end - start > 1;
  renderImageBulkBar();
  try {
    const items = await imageGallery.ensureRange(start, end);
    if (items.length !== end - start) throw new Error("Could not load the complete selection range");
    for (const item of items) on ? state.imageSelection.set(item.id, item) : state.imageSelection.delete(item.id);
    state.lastImageIndex = index;
  } catch (error) {
    if (error.name !== "AbortError") toast(error.message, "error");
  } finally {
    state.imageSelectionBusy = false;
    imageGallery.render();
    renderImageBulkBar();
  }
}

// Rectangle selection from the gallery: replaces the selection, or adds to it with Shift/Ctrl.
async function marqueeSelect(indices, additive) {
  if (!additive) state.imageSelection.clear();
  if (!indices.length) return renderImageBulkBar(), imageGallery.render();
  const first = Math.min(...indices), last = Math.max(...indices);
  state.imageSelectionBusy = last - first > 0;
  renderImageBulkBar();
  try {
    await imageGallery.ensureRange(first, last + 1);
    for (const index of indices) {
      const item = imageGallery.cache.get(index);
      if (item) state.imageSelection.set(item.id, item);
    }
    state.lastImageIndex = last;
  } catch (error) {
    if (error.name !== "AbortError") toast(error.message, "error");
  } finally {
    state.imageSelectionBusy = false;
    imageGallery.render();
    renderImageBulkBar();
  }
}

function renderImageBulkBar() {
  const count = state.imageSelection.size;
  $("image-bulk").hidden = !count && !state.imageSelectionBusy;
  $("image-selected-count").textContent = state.imageSelectionBusy ? "Selecting range…" : `${count} selected`;
  $("image-bulk").querySelectorAll("button,select").forEach(control => control.disabled = state.imageSelectionBusy);
}

function clearImageSelection() {
  state.imageSelection.clear();
  state.lastImageIndex = null;
  renderImageBulkBar();
  imageGallery?.render();
}

/* ---------- review status ---------- */

const REVIEW_NAMES = { done: "done", needs_work: "needs work" };

async function setReview(imageIds, status) {
  const result = await api("/images/review", { method: "POST", body: JSON.stringify({ image_ids: imageIds, status }) });
  state.meta.review_counts = result.review_counts;
  return result;
}

async function reviewSelection(status) {
  const ids = [...state.imageSelection.keys()];
  if (!ids.length) return;
  try {
    const result = await setReview(ids, status);
    toast(status ? `Marked ${result.updated} image${result.updated === 1 ? "" : "s"} ${REVIEW_NAMES[status]}` : `Cleared the review status of ${result.updated} image${result.updated === 1 ? "" : "s"}`);
    clearImageSelection();
    await loadImages();
  } catch (error) { toast(error.message, "error"); }
}

// Editor: set the status of the open image; `advance` moves on to the next image of the list
// (under the "Unreviewed" filter the image drops out of the list, and its successor takes its place).
async function reviewCurrent(status, advance) {
  if (!state.detail) return;
  const id = state.detail.id;
  const next = advance || state.detail.review !== status ? status : null;
  try {
    if (state.detail.review !== next) {
      await setReview([id], next);
      if (state.detail?.id === id) state.detail.review = next;
      updateReviewButtons();
    }
    await loadImages();
    if (!advance || state.detail?.id !== id) return;
    const index = state.editorGridIndex, total = imageGallery.total;
    if (index == null) return;  // opened from outside the list (Validation, Label QA)
    if (state.editorDetached ? index >= total : index + 1 >= total) return toast("That was the last image in this list");
    await navigate(1);
  } catch (error) { toast(error.message, "error"); }
}

// Open the first unreviewed image after the editor's position (or from `start`), wrapping around the list.
async function openNextUnreviewed(start = null) {
  const from = start ?? (state.detail && state.editorGridIndex != null ? state.editorGridIndex + (state.editorDetached ? 0 : 1) : 0);
  const params = imageFilterParams();
  params.set("start", String(from));
  try {
    const result = await api(`/images/next-unreviewed?${params}`);
    if (result.index == null) return toast(result.total ? "Every image in this list is reviewed" : "No images match the current filters");
    openEditor(result.item.id, null, result.index);
  } catch (error) { toast(error.message, "error"); }
}

/* ---------- issue filter ---------- */

function renderIssueFilter() {
  const select = $("image-issues");
  const previous = select.value;
  const kinds = [...new Set(state.issues.filter(issue => issue.image_id).map(issue => issue.kind))].sort();
  // a kind that was just fixed away stays selectable, so the empty list explains itself
  if (previous && previous !== "any" && previous !== "none" && !kinds.includes(previous)) kinds.push(previous);
  select.innerHTML = `<option value="">All</option><option value="any">With issues</option><option value="none">Without issues</option>`
    + kinds.map(kind => `<option value="${esc(kind)}">${esc(kind.replaceAll("_", " "))}</option>`).join("");
  select.value = previous;
}

function showIssueImages(kind) {
  renderIssueFilter();
  $("image-issues").value = kind;
  showTab("annotation");
  loadImages({ reset: true });
}

async function runImageBulk(action) {
  const image_ids = [...state.imageSelection.keys()];
  if (!image_ids.length) return;
  const count = `${image_ids.length} image${image_ids.length === 1 ? "" : "s"}`;
  if (action === "delete" && !confirm(`Delete ${count}? They are moved to the dataset trash with their labels; Ctrl+Z undoes.`)) return;
  try {
    const target = $("image-move-split").value;
    const separator = target.indexOf("|");
    const split = target.slice(0, separator), folder = target.slice(separator + 1);
    const result = action === "delete"
      ? await api("/images/delete", { method: "POST", body: JSON.stringify({ image_ids }) })
      : await api("/images/move", { method: "POST", body: JSON.stringify({ image_ids, split, folder }) });
    state.canUndo = result.can_undo;
    state.canRedo = result.can_redo;
    toast(action === "delete" ? `Deleted ${count} — Ctrl+Z undoes` : `Moved ${result.moved} to ${split} (${folder})${result.skipped ? ` · ${result.skipped} already there` : ""} — Ctrl+Z undoes`);
    await datasetChanged();
  } catch (error) {
    toast(error.message, "error");
  }
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
        <img loading="lazy" src="${apiUrl(`/objects/${item.image_id}/${encodeURIComponent(item.id)}/crop?padding=${padding}`)}" alt="">
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
    const result = await api("/objects/bulk", { method: "POST", body: JSON.stringify({ operations }) });
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

// One image's labels were rewritten outside the editor's own save (raw label editor): refresh what shows them.
async function labelsChanged(imageId, detail) {
  state.canUndo = true;
  state.canRedo = false;
  updateHistoryButtons();
  statisticsChanged();
  if (state.detail?.id === imageId && detail) {
    state.detail = { ...detail, image_version: state.detail.image_version };
    savedRevisionByImage.set(imageId, detail.revision);
    state.selected = null;
    renderList();
    render();
  }
  await Promise.all([loadImages(), loadObjects(), loadIssues()].map(task => task.catch(error => toast(error.message, "error"))));
}

export async function editObject(imageId, id, change) {
  try {
    const detail = await api(`/images/${imageId}`);
    if (change) change(detail.annotations.find(a => a.id === id));
    else detail.annotations = detail.annotations.filter(a => a.id !== id);
    await api(`/images/${imageId}/annotations`, {
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
    const data = await api("/statistics");
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
  const data = await api("/issues");
  state.issues = data.items;
  $("issue-count").textContent = data.items.length || "";
  renderIssues();
  renderUnknownClasses(data.unknown_classes || []);
  renderIssueFilter();
}

// What "Fix" does for kinds where the generic word would be unclear: [button, bulk button, tooltip, confirm question]
const FIX_ACTIONS = {
  malformed_label: ["Remove line", "Remove all", "Remove the malformed line from the label file", count => `Remove ${count} malformed line${count === 1 ? "" : "s"} from the label files?`],
  orphan_label: ["Trash", "Trash all", "Move the label file to the dataset trash", count => `Move ${count} orphan label file${count === 1 ? "" : "s"} to the dataset trash?`],
  overlap: ["Delete later", "Delete later ones", "Delete the later of two same-class objects in the same place (other classes need a manual choice)", count => `Delete the later object of ${count} same-class overlap${count === 1 ? "" : "s"}, keeping the earlier one?`],
};

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
      ${fixable[kind] ? `<button class="fix-all" data-fix-kind="${esc(kind)}" data-count="${fixable[kind]}" title="${esc(FIX_ACTIONS[kind]?.[2] ?? "Fix")} — all ${fixable[kind]}">${esc(FIX_ACTIONS[kind]?.[1] ?? "Fix all")}</button>` : ""}
      ${kind !== "orphan_label" ? `<button class="show-kind" data-show-kind="${esc(kind)}" title="Show these images in the Annotation tab">Show images</button>` : ""}
    </span>`).join("");
  const visible = state.issueFilter ? items.filter(issue => issue.kind === state.issueFilter) : items;
  // Open comes last and keeps its slot on every row: fixing an issue removes its row, so the rows below move up
  // under the pointer, and clicking Open again in the same spot must not land on the next row's fix button.
  $("issues").innerHTML = visible.map(issue => `
    <div class="issue">
      <span class="severity-dot ${esc(issue.severity)}" title="${esc(issue.severity)}"></span>
      <span class="pill">${esc(issue.kind.replaceAll("_", " "))}</span>
      <div><strong title="${esc(issue.path || issue.image_name)}">${esc(issue.path || issue.image_name)}</strong><div class="message">${esc(issue.message)}</div></div>
      <span class="split-badge">${esc(issue.split || "—")}</span>
      <span class="issue-actions">
        ${issue.kind === "malformed_label" ? `<button data-edit-label="${issue.image_id}" title="Repair the line in the raw label file">Edit file</button>` : ""}
        ${issue.fixable ? `<button data-fix="${issue.id}" title="${esc(FIX_ACTIONS[issue.kind]?.[2] ?? "Fix automatically")} (Ctrl+Z undoes)">${esc(FIX_ACTIONS[issue.kind]?.[0] ?? "Fix")}</button>` : ""}
        ${issue.image_id ? `<button class="issue-open" data-issue-open="${issue.image_id}" data-object="${esc(issue.annotation_id || "")}">Open</button>` : `<span class="issue-open" aria-hidden="true"></span>`}
      </span>
    </div>`).join("") || `<div class="empty">No issues found. The dataset looks clean.</div>`;
}

async function fixAllIssues(kind, count) {
  const question = FIX_ACTIONS[kind]?.[3](count) ?? `Fix all ${count} ${kind.replaceAll("_", " ")} issue${count === 1 ? "" : "s"}?`;
  if (!confirm(`${question} Ctrl+Z undoes the whole batch.`)) return;
  try {
    const result = await api("/issues/fix-bulk", { method: "POST", body: JSON.stringify({ kind }) });
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
  datasetChanged,
  drawPredictions,
  drawSam,
  hitPrediction,
  imageGallery: () => imageGallery,
  loadEditorPredictions,
  loadImages,
  loadIssues,
  loadObjects,
  predictImage,
  renderPredList,
  samHint,
  samImageOpened,
  samPointerDown,
  samPointerMove,
  samPointerUp,
  samRightClick,
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
configureSam({ render, save, shapePath, toast, toScreen, updateStatus });
configureRepair({ datasetChanged, labelsChanged, toast });
configureLabelQA({ labelsChanged, openEditor, toast });
configureEmbeddings({ classOptions, editObject, toast });
configureManage({ datasetChanged, toast });
configureTraining({
  toast,
  showTab,
  // "Use for labeling": a run's weights become the prediction model of the Annotation tab
  useModel: async path => {
    $("model-path").value = path;
    showTab("annotation");
    await loadModel();
  },
});

init().catch(error => toast(error.message, "error"));
