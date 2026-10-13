import { $, api, esc, classColor } from "./api.js";
import { state } from "./state.js";
let deps;

export function configureManage(dependencies) {
  deps = dependencies;
}

const SPLITS = ["train", "val", "test"];
const NEW_FOLDER = "";  // folder paths are never empty, so "" marks the "New folder…" option
const UPLOAD_EXTENSIONS = /\.(jpe?g|png|webp|bmp|tiff?|txt|ya?ml|json)$/i;
const UPLOAD_CONCURRENCY = 4;
const plural = (count, noun) => `${count.toLocaleString()} ${noun}${count === 1 ? "" : noun.endsWith("s") ? "es" : "s"}`;

let classItems = [];
let folderInfo = { items: [], editable: false };
let removing = null;      // class being removed in the dialog
let importSource = null;  // { path } or { staging_id }
let importScan = null;
let importJob = null;
let uploadAbort = null;

export function bindManage() {
  $("class-add-form").onsubmit = e => { e.preventDefault(); addClasses(); };
  $("class-table").addEventListener("change", e => {
    const input = e.target.closest("[data-rename]");
    if (input) renameClass(+input.dataset.rename, input);
  });
  $("class-table").addEventListener("keydown", e => {
    const input = e.target.closest("[data-rename]");
    if (!input) return;
    if (e.key === "Enter") input.blur();
    if (e.key === "Escape") { input.value = input.dataset.original; input.blur(); }
  });
  $("class-table").addEventListener("click", e => {
    const remove = e.target.closest("[data-remove]");
    if (remove) return openRemoveDialog(+remove.dataset.remove);
    const move = e.target.closest("[data-move]");
    if (move) return moveClass(+move.dataset.move, +move.dataset.delta);
  });
  $("class-remove-form").addEventListener("change", updateRemoveDialog);
  $("class-remove-dialog").addEventListener("close", () => {
    if ($("class-remove-dialog").returnValue === "confirm") removeClass();
    removing = null;
  });
  $("rescan").onclick = rescan;
  $("split-overview").addEventListener("click", e => {
    const add = e.target.closest("[data-add-folder]");
    if (add) addFolder(add.dataset.addFolder);
  });

  $("import-pick-files").onclick = () => $("import-files").click();
  $("import-pick-folder").onclick = () => $("import-folder").click();
  $("import-files").onchange = e => { uploadFiles([...e.target.files].map(file => ({ file, path: file.name }))); e.target.value = ""; };
  $("import-folder").onchange = e => { uploadFiles([...e.target.files].map(file => ({ file, path: file.webkitRelativePath || file.name }))); e.target.value = ""; };
  const drop = $("import-drop");
  drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("dragging"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
  drop.addEventListener("drop", async e => {
    e.preventDefault();
    drop.classList.remove("dragging");
    const entries = [...e.dataTransfer.items].map(item => item.webkitGetAsEntry?.()).filter(Boolean);
    const files = entries.length ? await collectEntries(entries) : [...e.dataTransfer.files].map(file => ({ file, path: file.name }));
    uploadFiles(files);
  });
  $("import-path-form").onsubmit = e => {
    e.preventDefault();
    const path = $("import-path").value.trim();
    if (path) scanImport({ path });
  };
  $("import-cancel").onclick = cancelImport;
  $("import-review").addEventListener("click", e => {
    if (e.target.closest("#import-commit")) commitImport();
    if (e.target.closest("#import-discard")) resetImport();
  });
  $("import-review").addEventListener("change", async e => {
    if (e.target.name === "split-mode" || e.target.closest(".ratio-input")) updateImportSummary();
    const folderSelect = e.target.closest("[data-folder-split]");
    if (folderSelect && folderSelect.value === NEW_FOLDER) {
      const path = await addFolder(folderSelect.dataset.folderSplit);
      renderFolderSelects(path ? { [folderSelect.dataset.folderSplit]: path } : {});
    }
  });
  $("import-review").addEventListener("input", e => { if (e.target.closest(".ratio-input")) updateImportSummary(); });
}

/* ---------- dataset tab ---------- */

export function renderManageMeta() {
  const meta = state.meta;
  const caps = meta.capabilities;
  $("dataset-path").textContent = meta.yaml;
  $("dataset-path").title = meta.root;
  $("import-panel").hidden = !caps.image_ops;
  $("class-add-form").hidden = !caps.classes;
  $("class-note").textContent = meta.category === "classification"
    ? "Classes are the folders inside each split; ids follow the folder names alphabetically."
    : "Removing or reordering renumbers class ids in every label file. Ctrl+Z undoes any class change.";
  loadFolders();
  if (!importJob) resumeImportJob();
}

async function loadFolders() {
  try {
    folderInfo = await api("/folders");
  } catch { folderInfo = { items: [], editable: false }; }
  renderSplits();
}

function renderSplits() {
  const meta = state.meta;
  const configured = meta.capabilities.splits;
  const splits = [...new Set([...SPLITS.filter(split => configured.includes(split) || meta.split_counts[split]), ...Object.keys(meta.split_counts)])];
  const total = Math.max(1, meta.image_count);
  const editable = folderInfo.editable;
  $("split-overview").innerHTML = splits.map(split => {
    const count = meta.split_counts[split] || 0;
    const folders = folderInfo.items.filter(item => item.split === split);
    return `<div class="split-block">
      <div class="split-row"><span class="split-badge">${esc(split)}</span>
        <span class="stats-bar-track"><span class="stats-bar-fill" style="width:${count / total * 100}%"></span></span>
        <span class="stats-bar-value">${count.toLocaleString()} · ${Math.round(count / total * 100)}%</span></div>
      ${folders.map(folder => `<div class="folder-row" title="${esc(folder.path)}">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/></svg>
        <code>${esc(folder.path)}</code><span class="muted">${folder.images.toLocaleString()}${folder.exists ? "" : " · missing on disk"}</span></div>`).join("")}
      ${editable && SPLITS.includes(split) ? `<button class="link-btn" data-add-folder="${esc(split)}">＋ Add folder</button>` : ""}
    </div>`;
  }).join("") + (meta.capabilities.image_ops
    ? `<p class="panel-note">A split can use several folders. Select images in the Annotation tab to move them between splits and folders.</p>`
    : `<p class="panel-note">This dataset's splits are file lists, not plain folders, so images cannot be moved or imported here.</p>`);
}

function suggestFolder(split) {
  const taken = new Set(Object.values(state.meta.capabilities.folders || {}).flat());
  const base = state.meta.category === "classification" ? split : `images/${split}`;
  let index = 2;
  while (taken.has(`${base}_${index}`)) index++;
  return `${base}_${index}`;
}

// Ask for a folder path, create it for `split`, and return its path (or null when cancelled/failed).
async function addFolder(split) {
  const where = state.meta.category === "classification" ? "with one subfolder per class" : "labels go to the matching labels/ folder";
  const path = prompt(`New image folder for the ${split} split, relative to the dataset root (${where}):`, suggestFolder(split));
  if (!path || !path.trim()) return null;
  try {
    const result = await api("/folders", { method: "POST", body: JSON.stringify({ split, path: path.trim() }) });
    state.canUndo = result.can_undo;
    state.canRedo = result.can_redo;
    deps.toast(`Added ${result.path} to ${split} — data.yaml updated`);
    await deps.datasetChanged();
    return result.path;
  } catch (error) {
    deps.toast(error.message, "error");
    return null;
  }
}

export async function loadClasses() {
  if (!state.meta) return;
  try {
    const data = await api("/classes");
    classItems = data.items;
    renderClasses(data.reorderable);
  } catch (error) { deps.toast(error.message, "error"); }
}

function renderClasses(reorderable) {
  const classification = state.meta.category === "classification";
  const editable = state.meta.capabilities.classes;
  $("class-count").textContent = classItems.length ? `(${classItems.length})` : "";
  $("class-table").innerHTML = classItems.length ? `<table class="stats-table">
    <thead><tr>${reorderable && editable ? "<th></th>" : ""}<th class="left">Id</th><th class="left">Name</th><th>${classification ? "Images" : "Annotations"}</th>${classification ? "" : "<th>Images</th>"}<th></th></tr></thead>
    <tbody>${classItems.map((item, index) => `<tr>
      ${reorderable && editable ? `<td class="class-order">
        <button class="icon-btn mini" data-move="${item.id}" data-delta="-1" title="Move up" ${index === 0 ? "disabled" : ""}>↑</button>
        <button class="icon-btn mini" data-move="${item.id}" data-delta="1" title="Move down" ${index === classItems.length - 1 ? "disabled" : ""}>↓</button></td>` : ""}
      <td class="left"><span class="class-id"><span class="swatch" style="background:${classColor(item.id)}"></span>${item.id}</span></td>
      <td class="left"><input class="class-name" data-rename="${item.id}" data-original="${esc(item.name)}" value="${esc(item.name)}" ${editable ? "" : "disabled"} aria-label="Class name" spellcheck="false"></td>
      <td>${item.annotations.toLocaleString()}</td>${classification ? "" : `<td>${item.images.toLocaleString()}</td>`}
      <td>${editable ? `<button class="secondary mini" data-remove="${item.id}" title="Remove this class">Remove</button>` : ""}</td>
    </tr>`).join("")}</tbody></table>`
    : `<p class="empty-note">No classes yet. Add the classes you want to annotate, or import a dataset that ships with class names.</p>`;
}

async function classChange(request, message) {
  try {
    const result = await request();
    classItems = result.items;
    renderClasses(result.reorderable);
    state.canUndo = result.can_undo;
    state.canRedo = result.can_redo;
    if (message) deps.toast(typeof message === "function" ? message(result) : message);
    await deps.datasetChanged({ schema: true });
    return true;
  } catch (error) {
    deps.toast(error.message, "error");
    await loadClasses();
    return false;
  }
}

async function addClasses() {
  const names = $("class-add-name").value.split(",").map(name => name.trim()).filter(Boolean);
  if (!names.length) return;
  if (await classChange(() => api("/classes", { method: "POST", body: JSON.stringify({ names }) }), `Added ${plural(names.length, "class")}`))
    $("class-add-name").value = "";
}

function renameClass(id, input) {
  const name = input.value.trim();
  if (!name || name === input.dataset.original) { input.value = input.dataset.original; return; }
  classChange(() => api(`/classes/${id}`, { method: "PATCH", body: JSON.stringify({ name }) }), `Renamed to “${name}”`);
}

function moveClass(id, delta) {
  const order = classItems.map(item => item.id);
  const index = order.indexOf(id);
  const target = index + delta;
  if (index < 0 || target < 0 || target >= order.length) return;
  [order[index], order[target]] = [order[target], order[index]];
  classChange(() => api("/classes/reorder", { method: "POST", body: JSON.stringify({ order }) }), "Classes reordered — ids renumbered");
}

function openRemoveDialog(id) {
  const item = classItems.find(entry => entry.id === id);
  if (!item) return;
  removing = item;
  const classification = state.meta.category === "classification";
  const others = classItems.filter(entry => entry.id !== id);
  $("class-remove-title").textContent = `Remove class “${item.name}”?`;
  $("class-remove-usage").textContent = classification
    ? (item.images ? `${plural(item.images, "image")} in this class.` : "No images are in this class.")
    : (item.annotations ? `Used by ${plural(item.annotations, "annotation")} in ${plural(item.images, "image")}.` : "No annotations use this class.");
  $("class-remove-target").innerHTML = others.map(entry => `<option value="${entry.id}">${esc(entry.id)} · ${esc(entry.name)}</option>`).join("");
  const form = $("class-remove-form");
  form.querySelector('[value="merge"]').disabled = !others.length;
  const used = classification ? item.images : item.annotations;
  form.querySelector(`[value="${others.length && used ? "merge" : "delete"}"]`).checked = true;
  $("class-delete-label").textContent = classification ? "Delete its images" : "Delete its annotations";
  $("class-delete-hint").textContent = classification
    ? "Images are moved to the dataset's .yolo-workbench/trash folder."
    : "The annotations are removed from the label files.";
  updateRemoveDialog();
  $("class-remove-dialog").returnValue = "";
  $("class-remove-dialog").showModal();
}

function updateRemoveDialog() {
  if (!removing) return;
  const merge = $("class-remove-form").elements["remove-mode"].value === "merge";
  $("class-remove-target").disabled = !merge;
  const target = classItems.find(entry => entry.id === +$("class-remove-target").value);
  $("class-merge-hint").textContent = target ? `Its ${state.meta.category === "classification" ? "images move" : "annotations become"} “${target.name}”.` : "";
}

function removeClass() {
  const item = removing;
  if (!item) return;
  const mode = $("class-remove-form").elements["remove-mode"].value;
  const body = { mode, target_id: mode === "merge" ? +$("class-remove-target").value : null };
  classChange(() => api(`/classes/${item.id}/remove`, { method: "POST", body: JSON.stringify(body) }), `Removed “${item.name}” — Ctrl+Z undoes`);
}

async function rescan() {
  try {
    const result = await api("/rescan", { method: "POST" });
    const changes = [result.added && `${result.added} added`, result.removed && `${result.removed} removed`, result.updated && `${result.updated} labels re-read`].filter(Boolean);
    deps.toast(changes.length ? `Rescan: ${changes.join(", ")}` : "Rescan: nothing changed on disk");
    await deps.datasetChanged({ schema: result.names_changed });
  } catch (error) { deps.toast(error.message, "error"); }
}

/* ---------- import ---------- */

async function collectEntries(entries) {
  const files = [];
  const walk = async (entry, prefix) => {
    if (entry.isFile) {
      const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
      files.push({ file, path: `${prefix}${entry.name}` });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      // readEntries returns results in batches until it yields an empty one
      for (;;) {
        const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
        if (!batch.length) break;
        for (const child of batch) await walk(child, `${prefix}${entry.name}/`);
      }
    }
  };
  for (const entry of entries) await walk(entry, "");
  return files;
}

function showProgress(label, done, total, cancellable = true) {
  $("import-start").hidden = true;
  $("import-review").hidden = true;
  $("import-progress").hidden = false;
  $("import-progress-label").textContent = label;
  $("import-progress-count").textContent = total ? `${done.toLocaleString()} / ${total.toLocaleString()}` : "";
  $("import-progress-bar").max = Math.max(1, total);
  $("import-progress-bar").value = done;
  $("import-cancel").hidden = !cancellable;
}

async function uploadFiles(files) {
  const accepted = files.filter(item => UPLOAD_EXTENSIONS.test(item.path) && !item.path.split("/").some(part => part.startsWith(".")));
  if (!accepted.length) return deps.toast("No images or label files in the selection", "error");
  let staging;
  try {
    staging = (await api("/import/staging", { method: "POST" })).staging_id;
  } catch (error) { return deps.toast(error.message, "error"); }
  const abort = new AbortController();
  uploadAbort = abort;
  let done = 0, next = 0, failed = null;
  showProgress("Uploading…", 0, accepted.length);
  const worker = async () => {
    while (next < accepted.length && !failed && !abort.signal.aborted) {
      const item = accepted[next++];
      try {
        await api(`/import/staging/${staging}/files?path=${encodeURIComponent(item.path)}`, {
          method: "PUT", body: item.file, headers: { "Content-Type": "application/octet-stream" }, signal: abort.signal,
        });
      } catch (error) { failed = error; return; }
      showProgress("Uploading…", ++done, accepted.length);
    }
  };
  await Promise.all(Array.from({ length: UPLOAD_CONCURRENCY }, worker));
  uploadAbort = null;
  if (abort.signal.aborted || failed) {
    api(`/import/staging/${staging}`, { method: "DELETE" }).catch(() => {});
    resetImport(false);
    if (failed && failed.name !== "AbortError") deps.toast(`Upload failed: ${failed.message}`, "error");
    return;
  }
  scanImport({ staging_id: staging });
}

async function scanImport(source) {
  showProgress("Scanning…", 0, 0, false);
  try {
    importScan = await api("/import/scan", { method: "POST", body: JSON.stringify(source) });
    importSource = source;
    renderImportReview();
  } catch (error) {
    deps.toast(error.message, "error");
    resetImport();
  }
}

function targetOptions(item) {
  const classification = state.meta.category === "classification";
  const options = Object.entries(state.meta.names).map(([id, name]) =>
    `<option value="${esc(id)}" ${item.suggested === +id ? "selected" : ""}>${esc(id)} · ${esc(name)}</option>`);
  if (item.named) options.push(`<option value="new" ${item.suggested === "new" ? "selected" : ""}>＋ New class “${esc(item.name)}”</option>`);
  options.push(`<option value="" ${item.suggested == null ? "selected" : ""}>${classification ? "Skip these images" : "Drop these annotations"}</option>`);
  return options.join("");
}

function renderImportReview() {
  const scan = importScan;
  const classification = state.meta.category === "classification";
  const splitCounts = Object.entries(scan.splits);
  const targetSplits = scan.target_splits;
  const nameSource = classification ? null : scan.names_source;
  $("import-progress").hidden = true;
  $("import-start").hidden = true;
  $("import-review").hidden = false;
  $("import-review").innerHTML = `
    <p class="import-summary"><strong>${plural(scan.images, "image")}</strong> found${classification ? "" : ` · ${scan.labeled.toLocaleString()} with labels`}
      ${classification ? (scan.classes.some(item => item.named) ? " · classes from folder names" : "")
        : nameSource ? ` · class names from <code>${esc(nameSource)}</code>` : scan.classes.length ? " · no class names shipped: ids are kept as they are" : ""}</p>
    <fieldset class="import-fieldset"><legend>Splits</legend>
      <label class="radio-row"><input type="radio" name="split-mode" value="fixed" ${splitCounts.length ? "" : "checked"}><span>All into <select id="import-split">${targetSplits.map(split => `<option>${esc(split)}</option>`).join("")}</select></span></label>
      <label class="radio-row"><input type="radio" name="split-mode" value="auto"><span>Random split
        ${targetSplits.map(split => `<label class="ratio-input">${esc(split)} <input type="number" min="0" max="100" step="5" data-ratio="${esc(split)}" value="${{ train: 80, val: 20, test: 0 }[split] ?? 0}">%</label>`).join("")}
        <label class="ratio-input">seed <input type="number" id="import-seed" value="0" step="1"></label></span></label>
      ${splitCounts.length ? `<label class="radio-row"><input type="radio" name="split-mode" value="source" checked><span>Keep the source splits (${splitCounts.map(([split, count]) => `${esc(split)} ${count}`).join(" · ")}; the rest go to the split chosen above)</span></label>` : ""}
    </fieldset>
    <fieldset class="import-fieldset" id="import-folders"><legend>Destination folders</legend><div class="folder-targets"></div></fieldset>
    ${scan.classes.length ? `<fieldset class="import-fieldset"><legend>Classes</legend><table class="stats-table import-mapping">
      <thead><tr><th>Source</th><th>${classification ? "Images" : "Annotations"}</th><th>Becomes</th></tr></thead>
      <tbody>${scan.classes.map(item => `<tr>
        <td>${item.key === "" ? "<em>not in a class folder</em>" : item.named ? esc(item.name) : `id ${esc(item.key)}`}${!classification && item.named ? ` <span class="muted">(id ${esc(item.key)})</span>` : ""}</td>
        <td>${item.count.toLocaleString()}</td>
        <td>${item.named || item.key === "" ? `<select data-map="${esc(item.key)}">${targetOptions(item)}</select>`
          : `<span class="${state.meta.names[item.key] == null ? "warn-text" : "muted"}">kept as id ${esc(item.key)}${state.meta.names[item.key] == null ? " — unknown in this project" : ` (${esc(state.meta.names[item.key])})`}</span>`}</td>
      </tr>`).join("")}</tbody></table></fieldset>` : ""}
    <label class="toggle"><input type="checkbox" id="import-skip-duplicates" checked><span>Skip images that are already in the dataset (identical files)</span></label>
    <div class="modal-actions"><button id="import-discard" class="secondary" type="button">Cancel</button><button id="import-commit" type="button" class="primary-btn">Import</button></div>`;
  renderFolderSelects();
  updateImportSummary();
}

// One folder picker per split; keeps the current choices unless `chosen` overrides them.
function renderFolderSelects(chosen = {}) {
  const container = $("import-review").querySelector(".folder-targets");
  if (!container) return;
  const previous = {};
  container.querySelectorAll("[data-folder-split]").forEach(select => previous[select.dataset.folderSplit] = select.value);
  const folders = state.meta.capabilities.folders || {};
  const editable = state.meta.capabilities.folders_editable;
  container.innerHTML = importScan.target_splits.map(split => {
    const options = folders[split] || [];
    const selected = chosen[split] ?? (options.includes(previous[split]) ? previous[split] : options[0]);
    return `<label class="folder-target"><span class="split-badge">${esc(split)}</span>
      <select data-folder-split="${esc(split)}">${options.map(path => `<option value="${esc(path)}" ${path === selected ? "selected" : ""}>${esc(path)}</option>`).join("")}
      ${editable ? `<option value="${NEW_FOLDER}">＋ New folder…</option>` : ""}</select></label>`;
  }).join("");
}

function updateImportSummary() {
  const review = $("import-review");
  const mode = review.querySelector('[name="split-mode"]:checked')?.value;
  review.querySelectorAll(".ratio-input input").forEach(input => input.disabled = mode !== "auto");
  const button = $("import-commit");
  if (button) button.textContent = `Import ${plural(importScan.images, "image")}`;
}

async function commitImport() {
  const review = $("import-review");
  const mode = review.querySelector('[name="split-mode"]:checked').value;
  const ratios = {};
  review.querySelectorAll("[data-ratio]").forEach(input => ratios[input.dataset.ratio] = Math.max(0, +input.value || 0) / 100);
  const mapping = {};
  review.querySelectorAll("[data-map]").forEach(select => mapping[select.dataset.map] = select.value === "" ? null : select.value === "new" ? "new" : +select.value);
  const body = {
    ...importSource,
    split_mode: mode,
    split: $("import-split").value,
    ratios,
    seed: +$("import-seed").value || 0,
    mapping,
    folders: Object.fromEntries([...review.querySelectorAll("[data-folder-split]")].map(select => [select.dataset.folderSplit, select.value])),
    skip_duplicates: $("import-skip-duplicates").checked,
  };
  try {
    const job = await api("/import/commit", { method: "POST", body: JSON.stringify(body) });
    importSource = null;
    trackJob(job, Object.values(mapping).includes("new"));
  } catch (error) { deps.toast(error.message, "error"); }
}

function trackJob(job, schema) {
  importJob = job;
  showProgress(job.message || "Importing…", job.done, job.total);
  if (job.state === "running") {
    setTimeout(async () => {
      try { trackJob(await api(`/api/v1/jobs/${job.id}`), schema); }
      catch (error) { importJob = null; deps.toast(error.message, "error"); resetImport(false); }
    }, 600);
    return;
  }
  importJob = null;
  resetImport(false);
  const result = job.result;
  if (job.state === "done" && result) {
    const extra = [result.duplicates && `${result.duplicates} duplicates skipped`, result.renamed && `${result.renamed} renamed`, result.dropped && `${result.dropped} skipped`].filter(Boolean);
    deps.toast(`Imported ${plural(result.imported, "image")}${result.labels ? ` and ${plural(result.labels, "label")}` : ""}${extra.length ? ` · ${extra.join(", ")}` : ""}`);
  } else if (job.state === "cancelled") {
    deps.toast("Import cancelled — files copied so far were kept", "error");
  } else if (job.state === "failed") {
    deps.toast(`Import failed: ${job.error}`, "error");
  }
  deps.datasetChanged({ schema });
}

async function resumeImportJob() {
  if (!state.project) return;
  try {
    const { items } = await api(`/api/v1/jobs?project=${encodeURIComponent(state.project)}`);
    const running = items.find(job => job.kind === "import" && job.state === "running");
    if (running && !importJob) trackJob(running, true);
  } catch { /* job listing is best effort */ }
}

async function cancelImport() {
  if (uploadAbort) return uploadAbort.abort();
  if (!importJob) return;
  try { await api(`/api/v1/jobs/${importJob.id}/cancel`, { method: "POST" }); }
  catch (error) { deps.toast(error.message, "error"); }
}

function resetImport(discard = true) {
  if (discard && importSource?.staging_id) api(`/import/staging/${importSource.staging_id}`, { method: "DELETE" }).catch(() => {});
  importSource = null;
  importScan = null;
  $("import-review").hidden = true;
  $("import-review").innerHTML = "";
  $("import-progress").hidden = true;
  $("import-start").hidden = false;
}
