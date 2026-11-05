// The rebalance dialog (Dataset tab → Splits → Rebalance…): rules on the left, a live preview on the right
// (POST /rebalance/preview), then Apply (a snapshot first, then one undoable move) or Train with this split (a
// snapshot laid out by the plan; nothing moves). Also the Splits panel's leakage note.
import { $, api, debounce, esc } from "./api.js";
import { state } from "./state.js";
import { splitTable, trackJob } from "./snapshots.js";
import { newRunOn } from "./training.js";
let deps;

export function configureRebalance(dependencies) {
  deps = dependencies;
}

const SPLITS = ["train", "val", "test"];
const WEIGHTS = ["off", "low", "normal", "high"];
const PATTERNS = [
  ["", "None"],
  ["^(.*?)[_-]?\\d+$", "Everything before a trailing number (clip12_000345 → clip12)"],
  ["^(.*?)[_-]frame[_-]?\\d+", "Before “frame” and a number (cam1_frame_0042 → cam1)"],
  ["custom", "Custom regular expression…"],
];
const plural = (count, noun) => `${count.toLocaleString()} ${noun}${count === 1 ? "" : "s"}`;
let rules = null;
let info = null;       // GET /rebalance
let preview = null;    // the latest POST /rebalance/preview answer for `rules`
let previewToken = 0;
let busy = false;

export function bindRebalance() {
  $("rebalance-open").onclick = openDialog;
  const dialog = $("rebalance-dialog");
  dialog.addEventListener("input", e => { if (e.target.closest("[data-rule]")) readRules(); });
  dialog.addEventListener("change", e => {
    if (e.target.id === "rb-pattern-preset") {
      const custom = e.target.value === "custom";
      $("rb-pattern").hidden = !custom;
      if (!custom) $("rb-pattern").value = e.target.value;
    }
    if (e.target.closest("[data-rule]") || e.target.id === "rb-pattern-preset") readRules();
    if (e.target.matches("[data-folder-target]")) {
      rules.folders = { ...rules.folders, [e.target.dataset.folderTarget]: e.target.value || undefined };
      if (!e.target.value) delete rules.folders[e.target.dataset.folderTarget];
      schedulePreview();
    }
  });
  dialog.addEventListener("click", e => {
    if (e.target.closest("#rb-apply")) { e.preventDefault(); applyPlan(); }
    if (e.target.closest("#rb-train")) { e.preventDefault(); trainOnPlan(); }
    if (e.target.closest("#rb-defaults")) { e.preventDefault(); rules = structuredClone(info.defaults); renderRules(); schedulePreview(0); }
    if (e.target.closest("#rb-scan")) { e.preventDefault(); scanImages(); }
    const subset = e.target.closest("[data-subset]");
    if (subset) { e.preventDefault(); dialog.close(); deps.showImages(subset.dataset.subset, subset.dataset.label); }
  });
  $("splits-panel").addEventListener("click", e => {
    if (e.target.closest("[data-rebalance]")) openDialog();
    const subset = e.target.closest("[data-leak-subset]");
    if (subset) deps.showImages(subset.dataset.leakSubset, "Groups in more than one split");
  });
}

/* ---------- the Splits panel's leakage note ---------- */

export async function loadLeakage() {
  const note = $("split-leakage");
  if (!note) return;
  try {
    const leak = await api("/rebalance/leakage");
    note.hidden = !leak.configured || !leak.groups;
    note.innerHTML = leak.groups ? `${plural(leak.groups, "group")} of related images (by the rebalance's keep-together rules) are in more than one split: <button class="link-btn" data-leak-subset="${esc(leak.subset)}">show them</button> · <button class="link-btn" data-rebalance>rebalance…</button>` : "";
  } catch { note.hidden = true; }
}

/* ---------- the dialog ---------- */

async function openDialog(presetRules = null) {
  const dialog = $("rebalance-dialog");
  try {
    info = await api("/rebalance");
  } catch (error) { deps.toast(error.message, "error"); return; }
  if (!info.capabilities.image_ops) {
    deps.toast("This dataset's splits are file lists, not plain folders, so images cannot be moved between them", "error");
    return;
  }
  rules = structuredClone(presetRules && !(presetRules instanceof Event) ? presetRules : info.rules);
  preview = null;
  renderRules();
  $("rb-preview").innerHTML = `<p class="panel-note">Computing the preview…</p>`;
  dialog.showModal();
  schedulePreview(0);
}

const select = (rule, value, options) => `<select data-rule="${rule}">${options.map(([key, label]) => `<option value="${esc(key)}" ${String(value) === String(key) ? "selected" : ""}>${esc(label)}</option>`).join("")}</select>`;
const weight = (rule, value) => select(rule, value, WEIGHTS.map(key => [key, key]));
const depthOptions = [[0, "off"], [1, "1st folder below the split folder"], [2, "2 folders deep"], [3, "3 folders deep"]];

function renderRules() {
  const classification = state.meta.category === "classification";
  const counts = state.meta.split_counts;
  const pattern = rules.together.pattern || "";
  const preset = PATTERNS.some(([value]) => value === pattern) ? pattern : "custom";
  const folders = Object.entries(info.capabilities.folders).flatMap(([split, labels]) => labels.map(label => [split, label]));
  // batches (date / revision folders) matter once a split has more than one folder
  const batched = Object.values(info.capabilities.folders).some(labels => labels.length > 1);
  const eligible = rules.eligible || {};
  const pinned = rules.pinned || {};
  const scan = info.scan;
  $("rb-rules").innerHTML = `
    <fieldset class="import-fieldset"><legend>Targets</legend>
      <div class="rb-targets">${SPLITS.map(split => `<label class="rb-target"><span class="split-badge">${split}</span>
        <input type="number" min="0" max="100" step="1" data-rule="targets.${split}" value="${Math.round((rules.targets[split] || 0) * 100)}" aria-label="${split} share">%
        <label class="crop-check" title="Nothing leaves or joins a frozen split"><input type="checkbox" data-rule="frozen.${split}" ${rules.frozen_splits.includes(split) ? "checked" : ""}><span>frozen</span></label>
        <span class="muted">${(counts[split] || 0).toLocaleString()} now</span></label>`).join("")}</div>
      ${classification ? "" : `<label class="field rb-inline">Count shares in ${select("count", rules.count, [["images", "images"], ["objects", "objects"]])}</label>`}
    </fieldset>
    <fieldset class="import-fieldset"><legend>Keep together</legend>
      <p class="panel-note">Images that share one of these never end up in different splits.</p>
      <label class="field rb-inline">Subfolder ${select("together.depth", rules.together.depth, depthOptions)}</label>
      <label class="field rb-inline">File name <select id="rb-pattern-preset">${PATTERNS.map(([value, label]) => `<option value="${esc(value)}" ${preset === value ? "selected" : ""}>${esc(label)}</option>`).join("")}</select></label>
      <input id="rb-pattern" class="rb-pattern" type="text" data-rule="together.pattern" value="${esc(pattern)}" placeholder="^(.*)_\\d+$ — the first group names the group" spellcheck="false" ${preset === "custom" ? "" : "hidden"}>
      ${batched ? `<label class="crop-check"><input type="checkbox" data-rule="together.batch" ${rules.together.batch ? "checked" : ""}><span>each batch folder (date / revision) in one split — a time-based holdout</span></label>` : ""}
      <label class="crop-check"><input type="checkbox" data-rule="together.dup" ${rules.together.near_duplicates != null ? "checked" : ""}><span>near-duplicates, up to <input type="number" min="0" max="16" step="1" class="rb-small" data-rule="together.near_duplicates" value="${rules.together.near_duplicates ?? 4}"> bits apart</span></label>
      <label class="crop-check"><input type="checkbox" data-rule="together.burst" ${rules.together.bursts != null ? "checked" : ""}><span>shots from one camera within <input type="number" min="0" step="1" class="rb-small" data-rule="together.bursts" value="${rules.together.bursts ?? 2}"> s (EXIF time)</span></label>
      <p class="panel-note">${scan.scanned < scan.images ? `${plural(scan.images - scan.scanned, "image")} not scanned for near-duplicates and capture times. ` : "Every image is scanned. "}<button class="link-btn" id="rb-scan" type="button">${scan.scanned ? "Scan new images" : "Scan the images"}</button></p>
    </fieldset>
    <fieldset class="import-fieldset"><legend>Balance</legend>
      <div class="rb-balance">
        <label class="field">Images per class ${weight("balance.classes", rules.balance.classes)}</label>
        ${classification ? "" : `<label class="field">Objects per class ${weight("balance.objects", rules.balance.objects)}</label>
        <label class="field">Background images ${weight("balance.background", rules.balance.background)}</label>
        <label class="field">Object sizes ${weight("balance.sizes", rules.balance.sizes)}</label>`}
        ${batched ? `<label class="field" title="Each date / revision folder spread over the splits in proportion">Batches ${weight("balance.batches", rules.balance.batches)}</label>` : ""}
        <label class="field" title="Each subfolder spread over the splits in proportion (e.g. every camera in every split)">Spread subfolders ${weight("balance.spread", rules.balance.spread)}</label>
      </div>
      <label class="field rb-inline" ${rules.balance.spread === "off" ? "hidden" : ""}>Spread by ${select("spread.depth", rules.spread.depth || 1, depthOptions.slice(1))}</label>
    </fieldset>
    <fieldset class="import-fieldset"><legend>Limits</legend>
      <label class="field rb-inline">At least <input type="number" min="0" step="1" class="rb-small" data-rule="minimum" value="${rules.minimum}"> images of each class in val and test</label>
      <label class="crop-check"><input type="checkbox" data-rule="eligible.review" ${eligible.review === "done" ? "checked" : ""}><span>only reviewed images (Done) in val and test</span></label>
      <label class="crop-check"><input type="checkbox" data-rule="eligible.issue" ${eligible.issue === "none" ? "checked" : ""}><span>no images with validation issues in val and test</span></label>
      <label class="crop-check"><input type="checkbox" data-rule="pinned.review" ${pinned.review === "done" ? "checked" : ""}><span>reviewed images (Done) stay where they are</span></label>
      ${folders.length > 1 ? `<details class="rb-frozen-folders"${rules.frozen_folders.length ? " open" : ""}><summary>Folders that stay as they are${rules.frozen_folders.length ? ` (${rules.frozen_folders.length})` : ""}</summary>
        ${folders.map(([split, label]) => `<label class="crop-check"><input type="checkbox" data-rule="frozen_folder" value="${esc(label)}" ${rules.frozen_folders.includes(label) ? "checked" : ""}><span><span class="split-badge">${split}</span> <code>${esc(label)}</code></span></label>`).join("")}</details>` : ""}
    </fieldset>
    <fieldset class="import-fieldset"><legend>Stability</legend>
      <label class="radio-row"><input type="radio" name="rb-mode" value="minimal" data-rule="mode" ${rules.mode === "minimal" ? "checked" : ""}><span>Move as few images as possible <span class="muted">— val stays mostly the same, so runs before and after compare</span></span></label>
      <label class="radio-row"><input type="radio" name="rb-mode" value="reassign" data-rule="mode" ${rules.mode === "reassign" ? "checked" : ""}><span>Reassign every image that is not pinned</span></label>
      <div class="rb-inline-row"><label class="field rb-inline">At most <input type="number" min="0" step="1" class="rb-small" data-rule="max_moves" value="${rules.max_moves || ""}" placeholder="any"> moves</label>
      <label class="field rb-inline">Seed <input type="number" step="1" class="rb-small" data-rule="seed" value="${rules.seed}"></label></div>
      <button class="link-btn" id="rb-defaults" type="button">Back to the defaults</button>
    </fieldset>`;
}

function readRules() {
  const form = $("rb-rules");
  const value = name => form.querySelector(`[data-rule="${name}"]`);
  const number = (name, fallback = 0) => { const input = value(name); return input && input.value !== "" && Number.isFinite(+input.value) ? +input.value : fallback; };
  for (const split of SPLITS) rules.targets[split] = number(`targets.${split}`) / 100;
  rules.frozen_splits = SPLITS.filter(split => value(`frozen.${split}`)?.checked);
  if (value("count")) rules.count = value("count").value;
  rules.together.depth = number("together.depth");
  rules.together.pattern = $("rb-pattern").value.trim();
  if (value("together.batch")) rules.together.batch = value("together.batch").checked;
  rules.together.near_duplicates = value("together.dup").checked ? number("together.near_duplicates", 4) : null;
  rules.together.bursts = value("together.burst").checked ? number("together.bursts", 2) : null;
  for (const key of ["classes", "objects", "background", "sizes", "batches", "spread"]) if (value(`balance.${key}`)) rules.balance[key] = value(`balance.${key}`).value;
  rules.spread.depth = rules.balance.spread === "off" ? 0 : number("spread.depth", 1);
  const spreadField = value("spread.depth")?.closest("label");
  if (spreadField) spreadField.hidden = rules.balance.spread === "off";
  rules.minimum = number("minimum");
  const eligible = {};
  if (value("eligible.review").checked) eligible.review = "done";
  if (value("eligible.issue").checked) eligible.issue = "none";
  rules.eligible = Object.keys(eligible).length ? eligible : null;
  rules.pinned = value("pinned.review").checked ? { review: "done" } : null;
  rules.frozen_folders = [...form.querySelectorAll('[data-rule="frozen_folder"]:checked')].map(input => input.value);
  rules.mode = form.querySelector('[name="rb-mode"]:checked')?.value ?? "minimal";
  rules.max_moves = number("max_moves");
  rules.seed = number("seed");
  schedulePreview();
}

const schedulePreviewDebounced = debounce(() => runPreview(), 450);
function schedulePreview(delay) {
  preview = null;
  updateButtons();
  if (delay === 0) runPreview(); else schedulePreviewDebounced();
}

async function runPreview() {
  const token = ++previewToken;
  $("rb-preview").classList.add("loading");
  try {
    const result = await api("/rebalance/preview", { method: "POST", body: JSON.stringify({ rules }) });
    if (token !== previewToken) return;
    preview = result;
    renderPreview();
  } catch (error) {
    if (token !== previewToken) return;
    preview = null;
    $("rb-preview").innerHTML = `<p class="panel-note warn-text">${esc(error.message)}</p>`;
  } finally {
    if (token === previewToken) $("rb-preview").classList.remove("loading");
    updateButtons();
  }
}

function renderPreview() {
  const { counts, leakage, warnings, table } = preview;
  const names = Object.fromEntries(preview.names.map((name, id) => [id, name]));
  const target = Object.fromEntries(Object.entries(table.target).map(([split, row]) => [split, row]));
  const folderRows = preview.folders.map(item => {
    const key = `${item.from}|${item.split}`;
    const options = (info.capabilities.folders[item.split] || []).filter(label => label !== item.to);
    return `<tr><td><code>${esc(item.from ?? "—")}</code></td><td><span class="split-badge">${item.split}</span></td>
      <td><select data-folder-target="${esc(key)}"><option value="">${esc(item.to)}${item.create ? " (new)" : ""}</option>${options.map(label => `<option value="${esc(label)}" ${rules.folders[key] === label ? "selected" : ""}>${esc(label)}</option>`).join("")}</select></td>
      <td>${item.images.toLocaleString()}</td></tr>`;
  }).join("");
  const offTarget = SPLITS.filter(split => target[split]).map(split => {
    const key = preview.count === "objects" ? "objects" : "images";
    const now = table.after[split]?.[key] ?? 0;
    return `<span class="split-badge">${split}</span> ${now.toLocaleString()} <span class="muted">/ ${Math.round(target[split][key]).toLocaleString()}</span>`;
  }).join(" · ");
  $("rb-preview").innerHTML = `
    <p class="rb-summary"><strong>${counts.moves ? plural(counts.moves, "image") + " to move" : "Nothing to move"}</strong>${counts.moves && preview.subset ? ` · <button class="link-btn" data-subset="${esc(preview.subset)}" data-label="Moved by the rebalance (${counts.moves})">show them</button>` : ""}</p>
    <p class="panel-note">${offTarget}${counts.groups ? ` · ${plural(counts.groups, "group")} kept together (${counts.grouped_images.toLocaleString()} images)` : ""}${leakage.before ? ` · groups in more than one split: ${leakage.before} → <strong>${leakage.after}</strong>` : ""}${counts.frozen ? ` · ${plural(counts.frozen, "image")} in frozen splits` : ""}</p>
    ${warnings.length ? `<ul class="check-list">${warnings.map(text => `<li class="check-warning">${esc(text)}</li>`).join("")}</ul>` : ""}
    <div class="split-table-wrap">${splitTable(table.before, table.after, names)}</div>
    <p class="panel-note">Before → after, per split. The class rows count images that contain the class.</p>
    ${folderRows ? `<h3 class="rb-subtitle">Where moved images go</h3><p class="panel-note">Each image goes to its batch's folder in the new split (made when there is none); the path below the split folder stays.</p>
      <table class="split-table rb-folders"><thead><tr><th>From</th><th>To</th><th>Folder</th><th>Images</th></tr></thead><tbody>${folderRows}</tbody></table>` : ""}`;
}

function updateButtons() {
  const ready = !!preview && !busy;
  $("rb-apply").disabled = !ready || !preview.counts.moves;
  $("rb-train").disabled = !ready || !preview.counts.moves || state.meta.category === "classification";
  $("rb-train").title = state.meta.category === "classification" ? "A classification dataset's splits are folders: apply the split to train on it" : "Train on a snapshot laid out by this plan; nothing moves in the dataset";
}

async function applyPlan() {
  if (!preview) return;
  const created = preview.create.length ? ` and ${plural(preview.create.length, "new folder")}` : "";
  if (!confirm(`Move ${plural(preview.counts.moves, "image")}${created}? A snapshot of the dataset is taken first, and one Ctrl+Z puts every image back.`)) return;
  busy = true;
  updateButtons();
  try {
    const job = await api("/rebalance/apply", { method: "POST", body: JSON.stringify({ rules, digest: preview.digest }) });
    $("rebalance-dialog").close();
    deps.showTab("dataset");
    trackJob(job, result => {
      deps.toast(`Moved ${plural(result.moved, "image")}${result.created.length ? ` · ${plural(result.created.length, "folder")} added` : ""} · snapshot “${result.snapshot.name}” has the splits before — Ctrl+Z undoes`);
      deps.datasetChanged();
      deps.loadSnapshots();
    });
  } catch (error) {
    deps.toast(error.message, "error");
    if (error.status === 409) schedulePreview(0);
  } finally {
    busy = false;
    updateButtons();
  }
}

async function trainOnPlan() {
  if (!preview) return;
  busy = true;
  updateButtons();
  try {
    const job = await api("/rebalance/snapshot", { method: "POST", body: JSON.stringify({ rules, digest: preview.digest }) });
    $("rebalance-dialog").close();
    deps.showTab("dataset");
    trackJob(job, result => {
      deps.toast(`Snapshot “${result.name}” has the proposed split; nothing moved in the dataset`);
      newRunOn(result.id);
    });
  } catch (error) {
    deps.toast(error.message, "error");
    if (error.status === 409) schedulePreview(0);
  } finally {
    busy = false;
    updateButtons();
  }
}

async function scanImages() {
  try {
    const job = await api("/rebalance/scan", { method: "POST" });
    const button = $("rb-scan");
    const poll = async current => {
      if (current.state === "running") {
        if (button) button.textContent = `Scanning… ${current.done.toLocaleString()} / ${current.total.toLocaleString()}`;
        setTimeout(async () => poll(await api(`/api/v1/jobs/${current.id}`)), 700);
        return;
      }
      if (current.state === "failed") deps.toast(`Scan failed: ${current.error}`, "error");
      info = await api("/rebalance");
      if ($("rebalance-dialog").open) { renderRules(); schedulePreview(0); }
    };
    poll(job);
  } catch (error) { deps.toast(error.message, "error"); }
}

// "Apply this split" of a proposed-split snapshot: the dialog with that snapshot's rules
export function openWithRules(snapshotRules) {
  openDialog(snapshotRules);
}
