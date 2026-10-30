// New-run form of the Training tab. One args object is the source of truth: GUI fields edit it
// (a value equal to the framework's default is dropped, so only real choices are kept), and the
// raw YAML mode round-trips through the server's validator (POST /training/args/validate).
// The framework (Ultralytics YOLO or RF-DETR) follows `model:` — an RF-DETR variant such as
// rfdetr-medium — and decides the fields, the defaults and the model picker. The Training data panel
// limits the run to some of the dataset's split folders (not part of the args: sent as `folders`).
import { $, api, esc, debounce } from "./api.js";
import { state } from "./state.js";
import { augmentationArgChanged, augmentationHtml, mountAugmentations } from "./augments.js";
import { mountRfAugmentations, rfAugmentationArgChanged, rfAugmentationHtml } from "./rfaugments.js";

let deps;
export function configureRunForm(dependencies) {
  deps = dependencies;
}

const FAMILIES = [["yolo26", "YOLO26"], ["yolo11", "YOLO11"], ["yolov8", "YOLOv8"]];
const SIZES = [["n", "nano"], ["s", "small"], ["m", "medium"], ["l", "large"], ["x", "extra large"]];
const SUFFIX = { detection: "", segmentation: "-seg", classification: "-cls" };
const PRETRAINED = /^(yolo(?:v8|11|26))([nsmlx])(-seg|-cls|-pose|-obb)?\.pt$/;
const DET_SEG = ["detection", "segmentation"];
const CLS = ["classification"];
const RFDETR_MODEL = /^rfdetr-/;
const LABELS = { ultralytics: "Ultralytics YOLO", rfdetr: "RF-DETR" };

const GROUPS = {
  ultralytics: [
    { id: "basics", title: "Basics", open: true },
    { id: "optimizer", title: "Optimizer" },
    { id: "augmentation", title: "Augmentation" },
    { id: "advanced", title: "Advanced" },
  ],
  rfdetr: [
    { id: "basics", title: "Basics", open: true },
    { id: "optimizer", title: "Optimizer" },
    { id: "augmentation", title: "Augmentation" },
    { id: "stopping", title: "Stopping & checkpoints" },
    { id: "advanced", title: "Advanced" },
  ],
};
// `cats` limits a field to dataset categories, `fw` to a framework (default Ultralytics); everything else is reachable in raw YAML.
// The augmentation group renders as preview cards (augments.js; RF-DETR's aug_config: rfaugments.js); its entries here keep
// counts and raw YAML in step.
const FIELDS = [
  { group: "basics", key: "epochs", label: "Epochs", type: "int", min: 1 },
  { group: "basics", key: "imgsz", label: "Image size", type: "int", min: 32, step: 32 },
  { group: "basics", key: "batch", label: "Batch", type: "float", hint: "-1 = AutoBatch · 0–1 = share of GPU memory" },
  { group: "basics", key: "device", label: "Device", type: "device" },
  { group: "basics", key: "patience", label: "Patience", type: "int", min: 0, hint: "Epochs without improvement before stopping" },
  { group: "basics", key: "cache", label: "Cache images", type: "select", options: [[false, "Off"], ["ram", "RAM"]] },
  { group: "basics", key: "seed", label: "Seed", type: "int" },

  { group: "optimizer", key: "optimizer", label: "Optimizer", type: "select", options: ["auto", "SGD", "MuSGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"].map(name => [name, name]) },
  { group: "optimizer", key: "lr0", label: "Initial LR (lr0)", type: "float", min: 0, hint: "Ignored when optimizer is auto" },
  { group: "optimizer", key: "lrf", label: "Final LR factor (lrf)", type: "float", min: 0 },
  { group: "optimizer", key: "momentum", label: "Momentum", type: "float", min: 0, max: 1 },
  { group: "optimizer", key: "weight_decay", label: "Weight decay", type: "float", min: 0 },
  { group: "optimizer", key: "warmup_epochs", label: "Warmup epochs", type: "float", min: 0 },
  { group: "optimizer", key: "close_mosaic", label: "Close mosaic (last N epochs)", type: "int", min: 0, cats: DET_SEG },
  { group: "optimizer", key: "cos_lr", label: "Cosine LR schedule", type: "bool" },
  { group: "optimizer", key: "amp", label: "Mixed precision (AMP)", type: "bool" },

  { group: "augmentation", key: "hsv_h", label: "Hue (hsv_h)", type: "float", min: 0, max: 1 },
  { group: "augmentation", key: "hsv_s", label: "Saturation (hsv_s)", type: "float", min: 0, max: 1 },
  { group: "augmentation", key: "hsv_v", label: "Brightness (hsv_v)", type: "float", min: 0, max: 1 },
  { group: "augmentation", key: "degrees", label: "Rotation (degrees)", type: "float", min: 0, max: 180, cats: DET_SEG },
  { group: "augmentation", key: "translate", label: "Translate", type: "float", min: 0, max: 1, cats: DET_SEG },
  { group: "augmentation", key: "scale", label: "Scale", type: "float", min: 0 },
  { group: "augmentation", key: "shear", label: "Shear (degrees)", type: "float", cats: DET_SEG },
  { group: "augmentation", key: "perspective", label: "Perspective", type: "float", min: 0, max: 0.001, cats: DET_SEG },
  { group: "augmentation", key: "flipud", label: "Flip up-down", type: "float", min: 0, max: 1 },
  { group: "augmentation", key: "fliplr", label: "Flip left-right", type: "float", min: 0, max: 1 },
  { group: "augmentation", key: "bgr", label: "BGR swap", type: "float", min: 0, max: 1, cats: DET_SEG },
  { group: "augmentation", key: "mosaic", label: "Mosaic", type: "float", min: 0, max: 1, cats: DET_SEG },
  { group: "augmentation", key: "mixup", label: "MixUp", type: "float", min: 0, max: 1, cats: DET_SEG },
  { group: "augmentation", key: "cutmix", label: "CutMix", type: "float", min: 0, max: 1, cats: DET_SEG },
  { group: "augmentation", key: "copy_paste", label: "Copy-paste", type: "float", min: 0, max: 1, cats: ["segmentation"] },
  { group: "augmentation", key: "copy_paste_mode", label: "Copy-paste mode", type: "select", options: [["flip", "flip"], ["mixup", "mixup"]], cats: ["segmentation"] },
  { group: "augmentation", key: "erasing", label: "Random erasing", type: "float", min: 0, max: 1, cats: CLS },
  { group: "augmentation", key: "auto_augment", label: "Auto augment", type: "select", options: [["randaugment", "RandAugment"], ["autoaugment", "AutoAugment"], ["augmix", "AugMix"], [null, "Off"]], cats: CLS },

  { group: "advanced", key: "workers", label: "Dataloader workers", type: "int", min: 0 },
  { group: "advanced", key: "fraction", label: "Dataset fraction", type: "float", min: 0, max: 1 },
  { group: "advanced", key: "freeze", label: "Freeze first N layers", type: "int", min: 0 },
  { group: "advanced", key: "multi_scale", label: "Multi-scale range", type: "float", min: 0, max: 1, cats: DET_SEG },
  { group: "advanced", key: "dropout", label: "Dropout", type: "float", min: 0, max: 1, cats: CLS },
  { group: "advanced", key: "time", label: "Time limit (hours)", type: "float", min: 0, hint: "Overrides epochs" },
  { group: "advanced", key: "save_period", label: "Save checkpoint every N epochs", type: "int" },
  { group: "advanced", key: "rect", label: "Rectangular batches", type: "bool", cats: DET_SEG },
  { group: "advanced", key: "single_cls", label: "Single class", type: "bool", cats: DET_SEG },
  { group: "advanced", key: "deterministic", label: "Deterministic", type: "bool" },
  { group: "advanced", key: "val", label: "Validate every epoch", type: "bool" },
  { group: "advanced", key: "plots", label: "Save plots", type: "bool" },

  { fw: "rfdetr", group: "basics", key: "epochs", label: "Epochs", type: "int", min: 1 },
  { fw: "rfdetr", group: "basics", key: "batch_size", label: "Batch", type: "batch", hint: "auto = the largest that fits (CUDA only)" },
  { fw: "rfdetr", group: "basics", key: "grad_accum_steps", label: "Gradient accumulation", type: "int", min: 1, hint: "Effective batch = batch × this" },
  { fw: "rfdetr", group: "basics", key: "resolution", label: "Resolution", type: "int", min: 12 },
  { fw: "rfdetr", group: "basics", key: "device", label: "Device", type: "device" },
  { fw: "rfdetr", group: "basics", key: "num_workers", label: "Dataloader workers", type: "int", min: 0 },
  { fw: "rfdetr", group: "basics", key: "seed", label: "Seed", type: "int" },

  { fw: "rfdetr", group: "optimizer", key: "lr", label: "Learning rate", type: "float", min: 0 },
  { fw: "rfdetr", group: "optimizer", key: "lr_encoder", label: "Backbone learning rate", type: "float", min: 0 },
  { fw: "rfdetr", group: "optimizer", key: "weight_decay", label: "Weight decay", type: "float", min: 0 },
  { fw: "rfdetr", group: "optimizer", key: "warmup_epochs", label: "Warmup epochs", type: "float", min: 0 },
  { fw: "rfdetr", group: "optimizer", key: "lr_scheduler", label: "LR schedule", type: "select", options: [["step", "Step"], ["cosine", "Cosine"]] },
  { fw: "rfdetr", group: "optimizer", key: "clip_max_norm", label: "Gradient clipping", type: "float", min: 0 },
  { fw: "rfdetr", group: "optimizer", key: "drop_path", label: "Drop path", type: "float", min: 0, max: 1 },
  { fw: "rfdetr", group: "optimizer", key: "use_ema", label: "EMA weights", type: "bool" },

  { fw: "rfdetr", group: "augmentation", key: "aug_config", label: "Augmentations", type: "aug" },
  { fw: "rfdetr", group: "augmentation", key: "augmentation_backend", label: "Augment on", type: "select", options: [["cpu", "CPU (Albumentations)"], ["auto", "GPU when available (Kornia)"], ["kornia", "GPU (Kornia)"], ["torchvision", "CPU (torchvision)"]], hint: "Kornia runs only some transforms" },
  { fw: "rfdetr", group: "augmentation", key: "save_dataset_grids", label: "Save augmented batch grids", type: "bool" },

  { fw: "rfdetr", group: "stopping", key: "early_stopping", label: "Early stopping", type: "bool" },
  { fw: "rfdetr", group: "stopping", key: "early_stopping_patience", label: "Patience", type: "int", min: 0, hint: "Epochs without improvement before stopping" },
  { fw: "rfdetr", group: "stopping", key: "early_stopping_min_delta", label: "Minimum improvement", type: "float", min: 0 },
  { fw: "rfdetr", group: "stopping", key: "skip_best_epochs", label: "Ignore first N epochs", type: "int", min: 0, hint: "For the best weights and early stopping" },
  { fw: "rfdetr", group: "stopping", key: "eval_interval", label: "Validate every N epochs", type: "int", min: 1 },
  { fw: "rfdetr", group: "stopping", key: "checkpoint_interval", label: "Full checkpoint every N epochs", type: "int", min: 1, empty: "off", hint: "Each is weights + optimizer state (hundreds of MB); last.ckpt is always kept" },
  { fw: "rfdetr", group: "stopping", key: "best_model_metric", label: "Best weights by", type: "select", options: [["map", "mAP"], ["mar", "mAR"]] },

  { fw: "rfdetr", group: "advanced", key: "amp_dtype", label: "Mixed precision", type: "select", options: [["auto", "Auto"], ["bf16", "bf16"], ["fp16", "fp16"], ["fp8", "fp8 (NVIDIA)"], [null, "Off (FP32)"]] },
  { fw: "rfdetr", group: "advanced", key: "multi_scale", label: "Multi-scale", type: "select", options: [["per-batch", "Per batch"], ["per-sample", "Per sample"], ["off", "Off"]] },
  { fw: "rfdetr", group: "advanced", key: "expanded_scales", label: "Wider scale range", type: "bool" },
  { fw: "rfdetr", group: "advanced", key: "eval_max_dets", label: "Detections per image (eval)", type: "int", min: 1 },
  { fw: "rfdetr", group: "advanced", key: "eval_masks_head_resolution", label: "Validate masks at mask-head resolution", type: "bool", cats: ["segmentation"] },
];

let options = null;   // GET /training/options
let args = {};        // the run's arguments (non-default values + model)
let mode = "gui";     // "gui" | "raw"
let modelSource = "pretrained";
let folders = null;   // {split: [folder]} the run trains on; null = the whole dataset
let check = { errors: [], warnings: [], notes: [], splits: null, argErrors: [] };
let checkToken = 0;
let queueAhead = 0;

const category = () => options.category;
const frameworkOf = values => RFDETR_MODEL.test(String(values?.model ?? "")) ? "rfdetr" : "ultralytics";
const framework = () => frameworkOf(args);
const fwOptions = (id = framework()) => options.frameworks?.find(item => item.id === id) ?? { id, label: LABELS[id], available: id === "ultralytics", defaults: options.defaults, help: options.help, pretrained: options.pretrained };
const defaults = () => fwOptions().defaults ?? {};
const groups = () => GROUPS[framework()];
const fields = () => FIELDS.filter(field => (field.fw ?? "ultralytics") === framework() && (!field.cats || field.cats.includes(category())));
const runsOf = id => options.runs.filter(run => (run.framework ?? "ultralytics") === id);
// the RF-DETR variant the args name (its default resolution and the resolution step)
const variantInfo = () => framework() === "rfdetr" ? fwOptions().pretrained?.find(item => item.name === args.model) : null;
// Ultralytics args keep the original key; RF-DETR's get their own
const storageKey = (id = "ultralytics") => `yolo-workbench:train-args:${state.project ?? state.meta.yaml}${id === "ultralytics" ? "" : `:${id}`}`;
const frameworkKey = () => `yolo-workbench:train-framework:${state.project ?? state.meta.yaml}`;
const argsByFramework = {};  // the other framework's args while the form is open, so switching back keeps them
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export async function loadTrainingOptions() {
  options = await api("/training/options");
  return options;
}
export const trainingOptions = () => options;

// ---------- YAML (GUI → raw); raw → GUI goes through the server ----------

const PLAIN = /^[A-Za-z_][\w.\-/]*$/;
const RESERVED = /^(true|false|yes|no|on|off|null|~)$/i;
const yamlKey = key => PLAIN.test(key) && !RESERVED.test(key) ? key : JSON.stringify(key);
// flow style for what nests: `{p: 0.5, limit: [-15, 15]}`
function yamlValue(value) {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean" || typeof value === "number") return String(value);
  if (Array.isArray(value)) return `[${value.map(yamlValue).join(", ")}]`;
  if (typeof value === "object") return `{${Object.entries(value).map(([key, item]) => `${yamlKey(key)}: ${yamlValue(item)}`).join(", ")}}`;
  const text = String(value);
  return PLAIN.test(text) && !RESERVED.test(text) ? text : JSON.stringify(text);
}
// a mapping (aug_config) gets one indented line per entry
const yamlLine = (key, value) => value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length
  ? `${key}:\n${Object.entries(value).map(([name, item]) => `  ${yamlKey(name)}: ${yamlValue(item)}`).join("\n")}`
  : `${key}: ${yamlValue(value)}`;
export function argsToYaml(values) {
  return Object.entries(values).map(([key, value]) => yamlLine(key, value)).join("\n") + (Object.keys(values).length ? "\n" : "");
}

// ---------- open ----------

/**
 * @param {{from?: "duplicate"|"finetune", runId?: string}} source  prefill from a run, else the last-used args
 */
export async function openRunForm(source = {}) {
  const container = $("run-detail");
  container.innerHTML = `<div class="empty">Loading training options…</div>`;
  try {
    const [loaded, queue] = await Promise.all([loadTrainingOptions(), api("/api/v1/training/queue")]);
    queueAhead = (queue.active ? 1 : 0) + queue.queued.length;
    let name = loaded.next_name;
    let notes = "";
    Object.keys(argsByFramework).forEach(key => delete argsByFramework[key]);
    folders = null;
    if (source.runId) {
      const copied = await api(`/training/runs/${encodeURIComponent(source.runId)}/args`);
      args = { ...copied.args };
      folders = fitFolders(copied.folders);
      name = copied.name;
      if (source.from === "finetune") {
        const run = loaded.runs.find(item => item.id === source.runId);
        const weights = run?.weights.best ?? run?.weights.last;
        if (weights && run.framework === "rfdetr") {
          args = { model: run.variant ?? args.model, pretrain_weights: weights, ...withoutKey(withoutKey(args, "model"), "pretrain_weights") };
        } else if (weights) {
          args = { model: weights, ...withoutKey(args, "model") };
        }
        name = `${copied.name.replace(/-\d+$/, "")}-ft`;
        notes = `Fine-tuned from ${run?.name ?? "a previous run"}`;
      }
    } else {
      const last = readLastFramework();
      args = readLastArgs(last) ?? { model: defaultModel(last) };
    }
    if (!args.model) args = { model: defaultModel(framework()), ...args };
    mode = "gui";
    modelSource = detectModelSource(args);
    renderForm(name, notes, source);
    scheduleCheck(0);
  } catch (error) {
    container.innerHTML = `<div class="empty">${esc(error.message)}</div>`;
  }
}

const withoutKey = (object, key) => Object.fromEntries(Object.entries(object).filter(([name]) => name !== key));
const defaultModel = (id = "ultralytics") => id === "rfdetr" ? fwOptions("rfdetr").pretrained?.[0]?.name ?? "rfdetr-nano" : `${FAMILIES[0][0]}n${SUFFIX[category()]}.pt`;
const usable = id => fwOptions(id).available && options.frameworks?.some(item => item.id === id) !== false;

function readLastFramework() {
  try {
    const saved = localStorage.getItem(frameworkKey());
    return saved && usable(saved) ? saved : "ultralytics";
  } catch { return "ultralytics"; }
}
function readLastArgs(id = "ultralytics") {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey(id)) || "null");
    return saved && typeof saved === "object" && !Array.isArray(saved) && frameworkOf(saved) === id ? saved : null;
  } catch { return null; }
}
function saveLastArgs() {
  try {
    localStorage.setItem(storageKey(framework()), JSON.stringify(args));
    localStorage.setItem(frameworkKey(), framework());
  } catch { /* a convenience only */ }
}

function detectModelSource(values) {
  if (frameworkOf(values) === "rfdetr") {
    if (values.pretrain_weights) return runsOf("rfdetr").some(run => Object.values(run.weights).includes(values.pretrain_weights)) ? "run" : "custom";
    return fwOptions("rfdetr").pretrained?.some(item => item.name === values.model) ? "pretrained" : "custom";
  }
  const model = values.model;
  if (PRETRAINED.test(model || "")) return "pretrained";
  if (runsOf("ultralytics").some(run => Object.values(run.weights).includes(model))) return "run";
  return "custom";
}

// ---------- render ----------

function renderForm(name, notes, source) {
  const title = source.from === "duplicate" ? "New run (duplicated)" : source.from === "finetune" ? "New run (fine-tune)" : "New run";
  $("run-detail").innerHTML = `
    <form id="run-form" class="run-form" autocomplete="off" novalidate>
      <div class="run-head">
        <div class="run-title"><h1>${esc(title)}</h1><p class="section-subtitle" id="run-form-subtitle">${subtitleText()}</p></div>
        <div class="segmented" role="radiogroup" aria-label="Editor">
          <label><input type="radio" name="form-mode" value="gui" checked><span>Form</span></label>
          <label><input type="radio" name="form-mode" value="raw"><span>Raw YAML</span></label>
        </div>
      </div>
      <div class="stats-panel form-panel">
        <div class="form-grid form-identity">
          <label class="field">Name<input id="run-name" type="text" value="${esc(name)}" maxlength="80" required></label>
          <label class="field grow-2">Notes<input id="run-notes" type="text" value="${esc(notes)}" placeholder="What this experiment tries"></label>
        </div>
      </div>
      <div id="form-data" class="stats-panel form-panel"></div>
      <div id="form-gui"></div>
      <div id="form-raw" hidden>
        <div class="stats-panel form-panel">
          <div class="panel-head"><h2>Arguments</h2><span class="muted" id="raw-help">${rawHelp()}</span></div>
          <textarea id="raw-args" class="raw-args" spellcheck="false" rows="16" aria-label="Training arguments (YAML)"></textarea>
        </div>
      </div>
      <div id="form-check" class="stats-panel form-panel form-check" aria-live="polite"></div>
      <div class="form-actions">
        <button type="button" class="secondary" id="run-form-cancel">Cancel</button>
        <button type="submit" class="primary-btn" id="run-form-submit">Start</button>
      </div>
    </form>`;
  renderFolders();
  renderGui();
  const form = $("run-form");
  form.addEventListener("submit", e => { e.preventDefault(); submit(); });
  form.addEventListener("change", onChange);
  form.addEventListener("input", onInput);
  form.addEventListener("click", onClick);
  $("run-form-cancel").onclick = () => deps.closeForm();
  updateSubmit();
}

function subtitleText() {
  const engine = framework() === "rfdetr" ? "RF-DETR" : `Ultralytics ${esc(options.task)}`;
  return `${esc(category())} · ${engine} · values left empty use the ${framework() === "rfdetr" ? "RF-DETR" : "Ultralytics"} default (shown greyed)`;
}
const rawHelp = () => framework() === "rfdetr"
  ? "One <code>key: value</code> per line: RF-DETR's training arguments, plus <code>model</code> (the variant), <code>pretrain_weights</code>, <code>resolution</code> and <code>device</code>. <code>dataset_dir</code> and <code>output_dir</code> are set by the workbench."
  : "One <code>key: value</code> per line, as in <code>args.yaml</code>. <code>data</code>, <code>project</code> and <code>name</code> are set by the workbench.";

// ---------- training data: the split folders the run trains on ----------

const SPLITS = ["train", "val", "test"];
const folderItems = () => (options.folders ?? []).filter(item => SPLITS.includes(item.split));
const folderChosen = item => folders === null || (folders[item.split] ?? []).includes(item.path);
const foldersSummary = () => folders === null
  ? "All folders"
  : `${folderItems().filter(folderChosen).length} of ${folderItems().length} folders <button type="button" class="link-btn" data-all-folders>Use all</button>`;

// a copied run's folders that still exist; all of them → null (the whole dataset)
function fitFolders(selection) {
  if (!selection) return null;
  const kept = {};
  for (const item of folderItems()) if ((selection[item.split] ?? []).includes(item.path)) (kept[item.split] ??= []).push(item.path);
  return folderItems().every(item => (kept[item.split] ?? []).includes(item.path)) ? null : kept;
}

function renderFolders() {
  const el = $("form-data");
  const items = folderItems();
  // with one folder per split there is nothing to choose (unless a copied run used a subset)
  el.hidden = folders === null && !SPLITS.some(split => items.filter(item => item.split === split).length > 1);
  if (el.hidden) { el.innerHTML = ""; return; }
  el.innerHTML = `
    <div class="panel-head"><h2>Training data</h2><span class="muted" id="folders-summary">${foldersSummary()}</span></div>
    <div class="train-folders">
      ${SPLITS.filter(split => items.some(item => item.split === split)).map(split => `
        <div class="train-folder-split"><span class="split-badge">${split}</span>
          ${items.filter(item => item.split === split).map(item => `<label class="toggle train-folder" title="${esc(item.path)}">
            <input type="checkbox" data-train-split="${esc(split)}" data-train-folder="${esc(item.path)}" ${folderChosen(item) ? "checked" : ""}>
            <code>${esc(item.path)}</code><span class="muted">${item.images.toLocaleString()}</span></label>`).join("")}
        </div>`).join("")}
    </div>
    <p class="panel-note">The run trains, validates and tests only on the images in the ticked folders.</p>`;
}

function readFolders() {
  const boxes = [...$("form-data").querySelectorAll("input[data-train-folder]")];
  if (boxes.every(box => box.checked)) return null;
  const selection = {};
  for (const box of boxes) if (box.checked) (selection[box.dataset.trainSplit] ??= []).push(box.dataset.trainFolder);
  return selection;
}

function renderGui() {
  const changed = group => fields().filter(field => field.group === group && field.key in args).length;
  if ($("run-form-subtitle")) $("run-form-subtitle").innerHTML = subtitleText();
  if ($("raw-help")) $("raw-help").innerHTML = rawHelp();
  $("form-gui").innerHTML = `
    <div class="stats-panel form-panel">
      <div class="panel-head"><h2>Model</h2>${frameworkSwitchHtml()}</div>
      ${modelHtml()}
    </div>
    ${groups().map(group => `
      <details class="stats-panel form-panel form-group" data-group="${group.id}" ${group.open || changed(group.id) ? "open" : ""}>
        <summary><h2>${esc(group.title)}</h2><span class="changed-count" data-changed="${group.id}">${changed(group.id) ? `${changed(group.id)} changed` : ""}</span></summary>
        ${group.id !== "augmentation" ? `<div class="form-grid">${fields().filter(field => field.group === group.id).map(fieldHtml).join("")}</div>`
          : framework() === "rfdetr" ? rfAugmentationHtml(rfAugmentationContext()) : augmentationHtml(augmentationContext())}
      </details>`).join("")}
    ${extrasHtml()}`;
  const augmentation = $("form-gui").querySelector('[data-group="augmentation"]');
  if (augmentation && framework() === "rfdetr") mountRfAugmentations(augmentation);
  else if (augmentation) mountAugmentations(augmentation);
}

const augmentationContext = () => ({
  category: category(),
  defaults: defaults(),
  help: fwOptions().help ?? {},
  fields: fields(),
  args: () => args,
  setArg,
  changed: () => { updateChangedCounts(); scheduleCheck(); },
});

const rfAugmentationContext = () => ({
  category: category(),
  args: () => args,
  setArg,
  resolution: () => args.resolution ?? variantInfo()?.resolution ?? 640,
  changed: () => { updateChangedCounts(); scheduleCheck(); },
  // the group's other fields (where augmentations run, batch grids), below the cards
  extraHtml: () => `<div class="form-grid">${fields().filter(field => field.group === "augmentation" && field.type !== "aug").map(fieldHtml).join("")}</div>`,
});

function frameworkSwitchHtml() {
  const choices = options.frameworks ?? [];
  if (choices.length < 2) return "";
  return `<div class="segmented framework-switch" role="radiogroup" aria-label="Framework">
    ${choices.map(item => `<label title="${esc(item.available ? item.label : item.reason || "Not available")}"><input type="radio" name="framework" value="${esc(item.id)}" ${item.id === framework() ? "checked" : ""} ${item.available ? "" : "disabled"}><span>${esc(item.label)}</span></label>`).join("")}
  </div>`;
}

function rfdetrModelHtml() {
  const variants = fwOptions().pretrained ?? [];
  const runs = runsOf("rfdetr");
  const current = variants.find(item => item.name === args.model) ?? variants[0];
  const selectedRun = runs.find(run => Object.values(run.weights).includes(args.pretrain_weights));
  const weightKind = selectedRun ? Object.entries(selectedRun.weights).find(([, path]) => path === args.pretrain_weights)?.[0] : "best";
  const sources = [["pretrained", "Pretrained"], ["run", `Previous run${runs.length ? "" : " (none yet)"}`]];
  if (modelSource === "custom") sources.push(["custom", "Custom"]);
  const sizes = `<div class="field"><span class="field-label">Size</span><div class="segmented" role="radiogroup" aria-label="RF-DETR size">
      ${variants.map(item => `<label title="${esc(item.name)} · ${item.resolution} px"><input type="radio" name="rfdetr-variant" value="${esc(item.name)}" ${item === current ? "checked" : ""}><span>${esc(item.label)}</span></label>`).join("")}
    </div></div>`;
  return `
    <div class="segmented model-source" role="radiogroup" aria-label="Base model source">
      ${sources.map(([value, label]) => `<label><input type="radio" name="model-source" value="${value}" ${modelSource === value ? "checked" : ""} ${value === "run" && !runs.length ? "disabled" : ""}><span>${esc(label)}</span></label>`).join("")}
    </div>
    ${modelSource === "pretrained" ? `
      <div class="model-row">
        ${sizes}
        <div class="model-note"><code>${esc(current?.name ?? args.model)}</code> · ${esc(current?.resolution ?? "?")} px ${current?.cached ? `<span class="ok-text">✓ downloaded</span>` : `<span class="muted">downloads on first use</span>`}</div>
      </div>` : modelSource === "run" ? `
      <div class="model-row">
        <label class="field grow">Run<select id="model-run">${runs.map(run => `<option value="${esc(run.id)}" ${run === selectedRun ? "selected" : ""}>${esc(run.name)}${run.variant ? ` · ${esc(run.variant)}` : ""}${run.best_metrics && Object.keys(run.best_metrics).length ? ` — ${esc(bestSummary(run.best_metrics))}` : ""}</option>`).join("")}</select></label>
        <div class="field"><span class="field-label">Weights</span><div class="segmented" role="radiogroup" aria-label="Weights file">
          ${["best", "last"].map(kind => `<label><input type="radio" name="model-weights" value="${kind}" ${kind === weightKind ? "checked" : ""} ${(selectedRun ?? runs[0])?.weights[kind] ? "" : "disabled"}><span>${kind}</span></label>`).join("")}
        </div></div>
      </div>
      ${selectedRun?.names_match === false ? `<p class="warn-text">This run was trained with different classes; its class head will be re-initialized.</p>` : ""}
      <p class="panel-note">Fine-tuning starts a new <code>${esc(args.model)}</code> run from these weights; to continue an interrupted run use Resume instead.</p>` : `
      <div class="model-row"><code class="model-custom">${esc(args.model)}${args.pretrain_weights ? ` ← ${esc(args.pretrain_weights)}` : ""}</code><span class="muted">Set in raw YAML; pick a source above to replace it.</span></div>`}`;
}

function modelHtml() {
  if (framework() === "rfdetr") return rfdetrModelHtml();
  const model = args.model || "";
  const match = PRETRAINED.exec(model);
  const family = match?.[1] ?? FAMILIES[0][0];
  const size = match?.[2] ?? "n";
  const pretrainedName = `${family}${size}${SUFFIX[category()]}.pt`;
  const cached = (fwOptions().pretrained ?? []).find(item => item.name === pretrainedName)?.cached;
  const runs = runsOf("ultralytics");
  const selectedRun = runs.find(run => Object.values(run.weights).includes(model));
  const weightKind = selectedRun ? Object.entries(selectedRun.weights).find(([, path]) => path === model)?.[0] : "best";
  const sources = [["pretrained", "Pretrained"], ["run", `Previous run${runs.length ? "" : " (none yet)"}`]];
  if (modelSource === "custom") sources.push(["custom", "Custom"]);
  return `
    <div class="segmented model-source" role="radiogroup" aria-label="Base model source">
      ${sources.map(([value, label]) => `<label><input type="radio" name="model-source" value="${value}" ${modelSource === value ? "checked" : ""} ${value === "run" && !runs.length ? "disabled" : ""}><span>${esc(label)}</span></label>`).join("")}
    </div>
    ${modelSource === "pretrained" ? `
      <div class="model-row">
        <label class="field">Family<select id="model-family">${FAMILIES.map(([value, label]) => `<option value="${value}" ${value === family ? "selected" : ""}>${label}</option>`).join("")}</select></label>
        <div class="field"><span class="field-label">Size</span><div class="segmented" role="radiogroup" aria-label="Model size">
          ${SIZES.map(([value, label]) => `<label title="${label}"><input type="radio" name="model-size" value="${value}" ${value === size ? "checked" : ""}><span>${value}</span></label>`).join("")}
        </div></div>
        <div class="model-note"><code>${esc(pretrainedName)}</code> ${cached ? `<span class="ok-text">✓ downloaded</span>` : `<span class="muted">downloads on first use</span>`}</div>
      </div>` : modelSource === "run" ? `
      <div class="model-row">
        <label class="field grow">Run<select id="model-run">${runs.map(run => `<option value="${esc(run.id)}" ${run === selectedRun ? "selected" : ""}>${esc(run.name)}${run.best_metrics && Object.keys(run.best_metrics).length ? ` — ${esc(bestSummary(run.best_metrics))}` : ""}</option>`).join("")}</select></label>
        <div class="field"><span class="field-label">Weights</span><div class="segmented" role="radiogroup" aria-label="Weights file">
          ${["best", "last"].map(kind => `<label><input type="radio" name="model-weights" value="${kind}" ${kind === weightKind ? "checked" : ""} ${(selectedRun ?? runs[0])?.weights[kind] ? "" : "disabled"}><span>${kind}.pt</span></label>`).join("")}
        </div></div>
      </div>
      ${selectedRun?.names_match === false ? `<p class="warn-text">This run was trained with different classes; its class head will be re-initialized.</p>` : ""}
      <p class="panel-note">Fine-tuning starts a new run from these weights; to continue an interrupted run use Resume instead.</p>` : `
      <div class="model-row"><code class="model-custom">${esc(model)}</code><span class="muted">Set in raw YAML; pick a source above to replace it.</span></div>`}`;
}

function bestSummary(metrics) {
  const key = Object.keys(metrics).find(name => /mAP50-95\((M|B)\)$/.test(name) && (category() !== "segmentation" || name.endsWith("(M)"))) ?? Object.keys(metrics).find(name => name.includes("top1"));
  return key ? `${key.replace("metrics/", "")} ${metrics[key].toFixed(3)}` : "";
}

function fieldHtml(field) {
  let def = defaults()[field.key];
  let step = field.step;
  if (field.key === "resolution" && framework() === "rfdetr") {
    // the variant's own size, and RF-DETR's rule: a multiple of patch size × windows
    def = variantInfo()?.resolution ?? def;
    step = variantInfo()?.step ?? step;
  }
  const help = (fwOptions().help ?? {})[field.key] || "";
  const set = field.key in args;
  const value = args[field.key];
  const changed = set ? "changed" : "";
  const describe = field.hint ? `<span class="field-hint">${esc(field.hint)}</span>` : "";
  const title = `${field.key}: ${help}`;
  if (field.type === "bool") {
    const checked = set ? !!value : !!def;
    return `<label class="toggle form-toggle ${changed}"><input type="checkbox" data-key="${field.key}" aria-label="${esc(field.label)}" title="${esc(title)}" ${checked ? "checked" : ""}><span>${esc(field.label)}</span></label>`;
  }
  if (field.type === "select" || field.type === "device") {
    const choices = field.type === "device"
      ? [["cpu", "CPU"], ...options.devices.map(device => [device.id, `GPU ${device.id} · ${device.name}${device.memory_total ? ` (${Math.round(device.memory_total / 1024)} GB)` : ""}`])]
      : field.options;
    const current = set ? JSON.stringify(value) : "";
    const defaultLabel = field.type === "device" ? "Auto (first free GPU)" : `Default (${choices.find(([choice]) => same(choice, def))?.[1] ?? def})`;
    const known = choices.some(([choice]) => JSON.stringify(choice) === current);
    return `<label class="field ${changed}">${esc(field.label)}<select data-key="${field.key}" data-type="select" aria-label="${esc(field.label)}" title="${esc(title)}">
      <option value="" ${set ? "" : "selected"}>${esc(defaultLabel)}</option>
      ${choices.filter(([choice]) => field.type === "device" || !same(choice, def)).map(([choice, label]) => `<option value="${esc(JSON.stringify(choice))}" ${JSON.stringify(choice) === current ? "selected" : ""}>${esc(label)}</option>`).join("")}
      ${set && !known ? `<option value="${esc(current)}" selected>${esc(String(value))}</option>` : ""}
    </select>${describe}</label>`;
  }
  if (field.type === "batch") {
    return `<label class="field ${changed}">${esc(field.label)}<input aria-label="${esc(field.label)}" title="${esc(title)}" type="text" data-key="${field.key}" data-type="batch" value="${esc(set ? value : "")}" placeholder="${esc(def ?? "")}" inputmode="numeric" spellcheck="false">${describe}</label>`;
  }
  step = step ?? (field.type === "int" ? 1 : "any");
  const shown = set ? (Array.isArray(value) ? JSON.stringify(value) : value) : "";
  const isList = set && Array.isArray(value);
  return `<label class="field ${changed}">${esc(field.label)}<input aria-label="${esc(field.label)}" title="${esc(title)}" ${isList ? 'type="text" readonly' : `type="number" step="${step}"`} data-key="${field.key}" data-type="${field.type}" value="${esc(shown)}" placeholder="${esc(def ?? field.empty ?? "")}" ${field.min !== undefined ? `min="${field.min}"` : ""} ${field.max !== undefined ? `max="${field.max}"` : ""} inputmode="decimal">${isList ? `<span class="field-hint">A list — edit in raw YAML</span>` : describe}</label>`;
}

function extrasHtml() {
  const covered = new Set(["model", "pretrain_weights", ...fields().map(field => field.key)]);
  const extras = Object.entries(args).filter(([key]) => !covered.has(key));
  if (!extras.length) return "";
  return `<div class="stats-panel form-panel"><div class="panel-head"><h2>Other arguments</h2><span class="muted">Not in the form — edit them in raw YAML</span></div>
    <div class="chips extra-args">${extras.map(([key, value]) => `<span class="chip"><code>${esc(key)}: ${esc(yamlValue(value))}</code><button type="button" class="icon-btn mini" data-remove-arg="${esc(key)}" title="Remove ${esc(key)}" aria-label="Remove ${esc(key)}">×</button></span>`).join("")}</div></div>`;
}

// ---------- events ----------

function setArg(key, value) {
  if (value === undefined || same(value, defaults()[key])) delete args[key];
  else args[key] = value;
}

function onInput(e) {
  const input = e.target.closest("input[data-key][type=number], input[data-key][data-type=batch]");
  if (!input) {
    if (e.target.id === "raw-args") scheduleCheck();
    return;
  }
  const raw = input.value.trim();
  if (raw === "") setArg(input.dataset.key, undefined);
  else if (input.dataset.type === "batch") {
    // a number, or "auto"; anything else waits until it is one
    if (raw.toLowerCase() === "auto") setArg(input.dataset.key, "auto");
    else if (/^\d+$/.test(raw)) setArg(input.dataset.key, Number(raw));
    else return;
  } else {
    const number = Number(raw);
    if (!Number.isFinite(number)) return;
    setArg(input.dataset.key, input.dataset.type === "int" ? Math.trunc(number) : number);
  }
  input.closest(".field").classList.toggle("changed", input.dataset.key in args);
  augmentationArgChanged(input.dataset.key);
  rfAugmentationArgChanged(input.dataset.key);
  updateChangedCounts();
  scheduleCheck();
}

async function onChange(e) {
  const target = e.target;
  if (target.matches("input[data-train-folder]")) {
    folders = readFolders();
    $("folders-summary").innerHTML = foldersSummary();
    return scheduleCheck();
  }
  if (target.name === "form-mode") return switchMode(target.value, target);
  if (target.name === "framework") return switchFramework(target.value);
  if (target.name === "model-source" && framework() === "rfdetr") {
    modelSource = target.value;
    if (modelSource === "pretrained") args = withoutKey(args, "pretrain_weights");
    const run = runsOf("rfdetr")[0];
    if (modelSource === "run" && run) args = { ...args, model: run.variant ?? args.model, pretrain_weights: run.weights.best ?? run.weights.last };
    return refreshModel();
  }
  if (target.name === "rfdetr-variant") {
    args = { ...args, model: target.value };
    if (args.resolution !== undefined && variantInfo()?.step && args.resolution % variantInfo().step) delete args.resolution;
    return refreshModel();
  }
  if (target.name === "model-source") {
    modelSource = target.value;
    if (modelSource === "pretrained") args.model = PRETRAINED.test(args.model) ? args.model : defaultModel();
    if (modelSource === "run" && runsOf("ultralytics").length) {
      const run = runsOf("ultralytics")[0];
      args.model = run.weights.best ?? run.weights.last;
    }
    return refreshModel();
  }
  if (target.id === "model-family" || target.name === "model-size") {
    const family = $("model-family").value;
    const size = $("run-form").querySelector("input[name=model-size]:checked")?.value ?? "n";
    args.model = `${family}${size}${SUFFIX[category()]}.pt`;
    return refreshModel();
  }
  if (target.id === "model-run" || target.name === "model-weights") {
    const run = options.runs.find(item => item.id === $("model-run").value);
    const kind = target.name === "model-weights" ? target.value : (run.weights.best ? "best" : "last");
    const weights = run.weights[kind] ?? run.weights.best ?? run.weights.last;
    if (framework() === "rfdetr") args = { ...args, model: run.variant ?? args.model, pretrain_weights: weights };
    else args.model = weights;
    return refreshModel();
  }
  if (target.matches("input[type=checkbox][data-key]")) {
    setArg(target.dataset.key, target.checked);
    target.closest(".form-toggle").classList.toggle("changed", target.dataset.key in args);
  } else if (target.matches("select[data-key]")) {
    setArg(target.dataset.key, target.value === "" ? undefined : JSON.parse(target.value));
    target.closest(".field").classList.toggle("changed", target.dataset.key in args);
  } else return;
  augmentationArgChanged(target.dataset.key);
  rfAugmentationArgChanged(target.dataset.key);
  updateChangedCounts();
  scheduleCheck();
}

function onClick(e) {
  if (e.target.closest("[data-all-folders]")) {
    folders = null;
    renderFolders();
    return scheduleCheck(0);
  }
  const remove = e.target.closest("[data-remove-arg]");
  if (remove) {
    delete args[remove.dataset.removeArg];
    renderGui();
    scheduleCheck();
  }
}

function refreshModel() {
  const open = [...$("form-gui").querySelectorAll("details[open]")].map(item => item.dataset.group);
  renderGui();
  $("form-gui").querySelectorAll("details").forEach(item => { if (open.includes(item.dataset.group)) item.open = true; });
  scheduleCheck();
}

// each framework has its own arguments: switching keeps what was set for the other one while the form is open
function switchFramework(next) {
  if (next === framework() || !usable(next)) return;
  argsByFramework[framework()] = args;
  args = argsByFramework[next] ?? readLastArgs(next) ?? { model: defaultModel(next) };
  modelSource = detectModelSource(args);
  renderGui();
  scheduleCheck(0);
}

function updateChangedCounts() {
  for (const group of groups()) {
    const count = fields().filter(field => field.group === group.id && field.key in args).length;
    const el = $("form-gui").querySelector(`[data-changed="${group.id}"]`);
    if (el) el.textContent = count ? `${count} changed` : "";
  }
}

async function switchMode(next, radio) {
  if (next === mode) return;
  if (next === "raw") {
    $("raw-args").value = argsToYaml(args);
    mode = "raw";
  } else {
    const result = await api("/training/args/validate", { method: "POST", body: JSON.stringify({ yaml: $("raw-args").value }) });
    if (result.errors.length) {
      check.argErrors = result.errors;
      renderCheck();
      $("run-form").querySelector('input[name=form-mode][value="raw"]').checked = true;
      deps.toast("Fix the YAML errors before switching back to the form", "error");
      return;
    }
    args = result.args;
    modelSource = detectModelSource(args);
    mode = "gui";
    renderGui();
  }
  $("form-gui").hidden = mode !== "gui";
  $("form-raw").hidden = mode !== "raw";
  if (mode === "raw") $("raw-args").focus();
  scheduleCheck(0);
}

// ---------- validation + preflight ----------

const currentYaml = () => mode === "raw" ? $("raw-args").value : argsToYaml(args);

const runCheck = async () => {
  const token = ++checkToken;
  const yaml = currentYaml();
  try {
    const [validation, preflight] = await Promise.all([
      api("/training/args/validate", { method: "POST", body: JSON.stringify({ yaml }) }),
      api("/training/preflight", { method: "POST", body: JSON.stringify({ yaml, folders }) }),
    ]);
    if (token !== checkToken || !$("run-form")) return;
    check = { ...preflight, argErrors: validation.errors.concat(validation.warnings.map(item => ({ ...item, warning: true }))) };
  } catch (error) {
    if (token !== checkToken || !$("run-form")) return;
    check = { errors: [error.message], warnings: [], notes: [], splits: null, argErrors: [] };
  }
  renderCheck();
};
const scheduleCheckDebounced = debounce(runCheck, 400);
function scheduleCheck(delay) {
  if (delay === 0) runCheck(); else scheduleCheckDebounced();
}

function renderCheck() {
  const el = $("form-check");
  if (!el) return;
  const splits = check.splits;
  const argErrors = check.argErrors.filter(item => !item.warning);
  const argWarnings = check.argErrors.filter(item => item.warning);
  const line = item => `${item.line ? `<span class="line-ref">${mode === "raw" ? `line ${item.line}` : esc(lineKey(item.line))}</span> ` : ""}${esc(item.message)}`;
  el.innerHTML = `
    <div class="panel-head"><h2>Before starting</h2>${splits ? `<span class="muted">Training on ${["train", "val", "test"].map(split => `${splits[split] ?? 0} ${split}`).join(" · ")} images</span>` : ""}</div>
    <ul class="check-list">
      ${argErrors.map(item => `<li class="check-error">${line(item)}</li>`).join("")}
      ${check.errors.map(message => `<li class="check-error">${esc(message)}</li>`).join("")}
      ${[...argWarnings.map(line), ...check.warnings.map(esc)].map(message => `<li class="check-warning">${message}</li>`).join("")}
      ${check.notes.map(message => `<li class="check-note">${esc(message)}</li>`).join("")}
      ${!argErrors.length && !check.errors.length && !check.warnings.length && !argWarnings.length ? `<li class="check-ok">Ready to train</li>` : ""}
    </ul>`;
  updateSubmit();
}

function lineKey(number) {
  const text = currentYaml().split("\n")[number - 1] || "";
  return text.split(":")[0].trim() || `line ${number}`;
}

function updateSubmit() {
  const button = $("run-form-submit");
  if (!button) return;
  const blocked = check.errors.length || check.argErrors.some(item => !item.warning);
  const warned = check.warnings.length > 0;
  button.disabled = !!blocked;
  const label = queueAhead ? `Add to queue (#${queueAhead})` : "Start training";
  button.textContent = warned ? `${label} anyway` : label;
  button.classList.toggle("warn-btn", warned);
}

// ---------- submit ----------

async function submit() {
  const button = $("run-form-submit");
  const name = $("run-name").value.trim();
  if (!name) { $("run-name").focus(); return; }
  button.disabled = true;
  try {
    const result = await api("/training/runs", {
      method: "POST",
      body: JSON.stringify({ name, notes: $("run-notes").value.trim(), yaml: currentYaml(), confirm: check.warnings.length > 0, folders }),
    });
    if (!result.queued) {
      // new warnings since the last check (e.g. the dataset changed): show them and let the user confirm
      check = { ...check, warnings: result.warnings, notes: result.notes };
      renderCheck();
      return;
    }
    if (mode === "raw") {
      const parsed = await api("/training/args/validate", { method: "POST", body: JSON.stringify({ yaml: $("raw-args").value }) }).catch(() => null);
      if (parsed && !parsed.errors.length) args = parsed.args;
    }
    saveLastArgs();
    deps.toast(result.run.active ? `Training “${result.run.name}” started` : `“${result.run.name}” queued as #${result.run.queue_position}`);
    deps.started(result.run.id);
  } catch (error) {
    deps.toast(error.message, "error");
    check = { ...check, errors: [error.message] };
    renderCheck();
  } finally {
    if ($("run-form-submit")) updateSubmit();
  }
}
