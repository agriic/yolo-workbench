// New-run form of the Training tab. One args object is the source of truth: GUI fields edit it
// (a value equal to the Ultralytics default is dropped, so only real choices are kept), and the
// raw YAML mode round-trips through the server's validator (POST /training/args/validate).
import { $, api, esc, debounce } from "./api.js";
import { state } from "./state.js";
import { augmentationArgChanged, augmentationHtml, mountAugmentations } from "./augments.js";

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

const GROUPS = [
  { id: "basics", title: "Basics", open: true },
  { id: "optimizer", title: "Optimizer" },
  { id: "augmentation", title: "Augmentation" },
  { id: "advanced", title: "Advanced" },
];
// `cats` limits a field to dataset categories; everything else is reachable in raw YAML.
// The augmentation group renders as preview cards (augments.js); its entries here keep counts and raw YAML in step.
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
];

let options = null;   // GET /training/options
let args = {};        // the run's arguments (non-default values + model)
let mode = "gui";     // "gui" | "raw"
let modelSource = "pretrained";
let check = { errors: [], warnings: [], notes: [], splits: null, argErrors: [] };
let checkToken = 0;
let queueAhead = 0;

const category = () => options.category;
const fields = () => FIELDS.filter(field => !field.cats || field.cats.includes(category()));
const storageKey = () => `yolo-workbench:train-args:${state.project ?? state.meta.yaml}`;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export async function loadTrainingOptions() {
  options = await api("/training/options");
  return options;
}
export const trainingOptions = () => options;

// ---------- YAML (GUI → raw); raw → GUI goes through the server ----------

const PLAIN = /^[A-Za-z_][\w.\-/]*$/;
const RESERVED = /^(true|false|yes|no|on|off|null|~)$/i;
function yamlValue(value) {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean" || typeof value === "number") return String(value);
  if (Array.isArray(value)) return JSON.stringify(value);
  const text = String(value);
  return PLAIN.test(text) && !RESERVED.test(text) ? text : JSON.stringify(text);
}
export function argsToYaml(values) {
  return Object.entries(values).map(([key, value]) => `${key}: ${yamlValue(value)}`).join("\n") + (Object.keys(values).length ? "\n" : "");
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
    if (source.runId) {
      const copied = await api(`/training/runs/${encodeURIComponent(source.runId)}/args`);
      args = { ...copied.args };
      name = copied.name;
      if (source.from === "finetune") {
        const run = loaded.runs.find(item => item.id === source.runId);
        const weights = run?.weights.best ?? run?.weights.last;
        if (weights) args = { model: weights, ...withoutKey(args, "model") };
        name = `${copied.name.replace(/-\d+$/, "")}-ft`;
        notes = `Fine-tuned from ${run?.name ?? "a previous run"}`;
      }
    } else {
      args = readLastArgs() ?? { model: defaultModel() };
    }
    if (!args.model) args = { model: defaultModel(), ...args };
    mode = "gui";
    modelSource = detectModelSource(args.model);
    renderForm(name, notes, source);
    scheduleCheck(0);
  } catch (error) {
    container.innerHTML = `<div class="empty">${esc(error.message)}</div>`;
  }
}

const withoutKey = (object, key) => Object.fromEntries(Object.entries(object).filter(([name]) => name !== key));
const defaultModel = () => `${FAMILIES[0][0]}n${SUFFIX[category()]}.pt`;

function readLastArgs() {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey()) || "null");
    return saved && typeof saved === "object" && !Array.isArray(saved) ? saved : null;
  } catch { return null; }
}
function saveLastArgs() {
  try { localStorage.setItem(storageKey(), JSON.stringify(args)); } catch { /* a convenience only */ }
}

function detectModelSource(model) {
  if (PRETRAINED.test(model || "")) return "pretrained";
  if (options.runs.some(run => Object.values(run.weights).includes(model))) return "run";
  return "custom";
}

// ---------- render ----------

function renderForm(name, notes, source) {
  const title = source.from === "duplicate" ? "New run (duplicated)" : source.from === "finetune" ? "New run (fine-tune)" : "New run";
  $("run-detail").innerHTML = `
    <form id="run-form" class="run-form" autocomplete="off" novalidate>
      <div class="run-head">
        <div class="run-title"><h1>${esc(title)}</h1><p class="section-subtitle">${esc(category())} · Ultralytics ${esc(options.task)} · values left empty use the Ultralytics default (shown greyed)</p></div>
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
      <div id="form-gui"></div>
      <div id="form-raw" hidden>
        <div class="stats-panel form-panel">
          <div class="panel-head"><h2>Arguments</h2><span class="muted">One <code>key: value</code> per line, as in <code>args.yaml</code>. <code>data</code>, <code>project</code> and <code>name</code> are set by the workbench.</span></div>
          <textarea id="raw-args" class="raw-args" spellcheck="false" rows="16" aria-label="Training arguments (YAML)"></textarea>
        </div>
      </div>
      <div id="form-check" class="stats-panel form-panel form-check" aria-live="polite"></div>
      <div class="form-actions">
        <button type="button" class="secondary" id="run-form-cancel">Cancel</button>
        <button type="submit" class="primary-btn" id="run-form-submit">Start</button>
      </div>
    </form>`;
  renderGui();
  const form = $("run-form");
  form.addEventListener("submit", e => { e.preventDefault(); submit(); });
  form.addEventListener("change", onChange);
  form.addEventListener("input", onInput);
  form.addEventListener("click", onClick);
  $("run-form-cancel").onclick = () => deps.closeForm();
  updateSubmit();
}

function renderGui() {
  const changed = group => fields().filter(field => field.group === group && field.key in args).length;
  $("form-gui").innerHTML = `
    <div class="stats-panel form-panel">
      <div class="panel-head"><h2>Model</h2></div>
      ${modelHtml()}
    </div>
    ${GROUPS.map(group => `
      <details class="stats-panel form-panel form-group" data-group="${group.id}" ${group.open || changed(group.id) ? "open" : ""}>
        <summary><h2>${esc(group.title)}</h2><span class="changed-count" data-changed="${group.id}">${changed(group.id) ? `${changed(group.id)} changed` : ""}</span></summary>
        ${group.id === "augmentation" ? augmentationHtml(augmentationContext()) : `<div class="form-grid">${fields().filter(field => field.group === group.id).map(fieldHtml).join("")}</div>`}
      </details>`).join("")}
    ${extrasHtml()}`;
  mountAugmentations($("form-gui").querySelector('[data-group="augmentation"]'));
}

const augmentationContext = () => ({
  category: category(),
  defaults: options.defaults,
  help: options.help,
  fields: fields(),
  args: () => args,
  setArg,
  changed: () => { updateChangedCounts(); scheduleCheck(); },
});

function modelHtml() {
  const model = args.model || "";
  const match = PRETRAINED.exec(model);
  const family = match?.[1] ?? FAMILIES[0][0];
  const size = match?.[2] ?? "n";
  const pretrainedName = `${family}${size}${SUFFIX[category()]}.pt`;
  const cached = options.pretrained.find(item => item.name === pretrainedName)?.cached;
  const runs = options.runs;
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
  const def = options.defaults[field.key];
  const help = options.help[field.key] || "";
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
  const step = field.step ?? (field.type === "int" ? 1 : "any");
  const shown = set ? (Array.isArray(value) ? JSON.stringify(value) : value) : "";
  const isList = set && Array.isArray(value);
  return `<label class="field ${changed}">${esc(field.label)}<input aria-label="${esc(field.label)}" title="${esc(title)}" ${isList ? 'type="text" readonly' : `type="number" step="${step}"`} data-key="${field.key}" data-type="${field.type}" value="${esc(shown)}" placeholder="${esc(def ?? "")}" ${field.min !== undefined ? `min="${field.min}"` : ""} ${field.max !== undefined ? `max="${field.max}"` : ""} inputmode="decimal">${isList ? `<span class="field-hint">A list — edit in raw YAML</span>` : describe}</label>`;
}

function extrasHtml() {
  const covered = new Set(["model", ...fields().map(field => field.key)]);
  const extras = Object.entries(args).filter(([key]) => !covered.has(key));
  if (!extras.length) return "";
  return `<div class="stats-panel form-panel"><div class="panel-head"><h2>Other arguments</h2><span class="muted">Not in the form — edit them in raw YAML</span></div>
    <div class="chips extra-args">${extras.map(([key, value]) => `<span class="chip"><code>${esc(key)}: ${esc(yamlValue(value))}</code><button type="button" class="icon-btn mini" data-remove-arg="${esc(key)}" title="Remove ${esc(key)}" aria-label="Remove ${esc(key)}">×</button></span>`).join("")}</div></div>`;
}

// ---------- events ----------

function setArg(key, value) {
  if (value === undefined || same(value, options.defaults[key])) delete args[key];
  else args[key] = value;
}

function onInput(e) {
  const input = e.target.closest("input[data-key][type=number]");
  if (!input) {
    if (e.target.id === "raw-args") scheduleCheck();
    return;
  }
  const raw = input.value.trim();
  if (raw === "") setArg(input.dataset.key, undefined);
  else {
    const number = Number(raw);
    if (!Number.isFinite(number)) return;
    setArg(input.dataset.key, input.dataset.type === "int" ? Math.trunc(number) : number);
  }
  input.closest(".field").classList.toggle("changed", input.dataset.key in args);
  augmentationArgChanged(input.dataset.key);
  updateChangedCounts();
  scheduleCheck();
}

async function onChange(e) {
  const target = e.target;
  if (target.name === "form-mode") return switchMode(target.value, target);
  if (target.name === "model-source") {
    modelSource = target.value;
    if (modelSource === "pretrained") args.model = PRETRAINED.test(args.model) ? args.model : defaultModel();
    if (modelSource === "run" && options.runs.length) {
      const run = options.runs[0];
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
    args.model = run.weights[kind] ?? run.weights.best ?? run.weights.last;
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
  updateChangedCounts();
  scheduleCheck();
}

function onClick(e) {
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

function updateChangedCounts() {
  for (const group of GROUPS) {
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
    modelSource = detectModelSource(args.model);
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
      api("/training/preflight", { method: "POST", body: JSON.stringify({ yaml }) }),
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
  return text.split(":")[0] || `line ${number}`;
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
      body: JSON.stringify({ name, notes: $("run-notes").value.trim(), yaml: currentYaml(), confirm: check.warnings.length > 0 }),
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
