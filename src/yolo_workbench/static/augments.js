// Augmentation group of the new-run form: one card per augmentation with a slider and previews of the two ends of its
// range, plus training samples with every augmentation at once. The server renders them with Ultralytics' own transforms
// (GET /training/augment/{original,preview,sample}), so a preview is what training will do, labels included.
import { $, apiUrl, esc, debounce } from "./api.js";

const DET_SEG = ["detection", "segmentation"];
const CLS = ["classification"];
const pct = value => `${Math.round(value * 100)}%`;
const signed = (value, digits = 0, unit = "") => `${value < 0 ? "−" : value > 0 ? "+" : "±"}${Math.abs(value).toFixed(digits)}${unit}`;
const factor = value => `×${Math.max(0, value).toFixed(2)}`;

// kind: "range" previews both ends of the range; "chance" shows the effect plus how often it happens;
// "draws" shows two random draws. `slider` is the comfortable range; the number box accepts the field's full range.
// `fixed`: the preview image does not depend on the value (only how often it happens), so dragging re-renders nothing.
const AUGS = [
  { key: "hsv_h", label: "Hue", kind: "range", slider: [0, 0.1, 0.001],
    describe: () => "Shifts hue by up to ± this fraction of the colour wheel.",
    caption: (value, side) => `hue ${signed((side === "min" ? -value : value) * 360, 1, "°")}` },
  { key: "hsv_s", label: "Saturation", kind: "range", slider: [0, 1, 0.01],
    describe: () => "Multiplies saturation by a factor between 1 − value and 1 + value.",
    caption: (value, side) => `saturation ${factor(side === "min" ? 1 - value : 1 + value)}` },
  { key: "hsv_v", label: "Brightness", kind: "range", slider: [0, 1, 0.01],
    describe: category => category === "classification" ? "Multiplies brightness and contrast by 1 ± value." : "Multiplies brightness by a factor between 1 − value and 1 + value.",
    caption: (value, side) => `brightness ${factor(side === "min" ? 1 - value : 1 + value)}` },
  { key: "degrees", label: "Rotation", kind: "range", slider: [0, 90, 1],
    describe: () => "Rotates by up to ± this many degrees. Boxes grow to enclose rotated objects.",
    caption: (value, side) => signed(side === "min" ? -value : value, value % 1 ? 1 : 0, "°") },
  { key: "translate", label: "Translate", kind: "range", slider: [0, 0.5, 0.01],
    describe: () => "Shifts the image by up to ± this fraction of its size on both axes.",
    caption: (value, side) => `shift ${side === "min" ? "−" : "+"}${pct(value)}` },
  { key: "scale", label: "Scale", kind: "range", slider: [0, 1, 0.01],
    describe: category => category === "classification" ? "Random crop covering between 1 − value and all of the image, resized to the training size." : "Zooms by a factor between 1 − value and 1 + value. Objects that become too small are dropped.",
    caption: (value, side, category) => category === "classification" ? (side === "min" ? `crop ${pct(Math.max(0.01, 1 - value))} of area` : "whole image") : `zoom ${factor(side === "min" ? 1 - value : 1 + value)}` },
  { key: "shear", label: "Shear", kind: "range", slider: [0, 45, 0.5],
    describe: () => "Shears by up to ± this many degrees, both axes.",
    caption: (value, side) => `shear ${signed(side === "min" ? -value : value, value % 1 ? 1 : 0, "°")}` },
  { key: "perspective", label: "Perspective", kind: "range", slider: [0, 0.001, 0.00001],
    describe: () => "Perspective warp, usually 0–0.001. The same value warps more at a larger image size.",
    caption: (value, side) => value ? `${side === "min" ? "−" : "+"}${value.toPrecision(2)}` : "±0" },
  { key: "fliplr", label: "Flip left-right", kind: "chance", slider: [0, 1, 0.01],
    describe: () => "Share of images mirrored left to right. Avoid it when left and right mean different classes.",
    applied: "mirrored", fixed: true },
  { key: "flipud", label: "Flip up-down", kind: "chance", slider: [0, 1, 0.01],
    describe: () => "Share of images flipped upside down. Avoid it when orientation matters (text, street scenes).",
    applied: "upside down", fixed: true },
  { key: "bgr", label: "BGR swap", kind: "chance", slider: [0, 1, 0.01],
    describe: () => "Share of images with red and blue channels swapped.",
    applied: "channels swapped", fixed: true },
  { key: "mosaic", label: "Mosaic", kind: "chance", slider: [0, 1, 0.01],
    describe: () => "Share of images built from 4 training images. Switched off for the last close_mosaic epochs.",
    applied: "mosaic of 4", fixed: true },
  { key: "mixup", label: "MixUp", kind: "chance", slider: [0, 1, 0.01],
    describe: () => "Share of images blended about 50/50 with another image; both images' labels are kept.",
    applied: "blended", fixed: true },
  { key: "cutmix", label: "CutMix", kind: "chance", slider: [0, 1, 0.01],
    describe: () => "Share of images that get a patch of another image, placed where it covers none of this image's objects.",
    applied: "patch pasted", fixed: true },
  { key: "copy_paste", label: "Copy-paste", kind: "chance", slider: [0, 1, 0.01],
    describe: (category, mode) => mode === "mixup" ? "Pastes objects cut from another image: the value is both the chance and the share of its objects pasted." : "Pastes mirrored copies of this image's objects where they don't overlap others: the value is the share of objects copied.",
    applied: "objects pasted", unit: mode => mode === "mixup" ? "of images" : "of objects" },
  { key: "erasing", label: "Random erasing", kind: "draws", slider: [0, 1, 0.01],
    describe: () => "Share of images where a random rectangle (2–33% of the image) is erased.",
    caption: value => `erased · ${pct(value)} of images`, fixed: true },
  { key: "auto_augment", label: "Auto augment", kind: "draws", type: "select",
    describe: () => "A policy of random colour and geometry operations. While it is on, Ultralytics ignores Hue, Saturation and Brightness.",
    caption: (value, side) => value ? `random draw ${side === "min" ? 1 : 2}` : "off" },
];
const POLICIES = [["randaugment", "RandAugment"], ["autoaugment", "AutoAugment"], ["augmix", "AugMix"], [null, "Off"]];
const SAMPLE_COUNT = 6;
const TILE_SIZE = 256;
const ZOOM_SIZE = 640;

let ctx = null;        // { category, defaults, help, fields, args(), setArg(key, value), changed() }
let root = null;
let sample = 0;
let resample = 0;
let showLabels = true;
let observer = null;
const visible = new Set();
const stale = new Set();
const loaders = new WeakMap(); // img -> { controller, url }

const isClassification = () => ctx.category === "classification";
const specs = () => AUGS.filter(spec => ctx.fields.some(field => field.key === spec.key));
const spec = key => AUGS.find(item => item.key === key);
const value = key => ctx.args()[key] !== undefined ? ctx.args()[key] : ctx.defaults[key];
const numeric = key => Number(value(key)) || 0;
const imgsz = () => numeric("imgsz") || 640;
const colorJitterOff = () => isClassification() && !!value("auto_augment");

// ---------- render ----------

/** @param {object} context  see `ctx` above */
export function augmentationHtml(context) {
  ctx = context;
  return `
    <div class="aug-toolbar">
      <div class="aug-sample">
        <button type="button" class="aug-sample-thumb" data-aug-zoom="original" title="Enlarge"><img alt="Preview image" data-aug-original></button>
        <div class="aug-sample-text"><strong>Preview image</strong><span class="muted" data-aug-sample-name>Loading…</span></div>
        <div class="aug-sample-nav">
          <button type="button" class="icon-btn mini" data-aug-sample="-1" title="Previous image" aria-label="Previous preview image">‹</button>
          <button type="button" class="icon-btn mini" data-aug-sample="1" title="Next image" aria-label="Next preview image">›</button>
        </div>
      </div>
      ${isClassification() ? "" : `<label class="toggle"><input type="checkbox" data-aug-labels ${showLabels ? "checked" : ""}><span>Show labels</span></label>`}
      <span class="muted aug-toolbar-note">Rendered by Ultralytics' own transforms at the run's image size</span>
    </div>
    <div class="aug-samples-panel" data-aug-card="__samples">
      <div class="aug-samples-head"><h3>Training samples</h3><span class="muted">every augmentation at once, as the dataloader draws them</span>
        <button type="button" class="mini" data-aug-resample>Draw again</button></div>
      <div class="aug-samples">${Array.from({ length: SAMPLE_COUNT }, (_, index) => `
        <button type="button" class="aug-img" data-aug-zoom="sample" data-index="${index}" title="Enlarge"><img alt="Training sample ${index + 1}" data-aug-sample-img="${index}"></button>`).join("")}</div>
    </div>
    <div class="aug-cards">${specs().map(cardHtml).join("")}</div>`;
}

function cardHtml(item) {
  const key = item.key;
  const set = key in ctx.args();
  return `
    <section class="aug-card ${set ? "changed" : ""}" data-aug-card="${key}">
      <div class="aug-head">
        <span class="aug-title">${esc(item.label)}</span><code class="aug-key" title="${esc(ctx.help[key] || "")}">${esc(key)}</code>
        <button type="button" class="mini aug-reset" data-aug-reset="${key}" ${set ? "" : "hidden"} title="Back to the Ultralytics default (${esc(String(ctx.defaults[key]))})">Reset</button>
      </div>
      <p class="aug-desc" data-aug-desc>${esc(describe(item))}</p>
      ${controlHtml(item)}
      <div class="aug-inactive" data-aug-inactive hidden>Not applied while Auto augment is on.</div>
      <div class="aug-tiles">${tilesHtml(item)}</div>
    </section>`;
}

function describe(item) {
  return item.describe(ctx.category, value("copy_paste_mode"));
}

function controlHtml(item) {
  const key = item.key;
  if (item.type === "select") {
    const current = JSON.stringify(value(key) ?? null);
    return `<div class="aug-control single"><select data-aug-select="${key}" aria-label="${esc(item.label)}">
      ${POLICIES.map(([choice, label]) => `<option value="${esc(JSON.stringify(choice))}" ${JSON.stringify(choice) === current ? "selected" : ""}>${esc(label)}${choice === ctx.defaults[key] ? " (default)" : ""}</option>`).join("")}
    </select></div>`;
  }
  const field = ctx.fields.find(entry => entry.key === key) || {};
  const [low, high, step] = item.slider;
  const current = numeric(key);
  return `
    <div class="aug-control">
      <input type="range" data-aug-range="${key}" min="${low}" max="${high}" step="${step}" value="${Math.min(current, high)}" list="aug-default-${key}" aria-label="${esc(item.label)}">
      <datalist id="aug-default-${key}"><option value="${Number(ctx.defaults[key]) || 0}"></option></datalist>
      <input type="number" data-aug-number="${key}" step="any" ${field.min !== undefined ? `min="${field.min}"` : ""} ${field.max !== undefined ? `max="${field.max}"` : ""} value="${esc(key in ctx.args() ? ctx.args()[key] : "")}" placeholder="${esc(ctx.defaults[key] ?? "")}" inputmode="decimal" aria-label="${esc(item.label)} value">
    </div>
    ${key === "copy_paste" ? `<div class="aug-mode"><span class="muted">Mode</span><div class="segmented" role="radiogroup" aria-label="Copy-paste mode">
      ${["flip", "mixup"].map(mode => `<label><input type="radio" name="aug-copy-paste-mode" data-aug-mode value="${mode}" ${value("copy_paste_mode") === mode ? "checked" : ""}><span>${mode}</span></label>`).join("")}
    </div></div>` : ""}`;
}

function tilesHtml(item) {
  const tile = (side, caption) => `
    <figure class="aug-tile">
      <button type="button" class="aug-img" data-aug-zoom="${item.key}" title="Enlarge"><img alt="" data-aug-tile="${side}"></button>
      <figcaption><span data-aug-caption="${side}">${esc(caption)}</span><span class="tile-note" data-aug-note="${side}"></span></figcaption>
    </figure>`;
  if (item.kind === "range") return tile("min", "") + tile("original", "original") + tile("max", "");
  if (item.kind === "draws") return tile("original", "original") + tile("min", "") + tile("max", "");
  return tile("original", "") + tile("max", item.applied) + `<div class="aug-odds" data-aug-odds><div class="aug-ring"><span data-aug-odds-value></span></div><span class="muted" data-aug-odds-unit></span></div>`;
}

// ---------- mount ----------

/** Wire the group rendered by augmentationHtml(); previews load once the group is open and a card is on screen. */
export function mountAugmentations(details) {
  root = details;
  observer?.disconnect();
  visible.clear();
  stale.clear();
  root.querySelectorAll("[data-aug-card]").forEach(card => stale.add(card.dataset.augCard));
  stale.add("__original");
  observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      const key = entry.target.dataset.augCard;
      if (entry.isIntersecting) visible.add(key); else visible.delete(key);
    }
    flush();
  }, { rootMargin: "300px 0px" });
  root.querySelectorAll("[data-aug-card]").forEach(card => observer.observe(card));
  root.addEventListener("input", onInput);
  root.addEventListener("change", onChange);
  root.addEventListener("click", onClick);
  root.addEventListener("toggle", () => flush());
  for (const item of specs()) updateCard(item.key);
}

/** Another field changed (imgsz, or the raw editor): bring the previews it affects up to date. */
export function augmentationArgChanged(key) {
  if (!root?.isConnected) return;
  if (key === "imgsz") {
    markAll();
  } else if (spec(key) || key === "copy_paste_mode") {
    const card = key === "copy_paste_mode" ? "copy_paste" : key;
    syncControls(card);
    markStale(card);
  }
}

function markAll() {
  root.querySelectorAll("[data-aug-card]").forEach(card => stale.add(card.dataset.augCard));
  stale.add("__original");
  scheduleFlush();
}

function markStale(key) {
  stale.add(key);
  stale.add("__samples");
  if (key === "auto_augment") ["hsv_h", "hsv_s", "hsv_v"].forEach(name => updateCard(name));
  updateCard(key);
  scheduleFlush();
}

const scheduleFlush = debounce(() => flush(), 180);

function flush() {
  if (!root?.isConnected || !root.open) return;
  if (stale.has("__original")) {
    stale.delete("__original");
    loadOriginal();
  }
  for (const key of [...stale]) {
    if (!visible.has(key)) continue;
    stale.delete(key);
    if (key === "__samples") loadSamples(); else loadCard(key);
  }
}

// ---------- events ----------

function setValue(key, raw) {
  ctx.setArg(key, raw);
  root.querySelector(`[data-aug-card="${key}"]`)?.classList.toggle("changed", key in ctx.args());
  root.querySelector(`[data-aug-reset="${key}"]`)?.toggleAttribute("hidden", !(key in ctx.args()));
  markStale(key);
  ctx.changed();
}

function onInput(e) {
  const range = e.target.closest("[data-aug-range]");
  const number = e.target.closest("[data-aug-number]");
  if (range) {
    const key = range.dataset.augRange;
    setValue(key, Number(range.value));
    const box = root.querySelector(`[data-aug-number="${key}"]`);
    box.value = key in ctx.args() ? ctx.args()[key] : "";
  } else if (number) {
    const key = number.dataset.augNumber;
    const text = number.value.trim();
    if (text === "") setValue(key, undefined);
    else if (Number.isFinite(Number(text))) setValue(key, Number(text));
    else return;
    const slider = root.querySelector(`[data-aug-range="${key}"]`);
    slider.value = Math.min(numeric(key), Number(slider.max));
  }
}

function onChange(e) {
  const target = e.target;
  if (target.matches("[data-aug-select]")) setValue(target.dataset.augSelect, JSON.parse(target.value));
  else if (target.matches("[data-aug-mode]")) {
    ctx.setArg("copy_paste_mode", target.value);
    markStale("copy_paste");
    ctx.changed();
  } else if (target.matches("[data-aug-labels]")) {
    showLabels = target.checked;
    markAll();
    stale.add("__samples");
  }
}

function onClick(e) {
  const reset = e.target.closest("[data-aug-reset]");
  if (reset) {
    setValue(reset.dataset.augReset, undefined);
    syncControls(reset.dataset.augReset);
    return;
  }
  const step = e.target.closest("[data-aug-sample]");
  if (step) {
    sample = Math.max(0, sample + Number(step.dataset.augSample));
    markAll();
    stale.add("__samples");
    return;
  }
  if (e.target.closest("[data-aug-resample]")) {
    resample += 1;
    loadSamples();
    return;
  }
  const zoom = e.target.closest("[data-aug-zoom]");
  if (zoom) openZoom(zoom.dataset.augZoom, Number(zoom.dataset.index));
}

function syncControls(key) {
  const card = root.querySelector(`[data-aug-card="${key}"]`);
  if (!card) return;
  const slider = card.querySelector("[data-aug-range]");
  if (slider) slider.value = Math.min(numeric(key), Number(slider.max));
  const box = card.querySelector("[data-aug-number]");
  if (box) box.value = key in ctx.args() ? ctx.args()[key] : "";
  const select = card.querySelector("[data-aug-select]");
  if (select) select.value = JSON.stringify(value(key) ?? null);
  card.querySelectorAll("[data-aug-mode]").forEach(radio => { radio.checked = radio.value === value("copy_paste_mode"); });
  card.classList.toggle("changed", key in ctx.args());
  card.querySelector("[data-aug-reset]")?.toggleAttribute("hidden", !(key in ctx.args()));
}

// ---------- captions ----------

function updateCard(key) {
  const item = spec(key);
  const card = root?.querySelector(`[data-aug-card="${key}"]`);
  if (!item || !card) return;
  const current = item.type === "select" ? value(key) : numeric(key);
  const caption = (side, text) => { const el = card.querySelector(`[data-aug-caption="${side}"]`); if (el) el.textContent = text; };
  if (item.kind === "range" || item.kind === "draws") {
    caption("min", item.caption(current, "min", ctx.category));
    caption("max", item.caption(current, "max", ctx.category));
  }
  if (item.kind === "chance") {
    const unit = item.unit ? item.unit(value("copy_paste_mode")) : "of images";
    caption("original", unit !== "of images" ? "original" : current >= 1 ? "never as is" : `as is · ${pct(1 - Math.min(1, current))}`);
    card.querySelector("[data-aug-odds]").style.setProperty("--odds", Math.min(1, current));
    card.querySelector("[data-aug-odds-value]").textContent = pct(Math.min(1, current));
    card.querySelector("[data-aug-odds-unit]").textContent = unit;
  }
  card.querySelector("[data-aug-desc]").textContent = describe(item);
  const off = item.type !== "select" && current === 0;
  card.classList.toggle("is-off", off);
  const inactive = ["hsv_h", "hsv_s", "hsv_v"].includes(key) && colorJitterOff();
  card.classList.toggle("is-inactive", inactive);
  card.querySelector("[data-aug-inactive]").hidden = !inactive;
}

// ---------- loading ----------

function previewArgs(key) {
  const values = { imgsz: imgsz() };
  if (key) {
    if (!spec(key).fixed) values[key] = value(key);
    if (key === "copy_paste") values.copy_paste_mode = value("copy_paste_mode");
  } else {
    for (const item of specs()) values[item.key] = value(item.key);
    if (ctx.category === "segmentation") values.copy_paste_mode = value("copy_paste_mode");
  }
  return JSON.stringify(values);
}

const query = params => new URLSearchParams({ ...params, labels: String(showLabels && !isClassification()) }).toString();
const originalUrl = size => apiUrl(`/training/augment/original?${query({ sample, imgsz: imgsz(), size })}`);
const previewUrl = (key, side, size) => apiUrl(`/training/augment/preview?${query({ key, side, sample, size, args: previewArgs(key) })}`);
const sampleUrl = (index, size) => apiUrl(`/training/augment/sample?${query({ sample: sample + index, seed: resample * SAMPLE_COUNT + index, size, args: previewArgs(null) })}`);

async function loadImage(img, url, noteEl) {
  const previous = loaders.get(img);
  if (previous?.source === url && !previous.failed) return null;  // already showing (or loading) this image
  previous?.controller.abort();
  const controller = new AbortController();
  loaders.set(img, { controller, url: previous?.url, source: url });
  const frame = img.closest(".aug-img, .aug-sample-thumb");
  frame?.classList.add("loading");
  frame?.classList.remove("failed");
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      const body = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(body.detail || response.statusText);
    }
    const note = decodeURIComponent(response.headers.get("X-Preview-Note") || "");
    const objectUrl = URL.createObjectURL(await response.blob());
    if (previous?.url) URL.revokeObjectURL(previous.url);
    loaders.set(img, { controller, url: objectUrl, source: url });
    img.src = objectUrl;
    if (noteEl) noteEl.textContent = note;
    frame?.classList.remove("loading");
    return note;
  } catch (error) {
    if (error.name === "AbortError") return null;
    loaders.set(img, { ...loaders.get(img), failed: true });
    frame?.classList.remove("loading");
    frame?.classList.add("failed");
    frame?.setAttribute("title", error.message);
    if (noteEl) noteEl.textContent = error.message;
    return null;
  }
}

async function loadOriginal() {
  const note = await loadImage(root.querySelector("[data-aug-original]"), originalUrl(TILE_SIZE));
  const name = root.querySelector("[data-aug-sample-name]");
  if (note !== null && name) name.textContent = note;
}

function loadCard(key) {
  const card = root.querySelector(`[data-aug-card="${key}"]`);
  if (!card) return;
  card.querySelectorAll("[data-aug-tile]").forEach(img => {
    const side = img.dataset.augTile;
    if (side === "original") loadImage(img, originalUrl(TILE_SIZE));
    else loadImage(img, previewUrl(key, side, TILE_SIZE), card.querySelector(`[data-aug-note="${side}"]`));
  });
}

function loadSamples() {
  root.querySelectorAll("[data-aug-sample-img]").forEach(img => loadImage(img, sampleUrl(Number(img.dataset.augSampleImg), TILE_SIZE)));
}

// ---------- zoom ----------

function zoomDialog() {
  let dialog = $("aug-zoom");
  if (dialog) return dialog;
  dialog = document.createElement("dialog");
  dialog.id = "aug-zoom";
  dialog.className = "modal aug-zoom";
  dialog.innerHTML = `
    <div class="plot-dialog-head"><strong data-zoom-title></strong>
      <button type="button" class="icon-btn" data-close title="Close" aria-label="Close"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg></button></div>
    <div class="aug-zoom-body" data-zoom-body></div>`;
  dialog.addEventListener("click", e => { if (e.target === dialog || e.target.closest("[data-close]")) dialog.close(); });
  document.body.append(dialog);
  return dialog;
}

function openZoom(key, index) {
  const dialog = zoomDialog();
  const body = dialog.querySelector("[data-zoom-body]");
  let tiles;
  if (key === "original") tiles = [["original", originalUrl(ZOOM_SIZE), root.querySelector("[data-aug-sample-name]").textContent]];
  else if (key === "sample") tiles = [["sample", sampleUrl(index, ZOOM_SIZE), `Training sample ${index + 1}`]];
  else {
    const card = root.querySelector(`[data-aug-card="${key}"]`);
    tiles = [...card.querySelectorAll("[data-aug-tile]")].map(img => {
      const side = img.dataset.augTile;
      return [side, side === "original" ? originalUrl(ZOOM_SIZE) : previewUrl(key, side, ZOOM_SIZE), card.querySelector(`[data-aug-caption="${side}"]`).textContent];
    });
  }
  dialog.querySelector("[data-zoom-title]").textContent = key === "original" ? "Preview image" : key === "sample" ? "Training sample" : `${spec(key).label} (${key})`;
  body.style.setProperty("--zoom-count", tiles.length);
  body.innerHTML = tiles.map(([side, , caption]) => `
    <figure class="aug-tile"><div class="aug-img loading"><img alt="" data-zoom-img="${side}"></div>
      <figcaption><span>${esc(caption)}</span><span class="tile-note" data-zoom-note="${side}"></span></figcaption></figure>`).join("");
  for (const [side, url] of tiles) loadImage(body.querySelector(`[data-zoom-img="${side}"]`), url, side === "original" ? null : body.querySelector(`[data-zoom-note="${side}"]`));
  if (!dialog.open) dialog.showModal();
}
