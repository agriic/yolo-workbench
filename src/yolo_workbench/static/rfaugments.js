// RF-DETR's Augmentation group of the new-run form: `aug_config`, the Albumentations transforms RF-DETR applies after
// its resize, as a preset to start from plus one card per transform (its chance, its parameters, two random draws).
// No aug_config is RF-DETR's built-in augmentation (a horizontal flip half the time), {} none at all. The transforms,
// their parameters and defaults come from the installed Albumentations (GET /training/augment/rfdetr/catalog); previews
// are rendered by it on a dataset image (GET /training/augment/rfdetr), so every transform it has can be configured.
import { api, apiUrl, esc, debounce } from "./api.js";
import { loadImage, showZoom } from "./augments.js";

const PRESETS = [
  ["default", "Default", "RF-DETR's built-in augmentation: a horizontal flip half the time (torchvision). Edit or add a transform to configure your own."],
  ["none", "None", "No augmentation: images are only resized (and scaled, with multi-scale training)."],
  ["conservative", "Conservative", "RF-DETR's preset for small datasets (under 500 images)."],
  ["aggressive", "Aggressive", "RF-DETR's preset for large datasets (2000+ images)."],
  ["aerial", "Aerial", "RF-DETR's preset for satellite and overhead imagery."],
  ["industrial", "Industrial", "RF-DETR's preset for manufacturing and inspection data: lighting and sensor noise."],
];
const KINDS = {
  geometric: ["geometric", "Moves the image; RF-DETR moves the boxes and masks with it"],
  pixel: ["pixels", "Changes pixels only; labels stay as they are"],
  unmoved: ["labels not moved", "Moves the image, but RF-DETR leaves the labels where they were: they will be wrong"],
};
const CONTAINERS = ["OneOf", "SomeOf", "Sequential"];
const SAMPLE_COUNT = 6;
const TILE_SIZE = 256;
const ZOOM_SIZE = 640;

let catalogLoad = null; // the catalog request, once per page
let catalog = null;
let ctx = null;         // { category, args(), setArg(key, value), resolution(), changed(), extraHtml() }
let root = null;        // the group's <details>
let sample = 0;
let resample = 0;
let showLabels = true;
let observer = null;
const visible = new Set();
const stale = new Set();

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const config = () => ctx.args().aug_config;                                // undefined | null = default
const entries = () => Object.entries(config() ?? catalog?.default ?? { HorizontalFlip: { p: 0.5 } });
const info = name => catalog?.transforms?.find(item => item.name === name);
const isDefault = () => config() === undefined || config() === null;

function presetOf(value) {
  if (value === undefined || value === null) return "default";
  if (!Object.keys(value).length) return "none";
  return Object.entries(catalog?.presets ?? {}).find(([, preset]) => same(preset, value))?.[0] ?? "custom";
}

// ---------- values shown in text fields ----------

function formatValue(value) {
  if (value === undefined || value === null) return "";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value) && value.every(item => typeof item === "number")) return value.join(", ");
  return typeof value === "string" ? value : JSON.stringify(value);
}

// a number, "low, high" (a list), true/false/null, JSON, or else text; "" = the transform's default
function parseValue(text) {
  const raw = text.trim();
  if (raw === "") return undefined;
  const parts = raw.replace(/^[[(]|[\])]$/g, "").split(/\s*[,;]\s*|\s+/).filter(Boolean);
  if (parts.length && parts.every(part => part !== "" && Number.isFinite(Number(part)))) {
    const numbers = parts.map(Number);
    return numbers.length === 1 && !/^[[(]/.test(raw) ? numbers[0] : numbers;
  }
  if (/^(true|false)$/i.test(raw)) return raw.toLowerCase() === "true";
  if (raw === "null") return null;
  if (/^[[{]/.test(raw)) {
    try { return JSON.parse(raw); } catch { /* text, then */ }
  }
  return raw;
}

// ---------- render ----------

/** @param {object} context  see `ctx` above */
export function rfAugmentationHtml(context) {
  ctx = context;
  if (!catalog) {
    return `<div class="rfaug" data-rfaug><p class="muted">Loading the Albumentations transforms…</p>${ctx.extraHtml?.() ?? ""}</div>`;
  }
  return `<div class="rfaug" data-rfaug>${bodyHtml()}</div>`;
}

function bodyHtml() {
  const preset = presetOf(config());
  const note = PRESETS.find(([id]) => id === preset)?.[2] ?? "Your own transforms. RF-DETR applies them in this order, each at its own chance, after resizing.";
  const unavailable = catalog.available === false;
  const rows = entries();
  return `
    ${unavailable ? `<div class="banner run-error">${esc(catalog.reason)}. Custom augmentations cannot be previewed or used until it is installed.</div>` : ""}
    <div class="aug-toolbar">
      <div class="aug-sample">
        <button type="button" class="aug-sample-thumb" data-rf-zoom="original" title="Enlarge"><img alt="Preview image" data-rf-original></button>
        <div class="aug-sample-text"><strong>Preview image</strong><span class="muted" data-rf-sample-name>Loading…</span></div>
        <div class="aug-sample-nav">
          <button type="button" class="icon-btn mini" data-rf-sample="-1" title="Previous image" aria-label="Previous preview image">‹</button>
          <button type="button" class="icon-btn mini" data-rf-sample="1" title="Next image" aria-label="Next preview image">›</button>
        </div>
      </div>
      <label class="toggle"><input type="checkbox" data-rf-labels ${showLabels ? "checked" : ""}><span>Show labels</span></label>
      <span class="muted aug-toolbar-note">Rendered by Albumentations after RF-DETR's square resize to the resolution</span>
    </div>
    <div class="rfaug-presets">
      <span class="field-label">Start from</span>
      <div class="segmented" role="radiogroup" aria-label="Augmentation preset">
        ${PRESETS.map(([id, label]) => `<label><input type="radio" name="rfaug-preset" value="${id}" ${id === preset ? "checked" : ""} ${unavailable && id !== "default" && id !== "none" ? "disabled" : ""}><span>${label}</span></label>`).join("")}
        ${preset === "custom" ? `<label><input type="radio" name="rfaug-preset" value="custom" checked><span>Custom</span></label>` : ""}
      </div>
    </div>
    <p class="panel-note rfaug-note">${esc(note)}</p>
    ${rows.length ? `
    <div class="aug-samples-panel" data-rf-card="__samples">
      <div class="aug-samples-head"><h3>Training samples</h3><span class="muted">every transform at its own chance, as training draws them</span>
        <button type="button" class="mini" data-rf-resample>Draw again</button></div>
      <div class="aug-samples">${Array.from({ length: SAMPLE_COUNT }, (_, index) => `
        <button type="button" class="aug-img" data-rf-zoom="sample" data-index="${index}" title="Enlarge"><img alt="Training sample ${index + 1}" data-rf-sample-img="${index}"></button>`).join("")}</div>
    </div>` : ""}
    <div class="aug-cards">${rows.map(([name, params], index) => cardHtml(name, params, index, rows.length)).join("")}</div>
    <div class="rfaug-add">
      <label class="field">Add a transform<select data-rf-add ${unavailable ? "disabled" : ""}>
        <option value="">Choose…</option>
        ${Object.entries(catalog.groups ?? {}).map(([group, label]) => `<optgroup label="${esc(label)}">${(catalog.transforms ?? []).filter(item => item.group === group).map(item => `<option value="${esc(item.name)}" ${rows.some(([name]) => name === item.name) ? "disabled" : ""} title="${esc(item.summary)}">${esc(item.name)}${item.kind === "unmoved" ? " (labels not moved)" : ""}</option>`).join("")}</optgroup>`).join("")}
      </select><span class="field-hint">Every Albumentations transform; OneOf / SomeOf / Sequential groups go in raw YAML</span></label>
      ${ctx.extraHtml?.() ?? ""}
    </div>`;
}

function cardHtml(name, params, index, count) {
  const item = info(name);
  const container = CONTAINERS.includes(name);
  const values = params && typeof params === "object" && !Array.isArray(params) ? params : {};
  const chance = typeof values.p === "number" ? values.p : container && name !== "SomeOf" ? 1 : item?.p ?? 0.5;
  const [kindLabel, kindTitle] = KINDS[item?.kind] ?? (container ? ["group", "Applies the transforms it holds"] : ["unknown", "Not an Albumentations transform"]);
  const fields = (item?.params ?? []).map(param => paramHtml(index, param, values[param.name]));
  const main = fields.filter((_, at) => !item.params[at].advanced || item.params[at].name in values);
  const more = fields.filter((_, at) => item.params[at].advanced && !(item.params[at].name in values));
  const extras = container ? [] : Object.keys(values).filter(key => key !== "p" && !(item?.params ?? []).some(param => param.name === key));
  return `
    <section class="aug-card rfaug-card ${item?.kind === "unmoved" || (!item && !container) ? "rfaug-warn" : ""}" data-rf-card="${index}" data-rf-name="${esc(name)}">
      <div class="aug-head">
        <span class="aug-title">${esc(name)}</span><span class="rfaug-kind rfaug-kind-${esc(item?.kind ?? (container ? "group" : "unknown"))}" title="${esc(kindTitle)}">${esc(kindLabel)}</span>
        <span class="rfaug-actions">
          <button type="button" class="icon-btn mini" data-rf-move="-1" ${index === 0 ? "disabled" : ""} title="Earlier" aria-label="Move ${esc(name)} earlier">↑</button>
          <button type="button" class="icon-btn mini" data-rf-move="1" ${index === count - 1 ? "disabled" : ""} title="Later" aria-label="Move ${esc(name)} later">↓</button>
          <button type="button" class="icon-btn mini" data-rf-remove title="Remove" aria-label="Remove ${esc(name)}">×</button>
        </span>
      </div>
      <p class="aug-desc">${esc(item?.summary ?? (container ? "A group of transforms; edit what it holds in raw YAML." : "Not a transform the installed Albumentations has: RF-DETR would skip it."))}</p>
      <div class="rfaug-chance">
        <span class="field-label">Chance</span>
        <div class="aug-control">
          <input type="range" data-rf-p min="0" max="1" step="0.01" value="${chance}" aria-label="${esc(name)} chance">
          <input type="number" data-rf-p-number step="any" min="0" max="1" value="${esc("p" in values ? values.p : "")}" placeholder="${esc(chance)}" inputmode="decimal" aria-label="${esc(name)} chance value">
        </div>
      </div>
      ${main.length ? `<div class="form-grid rfaug-params">${main.join("")}</div>` : ""}
      ${more.length ? `<details class="rfaug-more"><summary>More parameters (${more.length})</summary><div class="form-grid rfaug-params">${more.join("")}</div></details>` : ""}
      ${container ? `<p class="rfaug-extra">${name === "OneOf" ? "One of" : name === "SomeOf" ? "Some of" : "In turn"}: <code>${esc(nested(params).join(", ") || "nothing")}</code> <span class="muted">· edit the group in raw YAML</span></p>`
        : extras.length ? `<p class="rfaug-extra"><code>${esc(extras.map(key => `${key}: ${formatValue(values[key])}`).join(" · "))}</code> <span class="muted">· not a parameter of ${esc(name)}</span></p>` : ""}
      <div class="aug-tiles">${["original", "a", "b"].map(side => `
        <figure class="aug-tile">
          <button type="button" class="aug-img" data-rf-zoom="${index}" title="Enlarge"><img alt="" data-rf-tile="${side}"></button>
          <figcaption><span>${side === "original" ? "original" : `random draw ${side === "a" ? 1 : 2}`}</span><span class="tile-note" data-rf-note="${side}"></span></figcaption>
        </figure>`).join("")}</div>
    </section>`;
}

// the transforms a OneOf / SomeOf / Sequential holds (its `transforms` list, or the list itself)
const nested = params => (Array.isArray(params) ? params : params?.transforms ?? []).flatMap(entry => entry && typeof entry === "object" ? Object.keys(entry) : []);

function paramHtml(index, param, value) {
  const set = value !== undefined;
  const id = `rfaug-${index}-${param.name}`;
  const label = param.name.replace(/_/g, " ");
  if (typeof param.default === "boolean") {
    return `<label class="toggle form-toggle ${set ? "changed" : ""}"><input type="checkbox" data-rf-param="${esc(param.name)}" ${(set ? value : param.default) ? "checked" : ""}><span>${esc(label)}</span></label>`;
  }
  if (param.choices) {
    const current = set ? JSON.stringify(value) : "";
    const known = param.choices.some(([choice]) => JSON.stringify(choice) === current);
    const defaultLabel = param.choices.find(([choice]) => same(choice, param.default))?.[1] ?? formatValue(param.default);
    return `<label class="field ${set ? "changed" : ""}">${esc(label)}<select id="${id}" data-rf-param="${esc(param.name)}" data-rf-choice>
      <option value="" ${set ? "" : "selected"}>Default (${esc(defaultLabel)})</option>
      ${param.choices.filter(([choice]) => !same(choice, param.default)).map(([choice, text]) => `<option value="${esc(JSON.stringify(choice))}" ${JSON.stringify(choice) === current ? "selected" : ""}>${esc(text)}</option>`).join("")}
      ${set && !known ? `<option value="${esc(current)}" selected>${esc(formatValue(value))}</option>` : ""}
    </select></label>`;
  }
  return `<label class="field ${set ? "changed" : ""} ${param.required && !set ? "rfaug-required" : ""}" title="${esc(`${param.name}: a number, low, high, true/false, text or JSON; empty = ${param.required ? "required" : `the default (${formatValue(param.default) || "none"})`}`)}">${esc(label)}<input id="${id}" type="text" data-rf-param="${esc(param.name)}" value="${esc(formatValue(value))}" placeholder="${esc(param.required ? "required" : formatValue(param.default) || "none")}" spellcheck="false" autocomplete="off"></label>`;
}

// ---------- mount ----------

/** Wire the group rendered by rfAugmentationHtml(); the catalog loads first, previews once the group is open and on screen. */
export function mountRfAugmentations(details) {
  root = details;
  if (!catalog) {
    catalogLoad ??= api("/training/augment/rfdetr/catalog").catch(error => ({ available: false, reason: error.message, presets: {}, groups: {}, transforms: [] }));
    catalogLoad.then(loaded => {
      catalog = loaded;
      if (root === details && details.isConnected) rerender();
    });
    return;
  }
  wire();
}

function rerender() {
  const body = root.querySelector("[data-rfaug]");
  if (!body) return;
  body.innerHTML = bodyHtml();
  wire();
}

let wired = null;
function wire() {
  observer?.disconnect();
  visible.clear();
  stale.clear();
  root.querySelectorAll("[data-rf-card]").forEach(card => stale.add(card.dataset.rfCard));
  stale.add("__original");
  observer = new IntersectionObserver(items => {
    for (const entry of items) {
      const key = entry.target.dataset.rfCard;
      if (entry.isIntersecting) visible.add(key); else visible.delete(key);
    }
    flush();
  }, { rootMargin: "300px 0px" });
  root.querySelectorAll("[data-rf-card]").forEach(card => observer.observe(card));
  if (wired !== root) {
    wired = root;
    root.addEventListener("input", onInput);
    root.addEventListener("change", onChange);
    root.addEventListener("click", onClick);
    root.addEventListener("toggle", () => flush(), true);
  }
}

/** Another field changed (resolution, the variant): the previews depend on the resolution. */
export function rfAugmentationArgChanged(key) {
  if (!root?.isConnected || !catalog) return;
  if (key === "resolution" || key === "model") markAll();
}

function markAll() {
  root.querySelectorAll("[data-rf-card]").forEach(card => stale.add(card.dataset.rfCard));
  stale.add("__original");
  scheduleFlush();
}

const scheduleFlush = debounce(() => flush(), 220);

function flush() {
  if (!root?.isConnected || !root.open || !catalog) return;
  if (stale.has("__original")) {
    stale.delete("__original");
    loadOriginal();
  }
  for (const key of [...stale]) {
    if (!visible.has(key)) continue;
    stale.delete(key);
    if (key === "__samples") loadSamples(); else loadCard(Number(key));
  }
}

// ---------- editing ----------

// a copy of the current transforms (the default one materialized), changed by `edit`, stored as aug_config
function update(edit, { structural = false, card = null } = {}) {
  const next = Object.fromEntries(entries().map(([name, params]) => [name, params && typeof params === "object" && !Array.isArray(params) ? { ...params } : params]));
  const result = edit(next) ?? next;
  ctx.setArg("aug_config", result);
  ctx.changed();
  if (structural) {
    rerender();
    return;
  }
  stale.add("__samples");
  if (card !== null) stale.add(String(card));
  syncPreset();
  scheduleFlush();
}

function syncPreset() {
  const preset = presetOf(config());
  const radio = root.querySelector(`input[name=rfaug-preset][value="${preset}"]`);
  if (radio) radio.checked = true;
  else rerender();  // "Custom" appears or goes
}

function setParams(index, change) {
  update(next => {
    const [name] = Object.entries(next)[index];
    const params = next[name] && typeof next[name] === "object" && !Array.isArray(next[name]) ? next[name] : {};
    change(params);
    next[name] = params;
  }, { card: index });
}

function cardOf(target) {
  const card = target.closest("[data-rf-card]");
  return card && card.dataset.rfCard !== "__samples" ? Number(card.dataset.rfCard) : null;
}

function onInput(e) {
  const target = e.target;
  const index = cardOf(target);
  if (index === null) return;
  const card = root.querySelector(`[data-rf-card="${index}"]`);
  if (target.matches("[data-rf-p]")) {
    const value = Number(target.value);
    card.querySelector("[data-rf-p-number]").value = value;
    setParams(index, params => { params.p = value; });
  } else if (target.matches("[data-rf-p-number]")) {
    const raw = target.value.trim();
    if (raw !== "" && !Number.isFinite(Number(raw))) return;
    if (raw !== "") card.querySelector("[data-rf-p]").value = Number(raw);
    setParams(index, params => { if (raw === "") delete params.p; else params.p = Number(raw); });
  } else if (target.matches("input[type=text][data-rf-param]")) {
    const value = parseValue(target.value);
    target.closest(".field").classList.toggle("changed", value !== undefined);
    target.closest(".field").classList.remove("rfaug-required");
    setParams(index, params => { if (value === undefined) delete params[target.dataset.rfParam]; else params[target.dataset.rfParam] = value; });
  }
}

function onChange(e) {
  const target = e.target;
  if (target.name === "rfaug-preset") {
    const id = target.value;
    if (id === "custom") return;
    ctx.setArg("aug_config", id === "default" ? undefined : id === "none" ? {} : structuredClone(catalog.presets[id]));
    ctx.changed();
    rerender();
    return;
  }
  if (target.matches("[data-rf-add]")) {
    const name = target.value;
    if (!name) return;
    const item = info(name);
    const required = Object.fromEntries((item?.params ?? []).filter(param => param.required).map(param => [param.name, param.name === "size" ? [ctx.resolution(), ctx.resolution()] : param.default]).filter(([, value]) => value !== null && value !== undefined));
    update(next => { next[name] = { ...required, p: item?.p ?? 0.5 }; }, { structural: true });
    return;
  }
  if (target.matches("[data-rf-labels]")) {
    showLabels = target.checked;
    markAll();
    stale.add("__samples");
    return;
  }
  const index = cardOf(target);
  if (index === null) return;
  if (target.matches("input[type=checkbox][data-rf-param]")) {
    const param = info(entries()[index][0])?.params.find(item => item.name === target.dataset.rfParam);
    const value = target.checked;
    target.closest(".form-toggle").classList.toggle("changed", value !== param?.default);
    setParams(index, params => { if (value === param?.default) delete params[target.dataset.rfParam]; else params[target.dataset.rfParam] = value; });
  } else if (target.matches("select[data-rf-param]")) {
    const value = target.value === "" ? undefined : JSON.parse(target.value);
    target.closest(".field").classList.toggle("changed", value !== undefined);
    setParams(index, params => { if (value === undefined) delete params[target.dataset.rfParam]; else params[target.dataset.rfParam] = value; });
  }
}

function onClick(e) {
  const step = e.target.closest("[data-rf-sample]");
  if (step) {
    sample = Math.max(0, sample + Number(step.dataset.rfSample));
    markAll();
    stale.add("__samples");
    return;
  }
  if (e.target.closest("[data-rf-resample]")) {
    resample += 1;
    loadSamples();
    return;
  }
  const zoom = e.target.closest("[data-rf-zoom]");
  if (zoom) return openZoom(zoom.dataset.rfZoom, Number(zoom.dataset.index));
  const index = cardOf(e.target);
  if (index === null) return;
  if (e.target.closest("[data-rf-remove]")) {
    update(next => { delete next[entries()[index][0]]; }, { structural: true });
  } else if (e.target.closest("[data-rf-move]")) {
    const to = index + Number(e.target.closest("[data-rf-move]").dataset.rfMove);
    update(next => {
      const items = Object.entries(next);
      [items[index], items[to]] = [items[to], items[index]];
      return Object.fromEntries(items);
    }, { structural: true });
  }
}

// ---------- loading ----------

const query = params => new URLSearchParams({ ...params, resolution: ctx.resolution(), labels: String(showLabels) }).toString();
const originalUrl = size => apiUrl(`/training/augment/rfdetr?${query({ index: -1, sample, size })}`);
// one transform alone, always applied: only it goes into the URL, so its previews stay cached while others change
const drawUrl = (index, side, size) => {
  const [name, params] = entries()[index];
  return apiUrl(`/training/augment/rfdetr?${query({ aug: JSON.stringify({ [name]: params }), index: 0, sample, seed: side === "a" ? 1 : 2, size })}`);
};
const sampleUrl = (index, size) => apiUrl(`/training/augment/rfdetr?${query({ aug: isDefault() ? "null" : JSON.stringify(config()), sample: sample + index, seed: resample * SAMPLE_COUNT + index + 1, size })}`);

async function loadOriginal() {
  const note = await loadImage(root.querySelector("[data-rf-original]"), originalUrl(TILE_SIZE));
  const name = root.querySelector("[data-rf-sample-name]");
  if (note !== null && name) name.textContent = note;
}

function loadCard(index) {
  const card = root.querySelector(`[data-rf-card="${index}"]`);
  if (!card || catalog.available === false) return;
  card.querySelectorAll("[data-rf-tile]").forEach(img => {
    const side = img.dataset.rfTile;
    if (side === "original") loadImage(img, originalUrl(TILE_SIZE));
    else loadImage(img, drawUrl(index, side, TILE_SIZE), card.querySelector(`[data-rf-note="${side}"]`));
  });
}

function loadSamples() {
  if (catalog.available === false && !isDefault()) return;
  root.querySelectorAll("[data-rf-sample-img]").forEach(img => loadImage(img, sampleUrl(Number(img.dataset.rfSampleImg), TILE_SIZE)));
}

function openZoom(key, index) {
  if (key === "original") return showZoom("Preview image", [["original", originalUrl(ZOOM_SIZE), root.querySelector("[data-rf-sample-name]").textContent]]);
  if (key === "sample") return showZoom("Training sample", [["sample", sampleUrl(index, ZOOM_SIZE), `Training sample ${index + 1}`]]);
  const card = Number(key);
  showZoom(entries()[card][0], [["original", originalUrl(ZOOM_SIZE), "original"], ["a", drawUrl(card, "a", ZOOM_SIZE), "random draw 1"], ["b", drawUrl(card, "b", ZOOM_SIZE), "random draw 2"]]);
}
