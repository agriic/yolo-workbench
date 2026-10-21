import { $, api, esc } from "./api.js";
import { state } from "./state.js";
let deps;

// Repair tools for what the automatic fixes cannot decide: the raw label editor and unknown class ids.
export function configureRepair(dependencies) {
  deps = dependencies;
}

let editing = null;  // { imageId, revision } of the label file open in the raw editor

export function bindRepair() {
  $("label-text-input").addEventListener("input", renderLineProblems);
  $("label-text-problems").addEventListener("click", e => {
    const row = e.target.closest("[data-line]");
    if (row) selectLine(+row.dataset.line);
  });
  $("label-text-form").addEventListener("submit", e => {
    if (e.submitter?.value === "save") { e.preventDefault(); saveLabelText(); }
  });
  $("label-text-dialog").addEventListener("close", () => { editing = null; });
  $("unknown-classes").addEventListener("change", e => {
    const row = e.target.closest("[data-unknown]");
    if (row && e.target.matches("select")) row.querySelector("input").hidden = e.target.value !== "add";
  });
  $("unknown-classes").addEventListener("click", e => {
    const button = e.target.closest("[data-resolve]");
    if (button) resolveUnknown(button.closest("[data-unknown]"));
  });
}

/* ---------- raw label editor ---------- */

export async function openLabelText(imageId) {
  try {
    const data = await api(`/images/${imageId}/label-text`);
    editing = { imageId, revision: data.revision };
    $("label-text-title").textContent = data.name;
    $("label-text-path").textContent = data.exists ? data.path : `${data.path} (does not exist yet; saving creates it)`;
    $("label-text-input").value = data.text;
    $("label-text-format").textContent = state.meta.category === "detection"
      ? "One object per line: class cx cy w h, normalized 0..1."
      : "One object per line: class x1 y1 x2 y2 x3 y3 …, normalized 0..1.";
    renderLineProblems();
    $("label-text-dialog").showModal();
    const first = lineProblems($("label-text-input").value)[0];
    if (first) selectLine(first.line);
  } catch (error) { deps.toast(error.message, "error"); }
}

// The same checks as the server's parser, so problems show while typing.
function lineProblems(text) {
  const detection = state.meta.category === "detection";
  const problems = [];
  text.split("\n").forEach((raw, index) => {
    const line = raw.trim();
    if (!line) return;
    const add = message => problems.push({ line: index + 1, message });
    const [token, ...rest] = line.split(/\s+/);
    if (!/^[+-]?\d+$/.test(token)) return add(`class id “${token}” is not a whole number`);
    const values = rest.map(Number);
    if (values.some(value => !Number.isFinite(value))) return add("a coordinate is not a number");
    if (detection ? values.length !== 4 : values.length < 6 || values.length % 2)
      return add(detection ? `needs 4 coordinates (cx cy w h), has ${values.length}` : `needs an even number of at least 6 coordinates, has ${values.length}`);
    if (!(String(+token) in state.meta.names)) add(`class ${+token} is not a dataset class`);
    else if (values.some(value => value < 0 || value > 1)) add("coordinates outside 0..1");
  });
  return problems;
}

function renderLineProblems() {
  const problems = lineProblems($("label-text-input").value);
  $("label-text-problems").innerHTML = problems.length
    ? problems.slice(0, 50).map(problem => `<button type="button" class="line-problem" data-line="${problem.line}"><strong>Line ${problem.line}</strong> ${esc(problem.message)}</button>`).join("")
      + (problems.length > 50 ? `<p class="panel-note">…and ${problems.length - 50} more</p>` : "")
    : `<p class="panel-note ok-note">Every line parses.</p>`;
}

function selectLine(number) {
  const input = $("label-text-input");
  const lines = input.value.split("\n");
  const start = lines.slice(0, number - 1).reduce((sum, line) => sum + line.length + 1, 0);
  input.focus();
  input.setSelectionRange(start, start + (lines[number - 1]?.length ?? 0));
  // scroll the selected line into view (textareas do not do it on setSelectionRange everywhere)
  const lineHeight = parseFloat(getComputedStyle(input).lineHeight) || 18;
  input.scrollTop = Math.max(0, (number - 3) * lineHeight);
}

async function saveLabelText() {
  if (!editing) return;
  const { imageId, revision } = editing;
  const button = $("label-text-save");
  button.disabled = true;
  try {
    const detail = await api(`/images/${imageId}/label-text`, { method: "PUT", body: JSON.stringify({ revision, schema_generation: state.meta.schema_generation, text: $("label-text-input").value }) });
    $("label-text-dialog").close();
    const malformed = detail.issues.filter(issue => issue.kind === "malformed_label").length;
    deps.toast(malformed ? `Saved · ${malformed} line${malformed === 1 ? "" : "s"} still malformed — Ctrl+Z undoes` : "Label file saved — Ctrl+Z undoes", malformed ? "error" : "info");
    deps.labelsChanged(imageId, detail);
  } catch (error) {
    deps.toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

/* ---------- unknown class ids ---------- */

export function renderUnknownClasses(items) {
  const panel = $("unknown-classes");
  panel.hidden = !items.length;
  if (!items.length) { panel.innerHTML = ""; return; }
  const options = Object.entries(state.meta.names).map(([id, name]) => `<option value="map:${esc(id)}">Map to ${esc(id)} · ${esc(name)}</option>`).join("");
  panel.innerHTML = `<h2>Unknown class ids</h2>
    <p class="panel-note">Label files use class ids that data.yaml does not name. Images with them cannot be saved in the editor until the id is resolved. Ctrl+Z undoes each change.</p>
    ${items.map(item => `<div class="unknown-row" data-unknown="${item.id}">
      <span class="unknown-id">${esc(item.id)}</span>
      <span class="muted">${item.annotations.toLocaleString()} annotation${item.annotations === 1 ? "" : "s"} in ${item.images.toLocaleString()} image${item.images === 1 ? "" : "s"}</span>
      <select aria-label="What to do with class ${esc(item.id)}">${options}<option value="add">Add as a new class…</option><option value="delete">Delete these annotations</option></select>
      <input type="text" placeholder="New class name" aria-label="New class name" ${Object.keys(state.meta.names).length ? "hidden" : ""}>
      <button type="button" class="secondary mini" data-resolve>Apply</button>
    </div>`).join("")}`;
  // with no dataset classes the only choices are add and delete
  panel.querySelectorAll("[data-unknown] select").forEach(select => { if (!Object.keys(state.meta.names).length) select.value = "add"; });
}

async function resolveUnknown(row) {
  const classId = +row.dataset.unknown;
  const choice = row.querySelector("select").value;
  const body = choice.startsWith("map:") ? { action: "map", target_id: +choice.slice(4) }
    : choice === "add" ? { action: "add", name: row.querySelector("input").value.trim() }
    : { action: "delete" };
  if (body.action === "add" && !body.name) return deps.toast("Enter a name for the new class", "error");
  if (body.action === "delete" && !confirm(`Delete every annotation with class ${classId}? Ctrl+Z undoes.`)) return;
  try {
    const result = await api(`/classes/unknown/${classId}`, { method: "POST", body: JSON.stringify(body) });
    state.canUndo = result.can_undo;
    state.canRedo = result.can_redo;
    deps.toast(body.action === "add" ? `Added class “${body.name}” for id ${classId} — Ctrl+Z undoes` : `Resolved class ${classId} in ${result.files} file${result.files === 1 ? "" : "s"} — Ctrl+Z undoes`);
    await deps.datasetChanged({ schema: body.action === "add" });
  } catch (error) { deps.toast(error.message, "error"); }
}
