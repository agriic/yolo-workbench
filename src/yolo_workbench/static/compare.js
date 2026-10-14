// Run comparison (Training tab): the same metric for every selected run on one chart, two charts
// side by side, a results table, and only the arguments that differ. Each run keeps its color
// slot while it stays selected (color follows the run, not its position).
import { api, esc } from "./api.js";
import { LineChart, formatValue } from "./charts.js";

export const MAX_COMPARE = 8;  // validated categorical slots (--series-1..8)
const MANAGED = new Set(["data", "project", "name", "exist_ok", "resume", "mode", "task", "save_dir"]);

let charts = [];
let loaded = [];      // [{run, rows, slot}]
let picks = [null, null];

export function destroyCompare() {
  charts.forEach(chart => chart.destroy());
  charts = [];
}

const primaryMetric = category => category === "classification" ? "metrics/accuracy_top1" : category === "segmentation" ? "metrics/mAP50-95(M)" : "metrics/mAP50-95(B)";

function metricLabel(key) {
  if (key === "fitness") return "fitness";
  if (key.startsWith("lr/")) return key === "lr/pg0" ? "learning rate" : key;
  if (key.startsWith("train/") || key.startsWith("val/")) return key.replace("/", " ").replace(/_loss$/, " loss");
  return key.replace("metrics/", "").replace("precision", "P").replace("recall", "R").replace("accuracy_", "").replace("(B)", " box").replace("(M)", " mask");
}

/**
 * @param {HTMLElement} container
 * @param {{ids: string[], slots: Map<string, number>, category: string, defaults: object|null, onClose: () => void, onOpen: (id) => void}} options
 */
export async function renderCompare(container, options) {
  destroyCompare();
  container.innerHTML = `<div class="empty">Loading ${options.ids.length} runs…</div>`;
  try {
    loaded = await Promise.all(options.ids.map(async id => {
      const [run, epochs] = await Promise.all([api(`/training/runs/${encodeURIComponent(id)}`), api(`/training/runs/${encodeURIComponent(id)}/epochs`)]);
      return { run, rows: epochs.items, slot: options.slots.get(id) };
    }));
  } catch (error) {
    container.innerHTML = `<div class="empty">${esc(error.message)}</div>`;
    return;
  }
  const keys = [...new Set(loaded.flatMap(item => item.rows.flatMap(row => Object.keys(row))))]
    .filter(key => key !== "epoch" && key !== "time" && key !== "is_best" && key !== "epoch_time" && key !== "gpu_mem" && (key === "fitness" || key.includes("/")) && !/^lr\/pg[12]$/.test(key));
  const groups = [
    ["Validation metrics", keys.filter(key => key.startsWith("metrics/"))],
    ["Validation loss", keys.filter(key => key.startsWith("val/"))],
    ["Train loss", keys.filter(key => key.startsWith("train/"))],
    ["Other", keys.filter(key => key === "fitness" || key.startsWith("lr/"))],
  ].filter(([, items]) => items.length);
  const primary = primaryMetric(options.category);
  const valLoss = keys.find(key => key === "val/box_loss") ?? keys.find(key => key.startsWith("val/")) ?? keys.find(key => key.startsWith("train/"));
  picks = [keys.includes(picks[0]) ? picks[0] : keys.includes(primary) ? primary : keys[0], keys.includes(picks[1]) ? picks[1] : valLoss ?? keys[1]];
  const select = index => `<select data-compare-metric="${index}" aria-label="Metric for chart ${index + 1}">${groups.map(([title, items]) =>
    `<optgroup label="${esc(title)}">${items.map(key => `<option value="${esc(key)}" ${key === picks[index] ? "selected" : ""}>${esc(metricLabel(key))}</option>`).join("")}</optgroup>`).join("")}</select>`;

  container.innerHTML = `
    <div class="run-head">
      <div class="run-title"><h1>Compare ${loaded.length} runs</h1><p class="section-subtitle">Same metric per run on each chart; the table lists only arguments that differ.</p></div>
      <div class="run-actions"><button type="button" class="secondary" data-compare-close>Close</button></div>
    </div>
    <div class="stats-grid run-grid">
      ${[0, 1].map(index => `<article class="stats-panel"><div class="panel-head"><h2>Chart ${index + 1}</h2>${select(index)}</div><div id="compare-chart-${index}"></div></article>`).join("")}
      <article class="stats-panel stats-wide"><h2>Results</h2><div class="table-scroll">${resultsTable(options.category)}</div></article>
      <article class="stats-panel stats-wide"><h2>Arguments that differ</h2><div class="table-scroll">${argsDiff(options.defaults)}</div></article>
    </div>`;
  charts = [0, 1].map(index => new LineChart(container.querySelector(`#compare-chart-${index}`), { label: `Comparison chart ${index + 1}` }));
  drawCharts();
  container.querySelectorAll("[data-compare-metric]").forEach(el => el.addEventListener("change", () => {
    picks[+el.dataset.compareMetric] = el.value;
    drawCharts();
  }));
  container.querySelector("[data-compare-close]").onclick = options.onClose;
  container.querySelectorAll("[data-open-run]").forEach(el => el.addEventListener("click", () => options.onOpen(el.dataset.openRun)));
}

function drawCharts() {
  const epochs = [...new Set(loaded.flatMap(item => item.rows.map(row => row.epoch)))].sort((a, b) => a - b);
  charts.forEach((chart, index) => {
    const key = picks[index];
    const series = loaded.map(item => {
      const byEpoch = new Map(item.rows.map(row => [row.epoch, row[key]]));
      return {
        key: item.run.id, label: item.run.name, group: item.run.id, groupLabel: item.run.name,
        color: `var(--series-${item.slot})`, values: epochs.map(epoch => byEpoch.get(epoch) ?? null),
      };
    });
    const isRatio = key.startsWith("metrics/");
    chart.options.yMin = isRatio ? 0 : undefined;
    chart.update({ x: epochs, series, empty: "No epochs recorded" });
  });
}

const shortModel = model => {
  const parts = String(model ?? "?").split("/");
  return parts.length > 2 && parts[parts.length - 2] === "weights" ? `${parts[parts.length - 3]}/${parts[parts.length - 1]}` : parts[parts.length - 1];
};

function bestRow(rows) {
  let best = null;
  for (const row of rows) if (row.fitness !== null && row.fitness !== undefined && (!best || row.fitness >= best.fitness)) best = row;
  return best;
}

function resultsTable(category) {
  const metric = primaryMetric(category);
  const label = metricLabel(metric);
  const reference = loaded[0]?.run.dataset;
  const datasetCell = run => {
    const info = run.dataset;
    if (!info) return `<span class="muted">not recorded</span>`;
    const counts = ["train", "val", "test"].map(split => info.splits?.[split] ?? 0).join("/");
    const differs = reference && info.fingerprint !== reference.fingerprint;
    return `<span class="${differs ? "warn-text" : ""}" title="Dataset fingerprint ${esc(info.fingerprint)} (train/val/test images)${differs ? " — differs from the first run" : ""}">${esc(counts)}${differs ? " · changed" : ""}</span>`;
  };
  return `<table class="stats-table compare-table"><thead><tr><th>Run</th><th>Model</th><th>Epochs</th><th>Best epoch</th><th>Best ${esc(label)} (val)</th><th>Final val ${esc(label)}</th><th>Test ${esc(label)}</th><th>Data (train/val/test)</th></tr></thead><tbody>
    ${loaded.map(({ run, rows, slot }) => {
      const best = bestRow(rows);
      return `<tr>
        <td><button type="button" class="link-btn compare-run" data-open-run="${esc(run.id)}"><svg width="16" height="8" aria-hidden="true"><line x1="0" y1="4" x2="16" y2="4" stroke="var(--series-${slot})" stroke-width="2.5" stroke-linecap="round"/></svg>${esc(run.name)}</button></td>
        <td title="${esc(run.model ?? "")}">${esc(shortModel(run.model))}</td>
        <td>${rows.length ? rows[rows.length - 1].epoch : 0}/${esc(run.epochs ?? "?")}</td>
        <td>${best ? `★ ${best.epoch}` : "–"}</td>
        <td>${formatValue(best?.[metric])}</td>
        <td>${formatValue(run.val_metrics?.[metric])}</td>
        <td>${formatValue(run.test_metrics?.[metric])}</td>
        <td>${datasetCell(run)}</td>
      </tr>`;
    }).join("")}</tbody></table>`;
}

function argsDiff(defaults) {
  const normalized = (key, args) => {
    const value = args?.[key];
    if (value === undefined || value === "" || value === null) return defaults && key in defaults ? defaults[key] : null;
    return value;
  };
  // the base model (a resume rewrites args.yaml's model to the run's own last.pt)
  for (const item of loaded) item.run.args = { ...(item.run.args || {}), model: shortModel(item.run.model) };
  const keys = [...new Set(loaded.flatMap(item => Object.keys(item.run.args || {})))].filter(key => !MANAGED.has(key));
  const differing = keys.filter(key => new Set(loaded.map(item => JSON.stringify(normalized(key, item.run.args)))).size > 1);
  if (!differing.length) return `<p class="empty-note">All runs used the same arguments.</p>`;
  const show = value => value === null || value === undefined ? "–" : typeof value === "object" ? JSON.stringify(value) : String(value);
  return `<table class="stats-table compare-table args-diff"><thead><tr><th>Argument</th>${loaded.map(({ run, slot }) => `<th><svg width="12" height="6" aria-hidden="true"><line x1="0" y1="3" x2="12" y2="3" stroke="var(--series-${slot})" stroke-width="2.5"/></svg> ${esc(run.name)}</th>`).join("")}</tr></thead><tbody>
    ${differing.map(key => `<tr><td><code>${esc(key)}</code></td>${loaded.map(item => {
      const value = normalized(key, item.run.args);
      const isDefault = defaults && key in defaults && JSON.stringify(value) === JSON.stringify(defaults[key]);
      return `<td class="${isDefault ? "muted" : ""}" title="${isDefault ? "Ultralytics default" : ""}">${esc(show(value))}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table>`;
}
