// Small SVG line chart for training curves: one y axis, live updates, a crosshair tooltip that
// lists every series at the hovered epoch, a legend that toggles series groups, and an optional
// vertical marker (the best epoch). Colors come from CSS variables (--series-N), so dark mode
// swaps them in styles.css.

const SVG = "http://www.w3.org/2000/svg";
const HEIGHT = 230;
const MARGIN = { top: 14, right: 14, bottom: 26, left: 48 };

const node = (tag, attrs = {}, parent = null) => {
  const el = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  if (parent) parent.appendChild(el);
  return el;
};

function niceStep(span, count) {
  const raw = span / Math.max(count, 1);
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  return (unit >= 5 ? 10 : unit >= 2 ? 5 : unit >= 1 ? 2 : 1) * power;
}

function ticks(low, high, count) {
  if (!(high > low)) return [low];
  const step = niceStep(high - low, count);
  const out = [];
  for (let value = Math.ceil(low / step) * step; value <= high + step * 1e-9; value += step) out.push(+value.toPrecision(12));
  return out;
}

// axis labels: as short as the tick step allows (0.5, 1, 1.25 — not 1.000)
const formatTick = value => String(Number(value.toPrecision(6)));

export function formatValue(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1000) return Math.round(value).toLocaleString();
  if (magnitude < 0.001) return value.toExponential(1);
  if (magnitude < 0.01) return value.toPrecision(2);
  return value.toFixed(magnitude >= 100 ? 1 : 3);
}

export class LineChart {
  /**
   * @param {HTMLElement} container
   * @param {{label: string, yMin?: number, yMax?: number}} options  label names the chart for screen readers
   */
  constructor(container, options) {
    this.options = options;
    this.data = { x: [], series: [], marker: null, xMax: 0 };
    this.hidden = new Set();   // hidden series groups
    this.hoverIndex = null;
    container.classList.add("line-chart");
    container.innerHTML = "";
    this.legend = document.createElement("div");
    this.legend.className = "chart-legend";
    this.frame = document.createElement("div");
    this.frame.className = "chart-frame";
    this.tooltip = document.createElement("div");
    this.tooltip.className = "chart-tooltip";
    this.tooltip.hidden = true;
    this.empty = document.createElement("div");
    this.empty.className = "chart-empty";
    this.svg = node("svg", { role: "img", "aria-label": options.label, tabindex: "0", height: HEIGHT });
    this.frame.append(this.svg, this.tooltip, this.empty);
    container.append(this.legend, this.frame);
    this.svg.addEventListener("pointermove", e => this.hoverAt(e.clientX));
    this.svg.addEventListener("pointerleave", () => this.setHover(null));
    this.svg.addEventListener("blur", () => this.setHover(null));
    this.svg.addEventListener("keydown", e => this.onKey(e));
    this.legend.addEventListener("click", e => {
      const item = e.target.closest("[data-group]");
      if (!item) return;
      const group = item.dataset.group;
      if (this.hidden.has(group)) this.hidden.delete(group); else this.hidden.add(group);
      this.render();
    });
    this.width = 0;
    this.observer = new ResizeObserver(() => {
      const width = this.frame.clientWidth;
      if (width && width !== this.width) { this.width = width; this.render(); }
    });
    this.observer.observe(this.frame);
  }

  /**
   * @param {{x: number[], series: {key, label, group, groupLabel, color, dashed?, values: (number|null)[]}[],
   *          marker?: number|null, markerLabel?: string, xMax?: number, styles?: {label, dashed}[], empty?: string}} data
   */
  update(data) {
    this.data = data;
    this.renderLegend();
    this.render();
  }

  destroy() { this.observer.disconnect(); }

  renderLegend() {
    const groups = [];
    for (const series of this.data.series) if (!groups.some(group => group.group === series.group)) groups.push(series);
    this.legend.replaceChildren();
    if (groups.length + (this.data.styles?.length ?? 0) < 2) return;  // one series: the panel title names it
    for (const series of groups) {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "legend-item";
      item.dataset.group = series.group;
      item.setAttribute("aria-pressed", String(!this.hidden.has(series.group)));
      item.title = "Show or hide";
      item.innerHTML = `<svg width="16" height="8" aria-hidden="true"><line x1="0" y1="4" x2="16" y2="4" stroke="${series.color}" stroke-width="2.5" stroke-linecap="round"/></svg>`;
      item.append(document.createTextNode(series.groupLabel));
      this.legend.append(item);
    }
    for (const style of this.data.styles ?? []) {
      const key = document.createElement("span");
      key.className = "legend-style";
      key.innerHTML = `<svg width="18" height="8" aria-hidden="true"><line x1="1" y1="4" x2="17" y2="4" stroke="currentColor" stroke-width="2" ${style.dashed ? 'stroke-dasharray="4 3"' : ""}/></svg>`;
      key.append(document.createTextNode(style.label));
      this.legend.append(key);
    }
  }

  visibleSeries() {
    return this.data.series.filter(series => !this.hidden.has(series.group));
  }

  layout() {
    const { x } = this.data;
    const width = Math.max(this.width || this.frame.clientWidth || 400, 200);
    const xLow = 1;
    const xHigh = Math.max(x.length ? x[x.length - 1] : 1, 2);  // the data's extent: early epochs stay readable
    let yLow = this.options.yMin ?? Infinity;
    let yHigh = this.options.yMax ?? -Infinity;
    for (const series of this.visibleSeries()) for (const value of series.values) {
      if (value === null || value === undefined) continue;
      if (this.options.yMin === undefined) yLow = Math.min(yLow, value);
      if (this.options.yMax === undefined) yHigh = Math.max(yHigh, value);
    }
    if (!Number.isFinite(yLow) || !Number.isFinite(yHigh)) { yLow = 0; yHigh = 1; }
    if (yHigh === yLow) { yHigh += 1; yLow = Math.max(0, yLow - 1); }
    const pad = (yHigh - yLow) * 0.06;
    if (this.options.yMin === undefined) yLow = Math.max(0, yLow - pad);  // losses and metrics are never negative
    if (this.options.yMax === undefined) yHigh += pad;
    const plotWidth = width - MARGIN.left - MARGIN.right;
    const plotHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
    return {
      width, xLow, xHigh, yLow, yHigh,
      sx: value => MARGIN.left + (xHigh === xLow ? 0 : (value - xLow) / (xHigh - xLow)) * plotWidth,
      sy: value => MARGIN.top + (1 - (value - yLow) / (yHigh - yLow)) * plotHeight,
      plotWidth, plotHeight,
    };
  }

  render() {
    const svg = this.svg;
    svg.replaceChildren();
    const { x, series } = this.data;
    this.empty.hidden = x.length > 0;
    this.empty.textContent = this.data.empty ?? "No epochs yet";
    if (!x.length || !series.length) { this.tooltip.hidden = true; return; }
    const geometry = this.geometry = this.layout();
    const { width, sx, sy } = geometry;
    svg.setAttribute("width", width);
    svg.setAttribute("viewBox", `0 0 ${width} ${HEIGHT}`);

    const grid = node("g", { class: "chart-grid" }, svg);
    for (const value of ticks(geometry.yLow, geometry.yHigh, 4)) {
      const y = sy(value);
      node("line", { x1: MARGIN.left, x2: width - MARGIN.right, y1: y, y2: y }, grid);
      node("text", { x: MARGIN.left - 7, y: y + 4, "text-anchor": "end", class: "chart-tick" }, grid).textContent = formatTick(value);
    }
    const xCount = Math.max(2, Math.floor(geometry.plotWidth / 70));
    for (const value of ticks(geometry.xLow, geometry.xHigh, xCount).filter(value => Number.isInteger(value) && value >= 1)) {
      node("text", { x: sx(value), y: HEIGHT - 8, "text-anchor": "middle", class: "chart-tick" }, grid).textContent = value;
    }
    node("line", { x1: MARGIN.left, x2: width - MARGIN.right, y1: HEIGHT - MARGIN.bottom, y2: HEIGHT - MARGIN.bottom, class: "chart-axis" }, grid);

    if (this.data.marker) {
      const mx = sx(this.data.marker);
      const marker = node("g", { class: "chart-marker" }, svg);
      node("line", { x1: mx, x2: mx, y1: MARGIN.top - 4, y2: HEIGHT - MARGIN.bottom }, marker);
      const label = node("text", { x: mx + (mx > width - 80 ? -5 : 5), y: MARGIN.top + 6, "text-anchor": mx > width - 80 ? "end" : "start" }, marker);
      label.textContent = this.data.markerLabel ?? `best ${this.data.marker}`;
    }

    const lines = node("g", { class: "chart-lines" }, svg);
    for (const item of this.visibleSeries()) {
      let path = "";
      let pen = false;
      item.values.forEach((value, index) => {
        if (value === null || value === undefined) { pen = false; return; }
        path += `${pen ? "L" : "M"}${sx(x[index]).toFixed(1)},${sy(value).toFixed(1)}`;
        pen = true;
      });
      const attrs = { d: path, stroke: item.color, fill: "none", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" };
      if (item.dashed) attrs["stroke-dasharray"] = "5 4";
      node("path", attrs, lines);
      // a lone point (first epoch) has no segment to draw
      const points = item.values.filter(value => value !== null && value !== undefined);
      if (points.length === 1) {
        const index = item.values.findIndex(value => value !== null && value !== undefined);
        node("circle", { cx: sx(x[index]), cy: sy(item.values[index]), r: 3, fill: item.color }, lines);
      }
    }
    this.hoverLayer = node("g", { class: "chart-hover" }, svg);
    if (this.hoverIndex !== null) this.setHover(Math.min(this.hoverIndex, x.length - 1));
  }

  hoverAt(clientX) {
    if (!this.geometry || !this.data.x.length) return;
    const box = this.svg.getBoundingClientRect();
    const px = clientX - box.left;
    let best = 0;
    let distance = Infinity;
    this.data.x.forEach((value, index) => {
      const d = Math.abs(this.geometry.sx(value) - px);
      if (d < distance) { distance = d; best = index; }
    });
    this.setHover(best);
  }

  onKey(e) {
    if (!this.data.x.length) return;
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight" && e.key !== "Home" && e.key !== "End") return;
    e.preventDefault();
    const last = this.data.x.length - 1;
    const current = this.hoverIndex ?? last;
    const next = e.key === "Home" ? 0 : e.key === "End" ? last : current + (e.key === "ArrowLeft" ? -1 : 1);
    this.setHover(Math.max(0, Math.min(last, next)));
  }

  setHover(index) {
    this.hoverIndex = index;
    if (!this.hoverLayer) return;
    this.hoverLayer.replaceChildren();
    if (index === null || !this.geometry) { this.tooltip.hidden = true; return; }
    const { sx, sy } = this.geometry;
    const epoch = this.data.x[index];
    const cx = sx(epoch);
    node("line", { x1: cx, x2: cx, y1: MARGIN.top, y2: HEIGHT - MARGIN.bottom, class: "chart-crosshair" }, this.hoverLayer);
    const rows = [];
    for (const item of this.visibleSeries()) {
      const value = item.values[index];
      if (value === null || value === undefined) continue;
      node("circle", { cx, cy: sy(value), r: 4, fill: item.color, class: "chart-dot" }, this.hoverLayer);
      rows.push(item);
    }
    const tip = this.tooltip;
    tip.replaceChildren();
    const head = document.createElement("div");
    head.className = "chart-tooltip-head";
    head.textContent = `Epoch ${epoch}${epoch === this.data.marker ? " · best" : ""}`;
    tip.append(head);
    for (const item of rows) {
      const row = document.createElement("div");
      row.className = "chart-tooltip-row";
      row.innerHTML = `<svg width="14" height="6" aria-hidden="true"><line x1="0" y1="3" x2="14" y2="3" stroke="${item.color}" stroke-width="2.5" ${item.dashed ? 'stroke-dasharray="4 2"' : ""}/></svg>`;
      const value = document.createElement("strong");
      value.textContent = formatValue(item.values[index]);
      const label = document.createElement("span");
      label.textContent = item.label;
      row.append(value, label);
      tip.append(row);
    }
    tip.hidden = false;
    const frameWidth = this.frame.clientWidth;
    const tipWidth = tip.offsetWidth;
    tip.style.left = `${cx + 14 + tipWidth > frameWidth ? cx - 14 - tipWidth : cx + 14}px`;
    tip.style.top = `${MARGIN.top}px`;
  }
}
