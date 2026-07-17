import { $, esc, clamp } from "./api.js";

export class VirtualGallery {
  constructor({ gridId, scrollId, railId, errorId, noun, emptyMessage, fetchPage, renderCard, isPinned = () => false, onUpdate = () => {} }) {
    this.grid = $(gridId);
    this.scroll = $(scrollId);
    this.rail = $(railId);
    this.error = $(errorId);
    this.noun = noun;
    this.emptyMessage = emptyMessage;
    this.fetchPage = fetchPage;
    this.renderCard = renderCard;
    this.isPinned = isPinned;
    this.onUpdate = onUpdate;
    this.total = null;
    this.cache = new Map();
    this.columns = 1;
    this.rowHeight = 276;
    this.gap = 14;
    this.window = { start: 0, end: 0 };
    this.generation = 0;
    this.windowAbort = null;
    this.rangeControllers = new Set();
    this.loadTimer = null;
    this.scrollFrame = null;
    this.labelTimer = null;
    this.reloading = false;

    this.scroll.addEventListener("scroll", () => {
      if (this.scrollFrame) return;
      this.scrollFrame = requestAnimationFrame(() => {
        this.scrollFrame = null;
        this.render();
        this.showPosition();
      });
    });
    this.error.querySelector("button").onclick = () => {
      this.hideError();
      this.loadWindow();
    };
    new ResizeObserver(() => this.refreshLayout()).observe(this.scroll);
  }

  refreshLayout() {
    const previousColumns = this.columns;
    const previousIndex = this.firstVisibleIndex();
    const computed = getComputedStyle(this.grid);
    const template = computed.gridTemplateColumns;
    const resolvedColumns = template && template !== "none" ? template.trim().split(/\s+/).length : 1;
    this.columns = Math.max(1, resolvedColumns);
    this.gap = parseFloat(computed.rowGap) || 14;
    this.rowHeight = parseFloat(computed.getPropertyValue("--virtual-card-height")) || this.rowHeight;
    if (previousColumns !== this.columns && this.total != null) {
      this.scroll.scrollTop = Math.floor(previousIndex / this.columns) * this.stride();
    }
    this.render();
  }

  stride() { return this.rowHeight + this.gap; }

  firstVisibleIndex() {
    return Math.max(0, Math.floor(this.scroll.scrollTop / this.stride()) * this.columns);
  }

  visibleBounds() {
    if (!this.total) return { start: 0, end: 0 };
    const firstRow = Math.floor(this.scroll.scrollTop / this.stride());
    const visibleRows = Math.max(1, Math.ceil((this.scroll.clientHeight || this.rowHeight * 6) / this.stride()));
    return {
      start: Math.min(this.total, firstRow * this.columns),
      end: Math.min(this.total, (firstRow + visibleRows) * this.columns),
    };
  }

  windowBounds() {
    if (!this.total) return { start: 0, end: 0 };
    const totalRows = Math.ceil(this.total / this.columns);
    const firstRow = Math.floor(this.scroll.scrollTop / this.stride());
    const visibleRows = Math.max(1, Math.ceil((this.scroll.clientHeight || this.rowHeight * 6) / this.stride()));
    const startRow = Math.max(0, firstRow - visibleRows * 2);
    const endRow = Math.min(totalRows, firstRow + visibleRows * 3);
    return { start: startRow * this.columns, end: Math.min(this.total, endRow * this.columns) };
  }

  async reload({ reset = false } = {}) {
    const anchor = reset ? 0 : this.firstVisibleIndex();
    const hadTotal = this.total != null;
    const generation = ++this.generation;
    this.abortRequests();
    this.reloading = true;
    this.cache.clear();
    this.hideError();
    if (reset) this.scroll.scrollTop = 0;
    if (!hadTotal) this.total = null;
    this.render();
    clearTimeout(this.loadTimer);

    const offset = Math.max(0, Math.floor(anchor / this.columns) * this.columns);
    const limit = Math.min(500, Math.max(50, this.windowItemCount()));
    const controller = new AbortController();
    this.windowAbort = controller;
    let succeeded = false;
    try {
      const data = await this.fetchPage(offset, limit, controller.signal);
      if (generation !== this.generation) return;
      this.applyPage(offset, data);
      if (this.total && offset >= this.total) {
        const last = this.total - 1;
        this.scroll.scrollTop = Math.floor(last / this.columns) * this.stride();
      }
      succeeded = true;
    } catch (error) {
      if (error.name !== "AbortError" && generation === this.generation) this.showError(error);
    } finally {
      if (this.windowAbort === controller) this.windowAbort = null;
      if (generation === this.generation) {
        this.reloading = false;
        if (succeeded) this.render();
      }
    }
  }

  abortRequests() {
    if (this.windowAbort) this.windowAbort.abort();
    this.windowAbort = null;
    for (const controller of this.rangeControllers) controller.abort();
    this.rangeControllers.clear();
    clearTimeout(this.loadTimer);
  }

  windowItemCount() {
    const rows = Math.max(6, Math.ceil((this.scroll.clientHeight || this.rowHeight * 6) / this.stride()) * 5);
    return rows * this.columns;
  }

  applyPage(offset, data) {
    this.total = data.total;
    data.items.forEach((item, index) => this.cache.set(offset + index, item));
    this.onUpdate(this.total);
  }

  render() {
    this.refreshMetrics();
    if (this.total == null) {
      this.grid.innerHTML = `<p class="empty">Loading ${this.noun.toLowerCase()}…</p>`;
      this.rail.hidden = true;
      return;
    }
    if (this.total === 0) {
      this.grid.innerHTML = `<p class="empty">${esc(this.emptyMessage)}</p>`;
      this.rail.hidden = true;
      return;
    }

    const { start, end } = this.windowBounds();
    this.window = { start, end };
    const startRow = Math.floor(start / this.columns);
    const renderedRows = Math.ceil((end - start) / this.columns);
    const totalRows = Math.ceil(this.total / this.columns);
    const remainingRows = Math.max(0, totalRows - startRow - renderedRows);
    const topHeight = startRow ? startRow * this.stride() - this.gap : 0;
    const bottomHeight = remainingRows ? remainingRows * this.stride() - this.gap : 0;
    const cards = [];
    for (let index = start; index < end; index++) {
      const item = this.cache.get(index);
      cards.push(item ? this.renderCard(item, index) : `<article class="virtual-placeholder" aria-hidden="true"></article>`);
    }
    this.grid.innerHTML = `${topHeight ? `<div class="virtual-spacer" style="height:${topHeight}px" aria-hidden="true"></div>` : ""}${cards.join("")}${bottomHeight ? `<div class="virtual-spacer" style="height:${bottomHeight}px" aria-hidden="true"></div>` : ""}`;
    this.renderRail(totalRows);
    if (!this.reloading) this.scheduleWindowLoad();
    this.evictDistant();
  }

  refreshMetrics() {
    const computed = getComputedStyle(this.grid);
    const template = computed.gridTemplateColumns;
    if (template && template !== "none") this.columns = Math.max(1, template.trim().split(/\s+/).length);
    this.gap = parseFloat(computed.rowGap) || this.gap;
    this.rowHeight = parseFloat(computed.getPropertyValue("--virtual-card-height")) || this.rowHeight;
  }

  renderRail(totalRows) {
    const overflow = totalRows * this.stride() - this.gap > this.scroll.clientHeight + 1;
    this.rail.hidden = !overflow;
    if (!overflow) return;
    const markers = [];
    const seen = new Set();
    for (const position of [0, .25, .5, .75, 1]) {
      const number = Math.round(position * (this.total - 1)) + 1;
      if (seen.has(number)) continue;
      seen.add(number);
      const approximate = position > 0 && position < 1 ? "~" : "";
      const endpoint = position === 0 ? " start" : position === 1 ? " end" : "";
      markers.push(`<span class="virtual-marker${endpoint}" style="top:${position * 100}%">${approximate}${number.toLocaleString()}</span>`);
    }
    this.rail.innerHTML = `${markers.join("")}<span class="virtual-position"></span>`;
  }

  showPosition() {
    const label = this.rail.querySelector(".virtual-position");
    if (!label || this.rail.hidden || !this.total) return;
    const { start, end } = this.visibleBounds();
    const maximum = Math.max(1, this.scroll.scrollHeight - this.scroll.clientHeight);
    const position = clamp(this.scroll.scrollTop / maximum, 0, 1);
    label.style.top = `${clamp(position * 100, 3, 97)}%`;
    label.textContent = `${this.noun} ${(start + 1).toLocaleString()}–${end.toLocaleString()} of ${this.total.toLocaleString()}`;
    label.classList.add("show");
    clearTimeout(this.labelTimer);
    this.labelTimer = setTimeout(() => label.classList.remove("show"), 900);
  }

  scheduleWindowLoad() {
    clearTimeout(this.loadTimer);
    const { start, end } = this.window;
    let first = -1, last = -1;
    for (let index = start; index < end; index++) {
      if (this.cache.has(index)) continue;
      if (first === -1) first = index;
      last = index;
    }
    if (first === -1) return;
    this.loadTimer = setTimeout(() => this.loadWindow(first, last + 1), 40);
  }

  async loadWindow(start = this.window.start, end = this.window.end) {
    if (this.windowAbort) this.windowAbort.abort();
    const generation = this.generation;
    const controller = new AbortController();
    this.windowAbort = controller;
    try {
      const limit = Math.min(500, Math.max(1, end - start));
      const data = await this.fetchPage(start, limit, controller.signal);
      if (generation !== this.generation) return;
      this.applyPage(start, data);
      this.hideError();
      this.render();
    } catch (error) {
      if (error.name !== "AbortError" && generation === this.generation) this.showError(error);
    } finally {
      if (this.windowAbort === controller) this.windowAbort = null;
    }
  }

  async ensureRange(start, end) {
    const generation = this.generation;
    const controller = new AbortController();
    this.rangeControllers.add(controller);
    try {
      for (let offset = start; offset < end;) {
        while (offset < end && this.cache.has(offset)) offset += 1;
        if (offset >= end) break;
        let missingEnd = offset;
        while (missingEnd < end && !this.cache.has(missingEnd) && missingEnd - offset < 500) missingEnd += 1;
        const data = await this.fetchPage(offset, missingEnd - offset, controller.signal);
        if (generation !== this.generation) throw new DOMException("Gallery changed", "AbortError");
        this.applyPage(offset, data);
        if (!data.items.length) break;
        offset = missingEnd;
      }
      return Array.from({ length: Math.max(0, Math.min(end, this.total ?? end) - start) }, (_, i) => this.cache.get(start + i)).filter(Boolean);
    } finally {
      this.rangeControllers.delete(controller);
    }
  }

  async itemAt(index) {
    if (index < 0 || (this.total != null && index >= this.total)) return null;
    if (!this.cache.has(index)) await this.ensureRange(index, index + 1);
    return this.cache.get(index) || null;
  }

  evictDistant() {
    const length = Math.max(1, this.window.end - this.window.start);
    const low = Math.max(0, this.window.start - length);
    const high = this.window.end + length;
    for (const [index, item] of this.cache) {
      if ((index < low || index >= high) && !this.isPinned(item, index)) this.cache.delete(index);
    }
  }

  showError(error) {
    this.error.querySelector("span").textContent = error.message || String(error);
    this.error.hidden = false;
  }

  hideError() { this.error.hidden = true; }
}

