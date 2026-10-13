export const $ = id => document.getElementById(id);

// Dataset endpoints live at /api/v1 (single-dataset mode) or /api/v1/projects/<slug> (projects mode).
// Paths starting with /api/ are absolute (workspace, jobs); anything else is relative to the dataset base.
let apiBase = "/api/v1";
export const setApiBase = base => { apiBase = base; };
export const apiUrl = path => path.startsWith("/api/") ? path : `${apiBase}${path}`;
export const api = async (path, options = {}) => {
  // X-Workbench marks the request as coming from this UI; the server refuses mutating requests without it
  const headers = { "Content-Type": "application/json", "X-Workbench": "1", ...(options.headers || {}) };
  const response = await fetch(apiUrl(path), { ...options, headers });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail || response.statusText);
  }
  return response.json();
};
export const esc = value => String(value).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
export const debounce = (fn, wait) => { let timer; return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); }; };
export const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

// Keep in sync with PALETTE in web.py so grid overlays and crops match the editor.
export const PALETTE = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#06b6d4", "#84cc16", "#a855f7", "#64748b"];
export const classColor = id => PALETTE[((id % PALETTE.length) + PALETTE.length) % PALETTE.length];
export const HANDLE_CURSORS = ["nwse-resize", "ns-resize", "nesw-resize", "ew-resize", "nwse-resize", "ns-resize", "nesw-resize", "ew-resize"];


export function pointInPolygon(x, y, points) {
  let inside = false;
  for (let i = 0, j = points.length - 2; i < points.length; j = i, i += 2)
    if ((points[i + 1] > y) !== (points[j + 1] > y) && x < (points[j] - points[i]) * (y - points[i + 1]) / (points[j + 1] - points[i + 1]) + points[i])
      inside = !inside;
  return inside;
}
