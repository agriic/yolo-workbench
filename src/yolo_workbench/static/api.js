export const $ = id => document.getElementById(id);

// Dataset endpoints live at /api/v1 (single-dataset mode) or /api/v1/projects/<slug> (projects mode).
// Paths starting with /api/ are absolute (workspace, jobs); anything else is relative to the dataset base.
let apiBase = "/api/v1";
export const setApiBase = base => { apiBase = base; };
export const apiUrl = path => path.startsWith("/api/") ? path : `${apiBase}${path}`;
// This browser tab, so the server can tell its changes from other people's (and other tabs').
export const CLIENT_ID = (crypto.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`).replace(/[^A-Za-z0-9-]/g, "");

// 401: the sign-in ended (server profile). Go to the login page, which comes back here afterwards.
export const signIn = () => {
  if (location.pathname !== "/login.html") location.assign(`/login.html?next=${encodeURIComponent(location.pathname + location.search)}`);
};
export const api = async (path, options = {}) => {
  // X-Workbench marks the request as coming from this UI; the server refuses mutating requests without it
  const headers = { "Content-Type": "application/json", "X-Workbench": "1", "X-Workbench-Client": CLIENT_ID, ...(options.headers || {}) };
  const response = await fetch(apiUrl(path), { ...options, headers });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    if (response.status === 401) signIn();
    // status and body travel with the error: a 409 may say more (e.g. a changed class list)
    throw Object.assign(new Error(body.detail || response.statusText), { status: response.status, body });
  }
  return response.json();
};
export const esc = value => String(value).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
export const debounce = (fn, wait) => { let timer; return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); }; };
export const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
export const formatBytes = bytes => bytes >= 1 << 30 ? `${(bytes / (1 << 30)).toFixed(1)} GB`
  : bytes >= 1 << 20 ? `${(bytes / (1 << 20)).toFixed(1)} MB` : bytes >= 1024 ? `${Math.round(bytes / 1024)} KB` : `${bytes} B`;

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
