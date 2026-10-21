import { api, esc } from "./api.js";

// Server-wide warnings (low disk space) on every page, under the header. Polled: nothing else reports them.
const POLL = 60000;

export function watchServerStatus() {
  let banner = document.getElementById("server-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "server-banner";
    banner.className = "server-banner";
    banner.setAttribute("role", "status");
    banner.hidden = true;
    document.querySelector("header")?.after(banner);
  }
  const poll = async () => {
    try {
      const status = await api("/api/v1/status");
      banner.hidden = !status.warnings.length;
      banner.innerHTML = status.warnings.map(text => `<div>${esc(text)}</div>`).join("");
    } catch { /* the server may be restarting; the next poll tells */ }
    setTimeout(poll, POLL);
  };
  poll();
}
