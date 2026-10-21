import { $, api, esc } from "./api.js";
import { state } from "./state.js";

// Several people in one project. Other people's changes reach this page by polling /changes (every few
// seconds while the tab is visible; changes made from this tab are left out by the server). The editor
// reports which image it shows (presence, signed-in users only), so the gallery and the editor can say who
// else has an image open; the editor also says who last edited and reviewed its image.

const POLL = 3000;
const HIDDEN_POLL = 30000;
const HEARTBEAT = 15000;  // the server forgets a presence after 30 s without one
let deps;
let timer = null;
let polling = false;
let heartbeat = null;
let shown = null;  // the image this tab's editor shows

export function configureCollab(dependencies) {
  deps = dependencies;
}

export const personName = who => who ? (who.display_name || who.name) : "Someone";
export const initials = who => personName(who).split(/[\s._-]+/).filter(Boolean).slice(0, 2).map(part => part[0].toUpperCase()).join("") || "?";
const isMe = who => !!who && who.name === state.user?.name;

// "13:02" today, "21 Oct 13:02" otherwise; `at` is an ISO time or seconds since the epoch
export function when(at) {
  const date = typeof at === "number" ? new Date(at * 1000) : new Date(at);
  if (Number.isNaN(date.getTime())) return "";
  const clock = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return date.toDateString() === new Date().toDateString() ? clock : `${date.toLocaleDateString([], { day: "numeric", month: "short" })} ${clock}`;
}

// "· by Alice" for something another person started (nothing for your own)
export const startedBy = who => who && !isMe(who) ? ` · started by ${personName(who)}` : "";

/* ---------- the change feed ---------- */

export function startChangeFeed() {
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") poll();
  });
  schedule();
}

function schedule(delay = document.visibilityState === "visible" ? POLL : HIDDEN_POLL) {
  clearTimeout(timer);
  timer = setTimeout(poll, delay);
}

async function poll() {
  if (polling || !state.meta) return schedule();
  polling = true;
  try {
    const feed = await api(`/changes?since=${state.revision}${state.feed ? `&feed=${state.feed}` : ""}`);
    applyPresence(feed.presence || {});
    // closed: the project closed while idle; it opens again (as a new feed) with the next real request
    if (!feed.closed && (feed.revision !== state.revision || feed.feed !== state.feed)) await applyChanges(feed);
  } catch { /* the server may be restarting or the project closing; the next poll tells */ }
  finally {
    polling = false;
    schedule();
  }
}

async function applyChanges(feed) {
  state.revision = feed.revision;
  state.feed = feed.feed;
  const changed = Object.entries(feed.images);
  if (!feed.reset && !feed.structure && !feed.schema && !changed.length) return;  // only this tab's own changes
  const people = [...new Set(changed.map(([, change]) => personName(change.by)))];
  const open = state.detail?.id;
  if (feed.reset || feed.schema) {
    // reset: the project was opened again on the server, or this page fell too far behind; reload everything
    if (!feed.reset) deps.toast(`${people.join(", ") || "Someone"} changed the class list; the classes were reloaded`);
    await deps.datasetChanged({ schema: true, keepEditor: true, keepSelection: true });
  } else if (feed.structure) {
    await deps.datasetChanged({ keepSelection: true });
  } else {
    await deps.imagesChanged(changed.map(([id]) => id));
  }
  if (open && state.detail?.id === open && (feed.images[open] || feed.reset || feed.schema)) await openImageChanged(open, feed.images[open], feed);
}

// The editor's image changed elsewhere: reload what cannot conflict, offer a reload for label edits.
async function openImageChanged(id, change, feed) {
  if (feed.reset || feed.schema || change?.kind === "review") {
    const result = await deps.reloadEditor(id);
    if (result === "missing") showEditorNotice(`${esc(personName(change?.by))} removed this image from the dataset.`);
    return;
  }
  if (feed.structure) {
    const result = await deps.reloadEditor(id, { quiet: true });
    if (result === "missing") return showEditorNotice(`${esc(personName(change?.by))} removed this image from the dataset (deleted, or moved to another split).`);
  }
  showEditorNotice(`<strong>${esc(personName(change?.by))}</strong> changed this image's labels at ${esc(when(change?.at))}. Reload before editing further: saving over their change is refused.`, true);
}

function showEditorNotice(html, reload = false) {
  const notice = $("editor-notice");
  notice.innerHTML = `<span>${html}</span>${reload ? '<button type="button" class="secondary mini" data-reload>Reload</button>' : ""}`;
  notice.hidden = false;
  const button = notice.querySelector("[data-reload]");
  if (button) button.onclick = async () => {
    notice.hidden = true;
    if (state.detail) await deps.reloadEditor(state.detail.id);
  };
}

export function clearEditorNotice() {
  $("editor-notice").hidden = true;
}

/* ---------- presence: who has which image open ---------- */

function applyPresence(presence) {
  const before = JSON.stringify(state.presence);
  state.presence = presence;
  if (JSON.stringify(presence) === before) return;
  deps.presenceChanged();
  renderEditorPeople();
}

// The editor shows `imageId` (null: it closed); a heartbeat keeps telling the server while it stays open.
export function editorShows(imageId) {
  if (!state.user || imageId === shown) return renderEditorPeople();
  shown = imageId;
  clearInterval(heartbeat);
  heartbeat = imageId ? setInterval(() => sendPresence(shown), HEARTBEAT) : null;
  sendPresence(imageId);
  renderEditorPeople();
}

async function sendPresence(imageId) {
  try {
    applyPresence((await api("/presence", { method: "POST", body: JSON.stringify({ image_id: imageId }) })).presence || {});
  } catch { /* presence is a courtesy; the next heartbeat retries */ }
}

export function presenceBadge(imageId) {
  const people = state.presence[imageId];
  if (!people?.length) return "";
  const names = people.map(personName).join(", ");
  return `<span class="presence-flag" title="Open in the editor of ${esc(names)}">${people.slice(0, 2).map(who => `<span>${esc(initials(who))}</span>`).join("")}${people.length > 2 ? `<span>+${people.length - 2}</span>` : ""}</span>`;
}

function renderEditorPeople() {
  const el = $("editor-people");
  const people = state.detail ? state.presence[state.detail.id] || [] : [];
  el.hidden = !people.length;
  el.textContent = people.length ? `${people.map(personName).join(", ")} ${people.length === 1 ? "has" : "have"} this image open` : "";
}

/* ---------- who edited and reviewed the editor's image ---------- */

export function renderAttribution() {
  const el = $("editor-attribution");
  const detail = state.detail;
  const parts = [];
  if (detail?.edited) parts.push(`Last edited by ${esc(personName(detail.edited.by))} · ${esc(when(detail.edited.at))}`);
  if (detail?.reviewed) parts.push(`${detail.review === "done" ? "Marked done" : "Flagged"} by ${esc(personName(detail.reviewed.by))} · ${esc(when(detail.reviewed.at))}`);
  el.hidden = !parts.length;
  el.innerHTML = parts.map(part => `<div>${part}</div>`).join("");
}

/* ---------- the activity list (Dataset tab) ---------- */

export async function loadActivity() {
  const panel = $("activity-panel");
  panel.hidden = !state.user;
  if (!state.user) return;
  try {
    const { items } = await api("/activity?limit=60");
    $("activity-list").innerHTML = items.length ? items.map(item => `
      <div class="activity-row">
        <span class="activity-time">${esc(when(item.time))}</span>
        <span class="activity-who" title="${esc(item.user)}">${esc(item.display_name || item.user)}</span>
        <span class="activity-what">${esc(item.action)}${item.count ? ` · ${item.count.toLocaleString()} image${item.count === 1 ? "" : "s"}` : ""}</span>
      </div>`).join("") : '<p class="empty">Nothing yet.</p>';
  } catch (error) {
    $("activity-list").innerHTML = `<p class="warn-text">${esc(error.message)}</p>`;
  }
}
