import { watchServerStatus } from "./status.js";
import { $, api, esc } from "./api.js";

let projects = [];
let editing = null;   // slug being edited in the project dialog, null when creating
let adopting = null;

function toast(message, kind = "info") {
  const el = $("toast");
  el.textContent = message;
  el.className = `${kind} show`;
  clearTimeout(el.timer);
  el.timer = setTimeout(() => el.classList.remove("show"), 3200);
}

const plural = (count, noun) => `${(count || 0).toLocaleString()} ${noun}${count === 1 ? "" : noun.endsWith("s") ? "es" : "s"}`;
const openUrl = slug => `/?project=${encodeURIComponent(slug)}`;

async function init() {
  const info = await api("/api/v1/app");
  if (info.mode !== "projects") { location.replace("/"); return; }
  watchServerStatus();
  const message = sessionStorage.getItem("projects-message");
  if (message) {
    sessionStorage.removeItem("projects-message");
    $("projects-message").textContent = message;
    $("projects-message").hidden = false;
  }
  bind();
  await load();
}

async function load() {
  const [data, trash] = await Promise.all([api("/api/v1/projects"), api("/api/v1/trash")]);
  projects = data.projects;
  $("projects-root").textContent = data.root;
  $("projects-root").title = data.root;
  renderProjects();
  renderUnmanaged(data.unmanaged);
  renderTrash(trash.items, trash.retention_days);
  pollTraining();
}

// training badges on the cards: the active run and queued runs (one queue for all projects)
let trainingTimer = null;
async function pollTraining() {
  clearTimeout(trainingTimer);
  try {
    const queue = await api("/api/v1/training/queue");
    const badges = new Map();
    if (queue.active) badges.set(queue.active.project, { text: `Training ${queue.active.name} · ${queue.active.epoch ?? 0}/${queue.active.epochs ?? "?"}`, run: queue.active, queued: false });
    queue.queued.forEach((item, index) => {
      if (!badges.has(item.project)) badges.set(item.project, { text: `Queued #${index + 1}`, run: item, queued: true });
    });
    document.querySelectorAll("[data-training-flag]").forEach(flag => {
      const badge = badges.get(flag.dataset.trainingFlag);
      flag.hidden = !badge;
      if (!badge) return;
      flag.classList.toggle("queued", badge.queued);
      flag.innerHTML = `${badge.queued ? "" : '<span class="live-dot" aria-hidden="true"></span>'}${esc(badge.text)}`;
      flag.title = `${badge.run.name}: ${badge.queued ? "waiting in the training queue" : "training"}`;
    });
    trainingTimer = setTimeout(pollTraining, queue.active ? 5000 : 15000);
  } catch {
    trainingTimer = setTimeout(pollTraining, 15000);
  }
}

function renderProjects() {
  $("project-grid").innerHTML = projects.map(project => {
    const summary = project.summary || {};
    const splits = Object.entries(summary.splits || {}).map(([split, count]) => `<span class="split-badge" title="${esc(split)}">${esc(split)} ${count.toLocaleString()}</span>`).join("");
    return `<article class="card project-card" data-slug="${esc(project.slug)}">
      <a class="thumb project-cover" href="${openUrl(project.slug)}" aria-label="Open ${esc(project.name)}">
        ${project.has_cover ? `<img src="/api/v1/projects/${encodeURIComponent(project.slug)}/cover?v=${encodeURIComponent(project.updated || "")}" alt="" loading="lazy">`
          : `<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/></svg>`}
        ${project.active ? `<span class="pred-flag" title="Currently open">open</span>` : ""}
        <span class="training-flag" data-training-flag="${esc(project.slug)}" hidden></span>
      </a>
      <div class="card-body">
        <div class="project-card-head">
          <a class="name" href="${openUrl(project.slug)}" title="${esc(project.name)}">${esc(project.name)}</a>
          <span class="pill">${esc(project.category)}</span>
        </div>
        ${project.description ? `<p class="project-description" title="${esc(project.description)}">${esc(project.description)}</p>` : ""}
        <div class="meta"><span>${plural(summary.images, "image")}</span><span>${plural(summary.classes, "class")}</span>${summary.images ? `<span>${Math.round((summary.annotated_images || 0) * 100 / summary.images)}% labeled</span>` : ""}</div>
        <div class="chips">${splits}</div>
        <div class="project-actions">
          <a class="button-link primary-btn" href="${openUrl(project.slug)}">Open</a>
          <button class="secondary" data-edit="${esc(project.slug)}">Edit</button>
          <button class="secondary danger-btn" data-trash="${esc(project.slug)}" title="Move to trash">Delete</button>
        </div>
      </div>
    </article>`;
  }).join("") || `<div class="empty">No projects yet. Create one to start a new dataset.</div>`;
}

function renderUnmanaged(items) {
  $("unmanaged-section").hidden = !items.length;
  $("unmanaged-list").innerHTML = items.map(item => `<div class="plain-row">
      <strong>${esc(item.slug)}/</strong>
      <span class="muted">${esc(item.yaml ? `${item.yaml} · ` : "")}${esc(item.reason)}</span>
      ${item.adoptable ? `<button class="secondary" data-adopt="${esc(item.slug)}" data-category="${esc(item.category || "")}">Adopt</button>` : ""}
    </div>`).join("");
}

function renderTrash(items, retentionDays = 0) {
  $("trash-section").hidden = !items.length;
  $("trash-count").textContent = items.length || "";
  $("trash-retention").textContent = retentionDays ? ` They are deleted permanently by themselves ${plural(retentionDays, "day")} after deletion.` : "";
  $("trash-list").innerHTML = items.map(item => `<div class="plain-row">
      <strong>${esc(item.name)}</strong>
      <span class="muted">${esc(item.category || "")} · ${plural(item.summary?.images, "image")} · deleted ${esc(formatStamp(item.deleted))}</span>
      <span class="row-actions"><button class="secondary" data-restore="${esc(item.entry)}">Restore</button>
      <button class="danger-btn" data-purge="${esc(item.entry)}" data-name="${esc(item.name)}">Delete permanently</button></span>
    </div>`).join("");
}

const formatStamp = stamp => {
  const match = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/.exec(stamp || "");
  return match ? `${match[1]}-${match[2]}-${match[3]} ${match[4]}:${match[5]}` : stamp;
};

function openProjectDialog(project = null) {
  editing = project?.slug ?? null;
  const form = $("project-form");
  form.reset();
  $("project-dialog-title").textContent = project ? `Edit ${project.name}` : "New project";
  $("project-dialog-confirm").textContent = project ? "Save" : "Create";
  $("category-field").hidden = !!project;
  $("classes-field").hidden = !!project;
  $("project-dialog-error").hidden = true;
  if (project) {
    form.elements.name.value = project.name;
    form.elements.description.value = project.description || "";
  }
  $("project-dialog").showModal();
  form.elements.name.focus();
}

async function submitProject(e) {
  if (e.submitter?.value !== "confirm") return;
  e.preventDefault();  // keep the dialog open until the server accepts
  const form = $("project-form");
  const body = { name: form.elements.name.value.trim(), description: form.elements.description.value };
  try {
    if (editing) {
      await api(`/api/v1/projects/${encodeURIComponent(editing)}`, { method: "PATCH", body: JSON.stringify(body) });
      $("project-dialog").close();
      toast("Project updated");
      await load();
    } else {
      const project = await api("/api/v1/projects", { method: "POST", body: JSON.stringify({
        ...body,
        category: form.elements.category.value,
        classes: form.elements.classes.value.split("\n").map(name => name.trim()).filter(Boolean),
      }) });
      location.href = openUrl(project.slug);
    }
  } catch (error) {
    $("project-dialog-error").textContent = error.message;
    $("project-dialog-error").hidden = false;
  }
}

async function action(promise, message) {
  try {
    await promise;
    if (message) toast(message);
    await load();
  } catch (error) { toast(error.message, "error"); }
}

function bind() {
  $("new-project").onclick = () => openProjectDialog();
  $("project-form").addEventListener("submit", submitProject);
  $("project-grid").addEventListener("click", e => {
    const edit = e.target.closest("[data-edit]");
    if (edit) return openProjectDialog(projects.find(project => project.slug === edit.dataset.edit));
    const trash = e.target.closest("[data-trash]");
    if (trash) {
      const project = projects.find(item => item.slug === trash.dataset.trash);
      if (confirm(`Move “${project.name}” to the trash? You can restore it from the Trash section below.`))
        action(api(`/api/v1/projects/${encodeURIComponent(project.slug)}`, { method: "DELETE" }), `Moved “${project.name}” to the trash`);
    }
  });
  $("trash-list").addEventListener("click", e => {
    const restore = e.target.closest("[data-restore]");
    if (restore) return action(api(`/api/v1/trash/${encodeURIComponent(restore.dataset.restore)}/restore`, { method: "POST" }), "Project restored");
    const purge = e.target.closest("[data-purge]");
    if (purge) {
      const typed = prompt(`This permanently deletes “${purge.dataset.name}” and all its images and labels. Type the project name to confirm:`);
      if (typed === null) return;
      if (typed.trim() !== purge.dataset.name) return toast("The name did not match; nothing was deleted", "error");
      action(api(`/api/v1/trash/${encodeURIComponent(purge.dataset.purge)}`, { method: "DELETE" }), "Deleted permanently");
    }
  });
  $("unmanaged-list").addEventListener("click", e => {
    const adopt = e.target.closest("[data-adopt]");
    if (!adopt) return;
    adopting = adopt.dataset.adopt;
    const form = $("adopt-form");
    form.reset();
    $("adopt-slug").textContent = `${adopting}/`;
    form.elements.name.value = adopting;
    if (adopt.dataset.category) form.elements.category.value = adopt.dataset.category;
    $("adopt-dialog").showModal();
  });
  $("adopt-dialog").addEventListener("close", () => {
    if ($("adopt-dialog").returnValue !== "confirm" || !adopting) return;
    const form = $("adopt-form");
    const slug = adopting;
    adopting = null;
    action(api(`/api/v1/projects/${encodeURIComponent(slug)}/adopt`, { method: "POST", body: JSON.stringify({ category: form.elements.category.value, name: form.elements.name.value }) }), "Dataset adopted");
  });
  $("theme-toggle").onclick = () => {
    const root = document.documentElement;
    const dark = root.dataset.theme !== "dark";
    if (dark) root.dataset.theme = "dark"; else delete root.dataset.theme;
    localStorage.setItem("theme", dark ? "dark" : "light");
  };
}

init().catch(error => toast(error.message, "error"));
