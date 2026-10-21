import { api, esc } from "./api.js";

// The signed-in user in the header (server with sign-in): who you are, change password, sign out.
// `info` is /api/v1/app; without sign-in (desktop, or [auth] enabled = false) nothing is shown.

const initials = user => (user.display_name || user.name).split(/[\s._-]+/).filter(Boolean).slice(0, 2).map(part => part[0].toUpperCase()).join("") || "?";

export function showUser(info) {
  const user = info?.auth ? info.user : null;
  const header = document.querySelector("header");
  if (!user || !header || document.getElementById("user-menu")) return;
  const toggle = document.getElementById("theme-toggle");
  const menu = document.createElement("div");
  menu.id = "user-menu";
  menu.className = "user-menu";
  menu.innerHTML = `
    <button id="user-button" class="user-button" type="button" aria-haspopup="menu" aria-expanded="false" title="${esc(user.display_name)} (${esc(user.name)})">
      <span class="user-avatar" aria-hidden="true">${esc(initials(user))}</span>
    </button>
    <div id="user-panel" class="user-panel" role="menu" hidden>
      <div class="user-panel-who"><strong>${esc(user.display_name)}</strong><span class="muted">${esc(user.name)}${user.admin ? " · administrator" : ""}</span></div>
      <button type="button" role="menuitem" data-action="password">Change password…</button>
      <button type="button" role="menuitem" data-action="logout">Sign out</button>
    </div>`;
  // the projects page pushes the last header item to the right
  if (toggle?.classList.contains("header-end")) {
    toggle.classList.remove("header-end");
    menu.classList.add("header-end");
  }
  header.insertBefore(menu, toggle || null);

  const button = menu.querySelector("#user-button");
  const panel = menu.querySelector("#user-panel");
  const close = () => { panel.hidden = true; button.setAttribute("aria-expanded", "false"); };
  button.onclick = () => {
    panel.hidden = !panel.hidden;
    button.setAttribute("aria-expanded", String(!panel.hidden));
  };
  document.addEventListener("pointerdown", event => { if (!menu.contains(event.target)) close(); });
  document.addEventListener("keydown", event => { if (event.key === "Escape" && !panel.hidden) { close(); button.focus(); } });
  panel.onclick = async event => {
    const action = event.target.closest("[data-action]")?.dataset.action;
    if (!action) return;
    close();
    if (action === "logout") {
      await api("/api/v1/auth/logout", { method: "POST" }).catch(() => {});
      location.assign("/login.html");
    } else if (action === "password") {
      changePassword();
    }
  };
}

function changePassword() {
  let dialog = document.getElementById("password-dialog");
  if (!dialog) {
    dialog = document.createElement("dialog");
    dialog.id = "password-dialog";
    dialog.className = "modal";
    dialog.innerHTML = `
      <form method="dialog">
        <h2>Change password</h2>
        <label class="field">Current password<input name="current" type="password" autocomplete="current-password" required></label>
        <label class="field">New password<input name="new" type="password" autocomplete="new-password" minlength="8" required></label>
        <label class="field">New password again<input name="repeat" type="password" autocomplete="new-password" required></label>
        <p class="modal-text muted">At least 8 characters. Your other sign-ins (other browsers and computers) end.</p>
        <p class="warn-text" data-error hidden></p>
        <div class="modal-actions"><button type="button" class="secondary" data-cancel>Cancel</button><button value="confirm" class="primary-btn">Change password</button></div>
      </form>`;
    document.body.append(dialog);
    const form = dialog.querySelector("form");
    const error = dialog.querySelector("[data-error]");
    dialog.querySelector("[data-cancel]").onclick = () => dialog.close();  // a plain button: Enter confirms
    form.addEventListener("submit", async event => {
      event.preventDefault();
      const { current, new: next, repeat } = form.elements;
      const fail = message => { error.textContent = message; error.hidden = false; };
      if (next.value !== repeat.value) return fail("The new passwords differ.");
      if (next.value.length < 8) return fail("The new password needs at least 8 characters.");
      try {
        await api("/api/v1/auth/password", { method: "POST", body: JSON.stringify({ current: current.value, new: next.value }) });
        dialog.close();
        notify("Password changed. Your other sign-ins have ended.");
      } catch (failure) {
        fail(failure.message);
      }
    });
  }
  const form = dialog.querySelector("form");
  form.reset();
  dialog.querySelector("[data-error]").hidden = true;
  dialog.showModal();
}

function notify(message) {
  const toast = document.getElementById("toast");
  if (!toast) return;
  toast.textContent = message;
  toast.className = "info show";
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => toast.classList.remove("show"), 3200);
}
