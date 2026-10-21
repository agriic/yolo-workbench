// The sign-in page. Only /login.html, this script and styles.css load without a session (auth.py OPEN_PATHS),
// so it imports nothing.
const form = document.getElementById("login-form");
const error = document.getElementById("login-error");

// where to go afterwards: a page of this server only. Resolved the way the browser will resolve it, so tricks
// like "/\t/other-host" (the tab is dropped) cannot lead elsewhere.
function nextUrl() {
  const next = new URLSearchParams(location.search).get("next") || "/";
  try {
    const url = new URL(next, location.origin);
    if (url.origin === location.origin && next.startsWith("/")) return url.pathname + url.search + url.hash;
  } catch { /* not a URL */ }
  return "/";
}

function showError(message) {
  error.textContent = message;
  error.hidden = !message;
}

async function init() {
  try {
    const me = await (await fetch("/api/v1/auth/me", { headers: { "X-Workbench": "1" } })).json();
    if (!me.auth || me.user) { location.replace(nextUrl()); return; }  // no sign-in here, or already signed in
  } catch { /* the server may be starting; the form still works once it answers */ }
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  const name = form.elements.name.value.trim();
  const password = form.elements.password.value;
  if (!name || !password) { showError("Enter your user name and password."); return; }
  showError("");
  const button = document.getElementById("login-submit");
  button.disabled = true;
  try {
    const response = await fetch("/api/v1/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Workbench": "1" },
      body: JSON.stringify({ name, password }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      showError(body.detail || response.statusText);
      form.elements.password.select();
      return;
    }
    location.replace(nextUrl());
  } catch {
    showError("The server did not answer. Try again in a moment.");
  } finally {
    button.disabled = false;
  }
});

init();
