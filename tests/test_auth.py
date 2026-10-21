"""Sign-in (server profile): accounts, sessions, the middleware, password changes, the `user` commands."""

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import fields, replace as dc_replace

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from typer.testing import CliRunner

from test_dataset import make_dataset
from yolo_workbench import auth as auth_module
from yolo_workbench.auth import Auth, AuthError, User, UserStore, acting_as, hash_password, require_admin, verify_password
from yolo_workbench.config import load_config
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.runtime import RUNTIME
from yolo_workbench.server import app as server_cli, build_app
from yolo_workbench.web import create_app
from yolo_workbench.workspace import Workspace

PASSWORD = "correct horse"


@pytest.fixture(autouse=True)
def restore_runtime():
    saved = dc_replace(RUNTIME)
    yield
    for item in fields(RUNTIME):
        setattr(RUNTIME, item.name, getattr(saved, item.name))


@pytest.fixture(autouse=True)
def cheap_scrypt(monkeypatch):
    monkeypatch.setattr(auth_module, "SCRYPT", {"n": 2**10, "r": 8, "p": 1})  # tests need no real work factor


def server_app(tmp_path, monkeypatch, **overrides):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    root = tmp_path / "projects"
    root.mkdir(exist_ok=True)
    config = load_config(None, {"projects_root": root, "allowed_hosts": ["gpu-box"], "idle_unload_minutes": 0, **overrides}, environ={})
    return build_app(config, root), root


def client(app, cookies=None):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://gpu-box", headers={"X-Workbench": "1"}, cookies=cookies)


@asynccontextmanager
async def login(app, name="alice", password=PASSWORD):
    async with client(app) as signed_in:
        response = await signed_in.post("/api/v1/auth/login", json={"name": name, "password": password})
        assert response.status_code == 200, response.text
        signed_in.login_response = response
        yield signed_in


# ---------- accounts ----------


def test_password_hashes_are_salted_and_verified():
    first, second = hash_password(PASSWORD), hash_password(PASSWORD)
    assert first != second and first.startswith("scrypt$")
    assert verify_password(PASSWORD, first) and not verify_password("wrong password", first)
    assert not verify_password(PASSWORD, "not-a-hash")


def test_user_store_rules_and_generations(tmp_path):
    store = UserStore(tmp_path)
    alice = store.add("Alice", PASSWORD, "Alice A.", admin=True)
    assert alice == User("alice", "Alice A.", True)
    with pytest.raises(AuthError, match="already exists"):
        store.add("alice", PASSWORD)
    with pytest.raises(AuthError, match="user name"):
        store.add("../evil", PASSWORD)
    with pytest.raises(AuthError, match="at least 8"):
        store.add("bob", "short")
    assert store.verify("ALICE", PASSWORD) == alice and store.verify("alice", "wrong password") is None and store.verify("nobody", PASSWORD) is None
    generation = store.get("alice")["generation"]
    assert store.set_password("alice", "another password") == generation + 1
    assert store.verify("alice", "another password") and not store.verify("alice", PASSWORD)
    stored = json.loads((tmp_path / "users.json").read_text())
    assert PASSWORD not in json.dumps(stored) and "another password" not in json.dumps(stored)
    if os.name != "nt":
        assert (tmp_path / "users.json").stat().st_mode & 0o077 == 0  # hashes are private to the service account
    # a second store (the command line) sees the changes, and the first re-reads the file
    other = UserStore(tmp_path)
    other.add("bob", PASSWORD)
    assert [user["name"] for user in store.list()] == ["alice", "bob"]
    other.remove("bob")
    assert store.get("bob") is None


# ---------- the middleware ----------


def test_requests_need_a_session_except_login_and_health(tmp_path, monkeypatch):
    app, root = server_app(tmp_path, monkeypatch)
    app.state.auth.users.add("alice", PASSWORD)

    async def run():
        async with client(app) as anonymous:
            assert (await anonymous.get("/api/v1/projects")).status_code == 401
            response = await anonymous.get("/projects.html", params={"x": "1"})
            assert response.status_code == 303 and response.headers["location"] == "/login.html?next=%2Fprojects.html%3Fx%3D1"
            assert (await anonymous.get("/")).headers["location"] == "/login.html?next=%2F"
            assert (await anonymous.get("/app.js")).status_code == 401
            for path in ("/healthz", "/readyz", "/login.html", "/login.js", "/styles.css"):
                assert (await anonymous.get(path)).status_code in (200, 503), path  # readyz: 503 before the lifespan
            assert (await anonymous.get("/api/v1/auth/me")).json() == {"auth": True, "user": None}
            wrong = await anonymous.post("/api/v1/auth/login", json={"name": "alice", "password": "wrong password"})
            assert wrong.status_code == 400 and wrong.json()["detail"] == "Wrong user name or password"
            assert (await anonymous.post("/api/v1/projects", json={"name": "X", "category": "detection"})).status_code == 401
        async with client(app) as no_header:
            no_header.headers.pop("X-Workbench")
            assert (await no_header.post("/api/v1/auth/login", json={"name": "alice", "password": PASSWORD})).status_code == 403

        async with login(app) as signed_in:
            cookie = signed_in.cookies.jar
            raw = signed_in.login_response.headers["set-cookie"]
            assert "HttpOnly" in raw and "SameSite=Lax" in raw and "Secure" not in raw
            info = (await signed_in.get("/api/v1/app")).json()
            assert info["auth"] and info["user"] == {"name": "alice", "display_name": "alice", "admin": False}
            assert (await signed_in.post("/api/v1/projects", json={"name": "Cars", "category": "detection"})).status_code == 200
            assert (await signed_in.get("/api/v1/projects/cars/dataset")).status_code == 200
            assert (await signed_in.get("/projects.html")).status_code == 200
            assert cookie
            token = _token(signed_in)
            logout = await signed_in.post("/api/v1/auth/logout")
            assert "Max-Age=0" in logout.headers["set-cookie"]
        async with client(app, {app.state.auth.cookie: token}) as stale:
            assert (await stale.get("/api/v1/projects")).status_code == 401
        assert not app.state.auth.sessions.count()
        assert json.loads((root / ".workbench" / "users.json").read_text())["users"]["alice"]["last_login"]

    asyncio.run(run())


def _token(signed_in: AsyncClient) -> str:
    return next(cookie.value for cookie in signed_in.cookies.jar if cookie.name.startswith("workbench_session_"))


def test_servers_on_one_host_keep_separate_cookies(tmp_path):
    assert Auth(tmp_path / "a").cookie != Auth(tmp_path / "b").cookie


def test_no_accounts_yet_explains_how_to_add_one(tmp_path, monkeypatch):
    app, _ = server_app(tmp_path, monkeypatch)

    async def run():
        async with client(app) as anonymous:
            response = await anonymous.post("/api/v1/auth/login", json={"name": "alice", "password": PASSWORD})
            assert response.status_code == 400 and "user add" in response.json()["detail"]

    asyncio.run(run())


def test_sessions_expire_slide_survive_restarts_and_end_with_the_password(tmp_path, monkeypatch):
    app, root = server_app(tmp_path, monkeypatch)
    auth = app.state.auth
    auth.users.add("alice", PASSWORD)

    async def run():
        async with login(app) as first, login(app) as second:
            token = _token(first)
            # sliding renewal: a session not renewed for a while gets a fresh expiry and cookie
            key = auth.sessions._key(token)
            auth.sessions._sessions[key]["renewed"] -= 2 * 3600
            before = auth.sessions._sessions[key]["expires"]
            response = await first.get("/api/v1/projects")
            assert response.status_code == 200 and f"{auth.cookie}={token}" in response.headers.get("set-cookie", "")
            assert auth.sessions._sessions[key]["expires"] > before
            assert "set-cookie" not in (await first.get("/api/v1/projects")).headers  # not on every request

            # the server restarts: sessions are kept (only their digests are on disk)
            assert token not in (root / ".workbench" / "sessions.json").read_text()
            restarted, _ = server_app(tmp_path, monkeypatch)
            assert restarted.state.auth.cookie == auth.cookie  # the same folder: the same cookie after a restart
            async with client(restarted, {auth.cookie: token}) as again:
                assert (await again.get("/api/v1/projects")).status_code == 200

            # a password change ends the other sessions; the browser that changed it stays signed in
            wrong = await first.post("/api/v1/auth/password", json={"current": "nope nope", "new": "brand new password"})
            assert wrong.status_code == 400 and "current password" in wrong.json()["detail"]
            changed = await first.post("/api/v1/auth/password", json={"current": PASSWORD, "new": "brand new password"})
            assert changed.status_code == 200
            assert (await first.get("/api/v1/projects")).status_code == 200
            response = await second.get("/api/v1/projects")
            assert response.status_code == 401 and "expired" in response.json()["detail"]

            # the command line changes the password: the server sees it and the session ends
            UserStore(root / ".workbench").set_password("alice", "set by the admin")
            assert (await first.get("/api/v1/projects")).status_code == 401

        # an expired session is refused and dropped
        async with login(app, password="set by the admin") as third:
            auth.sessions._sessions[auth.sessions._key(_token(third))]["expires"] = time.time() - 1
            assert (await third.get("/api/v1/projects")).status_code == 401
        assert auth.sessions.count("alice") == 0

        # removing the account ends its sessions
        async with login(app, password="set by the admin") as fourth:
            UserStore(root / ".workbench").remove("alice")
            assert (await fourth.get("/api/v1/projects")).status_code == 401

    asyncio.run(run())


def test_current_user_reaches_threadpool_endpoints_and_require_admin(tmp_path, monkeypatch):
    app, _ = server_app(tmp_path, monkeypatch)
    app.state.auth.users.add("alice", PASSWORD)
    app.state.auth.users.add("root", PASSWORD, admin=True)
    seen = []

    @app.get("/api/v1/test-admin")
    def admin_only(user=Depends(require_admin)):  # a sync endpoint: FastAPI runs it in a thread
        seen.append(auth_module.current_user_name())
        return {"user": user.name}

    app.router.routes.insert(0, app.router.routes.pop())  # before the static mount

    async def run():
        async with login(app) as alice:
            assert (await alice.get("/api/v1/test-admin")).status_code == 403
        async with login(app, "root") as root:
            assert (await root.get("/api/v1/test-admin")).json() == {"user": "root"}
        assert seen == ["root"]

    asyncio.run(run())
    # without sign-in (desktop) nobody is refused
    plain = FastAPI()
    plain.get("/x")(lambda user=Depends(require_admin): {"user": user})

    async def desktop():
        async with AsyncClient(transport=ASGITransport(app=plain), base_url="http://test") as anonymous:
            assert (await anonymous.get("/x")).json() == {"user": None}

    asyncio.run(desktop())
    with acting_as(User("bob", "Bob")):
        with pytest.raises(Exception, match="administrators"):
            require_admin()


def test_desktop_and_servers_without_sign_in_are_unaffected(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    dataset, _ = make_dataset(tmp_path / "data")
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media"))

    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as anonymous:
            info = (await anonymous.get("/api/v1/app")).json()
            assert info["auth"] is False and info["user"] is None
            assert (await anonymous.get("/api/v1/dataset")).status_code == 200
            assert (await anonymous.post("/api/v1/auth/login", json={"name": "a", "password": "b"})).status_code == 404
        server, _ = server_app(tmp_path, monkeypatch, auth_enabled=False)
        async with client(server) as anonymous:
            assert (await anonymous.get("/api/v1/projects")).status_code == 200

    asyncio.run(run())


# ---------- the command line ----------


def test_user_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    root = tmp_path / "projects"
    root.mkdir()
    runner = CliRunner()
    projects = ["--projects", str(root)]  # no config file: the listening checks do not apply to these commands
    result = runner.invoke(server_cli, ["user", "add", "Alice", "--admin", "--display-name", "Alice A.", "--password-stdin", *projects], input=f"{PASSWORD}\n")
    assert result.exit_code == 0, result.output
    assert "Added alice (administrator)" in result.output
    assert runner.invoke(server_cli, ["user", "add", "alice", "--password-stdin", *projects], input=f"{PASSWORD}\n").exit_code == 1
    bad = runner.invoke(server_cli, ["user", "add", "no way", "--password-stdin", *projects], input=f"{PASSWORD}\n")
    assert bad.exit_code == 1 and "user name" in bad.output
    short = runner.invoke(server_cli, ["user", "add", "bob", "--password-stdin", *projects], input="short\n")
    assert short.exit_code == 1 and "at least 8" in short.output
    prompted = runner.invoke(server_cli, ["user", "add", "bob", *projects], input=f"{PASSWORD}\n{PASSWORD}\n")
    assert prompted.exit_code == 0, prompted.output
    listing = runner.invoke(server_cli, ["user", "list", *projects])
    assert "alice" in listing.output and "Alice A." in listing.output and "admin" in listing.output and "bob" in listing.output
    assert runner.invoke(server_cli, ["user", "passwd", "bob", "--password-stdin", *projects], input="new password!\n").exit_code == 0
    store = UserStore(root / ".workbench")
    assert store.verify("bob", "new password!")
    assert runner.invoke(server_cli, ["user", "admin", "bob", *projects]).exit_code == 0 and store.user("bob").admin
    assert runner.invoke(server_cli, ["user", "admin", "bob", "--revoke", *projects]).exit_code == 0 and not store.user("bob").admin
    assert runner.invoke(server_cli, ["user", "remove", "bob", *projects]).exit_code == 0 and store.get("bob") is None
    assert runner.invoke(server_cli, ["user", "remove", "bob", *projects]).exit_code == 1
    check = runner.invoke(server_cli, ["check", *projects, "--config", str(_config(tmp_path, root))])
    assert "sign-in: 1 account(s), 1 administrator(s)" in check.output, check.output


def _config(tmp_path, root):
    path = tmp_path / "config.toml"
    path.write_text(f'[server]\nprojects_root = "{root.as_posix()}"\nhost = "127.0.0.1"\n[storage]\ncache_dir = "{(tmp_path / "cache").as_posix()}"\n[embeddings]\nenabled = false\n')
    return path



# ---------- found in review ----------


def test_a_removed_and_re_added_name_does_not_revive_old_sessions(tmp_path):
    auth = Auth(tmp_path)
    auth.users.add("alice", PASSWORD)
    _, token = auth.login("alice", PASSWORD)
    auth.users.remove("ALICE")  # any spelling of the name
    auth.users.add("alice", PASSWORD)
    assert auth.authenticate(token) is None


def test_a_password_change_on_a_renewing_request_keeps_the_new_session(tmp_path, monkeypatch):
    app, _ = server_app(tmp_path, monkeypatch)
    auth = app.state.auth
    auth.users.add("alice", PASSWORD)

    async def run():
        async with login(app) as alice:
            auth.sessions._sessions[auth.sessions._key(_token(alice))]["renewed"] -= 2 * 3600  # due for renewal
            changed = await alice.post("/api/v1/auth/password", json={"current": PASSWORD, "new": "brand new password"})
            assert changed.status_code == 200 and len(changed.headers.get_list("set-cookie")) == 1
            assert (await alice.get("/api/v1/projects")).status_code == 200

    asyncio.run(run())
