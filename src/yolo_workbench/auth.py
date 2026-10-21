"""Users and sign-in for the server profile.

Accounts live in `<projects>/.workbench/users.json` (managed by `yolo-workbench-server user ...`), sessions
in `sessions.json` beside it. Passwords are hashed with scrypt and a per-user salt. A session is a random
token in an HttpOnly cookie; only its SHA-256 is stored, so the file holds no usable tokens. Sessions slide:
each use extends them to `session_days` from now (persisted at most every RENEW_INTERVAL). Changing a
password or removing a user ends that user's sessions (the account's `generation` changes).

The command line and the server both edit users.json, so every read-modify-write holds a file lock
(`users.lock`), and the server re-reads the file whenever it changed on disk.

`AuthMiddleware` lets a request through only with a valid session, except the login page and its assets,
`/healthz` and `/readyz`: /api/ requests get 401, pages redirect to the login page. The signed-in user is in
`current_user()` for the request, including the threads FastAPI runs endpoints in.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from .fsutil import WINDOWS, replace
from .identity import _USER, User, acting_as, current_user, current_user_name  # noqa: F401 (re-exported)

USERS_FILE = "users.json"
SESSIONS_FILE = "sessions.json"
LOCK_FILE = "users.lock"
COOKIE = "workbench_session"  # + a suffix per projects folder (Auth.cookie): cookies do not tell ports apart
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
MIN_PASSWORD = 8
RENEW_INTERVAL = 3600.0  # seconds between persisted renewals of one session
SCRYPT = {"n": 2**14, "r": 8, "p": 1}
# without a session: the login page and what it loads, and the probes a monitoring system polls
OPEN_PATHS = frozenset({"/login.html", "/login.js", "/styles.css", "/healthz", "/readyz", "/api/v1/auth/login", "/api/v1/auth/logout", "/api/v1/auth/me"})


class AuthError(ValueError):
    """A sign-in or account change that was refused; the message is for the person."""


def require_admin() -> User | None:
    """FastAPI dependency for routes only administrators may use (with sign-in turned off, everyone may)."""
    user = _USER.get()
    if user is not None and not user.admin:
        raise HTTPException(403, "Only administrators can do this")
    return user


# ---------- passwords ----------


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, dklen=32, maxmem=64 * 2**20, **SCRYPT)
    return "scrypt${n}${r}${p}${salt}${digest}".format(**SCRYPT, salt=_b64(salt), digest=_b64(digest))


def verify_password(password: str, stored: str) -> bool:
    try:
        kind, n, r, p, salt, digest = stored.split("$")
        if kind != "scrypt":
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=len(expected), maxmem=64 * 2**20)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


_dummy_hash: str | None = None


def _dummy() -> str:
    """A hash to verify for unknown names, so they take as long as known ones."""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password(secrets.token_hex(8))
    return _dummy_hash


def check_password_rules(password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise AuthError(f"The password must have at least {MIN_PASSWORD} characters")
    if password != password.strip():
        raise AuthError("The password cannot start or end with a space")


def normalize_name(name: str) -> str:
    clean = str(name or "").strip().lower()
    if not NAME_PATTERN.match(clean):
        raise AuthError("A user name has 1-64 lowercase letters, digits, '.', '_' or '-', and starts with a letter or digit")
    return clean


# ---------- files ----------


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _write_json(path: Path, data: dict, private: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if private and not WINDOWS:
            os.chmod(temporary, 0o600)  # password hashes and session digests
        _match_owner(Path(temporary), path.parent)
        replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _match_owner(path: Path, directory: Path) -> None:
    """Run as root (`sudo yolo-workbench-server user add`), files take the owner of the folder they are in, so
    the service account can still write them."""
    if WINDOWS or os.geteuid() != 0:
        return
    try:
        info = directory.stat()
        if info.st_uid != 0:
            os.chown(path, info.st_uid, info.st_gid)
    except OSError:
        pass


@contextmanager
def _file_lock(path: Path):
    """An exclusive lock between processes (the server and the command line) for a read-modify-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    stream = open(path, "a+b")
    if not existed:
        _match_owner(path, path.parent)
    try:
        if WINDOWS:
            import msvcrt

            while True:
                try:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)  # retries for about 10 s, then raises
                    break
                except OSError:
                    continue
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if WINDOWS:
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        stream.close()


def _stamp(path: Path):
    try:
        info = path.stat()
        return (info.st_mtime_ns, info.st_size, info.st_ino)
    except OSError:
        return None


# ---------- accounts ----------


class UserStore:
    """users.json: {"version": 1, "users": {name: {display_name, password, admin, generation, created, last_login}}}."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.path = self.directory / USERS_FILE
        self.lock_path = self.directory / LOCK_FILE
        self._lock = threading.RLock()
        self._users: dict[str, dict] = {}
        self._stamp = None

    def _load(self) -> dict[str, dict]:
        """The accounts as on disk (re-read when the file changed). Caller holds _lock."""
        stamp = _stamp(self.path)
        if stamp != self._stamp or stamp is None:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                users = data.get("users") if isinstance(data, dict) else None
                self._users = {name: entry for name, entry in (users or {}).items() if isinstance(entry, dict)}
            except FileNotFoundError:
                self._users = {}
            except (OSError, ValueError) as exc:
                raise AuthError(f"Cannot read {self.path}: {exc}") from exc
            self._stamp = stamp
        return self._users

    @contextmanager
    def _editing(self):
        """Read-modify-write under the file lock; yields the current accounts, saved afterwards."""
        with self._lock, _file_lock(self.lock_path):
            self._stamp = None  # always re-read under the lock
            users = {name: dict(entry) for name, entry in self._load().items()}
            yield users
            _write_json(self.path, {"version": 1, "users": users})
            self._users, self._stamp = users, _stamp(self.path)

    def get(self, name: str) -> dict | None:
        with self._lock:
            entry = self._load().get(name)
            return dict(entry) if entry else None

    def user(self, name: str) -> User | None:
        entry = self.get(name)
        return _user(name, entry) if entry else None

    def list(self) -> list[dict]:
        with self._lock:
            return [{"name": name, "display_name": entry.get("display_name") or name, "admin": bool(entry.get("admin")), "created": entry.get("created"), "last_login": entry.get("last_login")} for name, entry in sorted(self._load().items())]

    def count(self) -> int:
        with self._lock:
            return len(self._load())

    def add(self, name: str, password: str, display_name: str | None = None, admin: bool = False) -> User:
        name = normalize_name(name)
        check_password_rules(password)
        hashed = hash_password(password)
        with self._editing() as users:
            if name in users:
                raise AuthError(f"User {name} already exists")
            # a random first generation: sessions of an earlier account with this name never match it
            users[name] = {"display_name": (display_name or "").strip()[:80] or name, "password": hashed, "admin": bool(admin), "generation": secrets.randbelow(2**31), "created": _now(), "last_login": None}
            return _user(name, users[name])

    def set_password(self, name: str, password: str) -> int:
        """A new password; returns the account's new generation (its earlier sessions end)."""
        check_password_rules(password)
        hashed = hash_password(password)
        with self._editing() as users:
            entry = _require(users, name)
            entry["password"] = hashed
            entry["generation"] = int(entry.get("generation", 1)) + 1
            entry["password_changed"] = _now()
            return entry["generation"]

    def update(self, name: str, display_name: str | None = None, admin: bool | None = None) -> User:
        with self._editing() as users:
            entry = _require(users, name)
            if display_name is not None:
                entry["display_name"] = display_name.strip()[:80] or name
            if admin is not None:
                entry["admin"] = bool(admin)
            return _user(name, entry)

    def remove(self, name: str) -> None:
        with self._editing() as users:
            _require(users, name)
            del users[str(name).strip().lower()]

    def verify(self, name: str, password: str) -> User | None:
        """The account when the password is right; unknown names cost the same time as known ones."""
        try:
            name = normalize_name(name)
        except AuthError:
            name = ""
        entry = self.get(name) if name else None
        if not verify_password(password, entry["password"] if entry else _dummy()) or entry is None:
            return None
        return _user(name, entry)

    def record_login(self, name: str) -> None:
        try:
            with self._editing() as users:
                if name in users:
                    users[name]["last_login"] = _now()
        except OSError:
            pass  # only informative


def _require(users: dict, name: str) -> dict:
    entry = users.get(str(name).strip().lower())
    if entry is None:
        raise AuthError(f"There is no user {name}")
    return entry


def _user(name: str, entry: dict) -> User:
    return User(name, entry.get("display_name") or name, bool(entry.get("admin")))


# ---------- sessions ----------


class SessionStore:
    """sessions.json: {sha256(token): {user, generation, created, expires, renewed}}; written by the server only."""

    def __init__(self, directory: Path, days: float = 30.0):
        self.path = Path(directory) / SESSIONS_FILE
        self.lifetime = max(60.0, float(days) * 86400)
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            now = time.time()
            self._sessions = {key: entry for key, entry in (data.get("sessions") or {}).items() if isinstance(entry, dict) and entry.get("expires", 0) > now}
        except (OSError, ValueError, AttributeError):
            self._sessions = {}

    @staticmethod
    def _key(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _save(self) -> None:
        _write_json(self.path, {"version": 1, "sessions": self._sessions})

    def create(self, user: str, generation: int) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            self._sessions = {key: entry for key, entry in self._sessions.items() if entry.get("expires", 0) > now}  # expired ones go here
            self._sessions[self._key(token)] = {"user": user, "generation": generation, "created": now, "expires": now + self.lifetime, "renewed": now}
            self._save()
        return token

    def lookup(self, token: str) -> tuple[dict, bool] | None:
        """(session, renewed) for a live token; using a session slides its expiry (saved at most hourly)."""
        key = self._key(token)
        now = time.time()
        with self._lock:
            entry = self._sessions.get(key)
            if entry is None:
                return None
            if entry.get("expires", 0) <= now:
                del self._sessions[key]
                return None
            renewed = now - entry.get("renewed", 0) >= min(RENEW_INTERVAL, self.lifetime / 10)
            if renewed:
                entry["expires"] = now + self.lifetime
                entry["renewed"] = now
                try:
                    self._save()
                except OSError:
                    pass  # the renewal is kept in memory; the next one saves it
            return dict(entry), renewed

    def delete(self, token: str) -> None:
        with self._lock:
            if self._sessions.pop(self._key(token), None) is not None:
                self._save()

    def delete_user(self, user: str, keep: str | None = None) -> int:
        keep_key = self._key(keep) if keep else None
        with self._lock:
            doomed = [key for key, entry in self._sessions.items() if entry.get("user") == user and key != keep_key]
            for key in doomed:
                del self._sessions[key]
            if doomed:
                self._save()
            return len(doomed)

    def count(self, user: str | None = None) -> int:
        now = time.time()
        with self._lock:
            return sum(1 for entry in self._sessions.values() if entry.get("expires", 0) > now and (user is None or entry.get("user") == user))


# ---------- the server's view ----------


class Auth:
    """Sign-in for one projects folder (`directory` is its .workbench/)."""

    def __init__(self, directory: Path, session_days: float = 30.0):
        self.users = UserStore(directory)
        self.sessions = SessionStore(directory, session_days)
        # two servers on one host (another port) must not overwrite each other's sign-in cookie
        self.cookie = f"{COOKIE}_{hashlib.sha256(str(Path(directory).resolve()).encode()).hexdigest()[:8]}"

    @property
    def max_age(self) -> int:
        return int(self.sessions.lifetime)

    def login(self, name: str, password: str) -> tuple[User, str]:
        user = self.users.verify(name, password)
        if user is None:
            if self.users.count() == 0:
                raise AuthError("No user accounts exist yet. Create one on the server: yolo-workbench-server user add <name> --admin")
            raise AuthError("Wrong user name or password")
        entry = self.users.get(user.name) or {}
        token = self.sessions.create(user.name, int(entry.get("generation", 1)))
        self.users.record_login(user.name)
        return user, token

    def authenticate(self, token: str) -> tuple[User, bool] | None:
        """(user, renewed) for a valid session token; None when it expired, was revoked, or its user is gone."""
        found = self.sessions.lookup(token)
        if found is None:
            return None
        session, renewed = found
        entry = self.users.get(session["user"])
        if entry is None or int(entry.get("generation", 1)) != session.get("generation"):
            self.sessions.delete(token)  # password changed or the user was removed
            return None
        return _user(session["user"], entry), renewed

    def logout(self, token: str | None) -> None:
        if token:
            self.sessions.delete(token)

    def change_password(self, user: User, current: str, new: str) -> str:
        """Set a new password after checking the current one; every session of the user ends, and the new
        token returned keeps the browser that asked signed in."""
        if self.users.verify(user.name, current) is None:
            raise AuthError("The current password is wrong")
        generation = self.users.set_password(user.name, new)
        self.sessions.delete_user(user.name)
        return self.sessions.create(user.name, generation)


    def session_cookie(self, token: str) -> str:
        # HttpOnly: scripts cannot read it; Lax: other sites' forms do not send it; no Secure: plain HTTP on this network
        return f"{self.cookie}={token}; Max-Age={self.max_age}; Path=/; HttpOnly; SameSite=Lax"

    def clear_cookie(self) -> str:
        return f"{self.cookie}=; Max-Age=0; Path=/; HttpOnly; SameSite=Lax"

    def cookie_token(self, header: str | None) -> str | None:
        for part in (header or "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == self.cookie and value:
                return value
        return None


class AuthMiddleware:
    """Requests need a signed-in session (see OPEN_PATHS); sets current_user() for everything downstream."""

    def __init__(self, app, auth: Auth):
        self.app = app
        self.auth = auth

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        headers = dict((key.decode("latin-1"), value.decode("latin-1")) for key, value in scope.get("headers") or [])
        token = self.auth.cookie_token(headers.get("cookie"))
        found = await run_in_threadpool(self.auth.authenticate, token) if token else None
        if found is None:
            if path in OPEN_PATHS:
                await self.app(scope, receive, send)
                return
            await self._refuse(scope, send, path, bool(token))
            return
        user, renewed = found
        if renewed:
            cookie = self.auth.session_cookie(token).encode("latin-1")

            prefix = f"{self.auth.cookie}=".encode("latin-1")

            async def send_renewed(message):
                # not when the response sets the cookie itself (a password change issues a new token)
                if message["type"] == "http.response.start" and not any(key.lower() == b"set-cookie" and value.startswith(prefix) for key, value in message.get("headers", [])):
                    message = {**message, "headers": [*message.get("headers", []), (b"set-cookie", cookie)]}
                await send(message)
        else:
            send_renewed = send
        reset = _USER.set(user)
        try:
            await self.app(scope, receive, send_renewed)
        finally:
            _USER.reset(reset)

    async def _refuse(self, scope, send, path: str, had_token: bool) -> None:
        extra = [(b"set-cookie", self.auth.clear_cookie().encode("latin-1"))] if had_token else []
        if path == "/" or path.endswith(".html"):
            query = scope.get("query_string", b"").decode("latin-1")
            target = "/login.html?next=" + quote(path + (f"?{query}" if query else ""), safe="")
            await send({"type": "http.response.start", "status": 303, "headers": [(b"location", target.encode("latin-1")), (b"cache-control", b"no-store"), (b"content-length", b"0"), *extra]})
            await send({"type": "http.response.body", "body": b""})
            return
        body = json.dumps({"detail": "Sign in to continue" if not had_token else "Your sign-in has expired; sign in again"}).encode("utf-8")
        await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode("ascii")), (b"cache-control", b"no-store"), *extra]})
        await send({"type": "http.response.body", "body": body})


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
