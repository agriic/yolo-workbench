"""Who is acting: the signed-in user, the browser tab, and what they are doing, for the code a request runs.

Context variables, so they follow a request into the threads FastAPI runs endpoints in; background work
started by a request captures them (`snapshot`) and restores them in its thread (`restored`). Standard
library only: the dataset records them in its history, review bookkeeping and activity log.

- user: set by auth.AuthMiddleware (None without sign-in: the desktop app, or a server with sign-in off)
- client: the browser tab (X-Workbench-Client header), so a tab can tell its own changes from others'
- action: a short label of the operation ("delete images"); the outermost label wins
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

CLIENT_HEADER = "x-workbench-client"
_CLIENT_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class User:
    name: str
    display_name: str
    admin: bool = False

    def public(self) -> dict:
        return {"name": self.name, "display_name": self.display_name, "admin": self.admin}


_USER: ContextVar[User | None] = ContextVar("workbench_user", default=None)
_CLIENT: ContextVar[str | None] = ContextVar("workbench_client", default=None)
_ACTION: ContextVar[str | None] = ContextVar("workbench_action", default=None)


def current_user() -> User | None:
    """Who made the request being handled (None without sign-in)."""
    return _USER.get()


def current_user_name() -> str | None:
    user = _USER.get()
    return user.name if user else None


def who() -> dict | None:
    """The acting user as stored with what they did ({name, display_name}), or None."""
    user = _USER.get()
    return {"name": user.name, "display_name": user.display_name} if user else None


def current_client() -> str | None:
    return _CLIENT.get()


def current_action(default: str = "edit") -> str:
    return _ACTION.get() or default


@contextmanager
def acting_as(user: User | None, client: str | None = None):
    """Run code as `user` (from browser tab `client`)."""
    user_token, client_token = _USER.set(user), _CLIENT.set(client)
    try:
        yield
    finally:
        _USER.reset(user_token)
        _CLIENT.reset(client_token)


@contextmanager
def action(label: str):
    """Label what the enclosed code does; an enclosing label wins ("fix issue" stays, though it saves labels)."""
    if _ACTION.get() is not None:
        yield
        return
    token = _ACTION.set(label)
    try:
        yield
    finally:
        _ACTION.reset(token)


def snapshot() -> tuple[User | None, str | None]:
    """The acting user and tab, for background work started now."""
    return _USER.get(), _CLIENT.get()


@contextmanager
def restored(saved: tuple[User | None, str | None]):
    """Act as the snapshot taken when the background work was started."""
    with acting_as(*saved):
        yield


def client_from_header(value: str | None) -> str | None:
    return value if value and _CLIENT_PATTERN.match(value) else None


class RequestContextMiddleware:
    """Sets the browser tab (X-Workbench-Client) for everything downstream."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        value = next((raw.decode("latin-1") for key, raw in scope.get("headers") or [] if key == CLIENT_HEADER.encode()), None)
        token = _CLIENT.set(client_from_header(value))
        try:
            await self.app(scope, receive, send)
        finally:
            _CLIENT.reset(token)
