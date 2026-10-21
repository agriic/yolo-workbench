"""File system helpers that behave the same on Windows. Standard library only (train_runner uses it)."""

from __future__ import annotations

import os
import shutil
import time

WINDOWS = os.name == "nt"
RETRY_DELAYS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)  # about 0.9 s in all


def replace(src, dst) -> None:
    """os.replace, retried briefly on Windows.

    Windows refuses to replace or move a file another process has open (the server reading a
    runner's state.json, a thumbnail render, an indexer or antivirus scan); those holds are short.
    """
    if not WINDOWS:
        os.replace(src, dst)
        return
    for delay in RETRY_DELAYS:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(src, dst)


def ensure_free_space(path, needed: int = 0, what: str = "this") -> None:
    """Raise OSError(ENOSPC) when the disk holding `path` would drop below the configured minimum
    (runtime.min_free_bytes) after writing `needed` more bytes. No check when no minimum is set."""
    from .runtime import RUNTIME

    if not RUNTIME.min_free_bytes:
        return
    probe = os.fspath(path)
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return  # cannot tell: let the write itself fail if it must
    if free - needed < RUNTIME.min_free_bytes:
        raise InsufficientSpace(f"Not enough free disk space for {what}: {free / 2**30:.1f} GB free, at least {RUNTIME.min_free_bytes / 2**30:.1f} GB must stay free")


class InsufficientSpace(OSError):
    pass


def occupied(path, source) -> bool:
    """Does `path` exist as something other than `source`?

    On a case-insensitive file system (Windows, macOS) renaming Cat/ to cat/ finds cat/ already
    there: it is the same folder, so the rename is fine.
    """
    if not os.path.exists(path):
        return False
    try:
        return not os.path.samefile(path, source)
    except OSError:
        return True
