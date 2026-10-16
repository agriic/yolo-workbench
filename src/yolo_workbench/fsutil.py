"""File system helpers that behave the same on Windows. Standard library only (train_runner uses it)."""

from __future__ import annotations

import os
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
