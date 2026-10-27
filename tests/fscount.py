"""Count the filesystem calls made under a folder: on a network drive each is a round trip to the file server.

Python 3.11+ (pathlib calls os.stat/os.scandir/io.open at call time; on 3.10 Path methods are not counted).
"""

from __future__ import annotations

import builtins
import collections
import contextlib
import io
import os
import time


@contextlib.contextmanager
def counting(root, latency: float = 0.0):
    """Within the block, calls (stat, lstat, scandir, listdir, open) on paths under `root` are counted by name.
    `latency`: seconds each of them waits, to time code as on a network drive (1-3 ms is typical for SMB/NFS)."""
    root = os.fspath(root)
    counts: collections.Counter = collections.Counter()

    def wrap(name, function):
        def wrapper(path=None, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                text = os.fspath(path)
                if (text.decode(errors="ignore") if isinstance(text, bytes) else text).startswith(root):
                    counts[name] += 1
                    if latency:
                        time.sleep(latency)
            return function(*args, **kwargs) if path is None else function(path, *args, **kwargs)
        return wrapper

    originals = {(os, name): getattr(os, name) for name in ("stat", "lstat", "scandir", "listdir", "open")}
    originals[(io, "open")] = io.open
    originals[(builtins, "open")] = builtins.open
    try:
        for (module, name), function in originals.items():
            setattr(module, name, wrap(name, function))
        yield counts
    finally:
        for (module, name), function in originals.items():
            setattr(module, name, function)
