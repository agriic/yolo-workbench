"""Folder listings, so a network drive is asked once per folder instead of once per file. Standard library only.

On a network drive (an SMB share, a mapped Windows drive, NFS) every stat, exists or open is a round trip to the
file server, milliseconds instead of microseconds, so asking about thousands of files one at a time takes
seconds. One directory listing returns a whole folder in a few round trips, with each file's size and
modification time: on Windows os.scandir's entries carry them (DirEntry.stat() makes no call), and Linux's NFS
and SMB clients return them with the listing (READDIRPLUS, QUERY_DIRECTORY) and answer the stat right after it
from their cache.
"""

from __future__ import annotations

import os
import stat as stat_module
from pathlib import Path
from typing import Callable

_FILE, _LINK, _DIR, _LINK_DIR = 0, 1, 2, 3  # a file, a symlink to one, a folder, a symlink to one


class Listing:
    """What folders held when they were first asked about (each is listed once). `stat=False` lists names only.

    Lookups follow the filesystem's case rules: Windows' (os.path.normcase), and elsewhere a name differing only
    in case from a listed one is asked about directly (a case-insensitive macOS volume may still open it).
    """

    def __init__(self, stat: bool = True):
        self._stat = stat
        self._folders: dict[str, dict[str, tuple] | None] = {}
        self._folded: dict[str, set[str]] = {}

    def folder(self, path) -> dict[str, tuple] | None:
        """normcase(name) -> (name, kind, mtime_ns, size) of a folder's entries; None when it is not a folder."""
        key = os.path.normcase(os.fspath(path))
        try:
            return self._folders[key]
        except KeyError:
            pass
        entries: dict[str, tuple] | None = None
        try:
            with os.scandir(path) as iterator:
                entries = {}
                for entry in iterator:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            info = (entry.name, _DIR, None, None)
                        elif entry.is_symlink() and entry.is_dir():
                            info = (entry.name, _LINK_DIR, None, None)
                        else:
                            kind = _LINK if entry.is_symlink() else _FILE
                            if self._stat:
                                result = entry.stat()
                                info = (entry.name, kind, result.st_mtime_ns, result.st_size)
                            else:
                                info = (entry.name, kind, None, None)
                    except OSError:
                        continue  # removed meanwhile, or a broken link
                    entries[os.path.normcase(entry.name)] = info
        except OSError:
            entries = None
        self._folders[key] = entries
        return entries

    def _entry(self, path) -> tuple | None:
        directory, name = os.path.split(os.fspath(path))
        entries = self.folder(directory)
        if entries is None:
            return None
        info = entries.get(os.path.normcase(name))
        if info is None and os.path.normcase("A") == "A":
            folded = self._folded.get(directory)
            if folded is None:
                folded = self._folded[directory] = {listed[0].casefold() for listed in entries.values()}
            if name.casefold() in folded:
                try:
                    result = os.stat(path)
                except OSError:
                    return None
                kind = _DIR if stat_module.S_ISDIR(result.st_mode) else _FILE
                info = (name, kind, result.st_mtime_ns, result.st_size)
        return info

    def exists(self, path) -> bool:
        return self._entry(path) is not None

    def is_dir(self, path) -> bool:
        info = self._entry(path)
        return info is not None and info[1] in (_DIR, _LINK_DIR)

    def is_symlink(self, path) -> bool:
        info = self._entry(path)
        return info is not None and info[1] in (_LINK, _LINK_DIR)

    def stat(self, path) -> tuple[int, int] | None:
        """(mtime_ns, size) of a file; None when there is none (or names only were listed)."""
        info = self._entry(path)
        if info is None or info[1] not in (_FILE, _LINK) or info[2] is None:
            return None
        return info[2], info[3]

    def subfolders(self, path) -> list[Path]:
        """The folders in a folder (symlinked ones too), sorted by name."""
        entries = self.folder(path) or {}
        return sorted((Path(path, info[0]) for info in entries.values() if info[1] in (_DIR, _LINK_DIR)), key=lambda item: item.name)

    def files(self, root, prune: Callable[[str, str], bool] | None = None) -> list[str]:
        """Every file below `root`, recursively, as paths joined onto it. Like Path.rglob, symlinked folders below
        it are not entered; `prune(path, name)` leaves out a folder and everything in it."""
        found: list[str] = []
        stack = [os.fspath(root)]
        while stack:
            folder = stack.pop()
            for name, kind, _, _ in (self.folder(folder) or {}).values():
                path = os.path.join(folder, name)
                if kind == _DIR:
                    if prune is None or not prune(path, name):
                        stack.append(path)
                elif kind in (_FILE, _LINK):
                    found.append(path)
        return found


def read_text(path) -> tuple[str | None, int | None, str | None]:
    """A text file's (content, mtime_ns, error) with one open: (None, None, None) when it does not exist,
    (None, mtime or None, why) when it cannot be read. Newlines are read as Path.read_text reads them."""
    try:
        with open(path, encoding="utf-8") as stream:
            mtime = os.fstat(stream.fileno()).st_mtime_ns
            return stream.read(), mtime, None
    except FileNotFoundError:
        return None, None, None
    except (OSError, UnicodeDecodeError) as exc:
        try:
            mtime = os.stat(path).st_mtime_ns
        except OSError:
            mtime = None
        return None, mtime, str(exc)
