"""One workbench process per projects folder (or dataset): an OS file lock held for the process lifetime.

Locks and mtime checks inside one process do not coordinate two processes, so a second server, or a
desktop session on the same folder, refuses to start. The lock is released by the OS when the process
ends, so a crash leaves nothing stale; `owner.json` beside it only says who holds it.

Network filesystems with lock support (SMB, NFS with its lock service) honour the lock across machines. One
without it (some NAS and FUSE mounts) fails every lock attempt: that is logged and the folder opens anyway,
since refusing would make it unusable; only a lock held by another process refuses.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

from .release import release_name

LOCK_NAME = "owner.lock"
INFO_NAME = "owner.json"
# what a filesystem without lock support answers (not "held by someone else", which is EAGAIN/EACCES)
UNSUPPORTED = {errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOSYS, errno.EINVAL}

log = logging.getLogger(__name__)


class AlreadyOwned(RuntimeError):
    pass


class OwnerLock:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.path = self.directory / LOCK_NAME
        self._stream = None
        self.unlocked: str | None = None  # why no OS lock is held (the filesystem has no lock support)

    @property
    def held(self) -> bool:
        return self._stream is not None

    def acquire(self) -> "OwnerLock":
        self.directory.mkdir(parents=True, exist_ok=True)
        stream = open(self.path, "a+b")
        try:
            _lock(stream)
        except OSError as exc:
            if isinstance(exc, BlockingIOError) or exc.errno not in UNSUPPORTED:
                stream.close()
                raise AlreadyOwned(self._describe_owner()) from None
            self.unlocked = f"{exc.strerror or exc} (errno {exc.errno})"
            log.warning("%s: the filesystem does not support file locks (%s): a second workbench on this folder would not be refused", self.path, self.unlocked)
        self._stream = stream
        info = {"pid": os.getpid(), "host": socket.gethostname(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "command": " ".join(sys.argv), "release": release_name()}
        try:
            (self.directory / INFO_NAME).write_text(json.dumps(info), encoding="utf-8")
        except OSError:
            pass  # only informative
        return self

    def release(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            (self.directory / INFO_NAME).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            if self.unlocked is None:
                _unlock(stream)
        finally:
            stream.close()

    def __enter__(self) -> "OwnerLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()

    def _describe_owner(self) -> str:
        try:
            info = json.loads((self.directory / INFO_NAME).read_text(encoding="utf-8"))
            release = f", release {info['release']}" if info.get("release") else ""
            owner = f"process {info.get('pid')} on {info.get('host')} since {info.get('started')}{release} ({info.get('command')})"
        except (OSError, ValueError, AttributeError):
            owner = "another process"
        return f"{self.directory.parent} is already in use by {owner}"


if sys.platform == "win32":
    import msvcrt

    def _lock(stream) -> None:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(stream) -> None:
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

else:
    import fcntl

    def _lock(stream) -> None:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(stream) -> None:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
