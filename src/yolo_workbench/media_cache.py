from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Callable

# Bump when PALETTE or the rendering style changes so stale cached media self-invalidates.
PALETTE_VERSION = 1


def default_cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "yolo-workbench" / "media"


class MediaCache:
    """Disk cache for rendered thumbnails/crops, keyed by content-derived sha1 keys.

    Keys include image/label mtimes, so edits self-invalidate — no explicit hooks needed.
    """

    def __init__(self, directory: Path | None = None, max_bytes: int = 512 * 1024 * 1024, prune_every: int = 50):
        self.directory = directory or default_cache_dir()
        self.max_bytes = max_bytes
        self.prune_every = prune_every
        self._puts = 0
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha1("|".join(str(part) for part in parts).encode()).hexdigest()

    def get(self, key: str) -> bytes | None:
        path = self.directory / f"{key}.jpg"
        try:
            content = path.read_bytes()
            os.utime(path)  # bump mtime so LRU pruning works on relatime/noatime mounts
            return content
        except OSError:
            return None

    def put(self, key: str, content: bytes) -> None:
        path = self.directory / f"{key}.jpg"
        try:
            fd, temporary = tempfile.mkstemp(prefix=f".{key}.", dir=self.directory)
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
            os.replace(temporary, path)
        except OSError:
            return  # a cache write failure must never break serving
        self._puts += 1
        if self._puts % self.prune_every == 0:
            self.prune()

    def get_or_render(self, key: str, render: Callable[[], bytes]) -> bytes:
        cached = self.get(key)
        if cached is not None:
            return cached
        content = render()
        self.put(key, content)
        return content

    def prune(self) -> None:
        entries = []
        for path in self.directory.glob("*.jpg"):
            try:
                stat = path.stat()
                entries.append((stat.st_mtime_ns, stat.st_size, path))
            except OSError:
                continue
        total = sum(size for _, size, _ in entries)
        if total <= self.max_bytes:
            return
        for _, size, path in sorted(entries):
            try:
                path.unlink()
                total -= size
            except OSError:
                continue
            if total <= self.max_bytes:
                return
