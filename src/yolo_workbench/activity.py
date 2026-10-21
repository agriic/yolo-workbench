"""Who did what in a dataset: <dataset>/.yolo-workbench/activity.jsonl, one JSON object per line.

    {"time": "2026-10-21T13:02:07", "user": "alice", "display_name": "Alice", "action": "delete images",
     "kind": "labels", "count": 3, "images": ["3f2a…", …]}

Written only when someone is signed in (the server); the desktop app writes nothing. At most MAX_IDS image
ids per line (`count` has them all). Past MAX_BYTES the file moves to activity.1.jsonl (replacing the one
before), so the log keeps roughly the last 2 × MAX_BYTES.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

from .fsutil import replace

MAX_IDS = 200
MAX_BYTES = 16 * 1024 * 1024

log = logging.getLogger(__name__)


class ActivityLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, who: dict, action: str, image_ids, kind: str = "labels", **extra) -> None:
        ids = list(image_ids)
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": who["name"], "display_name": who.get("display_name") or who["name"], "action": action, "kind": kind, "count": len(ids), "images": ids[:MAX_IDS], **extra}
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.exists() and self.path.stat().st_size > MAX_BYTES:
                    replace(self.path, self.path.with_name(self.path.stem + ".1" + self.path.suffix))
                with open(self.path, "a", encoding="utf-8", newline="\n") as stream:
                    stream.write(line)
            except OSError as exc:  # bookkeeping: never fail the change it describes
                log.warning("activity: cannot write %s: %s", self.path, exc)

    def recent(self, limit: int = 100, image_id: str | None = None, user: str | None = None) -> list[dict]:
        """The latest entries, newest first (optionally of one image or one user)."""
        found: deque = deque(maxlen=limit)
        for path in (self.path.with_name(self.path.stem + ".1" + self.path.suffix), self.path):
            try:
                with open(path, encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            entry = json.loads(line)
                        except ValueError:
                            continue  # a line cut short by a crash
                        if image_id is not None and image_id not in entry.get("images", ()):
                            continue
                        if user is not None and entry.get("user") != user:
                            continue
                        found.append(entry)
            except OSError:
                continue
        return list(reversed(found))

    def size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0
