"""Per-image review status ("done" / "needs work"), so a pass over a large dataset can be tracked.

Kept in <dataset>/.yolo-workbench/review.json keyed by the dataset-relative image path, so it
survives restarts. It is bookkeeping, not dataset content: not on the undo stack and allowed
while a training run makes the dataset read-only. Statuses of deleted images are kept, so an
undone delete gets its status back.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

STATUSES = ("done", "needs_work")
VERSION = 1


class ReviewStore:
    def __init__(self, path: Path, root: Path):
        self.path = path
        self.root = root
        self.entries: dict[str, dict] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            images = data.get("images") if isinstance(data, dict) else None
            if isinstance(images, dict):
                self.entries = {str(key): value for key, value in images.items() if isinstance(value, dict) and value.get("status") in STATUSES}
        except (OSError, ValueError):
            pass  # no file yet, or unreadable: start empty

    def key(self, image_path: Path) -> str:
        return image_path.relative_to(self.root).as_posix() if image_path.is_relative_to(self.root) else image_path.as_posix()

    def get(self, image_path: Path) -> str | None:
        entry = self.entries.get(self.key(image_path))
        return entry["status"] if entry else None

    def set(self, image_path: Path, status: str | None) -> None:
        if status is None:
            self.entries.pop(self.key(image_path), None)
        else:
            self.entries[self.key(image_path)] = {"status": status, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def move(self, old: Path, new: Path) -> bool:
        """Follow an image to its new path; True when a status moved (and the store needs saving)."""
        entry = self.entries.pop(self.key(old), None)
        if entry is None:
            return False
        self.entries[self.key(new)] = entry
        return True

    def save(self, write) -> None:
        """Persist with `write(path, text)` (the dataset's atomic write); raises OSError."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write(self.path, json.dumps({"version": VERSION, "images": self.entries}, separators=(",", ":")))
