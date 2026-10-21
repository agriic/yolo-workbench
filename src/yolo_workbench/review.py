"""Per-image bookkeeping: review status ("done" / "needs work") and who last reviewed and edited an image.

Kept in <dataset>/.yolo-workbench/review.json keyed by the dataset-relative image path, so it
survives restarts. It is bookkeeping, not dataset content: not on the undo stack and allowed
while a training run makes the dataset read-only. Entries of deleted images are kept, so an
undone delete gets its status back.

An entry: {"status", "time", "by"} for the review (by: {name, display_name}, only with sign-in), and
{"edited_by", "edited_at"} for the last change of the image's labels by a signed-in user.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

STATUSES = ("done", "needs_work")
VERSION = 1


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class ReviewStore:
    def __init__(self, path: Path, root: Path):
        self.path = path
        self.root = root
        self.entries: dict[str, dict] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            images = data.get("images") if isinstance(data, dict) else None
            if isinstance(images, dict):
                self.entries = {str(key): value for key, value in images.items() if isinstance(value, dict) and (value.get("status") in STATUSES or value.get("edited_by"))}
        except (OSError, ValueError):
            pass  # no file yet, or unreadable: start empty

    def key(self, image_path: Path) -> str:
        return image_path.relative_to(self.root).as_posix() if image_path.is_relative_to(self.root) else image_path.as_posix()

    def get(self, image_path: Path) -> str | None:
        entry = self.entries.get(self.key(image_path))
        status = entry.get("status") if entry else None
        return status if status in STATUSES else None

    def info(self, image_path: Path) -> dict:
        return dict(self.entries.get(self.key(image_path)) or {})

    def reviewer(self, image_path: Path) -> str | None:
        """The name of who set the current review status (None: unreviewed, or set without sign-in)."""
        entry = self.entries.get(self.key(image_path)) or {}
        by = entry.get("by") if entry.get("status") in STATUSES else None
        return by.get("name") if isinstance(by, dict) else None

    def set(self, image_path: Path, status: str | None, by: dict | None = None) -> None:
        key = self.key(image_path)
        entry = dict(self.entries.get(key) or {})
        for field in ("status", "time", "by"):
            entry.pop(field, None)
        if status is not None:
            entry.update(status=status, time=_now())
            if by:
                entry["by"] = by
        self._put(key, entry)

    def edited(self, image_path: Path, by: dict) -> None:
        key = self.key(image_path)
        self._put(key, {**(self.entries.get(key) or {}), "edited_by": by, "edited_at": _now()})

    def _put(self, key: str, entry: dict) -> None:
        if entry:
            self.entries[key] = entry
        else:
            self.entries.pop(key, None)

    def move(self, old: Path, new: Path) -> bool:
        """Follow an image to its new path; True when an entry moved (and the store needs saving)."""
        entry = self.entries.pop(self.key(old), None)
        if entry is None:
            return False
        self.entries[self.key(new)] = entry
        return True

    def save(self, write) -> None:
        """Persist with `write(path, text)` (the dataset's atomic write); raises OSError."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write(self.path, json.dumps({"version": VERSION, "images": self.entries}, separators=(",", ":")))
