from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Category = Literal["detection", "segmentation", "classification"]


@dataclass
class Annotation:
    id: str
    class_id: int
    points: list[float]

    def as_dict(self) -> dict:
        return {"id": self.id, "class_id": self.class_id, "points": self.points}


@dataclass
class ImageRecord:
    id: str
    split: str
    path: Path
    label_path: Path
    width: int = 0
    height: int = 0
    annotations: list[Annotation] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    class_ids: set[int] = field(default_factory=set)
    name_cf: str = ""
    probe_error: str | None = None
    review: str | None = None  # "done", "needs_work", or None (unreviewed); see review.py
    reviewed_by: str | None = None  # the user name that set `review` (with sign-in)
    # the image file's modification time and size as last seen (opening, rescan, size probe): what cache keys and
    # Label QA compare with, without asking a network drive about every image (see Dataset.image_stamp)
    image_mtime: int | None = None
    image_size: int | None = None

    def summary(self) -> dict:
        return {
            "id": self.id,
            "split": self.split,
            "name": self.path.name,
            "width": self.width,
            "height": self.height,
            "annotation_count": len(self.annotations),
            "classes": sorted(self.class_ids),
            "issue_count": len(self.issues),
            "review": self.review,
            "reviewed_by": self.reviewed_by,
        }
