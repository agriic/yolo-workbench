from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError

from .models import Annotation, Category, ImageRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class DatasetError(ValueError):
    pass


class WriteConflict(RuntimeError):
    pass


@dataclass
class HistoryEntry:
    label_path: Path
    before: str
    after: str


class Dataset:
    def __init__(self, yaml_path: Path, category: Category):
        self.yaml_path = yaml_path.resolve()
        self.category = category
        self.root = self.yaml_path.parent
        self.names: dict[int, str] = {}
        self.images: dict[str, ImageRecord] = {}
        self.orphan_labels: list[Path] = []
        self._mtimes: dict[Path, int | None] = {}
        self._lock = threading.RLock()
        self._undo: list[HistoryEntry] = []
        self._redo: list[HistoryEntry] = []
        self.session_id = time.strftime("%Y%m%d-%H%M%S")
        self._backed_up: set[Path] = set()
        self._load()

    def _load(self) -> None:
        try:
            config = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise DatasetError(f"Cannot read dataset YAML: {exc}") from exc
        raw_names = config.get("names")
        if isinstance(raw_names, list):
            self.names = dict(enumerate(str(value) for value in raw_names))
        elif isinstance(raw_names, dict):
            try:
                self.names = {int(key): str(value) for key, value in raw_names.items()}
            except (TypeError, ValueError) as exc:
                raise DatasetError("Dataset names must use integer class IDs") from exc
        if not self.names:
            raise DatasetError("Dataset YAML has no class names")
        root_value = config.get("path", ".")
        root = Path(root_value)
        self.root = (self.yaml_path.parent / root).resolve() if not root.is_absolute() else root.resolve()
        for split in ("train", "val", "test"):
            for image_path in self._resolve_sources(config.get(split)):
                record_id = hashlib.sha1(f"{split}:{image_path}".encode()).hexdigest()[:16]
                label_path = self._label_for(image_path)
                record = ImageRecord(record_id, split, image_path, label_path)
                self._read_record(record)
                self.images[record_id] = record
        if not self.images:
            raise DatasetError("No supported images found in dataset splits")
        self._find_orphans()

    def _resolve_sources(self, value) -> list[Path]:
        if not value:
            return []
        values = value if isinstance(value, list) else [value]
        found: list[Path] = []
        for item in values:
            source = Path(str(item))
            source = (self.root / source).resolve() if not source.is_absolute() else source.resolve()
            if source.is_dir():
                found.extend(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
            elif source.suffix.lower() == ".txt" and source.exists():
                for line in source.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        path = Path(line.strip())
                        found.append((self.root / path).resolve() if not path.is_absolute() else path.resolve())
            elif source.suffix.lower() in IMAGE_EXTENSIONS and source.exists():
                found.append(source)
        return sorted(set(found))

    def _label_for(self, image_path: Path) -> Path:
        parts = list(image_path.parts)
        if "images" in parts:
            index = len(parts) - 1 - parts[::-1].index("images")
            candidate = Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt")
            label_root = self.root / "labels"
            if candidate.exists() or label_root.exists():
                return candidate
        return image_path.with_suffix(".txt")

    def _read_record(self, record: ImageRecord) -> None:
        record.issues.clear()
        try:
            with Image.open(record.path) as image:
                record.width, record.height = image.size
        except (OSError, UnidentifiedImageError) as exc:
            record.issues.append(self._issue("unreadable_image", "error", str(exc), record))
        text = ""
        if record.label_path.exists():
            try:
                text = record.label_path.read_text(encoding="utf-8")
            except OSError as exc:
                record.issues.append(self._issue("unreadable_label", "error", str(exc), record))
        else:
            record.issues.append(self._issue("missing_label", "warning", "Label file is missing", record, fixable=True))
        record.annotations = []
        seen: dict[tuple, str] = {}
        for line_number, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                values = stripped.split()
                class_id = int(values[0])
                points = [float(value) for value in values[1:]]
                expected = len(points) == 4 if self.category == "detection" else len(points) >= 6 and len(points) % 2 == 0
                if not expected or not all(value == value and abs(value) != float("inf") for value in points):
                    raise ValueError("invalid coordinate count or value")
                annotation = Annotation(f"{record.id}:{line_number}", class_id, points)
                record.annotations.append(annotation)
                if class_id not in self.names:
                    record.issues.append(self._issue("unknown_class", "error", f"Unknown class {class_id}", record, annotation.id))
                if any(value < 0 or value > 1 for value in points):
                    record.issues.append(self._issue("out_of_range", "warning", "Coordinates outside 0..1", record, annotation.id, True))
                if self._zero_area(annotation):
                    record.issues.append(self._issue("zero_area", "warning", "Annotation has zero area", record, annotation.id, True))
                key = (class_id, *points)
                if key in seen:
                    record.issues.append(self._issue("duplicate", "warning", "Exact duplicate annotation", record, annotation.id, True))
                seen[key] = annotation.id
            except (ValueError, IndexError) as exc:
                record.issues.append(self._issue("malformed_label", "error", f"Line {line_number}: {exc}", record))
        self._mtimes[record.label_path] = record.label_path.stat().st_mtime_ns if record.label_path.exists() else None

    def _zero_area(self, annotation: Annotation) -> bool:
        if self.category == "detection":
            return annotation.points[2] <= 0 or annotation.points[3] <= 0
        points = list(zip(annotation.points[::2], annotation.points[1::2]))
        area = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))
        return abs(area) < 1e-10

    def _issue(self, kind, severity, message, record, annotation_id=None, fixable=False):
        return {"id": hashlib.sha1(f"{record.id}:{kind}:{annotation_id}:{message}".encode()).hexdigest()[:16], "kind": kind, "severity": severity, "message": message, "image_id": record.id, "image_name": record.path.name, "split": record.split, "annotation_id": annotation_id, "fixable": fixable}

    def _find_orphans(self) -> None:
        expected = {record.label_path.resolve() for record in self.images.values()}
        candidates = set()
        for parent in {record.label_path.parent for record in self.images.values()}:
            if parent.exists():
                candidates.update(path.resolve() for path in parent.glob("*.txt"))
        self.orphan_labels = sorted(candidates - expected)

    def list_images(self, split="all", class_id=None, search="", offset=0, limit=100) -> dict:
        records = list(self.images.values())
        if split != "all":
            records = [record for record in records if record.split == split]
        if class_id is not None:
            records = [record for record in records if any(a.class_id == class_id for a in record.annotations)]
        if search:
            query = search.casefold()
            records = [record for record in records if query in record.path.name.casefold()]
        records.sort(key=lambda record: (record.split, record.path.name.casefold()))
        return {"total": len(records), "items": [record.summary() for record in records[offset : offset + limit]]}

    def detail(self, image_id: str) -> dict:
        record = self.require_image(image_id)
        return {**record.summary(), "path": str(record.path), "annotations": [a.as_dict() for a in record.annotations], "issues": record.issues}

    def require_image(self, image_id: str) -> ImageRecord:
        try:
            return self.images[image_id]
        except KeyError as exc:
            raise KeyError("Image not found") from exc

    def replace_annotations(self, image_id: str, raw_annotations: list[dict]) -> dict:
        record = self.require_image(image_id)
        if any(issue["kind"] == "malformed_label" for issue in record.issues):
            raise DatasetError("Resolve or remove malformed label rows before editing this image")
        annotations = []
        for index, raw in enumerate(raw_annotations):
            class_id = int(raw["class_id"])
            points = [float(value) for value in raw["points"]]
            if class_id not in self.names:
                raise DatasetError(f"Unknown class {class_id}")
            valid_count = len(points) == 4 if self.category == "detection" else len(points) >= 6 and len(points) % 2 == 0
            if not valid_count or any(value < 0 or value > 1 for value in points):
                raise DatasetError("Invalid annotation geometry")
            annotations.append(Annotation(raw.get("id") or f"{record.id}:new:{index}", class_id, points))
        content = "".join(f"{a.class_id} {' '.join(self._number(v) for v in a.points)}\n" for a in annotations)
        self._commit(record, content, record.label_path.read_text(encoding="utf-8") if record.label_path.exists() else "")
        return self.detail(image_id)

    def _number(self, value: float) -> str:
        return f"{value:.6f}".rstrip("0").rstrip(".") or "0"

    def _commit(self, record: ImageRecord, content: str, before: str, history=True) -> None:
        with self._lock:
            current_mtime = record.label_path.stat().st_mtime_ns if record.label_path.exists() else None
            if current_mtime != self._mtimes.get(record.label_path):
                raise WriteConflict("Label changed outside this application; reload before saving")
            self._backup(record.label_path, before)
            self._atomic_write(record.label_path, content)
            if history:
                self._undo.append(HistoryEntry(record.label_path, before, content))
                self._redo.clear()
            self._read_record(record)

    def _backup(self, label_path: Path, content: str) -> None:
        if label_path in self._backed_up:
            return
        try:
            relative = label_path.resolve().relative_to(self.root)
        except ValueError:
            relative = Path(label_path.name)
        backup = self.root / ".yolo-workbench" / "backups" / self.session_id / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(content, encoding="utf-8")
        self._backed_up.add(label_path)

    def _atomic_write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def history(self, direction: str) -> dict:
        source, target = (self._undo, self._redo) if direction == "undo" else (self._redo, self._undo)
        if not source:
            raise DatasetError(f"Nothing to {direction}")
        entry = source.pop()
        record = next(record for record in self.images.values() if record.label_path == entry.label_path)
        content = entry.before if direction == "undo" else entry.after
        before = entry.after if direction == "undo" else entry.before
        self._commit(record, content, before, history=False)
        target.append(entry)
        return {"image_id": record.id, "can_undo": bool(self._undo), "can_redo": bool(self._redo)}

    def issues(self) -> list[dict]:
        issues = [issue for record in self.images.values() for issue in record.issues]
        issues.extend({"id": hashlib.sha1(str(path).encode()).hexdigest()[:16], "kind": "orphan_label", "severity": "warning", "message": "Label has no matching dataset image", "image_id": None, "image_name": path.name, "split": None, "annotation_id": None, "fixable": False} for path in self.orphan_labels)
        return issues

    def fix_issue(self, issue_id: str) -> dict:
        for record in self.images.values():
            issue = next((item for item in record.issues if item["id"] == issue_id), None)
            if not issue or not issue["fixable"]:
                continue
            if issue["kind"] == "missing_label":
                return self.replace_annotations(record.id, [])
            annotation = next((a for a in record.annotations if a.id == issue["annotation_id"]), None)
            if not annotation:
                raise DatasetError("Annotation no longer exists")
            annotations = [Annotation(a.id, a.class_id, list(a.points)) for a in record.annotations]
            annotation = next(a for a in annotations if a.id == annotation.id)
            if issue["kind"] == "out_of_range":
                annotation.points = [min(1, max(0, value)) for value in annotation.points]
            elif issue["kind"] in {"zero_area", "duplicate"}:
                annotations.remove(annotation)
            return self.replace_annotations(record.id, [a.as_dict() for a in annotations])
        raise DatasetError("Issue not found or cannot be fixed automatically")
