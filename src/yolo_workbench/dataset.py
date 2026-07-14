from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
    def __init__(self, yaml_path: Path, category: Category, background_probe: bool = False):
        self.yaml_path = yaml_path.resolve()
        self.category = category
        self.root = self.yaml_path.parent
        self.names: dict[int, str] = {}
        self.images: dict[str, ImageRecord] = {}
        self.orphan_labels: list[Path] = []
        self._mtimes: dict[Path, int | None] = {}
        self._lock = threading.RLock()
        # each history entry is a transaction: one or more label files written together
        self._undo: list[list[HistoryEntry]] = []
        self._redo: list[list[HistoryEntry]] = []
        self.session_id = time.strftime("%Y%m%d-%H%M%S")
        self._backed_up: set[Path] = set()
        self.indexing: dict[str, int] = {"done": 0, "total": 0}
        self._load()
        pending = [record for record in self.images.values() if record.width == 0 and record.probe_error is None]
        self.indexing = {"done": len(self.images) - len(pending), "total": len(self.images)}
        if not pending:
            return
        if background_probe:
            threading.Thread(target=self._probe_all, args=(pending,), daemon=True).start()
        else:
            self._probe_all(pending)

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
        size_index = self._load_size_index()
        for split in ("train", "val", "test"):
            for image_path in self._resolve_sources(config.get(split)):
                record_id = hashlib.sha1(f"{split}:{image_path}".encode()).hexdigest()[:16]
                label_path = self._label_for(image_path)
                record = ImageRecord(record_id, split, image_path, label_path)
                record.name_cf = image_path.name.casefold()
                self._apply_cached_size(record, size_index)
                self._read_record(record)
                self.images[record_id] = record
        if not self.images:
            raise DatasetError("No supported images found in dataset splits")
        self._find_orphans()

    @property
    def _index_path(self) -> Path:
        return self.root / ".yolo-workbench" / "index.json"

    def _load_size_index(self) -> dict:
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _apply_cached_size(self, record: ImageRecord, size_index: dict) -> None:
        entry = size_index.get(str(record.path))
        if not entry:
            return
        try:
            stat = record.path.stat()
        except OSError:
            return
        if entry.get("mtime_ns") == stat.st_mtime_ns and entry.get("size") == stat.st_size:
            record.width, record.height = int(entry.get("width", 0)), int(entry.get("height", 0))

    def _probe_all(self, pending: list[ImageRecord]) -> None:
        with ThreadPoolExecutor(max_workers=12) as pool:
            for _ in pool.map(self._probe, pending):
                pass
        self._save_size_index()

    def _probe(self, record: ImageRecord) -> None:
        try:
            with Image.open(record.path) as image:
                width, height = image.size
            error = None
        except (OSError, UnidentifiedImageError) as exc:
            width = height = 0
            error = str(exc)
        with self._lock:
            record.width, record.height = width, height
            record.probe_error = error
            if error:
                record.issues.append(self._issue("unreadable_image", "error", error, record))
            self.indexing["done"] += 1

    def _save_size_index(self) -> None:
        entries = {str(record.path): {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size, "width": record.width, "height": record.height} for record in self.images.values() if record.width > 0 and (stat := self._safe_stat(record.path))}
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write(self._index_path, json.dumps(entries))
        except OSError:
            pass  # the index is only a cache; failing to persist it must never break the app

    @staticmethod
    def _safe_stat(path: Path):
        try:
            return path.stat()
        except OSError:
            return None

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
        if record.probe_error:
            record.issues.append(self._issue("unreadable_image", "error", record.probe_error, record))
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
        record.class_ids = {annotation.class_id for annotation in record.annotations}
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
            records = [record for record in records if class_id in record.class_ids]
        if search:
            query = search.casefold()
            records = [record for record in records if query in record.name_cf]
        records.sort(key=lambda record: (record.split, record.name_cf))
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
        self._commit_many([self._prepare(record, raw_annotations)])
        return self.detail(image_id)

    def _prepare(self, record: ImageRecord, raw_annotations: list[dict]) -> tuple[ImageRecord, str, str]:
        """Validate raw annotations and return a (record, new content, current content) change."""
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
        before = record.label_path.read_text(encoding="utf-8") if record.label_path.exists() else ""
        return record, content, before

    def _number(self, value: float) -> str:
        return f"{value:.6f}".rstrip("0").rstrip(".") or "0"

    def _commit_many(self, changes: list[tuple[ImageRecord, str, str]], history=True) -> None:
        """Write all changes as one transaction: every mtime is checked before any file is touched."""
        with self._lock:
            for record, _, _ in changes:
                current_mtime = record.label_path.stat().st_mtime_ns if record.label_path.exists() else None
                if current_mtime != self._mtimes.get(record.label_path):
                    raise WriteConflict(f"{record.label_path.name} changed outside this application; reload before saving")
            transaction = []
            for record, content, before in changes:
                self._backup(record.label_path, before)
                self._atomic_write(record.label_path, content)
                transaction.append(HistoryEntry(record.label_path, before, content))
                self._read_record(record)
            if history and transaction:
                self._undo.append(transaction)
                self._redo.clear()

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
        transaction = source.pop()
        entries = list(reversed(transaction)) if direction == "undo" else transaction
        changes = []
        for entry in entries:
            record = next(record for record in self.images.values() if record.label_path == entry.label_path)
            content = entry.before if direction == "undo" else entry.after
            before = entry.after if direction == "undo" else entry.before
            changes.append((record, content, before))
        try:
            self._commit_many(changes, history=False)
        except WriteConflict:
            source.append(transaction)
            raise
        target.append(transaction)
        image_ids = [record.id for record, _, _ in changes]
        return {"image_ids": image_ids, "image_id": image_ids[0], "can_undo": bool(self._undo), "can_redo": bool(self._redo)}

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

    def fix_issues_bulk(self, kind: str, split: str | None = None, issue_ids: list[str] | None = None) -> dict:
        """Fix every matching fixable issue in one transaction; skips images that fail validation."""
        wanted = set(issue_ids) if issue_ids else None
        changes, skipped, fixed = [], [], 0
        for record in self.images.values():
            if split is not None and record.split != split:
                continue
            issues = [issue for issue in record.issues if issue["fixable"] and issue["kind"] == kind and (wanted is None or issue["id"] in wanted)]
            if not issues:
                continue
            try:
                changes.append(self._prepare(record, self._repair(record, issues)))
                fixed += len(issues)
            except DatasetError as exc:
                skipped.append({"image_id": record.id, "image_name": record.path.name, "reason": str(exc)})
        if not changes and not skipped:
            raise DatasetError(f"No fixable {kind} issues match")
        if changes:
            self._commit_many(changes)
        return {"fixed": fixed, "files": len(changes), "skipped": skipped}

    def _repair(self, record: ImageRecord, issues: list[dict]) -> list[dict]:
        if any(issue["kind"] == "missing_label" for issue in issues):
            return []
        annotations = [Annotation(a.id, a.class_id, list(a.points)) for a in record.annotations]
        removed: set[str] = set()
        for issue in issues:
            annotation = next((a for a in annotations if a.id == issue["annotation_id"]), None)
            if not annotation:
                continue
            if issue["kind"] == "out_of_range":
                annotation.points = [min(1, max(0, value)) for value in annotation.points]
            elif issue["kind"] in {"zero_area", "duplicate"}:
                removed.add(annotation.id)
        return [a.as_dict() for a in annotations if a.id not in removed]

    def bulk_edit_objects(self, operations: list[dict]) -> dict:
        """Relabel or delete many annotations across images in one transaction."""
        grouped: dict[str, list[dict]] = {}
        for operation in operations:
            grouped.setdefault(operation["image_id"], []).append(operation)
        changes, skipped, applied = [], [], 0
        for image_id, group in grouped.items():
            record = self.images.get(image_id)
            if not record:
                skipped.append({"image_id": image_id, "annotation_id": None, "reason": "Image not found"})
                continue
            annotations = [Annotation(a.id, a.class_id, list(a.points)) for a in record.annotations]
            count = 0
            for operation in group:
                annotation = next((a for a in annotations if a.id == operation["annotation_id"]), None)
                if not annotation:
                    skipped.append({"image_id": image_id, "annotation_id": operation["annotation_id"], "reason": "Annotation not found"})
                    continue
                if operation["action"] == "delete":
                    annotations.remove(annotation)
                elif operation.get("class_id") is None:
                    skipped.append({"image_id": image_id, "annotation_id": operation["annotation_id"], "reason": "Relabel requires class_id"})
                    continue
                else:
                    annotation.class_id = int(operation["class_id"])
                count += 1
            if not count:
                continue
            try:
                changes.append(self._prepare(record, [a.as_dict() for a in annotations]))
                applied += count
            except DatasetError as exc:
                skipped.append({"image_id": image_id, "annotation_id": None, "reason": str(exc)})
        if changes:
            self._commit_many(changes)
        return {"applied": applied, "files": len(changes), "skipped": skipped}
