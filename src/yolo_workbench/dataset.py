from __future__ import annotations

import hashlib
import json
import math
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
WORKBENCH_DIR = ".yolo-workbench"  # per-dataset state: backups, size index, trash, upload staging


class DatasetError(ValueError):
    pass


class WriteConflict(RuntimeError):
    pass


@dataclass
class HistoryEntry:
    label_path: Path
    before: str
    after: str
    before_exists: bool
    after_exists: bool


@dataclass
class MoveEntry:
    """History entry for classification datasets: the class edit is a file move."""

    image_id: str
    src: Path
    dst: Path


@dataclass
class FileEntry:
    """History entry for a non-label file the dataset owns (data.yaml); replay compares content, not mtimes."""

    path: Path
    before: str
    after: str
    before_exists: bool
    after_exists: bool


@dataclass
class RecordMoveEntry:
    """History entry moving one image (and its label) between splits, into the trash, or back.

    split_before=None means the record did not exist before (restore from trash);
    split_after=None means it is removed from the dataset (moved to the trash).
    """

    image_id: str
    split_before: str | None
    split_after: str | None
    files: list[tuple[Path, Path]]  # (src, dst) pairs; the image comes first


@dataclass
class DirEntry:
    """History entry creating (created=True) or removing an empty directory."""

    path: Path
    created: bool


@dataclass
class DirMoveEntry:
    """History entry renaming a directory (classification class folders)."""

    src: Path
    dst: Path


@dataclass
class SchemaEntry:
    """Marks a transaction that changed class names; replay reloads names from disk.

    mapping is old class id -> new class id (None = removed) for listeners such as the predictor.
    """

    mapping: dict[int, int | None]
    layout_only: bool = False  # only split folders changed; class ids are untouched

    def inverse(self) -> dict[int, int | None]:
        return {new: old for old, new in self.mapping.items() if new is not None}


@dataclass
class PreparedChange:
    record: ImageRecord
    before: str
    after: str
    before_exists: bool
    after_exists: bool
    base_mtime: int | None
    base_revision: int
    expected_revision: int | None = None


class Dataset:
    def __init__(self, yaml_path: Path, category: Category, background_probe: bool = False, allow_empty: bool = False):
        self.yaml_path = yaml_path.resolve()
        self.category = category
        self.root = self.yaml_path.parent
        # managed projects start with no images and no classes; external datasets must have both
        self.allow_empty = allow_empty
        self.background_probe = background_probe
        self.names: dict[int, str] = {}
        self.names_style = "dict"  # how data.yaml spells names ("list" or "dict"); preserved on rewrite
        self.images: dict[str, ImageRecord] = {}
        self.orphan_labels: list[Path] = []
        self._label_scan_roots: set[Path] = set()
        # image folders per split, in YAML order; a split may list several (train: [images/a, images/b])
        self._split_folders: dict[str, list[Path]] = {}
        self._split_sources: dict[str, object] = {}  # raw YAML value per split (folders, .txt lists, files)
        self._class_ids_by_name: dict[str, int] = {}
        self._mtimes: dict[Path, int | None] = {}
        self._revisions: dict[Path, int] = {}
        self._lock = threading.RLock()
        self._change_listeners: list = []
        self._schema_listeners: list = []
        self.annotation_generation = 0
        self.schema_generation = 0
        self._sorted_image_ids: list[str] = []
        self._split_image_ids: dict[str, set[str]] = {}
        self._class_image_ids: dict[int, set[str]] = {}
        self._object_index: dict[int, dict[str, dict]] = {}
        self._object_keys_by_image: dict[str, list[tuple[int, str]]] = {}
        self._issues_by_image: dict[str, list[dict]] = {}
        self._issue_index: dict[str, tuple[ImageRecord, dict]] = {}
        self._issue_count = 0
        self._orphan_issues: list[dict] = []
        self._statistics_cache: dict | None = None
        # each history entry is a transaction: one or more files written or moved together
        self._undo: list[list] = []
        self._redo: list[list] = []
        self.session_id = time.strftime("%Y%m%d-%H%M%S")
        self._backed_up: set[Path] = set()
        self.indexing: dict[str, int] = {"done": 0, "total": 0}
        self._load()
        pending = [record for record in self.images.values() if record.width == 0 and record.probe_error is None]
        self.indexing = {"done": len(self.images) - len(pending), "total": len(self.images)}
        self._start_probe(pending)

    def _start_probe(self, pending: list[ImageRecord]) -> None:
        if not pending:
            return
        if self.background_probe:
            threading.Thread(target=self._probe_all, args=(pending,), daemon=True).start()
        else:
            self._probe_all(pending)

    def _read_config(self) -> dict:
        if not self.yaml_path.is_file():
            return {}
        try:
            config = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise DatasetError(f"Cannot read dataset YAML: {exc}") from exc
        if not isinstance(config, dict):
            raise DatasetError("Dataset YAML must be a mapping")
        return config

    def _names_from_config(self, config: dict) -> dict[int, str]:
        raw_names = config.get("names")
        if isinstance(raw_names, list):
            self.names_style = "list"
            return dict(enumerate(str(value) for value in raw_names))
        if isinstance(raw_names, dict):
            self.names_style = "dict"
            try:
                return {int(key): str(value) for key, value in raw_names.items()}
            except (TypeError, ValueError) as exc:
                raise DatasetError("Dataset names must use integer class IDs") from exc
        return {}

    def _classification_names(self) -> dict[int, str]:
        class_names = sorted({child.name for directory in self._all_folders() if directory.is_dir() for child in directory.iterdir() if child.is_dir() and not child.name.startswith(".")})
        return dict(enumerate(class_names))

    def _all_folders(self) -> list[Path]:
        return [folder for folders in self._split_folders.values() for folder in folders]

    def _resolve_path(self, value) -> Path:
        path = Path(str(value))
        return (self.root / path).resolve() if not path.is_absolute() else path.resolve()

    def _read_layout(self) -> dict:
        """Read the YAML: names, dataset root, and the image folders of every split."""
        if self.category == "classification":
            self.root = (self.yaml_path if self.yaml_path.is_dir() else self.yaml_path.parent).resolve()
        config = self._read_config()
        if self.category == "classification":
            if config:
                self._names_from_config(config)  # names come from folders; this only records the YAML's style
        else:
            self.names = self._names_from_config(config)
        if config or self.category != "classification":
            root_value = Path(str(config.get("path", ".")))
            self.root = (self.yaml_path.parent / root_value).resolve() if not root_value.is_absolute() else root_value.resolve()
        self._split_folders = {}
        self._split_sources = {}
        for split in ("train", "val", "test"):
            value = config.get(split)
            if self.category == "classification" and not value:
                value = split  # <root>/<split>/ is the convention without a YAML entry
            values = value if isinstance(value, list) else [value] if value else []
            folders = []
            for item in values:
                path = self._resolve_path(item)
                if self.category == "classification":
                    if path.is_dir():
                        folders.append(path)
                elif path.suffix.lower() != ".txt" and path.suffix.lower() not in IMAGE_EXTENSIONS:
                    folders.append(path)
            if folders:
                self._split_folders[split] = folders
            if value:
                self._split_sources[split] = value
        if self.category == "classification":
            self.names = self._classification_names()
        self._class_ids_by_name = {name: class_id for class_id, name in self.names.items()}
        return config

    def _discover(self) -> list[tuple[str, Path]]:
        """Read the layout and list every (split, image path) on disk."""
        self._read_layout()
        if self.category == "classification":
            if not self._split_folders and not self.allow_empty:
                raise DatasetError("Classification datasets need train/, val/, or test/ directories with one folder per class")
            if not self.names and not self.allow_empty:
                raise DatasetError("No class directories found under the dataset splits")
            found = []
            for split, folders in self._split_folders.items():
                for folder in folders:
                    for class_dir in sorted(child for child in folder.iterdir() if child.is_dir() and not child.name.startswith(".")):
                        found.extend((split, path) for path in sorted(class_dir.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS and WORKBENCH_DIR not in path.relative_to(class_dir).parts)
            return found
        if not self.names and not self.allow_empty:
            raise DatasetError("Dataset YAML has no class names")
        self._label_scan_roots = set()
        found = []
        for split in ("train", "val", "test"):
            found.extend((split, image_path) for image_path in self._resolve_sources(self._split_sources.get(split)))
        return found

    def folder_label(self, folder: Path) -> str:
        """How a folder is shown and addressed in the API: relative to the dataset root when inside it."""
        return folder.relative_to(self.root).as_posix() if folder.is_relative_to(self.root) else str(folder)

    def _folder_of(self, record: ImageRecord) -> Path | None:
        for folder in self._split_folders.get(record.split, []):
            if record.path.is_relative_to(folder):
                return folder
        return None

    def resolve_folder(self, split: str, folder: str | None = None) -> Path:
        """The split folder named by `folder` (as folder_label shows it), or the split's first folder."""
        folders = self._split_folders.get(split)
        if not folders:
            raise DatasetError(f"Split {split} has no image folder in this dataset")
        if not folder:
            return folders[0]
        for candidate in folders:
            if folder in (self.folder_label(candidate), str(candidate)):
                return candidate
        raise DatasetError(f"{folder} is not a folder of the {split} split")

    def folders(self) -> dict:
        """Every split folder with its image count (for the Dataset tab)."""
        with self._lock:
            counts: dict[Path, int] = {}
            for record in self.images.values():
                folder = self._folder_of(record)
                if folder is not None:
                    counts[folder] = counts.get(folder, 0) + 1
            items = [
                {"split": split, "path": self.folder_label(folder), "images": counts.get(folder, 0), "exists": folder.is_dir()}
                for split, folders in self._split_folders.items()
                for folder in folders
            ]
            return {"items": items, "editable": self.yaml_path.is_file()}

    def add_folder(self, split: str, path: str) -> dict:
        """Create a new image folder and list it under `split` in the YAML (one undoable step)."""
        from .yamledit import write_key

        if split not in ("train", "val", "test"):
            raise DatasetError("Split must be train, val, or test")
        if not self.yaml_path.is_file():
            raise DatasetError("This dataset has no YAML file to record the new folder in")
        raw = str(path).strip().replace("\\", "/").strip("/")
        parts = [part for part in raw.split("/") if part not in ("", ".")]
        if not parts or any(part == ".." or part.startswith(".") for part in parts):
            raise DatasetError("Use a folder path inside the dataset, without '..' or hidden folders")
        with self._lock:
            folder = self.root.joinpath(*parts).resolve()
            if not folder.is_relative_to(self.root) or folder == self.root:
                raise DatasetError("The folder must be inside the dataset root")
            for existing in self._all_folders():
                if folder == existing or folder.is_relative_to(existing) or existing.is_relative_to(folder):
                    raise DatasetError(f"{self.folder_label(folder)} overlaps the existing split folder {self.folder_label(existing)}")
            if folder.exists() and any(folder.iterdir()):
                raise DatasetError(f"{self.folder_label(folder)} already exists and is not empty; add it to data.yaml by hand or rescan")
            entries: list = []
            created = []
            for directory in [folder] + ([] if self.category == "classification" else [self._label_root_for(folder)]):
                if directory != folder and directory.exists():
                    continue
                missing = []
                current = directory
                while not current.exists() and current != self.root:
                    missing.append(current)
                    current = current.parent
                for item in reversed(missing):
                    if item not in created:
                        entries.append(DirEntry(item, True))
                        created.append(item)
            config = self._read_config()
            current_value = config.get(split)
            listed = current_value if isinstance(current_value, list) else [current_value] if current_value else []
            if self.category == "classification" and not listed and split in self._split_folders:
                listed = [self.folder_label(self._split_folders[split][0])]
            new_value = listed + ["/".join(parts)]
            before = self.yaml_path.read_text(encoding="utf-8")
            entries.append(FileEntry(self.yaml_path, before, write_key(before, split, new_value[0] if len(new_value) == 1 else new_value), True, True))
            entries.append(SchemaEntry({class_id: class_id for class_id in self.names}, layout_only=True))
            self._execute(entries)
            return {"split": split, "path": self.folder_label(folder), **self.folders(), **self.history_state()}

    def _make_record(self, split: str, image_path: Path, record_id: str | None = None) -> ImageRecord:
        record_id = record_id or hashlib.sha1(f"{split}:{image_path}".encode()).hexdigest()[:16]
        label_path = image_path if self.category == "classification" else self._label_for(image_path)
        record = ImageRecord(record_id, split, image_path, label_path)
        record.name_cf = image_path.name.casefold()
        return record

    def _load(self) -> None:
        size_index = self._load_size_index()
        for split, image_path in self._discover():
            record = self._make_record(split, image_path)
            self._apply_cached_size(record, size_index)
            self._read_record(record)
            self.images[record.id] = record
        if not self.images and not self.allow_empty:
            raise DatasetError("No supported images found in dataset splits")
        self._find_orphans()
        self._rebuild_indexes()

    def _resort(self) -> None:
        self._sorted_image_ids = [record.id for record in sorted(self.images.values(), key=lambda item: (item.split, item.name_cf))]

    def _add_record(self, record: ImageRecord, read: bool = True) -> None:
        """Insert a record discovered (or restored) after load; caller re-sorts once per batch."""
        self.images[record.id] = record
        self._split_image_ids.setdefault(record.split, set()).add(record.id)
        if read:
            self._read_record(record)

    def _remove_record(self, record: ImageRecord) -> None:
        record.annotations = []
        record.issues = []
        self._refresh_record_indexes(record)
        self.images.pop(record.id, None)
        self._object_keys_by_image.pop(record.id, None)
        self._issues_by_image.pop(record.id, None)
        image_ids = self._split_image_ids.get(record.split)
        if image_ids is not None:
            image_ids.discard(record.id)
            if not image_ids:
                del self._split_image_ids[record.split]
        self._mtimes.pop(record.label_path, None)

    def _set_split(self, record: ImageRecord, split: str) -> None:
        if record.split == split:
            return
        image_ids = self._split_image_ids.get(record.split)
        if image_ids is not None:
            image_ids.discard(record.id)
            if not image_ids:
                del self._split_image_ids[record.split]
        record.split = split
        self._split_image_ids.setdefault(split, set()).add(record.id)

    def rescan(self) -> dict:
        """Pick up images and labels added, removed, or edited on disk since load."""
        with self._lock:
            before_names = dict(self.names)
            discovered = self._discover()
            by_key = {(record.split, record.path): record for record in self.images.values()}
            wanted = set(discovered)
            removed = [record for key, record in by_key.items() if key not in wanted]
            for record in removed:
                self._remove_record(record)
            size_index = self._load_size_index()
            added = []
            for split, image_path in discovered:
                if (split, image_path) in by_key:
                    continue
                record = self._make_record(split, image_path)
                if record.id in self.images:
                    continue
                self._apply_cached_size(record, size_index)
                self._add_record(record)
                added.append(record)
            self._class_ids_by_name = {name: class_id for class_id, name in self.names.items()}
            names_changed = before_names != self.names
            updated = 0
            added_ids = {record.id for record in added}
            for record in self.images.values():
                if record.id in added_ids:
                    continue
                current = self._safe_stat(record.label_path)
                if names_changed or (current.st_mtime_ns if current else None) != self._mtimes.get(record.label_path):
                    self._read_record(record)
                    updated += 1
            self._resort()
            self._find_orphans()
            self._rebuild_orphan_issues()
            self._statistics_cache = None
            pending = [record for record in added if record.width == 0 and record.probe_error is None]
            self.indexing = {"done": self.indexing["done"] + len(added) - len(pending), "total": self.indexing["total"] + len(added) - len(removed)}
            self.indexing["done"] = min(self.indexing["done"], self.indexing["total"])
            changed = bool(added or removed or updated or names_changed)
            if changed:
                self.annotation_generation += 1
            if names_changed:
                self.schema_generation += 1
        self._start_probe(pending)
        if changed:
            self._notify_changes([record.id for record in added])
        if names_changed:
            self._notify_schema(None)
        return {"added": len(added), "removed": len(removed), "updated": updated, "names_changed": names_changed}

    def _class_name_for(self, record: ImageRecord) -> str:
        folder = self._folder_of(record)
        return record.path.relative_to(folder).parts[0] if folder else record.path.parent.name

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
            self._refresh_record_indexes(record)
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
                self._label_scan_roots.add(self._label_root_for(source))
                found.extend(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS and WORKBENCH_DIR not in path.relative_to(source).parts)
            elif source.suffix.lower() == ".txt" and source.exists():
                for line in source.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        path = Path(line.strip())
                        resolved = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
                        if not resolved.exists():
                            resolved = self._relocate_listed_image(path, resolved)
                        found.append(resolved)
            elif source.suffix.lower() in IMAGE_EXTENSIONS and source.exists():
                found.append(source)
        return sorted(set(found))

    def _relocate_listed_image(self, path: Path, resolved: Path) -> Path:
        # File-list entries exported from another machine or layout often carry a stale
        # prefix (e.g. "data/images/train/x.png" when the file is at "<root>/images/train/x.png").
        # Retry with leading components stripped until the entry matches under the root.
        parts = path.parts[1:] if path.is_absolute() else path.parts
        for start in range(1, len(parts)):
            candidate = self.root.joinpath(*parts[start:])
            if candidate.exists():
                return candidate.resolve()
        return resolved

    def _label_root_for(self, image_root: Path) -> Path:
        parts = list(image_root.parts)
        if "images" in parts:
            index = len(parts) - 1 - parts[::-1].index("images")
            candidate = Path(*parts[:index], "labels", *parts[index + 1 :])
            if candidate.exists() or (self.root / "labels").exists():
                return candidate
        return image_root

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
        if self.category == "classification":
            record.issues.clear()
            if record.probe_error:
                record.issues.append(self._issue("unreadable_image", "error", record.probe_error, record))
            class_id = self._class_ids_by_name[self._class_name_for(record)]
            record.annotations = [Annotation(f"{record.id}:1", class_id, [])]
            record.class_ids = {class_id}
            self._mtimes[record.label_path] = record.path.stat().st_mtime_ns if record.path.exists() else None
            self._revisions[record.label_path] = self._revisions.get(record.label_path, 0) + 1
            self._refresh_record_indexes(record)
            return
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
                if not expected or not all(math.isfinite(value) for value in points):
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
        self._revisions[record.label_path] = self._revisions.get(record.label_path, 0) + 1
        self._refresh_record_indexes(record)

    def _zero_area(self, annotation: Annotation) -> bool:
        if self.category == "detection":
            return annotation.points[2] <= 0 or annotation.points[3] <= 0
        points = list(zip(annotation.points[::2], annotation.points[1::2]))
        area = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))
        return abs(area) < 1e-10

    def _issue(self, kind, severity, message, record, annotation_id=None, fixable=False):
        return {"id": hashlib.sha1(f"{record.id}:{kind}:{annotation_id}:{message}".encode()).hexdigest()[:16], "kind": kind, "severity": severity, "message": message, "image_id": record.id, "image_name": record.path.name, "split": record.split, "annotation_id": annotation_id, "fixable": fixable}

    def _find_orphans(self) -> None:
        if self.category == "classification":
            self.orphan_labels = []
            return
        expected = {record.label_path.resolve() for record in self.images.values()}
        candidates = set()
        roots = self._label_scan_roots | {record.label_path.parent for record in self.images.values()}
        for root in roots:
            if root.exists():
                candidates.update(path.resolve() for path in root.rglob("*.txt") if WORKBENCH_DIR not in path.parts)
        self.orphan_labels = sorted(candidates - expected)

    def _rebuild_indexes(self) -> None:
        """Build immutable ordering/split indexes and mutable annotation indexes once."""
        self._sorted_image_ids = [
            record.id for record in sorted(self.images.values(), key=lambda item: (item.split, item.name_cf))
        ]
        self._split_image_ids = {}
        self._class_image_ids = {}
        self._object_index = {}
        self._object_keys_by_image = {}
        self._issues_by_image = {}
        self._issue_index = {}
        self._issue_count = 0
        for record in self.images.values():
            self._split_image_ids.setdefault(record.split, set()).add(record.id)
            self._refresh_record_indexes(record)
        self._rebuild_orphan_issues()

    def _rebuild_orphan_issues(self) -> None:
        self._orphan_issues = [
            {
                "id": hashlib.sha1(str(path).encode()).hexdigest()[:16],
                "kind": "orphan_label",
                "severity": "warning",
                "message": "Label has no matching dataset image",
                "image_id": None,
                "image_name": path.name,
                "split": None,
                "annotation_id": None,
                "fixable": False,
            }
            for path in self.orphan_labels
        ]

    def _refresh_record_indexes(self, record: ImageRecord) -> None:
        """Refresh only the indexes affected by one record mutation."""
        self._statistics_cache = None
        if record.id not in self.images:
            return
        old_keys = self._object_keys_by_image.pop(record.id, [])
        for class_id in {class_id for class_id, _ in old_keys}:
            image_ids = self._class_image_ids[class_id]
            image_ids.discard(record.id)
            if not image_ids:
                del self._class_image_ids[class_id]
        for class_id, annotation_id in old_keys:
            objects = self._object_index.get(class_id)
            if objects is not None:
                objects.pop(annotation_id, None)
                if not objects:
                    del self._object_index[class_id]

        keys = []
        for annotation in record.annotations:
            self._class_image_ids.setdefault(annotation.class_id, set()).add(record.id)
            item = {
                "id": annotation.id,
                "image_id": record.id,
                "image_name": record.path.name,
                "split": record.split,
                "class_id": annotation.class_id,
            }
            self._object_index.setdefault(annotation.class_id, {})[annotation.id] = item
            keys.append((annotation.class_id, annotation.id))
        self._object_keys_by_image[record.id] = keys
        old_issues = self._issues_by_image.get(record.id, [])
        self._issue_count -= len(old_issues)
        for issue in old_issues:
            self._issue_index.pop(issue["id"], None)
        self._issues_by_image[record.id] = list(record.issues)
        for issue in record.issues:
            self._issue_index[issue["id"]] = (record, issue)
        self._issue_count += len(record.issues)

    def add_change_listener(self, listener) -> None:
        self._change_listeners.append(listener)

    def add_schema_listener(self, listener) -> None:
        """listener(mapping) after class names change; mapping is old id -> new id (None = removed), or None if unknown."""
        self._schema_listeners.append(listener)

    def _notify_schema(self, mapping: dict[int, int | None] | None) -> None:
        for listener in tuple(self._schema_listeners):
            try:
                listener(mapping)
            except Exception:
                pass  # listeners hold derived state only; a committed schema change must stand

    def _notify_changes(self, image_ids: list[str]) -> None:
        for listener in tuple(self._change_listeners):
            try:
                listener(image_ids, self.annotation_generation)
            except Exception:
                pass  # cache invalidation must never make a committed label write fail

    def metadata(self) -> dict:
        with self._lock:
            return {
                "yaml": str(self.yaml_path),
                "root": str(self.root),
                "category": self.category,
                "names": self.names,
                "image_count": len(self.images),
                "split_counts": {split: len(image_ids) for split, image_ids in self._split_image_ids.items()},
                "issue_count": self._issue_count + len(self._orphan_issues),
                "session_id": self.session_id,
                "indexing": dict(self.indexing),
                "schema_generation": self.schema_generation,
                "capabilities": self.capabilities(),
                "can_undo": bool(self._undo),
                "can_redo": bool(self._redo),
            }

    def capabilities(self) -> dict:
        """What the UI may offer: image moves/imports need each split to be a single image directory."""
        return {
            "classes": self.category == "classification" or self.yaml_path.is_file(),
            "image_ops": bool(self._split_folders),
            "splits": [split for split in ("train", "val", "test") if split in self._split_folders],
            "folders": {split: [self.folder_label(folder) for folder in folders] for split, folders in self._split_folders.items()},
            "folders_editable": self.yaml_path.is_file(),
        }

    def statistics(self) -> dict:
        """Return cached dataset-wide annotation statistics.

        Geometry is normalized to the image and segmentation objects use their
        enclosing box, so the same distributions work for both dataset types.
        """
        with self._lock:
            if self._statistics_cache is None:
                self._statistics_cache = self._build_statistics()
            return self._statistics_cache

    def _build_statistics(self) -> dict:
        records = list(self.images.values())
        annotation_counts = [len(record.annotations) for record in records]
        annotation_total = sum(annotation_counts)
        annotated_images = sum(count > 0 for count in annotation_counts)
        class_ids = sorted(set(self.names) | {annotation.class_id for record in records for annotation in record.annotations})
        class_annotations = {class_id: 0 for class_id in class_ids}
        class_images = {class_id: 0 for class_id in class_ids}
        split_rows = {
            split: {
                "split": split,
                "images": len(image_ids),
                "annotated_images": 0,
                "annotations": 0,
                "class_annotations": {str(class_id): 0 for class_id in class_ids},
            }
            for split, image_ids in sorted(self._split_image_ids.items())
        }
        cooccurrence = [[0 for _ in class_ids] for _ in class_ids]
        class_positions = {class_id: index for index, class_id in enumerate(class_ids)}
        geometry = []

        for record in records:
            split = split_rows[record.split]
            split["annotations"] += len(record.annotations)
            split["annotated_images"] += bool(record.annotations)
            present = sorted({annotation.class_id for annotation in record.annotations})
            for class_id in present:
                class_images[class_id] += 1
                position = class_positions[class_id]
                cooccurrence[position][position] += 1
            for left, class_id in enumerate(present):
                for other_id in present[left + 1 :]:
                    a, b = class_positions[class_id], class_positions[other_id]
                    cooccurrence[a][b] += 1
                    cooccurrence[b][a] += 1
            for annotation in record.annotations:
                class_annotations[annotation.class_id] += 1
                split["class_annotations"][str(annotation.class_id)] += 1
                width, height = self._annotation_size(annotation)
                area = width * height if width > 0 and height > 0 else 0
                aspect = width / height if width > 0 and height > 0 else None
                geometry.append({
                    "record": record,
                    "annotation": annotation,
                    "area": area,
                    "aspect": aspect,
                })

        for row in split_rows.values():
            row["average_annotations"] = round(row["annotations"] / row["images"], 3) if row["images"] else 0

        area_values = [item["area"] for item in geometry]
        aspect_values = [item["aspect"] for item in geometry if item["aspect"] is not None]
        density_bins = [
            {"label": "0", "count": sum(value == 0 for value in annotation_counts)},
            {"label": "1", "count": sum(value == 1 for value in annotation_counts)},
            {"label": "2", "count": sum(value == 2 for value in annotation_counts)},
            {"label": "3–4", "count": sum(3 <= value <= 4 for value in annotation_counts)},
            {"label": "5–9", "count": sum(5 <= value <= 9 for value in annotation_counts)},
            {"label": "10–19", "count": sum(10 <= value <= 19 for value in annotation_counts)},
            {"label": "20+", "count": sum(value >= 20 for value in annotation_counts)},
        ]
        size_bins = [
            {"label": "≤0", "count": sum(value <= 0 for value in area_values)},
            {"label": "<1%", "count": sum(0 < value < 0.01 for value in area_values)},
            {"label": "1–5%", "count": sum(0.01 <= value < 0.05 for value in area_values)},
            {"label": "5–20%", "count": sum(0.05 <= value < 0.2 for value in area_values)},
            {"label": "≥20%", "count": sum(value >= 0.2 for value in area_values)},
        ]
        aspect_bins = [
            {"label": "<0.5", "count": sum(value < 0.5 for value in aspect_values)},
            {"label": "0.5–0.8", "count": sum(0.5 <= value < 0.8 for value in aspect_values)},
            {"label": "0.8–1.25", "count": sum(0.8 <= value < 1.25 for value in aspect_values)},
            {"label": "1.25–2", "count": sum(1.25 <= value < 2 for value in aspect_values)},
            {"label": "≥2", "count": sum(value >= 2 for value in aspect_values)},
        ]

        return {
            "source_generation": self.annotation_generation,
            "summary": {
                "images": len(records),
                "annotated_images": annotated_images,
                "unlabeled_images": len(records) - annotated_images,
                "annotations": annotation_total,
                "average_annotations": round(annotation_total / len(records), 3) if records else 0,
            },
            "class_balance": [
                {
                    "class_id": class_id,
                    "name": self.names.get(class_id, f"Unknown class {class_id}"),
                    "annotations": class_annotations[class_id],
                    "images": class_images[class_id],
                    "percent": round(class_annotations[class_id] * 100 / annotation_total, 2) if annotation_total else 0,
                }
                for class_id in class_ids
            ],
            "annotations_per_image": {
                "bins": density_bins,
                **self._distribution_summary(annotation_counts),
            },
            "box_size": {"bins": size_bins, **self._distribution_summary(area_values)},
            "aspect_ratio": {"bins": aspect_bins, **self._distribution_summary(aspect_values)},
            "split_comparison": list(split_rows.values()),
            "cooccurrence": {
                "class_ids": class_ids,
                "names": [self.names.get(class_id, f"Unknown {class_id}") for class_id in class_ids],
                "matrix": cooccurrence,
            },
            "outliers": self._statistics_outliers(records, annotation_counts, geometry),
        }

    def _annotation_size(self, annotation: Annotation) -> tuple[float, float]:
        if self.category == "classification":
            return 1.0, 1.0
        if self.category == "detection":
            return annotation.points[2], annotation.points[3]
        xs, ys = annotation.points[::2], annotation.points[1::2]
        return max(xs) - min(xs), max(ys) - min(ys)

    @staticmethod
    def _percentile(values: list[float] | list[int], percentile: float) -> float:
        if not values:
            return 0
        ordered = sorted(values)
        position = (len(ordered) - 1) * percentile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    @classmethod
    def _distribution_summary(cls, values: list[float] | list[int]) -> dict:
        return {
            "count": len(values),
            "min": min(values) if values else 0,
            "q1": cls._percentile(values, 0.25),
            "median": cls._percentile(values, 0.5),
            "q3": cls._percentile(values, 0.75),
            "p95": cls._percentile(values, 0.95),
            "max": max(values) if values else 0,
        }

    @classmethod
    def _iqr_fences(cls, values: list[float], multiplier: float = 3) -> tuple[float, float] | None:
        if len(values) < 8:
            return None
        q1, q3 = cls._percentile(values, 0.25), cls._percentile(values, 0.75)
        spread = q3 - q1
        if spread <= 1e-12:
            return (q1, q3) if min(values) < q1 or max(values) > q3 else None
        return q1 - multiplier * spread, q3 + multiplier * spread

    def _statistics_outliers(self, records: list[ImageRecord], counts: list[int], geometry: list[dict]) -> list[dict]:
        outliers = []
        count_fence = self._iqr_fences(counts)
        if count_fence:
            for record, count in zip(records, counts):
                if count > count_fence[1]:
                    outliers.append({
                        "image_id": record.id,
                        "image_name": record.path.name,
                        "split": record.split,
                        "annotation_id": None,
                        "kind": "annotation_count",
                        "reason": f"Unusually many annotations ({count})",
                        "score": count / max(count_fence[1], 1),
                    })

        by_class: dict[int, list[dict]] = {}
        for item in geometry:
            by_class.setdefault(item["annotation"].class_id, []).append(item)
        for class_items in by_class.values():
            positive_areas = [math.log(item["area"]) for item in class_items if item["area"] > 0]
            positive_aspects = [math.log(item["aspect"]) for item in class_items if item["aspect"] is not None]
            area_fence = self._iqr_fences(positive_areas)
            aspect_fence = self._iqr_fences(positive_aspects)
            for item in class_items:
                record, annotation = item["record"], item["annotation"]
                candidates = []
                if item["area"] <= 0:
                    candidates.append(("box_size", "Non-positive box size", 1000))
                elif area_fence:
                    value = math.log(item["area"])
                    if value < area_fence[0]:
                        candidates.append(("box_size", f"Unusually small box ({item['area'] * 100:.3g}% of image)", area_fence[0] - value))
                    elif value > area_fence[1]:
                        candidates.append(("box_size", f"Unusually large box ({item['area'] * 100:.3g}% of image)", value - area_fence[1]))
                if item["aspect"] is not None and aspect_fence:
                    value = math.log(item["aspect"])
                    if value < aspect_fence[0] or value > aspect_fence[1]:
                        candidates.append(("aspect_ratio", f"Unusual aspect ratio ({item['aspect']:.3g}:1)", min(abs(value - aspect_fence[0]), abs(value - aspect_fence[1]))))
                for kind, reason, score in candidates:
                    outliers.append({
                        "image_id": record.id,
                        "image_name": record.path.name,
                        "split": record.split,
                        "annotation_id": annotation.id,
                        "class_id": annotation.class_id,
                        "kind": kind,
                        "reason": reason,
                        "score": score,
                    })
        outliers.sort(key=lambda item: item["score"], reverse=True)
        for item in outliers:
            item.pop("score", None)
        return outliers[:100]

    def list_images(self, split="all", class_id=None, search="", offset=0, limit=100, prediction_ids: set[str] | None = None, min_labels: int | None = None, max_labels: int | None = None) -> dict:
        with self._lock:
            allowed: set[str] | None = None
            if split != "all":
                allowed = set(self._split_image_ids.get(split, set()))
            if class_id is not None:
                class_images = self._class_image_ids.get(class_id, set())
                allowed = set(class_images) if allowed is None else allowed & class_images
            if prediction_ids is not None:
                allowed = set(prediction_ids) if allowed is None else allowed & prediction_ids
            records = [self.images[image_id] for image_id in self._sorted_image_ids if allowed is None or image_id in allowed]
            if search:
                query = search.casefold()
                records = [record for record in records if query in record.name_cf]
            if min_labels is not None or max_labels is not None:
                # With a class filter, count only that class's labels ("images with >= 3 cars").
                def label_count(record: ImageRecord) -> int:
                    if class_id is None:
                        return len(record.annotations)
                    return sum(1 for a in record.annotations if a.class_id == class_id)
                low = min_labels if min_labels is not None else 0
                high = max_labels if max_labels is not None else math.inf
                records = [record for record in records if low <= label_count(record) <= high]
            return {"total": len(records), "items": [record.summary() for record in records[offset : offset + limit]]}

    def list_objects(self, class_id: int, split="all", offset=0, limit=100) -> dict:
        with self._lock:
            items = list(self._object_index.get(class_id, {}).values())
            if split != "all":
                items = [item for item in items if item["split"] == split]
            items.sort(key=lambda item: (item["split"], item["image_name"].casefold(), item["id"]))
            return {"total": len(items), "items": items[offset : offset + limit]}

    def detail(self, image_id: str) -> dict:
        with self._lock:
            record = self.require_image(image_id)
            return {
                **record.summary(),
                "path": str(record.path),
                "revision": self._revisions[record.label_path],
                "annotations": [a.as_dict() for a in record.annotations],
                "issues": list(record.issues),
            }

    def require_image(self, image_id: str) -> ImageRecord:
        try:
            return self.images[image_id]
        except KeyError as exc:
            raise KeyError("Image not found") from exc

    def replace_annotations(self, image_id: str, raw_annotations: list[dict], expected_revision: int | None = None) -> dict:
        record = self.require_image(image_id)
        if self.category == "classification":
            if len(raw_annotations) != 1 or raw_annotations[0].get("points"):
                raise DatasetError("Classification images have exactly one class and no geometry")
            class_id = int(raw_annotations[0]["class_id"])
            if class_id not in self.names:
                raise DatasetError(f"Unknown class {class_id}")
            self._commit_moves([(record, class_id)], expected_revision=expected_revision)
            return self.detail(image_id)
        self._commit_many([self._prepare(record, raw_annotations, expected_revision)])
        return self.detail(image_id)

    def _commit_moves(self, moves: list[tuple[ImageRecord, int]], history: bool = True, expected_revision: int | None = None) -> None:
        """Reassign classification classes by moving files; all targets are checked before any move."""
        with self._lock:
            planned = []
            for record, class_id in moves:
                src = record.path
                if not src.exists() or src.stat().st_mtime_ns != self._mtimes.get(record.label_path):
                    self._mtimes[record.label_path] = src.stat().st_mtime_ns if src.exists() else None
                    raise WriteConflict(f"{src.name} changed outside this application; reload before saving")
                if expected_revision is not None and expected_revision != self._revisions[record.label_path]:
                    raise WriteConflict(f"{src.name} has a newer revision; reload before saving")
                if self.names[class_id] == self._class_name_for(record):
                    continue
                dst = (self._folder_of(record) or src.parent.parent) / self.names[class_id] / src.name
                if dst.exists():
                    raise DatasetError(f"Cannot move {src.name}: a file with that name already exists in {self.names[class_id]}/")
                planned.append((record, src, dst))
            entries = [MoveEntry(record.id, src, dst) for record, src, dst in planned]
            if not entries:
                return
            if history:
                self._execute(entries)
            else:
                self._run_transaction(entries, "redo")

    @staticmethod
    def _move_file(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)

    def _prepare(self, record: ImageRecord, raw_annotations: list[dict], expected_revision: int | None = None) -> PreparedChange:
        """Validate raw annotations and capture the label state this change is based on."""
        if any(issue["kind"] == "malformed_label" for issue in record.issues):
            raise DatasetError("Resolve or remove malformed label rows before editing this image")
        annotations = []
        for index, raw in enumerate(raw_annotations):
            class_id = int(raw["class_id"])
            points = [float(value) for value in raw["points"]]
            if class_id not in self.names:
                raise DatasetError(f"Unknown class {class_id}")
            valid_count = len(points) == 4 if self.category == "detection" else len(points) >= 6 and len(points) % 2 == 0
            if not valid_count or not all(math.isfinite(value) and 0 <= value <= 1 for value in points):
                raise DatasetError("Invalid annotation geometry")
            annotations.append(Annotation(raw.get("id") or f"{record.id}:new:{index}", class_id, points))
        content = "".join(f"{a.class_id} {' '.join(self._number(v) for v in a.points)}\n" for a in annotations)
        before_exists = record.label_path.exists()
        before = record.label_path.read_text(encoding="utf-8") if before_exists else ""
        base_mtime = record.label_path.stat().st_mtime_ns if before_exists else None
        return PreparedChange(
            record=record,
            before=before,
            after=content,
            before_exists=before_exists,
            after_exists=True,
            base_mtime=base_mtime,
            base_revision=self._revisions[record.label_path],
            expected_revision=expected_revision,
        )

    def _number(self, value: float) -> str:
        return f"{value:.6f}".rstrip("0").rstrip(".") or "0"

    def _commit_many(self, changes: list[PreparedChange], history=True) -> None:
        """Write all changes as one transaction: every mtime is checked before any file is touched."""
        with self._lock:
            for change in changes:
                record = change.record
                current_mtime = record.label_path.stat().st_mtime_ns if record.label_path.exists() else None
                if current_mtime != self._mtimes.get(record.label_path):
                    self._read_record(record)
                    raise WriteConflict(f"{record.label_path.name} changed outside this application; reload before saving")
                if current_mtime != change.base_mtime or self._revisions[record.label_path] != change.base_revision:
                    raise WriteConflict(f"{record.label_path.name} changed while this edit was pending; reload before saving")
                if change.expected_revision is not None and change.expected_revision != change.base_revision:
                    raise WriteConflict(f"{record.label_path.name} has a newer revision; reload before saving")
            transaction = []
            for change in changes:
                record = change.record
                self._backup(record.label_path, change.before)
                self._write_state(record.label_path, change.after, change.after_exists)
                transaction.append(HistoryEntry(
                    record.label_path,
                    change.before,
                    change.after,
                    change.before_exists,
                    change.after_exists,
                ))
                self._read_record(record)
            if history and transaction:
                self._undo.append(transaction)
                self._redo.clear()
            changed_ids = [change.record.id for change in changes]
            if changed_ids:
                self.annotation_generation += 1
        if changed_ids:
            self._notify_changes(changed_ids)

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

    def _write_state(self, path: Path, content: str, exists: bool) -> None:
        if exists:
            self._atomic_write(path, content)
        else:
            path.unlink(missing_ok=True)

    def history(self, direction: str) -> dict:
        with self._lock:
            source, target = (self._undo, self._redo) if direction == "undo" else (self._redo, self._undo)
            if not source:
                raise DatasetError(f"Nothing to {direction}")
            transaction = source[-1]
            image_ids, schema_changed = self._run_transaction(transaction, direction)
            source.pop()
            target.append(transaction)
            return {
                "image_ids": image_ids,
                "image_id": image_ids[0] if image_ids else None,
                "schema_changed": schema_changed,
                "can_undo": bool(self._undo),
                "can_redo": bool(self._redo),
            }

    def history_state(self) -> dict:
        with self._lock:
            return {"can_undo": bool(self._undo), "can_redo": bool(self._redo)}

    def _execute(self, entries: list) -> tuple[list[str], bool]:
        """Apply a new transaction forward and make it undoable as one unit."""
        with self._lock:
            result = self._run_transaction(entries, "redo")
            if entries:
                self._undo.append(entries)
                self._redo.clear()
            return result

    def _run_transaction(self, entries: list, direction: str) -> tuple[list[str], bool]:
        """Replay a transaction in either direction: every entry is checked before anything is touched.

        Records are re-read once at the end so that a class-folder rename or a names change
        inside the same transaction is in place before labels are interpreted.
        """
        with self._lock:
            ordered = list(reversed(entries)) if direction == "undo" else list(entries)
            records_by_label = {record.label_path: record for record in self.images.values()}
            for entry in ordered:
                self._check_entry(entry, direction, records_by_label)
            touched: dict[str, ImageRecord] = {}
            schema: SchemaEntry | None = None
            resort = False
            for entry in ordered:
                if isinstance(entry, SchemaEntry):
                    schema = entry
                else:
                    resort |= self._apply_entry(entry, direction, records_by_label, touched)
            if schema is not None:
                self._reload_schema()
            else:
                for record in touched.values():
                    if record.id in self.images:
                        self._read_record(record)
            if resort:
                self._resort()
                self._find_orphans()
                self._rebuild_orphan_issues()
            image_ids = list(touched)
            if image_ids or schema is not None:
                self.annotation_generation += 1
        if image_ids or schema is not None:
            self._notify_changes(image_ids)
        if schema is not None and not schema.layout_only:
            self._notify_schema(schema.mapping if direction == "redo" else schema.inverse())
        return image_ids, schema is not None

    @staticmethod
    def _states(entry, direction: str):
        """(expected current content/exists, target content/exists) for content entries."""
        if direction == "undo":
            return (entry.after, entry.after_exists), (entry.before, entry.before_exists)
        return (entry.before, entry.before_exists), (entry.after, entry.after_exists)

    def _check_entry(self, entry, direction: str, records_by_label: dict) -> None:
        if isinstance(entry, HistoryEntry):
            record = records_by_label.get(entry.label_path)
            if record is None:
                raise WriteConflict(f"{entry.label_path.name} is no longer part of the dataset; cannot {direction}")
            current = entry.label_path.stat().st_mtime_ns if entry.label_path.exists() else None
            if current != self._mtimes.get(entry.label_path):
                self._read_record(record)
                raise WriteConflict(f"{entry.label_path.name} changed outside this application; reload before saving")
        elif isinstance(entry, FileEntry):
            (expected, expected_exists), _ = self._states(entry, direction)
            exists = entry.path.exists()
            if exists != expected_exists or (exists and entry.path.read_text(encoding="utf-8") != expected):
                raise WriteConflict(f"{entry.path.name} changed outside this application; cannot {direction}")
        elif isinstance(entry, MoveEntry):
            src, dst = (entry.dst, entry.src) if direction == "undo" else (entry.src, entry.dst)
            record = self.images.get(entry.image_id)
            if record is None or record.path != src or not src.exists():
                raise WriteConflict(f"{src.name} changed outside this application; cannot {direction}")
            if dst.exists():
                raise WriteConflict(f"Cannot {direction}: {dst.name} already exists in {dst.parent.name}/")
        elif isinstance(entry, RecordMoveEntry):
            pairs = [(dst, src) for src, dst in entry.files] if direction == "undo" else entry.files
            present = entry.split_after if direction == "undo" else entry.split_before
            if present is not None:
                record = self.images.get(entry.image_id)
                if record is None or record.path != pairs[0][0]:
                    raise WriteConflict(f"{pairs[0][0].name} changed outside this application; cannot {direction}")
            elif entry.image_id in self.images:
                raise WriteConflict(f"{pairs[0][1].name} is already part of the dataset; cannot {direction}")
            for src, dst in pairs:
                if not src.exists():
                    raise WriteConflict(f"{src.name} changed outside this application; cannot {direction}")
                if dst.exists():
                    raise WriteConflict(f"Cannot {direction}: {dst.name} already exists in {dst.parent.name}/")
        elif isinstance(entry, DirMoveEntry):
            src, dst = (entry.dst, entry.src) if direction == "undo" else (entry.src, entry.dst)
            if not src.is_dir():
                raise WriteConflict(f"Folder {src.name}/ changed outside this application; cannot {direction}")
            if dst.exists():
                raise WriteConflict(f"Cannot {direction}: folder {dst.name}/ already exists")

    def _apply_entry(self, entry, direction: str, records_by_label: dict, touched: dict) -> bool:
        """Apply one checked entry; returns True when record membership or order changed."""
        if isinstance(entry, HistoryEntry):
            record = records_by_label[entry.label_path]
            (before, _), (after, after_exists) = self._states(entry, direction)
            self._backup(entry.label_path, before)
            self._write_state(entry.label_path, after, after_exists)
            touched[record.id] = record
            return False
        if isinstance(entry, FileEntry):
            (before, before_exists), (after, after_exists) = self._states(entry, direction)
            if before_exists:
                self._backup(entry.path, before)
            self._write_state(entry.path, after, after_exists)
            return False
        if isinstance(entry, MoveEntry):
            record = self.images[entry.image_id]
            dst = entry.src if direction == "undo" else entry.dst
            self._move_file(record.path, dst)
            self._relocate(record, dst, dst)
            touched[record.id] = record
            return True
        if isinstance(entry, RecordMoveEntry):
            pairs = [(dst, src) for src, dst in entry.files] if direction == "undo" else entry.files
            split_from, split_to = (entry.split_after, entry.split_before) if direction == "undo" else (entry.split_before, entry.split_after)
            for src, dst in pairs:
                self._move_file(src, dst)
            if split_to is None:
                self._remove_record(self.images[entry.image_id])
                return True
            image_path = pairs[0][1]
            if split_from is None:
                record = self._make_record(split_to, image_path, entry.image_id)
                self._add_record(record, read=False)
            else:
                record = self.images[entry.image_id]
                label_path = image_path if self.category == "classification" else self._label_for(image_path)
                self._relocate(record, image_path, label_path)
                self._set_split(record, split_to)
            touched[record.id] = record
            return True
        if isinstance(entry, DirEntry):
            create = entry.created != (direction == "undo")
            if create:
                entry.path.mkdir(parents=True, exist_ok=True)
            else:
                try:
                    entry.path.rmdir()
                except OSError:
                    pass  # a folder that gained files outside the app is left in place
            return False
        if isinstance(entry, DirMoveEntry):
            src, dst = (entry.dst, entry.src) if direction == "undo" else (entry.src, entry.dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
            for record in list(self.images.values()):
                if record.path.is_relative_to(src):
                    moved = dst / record.path.relative_to(src)
                    label = dst / record.label_path.relative_to(src) if record.label_path.is_relative_to(src) else record.label_path
                    self._relocate(record, moved, label)
                    touched[record.id] = record
            return True
        raise TypeError(f"Unknown history entry {entry!r}")

    def _relocate(self, record: ImageRecord, image_path: Path, label_path: Path) -> None:
        """Point a record at moved files, carrying its revision and mtime bookkeeping along."""
        old_label = record.label_path
        record.path = image_path
        record.label_path = label_path
        record.name_cf = image_path.name.casefold()
        if old_label != label_path:
            self._revisions[label_path] = self._revisions.pop(old_label, 0)
            self._mtimes.pop(old_label, None)
        stat = self._safe_stat(label_path)
        self._mtimes[label_path] = stat.st_mtime_ns if stat else None

    def _reload_schema(self) -> None:
        """Re-read class names and split folders from disk and re-interpret every record against them."""
        self._read_layout()
        self.schema_generation += 1
        for record in list(self.images.values()):
            self._read_record(record)
        self._statistics_cache = None

    def _trash_path(self, path: Path) -> Path:
        try:
            relative = path.resolve().relative_to(self.root)
        except ValueError:
            relative = Path(path.name)
        target = self.root / WORKBENCH_DIR / "trash" / self.session_id / relative
        counter = 1
        while target.exists():
            target = target.with_name(f"{path.stem}.{counter}{path.suffix}")
            counter += 1
        return target

    def delete_images(self, image_ids: list[str]) -> dict:
        """Move images (and their labels) into the dataset trash as one undoable transaction."""
        with self._lock:
            entries = []
            for image_id in dict.fromkeys(image_ids):
                record = self.require_image(image_id)
                files = [(record.path, self._trash_path(record.path))]
                if record.label_path != record.path and record.label_path.exists():
                    files.append((record.label_path, self._trash_path(record.label_path)))
                entries.append(RecordMoveEntry(record.id, record.split, None, files))
            if not entries:
                raise DatasetError("No images selected")
            self._execute(entries)
            return {"deleted": len(entries), **self.history_state()}

    def move_images(self, image_ids: list[str], split: str, folder: str | None = None) -> dict:
        """Move images (and their labels) to a split folder as one undoable transaction."""
        with self._lock:
            target_root = self.resolve_folder(split, folder)
            entries, skipped, targets = [], 0, set()
            for image_id in dict.fromkeys(image_ids):
                record = self.require_image(image_id)
                source_root = self._folder_of(record)
                if record.split == split and source_root == target_root:
                    skipped += 1
                    continue
                relative = record.path.relative_to(source_root) if source_root else Path(record.path.name)
                image_path = target_root / relative
                where = f"{split} ({self.folder_label(target_root)})"
                if image_path.exists() or image_path in targets:
                    raise DatasetError(f"Cannot move {record.path.name}: {where} already has a file with that name")
                targets.add(image_path)
                files = [(record.path, image_path)]
                if record.label_path != record.path and record.label_path.exists():
                    label_path = self._label_for(image_path)
                    if label_path.exists():
                        raise DatasetError(f"Cannot move {record.path.name}: {label_path.name} already exists in {where}")
                    files.append((record.label_path, label_path))
                entries.append(RecordMoveEntry(record.id, record.split, split, files))
            if entries:
                self._execute(entries)
            return {"moved": len(entries), "skipped": skipped, **self.history_state()}

    def issues(self) -> list[dict]:
        with self._lock:
            return [issue for _, issue in self._issue_index.values()] + list(self._orphan_issues)

    def fix_issue(self, issue_id: str) -> dict:
        indexed = self._issue_index.get(issue_id)
        if not indexed or not indexed[1]["fixable"]:
            raise DatasetError("Issue not found or cannot be fixed automatically")
        record, issue = indexed
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

    def fix_issues_bulk(self, kind: str, split: str | None = None, issue_ids: list[str] | None = None) -> dict:
        """Fix every matching fixable issue in one transaction; skips images that fail validation."""
        wanted = set(issue_ids) if issue_ids else None
        changes, skipped, fixed = [], [], 0
        grouped: dict[str, tuple[ImageRecord, list[dict]]] = {}
        for record, issue in self._issue_index.values():
            if issue["fixable"] and issue["kind"] == kind and (split is None or record.split == split) and (wanted is None or issue["id"] in wanted):
                grouped.setdefault(record.id, (record, []))[1].append(issue)
        for record, issues in grouped.values():
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
        if self.category == "classification":
            return self._bulk_reclassify(operations)
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

    def _bulk_reclassify(self, operations: list[dict]) -> dict:
        moves, skipped, seen = [], [], set()
        for operation in operations:
            record = self.images.get(operation["image_id"])
            if not record:
                skipped.append({"image_id": operation["image_id"], "annotation_id": None, "reason": "Image not found"})
            elif operation["action"] == "delete":
                skipped.append({"image_id": record.id, "annotation_id": None, "reason": "Deleting images is not supported for classification datasets"})
            elif operation.get("class_id") is None or int(operation["class_id"]) not in self.names:
                skipped.append({"image_id": record.id, "annotation_id": None, "reason": "Relabel requires a known class_id"})
            elif record.id not in seen:
                seen.add(record.id)
                moves.append((record, int(operation["class_id"])))
        if moves:
            self._commit_moves(moves)
        return {"applied": len(moves), "files": len(moves), "skipped": skipped}
