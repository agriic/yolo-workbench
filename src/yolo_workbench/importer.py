"""Import images (and YOLO labels) from a folder into a dataset.

Browser uploads land in <dataset>/.yolo-workbench/staging/<id>/ first and are then
imported exactly like a local folder. Files are always copied; labels are rewritten
with the chosen class mapping. Imports are not on the undo stack (delete the images
to reverse one) but each is logged to .yolo-workbench/imports.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import secrets
import shutil
import time
from pathlib import Path, PurePosixPath

import yaml

from . import classes
from .dataset import IMAGE_EXTENSIONS, WORKBENCH_DIR, Dataset, DatasetError
from .jobs import Job

SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val", "test": "test", "testing": "test"}
UPLOAD_EXTENSIONS = IMAGE_EXTENSIONS | {".txt", ".yaml", ".yml", ".json"}
STAGING_PATTERN = re.compile(r"^[a-f0-9]{16}$")
UNASSIGNED = ""  # classification images that are not inside a class folder


# ---------- staging (browser uploads) ----------


def staging_root(dataset: Dataset) -> Path:
    return dataset.root / WORKBENCH_DIR / "staging"


def create_staging(dataset: Dataset) -> dict:
    staging_id = secrets.token_hex(8)
    (staging_root(dataset) / staging_id).mkdir(parents=True)
    return {"staging_id": staging_id}


def staging_dir(dataset: Dataset, staging_id: str) -> Path:
    if not STAGING_PATTERN.match(staging_id or ""):
        raise DatasetError("Unknown upload")
    path = staging_root(dataset) / staging_id
    if not path.is_dir():
        raise DatasetError("Unknown upload; it may have expired")
    return path


def safe_relative(raw: str) -> PurePosixPath:
    """Normalize a client-supplied relative path; rejects absolute paths and '..'."""
    text = str(raw).replace("\\", "/").strip()
    path = PurePosixPath(text)
    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts or path.is_absolute() or any(part == ".." or part.startswith(".") for part in parts) or re.match(r"^[A-Za-z]:", parts[0]):
        raise DatasetError(f"Invalid upload path: {raw!r}")
    clean = PurePosixPath(*parts)
    if clean.suffix.lower() not in UPLOAD_EXTENSIONS:
        raise DatasetError(f"Unsupported file type: {clean.name}")
    return clean


def write_staged_file(dataset: Dataset, staging_id: str, relative: str, content: bytes) -> dict:
    root = staging_dir(dataset, staging_id)
    target = root.joinpath(*safe_relative(relative).parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise DatasetError(f"Invalid upload path: {relative!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return {"path": str(PurePosixPath(*safe_relative(relative).parts)), "size": len(content)}


def discard_staging(dataset: Dataset, staging_id: str) -> None:
    shutil.rmtree(staging_dir(dataset, staging_id), ignore_errors=True)


# ---------- scanning ----------


def resolve_source(dataset: Dataset, path: str | None, staging_id: str | None) -> Path:
    if staging_id:
        return staging_dir(dataset, staging_id)
    if not path:
        raise DatasetError("Choose a folder to import from")
    source = Path(path).expanduser()
    if not source.is_dir():
        raise DatasetError(f"Folder not found: {source}")
    source = source.resolve()
    if source == dataset.root or dataset.root.is_relative_to(source) or (source.is_relative_to(dataset.root) and WORKBENCH_DIR not in source.relative_to(dataset.root).parts):
        raise DatasetError("Import from a folder outside this dataset")
    return source


def _visible(path: Path, source: Path) -> bool:
    return not any(part.startswith(".") for part in path.relative_to(source).parts)


def _names_from_source(source: Path) -> tuple[dict[int, str] | None, str | None]:
    """Class names shipped with the source: data.yaml (or any dataset YAML), classes.txt, or notes.json."""
    candidates = sorted(source.glob("*.y*ml")) + sorted(source.glob("*/*.y*ml"))
    candidates.sort(key=lambda path: (path.parent != source, path.name != "data.yaml"))
    for candidate in candidates:
        try:
            config = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError, UnicodeDecodeError):
            continue
        names = config.get("names") if isinstance(config, dict) else None
        if isinstance(names, list):
            return dict(enumerate(str(name) for name in names)), candidate.name
        if isinstance(names, dict):
            try:
                return {int(key): str(value) for key, value in names.items()}, candidate.name
            except (TypeError, ValueError):
                continue
    for candidate in sorted(source.glob("classes.txt")) + sorted(source.glob("*/classes.txt")):
        try:
            lines = [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines()]
        except (OSError, UnicodeDecodeError):
            continue
        if any(lines):
            return {index: name for index, name in enumerate(lines) if name}, candidate.name
    for candidate in sorted(source.glob("notes.json")):
        try:
            categories = json.loads(candidate.read_text(encoding="utf-8")).get("categories", [])
            return {int(item["id"]): str(item["name"]) for item in categories}, candidate.name
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return None, None


def _label_candidates(image: Path, source: Path) -> list[Path]:
    relative = image.relative_to(source)
    parts = list(relative.parts)
    candidates = []
    if "images" in parts[:-1]:
        index = len(parts) - 1 - parts[::-1].index("images")
        candidates.append(source.joinpath(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt"))
    candidates.append(image.with_suffix(".txt"))
    return candidates


def _detected_split(relative: Path) -> str | None:
    for part in relative.parts[:-1]:
        split = SPLIT_ALIASES.get(part.casefold())
        if split:
            return split
    return None


def _scan_items(source: Path, category: str) -> list[dict]:
    items = []
    for image in sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file() and _visible(path, source)):
        relative = image.relative_to(source)
        item = {"image": image, "relative": relative, "split": _detected_split(relative), "label": None, "class": None}
        if category == "classification":
            parent = image.parent.name
            inside_class = image.parent != source and SPLIT_ALIASES.get(parent.casefold()) is None and parent.casefold() != "images"
            item["class"] = parent if inside_class else UNASSIGNED
        else:
            item["label"] = next((candidate for candidate in _label_candidates(image, source) if candidate.is_file()), None)
        items.append(item)
    return items


def _label_ids(path: Path) -> list[int]:
    ids = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ids
    for line in text.splitlines():
        token = line.split(maxsplit=1)[0] if line.strip() else None
        if token is None:
            continue
        try:
            ids.append(int(token))
        except ValueError:
            continue
    return ids


def scan(dataset: Dataset, source: Path) -> dict:
    """Summarize what an import from `source` would bring in, with a suggested class mapping."""
    items = _scan_items(source, dataset.category)
    by_name = {name.casefold(): class_id for class_id, name in dataset.names.items()}
    split_counts: dict[str, int] = {}
    for item in items:
        if item["split"]:
            split_counts[item["split"]] = split_counts.get(item["split"], 0) + 1
    result = {
        "source": str(source),
        "images": len(items),
        "splits": split_counts,
        "target_splits": dataset.capabilities()["splits"],
        "names_source": None,
        "labeled": 0,
        "classes": [],
    }
    if dataset.category == "classification":
        counts: dict[str, int] = {}
        for item in items:
            counts[item["class"]] = counts.get(item["class"], 0) + 1
        result["labeled"] = len(items) - counts.get(UNASSIGNED, 0)
        result["classes"] = [
            {"key": key, "name": key or None, "count": count, "named": bool(key), "suggested": by_name.get(key.casefold(), "new") if key else None}
            for key, count in sorted(counts.items())
        ]
        return result
    names, names_source = _names_from_source(source)
    result["names_source"] = names_source
    counts = {}
    for item in items:
        if item["label"] is not None:
            result["labeled"] += 1
            for class_id in _label_ids(item["label"]):
                counts[class_id] = counts.get(class_id, 0) + 1
    keys = sorted(set(counts) | set(names or {}))
    for class_id in keys:
        name = (names or {}).get(class_id)
        if names is not None and name is not None:
            suggested = by_name.get(name.casefold(), "new")
        else:
            # no names shipped: ids are assumed to match the project (unknown ids show up in Validation)
            suggested = class_id
        result["classes"].append({"key": str(class_id), "name": name, "count": counts.get(class_id, 0), "named": name is not None, "suggested": suggested})
    return result


# ---------- committing ----------


def _assign_splits(items: list[dict], options: dict, allowed: list[str]) -> None:
    mode = options.get("split_mode", "fixed")
    fallback = options.get("split") or (allowed[0] if allowed else "train")
    if mode == "auto":
        ratios = {split: max(0.0, float((options.get("ratios") or {}).get(split, 0))) for split in ("train", "val", "test")}
        ratios = {split: value for split, value in ratios.items() if value > 0}
        if not ratios or sum(ratios.values()) <= 0:
            raise DatasetError("Split ratios must add up to more than zero")
        order = list(range(len(items)))
        random.Random(int(options.get("seed", 0))).shuffle(order)
        total = sum(ratios.values())
        # largest-remainder allocation so counts always add up to the number of images
        exact = {split: len(items) * value / total for split, value in ratios.items()}
        counts = {split: int(value) for split, value in exact.items()}
        for split in sorted(exact, key=lambda key: exact[key] - counts[key], reverse=True)[: len(items) - sum(counts.values())]:
            counts[split] += 1
        position = 0
        for split in ("train", "val", "test"):
            for index in order[position : position + counts.get(split, 0)]:
                items[index]["target"] = split
            position += counts.get(split, 0)
    elif mode == "source":
        for item in items:
            item["target"] = item["split"] if item["split"] in allowed else fallback
    elif mode == "fixed":
        for item in items:
            item["target"] = fallback
    else:
        raise DatasetError("Split mode must be fixed, auto, or source")
    used = {item["target"] for item in items}
    if not used <= set(allowed):
        raise DatasetError(f"Split {', '.join(sorted(used - set(allowed)))} is not available in this dataset")


def _resolve_mapping(dataset: Dataset, scanned: dict, raw: dict) -> dict:
    """Turn {source key: class id | "new" | None} into source key -> dataset class id (None = drop).

    Classes marked "new" are added first, as one undoable schema change.
    """
    wanted: dict[str, object] = {}
    for item in scanned["classes"]:
        key = item["key"]
        choice = raw.get(key, item["suggested"]) if raw is not None else item["suggested"]
        if choice == "new" and not item["named"]:
            choice = None
        wanted[key] = choice
    new_names = []
    by_name = {name.casefold(): class_id for class_id, name in dataset.names.items()}
    for item in scanned["classes"]:
        if wanted[item["key"]] == "new" and item["name"].casefold() not in by_name and item["name"].casefold() not in {name.casefold() for name in new_names}:
            new_names.append(item["name"])
    if new_names:
        classes.add_classes(dataset, new_names)
        by_name = {name.casefold(): class_id for class_id, name in dataset.names.items()}
    mapping: dict[str, int | None] = {}
    for item in scanned["classes"]:
        choice = wanted[item["key"]]
        if choice == "new":
            mapping[item["key"]] = by_name[item["name"].casefold()]
        elif choice is None:
            mapping[item["key"]] = None
        else:
            class_id = int(choice)
            if item["named"] and class_id not in dataset.names:
                raise DatasetError(f"Unknown target class {class_id}")
            mapping[item["key"]] = class_id
    return mapping


def _unique(path: Path, planned: set[Path]) -> Path:
    candidate, counter = path, 1
    while candidate.exists() or candidate in planned:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        counter += 1
    return candidate


class _Fingerprints:
    """Exact-duplicate detection: compare sizes first, hash only when sizes collide."""

    def __init__(self, paths):
        self.by_size: dict[int, list[Path]] = {}
        self.hashes: dict[Path, str] = {}
        for path in paths:
            try:
                self.by_size.setdefault(path.stat().st_size, []).append(path)
            except OSError:
                continue

    def _hash(self, path: Path) -> str:
        if path not in self.hashes:
            digest = hashlib.sha1()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            self.hashes[path] = digest.hexdigest()
        return self.hashes[path]

    def seen(self, path: Path) -> bool:
        size = path.stat().st_size
        others = self.by_size.get(size, [])
        if others:
            digest = self._hash(path)
            if any(self._hash(other) == digest for other in others):
                return True
        self.by_size.setdefault(size, []).append(path)
        return False


def run_import(dataset: Dataset, source: Path, options: dict, job: Job | None = None) -> dict:
    allowed = dataset.capabilities()["splits"]
    if not allowed:
        raise DatasetError("This dataset's splits are not plain image folders, so images cannot be imported into it")
    scanned = scan(dataset, source)
    items = _scan_items(source, dataset.category)
    if not items:
        raise DatasetError("No images found in the selected folder")
    _assign_splits(items, options, allowed)
    # each split may have several folders; the caller picks one per split (default: the first)
    chosen = options.get("folders") or {}
    folders = {split: dataset.resolve_folder(split, chosen.get(split)) for split in {item["target"] for item in items}}
    mapping = _resolve_mapping(dataset, scanned, options.get("mapping"))
    label_mapping = {int(key): value for key, value in mapping.items()} if dataset.category != "classification" and scanned["names_source"] else None
    fingerprints = _Fingerprints(record.path for record in list(dataset.images.values())) if options.get("skip_duplicates", True) else None
    planned: set[Path] = set()
    stats = {"imported": 0, "labels": 0, "duplicates": 0, "dropped": 0, "renamed": 0, "by_split": {}}
    rescan: dict = {}
    if job:
        job.progress(0, len(items), "Copying files…")
    try:
        for index, item in enumerate(items):
            if job:
                job.progress(index, len(items))
            _copy_item(dataset, item, mapping, label_mapping, fingerprints, planned, stats, folders)
    finally:
        # also after a cancel: whatever was copied is part of the dataset now
        if job and not job.cancel_requested:
            job.progress(message="Indexing…")
        rescan = dataset.rescan()
        _log(dataset, source, options, mapping, stats)
    return {**stats, "rescan": rescan}


def _target_paths(dataset: Dataset, directory: Path, name: str, planned: set[Path]) -> tuple[Path, Path | None]:
    """A free image path (and its label path) in `directory`; numbered when the name is taken.

    The label is checked too: an orphan label with the same stem would silently attach to the new image.
    """
    base = directory / name
    candidate, counter = base, 0
    while True:
        label = None if dataset.category == "classification" else dataset._label_for(candidate)
        if not candidate.exists() and candidate not in planned and (label is None or (not label.exists() and label not in planned)):
            return candidate, label
        counter += 1
        candidate = base.with_name(f"{base.stem}_{counter}{base.suffix}")


def _copy_item(dataset: Dataset, item: dict, mapping: dict, label_mapping: dict | None, fingerprints: _Fingerprints | None, planned: set[Path], stats: dict, folders: dict[str, Path]) -> None:
    if dataset.category == "classification":
        class_id = mapping.get(item["class"])
        if class_id is None:
            stats["dropped"] += 1
            return
        directory = folders[item["target"]] / dataset.names[class_id]
    else:
        directory = folders[item["target"]]
    if fingerprints is not None and fingerprints.seen(item["image"]):
        stats["duplicates"] += 1
        return
    target, label_target = _target_paths(dataset, directory, item["image"].name, planned)
    if target.name != item["image"].name:
        stats["renamed"] += 1
    planned.add(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item["image"], target)
    if label_target is not None and item["label"] is not None:
        text = item["label"].read_text(encoding="utf-8")
        if label_mapping is not None:
            text = classes.remap_label_text(text, label_mapping)
        planned.add(label_target)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.write_text(text, encoding="utf-8")
        stats["labels"] += 1
    stats["imported"] += 1
    stats["by_split"][item["target"]] = stats["by_split"].get(item["target"], 0) + 1


def _log(dataset: Dataset, source: Path, options: dict, mapping: dict, stats: dict) -> None:
    entry = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "browser upload" if options.get("staging_id") else str(source),
        "options": {key: value for key, value in options.items() if key in {"split_mode", "split", "ratios", "seed", "skip_duplicates", "folders"}},
        "mapping": mapping,
        **stats,
    }
    try:
        log_path = dataset.root / WORKBENCH_DIR / "imports.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # the log is provenance only
