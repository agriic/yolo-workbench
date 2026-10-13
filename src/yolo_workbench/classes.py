"""Class schema operations: add, rename, remove (delete or merge), merge, reorder.

Every operation compiles to one undoable Dataset transaction. For detection and
segmentation that is a remap of class ids across label files plus a rewrite of the
YAML `names` block; for classification (class ids = sorted folder names) it is a set
of folder and file moves. A SchemaEntry closes each transaction so replay reloads
names from disk and listeners (predictor, embeddings) learn the id mapping.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .dataset import (
    WORKBENCH_DIR,
    Dataset,
    DatasetError,
    DirEntry,
    DirMoveEntry,
    FileEntry,
    HistoryEntry,
    MoveEntry,
    RecordMoveEntry,
    SchemaEntry,
)

MAX_NAME_LENGTH = 100


def class_summary(dataset: Dataset) -> dict:
    with dataset._lock:
        items = [
            {
                "id": class_id,
                "name": name,
                "annotations": len(dataset._object_index.get(class_id, {})),
                "images": len(dataset._class_image_ids.get(class_id, ())),
            }
            for class_id, name in sorted(dataset.names.items())
        ]
        return {"items": items, "category": dataset.category, "reorderable": dataset.category != "classification"}


def validate_name(dataset: Dataset, name: str, exclude_id: int | None = None, pending: tuple[str, ...] = ()) -> str:
    name = str(name).strip()
    if not name:
        raise DatasetError("Class name cannot be empty")
    if len(name) > MAX_NAME_LENGTH:
        raise DatasetError(f"Class name is longer than {MAX_NAME_LENGTH} characters")
    if any(ch in name for ch in "/\\\n\r\t") or name in {".", ".."}:
        raise DatasetError(f"Class name {name!r} contains characters that are not allowed")
    if dataset.category == "classification" and name.startswith("."):
        raise DatasetError("Classification class names cannot start with '.' (they are folder names)")
    taken = {other.casefold() for class_id, other in dataset.names.items() if class_id != exclude_id}
    taken |= {other.casefold() for other in pending}
    if name.casefold() in taken:
        raise DatasetError(f"A class named {name!r} already exists")
    return name


def add_classes(dataset: Dataset, names: list[str]) -> dict:
    with dataset._lock:
        clean: list[str] = []
        for name in names:
            clean.append(validate_name(dataset, name, pending=tuple(clean)))
        if not clean:
            raise DatasetError("No class names given")
        if dataset.category == "classification":
            return _classification_add(dataset, clean)
        start = max(dataset.names, default=-1) + 1
        new_names = {**dataset.names, **{start + index: name for index, name in enumerate(clean)}}
        return _remap(dataset, {class_id: class_id for class_id in dataset.names}, new_names)


def rename_class(dataset: Dataset, class_id: int, name: str) -> dict:
    with dataset._lock:
        old = _require(dataset, class_id)
        name = validate_name(dataset, name, exclude_id=class_id)
        if name == old:
            return _result(dataset, {class_id: class_id for class_id in dataset.names})
        if dataset.category == "classification":
            return _classification_rename(dataset, old, name)
        return _remap(dataset, {key: key for key in dataset.names}, {**dataset.names, class_id: name})


def remove_class(dataset: Dataset, class_id: int, mode: str, target_id: int | None = None) -> dict:
    if mode == "merge":
        if target_id is None:
            raise DatasetError("Merging needs a target class")
        return merge_classes(dataset, [class_id], target_id)
    if mode != "delete":
        raise DatasetError("Mode must be 'delete' or 'merge'")
    with dataset._lock:
        _require(dataset, class_id)
        if dataset.category == "classification":
            return _classification_remove(dataset, [class_id], None)
        return _remap(dataset, *_compact(dataset, {class_id: None}))


def merge_classes(dataset: Dataset, source_ids: list[int], target_id: int) -> dict:
    with dataset._lock:
        _require(dataset, target_id)
        sources = list(dict.fromkeys(int(class_id) for class_id in source_ids))
        for class_id in sources:
            _require(dataset, class_id)
        if not sources or target_id in sources:
            raise DatasetError("Choose source classes different from the target class")
        if dataset.category == "classification":
            return _classification_remove(dataset, sources, target_id)
        return _remap(dataset, *_compact(dataset, {class_id: target_id for class_id in sources}))


def reorder_classes(dataset: Dataset, order: list[int]) -> dict:
    with dataset._lock:
        if dataset.category == "classification":
            raise DatasetError("Classification class ids follow the folder names alphabetically and cannot be reordered")
        order = [int(class_id) for class_id in order]
        if sorted(order) != sorted(dataset.names):
            raise DatasetError("The new order must list every class exactly once")
        mapping = {old: new for new, old in enumerate(order)}
        return _remap(dataset, mapping, {new: dataset.names[old] for new, old in enumerate(order)})


def _require(dataset: Dataset, class_id: int) -> str:
    if class_id not in dataset.names:
        raise DatasetError(f"Unknown class {class_id}")
    return dataset.names[class_id]


def _compact(dataset: Dataset, removed: dict[int, int | None]) -> tuple[dict[int, int | None], dict[int, str]]:
    """Drop classes (mapped to None or merged into a kept class) and renumber the rest to 0..n-1."""
    kept = [class_id for class_id in sorted(dataset.names) if class_id not in removed]
    renumber = {old: new for new, old in enumerate(kept)}
    mapping: dict[int, int | None] = dict(renumber)
    for class_id, target in removed.items():
        mapping[class_id] = None if target is None else renumber[target]
    return mapping, {renumber[old]: dataset.names[old] for old in kept}


def _result(dataset: Dataset, mapping: dict, changed_files: int = 0) -> dict:
    return {"mapping": {str(key): value for key, value in mapping.items()}, "files": changed_files, **class_summary(dataset), **dataset.history_state()}


# ---------- detection / segmentation ----------

_LINE = re.compile(r"^(\s*)(\S+)(.*)$", re.DOTALL)


def remap_label_text(text: str, mapping: dict[int, int | None]) -> str:
    """Rewrite the class token of each label line; geometry text is kept byte for byte.

    Lines whose class is not in the mapping (unknown ids, malformed rows) are left untouched.
    """
    lines = []
    for line in text.splitlines(keepends=True):
        match = _LINE.match(line)
        if not match:
            lines.append(line)
            continue
        indent, token, rest = match.groups()
        try:
            class_id = int(token)
        except ValueError:
            lines.append(line)
            continue
        if class_id not in mapping:
            lines.append(line)
        elif mapping[class_id] is not None:
            lines.append(f"{indent}{mapping[class_id]}{rest}")
    return "".join(lines)


def _remap(dataset: Dataset, mapping: dict[int, int | None], new_names: dict[int, str]) -> dict:
    entries: list = []
    changes = {class_id: target for class_id, target in mapping.items() if target != class_id}
    if changes:
        for record in dataset.images.values():
            if not record.label_path.exists():
                continue
            # malformed rows are not parsed into annotations but may still carry a class token
            affected = any(annotation.class_id in changes for annotation in record.annotations) or any(issue["kind"] == "malformed_label" for issue in record.issues)
            if not affected:
                continue
            before = record.label_path.read_text(encoding="utf-8")
            after = remap_label_text(before, mapping)
            if after != before:
                entries.append(HistoryEntry(record.label_path, before, after, True, True))
    entries.append(_names_entry(dataset, new_names))
    entries.append(SchemaEntry(dict(mapping)))
    dataset._execute(entries)
    return _result(dataset, mapping, len(entries) - 2)


def _names_entry(dataset: Dataset, new_names: dict[int, str]) -> FileEntry:
    path = dataset.yaml_path
    exists = path.is_file()
    before = path.read_text(encoding="utf-8") if exists else ""
    after = write_names(before, new_names, dataset.names_style)
    return FileEntry(path, before, after, exists, True)


def _names_block(names: dict[int, str], style: str) -> str:
    contiguous = sorted(names) == list(range(len(names)))
    value = [names[key] for key in sorted(names)] if style == "list" and contiguous else {key: names[key] for key in sorted(names)}
    return yaml.safe_dump({"names": value}, sort_keys=False, allow_unicode=True, default_flow_style=False)


def write_names(text: str, names: dict[int, str], style: str = "dict") -> str:
    """Replace only the top-level `names:` block, keeping every other line (and comment) intact.

    Falls back to a full dump when the block cannot be located or the edit would not parse back.
    """
    block = _names_block(names, style)
    lines = text.splitlines(keepends=True)
    start = next((index for index, line in enumerate(lines) if re.match(r"^names\s*:", line)), None)
    if start is None:
        candidate = text + ("" if not text or text.endswith("\n") else "\n") + block
    else:
        end = start + 1
        while end < len(lines):
            line = lines[end]
            if line.strip() and not line[0].isspace() and not line.startswith("-") and not line.lstrip().startswith("#"):
                break
            end += 1
        while end > start + 1 and (not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")):
            end -= 1  # trailing blank lines and comments belong to whatever follows
        candidate = "".join(lines[:start]) + block + "".join(lines[end:])
    try:
        original = yaml.safe_load(text) or {} if text.strip() else {}
        parsed = yaml.safe_load(candidate) or {}
        expected_names = yaml.safe_load(block)["names"]
        if isinstance(original, dict) and parsed.get("names") == expected_names and {k: v for k, v in parsed.items() if k != "names"} == {k: v for k, v in original.items() if k != "names"}:
            return candidate
    except yaml.YAMLError:
        pass
    config = yaml.safe_load(text) if text.strip() else {}
    config = config if isinstance(config, dict) else {}
    config["names"] = yaml.safe_load(block)["names"]
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


# ---------- classification (classes are folders) ----------


def _name_mapping(dataset: Dataset, renamed: dict[str, str | None]) -> dict[int, int | None]:
    """old id -> new id, given old name -> new name (None = gone) and sorted-name ids afterwards."""
    final = sorted({renamed.get(name, name) for name in dataset.names.values()} - {None})
    position = {name: index for index, name in enumerate(final)}
    mapping: dict[int, int | None] = {}
    for class_id, name in dataset.names.items():
        target = renamed.get(name, name)
        mapping[class_id] = None if target is None else position[target]
    return mapping


def _classification_yaml(dataset: Dataset, names: list[str]) -> list:
    if not dataset.yaml_path.is_file():
        return []
    return [_names_entry(dataset, dict(enumerate(sorted(names))))]


def _split_dirs(dataset: Dataset) -> dict[str, Path]:
    if dataset._split_dirs:
        return dataset._split_dirs
    # an empty managed project may not have split folders yet
    return {"train": dataset.root / "train"}


def _classification_add(dataset: Dataset, names: list[str]) -> dict:
    entries: list = []
    for directory in _split_dirs(dataset).values():
        for name in names:
            entries.append(DirEntry(directory / name, True))
    all_names = list(dataset.names.values()) + names
    position = {name: index for index, name in enumerate(sorted(all_names))}
    mapping = {class_id: position[name] for class_id, name in dataset.names.items()}
    entries += _classification_yaml(dataset, all_names)
    entries.append(SchemaEntry(mapping))
    if not dataset._split_dirs:
        dataset._split_dirs = dict(_split_dirs(dataset))
        dataset._split_image_dirs = dict(dataset._split_dirs)
    dataset._execute(entries)
    return _result(dataset, mapping)


def _classification_rename(dataset: Dataset, old: str, new: str) -> dict:
    entries: list = []
    for directory in dataset._split_dirs.values():
        if (directory / old).is_dir():
            if (directory / new).exists():
                raise DatasetError(f"Folder {new}/ already exists in {directory.name}/")
            entries.append(DirMoveEntry(directory / old, directory / new))
    names = [new if name == old else name for name in dataset.names.values()]
    mapping = _name_mapping(dataset, {old: new})
    entries += _classification_yaml(dataset, names)
    entries.append(SchemaEntry(mapping))
    dataset._execute(entries)
    return _result(dataset, mapping)


def _trash_dir(dataset: Dataset, directory: Path) -> Path:
    # kept apart from the per-image trash paths, which recreate the same relative folders
    relative = directory.relative_to(dataset.root) if directory.is_relative_to(dataset.root) else Path(directory.name)
    target = dataset.root / WORKBENCH_DIR / "trash" / dataset.session_id / "_folders" / relative
    counter = 1
    while target.exists():
        target = target.with_name(f"{directory.name}.{counter}")
        counter += 1
    return target


def _classification_remove(dataset: Dataset, class_ids: list[int], target_id: int | None) -> dict:
    """Delete (images to the trash) or merge (images into the target folder) whole class folders."""
    removed = {dataset.names[class_id] for class_id in class_ids}
    target_name = None if target_id is None else dataset.names[target_id]
    entries: list = []
    destinations: set[Path] = set()
    for record in sorted(dataset.images.values(), key=lambda item: str(item.path)):
        if dataset._class_name_for(record) not in removed:
            continue
        if target_name is None:
            entries.append(RecordMoveEntry(record.id, record.split, None, [(record.path, dataset._trash_path(record.path))]))
            continue
        dst = dataset._split_dirs[record.split] / target_name / record.path.name
        if dst.exists() or dst in destinations:
            raise DatasetError(f"Cannot merge: {target_name}/ in {record.split} already has a file named {record.path.name}")
        destinations.add(dst)
        entries.append(MoveEntry(record.id, record.path, dst))
    for directory in dataset._split_dirs.values():
        for name in sorted(removed):
            folder = directory / name
            if folder.is_dir():
                # whatever is left (non-image files) goes to the trash with the folder
                entries.append(DirMoveEntry(folder, _trash_dir(dataset, folder)))
    mapping = _name_mapping(dataset, {name: target_name for name in removed})
    entries += _classification_yaml(dataset, [name for name in dataset.names.values() if name not in removed])
    entries.append(SchemaEntry(mapping))
    dataset._execute(entries)
    return _result(dataset, mapping, len([entry for entry in entries if isinstance(entry, (MoveEntry, RecordMoveEntry))]))
