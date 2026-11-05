"""Splits in numbers and by name: per-split class tables, the batch a split folder belongs to, and what changed
between a snapshot and the dataset. Standard library only.

New data usually lands in a split folder per date or revision (`images/train_2024-03`, `2024-03/images/train`).
Such a folder's *batch* is its label with the split name masked (`images/*_2024-03`), so the train and val
folders of one batch share it; a label without a split name gets an implicit `_*` suffix (`batch7` and
`batch7_val` are one batch). Moving an image to another split puts it into its batch's folder there
(`twin_label`), which keeps where the data came from.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val", "test": "test", "testing": "test"}
# how a split is spelled in each naming family, so a twin folder reads like its source (`training` → `validation`)
_FAMILIES = (
    {"train": "train", "val": "val", "test": "test"},
    {"train": "training", "val": "validation", "test": "testing"},
    {"train": "train", "val": "valid", "test": "test"},
)
_TOKENS = re.compile(r"([_.\-]+)")


# ---------- tables ----------


def split_table(rows: Iterable[tuple[str, list[int]]]) -> dict:
    """Per split: images, background images (no objects), objects, and per class (str id) the images containing it
    and its objects. `rows`: (split, the class id of every object) per image."""
    table: dict[str, dict] = {}
    for split, class_ids in rows:
        row = table.setdefault(split, {"images": 0, "background": 0, "objects": 0, "classes": {}})
        row["images"] += 1
        row["objects"] += len(class_ids)
        if not class_ids:
            row["background"] += 1
        for class_id in set(class_ids):
            entry = row["classes"].setdefault(str(class_id), {"images": 0, "objects": 0})
            entry["images"] += 1
        for class_id in class_ids:
            row["classes"][str(class_id)]["objects"] += 1
    return table


def dataset_rows(dataset) -> list[tuple[str, list[int]]]:
    """(split, class ids of its objects) of every image. Caller holds the dataset lock."""
    return [(record.split, [annotation.class_id for annotation in record.annotations]) for record in dataset.images.values()]


# ---------- batches and twin folders ----------


def _mask(label: str) -> tuple[str, bool]:
    """The label with every split-name token replaced by `*`, and whether it had one."""
    found = False
    parts = []
    for component in label.split("/"):
        pieces = _TOKENS.split(component)
        for index in range(0, len(pieces), 2):
            if pieces[index].casefold() in SPLIT_ALIASES:
                pieces[index] = "*"
                found = True
        parts.append("".join(pieces))
    return "/".join(parts), found


def batch_key(label: str) -> str:
    """The batch of a split folder (as folder_label names it): its label with the split name masked."""
    masked, found = _mask(label)
    return masked if found else f"{masked}_*"


def _spelled(token: str, split: str) -> str:
    folded = token.casefold()
    family = next((family for family in _FAMILIES if family[SPLIT_ALIASES[folded]] == folded), _FAMILIES[0])
    word = family[split]
    if token.isupper():
        return word.upper()
    return word.capitalize() if token[:1].isupper() else word


def twin_label(label: str, split: str) -> str:
    """The folder label of `label`'s batch in `split`: its split name swapped (`images/train_2024-03` →
    `images/val_2024-03`), or `_<split>` appended when it has none (`batch7` → `batch7_val`)."""
    parts = []
    found = False
    for component in label.split("/"):
        pieces = _TOKENS.split(component)
        for index in range(0, len(pieces), 2):
            if pieces[index].casefold() in SPLIT_ALIASES:
                pieces[index] = _spelled(pieces[index], split)
                found = True
        parts.append("".join(pieces))
    return "/".join(parts) if found else f"{label}_{split}"


# ---------- snapshot ↔ dataset ----------


def relative_path(root: Path, path: Path) -> str:
    """A dataset file as manifests name it: POSIX, relative to the root when inside it. A string prefix test:
    Path.relative_to takes seconds over 100k images."""
    text, base = os.fspath(path), os.path.join(os.fspath(root), "")
    if os.path.normcase(text).startswith(os.path.normcase(base)):
        return text[len(base) :].replace(os.sep, "/")
    return Path(text).as_posix()


def below(folder: Path, path: Path) -> str:
    """A file's path below a folder holding it (POSIX)."""
    return os.fspath(path)[len(os.path.join(os.fspath(folder), "")) :].replace(os.sep, "/")


def image_key(path: str, folder: str | None, classification: bool) -> str:
    """What names one image across moves between splits: its batch and its path below the split folder (and, in
    classification, below its class folder, which a class change moves it between). Files of no split folder
    (`.txt` lists) are named by their path."""
    if not folder or not path.startswith(folder.rstrip("/") + "/"):
        return path
    below = path[len(folder.rstrip("/")) + 1 :]
    if classification and "/" in below:
        below = below.split("/", 1)[1]
    return f"{batch_key(folder)}|{below}"


def _keyed(items: list[tuple[str, dict]]) -> dict[str, dict]:
    """key -> item; a key two items share falls back to each one's path."""
    counts: dict[str, int] = {}
    for key, _ in items:
        counts[key] = counts.get(key, 0) + 1
    return {(key if counts[key] == 1 else f"path|{item['path']}"): item for key, item in items}


def diff(dataset, meta: dict, manifest: list[dict]) -> dict:
    """What changed from a snapshot (its meta and manifest rows) to the dataset now: images added, removed, moved
    to another split, relabelled (label content, or the class in classification); classes and split folders; and
    both split tables. Answered from memory: no filesystem call per image."""
    classification = dataset.category == "classification"
    with dataset._lock:
        names = [dataset.names[class_id] for class_id in sorted(dataset.names)]
        folders = {split: [dataset.folder_label(folder) for folder in listed] for split, listed in dataset._split_folders.items()}
        current_items = []
        for record in dataset.images.values():
            folder = dataset._folder_of(record)
            label = dataset._digests.get(record.label_path)
            item = {
                "id": record.id,
                "path": relative_path(dataset.root, record.path),
                "split": record.split,
                "digest": label.hex() if label else None,
                "classes": [annotation.class_id for annotation in record.annotations],
            }
            current_items.append((image_key(item["path"], dataset.folder_label(folder) if folder else None, classification), item))
        table = split_table(dataset_rows(dataset))
    then = _keyed([(image_key(row["path"], row.get("folder"), classification), row) for row in manifest])
    now = _keyed(current_items)
    added = [now[key]["id"] for key in now.keys() - then.keys()]
    removed = sorted(then[key]["path"] for key in then.keys() - now.keys())
    moved, changed = [], []
    for key in now.keys() & then.keys():
        before, after = then[key], now[key]
        if before["split"] != after["split"]:
            moved.append(after["id"])
        relabelled = sorted(before.get("classes") or []) != sorted(after["classes"]) if classification else before.get("digest") != after["digest"]
        if relabelled:
            changed.append(after["id"])
    then_names = meta.get("names") or []
    return {
        "counts": {"added": len(added), "removed": len(removed), "moved": len(moved), "changed": len(changed), "unchanged": len(now.keys() & then.keys()) - len(set(moved) | set(changed))},
        "added": added,
        "removed": removed,
        "moved": moved,
        "changed": changed,
        "names": {"then": then_names, "now": names, "same": then_names == names},
        "folders": {"then": meta.get("folders") or {}, "now": folders},
        "table": {"then": meta.get("table") or {}, "now": table},
    }
