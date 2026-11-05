"""Snapshots: named, frozen copies of a dataset (images, labels, classes, splits) to download, train on, compare
with the dataset, and restore.

A snapshot is `<dataset>/.yolo-workbench/snapshots/<id>/`, laid out like the full download (download.plan): a
dataset of its own beside its data.yaml, which `open_dataset` opens read-only. Its bookkeeping is in its own
`.yolo-workbench/` (pruned by every walk): `snapshot.json` (name, kind, counts, split table, …), `manifest.jsonl`
(one line per image: where it was in the dataset, its split, label digest and classes), `index.json` (image
sizes, so opening it probes nothing) and `source.yaml` (the dataset's own YAML, for restoring in place).

The workbench never edits a file in place (labels: temp file + rename; images: moved, swapped or trashed by
rename), so a hardlink keeps the version it links whatever happens to the dataset later. Images are hardlinked,
copied where that fails (another volume, a share without hardlinks), never symlinked; labels are copied. While the
images are linked, a structure hold refuses image moves, deletes and swaps (Dataset.hold_structure); labels were
copied under the dataset lock before, so label edits go on.

Kinds: `manual` (kept until deleted), `auto` (before a rebalance or a restore, a proposed split: deleted after
runtime.auto_snapshot_retention_days unless a run uses it), `run` (taken for a training run: deleted with the
last run that uses it, see `release`).
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import download, identity
from .dataset import IMAGE_EXTENSIONS, SIZE_INDEX_VERSION, WORKBENCH_DIR, Dataset, DatasetError
from .fsutil import InsufficientSpace, ensure_free_space, replace
from .jobs import Job
from .release import release_name
from .runtime import RUNTIME
from .splitstats import SPLITS, relative_path, split_table
from .yamledit import remove_key, write_key

log = logging.getLogger(__name__)

FOLDER = "snapshots"
KINDS = ("manual", "auto", "run")
ID_PATTERN = re.compile(r"^\d{8}-\d{6}(-\d+)?$")
META = "snapshot.json"
MANIFEST = "manifest.jsonl"
SOURCE_YAML = "source.yaml"
WORKERS = 8  # a link or a copy on a network drive waits on round trips
BATCH = 64  # images placed between progress reports (and cancel checks)
MAX_NAME = 120
MAX_NOTE = 2000
COPY_MARGIN = 256 * 1024 * 1024  # free space a full copy must leave when no minimum is configured

_lock = threading.Lock()  # snapshot.json updates (used_by), and the partial folders being written
_writing: set[Path] = set()


class SnapshotInUse(DatasetError):
    pass


def folder(dataset_root: Path) -> Path:
    return Path(dataset_root) / WORKBENCH_DIR / FOLDER


def snapshot_dir(dataset_root: Path, snapshot_id: str) -> Path:
    if not ID_PATTERN.match(snapshot_id or ""):
        raise DatasetError("Invalid snapshot id")
    path = folder(dataset_root) / snapshot_id
    if not (path / WORKBENCH_DIR / META).is_file():
        raise KeyError("Snapshot not found")
    return path


def _new_id(base: Path) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate, index = stamp, 2
    while (base / candidate).exists() or (base / f".partial-{candidate}").exists():
        candidate = f"{stamp}-{index}"
        index += 1
    return candidate


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
    temporary.write_text(json.dumps(data, indent=1), encoding="utf-8", newline="\n")
    replace(temporary, path)


def read_meta(path: Path) -> dict | None:
    try:
        data = json.loads((path / WORKBENCH_DIR / META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def manifest(dataset_root: Path, snapshot_id: str) -> list[dict]:
    path = snapshot_dir(dataset_root, snapshot_id) / WORKBENCH_DIR / MANIFEST
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


# ---------- fingerprints and link checks ----------


def fingerprint(dataset: Dataset) -> str:
    """What the dataset holds, digested: every image's path, split, file stamp and label content, plus the class
    list and split folders. Two equal fingerprints train on the same data (a snapshot taken for a run is reused)."""
    digest = hashlib.sha1()
    with dataset._lock:
        root = dataset.root
        for record in sorted(dataset.images.values(), key=lambda item: str(item.path)):
            label = dataset._digests.get(record.label_path)
            digest.update(f"{relative_path(root, record.path)}\0{record.split}\0{record.image_mtime}\0{record.image_size}\0{label.hex() if label else '-'}\n".encode())
        digest.update(json.dumps([dataset.names.get(class_id) for class_id in sorted(dataset.names)]).encode())
        digest.update(json.dumps({split: [dataset.folder_label(path) for path in folders] for split, folders in sorted(dataset._split_folders.items())}).encode())
    return digest.hexdigest()[:16]


def _groups(dataset: Dataset) -> dict[Path, list]:
    """The dataset's records by split folder (a file of none: by its own folder). Caller holds the lock."""
    groups: dict[Path, list] = {}
    for record in dataset.images.values():
        groups.setdefault(dataset._folder_of(record) or record.path.parent, []).append(record)
    return groups


def _links(source: Path, base: Path) -> bool:
    """Whether a file of `source`'s folder can be hardlinked into `base` (one link made and removed)."""
    probe = base / f".linktest-{secrets.token_hex(4)}"
    try:
        os.link(source, probe)
    except OSError:
        return False
    try:
        probe.unlink()
    except OSError:
        pass
    return True


def link_check(dataset: Dataset) -> dict:
    """Whether a snapshot would hardlink the images or copy them: {"mode": link | copy | mixed | none, "images",
    "copy_images", "copy_bytes"}. One image per split folder is linked and unlinked to tell."""
    base = folder(dataset.root)
    base.mkdir(parents=True, exist_ok=True)
    with dataset._lock:
        groups = [(records[0].path, len(records), sum(record.image_size or 0 for record in records)) for records in _groups(dataset).values()]
    images = sum(count for _, count, _ in groups)
    copy = [(count, size) for source, count, size in groups if not _links(source, base)]
    copy_images = sum(count for count, _ in copy)
    mode = "none" if not images else "copy" if copy_images == images else "mixed" if copy_images else "link"
    return {"mode": mode, "images": images, "copy_images": copy_images, "copy_bytes": sum(size for _, size in copy)}


def _check_space(base: Path, needed: int) -> None:
    ensure_free_space(base, needed, "the snapshot's copies of the images")
    if not needed or RUNTIME.min_free_bytes:
        return
    try:
        free = shutil.disk_usage(base).free
    except OSError:
        return
    if free - needed < COPY_MARGIN:
        raise InsufficientSpace(f"Not enough free disk space for the snapshot: its images are copied here (no hardlinks on this drive), {needed / 2**30:.1f} GB, and {free / 2**30:.1f} GB is free")


# ---------- taking a snapshot ----------


def _place(source: Path, target: Path) -> str:
    """Put an image into the snapshot: "linked", "copied", or "missing" (gone meanwhile)."""
    try:
        os.link(source, target)
        return "linked"
    except FileNotFoundError:
        return "missing"
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            return "linked"
    try:
        shutil.copy2(source, target)
        return "copied"
    except FileNotFoundError:
        return "missing"


def _split_lists(yaml_text: str | None, assignment: dict[str, str], placed: dict[str, tuple[str, str | None]], extra: list[download.Entry]) -> str:
    """data.yaml naming a `<split>.txt` list per split, for a snapshot laid out by assigned splits."""
    text = yaml_text or ""
    for split in SPLITS:
        names = sorted(placed[image_id][0] for image_id, assigned in assignment.items() if assigned == split and image_id in placed)
        if names:
            extra.append(download.Entry(f"{split}.txt", text="".join(f"./{name}\n" for name in names)))
            text = write_key(text, split, f"{split}.txt")
        elif text:
            text = remove_key(text, split)
    return text


def take(dataset: Dataset, name: str = "", note: str = "", kind: str = "manual", job: Job | None = None, assignment: dict[str, str] | None = None, extra: dict | None = None) -> dict:
    """Take a snapshot of the dataset now; returns its summary (snapshot.json). `assignment` ({image id: split})
    records other splits than the dataset's (a proposed split): the files keep their places and data.yaml lists
    them per split. `extra` is kept in snapshot.json (a rebalance's rules and plan digest)."""
    if kind not in KINDS:
        raise DatasetError("Unknown snapshot kind")
    if assignment and dataset.category == "classification":
        raise DatasetError("A classification dataset's splits are folders; apply the split to train on it")
    name = (name or "").strip()[:MAX_NAME] or time.strftime("Snapshot %Y-%m-%d %H:%M")
    base = folder(dataset.root)
    base.mkdir(parents=True, exist_ok=True)
    purge_expired(dataset.root)
    with _lock:
        snapshot_id = _new_id(base)
        partial = base / f".partial-{snapshot_id}"
        partial.mkdir()
        _writing.add(partial)
    final = base / snapshot_id
    token = None
    try:
        if job:
            job.progress(0, 0, "Copying the labels…")
        with dataset._lock:
            placed: dict[str, tuple[str, str | None]] = {}
            entries = download.plan(dataset, "full", placed)
            records = [dataset.images[image_id] for image_id in placed]
            image_names = {image for image, _ in placed.values()}
            labels = [entry for entry in entries if entry.path is not None and entry.name not in image_names]
            texts = dataset._read_texts([entry.path for entry in labels])
            yaml_text = dataset.yaml_path.read_text(encoding="utf-8") if dataset.yaml_path.is_file() else None
            if assignment:
                assigned = {record.id: assignment.get(record.id, record.split) for record in records}
                lists: list[download.Entry] = []
                data_entry = next((entry for entry in entries if entry.name == "data.yaml"), None)
                text = _split_lists(data_entry.text if data_entry else None, assigned, placed, lists)
                entries = [entry for entry in entries if entry.path is not None or entry.name not in ("data.yaml", *(f"{split}.txt" for split in SPLITS))] + lists + [download.Entry("data.yaml", text=text)]
            else:
                assigned = {record.id: record.split for record in records}
            for entry in entries:
                if entry.path is None:
                    target = partial / entry.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(entry.text, encoding="utf-8", newline="\n")
            label_bytes = 0
            for entry in labels:
                content, _, error = texts.get(entry.path, (None, None, None))
                if content is None:
                    continue  # unreadable or gone: the snapshot has no label for it, as the dataset reads it
                target = partial / entry.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8", newline="\n")
                label_bytes += len(content.encode("utf-8"))
            rows, index = [], {}
            for record in records:
                image_name, label_name = placed[record.id]
                folder_path = dataset._folder_of(record)
                label = dataset._digests.get(record.label_path)
                rows.append({
                    "path": relative_path(dataset.root, record.path),
                    "file": image_name,
                    "label": label_name,
                    "label_path": relative_path(dataset.root, record.label_path) if label_name else None,
                    "split": assigned[record.id],
                    "folder": dataset.folder_label(folder_path) if folder_path else None,
                    "digest": label.hex() if label else None,
                    "classes": [annotation.class_id for annotation in record.annotations],
                    "size": [record.width, record.height],
                    "stamp": [record.image_mtime, record.image_size],
                    "review": record.review,
                })
                if record.width > 0 and record.image_mtime is not None:
                    index[str(final / image_name)] = {"mtime_ns": record.image_mtime, "size": record.image_size, "width": record.width, "height": record.height}
            meta = {
                "version": 1,
                "id": snapshot_id,
                "name": name,
                "note": (note or "").strip()[:MAX_NOTE],
                "kind": kind,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "by": identity.who(),
                "fingerprint": None,
                "category": dataset.category,
                "names": [dataset.names[class_id] for class_id in sorted(dataset.names)],
                "folders": {split: [dataset.folder_label(path) for path in listed] for split, listed in dataset._split_folders.items()},
                "images": len(records),
                "splits": {split: sum(1 for value in assigned.values() if value == split) for split in SPLITS if any(value == split for value in assigned.values())},
                "table": split_table((assigned[record.id], [annotation.class_id for annotation in record.annotations]) for record in records),
                "folder_counts": None if assignment else _folder_counts(rows),
                "revision": dataset.revision,
                "schema_generation": dataset.schema_generation,
                "has_yaml": (partial / "data.yaml").is_file(),
                "release": release_name(),
                "used_by": [],
                **(extra or {}),
            }
            meta["fingerprint"] = None if assignment else fingerprint(dataset)
            workdir = partial / WORKBENCH_DIR
            workdir.mkdir()
            with open(workdir / MANIFEST, "w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            (workdir / "index.json").write_text(json.dumps({"version": SIZE_INDEX_VERSION, "images": index}), encoding="utf-8")
            if yaml_text is not None:
                (workdir / SOURCE_YAML).write_text(yaml_text, encoding="utf-8", newline="\n")
            sources = [(record.path, partial / placed[record.id][0], record) for record in records]
            grouped = _groups(dataset)
            groups = {key: records_of[0].path for key, records_of in grouped.items()}
            group_of = {record.id: key for key, records_of in grouped.items() for record in records_of}
            token = dataset.hold_structure("a snapshot is being taken")
        # outside the lock: image moves wait for the hold, everything else goes on
        copied_groups = {key for key, source in groups.items() if not _links(source, base)}
        _check_space(base, sum(record.image_size or 0 for _, _, record in sources if group_of[record.id] in copied_groups))
        for parent in {target.parent for _, target, _ in sources}:
            parent.mkdir(parents=True, exist_ok=True)
        counts = {"linked": 0, "copied": 0, "missing": 0}
        copied_bytes = 0
        if job:
            job.progress(0, len(sources), "Linking the images…")
        with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="snapshot") as pool:
            for start in range(0, len(sources), BATCH):
                batch = sources[start : start + BATCH]
                for (_, _, record), result in zip(batch, pool.map(lambda item: _place(item[0], item[1]), batch)):
                    counts[result] += 1
                    if result == "copied":
                        copied_bytes += record.image_size or 0
                if job:
                    job.progress(start + len(batch), len(sources))
        image_bytes = sum(record.image_size or 0 for _, _, record in sources)
        meta.update(counts, copied_bytes=copied_bytes, bytes=image_bytes + label_bytes, own_bytes=copied_bytes + label_bytes, counted=meta["created"])
        _write_json(partial / WORKBENCH_DIR / META, meta)
        replace(partial, final)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    finally:
        if token is not None:
            dataset.release_structure(token)
        with _lock:
            _writing.discard(partial)
    log.info("snapshot %s of %s: %d images (%d linked, %d copied, %d missing)", snapshot_id, dataset.root.name, meta["images"], meta["linked"], meta["copied"], meta["missing"])
    return meta


def _folder_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Images per split folder ({split: {folder label: count}}), for the new-run form's folder choice."""
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.get("folder"):
            split = counts.setdefault(row["split"], {})
            split[row["folder"]] = split.get(row["folder"], 0) + 1
    return counts


def folder_counts(dataset_root: Path, meta: dict) -> dict[str, dict[str, int]] | None:
    """A snapshot's images per split folder; worked out from its manifest (and kept) for a snapshot taken before
    snapshot.json recorded them. None for a proposed split (its splits are lists, not folders)."""
    if meta.get("rebalance"):
        return None
    if meta.get("folder_counts") is not None:
        return meta["folder_counts"]
    try:
        counts = _folder_counts(manifest(dataset_root, meta["id"]))
        _update(dataset_root, meta["id"], lambda data: data.__setitem__("folder_counts", counts))
    except (KeyError, OSError, ValueError):
        return None
    return counts


# ---------- listing, changing, deleting ----------


def list_snapshots(dataset_root: Path) -> list[dict]:
    """Every snapshot's summary, newest first."""
    base = folder(dataset_root)
    items = []
    if base.is_dir():
        for path in base.iterdir():
            if path.is_dir() and ID_PATTERN.match(path.name):
                meta = read_meta(path)
                if meta is not None:
                    items.append(meta)
    return sorted(items, key=lambda item: item["id"], reverse=True)


def summary(dataset: Dataset) -> dict:
    purge_expired(dataset.root)
    items = list_snapshots(dataset.root)
    return {"items": items, "retention_days": RUNTIME.auto_snapshot_retention_days, "total_bytes": sum(item.get("own_bytes") or 0 for item in items)}


def get(dataset_root: Path, snapshot_id: str) -> dict:
    meta = read_meta(snapshot_dir(dataset_root, snapshot_id))
    if meta is None:
        raise KeyError("Snapshot not found")
    return meta


def _update(dataset_root: Path, snapshot_id: str, change) -> dict:
    path = snapshot_dir(dataset_root, snapshot_id)
    with _lock:
        meta = read_meta(path)
        if meta is None:
            raise KeyError("Snapshot not found")
        change(meta)
        _write_json(path / WORKBENCH_DIR / META, meta)
    return meta


def update(dataset_root: Path, snapshot_id: str, name: str | None = None, note: str | None = None, keep: bool | None = None) -> dict:
    """Rename, annotate, or keep a snapshot (an automatic or a run's snapshot kept is kept like one taken by hand)."""

    def change(meta: dict) -> None:
        if name is not None:
            if not name.strip():
                raise DatasetError("A snapshot needs a name")
            meta["name"] = name.strip()[:MAX_NAME]
        if note is not None:
            meta["note"] = note.strip()[:MAX_NOTE]
        if keep:
            meta["kind"] = "manual"

    return _update(dataset_root, snapshot_id, change)


def use(dataset_root: Path, snapshot_id: str, run: str) -> dict:
    """Record that training run `run` (its folder name) trains on this snapshot."""
    return _update(dataset_root, snapshot_id, lambda meta: meta.__setitem__("used_by", sorted({*meta.get("used_by", []), run})))


def release(dataset_root: Path, snapshot_id: str, run: str) -> bool:
    """Run `run` no longer uses the snapshot (it was deleted); a run's own snapshot no run uses any more is deleted.
    Returns whether it was."""
    try:
        meta = _update(dataset_root, snapshot_id, lambda meta: meta.__setitem__("used_by", [item for item in meta.get("used_by", []) if item != run]))
    except KeyError:
        return False
    if meta.get("kind") == "run" and not meta.get("used_by"):
        delete(dataset_root, snapshot_id)
        return True
    return False


def delete(dataset_root: Path, snapshot_id: str) -> dict:
    path = snapshot_dir(dataset_root, snapshot_id)
    doomed = path.with_name(f".deleting-{snapshot_id}-{time.time_ns()}")
    try:
        replace(path, doomed)
    except OSError as exc:
        raise DatasetError(f"Could not delete the snapshot ({exc.strerror or exc}); is one of its files open in another program?") from exc
    shutil.rmtree(doomed, ignore_errors=True)
    log.info("snapshot %s of %s deleted", snapshot_id, Path(dataset_root).name)
    return {"deleted": snapshot_id}


def purge_expired(dataset_root: Path, days: float | None = None) -> list[str]:
    """Delete automatic snapshots older than `days` (runtime.auto_snapshot_retention_days; 0 keeps them) that no
    run uses, and what a crash left behind (partial or half-deleted folders)."""
    days = RUNTIME.auto_snapshot_retention_days if days is None else days
    base = folder(dataset_root)
    if not base.is_dir():
        return []
    purged = []
    cutoff = time.time() - days * 86400
    for path in base.iterdir():
        if path.name.startswith((".partial-", ".deleting-")):
            with _lock:
                busy = path in _writing
            if not busy:
                shutil.rmtree(path, ignore_errors=True)
            continue
        if days <= 0 or not ID_PATTERN.match(path.name):
            continue
        meta = read_meta(path)
        if meta is None or meta.get("kind") != "auto" or meta.get("used_by"):
            continue
        try:
            created = time.mktime(time.strptime(meta["created"], "%Y-%m-%dT%H:%M:%S"))
        except (KeyError, ValueError):
            continue
        if created < cutoff:
            delete(dataset_root, path.name)
            purged.append(path.name)
    return purged


def recount(dataset_root: Path, snapshot_id: str, job: Job | None = None) -> dict:
    """What the snapshot holds on its own (files no other folder links: deleting it frees them) and what it shares
    with the dataset or other snapshots."""
    path = snapshot_dir(dataset_root, snapshot_id)
    files = [file for file in path.rglob("*") if file.is_file()]
    own = shared = 0
    if job:
        job.progress(0, len(files), "Counting…")
    for index, file in enumerate(files, 1):
        try:
            stat = file.stat()
        except OSError:
            continue
        if stat.st_nlink > 1:
            shared += stat.st_size
        else:
            own += stat.st_size
        if job and index % 500 == 0:
            job.progress(index, len(files))

    def change(meta: dict) -> None:
        meta.update(own_bytes=own, shared_bytes=shared, bytes=own + shared, counted=time.strftime("%Y-%m-%dT%H:%M:%S"))

    return _update(dataset_root, snapshot_id, change)


# ---------- restoring ----------


def _dataset_path(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def restore(dataset: Dataset, snapshot_id: str, job: Job | None = None) -> dict:
    """Make the dataset as it was in the snapshot: images added since go to the trash, removed ones are linked back,
    moved ones go back to their split, labels and data.yaml get the snapshot's content. Like an import it is not
    on the undo stack (the history starts over): the state before is kept as an automatic snapshot first."""
    root = dataset.root
    meta = get(root, snapshot_id)
    if meta.get("category") != dataset.category:
        raise DatasetError("This snapshot is of another kind of dataset")
    if meta.get("rebalance"):
        raise DatasetError("This snapshot is a proposed split: apply the split from the rebalance dialog instead")
    rows = manifest(root, snapshot_id)
    source = snapshot_dir(root, snapshot_id)
    dataset.check_writable(structure=True)
    before = take(dataset, f"Before restoring '{meta['name']}'", kind="auto", job=job)
    if job:
        job.progress(0, len(rows), "Restoring…")
    counts = {"trashed": 0, "restored": 0, "labels": 0, "skipped": []}
    with dataset._lock:
        dataset.check_writable(structure=True)
        wanted = {os.path.normcase(str(_dataset_path(root, row["path"]))): row for row in rows}
        kept_labels = {os.path.normcase(str(_dataset_path(root, row["label_path"]))) for row in rows if row.get("label_path")}
        taken: set[Path] = set()
        for record in list(dataset.images.values()):
            if os.path.normcase(str(record.path)) in wanted:
                continue
            files = [record.path]
            if record.label_path != record.path and dataset._mtimes.get(record.label_path) is not None and os.path.normcase(str(record.label_path)) not in kept_labels:
                files.append(record.label_path)
            for path in files:
                target = dataset._trash_path(path, taken)
                taken.add(target)
                dataset._move_file(path, target)
            counts["trashed"] += 1
        yaml_text = source / WORKBENCH_DIR / SOURCE_YAML
        if yaml_text.is_file() and dataset.yaml_path.is_file():
            text = yaml_text.read_text(encoding="utf-8")
            if dataset.yaml_path.read_text(encoding="utf-8") != text:
                dataset._backup(dataset.yaml_path, dataset.yaml_path.read_text(encoding="utf-8"))
                dataset._atomic_write(dataset.yaml_path, text)
        for done, row in enumerate(rows, 1):
            image = _dataset_path(root, row["path"])
            if not image.exists():
                image.parent.mkdir(parents=True, exist_ok=True)
                if _place(source / row["file"], image) == "missing":
                    counts["skipped"].append(row["path"])
                    continue
                counts["restored"] += 1
            if row.get("label_path"):
                label = _dataset_path(root, row["label_path"])
                content = (source / row["label"]).read_text(encoding="utf-8")
                current = dataset._digests.get(label)
                if current is None or current.hex() != row.get("digest") or not label.exists():
                    label.parent.mkdir(parents=True, exist_ok=True)
                    dataset._atomic_write(label, content)
                    counts["labels"] += 1
            if job and done % 200 == 0:
                job.progress(done, len(rows))
        by_path = {os.path.normcase(str(record.path)): record for record in dataset.images.values()}
        for row in rows:  # a label the snapshot's image did not have goes to the trash
            if row.get("label_path") or dataset.category == "classification":
                continue
            record = by_path.get(os.path.normcase(str(_dataset_path(root, row["path"]))))
            if record is not None and record.label_path != record.path and record.label_path.exists():
                target = dataset._trash_path(record.label_path, taken)
                taken.add(target)
                dataset._move_file(record.label_path, target)
                counts["labels"] += 1
        dataset.clear_history()
    rescan = dataset.rescan()  # outside the lock: its size probes take it in their threads
    log.info("snapshot %s restored into %s: %s", snapshot_id, root.name, counts)
    return {"name": meta["name"], "images": meta["images"], "before": {"id": before["id"], "name": before["name"]}, **{key: value for key, value in counts.items() if key != "skipped"}, "skipped": counts["skipped"][:50], "added": rescan.get("added"), "removed": rescan.get("removed")}


def copy_into(dataset_root: Path, snapshot_id: str, target: Path) -> dict:
    """The snapshot's files (not its bookkeeping) linked, or copied, into a new folder: a dataset of their own."""
    source = snapshot_dir(dataset_root, snapshot_id)
    counts = {"linked": 0, "copied": 0, "missing": 0}
    for file in sorted(source.rglob("*")):
        relative = file.relative_to(source)
        if relative.parts[0] == WORKBENCH_DIR or not file.is_file():
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if file.suffix.lower() in IMAGE_EXTENSIONS:
            counts[_place(file, destination)] += 1
        else:
            shutil.copy2(file, destination)
    return counts


# ---------- using a snapshot ----------


def open_dataset(dataset_root: Path, snapshot_id: str, category: str) -> Dataset:
    """The snapshot as a read-only dataset (nothing is probed or written)."""
    path = snapshot_dir(dataset_root, snapshot_id)
    yaml_path = path / "data.yaml"
    return Dataset(yaml_path if yaml_path.is_file() else path, category, allow_empty=True, read_only=True)


def download_entries(dataset_root: Path, snapshot_id: str, kind: download.Kind) -> list[download.Entry]:
    """The snapshot's files for a zip, as the dataset's own download lays them out."""
    path = snapshot_dir(dataset_root, snapshot_id)
    meta = read_meta(path) or {}
    if kind == "labels" and meta.get("category") == "classification":
        raise DatasetError("A classification snapshot's labels are its class folders: download it with the images")
    entries = []
    for file in sorted(path.rglob("*")):
        relative = file.relative_to(path)
        if relative.parts[0] == WORKBENCH_DIR or not file.is_file():
            continue
        image = file.suffix.lower() in IMAGE_EXTENSIONS
        if kind == "labels" and image:
            continue
        entries.append(download.Entry(relative.as_posix(), file, compress=not image or file.suffix.lower() in (".bmp", ".tif", ".tiff")))
    return entries
