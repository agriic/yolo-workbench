"""Download a dataset as a zip: its labels only, or its images with their labels.

The archive keeps the dataset's layout: every file at its path relative to the dataset root, beside a
data.yaml that resolves next to itself (no `path:`), so the labels can be laid over a copy of the images
and the full download trains as it is. Files outside the root go under `external/`, and a split that
names such files or `.txt` lists gets a `<split>.txt` list of what was exported.

The zip is written while it is sent: nothing is staged on disk, and a dataset of any size streams in
chunks (images stored as they are, text deflated; ZIP64 past 4 GB).
"""

from __future__ import annotations

import contextlib
import io
import os
import time
import zipfile
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from .dataset import IMAGE_EXTENSIONS, Dataset, DatasetError
from .yamledit import remove_key, write_key

Kind = Literal["labels", "full"]
CHUNK = 1 << 20
SPLITS = ("train", "val", "test")
# files opened ahead of the one being written: on a network drive every open waits on a round trip, so one
# file after another would take a round trip per label; files up to SMALL bytes are read whole meanwhile
READ_AHEAD = 16
SMALL = 1 << 20


@dataclass
class Entry:
    name: str  # path inside the archive (POSIX)
    path: Path | None = None  # file to copy, or None for `text`
    text: str | None = None
    compress: bool = True


class _Sink(io.RawIOBase):
    """A write-only, non-seekable stream the zip is written into; `drain()` hands out what was written."""

    def __init__(self):
        self._buffer = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        self._buffer += data
        return len(data)

    def pending(self) -> int:
        return len(self._buffer)

    def drain(self) -> bytes:
        data = bytes(self._buffer)
        self._buffer.clear()
        return data


class _Names:
    """Archive names handed out so far; a taken one gets `_2`, `_3`, … before its suffix."""

    def __init__(self):
        self.taken: set[str] = set()

    def claim(self, name: str) -> str:
        candidate, index = name, 2
        stem, dot, suffix = name.rpartition(".") if "." in name.rsplit("/", 1)[-1] else (name, "", "")
        while candidate.casefold() in self.taken:
            candidate = f"{stem}_{index}{dot}{suffix}"
            index += 1
        self.taken.add(candidate.casefold())
        return candidate


def plan(dataset: Dataset, kind: Kind) -> list[Entry]:
    """The archive's files, from a snapshot of the dataset's records (read under its lock)."""
    classification = dataset.category == "classification"
    if kind == "labels" and classification:
        raise DatasetError("A classification dataset's labels are its class folders: download it with the images")
    with dataset._lock:
        root = dataset.root
        records = sorted(dataset.images.values(), key=lambda record: str(record.path))
        split_folders = {split: list(folders) for split, folders in dataset._split_folders.items()}
        sources = dict(dataset._split_sources)
        folder_of = {record.id: dataset._folder_of(record) for record in records}
        # labels the dataset knows exist (an is_file() per label is a round trip each on a network drive)
        labelled = {record.label_path for record in records if dataset._mtimes.get(record.label_path) is not None}
        yaml_text = dataset.yaml_path.read_text(encoding="utf-8") if dataset.yaml_path.is_file() else None
        label_root_for = dataset._label_root_for

    names = _Names()
    names.taken.add("data.yaml")
    # split folders outside the root are given a place under external/ (labels in the matching labels/ folder)
    external: dict[Path, tuple[str, str]] = {}
    for split, folders in split_folders.items():
        outside = [folder for folder in folders if not folder.is_relative_to(root)]
        for index, folder in enumerate(outside):
            base = split if len(outside) == 1 else f"{split}_{index + 1}"
            external[folder] = (f"external/{base}", f"external/{base}") if classification else (f"external/images/{base}", f"external/labels/{base}")

    entries: list[Entry] = []
    exported: dict[str, list[str]] = {split: [] for split in SPLITS}  # image names per split (for .txt lists)
    labels_seen: set[Path] = set()  # a.jpg and a.png share a.txt
    for record in records:
        folder = folder_of[record.id]
        label_name = None
        if record.path.is_relative_to(root):
            image_name = names.claim(record.path.relative_to(root).as_posix())
            if record.label_path.is_relative_to(root):
                label_name = record.label_path.relative_to(root).as_posix()
        elif folder in external:
            image_base, label_base = external[folder]
            image_name = names.claim(f"{image_base}/{record.path.relative_to(folder).as_posix()}")
            label_root = label_root_for(folder)
            if record.label_path.is_relative_to(label_root):
                label_name = f"{label_base}/{record.label_path.relative_to(label_root).as_posix()}"
        else:
            image_name = names.claim(f"external/images/{record.split}/{record.path.name}")
        if label_name is None:  # labels outside the root: beside the image, as Ultralytics looks for them
            label_name = image_name.replace("external/images/", "external/labels/", 1).rsplit(".", 1)[0] + ".txt"
        exported.setdefault(record.split, []).append(image_name)
        if kind == "full":
            entries.append(Entry(image_name, record.path, compress=record.path.suffix.lower() in (".bmp", ".tif", ".tiff")))
        if not classification and record.label_path not in labels_seen and record.label_path in labelled:
            labels_seen.add(record.label_path)
            entries.append(Entry(names.claim(label_name), record.label_path))

    if yaml_text is not None:
        entries.insert(0, Entry("data.yaml", text=_data_yaml(yaml_text, root, split_folders, sources, external, exported, names, entries)))
    return entries


def _data_yaml(text: str, root: Path, split_folders: dict, sources: dict, external: dict, exported: dict, names: _Names, entries: list[Entry]) -> str:
    """The dataset's YAML without `path:` (paths resolve next to the file); a split naming files outside
    the root or `.txt` lists is rewritten to the exported folders, or to a list of the exported images."""
    text = remove_key(text, "path")
    for split in SPLITS:
        value = sources.get(split)
        if not value:
            continue
        items = value if isinstance(value, list) else [value]
        if all(_portable(item, root) for item in items):
            continue
        folders = split_folders.get(split, [])
        listed = any(Path(str(item)).suffix.lower() == ".txt" or Path(str(item)).suffix.lower() in IMAGE_EXTENSIONS for item in items)
        if folders and not listed:
            labels = [folder.relative_to(root).as_posix() if folder.is_relative_to(root) else external[folder][0] for folder in folders]
            text = write_key(text, split, labels[0] if len(labels) == 1 else labels)
        else:
            list_name = names.claim(f"{split}.txt")
            entries.append(Entry(list_name, text="".join(f"./{name}\n" for name in exported.get(split, []))))
            text = write_key(text, split, list_name)
    return text


def _portable(item, root: Path) -> bool:
    """A split entry that still means the same thing next to the exported data.yaml: a relative folder inside the root."""
    path = Path(str(item))
    if path.is_absolute() or path.suffix.lower() == ".txt" or path.suffix.lower() in IMAGE_EXTENSIONS:
        return False
    return (root / path).resolve().is_relative_to(root)


def _fetch(path: Path) -> tuple[os.stat_result, bytes | None, io.BufferedReader | None] | None:
    """A file's (stat, content) when small, else (stat, its open stream); None when it is gone."""
    try:
        source = path.open("rb")
    except OSError:
        return None
    try:
        stat = os.fstat(source.fileno())
        if stat.st_size <= SMALL:
            with source:
                return stat, source.read(), None
    except OSError:
        source.close()
        return None
    return stat, None, source


def stream(entries: list[Entry]) -> Iterator[bytes]:
    """Write the entries into a zip and yield it in chunks. A file that vanished meanwhile is left out."""
    sink = _Sink()
    queue = iter(entries)
    ahead: deque[tuple[Entry, Future | None]] = deque()
    with ThreadPoolExecutor(max_workers=READ_AHEAD, thread_name_prefix="download") as pool:

        def refill() -> None:
            while len(ahead) < 2 * READ_AHEAD and (entry := next(queue, None)) is not None:
                ahead.append((entry, pool.submit(_fetch, entry.path) if entry.path is not None else None))

        try:
            with zipfile.ZipFile(sink, "w", allowZip64=True) as archive:
                refill()
                while ahead:
                    entry, future = ahead.popleft()
                    refill()
                    if future is None:
                        info = zipfile.ZipInfo(entry.name, time.localtime()[:6])
                        info.compress_type = zipfile.ZIP_DEFLATED
                        archive.writestr(info, entry.text.encode("utf-8"))
                    elif (fetched := future.result()) is not None:
                        stat, data, source = fetched
                        with source or contextlib.nullcontext():
                            info = zipfile.ZipInfo(entry.name, time.localtime(max(stat.st_mtime, 315532800))[:6])  # zip dates start in 1980
                            info.compress_type = zipfile.ZIP_DEFLATED if entry.compress else zipfile.ZIP_STORED
                            info.file_size = len(data) if data is not None else stat.st_size  # decides ZIP64 for this entry up front
                            with archive.open(info, "w") as target:
                                if data is not None:
                                    target.write(data)
                                while source is not None and (chunk := source.read(CHUNK)):
                                    target.write(chunk)
                                    if sink.pending() >= CHUNK:
                                        yield sink.drain()
                    if sink.pending() >= CHUNK:
                        yield sink.drain()
            yield sink.drain()
        finally:  # stopped midway (the client went away): close what was opened ahead
            for _, future in ahead:
                fetched = future.result() if future is not None else None
                if fetched is not None and fetched[2] is not None:
                    fetched[2].close()


def archive_name(name: str, kind: Kind) -> str:
    return f"{name}-{'labels' if kind == 'labels' else 'dataset'}.zip"
