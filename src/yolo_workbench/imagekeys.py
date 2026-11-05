"""What the rebalance's keep-together rules need to know of an image's pixels and EXIF: a near-duplicate fingerprint
(a 64-bit difference hash of the upright image) and when and with which camera it was taken.

Reading them decodes every image once, so it is a job of its own (`scan`), cached in
`.yolo-workbench/imagekeys.json` by path with the file's time and size (the dataset knows those: no stat per
image); a later scan reads only images that are new or changed.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps, UnidentifiedImageError

from .dataset import WORKBENCH_DIR, Dataset
from .jobs import Job

FILE = "imagekeys.json"
VERSION = 1
WORKERS = min(8, os.cpu_count() or 1)
BATCH = 4 * WORKERS
DATE_TAGS = (0x9003, 0x0132)  # DateTimeOriginal, DateTime
MODEL_TAG = 0x0110
EXIF_IFD = 0x8769


def _path(dataset: Dataset):
    return dataset.root / WORKBENCH_DIR / FILE


def _read(dataset: Dataset) -> dict:
    try:
        data = json.loads(_path(dataset).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data.get("images", {}) if isinstance(data, dict) and data.get("version") == VERSION else {}


def load(dataset: Dataset) -> dict:
    """path -> {dhash, time, camera} of the images whose file is still the one scanned."""
    cached = _read(dataset)
    with dataset._lock:
        stamps = {str(record.path): (record.image_mtime, record.image_size) for record in dataset.images.values()}
    return {path: entry for path, entry in cached.items() if path in stamps and (entry.get("mtime_ns"), entry.get("size")) == stamps[path]}


def status(dataset: Dataset) -> dict:
    with dataset._lock:
        total = len(dataset.images)
    return {"scanned": len(load(dataset)), "images": total}


def dhash(image: Image.Image) -> str:
    """64-bit difference hash: a 9×8 grey thumbnail, one bit per left/right brightness step."""
    small = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    pixels = small.tobytes()  # one byte per pixel, row by row
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{bits:016x}"


def _taken(image: Image.Image) -> tuple[float | None, str | None]:
    try:
        exif = image.getexif()
    except Exception:  # a broken EXIF block says nothing
        return None, None
    camera = exif.get(MODEL_TAG)
    stamp = None
    try:
        stamp = exif.get_ifd(EXIF_IFD).get(DATE_TAGS[0])
    except Exception:
        pass
    stamp = stamp or exif.get(DATE_TAGS[1])
    when = None
    if isinstance(stamp, str):
        try:
            when = time.mktime(time.strptime(stamp.strip()[:19], "%Y:%m:%d %H:%M:%S"))
        except ValueError:
            when = None
    return when, str(camera).strip() if camera else None


def _keys(path) -> dict | None:
    try:
        with Image.open(path) as image:
            when, camera = _taken(image)
            image.draft("L", (64, 64))  # JPEG: decode at a fraction of the size
            return {"dhash": dhash(ImageOps.exif_transpose(image)), "time": when, "camera": camera}
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError):
        return None


def scan(dataset: Dataset, job: Job | None = None) -> dict:
    """Fingerprint the images not scanned yet (or changed since); saved as it goes."""
    cached = load(dataset)
    with dataset._lock:
        pending = [(str(record.path), record.image_mtime, record.image_size) for record in dataset.images.values() if str(record.path) not in cached]
    failed = 0
    if job:
        job.progress(0, len(pending), "Reading the images…")
    last_save = time.monotonic()
    with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="imagekeys") as pool:
        for start in range(0, len(pending), BATCH):
            batch = pending[start : start + BATCH]
            for (path, mtime, size), keys in zip(batch, pool.map(lambda item: _keys(item[0]), batch)):
                if keys is None:
                    failed += 1
                    continue
                cached[path] = {**keys, "mtime_ns": mtime, "size": size}
            if job:
                job.progress(start + len(batch), len(pending))
            if time.monotonic() - last_save > 10:
                _save(dataset, cached)
                last_save = time.monotonic()
    _save(dataset, cached)
    return {"read": len(pending) - failed, "failed": failed, **status(dataset)}


def _save(dataset: Dataset, cached: dict) -> None:
    path = _path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset._atomic_write(path, json.dumps({"version": VERSION, "images": cached}))
