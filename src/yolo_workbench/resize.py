"""Resize images to fit within a size, keeping their aspect ratio: on import (after the crop) and for the images
already in a dataset (the Dataset tab).

YOLO coordinates are relative to the image, so labels stay as they are. Images are resized upright (EXIF
orientation applied) and written without an orientation tag, in their own format (imaging.encode). Resizing a
dataset's images is one undoable transaction (Dataset.replace_images): the new versions are written into the
trash first, without holding the dataset, then swapped in; the originals wait in the trash.
"""

from __future__ import annotations

import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

from . import imaging
from .dataset import WORKBENCH_DIR, Dataset, DatasetError
from .fsutil import ensure_free_space
from .jobs import Job
from .models import ImageRecord
from .storage import VERSION_TRASH

MAX_SIDE = 16384  # the largest width or height asked for (a typo must not ask for a gigapixel image)
WORKERS = min(8, os.cpu_count() or 1)  # Pillow decodes, resizes and encodes outside the GIL
BATCH = 4 * WORKERS  # images resized between progress reports (and cancel checks)
MAX_FAILURES = 20  # unreadable images named in the result


def resize_options(raw: dict | None) -> dict | None:
    """Validate {mode: none|fit, width, height, enlarge}; None means no resize."""
    if not raw or raw.get("mode", "none") == "none":
        return None
    if raw.get("mode") != "fit":
        raise DatasetError("Resize mode must be none or fit")
    try:
        width, height = int(raw.get("width") or 0), int(raw.get("height") or 0)
    except (TypeError, ValueError) as exc:
        raise DatasetError("Resize width and height must be whole numbers") from exc
    if not (0 < width <= MAX_SIDE and 0 < height <= MAX_SIDE):
        raise DatasetError(f"Resize width and height must be between 1 and {MAX_SIDE} px")
    return {"mode": "fit", "width": width, "height": height, "enlarge": bool(raw.get("enlarge"))}


def fit_size(width: int, height: int, options: dict) -> tuple[int, int] | None:
    """The size a width×height image gets to fit within the options' width×height with its aspect ratio kept;
    None when it stays as it is (it fits already and enlarging is off). static/manage.js `fitSize` mirrors this."""
    if width <= 0 or height <= 0:
        return None
    scale = min(options["width"] / width, options["height"] / height)
    if scale >= 1 and not options["enlarge"]:
        return None
    size = (max(1, int(width * scale + 0.5)), max(1, int(height * scale + 0.5)))
    return None if size == (width, height) else size


def scaled(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """`image` resized to `size` with Lanczos (palette and bilevel images first get a mode it can blend in)."""
    if image.mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
    elif image.mode == "1":
        image = image.convert("L")
    return image.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)


# ---------- a dataset's images ----------


def _candidates(dataset: Dataset, options: dict, split: str) -> tuple[int, list[ImageRecord]]:
    """How many readable images `split` ("all": every split) has, and those that may need resizing by the sizes
    the dataset knows (one not probed yet is opened to tell)."""
    with dataset._lock:
        records = [record for record in dataset.images.values() if split in ("all", record.split) and record.probe_error is None]
    return len(records), [record for record in records if not record.width or fit_size(record.width, record.height, options)]


def summary(dataset: Dataset, options: dict, split: str = "all") -> dict:
    """What resizing would do, by the sizes the dataset knows: {"images", "resize", "unknown" (sizes not read
    yet), "bytes" (of the images to resize)}."""
    total, pending = _candidates(dataset, options, split)
    known = [record for record in pending if record.width]
    return {"images": total, "resize": len(known), "unknown": len(pending) - len(known), "bytes": sum(record.image_size or 0 for record in known)}


def _stage(path: Path, target: Path, options: dict) -> dict | str | None:
    """Write the resized version of the image at `path` to `target`: {"file", "before", "after"} with (width,
    height, bytes) sizes; None when it fits already; the error text when it cannot be read."""
    try:
        with open(path, "rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            with Image.open(stream) as source:
                original = imaging.upright_size(source)
                fitted = fit_size(*original, options)
                if fitted is None:
                    return None
                data = imaging.encode(scaled(ImageOps.exif_transpose(source), fitted), source, path.suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return f"{path.name}: {exc}"
    return {"file": target, "before": (*original, size), "after": (*fitted, len(data))}


def resize_images(dataset: Dataset, options: dict, split: str = "all", job: Job | None = None) -> dict:
    """Resize the images of `split` that do not fit, as one undoable change (see the module doc). Cancelled, it
    changes nothing."""
    dataset.check_writable()
    total, pending = _candidates(dataset, options, split)
    ensure_free_space(dataset.root, sum(record.image_size or 0 for record in pending), "resizing the images")
    # this run's own folder: an undone resize's versions wait in the trash for a redo, and must stay as they are
    staging = dataset.root / WORKBENCH_DIR / "trash" / dataset.session_id / VERSION_TRASH / secrets.token_hex(4)
    versions: list[dict] = []
    failed: list[str] = []
    replaced: set[str] = set()
    if job:
        job.progress(0, len(pending), "Resizing…")
    try:
        with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="resize") as pool:
            for start in range(0, len(pending), BATCH):
                batch = pending[start : start + BATCH]
                # named by record id: images outside the dataset folder may share a name
                results = pool.map(lambda record: _stage(record.path, staging / f"{record.id}{record.path.suffix}", options), batch)
                for record, result in zip(batch, results):
                    if isinstance(result, dict):
                        versions.append({"image_id": record.id, "path": record.path, **result})
                    elif result is not None:
                        failed.append(result)
                if job:
                    job.progress(start + len(batch), len(pending))
        if job:
            job.progress(message="Replacing the images…")
        replaced = set(dataset.replace_images(versions)["replaced"]) if versions else set()
    finally:
        for version in versions:
            if version["image_id"] not in replaced:
                version["file"].unlink(missing_ok=True)
        _remove_empty(staging)
    done = [version for version in versions if version["image_id"] in replaced]
    return {
        "images": total,
        "resized": len(done),
        "unchanged": total - len(pending) + (len(pending) - len(versions) - len(failed)),
        "skipped": len(versions) - len(done),  # moved or removed meanwhile
        "failed": len(failed),
        "errors": failed[:MAX_FAILURES],
        "bytes_before": sum(version["before"][2] for version in done),
        "bytes_after": sum(version["after"][2] for version in done),
    }


def _remove_empty(staging: Path) -> None:
    """The run's folder (and the trash folders above it) when nothing of it is kept."""
    for folder in (staging, staging.parent, staging.parent.parent):
        try:
            folder.rmdir()
        except OSError:
            return
