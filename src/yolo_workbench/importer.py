"""Import images (and YOLO or COCO labels) from a folder into a dataset.

Browser uploads land in <dataset>/.yolo-workbench/staging/<id>/ first and are then
imported exactly like a local folder. Files are always copied; labels are rewritten
with the chosen class mapping. A source with COCO annotation files has its objects
converted to YOLO lines (its .txt files are not used). An optional crop (centered, or
a region) re-encodes the images and clips the labels to it. Imports are not on the undo
stack (delete the images to reverse one) but each is logged to .yolo-workbench/imports.jsonl.
An optional resize (resize.py) then scales the images to fit within a size; the labels need no change.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import posixpath
import random
import re
import secrets
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import yaml
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from . import classes, fsscan, imaging, resize
from .dataset import IMAGE_EXTENSIONS, READ_WORKERS, WORKBENCH_DIR, Dataset, DatasetError
from .fsutil import ensure_free_space, replace
from .jobs import Job
from .runtime import RUNTIME

SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val", "test": "test", "testing": "test"}
UPLOAD_EXTENSIONS = IMAGE_EXTENSIONS | {".txt", ".yaml", ".yml", ".json"}
STAGING_PATTERN = re.compile(r"^[a-f0-9]{16}$")
UNASSIGNED = ""  # classification images that are not inside a class folder
SAMPLE_COUNT = 24  # images offered for the crop preview, spread over the whole import
MIN_VISIBLE = 0.1  # a cropped object keeping less of its area than this is dropped (Ultralytics' area_thr)


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


class UploadTooLarge(DatasetError):
    """An upload over the configured limits (HTTP 413)."""


_usage_lock = threading.Lock()
_usage: dict[str, list[int]] = {}  # staging dir -> [files, bytes] received so far


def _staging_usage(root: Path) -> list[int]:
    """Files and bytes already in a staging dir (counted once, then kept up to date). Caller holds _usage_lock."""
    key = str(root)
    if key not in _usage:
        files = [path for path in root.rglob("*") if path.is_file() and not path.name.startswith(".")]
        _usage[key] = [len(files), sum(path.stat().st_size for path in files)]
    return _usage[key]


def staged_target(dataset: Dataset, staging_id: str, relative: str) -> tuple[Path, Path, str]:
    """Where an uploaded file goes: (staging root, target path, clean relative path). Checks the upload's file
    count and free disk space before anything is received."""
    root = staging_dir(dataset, staging_id)
    clean = safe_relative(relative)
    target = root.joinpath(*clean.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise DatasetError(f"Invalid upload path: {relative!r}")
    with _usage_lock:
        files, _ = _staging_usage(root)
    if RUNTIME.max_upload_files and files >= RUNTIME.max_upload_files and not target.exists():
        raise UploadTooLarge(f"This upload already has {files} files, the most one import takes; import them first")
    ensure_free_space(root, what="the upload")
    return root, target, str(PurePosixPath(*clean.parts))


def upload_budget(root: Path) -> int:
    """How many more bytes this upload may receive in one file (per-file limit and what is left of the total)."""
    limit = RUNTIME.max_upload_file_bytes
    if RUNTIME.max_upload_total_bytes:
        with _usage_lock:
            _, used = _staging_usage(root)
        limit = min(limit, max(0, RUNTIME.max_upload_total_bytes - used))
    return limit


def partial_path(target: Path) -> Path:
    """The temporary file an upload streams into (dot-prefixed, so a scan skips it)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.with_name(f".{target.name}.{secrets.token_hex(4)}.part")


def commit_staged_file(root: Path, partial: Path, target: Path, size: int, relative: str) -> dict:
    """Move a completely received upload into place and count it."""
    previous = target.stat().st_size if target.exists() else None
    replace(partial, target)
    with _usage_lock:
        usage = _staging_usage(root)
        if previous is None:
            usage[0] += 1
            usage[1] += size
        else:
            usage[1] += size - previous
    _touch(root)
    return {"path": relative, "size": size}


def write_staged_file(dataset: Dataset, staging_id: str, relative: str, content: bytes) -> dict:
    root, target, clean = staged_target(dataset, staging_id, relative)
    if len(content) > upload_budget(root):
        raise UploadTooLarge(f"{target.name} is larger than this server accepts")
    partial = partial_path(target)
    try:
        partial.write_bytes(content)
        return commit_staged_file(root, partial, target, len(content), clean)
    finally:
        partial.unlink(missing_ok=True)


def _touch(path: Path) -> None:
    """Mark a staging dir as in use (cleanup goes by its modification time)."""
    try:
        os.utime(path)
    except OSError:
        pass


def discard_staging(dataset: Dataset, staging_id: str) -> None:
    root = staging_dir(dataset, staging_id)
    shutil.rmtree(root, ignore_errors=True)
    with _usage_lock:
        _usage.pop(str(root), None)
    _forget_coco(root)


def clean_staging(staging: Path, max_age: float) -> list[str]:
    """Remove upload dirs untouched for `max_age` seconds (every received file touches its dir)."""
    if not staging.is_dir():
        return []
    removed = []
    now = time.time()
    for entry in staging.iterdir():
        try:
            if now - entry.stat().st_mtime > max_age:
                shutil.rmtree(entry, ignore_errors=True)
                removed.append(entry.name)
                with _usage_lock:
                    _usage.pop(str(entry), None)
                _forget_coco(entry)
        except OSError:
            pass
    return removed


# ---------- scanning ----------


def resolve_source(dataset: Dataset, path: str | None, staging_id: str | None) -> Path:
    if staging_id:
        return staging_dir(dataset, staging_id)
    if not path:
        raise DatasetError("Choose a folder to import from")
    if not RUNTIME.allow_path_import:
        raise DatasetError("Importing from a folder on the server is turned off; upload the files from your browser instead")
    source = Path(path).expanduser()
    if not source.is_dir():
        raise DatasetError(f"Folder not found: {source}")
    source = source.resolve()
    if source == dataset.root or dataset.root.is_relative_to(source) or (source.is_relative_to(dataset.root) and WORKBENCH_DIR not in source.relative_to(dataset.root).parts):
        raise DatasetError("Import from a folder outside this dataset")
    return source


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


# ---------- COCO ----------
#
# A COCO annotation file (Roboflow's <split>/_annotations.coco.json, COCO's annotations/instances_<split>.json, …)
# lists images by file name and their objects in pixels. Its categories go through the class mapping like named
# YOLO classes (keyed by category id), and its objects become YOLO lines when the images are copied: boxes for
# detection, one polygon per object for segmentation. Crowd regions are left out, as Ultralytics' converter does.


class _CocoFile:
    """One COCO annotation file: its categories (and the mapping key each one got), and its images with their objects."""

    def __init__(self, relative: PurePosixPath, content: dict):
        self.relative = relative
        self.split = _detected_split(relative) or next((SPLIT_ALIASES[token] for token in re.split(r"[^a-z]+", relative.stem.casefold()) if token in SPLIT_ALIASES), None)
        categories = [category for category in content["categories"] if isinstance(category, dict) and _int(category.get("id")) is not None]
        self.categories = {_int(category["id"]): str(category.get("name") or category["id"]) for category in categories}
        self.supercategories = {str(category["supercategory"]) for category in categories if category.get("supercategory")}
        self.keypoints = bool(categories) and all(category.get("keypoints") for category in categories)
        self.keys: dict[int, str] = {}  # category id -> mapping key (_coco_keys)
        objects: dict[str, list[dict]] = {}
        for annotation in content["annotations"]:
            if isinstance(annotation, dict) and isinstance(annotation.get("image_id"), (int, str)):
                objects.setdefault(str(annotation["image_id"]), []).append(annotation)
        self.images = [(entry, objects.get(str(entry.get("id")), [])) for entry in content["images"] if isinstance(entry, dict) and entry.get("file_name")]


_coco_lock = threading.Lock()
_coco_cache: dict | None = None  # the last scanned source's COCO files, so the scan, the crop previews and the import parse them once


def _forget_coco(source: Path) -> None:
    global _coco_cache
    with _coco_lock:
        if _coco_cache is not None and _coco_cache["source"] == str(source):
            _coco_cache = None


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dimension(value) -> float:
    """A positive image width or height from a COCO entry; 0 when missing or invalid."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if 0 < number < float("inf") else 0.0


def _read_coco(path: str) -> dict | None:
    """The content of a COCO annotation file; None for any other JSON (most are never parsed)."""
    try:
        data = Path(path).read_bytes()
        if not all(key in data for key in (b'"images"', b'"annotations"', b'"categories"')):
            return None
        content = json.loads(data)
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(content, dict) or not all(isinstance(content.get(key), list) for key in ("images", "annotations", "categories")):
        return None
    if any(isinstance(annotation, dict) and "segments_info" in annotation for annotation in content["annotations"][:100]):
        return None  # panoptic segmentation: PNG masks, not objects
    return content


def _coco_keys(files: list[_CocoFile]) -> None:
    """Give each category its mapping key: its id, shared by every file with a class of that name; an id another
    file already gave to another name becomes "<id>:<file number>"."""
    by_name: dict[str, str] = {}
    for index, coco in enumerate(files):
        for category_id, name in sorted(coco.categories.items()):
            if name not in by_name:
                by_name[name] = str(category_id) if str(category_id) not in by_name.values() else f"{category_id}:{index}"
            coco.keys[category_id] = by_name[name]


def _coco_image(file_name: str, coco: _CocoFile, exact: dict, folded: dict, by_name: dict) -> PurePosixPath | None:
    """The source image a COCO entry names: next to the annotation file, beside its folder (annotations/ next to
    the image folders) or at the source root; else the one image with that name (ties go to the file's split,
    then to images below its folder)."""
    parts = [part for part in file_name.replace("\\", "/").split("/") if part not in ("", ".")]
    if not parts:
        return None
    name = "/".join(parts)
    folder = coco.relative.parent
    for base in (folder, folder.parent, PurePosixPath()):
        candidate = posixpath.normpath(posixpath.join(str(base), name))
        found = exact.get(candidate) or folded.get(candidate.casefold())
        if found is not None:
            return found
    candidates = by_name.get(parts[-1].casefold(), [])
    if len(parts) > 1:  # an absolute or differently rooted path: the longer one ends with the other
        tail = [path for path in candidates if f"/{path}".casefold().endswith(f"/{name}".casefold()) or f"/{name}".casefold().endswith(f"/{path}".casefold())]
        candidates = tail or candidates
    if len(candidates) > 1 and coco.split:
        candidates = [path for path in candidates if _detected_split(path) == coco.split] or candidates
    if len(candidates) > 1 and folder != PurePosixPath():
        candidates = [path for path in candidates if path.is_relative_to(folder)] or candidates
    return candidates[0] if len(candidates) == 1 else None


def _coco_source(source: Path, listing: fsscan.Listing, found: list[str], images: list[PurePosixPath]) -> dict | None:
    """The source's COCO annotation files and which of its images each entry describes:
    {"files", "matched": {image: [(file, entry, objects)]}, "missing": entries whose image is not there}; None without any."""
    global _coco_cache
    paths = sorted(path for path in found if path.lower().endswith(".json") and not os.path.basename(path).startswith("."))
    stamps = tuple((path, listing.stat(path)) for path in paths)
    with _coco_lock:
        cached = _coco_cache if _coco_cache is not None and _coco_cache["source"] == str(source) and _coco_cache["stamps"] == stamps else None
    if cached is not None:
        files = cached["files"]
    else:
        contents = []
        if paths:
            with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
                contents = list(pool.map(_read_coco, paths))
        files =[_CocoFile(PurePosixPath(*Path(path).relative_to(source).parts), content) for path, content in zip(paths, contents) if content is not None]
        if any(not coco.keypoints for coco in files):
            files = [coco for coco in files if not coco.keypoints]  # COCO's person_keypoints_* repeat the people of instances_*
        _coco_keys(files)
    result = None
    if files:
        exact = {str(path): path for path in images}
        folded = {key.casefold(): path for key, path in exact.items()}
        by_name: dict[str, list[PurePosixPath]] = {}
        for path in images:
            by_name.setdefault(path.name.casefold(), []).append(path)
        matched: dict[PurePosixPath, list] = {}
        missing = 0
        for coco in files:
            for entry, objects in coco.images:
                path = _coco_image(str(entry["file_name"]), coco, exact, folded, by_name)
                if path is None:
                    missing += 1
                else:
                    matched.setdefault(path, []).append((coco, entry, objects))
        result = {"files": files, "matched": matched, "missing": missing}
    with _coco_lock:
        _coco_cache = {"source": str(source), "stamps": stamps, "files": files, "coco": result}
    return result


def _cached_coco(source: Path, category: str) -> dict | None:
    """The source's COCO annotations as its last scan found them (scanned again when another source came since)."""
    with _coco_lock:
        if _coco_cache is not None and _coco_cache["source"] == str(source):
            return _coco_cache["coco"]
    return _scan_items(source, category)[1]


def _has_shape(annotation: dict) -> bool:
    segmentation, box = annotation.get("segmentation"), annotation.get("bbox")
    return bool(segmentation) or (isinstance(box, list) and len(box) == 4)


def _rle_counts(text: str) -> list[int]:
    """The run lengths of a compressed COCO RLE string (pycocotools' rleFrString)."""
    counts: list[int] = []
    position = 0
    while position < len(text):
        value = shift = 0
        more = True
        while more and position < len(text):
            char = ord(text[position]) - 48
            value |= (char & 0x1F) << shift
            more = bool(char & 0x20)
            position += 1
            shift += 5
            if not more and char & 0x10:
                value |= -1 << shift
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts


def _rle_polygons(rle: dict) -> list[list[tuple[float, float]]]:
    """The outlines of a run-length encoded COCO mask, in pixels."""
    import numpy as np

    from .masks import mask_outlines

    try:
        height, width = (int(value) for value in rle["size"])
        counts = rle["counts"]
        counts = _rle_counts(counts) if isinstance(counts, str) else [int(count) for count in counts]
    except (KeyError, TypeError, ValueError):
        return []
    if height <= 0 or width <= 0:
        return []
    flat = np.zeros(height * width, dtype=np.uint8)
    position = 0
    for index, count in enumerate(counts):
        if index % 2:
            flat[position : position + max(0, count)] = 1
        position += max(0, count)
    mask = flat.reshape((height, width), order="F")  # COCO masks run down the columns
    return [[(x * width, y * height) for x, y in zip(outline[::2], outline[1::2])] for outline in mask_outlines(mask)]


def _coco_polygons(segmentation) -> list[list[tuple[float, float]]]:
    """A COCO segmentation's parts as pixel (x, y) lists: polygons ([[x, y, …], …], or one flat list) or an RLE mask."""
    if isinstance(segmentation, dict):
        return _rle_polygons(segmentation)
    if not isinstance(segmentation, list) or not segmentation:
        return []
    polygons = []
    for flat in segmentation if isinstance(segmentation[0], list) else [segmentation]:
        try:
            values = [float(value) for value in flat]
        except (TypeError, ValueError):
            continue
        if len(values) >= 6:
            polygons.append(list(zip(values[::2], values[1::2])))
    return polygons


def _coco_box(box, width: float, height: float) -> tuple[float, float, float, float] | None:
    """A COCO [x, y, w, h] box clipped to the image as (x1, y1, x2, y2); None when nothing of it is inside."""
    try:
        x, y, box_width, box_height = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(width, x + box_width), min(height, y + box_height)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _merge_polygons(polygons: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """One outline for an object in several parts (YOLO has one polygon per object, so Ultralytics' COCO converter
    joins them too): each part, turned the same way round as the first, is joined to the outline so far by a
    zero-width bridge between their closest points, so the area is the sum of the parts'."""

    def signed(points):
        return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))

    merged = list(polygons[0])
    turn = signed(merged) >= 0
    for part in polygons[1:]:
        part = part if (signed(part) >= 0) == turn else part[::-1]
        _, i, j = min(((ax - bx) ** 2 + (ay - by) ** 2, i, j) for i, (ax, ay) in enumerate(merged) for j, (bx, by) in enumerate(part))
        merged[i + 1 : i + 1] = part[j:] + part[:j] + [part[j], merged[i]]
    return merged


def _coco_shape(annotation: dict, width: float, height: float, category: str) -> list[float] | None:
    """A COCO object's normalized YOLO geometry: its box (cx cy w h) for detection; for segmentation one polygon
    (parts joined; an object with only a box becomes its rectangle). None when nothing usable is in the image."""
    if category == "detection":
        box = _coco_box(annotation.get("bbox"), width, height)
        polygons = _coco_polygons(annotation.get("segmentation")) if box is None else []
        if polygons:  # no usable box: the one around the outline
            xs, ys = [x for polygon in polygons for x, _ in polygon], [y for polygon in polygons for _, y in polygon]
            box = _coco_box([min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], width, height)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        return [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]
    polygons = _coco_polygons(annotation.get("segmentation"))
    if not polygons:
        box = _coco_box(annotation.get("bbox"), width, height)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        polygons = [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]
    points = [(min(1.0, max(0.0, x / width)), min(1.0, max(0.0, y / height))) for x, y in _merge_polygons(polygons)]
    return [value for point in points for value in point] if _area(points) > 0 else None


def _coco_label_text(pairs: list, mapping: dict | None, category: str, image: Path) -> tuple[str, int]:
    """The YOLO label text of one image's COCO objects, with classes mapped (None: left out; no mapping: every
    class as 0, for the preview), and how many objects had no usable shape. Crowd regions are left out."""
    lines, unusable, size = [], 0, None
    for coco, entry, objects in pairs:
        width, height = _dimension(entry.get("width")), _dimension(entry.get("height"))
        if not (width and height):  # the entry has no size: the image's own (upright, as labels refer to it)
            if size is None:
                try:
                    with Image.open(image) as opened:
                        size = imaging.upright_size(opened)
                except (OSError, UnidentifiedImageError):
                    size = (0, 0)
            width, height = size
        for annotation in objects:
            key = coco.keys.get(_int(annotation.get("category_id")))
            if annotation.get("iscrowd") or key is None:
                continue
            class_id = 0 if mapping is None else mapping.get(key)
            if class_id is None:
                continue
            values = _coco_shape(annotation, width, height, category) if width and height else None
            if values is None:
                unusable += 1
                continue
            lines.append(" ".join([str(class_id), *map(_number, values)]) + "\n")
    return "".join(lines), unusable


def _scan_items(source: Path, category: str) -> tuple[list[dict], dict | None]:
    """The source's images (with their labels) and, when it has COCO annotation files, those (`_coco_source`)."""
    # one listing per folder (with sizes): an upload staged on a network drive holds thousands of files
    listing = fsscan.Listing()
    found = listing.files(source, lambda path, name: name.startswith("."))  # hidden folders are left out
    images = sorted(Path(path) for path in found if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS and not os.path.basename(path).startswith("."))
    relatives = [image.relative_to(source) for image in images]
    coco = None if category == "classification" else _coco_source(source, listing, found, [PurePosixPath(*relative.parts) for relative in relatives])
    items = []
    for image, relative in zip(images, relatives):
        stamp = listing.stat(image)
        item = {"image": image, "relative": relative, "split": _detected_split(relative), "label": None, "coco": None, "class": None, "size": stamp[1] if stamp else None}
        if category == "classification":
            parent = image.parent.name
            inside_class = image.parent != source and SPLIT_ALIASES.get(parent.casefold()) is None and parent.casefold() != "images"
            item["class"] = parent if inside_class else UNASSIGNED
        elif coco is not None:
            item["coco"] = coco["matched"].get(PurePosixPath(*relative.parts))
            if item["coco"] and item["split"] is None:
                item["split"] = item["coco"][0][0].split
        else:
            item["label"] = next((candidate for candidate in _label_candidates(image, source) if listing.stat(candidate) is not None), None)
        items.append(item)
    return items, coco


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
    return _scan(dataset, source)[0]


def _scan(dataset: Dataset, source: Path) -> tuple[dict, list[dict]]:
    items, coco = _scan_items(source, dataset.category)
    by_name = {name.casefold(): class_id for class_id, name in dataset.names.items()}
    sample_count = min(SAMPLE_COUNT, len(items))
    split_counts: dict[str, int] = {}
    for item in items:
        if item["split"]:
            split_counts[item["split"]] = split_counts.get(item["split"], 0) + 1
    result = {
        "source": str(source),
        "images": len(items),
        "splits": split_counts,
        "target_splits": dataset.capabilities()["splits"],
        "format": "coco" if coco is not None else "yolo",
        "names_source": None,
        "coco": None,
        "labeled": 0,
        "classes": [],
        "samples": [str(PurePosixPath(*items[index * len(items) // sample_count]["relative"].parts)) for index in range(sample_count)],
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
        return result, items
    if coco is not None:
        _scan_coco(result, items, coco, by_name)
        return result, items
    names, names_source = _names_from_source(source)
    result["names_source"] = names_source
    counts = {}
    labels = [item["label"] for item in items if item["label"] is not None]
    result["labeled"] = len(labels)
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:  # read in parallel: a round trip each on a network drive
        for ids in pool.map(_label_ids, labels):
            for class_id in ids:
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
    return result, items


def _scan_coco(result: dict, items: list[dict], coco: dict, by_name: dict[str, int]) -> None:
    """The scan's labels and classes from COCO files: one class per category key, counted over the images found."""
    counts: dict[str, int] = {}
    skipped = {"crowd": 0, "unknown_class": 0, "no_shape": 0}
    labeled = [item for item in items if item["coco"] is not None]
    for item in labeled:
        for file, _, objects in item["coco"]:
            for annotation in objects:
                key = file.keys.get(_int(annotation.get("category_id")))
                reason = "crowd" if annotation.get("iscrowd") else "unknown_class" if key is None else None if _has_shape(annotation) else "no_shape"
                if reason:
                    skipped[reason] += 1
                else:
                    counts[key] = counts.get(key, 0) + 1
    names: dict[str, str] = {}
    supercategories: set[str] = set()
    for file in coco["files"]:
        supercategories |= file.supercategories
        for category_id, key in file.keys.items():
            names.setdefault(key, file.categories[category_id])
    result["labeled"] = len(labeled)
    result["coco"] = {"files": [str(file.relative) for file in coco["files"]], "missing_images": coco["missing"], "skipped": skipped}
    for key, name in names.items():
        # Roboflow lists the project itself as an unused category, the supercategory of the real ones: not a class
        parent = not counts.get(key) and name in supercategories
        suggested = None if parent else by_name.get(name.casefold(), "new")
        result["classes"].append({"key": key, "name": name, "count": counts.get(key, 0), "named": True, "suggested": suggested})


# ---------- cropping ----------
#
# Crops work on the upright image (EXIF orientation applied, as Ultralytics loads it), and
# the cropped file is written without an orientation tag, so pixels and labels agree everywhere.


def crop_options(raw: dict | None) -> dict | None:
    """Validate {mode: none|center|box, unit: px|percent, x, y, width, height}; None means no crop."""
    if not raw or raw.get("mode", "none") == "none":
        return None
    mode, unit = raw.get("mode"), raw.get("unit", "px")
    if mode not in ("center", "box"):
        raise DatasetError("Crop mode must be none, center, or box")
    if unit not in ("px", "percent"):
        raise DatasetError("Crop unit must be px or percent")
    try:
        crop = {"mode": mode, "unit": unit, **{key: float(raw.get(key) or 0) for key in ("x", "y", "width", "height")}}
    except (TypeError, ValueError) as exc:
        raise DatasetError("Crop values must be numbers") from exc
    if crop["width"] <= 0 or crop["height"] <= 0 or crop["x"] < 0 or crop["y"] < 0:
        raise DatasetError("Crop width and height must be positive and the offset not negative")
    if unit == "percent" and (crop["width"] > 100 or crop["height"] > 100 or crop["x"] >= 100 or crop["y"] >= 100):
        raise DatasetError("Crop percentages must be within 0–100")
    if mode == "center":
        crop["x"] = crop["y"] = 0.0
    return crop


def crop_box(width: int, height: int, crop: dict) -> tuple[int, int, int, int] | None:
    """The pixel box (left, top, right, bottom) `crop` cuts from a width×height image.

    Center crops shrink to the image where it is smaller; a region is clipped to the image and
    is None when it misses the image entirely. static/manage.js `cropBox` mirrors this.
    """
    scale_x, scale_y = (width / 100, height / 100) if crop["unit"] == "percent" else (1, 1)
    crop_width, crop_height = int(crop["width"] * scale_x), int(crop["height"] * scale_y)
    if crop["mode"] == "center":
        crop_width, crop_height = max(1, min(width, crop_width)), max(1, min(height, crop_height))
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        return left, top, left + crop_width, top + crop_height
    x, y = int(crop["x"] * scale_x), int(crop["y"] * scale_y)
    left, top, right, bottom = max(0, x), max(0, y), min(width, x + crop_width), min(height, y + crop_height)
    return (left, top, right, bottom) if right > left and bottom > top else None


def _clip_polygon(points: list[tuple[float, float]], box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    """Sutherland–Hodgman clipping of a polygon against an axis-aligned box."""
    left, top, right, bottom = box
    edges = [
        (lambda p: p[0] >= left, lambda a, b: (left, a[1] + (b[1] - a[1]) * (left - a[0]) / (b[0] - a[0]))),
        (lambda p: p[0] <= right, lambda a, b: (right, a[1] + (b[1] - a[1]) * (right - a[0]) / (b[0] - a[0]))),
        (lambda p: p[1] >= top, lambda a, b: (a[0] + (b[0] - a[0]) * (top - a[1]) / (b[1] - a[1]), top)),
        (lambda p: p[1] <= bottom, lambda a, b: (a[0] + (b[0] - a[0]) * (bottom - a[1]) / (b[1] - a[1]), bottom)),
    ]
    for inside, intersect in edges:
        if not points:
            break
        clipped = []
        for index, current in enumerate(points):
            previous = points[index - 1]
            if inside(current):
                if not inside(previous):
                    clipped.append(intersect(previous, current))
                clipped.append(current)
            elif inside(previous):
                clipped.append(intersect(previous, current))
        points = clipped
    deduplicated = [point for index, point in enumerate(points) if point != points[index - 1]] if len(points) > 1 else points
    return deduplicated


def _area(points: list[tuple[float, float]]) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))) / 2


def _crop_geometry(values: list[float], box: tuple[int, int, int, int], width: int, height: int) -> list[float] | None:
    """Normalized geometry relative to the crop, or None when too little of the object is left."""
    left, top, right, bottom = box
    crop_width, crop_height = right - left, bottom - top
    if len(values) == 4:
        cx, cy, box_width, box_height = values
        x1, x2 = sorted(((cx - box_width / 2) * width, (cx + box_width / 2) * width))
        y1, y2 = sorted(((cy - box_height / 2) * height, (cy + box_height / 2) * height))
        cx1, cy1, cx2, cy2 = max(x1, left), max(y1, top), min(x2, right), min(y2, bottom)
        original, visible = (x2 - x1) * (y2 - y1), max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
        if visible <= 0 or visible < MIN_VISIBLE * original:
            return None
        return [((cx1 + cx2) / 2 - left) / crop_width, ((cy1 + cy2) / 2 - top) / crop_height, (cx2 - cx1) / crop_width, (cy2 - cy1) / crop_height]
    points = [(x * width, y * height) for x, y in zip(values[::2], values[1::2])]
    clipped = _clip_polygon(points, box)
    original, visible = _area(points), _area(clipped) if len(clipped) >= 3 else 0.0
    if visible <= 0 or visible < MIN_VISIBLE * original:
        return None
    return [min(1.0, max(0.0, value)) for x, y in clipped for value in ((x - left) / crop_width, (y - top) / crop_height)]


def _number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def crop_label_text(text: str, box: tuple[int, int, int, int], width: int, height: int) -> tuple[str, int]:
    """Rewrite YOLO label lines for an image cropped to `box`; returns (text, objects dropped).

    Boxes (4 values) are clipped, polygons (6+ even values) clipped with Sutherland–Hodgman.
    Lines that are neither are kept as they are, so Validation still reports them.
    """
    lines, dropped = [], 0
    for line in text.splitlines(keepends=True):
        parts = line.split()
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError:
            values = []
        if not parts or not (len(values) == 4 or (len(values) >= 6 and len(values) % 2 == 0)):
            lines.append(line)
            continue
        geometry = _crop_geometry(values, box, width, height)
        if geometry is None:
            dropped += 1
            continue
        lines.append(" ".join([parts[0], *map(_number, geometry)]) + "\n")
    return "".join(lines), dropped


def crop_image(path: Path, crop: dict | None, fit: dict | None = None) -> dict | None:
    """Crop one image (crop None: the whole image), then resize it to fit (`fit`: resize.resize_options):
    {"data": bytes or None (neither changes it: copy unchanged), "box" (None: not cropped), "size", "resized"}.

    None when the region misses the image or the image cannot be read.
    """
    try:
        with Image.open(path) as source:
            width, height = imaging.upright_size(source)
            box = crop_box(width, height, crop) if crop else (0, 0, width, height)
            if box is None:
                return None
            cropped = box != (0, 0, width, height)
            target = resize.fit_size(box[2] - box[0], box[3] - box[1], fit) if fit else None
            if not cropped and target is None:
                return {"data": None, "box": None, "size": (width, height), "resized": False}
            image = ImageOps.exif_transpose(source)
            image = image.crop(box) if cropped else image
            image = resize.scaled(image, target) if target else image
            return {"data": imaging.encode(image, source, path.suffix), "box": box if cropped else None, "size": (width, height), "resized": target is not None}
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError):
        return None


def sample_preview(source: Path, relative: str, category: str, size: int = 640) -> tuple[bytes, int, int]:
    """A JPEG of one source image (upright, its labels drawn) for the crop preview, with its full size."""
    clean = safe_relative(relative)
    image_path = source.joinpath(*clean.parts)
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.resolve().is_relative_to(source.resolve()) or not image_path.is_file():
        raise DatasetError(f"Not an image in this import: {relative!r}")
    text = None
    coco = None if category == "classification" else _cached_coco(source, category)
    if coco is not None:
        pairs = coco["matched"].get(clean)
        text = _coco_label_text(pairs, None, category, image_path)[0] if pairs else None
    elif category != "classification":
        label = next((candidate for candidate in _label_candidates(image_path, source) if candidate.is_file()), None)
        text = label.read_text(encoding="utf-8", errors="replace") if label is not None else None
    try:
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise DatasetError(f"Cannot read {relative}: {exc}") from exc
    width, height = image.size
    image.thumbnail((size, size))
    if text is not None:
        draw = ImageDraw.Draw(image)
        for line in text.splitlines():
            try:
                values = [float(value) for value in line.split()[1:]]
            except ValueError:
                continue
            if len(values) == 4:
                cx, cy, box_width, box_height = values
                xs = sorted(((cx - box_width / 2) * image.width, (cx + box_width / 2) * image.width))
                ys = sorted(((cy - box_height / 2) * image.height, (cy + box_height / 2) * image.height))
                draw.rectangle((xs[0], ys[0], xs[1], ys[1]), outline="#facc15", width=2)
            elif len(values) >= 6 and len(values) % 2 == 0:
                points = [(x * image.width, y * image.height) for x, y in zip(values[::2], values[1::2])]
                draw.line(points + points[:1], fill="#facc15", width=2)
    output = io.BytesIO()
    image.save(output, "JPEG", quality=85)
    return output.getvalue(), width, height


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

    def __init__(self, files):
        """files: (path, size or None when not known)."""
        self.by_size: dict[int, list[Path]] = {}
        self.hashes: dict[Path, str] = {}
        for path, size in files:
            try:
                self.by_size.setdefault(size if size is not None else path.stat().st_size, []).append(path)
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

    def seen(self, path: Path, data: bytes | None = None, size: int | None = None) -> bool:
        """Whether `path` (or `data`, the bytes about to be written for it) is already known."""
        if data is not None:
            self.hashes[path] = hashlib.sha1(data).hexdigest()
        size = len(data) if data is not None else size if size is not None else path.stat().st_size
        others = self.by_size.get(size, [])
        if others:
            digest = self._hash(path)
            if any(self._hash(other) == digest for other in others):
                return True
        self.by_size.setdefault(size, []).append(path)
        return False


def run_import(dataset: Dataset, source: Path, options: dict, job: Job | None = None) -> dict:
    dataset.check_writable()
    allowed = dataset.capabilities()["splits"]
    if not allowed:
        raise DatasetError("This dataset's splits are not plain image folders, so images cannot be imported into it")
    scanned, items = _scan(dataset, source)
    if not items:
        raise DatasetError("No images found in the selected folder")
    _assign_splits(items, options, allowed)
    ensure_free_space(dataset.root, sum(item["size"] or 0 for item in items), "the import")
    # each split may have several folders; the caller picks one per split (default: the first)
    chosen = options.get("folders") or {}
    folders = {split: dataset.resolve_folder(split, chosen.get(split)) for split in {item["target"] for item in items}}
    crop = crop_options(options.get("crop"))
    fit = resize.resize_options(options.get("resize"))
    mapping = _resolve_mapping(dataset, scanned, options.get("mapping"))
    label_mapping = {int(key): value for key, value in mapping.items()} if scanned["format"] == "yolo" and dataset.category != "classification" and scanned["names_source"] else None
    # the dataset's image sizes as it knows them: no stat per image
    fingerprints = _Fingerprints((record.path, record.image_size) for record in list(dataset.images.values())) if options.get("skip_duplicates", True) else None
    planned: set[Path] = set()
    stats = {"imported": 0, "labels": 0, "duplicates": 0, "dropped": 0, "renamed": 0, "cropped": 0, "crop_skipped": 0, "resized": 0, "unreadable": 0, "objects_cropped_out": 0, "objects_skipped": 0, "by_split": {}}
    rescan: dict = {}
    if job:
        job.progress(0, len(items), "Copying files…")
    try:
        for index, item in enumerate(items):
            if job:
                job.progress(index, len(items))
            _copy_item(dataset, item, mapping, label_mapping, fingerprints, planned, stats, folders, crop, fit)
    finally:
        # also after a cancel: whatever was copied is part of the dataset now
        if job and not job.cancel_requested:
            job.progress(message="Indexing…")
        rescan = dataset.rescan()
        if job and job.cancel_requested:
            stats["cancelled"] = True  # cancelled (or the server stopped): only the files copied so far came in
        _forget_coco(source)
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


def _copy_item(dataset: Dataset, item: dict, mapping: dict, label_mapping: dict | None, fingerprints: _Fingerprints | None, planned: set[Path], stats: dict, folders: dict[str, Path], crop: dict | None = None, fit: dict | None = None) -> None:
    if dataset.category == "classification":
        class_id = mapping.get(item["class"])
        if class_id is None:
            stats["dropped"] += 1
            return
        directory = folders[item["target"]] / dataset.names[class_id]
    else:
        directory = folders[item["target"]]
    cropped = crop_image(item["image"], crop, fit) if crop or fit else None
    if cropped is None and (crop or fit):
        # a region that misses the image (or one that cannot be read: a resize must read every image)
        stats["crop_skipped" if crop else "unreadable"] += 1
        return
    data = cropped["data"] if cropped else None
    # a cropped or resized image is compared by the bytes it will have, so re-importing with the same options is still caught
    if fingerprints is not None and fingerprints.seen(item["image"], data, item.get("size")):
        stats["duplicates"] += 1
        return
    target, label_target = _target_paths(dataset, directory, item["image"].name, planned)
    if target.name != item["image"].name:
        stats["renamed"] += 1
    planned.add(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if data is not None:
        target.write_bytes(data)
        stats["cropped"] += cropped["box"] is not None
        stats["resized"] += cropped["resized"]
    else:
        shutil.copy2(item["image"], target)
    if label_target is not None and (item["label"] is not None or item["coco"] is not None):
        if item["coco"] is not None:  # listed in a COCO file: its objects (none: a background image, an empty label)
            text, unusable = _coco_label_text(item["coco"], mapping, dataset.category, item["image"])
            stats["objects_skipped"] += unusable
        else:
            text = item["label"].read_text(encoding="utf-8")
            if label_mapping is not None:
                text = classes.remap_label_text(text, label_mapping)
        if data is not None and cropped["box"] is not None:  # a resize alone keeps the relative coordinates
            text, removed = crop_label_text(text, cropped["box"], *cropped["size"])
            stats["objects_cropped_out"] += removed
        planned.add(label_target)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.write_text(text, encoding="utf-8", newline="\n")
        stats["labels"] += 1
    stats["imported"] += 1
    stats["by_split"][item["target"]] = stats["by_split"].get(item["target"], 0) + 1


def _log(dataset: Dataset, source: Path, options: dict, mapping: dict, stats: dict) -> None:
    entry = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "browser upload" if options.get("staging_id") else str(source),
        "options": {key: value for key, value in options.items() if key in {"split_mode", "split", "ratios", "seed", "skip_duplicates", "folders", "crop", "resize"}},
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
