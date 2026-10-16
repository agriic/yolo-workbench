"""Import images (and YOLO labels) from a folder into a dataset.

Browser uploads land in <dataset>/.yolo-workbench/staging/<id>/ first and are then
imported exactly like a local folder. Files are always copied; labels are rewritten
with the chosen class mapping. An optional crop (centered, or a region) re-encodes the
images and clips the labels to it. Imports are not on the undo stack (delete the images
to reverse one) but each is logged to .yolo-workbench/imports.jsonl.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import secrets
import shutil
import time
from pathlib import Path, PurePosixPath

import yaml
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from . import classes
from .dataset import IMAGE_EXTENSIONS, WORKBENCH_DIR, Dataset, DatasetError
from .jobs import Job

SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val", "test": "test", "testing": "test"}
UPLOAD_EXTENSIONS = IMAGE_EXTENSIONS | {".txt", ".yaml", ".yml", ".json"}
STAGING_PATTERN = re.compile(r"^[a-f0-9]{16}$")
UNASSIGNED = ""  # classification images that are not inside a class folder
SAMPLE_COUNT = 24  # images offered for the crop preview, spread over the whole import
MIN_VISIBLE = 0.1  # a cropped object keeping less of its area than this is dropped (Ultralytics' area_thr)
ORIENTATION = 0x0112  # EXIF tag


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
        "names_source": None,
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


def _encode(image: Image.Image, source: Image.Image, suffix: str) -> bytes:
    """Encode a cropped image in the source's format, keeping its ICC profile and EXIF (minus orientation)."""
    image_format = Image.registered_extensions().get(suffix.lower(), "PNG")
    params: dict = {}
    if image_format in ("JPEG", "WEBP"):
        params["quality"] = 95
    if image_format == "JPEG" and image.mode not in ("RGB", "L", "CMYK"):
        image = image.convert("RGB")
    if source.info.get("icc_profile"):
        params["icc_profile"] = source.info["icc_profile"]
    exif = source.getexif()
    if exif and image_format in ("JPEG", "WEBP", "PNG", "TIFF"):
        exif.pop(ORIENTATION, None)
        params["exif"] = exif.tobytes()
    output = io.BytesIO()
    image.save(output, image_format, **params)
    return output.getvalue()


def crop_image(path: Path, crop: dict) -> dict | None:
    """Crop one image: {"data": bytes or None (the box is the whole image: copy unchanged), "box", "size"}.

    None when the region misses the image or the image cannot be read.
    """
    try:
        with Image.open(path) as source:
            upright = ImageOps.exif_transpose(source)
            width, height = upright.size
            box = crop_box(width, height, crop)
            if box is None:
                return None
            if box == (0, 0, width, height):
                return {"data": None, "box": box, "size": (width, height)}
            return {"data": _encode(upright.crop(box), source, path.suffix), "box": box, "size": (width, height)}
    except (OSError, UnidentifiedImageError, ValueError):
        return None


def sample_preview(source: Path, relative: str, category: str, size: int = 640) -> tuple[bytes, int, int]:
    """A JPEG of one source image (upright, its labels drawn) for the crop preview, with its full size."""
    image_path = source.joinpath(*safe_relative(relative).parts)
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.resolve().is_relative_to(source.resolve()) or not image_path.is_file():
        raise DatasetError(f"Not an image in this import: {relative!r}")
    label = None if category == "classification" else next((candidate for candidate in _label_candidates(image_path, source) if candidate.is_file()), None)
    try:
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise DatasetError(f"Cannot read {relative}: {exc}") from exc
    width, height = image.size
    image.thumbnail((size, size))
    if label is not None:
        draw = ImageDraw.Draw(image)
        for line in label.read_text(encoding="utf-8", errors="replace").splitlines():
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

    def seen(self, path: Path, data: bytes | None = None) -> bool:
        """Whether `path` (or `data`, the bytes about to be written for it) is already known."""
        if data is not None:
            self.hashes[path] = hashlib.sha1(data).hexdigest()
        size = len(data) if data is not None else path.stat().st_size
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
    scanned = scan(dataset, source)
    items = _scan_items(source, dataset.category)
    if not items:
        raise DatasetError("No images found in the selected folder")
    _assign_splits(items, options, allowed)
    # each split may have several folders; the caller picks one per split (default: the first)
    chosen = options.get("folders") or {}
    folders = {split: dataset.resolve_folder(split, chosen.get(split)) for split in {item["target"] for item in items}}
    crop = crop_options(options.get("crop"))
    mapping = _resolve_mapping(dataset, scanned, options.get("mapping"))
    label_mapping = {int(key): value for key, value in mapping.items()} if dataset.category != "classification" and scanned["names_source"] else None
    fingerprints = _Fingerprints(record.path for record in list(dataset.images.values())) if options.get("skip_duplicates", True) else None
    planned: set[Path] = set()
    stats = {"imported": 0, "labels": 0, "duplicates": 0, "dropped": 0, "renamed": 0, "cropped": 0, "crop_skipped": 0, "objects_cropped_out": 0, "by_split": {}}
    rescan: dict = {}
    if job:
        job.progress(0, len(items), "Copying files…")
    try:
        for index, item in enumerate(items):
            if job:
                job.progress(index, len(items))
            _copy_item(dataset, item, mapping, label_mapping, fingerprints, planned, stats, folders, crop)
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


def _copy_item(dataset: Dataset, item: dict, mapping: dict, label_mapping: dict | None, fingerprints: _Fingerprints | None, planned: set[Path], stats: dict, folders: dict[str, Path], crop: dict | None = None) -> None:
    if dataset.category == "classification":
        class_id = mapping.get(item["class"])
        if class_id is None:
            stats["dropped"] += 1
            return
        directory = folders[item["target"]] / dataset.names[class_id]
    else:
        directory = folders[item["target"]]
    cropped = crop_image(item["image"], crop) if crop else None
    if crop and cropped is None:
        stats["crop_skipped"] += 1
        return
    data = cropped["data"] if cropped else None
    # a cropped image is compared by the bytes it will have, so re-importing with the same crop is still caught
    if fingerprints is not None and fingerprints.seen(item["image"], data):
        stats["duplicates"] += 1
        return
    target, label_target = _target_paths(dataset, directory, item["image"].name, planned)
    if target.name != item["image"].name:
        stats["renamed"] += 1
    planned.add(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if data is not None:
        target.write_bytes(data)
        stats["cropped"] += 1
    else:
        shutil.copy2(item["image"], target)
    if label_target is not None and item["label"] is not None:
        text = item["label"].read_text(encoding="utf-8")
        if label_mapping is not None:
            text = classes.remap_label_text(text, label_mapping)
        if data is not None:
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
        "options": {key: value for key, value in options.items() if key in {"split_mode", "split", "ratios", "seed", "skip_duplicates", "folders", "crop"}},
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
