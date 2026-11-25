"""SAM box fit: tighten detection boxes (e.g. a model's, a few pixels off) around their objects with SAM.

Fitting one box (fit_one): SAM is prompted with the label box grown by a margin (so an object the box cuts off
is still inside it), a click on the box centre, and background clicks on the centres of neighbouring labels inside
the grown box (cards in a fan); of SAM's three masks the one with the best score × overlap with the label wins
(not the suit symbol, not the whole stack). The fitted box is the box around that mask, without specks and parts
mostly outside the label. A fit is "fitted" or "uncertain" with its reasons (SAM unsure, a box very different
from the label, a mask reaching the grown box's edge — the object is larger, or SAM spread into the background —,
a label overlapped by another one — SAM outlines the visible part only), or "failed" (no mask).

Precision: SAM encodes 1024 px and draws masks on a 256×256 grid, so on a whole 4K image an edge can be several
pixels off — about the error being fixed. crop_plan encodes a region around each box instead (a few times the
object's size, at least MIN_CROP px), boxes close together sharing one region.

BoxFitManager runs a fit over a class or chosen objects in a thread and keeps the fits per image in
.yolo-workbench/boxfit.json, keyed by the label's geometry. Findings are derived from the current labels on every
listing, so a label edited since its fit drops out, as does an image whose file changed. Applying goes through
Dataset.bulk_edit_objects ("reshape"): backups, undo, conflict detection and the training lock apply, and
"apply all" is one transaction.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import identity
from .dataset import WORKBENCH_DIR, Dataset, DatasetError, object_key
from .imaging import upright_rgb
from .labelqa import _dashed
from .models import ImageRecord
from .predictor import _bbox, _bbox_iou

MARGIN_PX = 12            # the prompt box reaches this far past the label on every side ...
MARGIN_FRACTION = 0.15    # ... or this fraction of the box's side, when that is more
CONTEXT = 2.0             # a region encoded for a box is this many times the grown box's longer side ...
MIN_CROP = 256            # ... and at least this many pixels
WHOLE_FRACTION = 0.75     # a region this close to the image's size: encode the whole image
GROUP_SLACK = 1.5         # boxes share a region while it stays within this factor of each one's own
MAX_NEGATIVES = 8         # background clicks on neighbouring labels, nearest first
MIN_AREA = 4              # px; a smaller mask is no mask
SIDE_PART = 0.1           # a mask part apart from the main one counts when it is this large (an occluded object's)
FRINGE_KERNEL = 5         # px; mask structures thinner than this are SAM's edge fringe, not the object
MIN_SCORE = 0.7           # SAM's own estimate of its mask's quality, below which a fit is uncertain
MIN_IOU = 0.5             # a fitted box overlapping its label less than this is something else
SIZE_RANGE = (0.5, 2.0)   # ... as is one whose area changed beyond these factors
OVERLAP_FRACTION = 0.15   # a label this much covered by another one: SAM's box is the visible part only
EDGE_PX = 1.0             # a mask edge this close to the prompt box's: it may continue past it
REASONS = {
    "low_score": "SAM is unsure of the outline",
    "different": "SAM's box is very different from the label",
    "edge": "the outline reaches the search area's edge",
    "overlap": "the label overlaps another one",
}
KINDS = ("fitted", "uncertain", "failed")
DEFAULT_TOLERANCE_PX = 2.0
SAVE_INTERVAL = 20.0
VERSION = 1


# ---------- geometry (pixels of the upright image) ----------

def grown(box, width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    margin_x = max(MARGIN_PX, MARGIN_FRACTION * (x2 - x1))
    margin_y = max(MARGIN_PX, MARGIN_FRACTION * (y2 - y1))
    return max(0.0, x1 - margin_x), max(0.0, y1 - margin_y), min(float(width), x2 + margin_x), min(float(height), y2 + margin_y)


def _region(box, width: int, height: int) -> tuple[int, int, int, int]:
    """A square around the box with context, shifted inside the image (clipped where the image is smaller)."""
    x1, y1, x2, y2 = grown(box, width, height)
    side = max(MIN_CROP, CONTEXT * max(x2 - x1, y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return _clamped(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, width, height)


def _clamped(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    w, h = min(x2 - x1, width), min(y2 - y1, height)
    left, top = min(max(0.0, x1), width - w), min(max(0.0, y1), height - h)
    return int(np.floor(left)), int(np.floor(top)), min(width, int(np.ceil(left + w))), min(height, int(np.ceil(top + h)))


def crop_plan(boxes: list, width: int, height: int) -> list[tuple[tuple[int, int, int, int], list[int]]]:
    """Regions to encode, each with the indices of the boxes fitted in it. A box gets a region a few times its size;
    one near another box's joins its region while the union stays within GROUP_SLACK of both (no resolution lost to
    speak of); a region nearly as large as the image becomes the whole image."""
    whole = (0, 0, width, height)
    groups: list[list] = []  # [region, the finest side a member asked for, members]
    order = sorted(range(len(boxes)), key=lambda index: ((boxes[index][1] + boxes[index][3]) / 2, (boxes[index][0] + boxes[index][2]) / 2))
    for index in order:
        region = _region(boxes[index], width, height)
        side = max(region[2] - region[0], region[3] - region[1])
        if side >= WHOLE_FRACTION * max(width, height):
            region, side = whole, max(width, height)
        for group in groups:
            union = (min(group[0][0], region[0]), min(group[0][1], region[1]), max(group[0][2], region[2]), max(group[0][3], region[3]))
            union_side = max(union[2] - union[0], union[3] - union[1])
            if union_side <= GROUP_SLACK * min(group[1], side) or union == whole and group[0] == whole:
                group[0] = whole if union_side >= WHOLE_FRACTION * max(width, height) else union
                group[1] = min(group[1], side)
                group[2].append(index)
                break
        else:
            groups.append([region, side, [index]])
    return [(tuple(region), sorted(members)) for region, _, members in groups]


def _negatives(box, prompt, others) -> list[tuple[float, float]]:
    """Centres of other labels inside the prompt box but outside this label, nearest first: SAM's background."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    centres = []
    for other in others:
        ox, oy = (other[0] + other[2]) / 2, (other[1] + other[3]) / 2
        inside_prompt = prompt[0] <= ox <= prompt[2] and prompt[1] <= oy <= prompt[3]
        inside_label = box[0] <= ox <= box[2] and box[1] <= oy <= box[3]
        if inside_prompt and not inside_label:
            centres.append((ox, oy))
    centres.sort(key=lambda point: (point[0] - cx) ** 2 + (point[1] - cy) ** 2)
    return centres[:MAX_NEGATIVES]


def _mask_extent(mask: np.ndarray, label) -> tuple[tuple[float, float, float, float], np.ndarray] | None:
    """The box around the mask's part under the label (crop pixels, edges exclusive), plus any other large part lying
    mostly inside the label, and that mask; specks, neighbours' slivers in the margin and edge fringe are left out."""
    import cv2

    from .masks import drop_specks

    mask = drop_specks(mask)
    if mask.sum() < MIN_AREA:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    height, width = mask.shape
    x1, y1 = int(max(0, np.floor(label[0]))), int(max(0, np.floor(label[1])))
    x2, y2 = int(min(width, np.ceil(label[2]))), int(min(height, np.ceil(label[3])))
    inside = np.bincount(labels[y1:y2, x1:x2].ravel(), minlength=count) if x2 > x1 and y2 > y1 else np.zeros(count, int)
    inside[0] = 0
    if inside.max() == 0:
        return None
    main = int(inside.argmax())
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = [part for part in range(1, count) if part == main or areas[part] >= SIDE_PART * areas[main] and inside[part] >= 0.5 * areas[part]]
    kept = _without_fringe(np.isin(labels, keep))
    ys, xs = np.nonzero(kept)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)), kept


def edge_strength(image: np.ndarray) -> np.ndarray:
    """How strong an edge each pixel of the (BGR) image is near: the gradient magnitude of the lightly smoothed
    image, spread by a pixel so that a mask outline a pixel off its edge still finds it."""
    import cv2

    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    magnitude = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    return cv2.dilate(magnitude, np.ones((3, 3), np.uint8))


def _outline_contrast(mask: np.ndarray, edges: np.ndarray) -> float:
    """The mean edge strength along the mask's outline: high where it follows the object's edge, low where SAM
    filled the prompt box with background (a ball on plain felt)."""
    import cv2

    solid = mask.astype(np.uint8)
    outline = (solid - cv2.erode(solid, np.ones((3, 3), np.uint8))).astype(bool)
    return float(edges[outline].mean()) if outline.any() else 0.0


def _without_fringe(mask: np.ndarray) -> np.ndarray:
    """The mask without the thin, dotted fringe SAM leaves along some edges (cells of its 256×256 grid just over
    the threshold), which widen a box by several pixels: an opening keeps the solid core, then the mask's own
    pixels within a pixel of the core come back, so corners (a rotated card's tip) are not cut."""
    import cv2

    ys, xs = np.nonzero(mask)
    side = min(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    size = FRINGE_KERNEL if side >= 4 * FRINGE_KERNEL else 3 if side >= 12 else 0
    if not size:
        return mask
    solid = mask.astype(np.uint8)
    core = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))
    if core.sum() < 0.5 * solid.sum():
        return mask  # mostly thin itself (a rope, a frame): nothing to tell apart
    return (solid & cv2.dilate(core, np.ones((3, 3), np.uint8))).astype(bool)


def fit_one(backend, features, region, box, others, width: int, height: int, edges: np.ndarray | None = None) -> dict:
    """Fit one label box (image pixels) with SAM features of `region`; see the module docstring. `edges` is
    edge_strength of the region: of SAM's masks, the one whose outline follows the image's edges is preferred."""
    left, top, right, bottom = region
    prompt = grown(box, width, height)
    shift = lambda points: [[x - left, y - top] for x, y in points]
    local_box = [box[0] - left, box[1] - top, box[2] - left, box[3] - top]
    points = shift([((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)] + _negatives(box, prompt, others))
    labels = [1] + [0] * (len(points) - 1)
    shape = (bottom - top, right - left)
    local_prompt = [max(0.0, prompt[0] - left), max(0.0, prompt[1] - top), min(float(shape[1]), prompt[2] - left), min(float(shape[0]), prompt[3] - top)]
    if hasattr(backend, "candidates"):
        candidates = backend.candidates(features, shape, points, labels, local_prompt)
    else:  # a backend with one mask per prompt
        candidates = [backend.segment(features, shape, points, labels, local_prompt)]
    # SAM's own score and the overlap with the label rule out a part (the suit symbol) or a group (the stack), but
    # not the box-filling mask SAM draws on a plain background: its overlap with a label a few pixels off is as
    # good as the object's. Its outline does not follow an edge of the image, though.
    found = []
    for mask, score in candidates:
        result = _mask_extent(mask, local_box)
        if result is not None:
            extent, kept = result
            found.append((extent, score, _bbox_iou(extent, local_box), 1.0 if edges is None else _outline_contrast(kept, edges)))
    if not found:
        return {"box": None, "score": 0.0, "status": "failed", "reasons": []}
    strongest = max(contrast for *_, contrast in found) or 1.0
    extent, score, _, _ = max(found, key=lambda item: item[1] * item[2] * item[3] / strongest)
    fitted = (extent[0] + left, extent[1] + top, extent[2] + left, extent[3] + top)
    reasons = judge(box, prompt, fitted, score, others, width, height)
    return {
        "box": [round(fitted[0] / width, 6), round(fitted[1] / height, 6), round(fitted[2] / width, 6), round(fitted[3] / height, 6)],
        "score": round(float(score), 4),
        "status": "uncertain" if reasons else "fitted",
        "reasons": reasons,
    }


def judge(box, prompt, fitted, score: float, others, width: int, height: int) -> list[str]:
    """Why a fit may be wrong (empty: none found); all boxes in pixels."""
    reasons = []
    if score < MIN_SCORE:
        reasons.append("low_score")
    area = lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    ratio = area(fitted) / area(box) if area(box) > 0 else 0.0
    if _bbox_iou(box, fitted) < MIN_IOU or not SIZE_RANGE[0] <= ratio <= SIZE_RANGE[1]:
        reasons.append("different")
    # an outline at (or past) the prompt box's edge, where that edge is not the image's: the object may go on
    reaches = (
        prompt[0] > EDGE_PX and fitted[0] - prompt[0] <= EDGE_PX,
        prompt[1] > EDGE_PX and fitted[1] - prompt[1] <= EDGE_PX,
        prompt[2] < width - EDGE_PX and prompt[2] - fitted[2] <= EDGE_PX,
        prompt[3] < height - EDGE_PX and prompt[3] - fitted[3] <= EDGE_PX,
    )
    if any(reaches):
        reasons.append("edge")
    for other in others:
        overlap_w = min(box[2], other[2]) - max(box[0], other[0])
        overlap_h = min(box[3], other[3]) - max(box[1], other[1])
        if overlap_w > 0 and overlap_h > 0 and overlap_w * overlap_h >= OVERLAP_FRACTION * area(box):
            reasons.append("overlap")
            break
    return reasons


def geometry_key(points) -> str:
    """A label's geometry, digested: what a fit belongs to (the class may change, the fit stays right)."""
    return hashlib.sha1(repr([float(value) for value in points]).encode()).hexdigest()[:16]


def corners(points) -> list[float]:
    return list(_bbox("detection", points))


def yolo_box(box) -> list[float]:
    """Normalized x1,y1,x2,y2 -> YOLO cx, cy, w, h."""
    x1, y1, x2, y2 = box
    return [round((x1 + x2) / 2, 6), round((y1 + y2) / 2, 6), round(x2 - x1, 6), round(y2 - y1, 6)]


# ---------- the batch ----------

class BoxFitManager:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.job: dict = {"state": "idle", "done": 0, "total": 0, "failed": 0, "objects": 0, "error": None}
        self.result: dict | None = None  # {version, model, updated, started_by, images: {key: {id, mtime, fits}}, dismissed}
        self._cache: dict[str, tuple] = {}  # image key -> (inputs, findings at tolerance 0)
        self._load()

    @property
    def available(self) -> bool:
        return self.dataset.category == "detection"

    @property
    def _path(self) -> Path:
        return self.dataset.root / WORKBENCH_DIR / "boxfit.json"

    def _key(self, path: Path) -> str:
        text, root = str(path), os.path.join(str(self.dataset.root), "")
        if os.path.normcase(text).startswith(os.path.normcase(root)):
            text = text[len(root):]
        return text.replace(os.sep, "/") if os.sep != "/" else text

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict) and data.get("version") == VERSION and isinstance(data.get("images"), dict):
            data.setdefault("dismissed", {})
            self.result = data

    def _save(self) -> None:
        with self._lock:
            if self.result is None:
                return
            text = json.dumps(self.result, separators=(",", ":"))
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self.dataset._atomic_write(self._path, text)
        except OSError:
            pass  # fits can be computed again; a failed save must not break the run

    def payload(self) -> dict:
        with self._lock:
            result = self.result
            summary = None
            if result is not None:
                summary = {
                    "model": result.get("model"),
                    "updated": result.get("updated"),
                    "started_by": result.get("started_by"),
                    "images": len(result["images"]),
                    "objects": sum(len(entry["fits"]) for entry in result["images"].values()),
                }
            return {"available": self.available, "category": self.dataset.category, "job": dict(self.job), "result": summary}

    def busy_reason(self) -> str | None:
        return "boxes are being fitted with SAM" if self.job["state"] == "running" else None

    # ---------- running ----------

    def start(self, sam, class_ids: list[int] | None = None, split: str | None = None, objects: list[dict] | None = None) -> dict:
        """Fit every box of `class_ids` (None: every class) in `split`, or the listed objects ({image_id, key}: the
        Exploration gallery's object key). Fits are added to the stored ones (an object fitted again is replaced)."""
        if not self.available:
            raise DatasetError("Box fit needs a detection dataset")
        dataset = self.dataset
        with dataset._lock:
            if objects is not None:
                wanted: dict[str, set[str]] = {}
                for item in objects:
                    wanted.setdefault(item["image_id"], set()).add(item["key"])
                tasks = [(dataset.images[image_id], keys) for image_id, keys in wanted.items() if image_id in dataset.images]
            else:
                classes = None if class_ids is None else set(class_ids)
                tasks = [
                    (record, None) for record in dataset.images.values()
                    if (split in (None, "all") or record.split == split)
                    and any(classes is None or annotation.class_id in classes for annotation in record.annotations)
                ]
        if not tasks:
            raise DatasetError("No boxes to fit here")
        with self._lock:
            if self.job["state"] == "running":
                raise DatasetError("Boxes are already being fitted; wait for it or cancel it")
            if self.result is None:
                self.result = {"version": VERSION, "images": {}, "dismissed": {}}
            self.result.update(updated=time.strftime("%Y-%m-%dT%H:%M:%S"), started_by=identity.who())
            self.job = {"state": "running", "done": 0, "total": len(tasks), "failed": 0, "objects": 0, "error": None}
            self._cancel.clear()
            classes = None if class_ids is None else set(class_ids)
            self._thread = threading.Thread(target=self._run, args=(sam, tasks, classes), name="boxfit", daemon=True)
            self._thread.start()
        return self.payload()

    def cancel(self) -> dict:
        with self._lock:
            if self.job["state"] != "running":
                raise DatasetError("Box fit is not running")
            self._cancel.set()
        return self.payload()

    def stop(self, timeout: float) -> None:
        """Cancel a running fit and wait (up to `timeout` seconds) for its current image."""
        with self._lock:
            if self.job["state"] != "running":
                return
            self._cancel.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def clear(self) -> dict:
        with self._lock:
            if self.job["state"] == "running":
                raise DatasetError("Boxes are being fitted; cancel it first")
            self.result = None
            self._cache.clear()
            self.job = {"state": "idle", "done": 0, "total": 0, "failed": 0, "objects": 0, "error": None}
        self._path.unlink(missing_ok=True)
        return self.payload()

    def _wait_for(self, sam) -> None:
        """SAM may still be loading (its first use downloads the weights)."""
        while True:
            state = sam.payload()
            if state["status"] == "ready":
                return
            if state["status"] != "loading":
                raise DatasetError(state.get("unloaded") or state.get("error") or "SAM is not loaded")
            if self._cancel.wait(0.5):
                return

    def _run(self, sam, tasks: list[tuple[ImageRecord, set[str] | None]], classes: set[int] | None) -> None:
        last_save = time.monotonic()
        try:
            for record, keys in tasks:
                self._wait_for(sam)
                if self._cancel.is_set():
                    break
                self._fit_record(sam, record, keys, classes)
                if time.monotonic() - last_save > SAVE_INTERVAL:
                    self._save()
                    last_save = time.monotonic()
            with self._lock:
                self.job["state"] = "cancelled" if self._cancel.is_set() else "done"
        except Exception as exc:  # surfaced to the UI, never crashes the server
            with self._lock:
                self.job["state"] = "error"
                self.job["error"] = str(exc)
        finally:
            self._save()

    def _fit_record(self, sam, record: ImageRecord, keys: set[str] | None, classes: set[int] | None) -> None:
        """Fit one image's chosen boxes (`keys`: object keys) or those of `classes` (None: all), and store the fits."""
        with self.dataset._lock:
            annotations = [annotation for annotation in record.annotations if annotation.points]
            stamp = self.dataset.image_stamp(record)
        targets = [
            index for index, annotation in enumerate(annotations)
            if (object_key(annotation) in keys if keys is not None else classes is None or annotation.class_id in classes)
        ]
        if not targets or stamp is None:
            with self._lock:
                self.job["done"] += 1
            return
        try:
            fits = sam.fit_boxes(record.path, [corners(annotation.points) for annotation in annotations], targets, cache=False)
        except DatasetError:
            raise  # SAM was unloaded: the run stops and says why
        except Exception:  # one unreadable image must not stop the run
            with self._lock:
                self.job["failed"] += 1
                self.job["done"] += 1
            return
        key = self._key(record.path)
        with self._lock:
            self.result["model"] = sam.payload().get("model") or self.result.get("model")
            entry = self.result["images"].get(key)
            if entry is None or entry["mtime"] != stamp[0]:
                entry = self.result["images"][key] = {"id": record.id, "mtime": stamp[0], "fits": {}}
            entry["id"] = record.id
            for index, fit in zip(targets, fits):
                entry["size"] = fit.pop("image_size", entry.get("size"))
                entry["fits"][geometry_key(annotations[index].points)] = fit
            entry["version"] = entry.get("version", 0) + 1
            self.job["done"] += 1
            self.job["objects"] += len(targets)

    # ---------- findings ----------

    def _record(self, key: str, entry: dict, by_key: dict | None) -> tuple[ImageRecord | None, dict | None]:
        record = self.dataset.images.get(entry.get("id"))
        if record is None or self._key(record.path) != key:
            if by_key is None:  # opened again after a move: found by path
                by_key = {self._key(candidate.path): candidate for candidate in self.dataset.images.values()}
            record = by_key.get(key)
        return record, by_key

    def _image_findings(self, key: str, entry: dict, record: ImageRecord) -> list[dict] | None:
        """Every finding of one image at tolerance 0 (None when the image file changed since its fit)."""
        stamp = self.dataset.image_stamp(record)
        if stamp is None or stamp[0] != entry["mtime"]:
            return None
        dismissed = tuple(self.result["dismissed"].get(key, ()))
        inputs = (id(entry), entry.get("version"), self.dataset._revisions.get(record.label_path), dismissed, record.width, record.height)
        cached = self._cache.get(key)
        if cached and cached[0] == inputs:
            return cached[1]
        size = entry.get("size") or (0, 0)  # the image's when it was fitted; the dataset may not have probed it yet
        width, height = record.width or size[0], record.height or size[1]
        findings = []
        for annotation in record.annotations:
            fit = entry["fits"].get(geometry_key(annotation.points)) if annotation.points else None
            if fit is None:
                continue
            label = corners(annotation.points)
            if fit["box"] is None:
                kind, edges, deviation, proposal = "failed", None, None, None
            else:
                kind, proposal = fit["status"], fit["box"]
                if not width or not height:
                    continue  # fitted before sizes were kept: nothing to measure the move in
                edges = [round((proposal[side] - label[side]) * (width if side % 2 == 0 else height), 1) for side in range(4)]
                deviation = max(abs(edge) for edge in edges)
                if deviation == 0:
                    continue
            finding_key = hashlib.sha1(f"{geometry_key(annotation.points)}|{proposal}".encode()).hexdigest()[:16]
            if finding_key in dismissed:
                continue
            findings.append({
                "key": finding_key,
                "kind": kind,
                "image_id": record.id,
                "image_name": record.path.name,
                "split": record.split,
                "class_id": annotation.class_id,
                "label": {"annotation_id": annotation.id, "class_id": annotation.class_id, "points": list(annotation.points), "key": object_key(annotation)},
                "prediction": None if proposal is None else {"class_id": annotation.class_id, "points": yolo_box(proposal)},
                "score": fit["score"],
                "reasons": fit["reasons"],
                "edges": edges,  # left, top, right, bottom: how far each edge moves (px, + is right/down)
                "deviation_px": deviation,
            })
        self._cache[key] = (inputs, findings)
        return findings

    def _all(self, kind: str | None, class_id: int | None, split: str | None, tolerance: float) -> tuple[list[dict], dict, int]:
        """(findings matching the filters but the kind, counts per kind, stale images). Caller holds both locks."""
        items, stale, by_key = [], 0, None
        for key, entry in self.result["images"].items():
            record, by_key = self._record(key, entry, by_key)
            if record is None or split not in (None, "all") and record.split != split:
                continue
            found = self._image_findings(key, entry, record)
            if found is None:
                stale += 1
                continue
            items.extend(
                finding for finding in found
                if (class_id is None or finding["class_id"] == class_id) and (finding["kind"] == "failed" or finding["deviation_px"] > tolerance)
            )
        counts = {name: 0 for name in KINDS}
        for finding in items:
            counts[finding["kind"]] += 1
        if kind:
            items = [finding for finding in items if finding["kind"] == kind]
        items.sort(key=lambda finding: (finding["kind"] == "failed", -(finding["deviation_px"] or 0), finding["image_name"], finding["key"]))
        return items, counts, stale

    def findings(self, kind: str | None = None, class_id: int | None = None, split: str | None = None, tolerance: float = DEFAULT_TOLERANCE_PX, offset: int = 0, limit: int = 60) -> dict:
        """Boxes SAM would move by more than `tolerance` px (biggest moves first), and the boxes it could not fit."""
        if kind is not None and kind not in KINDS:
            raise DatasetError(f"Unknown finding kind {kind}")
        with self.dataset._lock, self._lock:
            if self.result is None:
                return {"total": 0, "images": 0, "stale": 0, "counts": {name: 0 for name in KINDS}, "items": []}
            items, counts, stale = self._all(kind, class_id, split, tolerance)
            return {
                "total": len(items),
                "images": len({finding["image_id"] for finding in items}),
                "stale": stale,
                "counts": counts,
                "items": items[offset : offset + limit],
            }

    def _finding(self, image_id: str, key: str) -> tuple[ImageRecord, dict, str]:
        record = self.dataset.require_image(image_id)
        if self.result is None:
            raise DatasetError("Fit boxes with SAM first")
        image_key = self._key(record.path)
        entry = self.result["images"].get(image_key)
        found = self._image_findings(image_key, entry, record) if entry else None
        finding = next((item for item in found or () if item["key"] == key), None)
        if finding is None:
            raise DatasetError("This fit no longer applies; the label or the image changed")
        return record, finding, image_key

    @staticmethod
    def _operation(finding: dict) -> dict:
        label = finding["label"]
        return {"image_id": finding["image_id"], "annotation_id": label["annotation_id"], "action": "reshape", "points": finding["prediction"]["points"], "key": label["key"]}

    def apply(self, image_id: str, key: str) -> dict:
        with self.dataset._lock:
            with self._lock:
                _, finding, _ = self._finding(image_id, key)
            if finding["prediction"] is None:
                raise DatasetError("SAM found no outline for this box")
            result = self.dataset.bulk_edit_objects([self._operation(finding)])
            if not result["applied"]:
                raise DatasetError(result["skipped"][0]["reason"] if result["skipped"] else "The label changed")
            return {"detail": self.dataset.detail(image_id), **result}

    def apply_all(self, kind: str | None, class_id: int | None, split: str | None, tolerance: float) -> dict:
        """Apply every listed fit (failed ones have nothing to apply) as one transaction: one undo."""
        if kind == "failed":
            raise DatasetError("Failed fits have no box to apply")
        with self.dataset._lock:
            with self._lock:
                if self.result is None:
                    raise DatasetError("Fit boxes with SAM first")
                items, _, _ = self._all(kind, class_id, split, tolerance)
            operations = [self._operation(finding) for finding in items if finding["prediction"] is not None]
            if not operations:
                raise DatasetError("No fitted boxes to apply")
            result = self.dataset.bulk_edit_objects(operations)
        return {**result, "image_ids": sorted({operation["image_id"] for operation in operations})}

    def dismiss(self, image_id: str, key: str) -> dict:
        with self.dataset._lock, self._lock:
            _, finding, image_key = self._finding(image_id, key)
            self.result["dismissed"].setdefault(image_key, []).append(finding["key"])
        self._save()
        return {"dismissed": key}

    def crop(self, image_id: str, key: str, color) -> bytes:
        with self.dataset._lock, self._lock:
            record, finding, _ = self._finding(image_id, key)
        return render_fit(record.path, finding, color)


def render_fit(path: Path, finding: dict, color, size: tuple[int, int] = (420, 320)) -> bytes:
    """A crop around the label (solid, class color) and SAM's box (dashed white), scaled up to `size` (at most 4×)
    so that a move of a few pixels shows; lines are drawn after scaling, thin."""
    shapes = [corners(finding["label"]["points"])] + ([corners(finding["prediction"]["points"])] if finding["prediction"] else [])
    left, top = min(box[0] for box in shapes), min(box[1] for box in shapes)
    right, bottom = max(box[2] for box in shapes), max(box[3] for box in shapes)
    with Image.open(path) as source:
        image = upright_rgb(source)
    width, height = image.size
    pad_x, pad_y = max((right - left) * width * 0.2, 12), max((bottom - top) * height * 0.2, 12)
    box = (max(0, int(left * width - pad_x)), max(0, int(top * height - pad_y)), min(width, int(right * width + pad_x) + 1), min(height, int(bottom * height + pad_y) + 1))
    crop = image.crop(box)
    scale = min(size[0] / crop.width, size[1] / crop.height, 4.0)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(crop)

    def outline(corner_box):
        l, t, r, b = ((corner_box[0] * width - box[0]) * scale, (corner_box[1] * height - box[1]) * scale, (corner_box[2] * width - box[0]) * scale, (corner_box[3] * height - box[1]) * scale)
        return [(l, t), (r, t), (r, b), (l, b), (l, t)]

    label = outline(shapes[0])
    draw.line(label, fill="#000000", width=3)
    draw.line(label, fill=color(finding["label"]["class_id"]), width=1)
    if finding["prediction"]:
        proposal = outline(shapes[1])
        _dashed(draw, proposal, "#000000", 3, dash=6, gap=4)
        _dashed(draw, proposal, "#ffffff", 1, dash=6, gap=4)
    output = io.BytesIO()
    crop.save(output, "JPEG", quality=90)
    return output.getvalue()
