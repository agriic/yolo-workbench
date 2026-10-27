"""Label QA: rank likely label mistakes by where a trained model disagrees with the labels.

An analysis runs the predictor's loaded model over a split at a low confidence and stores the
raw predictions (in .yolo-workbench/labelqa.json, so results survive a restart). Findings are
derived from those predictions and the *current* labels each time they are listed, so fixing a
label makes its finding disappear without re-running the model:

- missing_label    a confident prediction no label matches (likely an object nobody labelled)
- wrong_class      a label and a confident prediction overlap well but disagree on the class
- box_mismatch     a confident prediction of the label's class whose box differs by more than the
                   tolerance (a few pixels, or a small fraction of the object's size)
- unmatched_label  a label nothing the model predicted overlaps (likely spurious or mislabelled)

Pairing is greedy by confidence × box IoU (segmentation polygons use their bounding boxes); see
compute_findings. The tolerance is chosen per listing, so changing it needs no new analysis.
Applying a finding's fix goes through Dataset.replace_annotations, so backups, undo, conflict
detection, and the training lock apply. Segmentation polygons are paired and keyed as the model
drew them; what is listed, drawn and applied is simplified to the predictor's point limit. Dismissing one ("the label is right") is remembered.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw

from . import identity
from .dataset import WORKBENCH_DIR, Dataset, DatasetError
from .imaging import upright_rgb
from .models import Annotation, ImageRecord
from .predictor import PredictorManager, _bbox, _bbox_iou

QA_CONF = 0.05        # predictions are kept down to this confidence, so "the model saw nothing here" means something
MATCH_IOU = 0.5       # a prediction of another class must overlap a label this well to be about the same object
LOOSE_IOU = 0.1       # a prediction of the label's class overlapping it this much is about the same object
CLOSE_IOU = 0.85      # without pixel sizes, a same-class pair overlapping this well counts as the same box
DEFAULT_TOLERANCE_PX = 3.0  # box edges this many pixels apart are the same box
TOLERANCE_FRACTION = 0.05   # ... or this fraction of the object's longer side, when that is more
DEFAULT_MIN_CONF = 0.5
KINDS = ("missing_label", "wrong_class", "box_mismatch", "unmatched_label")
SAVE_INTERVAL = 20.0  # seconds between saves while analyzing
VERSION = 1


class LabelQAManager:
    def __init__(self, dataset: Dataset, predictor: PredictorManager):
        self.dataset = dataset
        self.predictor = predictor
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.job: dict = {"state": "idle", "done": 0, "total": 0, "failed": 0, "error": None}
        self.result: dict | None = None  # the stored analysis (see _new_result)
        self._cache: dict[tuple, tuple] = {}  # (image key, tolerance) -> (inputs, findings)
        self._load()
        dataset.add_schema_listener(self._schema_changed)

    # ---------- state ----------

    @property
    def available(self) -> bool:
        return self.dataset.category != "classification" and self.predictor.status != "unavailable"

    @property
    def _path(self) -> Path:
        return self.dataset.root / WORKBENCH_DIR / "labelqa.json"

    def _key(self, path: Path) -> str:
        # a string prefix test: Path.relative_to for every analyzed image takes long on large datasets
        text, root = str(path), os.path.join(str(self.dataset.root), "")
        if os.path.normcase(text).startswith(os.path.normcase(root)):
            text = text[len(root):]
        return text.replace(os.sep, "/") if os.sep != "/" else text

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict) or data.get("version") != VERSION or not isinstance(data.get("images"), dict):
            return
        data["mapping"] = {int(key): value for key, value in (data.get("mapping") or {}).items()}
        data["names"] = {int(key): value for key, value in (data.get("names") or {}).items()}
        data.setdefault("dismissed", {})
        self.result = data

    def _save(self) -> None:
        with self._lock:
            if self.result is None:
                return
            text = json.dumps({**self.result, "mapping": {str(key): value for key, value in self.result["mapping"].items()}, "names": {str(key): value for key, value in self.result["names"].items()}}, separators=(",", ":"))
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self.dataset._atomic_write(self._path, text)
        except OSError:
            pass  # results can be recomputed; a failed save must not break the analysis

    def payload(self) -> dict:
        predictor = self.predictor
        with self._lock:
            result = self.result
            summary = None
            if result is not None:
                summary = {
                    "model": result["model"],
                    "model_name": result["model_name"],
                    "created": result["created"],
                    "finished": result.get("finished"),
                    "split": result["split"],
                    "started_by": result.get("started_by"),
                    "images": len(result["images"]),
                    "unmapped": sorted(result["names"][model_id] for model_id, target in result["mapping"].items() if target is None and model_id in result["names"]),
                }
            return {
                "available": self.available,
                "category": self.dataset.category,
                "job": dict(self.job),
                "predictor": {"status": predictor.status, "model": predictor.model_path, "max_points": predictor.max_points},
                "result": summary,
            }

    def busy_reason(self) -> str | None:
        return "Label QA is analyzing images" if self.job["state"] == "running" else None

    # ---------- analysis ----------

    def start(self, split: str | None = None) -> dict:
        if self.dataset.category == "classification":
            raise DatasetError("Label QA needs boxes or polygons; classification datasets are not supported")
        predictor = self.predictor
        predictor.reload()  # a model unloaded while idle comes back
        with predictor._lock:
            if predictor.status != "ready" or predictor.backend is None:
                raise DatasetError(predictor.unloaded or "Load a model first: pick a trained run or a .pt file above")
            if predictor.job["state"] == "running":
                raise DatasetError("A prediction run is in progress; wait for it to finish")
            backend, model_path, mapping = predictor.backend, predictor.model_path, dict(predictor.mapping)
        with self.dataset._lock:
            records = [record for record in self.dataset.images.values() if split in (None, "all") or record.split == split]
        if not records:
            raise DatasetError("No images in this split")
        with self._lock:
            if self.job["state"] == "running":
                raise DatasetError("Label QA is already running")
            try:
                model_mtime = Path(model_path).stat().st_mtime_ns
            except OSError:
                model_mtime = None
            previous = self.result
            # dismissals answer this model's findings; keep them when the same weights run again
            same_model = previous is not None and previous.get("model") == model_path and previous.get("model_mtime") == model_mtime
            self.result = {
                "version": VERSION,
                "model": model_path,
                "model_name": _model_name(Path(model_path)),
                "model_mtime": model_mtime,
                "names": dict(backend.names),
                "mapping": mapping,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "started_by": identity.who(),
                "finished": None,
                "split": split or "all",
                "images": {},
                "dismissed": previous["dismissed"] if same_model else {},
            }
            self._cache.clear()
            self.job = {"state": "running", "done": 0, "total": len(records), "failed": 0, "error": None}
            self._cancel.clear()
            predictor.external_busy = "Label QA is analyzing images"
            self._thread = threading.Thread(target=self._run, args=(records, backend), daemon=True)
            self._thread.start()
        return self.payload()

    def cancel(self) -> dict:
        with self._lock:
            if self.job["state"] != "running":
                raise DatasetError("Label QA is not running")
            self._cancel.set()
        return self.payload()

    def stop(self, timeout: float) -> None:
        """Cancel a running analysis and wait (up to `timeout` seconds) for its current image."""
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
                raise DatasetError("Label QA is running; cancel it first")
            self.result = None
            self._cache.clear()
            self.job = {"state": "idle", "done": 0, "total": 0, "failed": 0, "error": None}
        self._path.unlink(missing_ok=True)
        return self.payload()

    def _run(self, records: list[ImageRecord], backend) -> None:
        try:
            accepts_conf = "conf" in inspect.signature(backend.predict).parameters
        except (TypeError, ValueError):
            accepts_conf = False
        last_save = time.monotonic()
        try:
            for record in records:
                if self._cancel.is_set():
                    break
                try:
                    with self.predictor.backend_lock:
                        raw = backend.predict(record.path, conf=QA_CONF) if accepts_conf else backend.predict(record.path)
                    stamp = self.dataset.image_stamp(record)  # as the dataset knows it: what listings compare with
                    if stamp is None:
                        raise FileNotFoundError(record.path)
                    mtime = stamp[0]
                except Exception:  # one unreadable image must not stop the analysis
                    with self._lock:
                        self.job["failed"] += 1
                        self.job["done"] += 1
                    continue
                predictions = [[int(item["class_id"]), round(float(item["confidence"]), 4), *(round(float(value), 6) for value in item["points"])] for item in raw]
                with self._lock:
                    self.result["images"][self._key(record.path)] = {"id": record.id, "mtime": mtime, "predictions": predictions}
                    self.job["done"] += 1
                if time.monotonic() - last_save > SAVE_INTERVAL:
                    self._save()
                    last_save = time.monotonic()
            with self._lock:
                self.job["state"] = "cancelled" if self._cancel.is_set() else "done"
                self.result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception as exc:  # surfaced to the UI, never crashes the server
            with self._lock:
                self.job["state"] = "error"
                self.job["error"] = str(exc)
        finally:
            self.predictor.external_busy = None
            self._save()

    def _schema_changed(self, mapping: dict[int, int | None] | None) -> None:
        """Class ids were renumbered: keep the model -> dataset class mapping pointing at the same classes."""
        with self._lock:
            if self.result is None:
                return
            names = self.dataset.names
            if mapping is None:
                self.result["mapping"] = {model_id: target if target in names else None for model_id, target in self.result["mapping"].items()}
            else:
                self.result["mapping"] = {model_id: None if target is None else mapping.get(target) for model_id, target in self.result["mapping"].items()}
            self._cache.clear()
        self._save()

    # ---------- findings ----------

    def _image_findings(self, key: str, entry: dict, record: ImageRecord, tolerance: float = 0.0) -> list[dict] | None:
        """Every finding of one image at any confidence (None when the image changed since it was analyzed).

        tolerance 0 reports every box difference: a superset of what any listing shows, which is
        what looking a finding up by its key needs. Whether the image changed is judged by the time the
        dataset knows (opening, rescan): a stat per image made every listing wait on a network drive.
        """
        stamp = self.dataset.image_stamp(record)
        if stamp is None or stamp[0] != entry["mtime"]:
            return None
        dismissed = tuple(self.result["dismissed"].get(key, ()))
        inputs = (id(entry), self.dataset._revisions.get(record.label_path), dismissed, self.dataset.schema_generation, record.width, record.height)
        cached = self._cache.get((key, tolerance))
        if cached and cached[0] == inputs:
            return cached[1]
        found = compute_findings(self.dataset.category, record.annotations, self._predictions(entry), self.dataset.names, (record.width, record.height), tolerance)
        findings = [finding for finding in found if finding["key"] not in dismissed]
        for finding in findings:
            finding.update(image_id=record.id, image_name=record.path.name, split=record.split)
        self._cache[(key, tolerance)] = (inputs, findings)
        return findings

    def _predictions(self, entry: dict) -> list[dict]:
        mapping, names = self.result["mapping"], self.result["names"]
        predictions = []
        for model_class, confidence, *points in entry["predictions"]:
            target = mapping.get(model_class)
            if target is not None and target in self.dataset.names:
                predictions.append({"class_id": target, "model_class_id": model_class, "model_class_name": names.get(model_class, str(model_class)), "confidence": confidence, "points": points})
        return predictions

    def findings(self, kind: str | None = None, split: str | None = None, min_conf: float = DEFAULT_MIN_CONF, offset: int = 0, limit: int = 60, tolerance: float = DEFAULT_TOLERANCE_PX) -> dict:
        """Findings across the analyzed images, most suspicious first."""
        if kind is not None and kind not in KINDS:
            raise DatasetError(f"Unknown finding kind {kind}")
        with self.dataset._lock, self._lock:
            if self.result is None:
                return {"total": 0, "images": 0, "stale": 0, "counts": {name: 0 for name in KINDS}, "items": []}
            items, stale = [], 0
            by_key = None  # images analyzed under another id (opened again after a move): found by path
            for key, entry in self.result["images"].items():
                record = self.dataset.images.get(entry.get("id"))
                if record is None or self._key(record.path) != key:
                    if by_key is None:
                        by_key = {self._key(candidate.path): candidate for candidate in self.dataset.images.values()}
                    record = by_key.get(key)
                if record is None:
                    continue
                if split not in (None, "all") and record.split != split:
                    continue
                found = self._image_findings(key, entry, record, tolerance)
                if found is None:
                    stale += 1
                    continue
                items.extend(finding for finding in found if _confident(finding, min_conf))
            counts = {name: 0 for name in KINDS}
            for finding in items:
                counts[finding["kind"]] += 1
            if kind:
                items = [finding for finding in items if finding["kind"] == kind]
            items.sort(key=lambda finding: (-finding["score"], finding["image_name"], finding["key"]))
            return {
                "total": len(items),
                "images": len({finding["image_id"] for finding in items}),
                "stale": stale,
                "counts": counts,
                "items": [self._shaped(finding, self.dataset.images[finding["image_id"]]) for finding in items[offset : offset + limit]],
            }

    def _finding(self, image_id: str, key: str) -> tuple[ImageRecord, dict, str]:
        record = self.dataset.require_image(image_id)
        if self.result is None:
            raise DatasetError("Run Label QA first")
        image_key = self._key(record.path)
        entry = self.result["images"].get(image_key)
        found = self._image_findings(image_key, entry, record) if entry else None
        finding = next((item for item in found or () if item["key"] == key), None)
        if finding is None:
            raise DatasetError("This finding no longer applies; the labels or the image changed")
        return record, finding, image_key

    def apply(self, image_id: str, key: str) -> dict:
        """Make the label agree with the model: add the predicted object, take its class or box, or delete the label."""
        with self.dataset._lock:
            with self._lock:
                record, finding, _ = self._finding(image_id, key)
            finding = self._shaped(finding, record)
            annotations = [annotation.as_dict() for annotation in record.annotations]
            label, prediction = finding["label"], finding["prediction"]
            if finding["kind"] == "missing_label":
                annotations.append({"class_id": prediction["class_id"], "points": prediction["points"]})
            else:
                target = next(annotation for annotation in annotations if annotation["id"] == label["annotation_id"])
                if finding["kind"] == "wrong_class":
                    target["class_id"] = prediction["class_id"]
                elif finding["kind"] == "box_mismatch":
                    target["points"] = prediction["points"]
                else:
                    annotations.remove(target)
            detail = self.dataset.replace_annotations(record.id, annotations)
        return {"detail": detail, "kind": finding["kind"]}

    def dismiss(self, image_id: str, key: str) -> dict:
        with self.dataset._lock, self._lock:
            _, finding, image_key = self._finding(image_id, key)
            self.result["dismissed"].setdefault(image_key, []).append(finding["key"])
        self._save()
        return {"dismissed": key}

    def crop(self, image_id: str, key: str, color) -> bytes:
        with self.dataset._lock, self._lock:
            record, finding, _ = self._finding(image_id, key)
            finding = self._shaped(finding, record)
        return render_finding(record.path, self.dataset.category, finding, color)

    def _shaped(self, finding: dict, record: ImageRecord) -> dict:
        """A copy of the finding with the model's polygon as the predictor's point limit makes it (keys stay as found)."""
        finding = dict(finding)
        if finding["prediction"] is not None:
            finding["prediction"] = {**finding["prediction"], "points": self.predictor.shape(record, finding["prediction"]["points"])}
        return finding


def _confident(finding: dict, min_conf: float) -> bool:
    """Findings that rest on the model being sure of something need its confidence; an unmatched label does not."""
    return finding["kind"] == "unmatched_label" or finding["prediction"]["confidence"] >= min_conf


def _model_name(path: Path) -> str:
    # a training run's weights: Ultralytics keeps them in <run>/weights/, RF-DETR in the run folder itself
    for run_dir in (path.parent.parent, path.parent):
        try:
            name = json.loads((run_dir / "workbench" / "run.json").read_text(encoding="utf-8")).get("name")
        except (OSError, ValueError, AttributeError):
            continue
        if name:
            return f"{name} · {path.name}"
    return path.name


def _signature(kind: str, label: Annotation | None, prediction: dict | None, category: str) -> str:
    parts = [kind]
    for class_id, points in ((label.class_id, label.points) if label else (None, None), (prediction["class_id"], prediction["points"]) if prediction else (None, None)):
        parts.append("-" if points is None else f"{class_id}:" + ",".join(f"{value:.3f}" for value in _bbox(category, points)))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def compute_findings(category: str, annotations: list[Annotation], predictions: list[dict], names: dict, image_size: tuple[int, int] = (0, 0), tolerance: float = DEFAULT_TOLERANCE_PX) -> list[dict]:
    """Pair labels with predictions and report every disagreement (confidence filtering happens later).

    Pairing is one greedy pass over candidate pairs, best first by confidence × IoU: a prediction of
    the label's class that overlaps it at all (IoU >= LOOSE_IOU), or one of another class that overlaps
    it well (IoU >= MATCH_IOU). So a confident box of the right class a few pixels off is not outbid by
    a weak guess of another class that happens to overlap a little better, and a confident other-class
    box still beats a stray same-class one. A same-class pair is a finding only when the boxes differ by
    more than the tolerance (see _box_deviation). Unpaired predictions that overlap a label well are other
    guesses at an object that has a label, not missing labels.
    """
    labels = [annotation for annotation in annotations if annotation.class_id in names and annotation.points]
    label_boxes = [_bbox(category, label.points) for label in labels]
    prediction_boxes = [_bbox(category, prediction["points"]) for prediction in predictions]
    iou_of = {(li, pi): iou for li, label_box in enumerate(label_boxes) for pi, prediction_box in enumerate(prediction_boxes) if (iou := _bbox_iou(label_box, prediction_box)) > 0}

    def same(li: int, pi: int) -> bool:
        return labels[li].class_id == predictions[pi]["class_id"]

    candidates = sorted(
        ((predictions[pi]["confidence"] * iou, same(li, pi), iou, li, pi) for (li, pi), iou in iou_of.items() if iou >= (LOOSE_IOU if same(li, pi) else MATCH_IOU)),
        reverse=True,
    )
    label_match: dict[int, int] = {}
    prediction_match: dict[int, int] = {}
    for _, _, _, li, pi in candidates:
        if li not in label_match and pi not in prediction_match:
            label_match[li], prediction_match[pi] = pi, li
    findings: list[dict] = []

    def add(kind: str, score: float, li: int | None, pi: int | None, **extra) -> None:
        label = labels[li] if li is not None else None
        prediction = predictions[pi] if pi is not None else None
        findings.append({
            "key": _signature(kind, label, prediction, category),
            "kind": kind,
            "score": round(score, 4),
            "iou": round(iou_of.get((li, pi), 0.0), 3) if li is not None and pi is not None else None,
            "label": {"annotation_id": label.id, "class_id": label.class_id, "points": list(label.points)} if label else None,
            "prediction": dict(prediction) if prediction else None,
            **extra,
        })

    for li, pi in label_match.items():
        iou, confidence = iou_of[(li, pi)], predictions[pi]["confidence"]
        if not same(li, pi):
            add("wrong_class", confidence, li, pi)
            continue
        deviation, allowed = _box_deviation(label_boxes[li], prediction_boxes[pi], image_size, tolerance)
        if deviation is None:
            # no pixel size known (not probed yet): fall back to overlap alone
            if iou < CLOSE_IOU:
                add("box_mismatch", confidence * (1 - iou), li, pi)
        elif deviation > allowed:
            add("box_mismatch", confidence * (1 - iou), li, pi, deviation_px=round(deviation, 1))
    for pi, prediction in enumerate(predictions):
        if pi in prediction_match or any(iou_of.get((li, pi), 0.0) >= MATCH_IOU for li in range(len(labels))):
            continue
        add("missing_label", prediction["confidence"], None, pi)
    for li in range(len(labels)):
        if li in label_match:
            continue
        support = max((predictions[pi]["confidence"] for (other, pi), iou in iou_of.items() if other == li and iou >= LOOSE_IOU), default=0.0)
        # weighted below a confident miss: a weak model also fails to find objects that are labelled right
        add("unmatched_label", 0.9 * (1 - support), li, None)
    # many low-confidence predictions are noise; keep the strongest few per image
    return sorted(findings, key=lambda finding: -finding["score"])[:50]


def _box_deviation(label_box, prediction_box, image_size: tuple[int, int], tolerance: float) -> tuple[float | None, float]:
    """How far apart (pixels) the furthest pair of edges is, and how far is still "the same box".

    The allowance is the tolerance in pixels, or TOLERANCE_FRACTION of the label's longer side when
    that is more, so large objects are not flagged for a few pixels either.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        return None, 0.0
    deviation = max(abs(label_box[0] - prediction_box[0]) * width, abs(label_box[2] - prediction_box[2]) * width,
                    abs(label_box[1] - prediction_box[1]) * height, abs(label_box[3] - prediction_box[3]) * height)
    longer_side = max((label_box[2] - label_box[0]) * width, (label_box[3] - label_box[1]) * height)
    return deviation, max(tolerance, TOLERANCE_FRACTION * longer_side)


def render_finding(path: Path, category: str, finding: dict, color) -> bytes:
    """A crop around the finding: the label solid in its class color, the prediction dashed in white."""
    shapes = [(finding["label"], False), (finding["prediction"], True)]
    boxes = [_bbox(category, shape["points"]) for shape, _ in shapes if shape]
    left, top = min(box[0] for box in boxes), min(box[1] for box in boxes)
    right, bottom = max(box[2] for box in boxes), max(box[3] for box in boxes)
    with Image.open(path) as source:
        image = upright_rgb(source)
    width, height = image.size
    pad_x, pad_y = max((right - left) * 0.35, 24 / width), max((bottom - top) * 0.35, 24 / height)
    box = (max(0, int((left - pad_x) * width)), max(0, int((top - pad_y) * height)), min(width, int((right + pad_x) * width) + 1), min(height, int((bottom + pad_y) * height) + 1))
    crop = image.crop(box)
    draw = ImageDraw.Draw(crop)
    for shape, dashed in shapes:
        if not shape:
            continue
        if category == "detection":
            l, t, r, b = _bbox(category, shape["points"])
            outline = [(l, t), (r, t), (r, b), (l, b)]
        else:
            outline = list(zip(shape["points"][::2], shape["points"][1::2]))
        points = [(x * width - box[0], y * height - box[1]) for x, y in outline]
        points.append(points[0])
        if dashed:
            _dashed(draw, points, "#000000", 4)
            _dashed(draw, points, "#ffffff", 2)
        else:  # a dark halo keeps the outline visible on an object of the same color
            draw.line(points, fill="#000000", width=5)
            draw.line(points, fill=color(shape["class_id"]), width=3)
    crop.thumbnail((420, 320))
    output = io.BytesIO()
    crop.save(output, "JPEG", quality=88)
    return output.getvalue()


def _dashed(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], fill: str, width: int, dash: float = 7, gap: float = 5) -> None:
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if not length:
            continue
        position = 0.0
        while position < length:
            end = min(position + dash, length)
            draw.line([(x1 + (x2 - x1) * position / length, y1 + (y2 - y1) * position / length), (x1 + (x2 - x1) * end / length, y1 + (y2 - y1) * end / length)], fill=fill, width=width)
            position = end + gap
