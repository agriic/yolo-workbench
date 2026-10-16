"""Label QA: rank likely label mistakes by where a trained model disagrees with the labels.

An analysis runs the predictor's loaded model over a split at a low confidence and stores the
raw predictions (in .yolo-workbench/labelqa.json, so results survive a restart). Findings are
derived from those predictions and the *current* labels each time they are listed, so fixing a
label makes its finding disappear without re-running the model:

- missing_label    a confident prediction no label matches (likely an object nobody labelled)
- wrong_class      a label and a confident prediction overlap well but disagree on the class
- box_mismatch     a confident prediction of the label's class overlaps it only loosely
- unmatched_label  a label nothing the model predicted overlaps (likely spurious or mislabelled)

Matching is greedy by box IoU (segmentation polygons use their bounding boxes), same class first.
Applying a finding's fix goes through Dataset.replace_annotations, so backups, undo, conflict
detection, and the training lock apply. Dismissing one ("the label is right") is remembered.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .dataset import WORKBENCH_DIR, Dataset, DatasetError
from .imaging import upright_rgb
from .models import Annotation, ImageRecord
from .predictor import PredictorManager, _bbox, _bbox_iou

QA_CONF = 0.05        # predictions are kept down to this confidence, so "the model saw nothing here" means something
MATCH_IOU = 0.5       # a label and a prediction are the same object
LOOSE_IOU = 0.1       # below MATCH_IOU but above this, a same-class prediction says the box is drawn differently
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
        self._cache: dict[str, tuple] = {}  # image key -> (inputs, findings)
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
        root = self.dataset.root
        return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()

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
                    "images": len(result["images"]),
                    "unmapped": sorted(result["names"][model_id] for model_id, target in result["mapping"].items() if target is None and model_id in result["names"]),
                }
            return {
                "available": self.available,
                "category": self.dataset.category,
                "job": dict(self.job),
                "predictor": {"status": predictor.status, "model": predictor.model_path},
                "result": summary,
            }

    def busy_reason(self) -> str | None:
        return "Label QA is analyzing images" if self.job["state"] == "running" else None

    # ---------- analysis ----------

    def start(self, split: str | None = None) -> dict:
        if self.dataset.category == "classification":
            raise DatasetError("Label QA needs boxes or polygons; classification datasets are not supported")
        predictor = self.predictor
        with predictor._lock:
            if predictor.status != "ready" or predictor.backend is None:
                raise DatasetError("Load a model first: pick a trained run or a .pt file above")
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
                    mtime = record.path.stat().st_mtime_ns
                except Exception:  # one unreadable image must not stop the analysis
                    with self._lock:
                        self.job["failed"] += 1
                        self.job["done"] += 1
                    continue
                predictions = [[int(item["class_id"]), round(float(item["confidence"]), 4), *(round(float(value), 6) for value in item["points"])] for item in raw]
                with self._lock:
                    self.result["images"][self._key(record.path)] = {"id": record.id, "mtime": mtime, "predictions": predictions}
                    self._cache.pop(self._key(record.path), None)
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

    def _record_for(self, key: str, entry: dict) -> ImageRecord | None:
        record = self.dataset.images.get(entry.get("id"))
        if record is not None and self._key(record.path) == key:
            return record
        return next((candidate for candidate in self.dataset.images.values() if self._key(candidate.path) == key), None)

    def _image_findings(self, key: str, entry: dict, record: ImageRecord) -> list[dict] | None:
        """Every finding of one image at any confidence (None when the image changed since it was analyzed)."""
        try:
            mtime = record.path.stat().st_mtime_ns
        except OSError:
            return None
        if mtime != entry["mtime"]:
            return None
        dismissed = tuple(self.result["dismissed"].get(key, ()))
        inputs = (self.dataset._revisions.get(record.label_path), dismissed, self.dataset.schema_generation)
        cached = self._cache.get(key)
        if cached and cached[0] == inputs:
            return cached[1]
        findings = [finding for finding in compute_findings(self.dataset.category, record.annotations, self._predictions(entry), self.dataset.names) if finding["key"] not in dismissed]
        for finding in findings:
            finding.update(image_id=record.id, image_name=record.path.name, split=record.split)
        self._cache[key] = (inputs, findings)
        return findings

    def _predictions(self, entry: dict) -> list[dict]:
        mapping, names = self.result["mapping"], self.result["names"]
        predictions = []
        for model_class, confidence, *points in entry["predictions"]:
            target = mapping.get(model_class)
            if target is not None and target in self.dataset.names:
                predictions.append({"class_id": target, "model_class_id": model_class, "model_class_name": names.get(model_class, str(model_class)), "confidence": confidence, "points": points})
        return predictions

    def findings(self, kind: str | None = None, split: str | None = None, min_conf: float = DEFAULT_MIN_CONF, offset: int = 0, limit: int = 60) -> dict:
        """Findings across the analyzed images, most suspicious first."""
        if kind is not None and kind not in KINDS:
            raise DatasetError(f"Unknown finding kind {kind}")
        with self.dataset._lock, self._lock:
            if self.result is None:
                return {"total": 0, "images": 0, "stale": 0, "counts": {name: 0 for name in KINDS}, "items": []}
            items, stale = [], 0
            for key, entry in self.result["images"].items():
                record = self._record_for(key, entry)
                if record is None:
                    continue
                if split not in (None, "all") and record.split != split:
                    continue
                found = self._image_findings(key, entry, record)
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
                "items": [dict(finding) for finding in items[offset : offset + limit]],
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
        return render_finding(record.path, self.dataset.category, finding, color)


def _confident(finding: dict, min_conf: float) -> bool:
    """Findings that rest on the model being sure of something need its confidence; an unmatched label does not."""
    return finding["kind"] == "unmatched_label" or finding["prediction"]["confidence"] >= min_conf


def _model_name(path: Path) -> str:
    try:
        run_dir = path.parent.parent
        name = json.loads((run_dir / "workbench" / "run.json").read_text(encoding="utf-8")).get("name")
        if name:
            return f"{name} · {path.name}"
    except (OSError, ValueError, AttributeError):
        pass
    return path.name


def _signature(kind: str, label: Annotation | None, prediction: dict | None, category: str) -> str:
    parts = [kind]
    for class_id, points in ((label.class_id, label.points) if label else (None, None), (prediction["class_id"], prediction["points"]) if prediction else (None, None)):
        parts.append("-" if points is None else f"{class_id}:" + ",".join(f"{value:.3f}" for value in _bbox(category, points)))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def compute_findings(category: str, annotations: list[Annotation], predictions: list[dict], names: dict) -> list[dict]:
    """Match labels and predictions and report every disagreement (confidence filtering happens later)."""
    labels = [annotation for annotation in annotations if annotation.class_id in names and annotation.points]
    label_boxes = [_bbox(category, label.points) for label in labels]
    prediction_boxes = [_bbox(category, prediction["points"]) for prediction in predictions]
    pairs = sorted(
        ((iou, li, pi) for li, label_box in enumerate(label_boxes) for pi, prediction_box in enumerate(prediction_boxes) if (iou := _bbox_iou(label_box, prediction_box)) > 0),
        reverse=True,
    )
    label_match: dict[int, int] = {}
    prediction_match: dict[int, int] = {}
    for same_class in (True, False):  # a same-class match wins over a better-overlapping other class
        for iou, li, pi in pairs:
            if iou < MATCH_IOU:
                break
            if li in label_match or pi in prediction_match:
                continue
            if same_class and labels[li].class_id != predictions[pi]["class_id"]:
                continue
            label_match[li], prediction_match[pi] = pi, li
    iou_of = {(li, pi): iou for iou, li, pi in pairs}
    findings: list[dict] = []

    def add(kind: str, score: float, li: int | None, pi: int | None) -> None:
        label = labels[li] if li is not None else None
        prediction = predictions[pi] if pi is not None else None
        findings.append({
            "key": _signature(kind, label, prediction, category),
            "kind": kind,
            "score": round(score, 4),
            "iou": round(iou_of.get((li, pi), 0.0), 3) if li is not None and pi is not None else None,
            "label": {"annotation_id": label.id, "class_id": label.class_id, "points": list(label.points)} if label else None,
            "prediction": dict(prediction) if prediction else None,
        })

    for li, pi in label_match.items():
        if labels[li].class_id != predictions[pi]["class_id"]:
            add("wrong_class", predictions[pi]["confidence"], li, pi)
    explained: set[int] = set()
    for pi, prediction in enumerate(predictions):
        if pi in prediction_match:
            continue
        loose = [(iou, li) for (li, other), iou in iou_of.items() if other == pi and li not in label_match and li not in explained and labels[li].class_id == prediction["class_id"] and iou >= LOOSE_IOU]
        if loose:
            iou, li = max(loose)
            explained.add(li)
            add("box_mismatch", prediction["confidence"] * (1 - iou), li, pi)
        else:
            add("missing_label", prediction["confidence"], None, pi)
    for li in range(len(labels)):
        if li in label_match or li in explained:
            continue
        support = max((predictions[pi]["confidence"] for (other, pi), iou in iou_of.items() if other == li and iou >= LOOSE_IOU), default=0.0)
        # weighted below a confident miss: a weak model also fails to find objects that are labelled right
        add("unmatched_label", 0.9 * (1 - support), li, None)
    # many low-confidence predictions are noise; keep the strongest few per image
    return sorted(findings, key=lambda finding: -finding["score"])[:50]


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
