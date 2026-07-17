"""Model-assisted labeling: run an Ultralytics model, hold predictions in memory until accepted."""

from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path

from .dataset import Dataset, DatasetError
from .models import ImageRecord


def ultralytics_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


MODEL_EXTENSIONS = {".pt"}
SCAN_DEPTH = 4
SCAN_LIMIT = 50
DEDUP_IOU = 0.6  # predictions matching an existing same-class annotation at or above this IoU are dropped


def _bbox(category: str, points: list[float]) -> tuple[float, float, float, float]:
    if category == "segmentation":
        xs, ys = points[0::2], points[1::2]
        return min(xs), min(ys), max(xs), max(ys)
    cx, cy, w, h = points
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def _matches_existing(category: str, class_id: int | None, points: list[float], annotations) -> bool:
    """True when an existing annotation has the same class and overlaps at IoU >= DEDUP_IOU."""
    if class_id is None:
        return False
    box = _bbox(category, points)
    return any(
        annotation.class_id == class_id and _bbox_iou(box, _bbox(category, annotation.points)) >= DEDUP_IOU
        for annotation in annotations
    )


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


class UltralyticsBackend:
    """Thin wrapper so PredictorManager only depends on names + predict()."""

    def __init__(self, path: Path, category: str, conf: float, iou: float):
        from ultralytics import YOLO

        self.category = category
        self.conf = conf
        self.iou = iou
        self.model = YOLO(str(path))
        self.names = {int(key): str(value) for key, value in self.model.names.items()}

    def predict(self, image_path: Path) -> list[dict]:
        result = self.model.predict(str(image_path), conf=self.conf, iou=self.iou, verbose=False)[0]
        predictions: list[dict] = []
        if result.boxes is None:
            return predictions
        classes = result.boxes.cls.tolist()
        confidences = result.boxes.conf.tolist()
        if self.category == "segmentation":
            polygons = result.masks.xyn if result.masks is not None else []
            for class_id, confidence, polygon in zip(classes, confidences, polygons):
                points = [_clamp(value) for pair in polygon.tolist() for value in pair]
                if len(points) >= 6:
                    predictions.append({"class_id": int(class_id), "confidence": float(confidence), "points": points})
        else:
            for class_id, confidence, (cx, cy, w, h) in zip(classes, confidences, result.boxes.xywhn.tolist()):
                left, top = _clamp(cx - w / 2), _clamp(cy - h / 2)
                right, bottom = _clamp(cx + w / 2), _clamp(cy + h / 2)
                if right - left > 0 and bottom - top > 0:
                    points = [(left + right) / 2, (top + bottom) / 2, right - left, bottom - top]
                    predictions.append({"class_id": int(class_id), "confidence": float(confidence), "points": points})
        return predictions


class PredictorManager:
    """Loads a model on demand and keeps per-image predictions in memory until accepted.

    Predictions never touch label files directly: acceptance goes through
    Dataset.replace_annotations, so backups, undo, and conflict detection all apply.
    """

    def __init__(self, dataset: Dataset, backend_factory=None):
        self.dataset = dataset
        self._backend_factory = backend_factory or self._default_factory
        self._available = (backend_factory is not None or ultralytics_available()) and dataset.category != "classification"
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._failed_image_ids: list[str] = []
        self.status = "idle" if self._available else "unavailable"
        self.error: str | None = None
        self.backend = None
        self.model_path: str | None = None
        self.conf = 0.25
        self.iou = 0.45
        self.mapping: dict[int, int | None] = {}  # model class id -> dataset class id (None = unmapped)
        self.predictions: dict[str, list[dict]] = {}  # image_id -> pending predictions
        self.job: dict = self._new_job()
        self._counter = 0

    @staticmethod
    def _new_job(state: str = "idle", total: int = 0) -> dict:
        return {
            "state": state,
            "done": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "total": total,
            "failures": [],
            "cancel_requested": False,
            "error": None,
        }

    @staticmethod
    def _default_factory(path: Path, category: str, conf: float, iou: float):
        return UltralyticsBackend(path, category, conf, iou)

    def payload(self) -> dict:
        with self._lock:
            job = {**self.job, "failures": [dict(item) for item in self.job["failures"]]}
            return {
                "status": self.status,
                "error": self.error,
                "model": {"path": self.model_path, "names": self.backend.names if self.backend else {}},
                "conf": self.conf,
                "iou": self.iou,
                "mapping": {str(key): value for key, value in self.mapping.items()},
                "job": job,
                "pending": {image_id: len(items) for image_id, items in self.predictions.items() if items},
            }

    def pending_image_ids(self) -> set[str]:
        with self._lock:
            return {image_id for image_id, items in self.predictions.items() if items}

    def load(self, path: str, conf: float | None = None, iou: float | None = None) -> dict:
        if not self._available:
            if self.dataset.category == "classification":
                raise DatasetError("Model-assisted labeling is not supported for classification datasets")
            raise DatasetError("ultralytics is not installed. Run 'uv sync' to install all dependencies")
        model_path = Path(path).expanduser()
        if not model_path.exists():
            raise DatasetError(f"Model file not found: {model_path}")
        with self._lock:
            if self.job["state"] == "running":
                raise DatasetError("A prediction run is in progress; wait for it to finish before loading a model")
            self.status = "loading"
            self.error = None
        try:
            backend = self._backend_factory(model_path, self.dataset.category, conf if conf is not None else self.conf, iou if iou is not None else self.iou)
        except Exception as exc:
            with self._lock:
                self.status = "error"
                self.error = str(exc)
            raise DatasetError(f"Failed to load model: {exc}") from exc
        with self._lock:
            self.backend = backend
            self.model_path = str(model_path)
            if conf is not None:
                self.conf = conf
            if iou is not None:
                self.iou = iou
            self.mapping = self._auto_map(backend.names)
            self.predictions.clear()
            self.job = self._new_job()
            self._failed_image_ids = []
            self._cancel_event.clear()
            self.status = "ready"
        self._remember_model(str(model_path))
        return self.payload()

    @property
    def _recents_path(self) -> Path:
        return self.dataset.root / ".yolo-workbench" / "predictor.json"

    def _recent_models(self) -> list[str]:
        try:
            data = json.loads(self._recents_path.read_text(encoding="utf-8"))
            recents = data.get("recent_models", [])
            return [str(item) for item in recents] if isinstance(recents, list) else []
        except (OSError, ValueError):
            return []

    def _remember_model(self, path: str) -> None:
        recents = [path] + [item for item in self._recent_models() if item != path]
        try:
            self._recents_path.parent.mkdir(parents=True, exist_ok=True)
            self._recents_path.write_text(json.dumps({"recent_models": recents[:10]}), encoding="utf-8")
        except OSError:
            pass  # recents are a convenience; failing to persist them must never break loading

    def discover_models(self) -> dict:
        """Recently used models first, then .pt files found near the dataset, cwd, and recents' directories."""
        items: list[dict] = []
        seen: set[Path] = set()

        def add(path: Path, source: str) -> None:
            resolved = path.resolve()
            if resolved in seen or resolved.suffix.lower() not in MODEL_EXTENSIONS:
                return
            try:
                stat = resolved.stat()
            except OSError:
                return
            seen.add(resolved)
            items.append({"path": str(resolved), "name": resolved.name, "size": stat.st_size, "mtime": int(stat.st_mtime), "source": source})

        recents = self._recent_models()
        for recent in recents:
            add(Path(recent), "recent")
        # also scan the directories models were previously loaded from
        roots = {self.dataset.root, Path.cwd()} | {Path(recent).parent for recent in recents}
        found: list[Path] = []
        for root in roots:
            found.extend(self._scan(root))
        for path in sorted(found, key=lambda item: item.name.casefold()):
            if len(items) >= SCAN_LIMIT:
                break
            add(path, "found")
        return {"items": items}

    @staticmethod
    def _scan(root: Path) -> list[Path]:
        found: list[Path] = []

        def walk(directory: Path, depth: int) -> None:
            if depth > SCAN_DEPTH or len(found) >= SCAN_LIMIT:
                return
            try:
                entries = sorted(directory.iterdir())
            except OSError:
                return
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    walk(entry, depth + 1)
                elif entry.suffix.lower() in MODEL_EXTENSIONS:
                    found.append(entry)

        walk(root, 0)
        return found

    def _auto_map(self, model_names: dict[int, str]) -> dict[int, int | None]:
        by_name = {name.casefold(): class_id for class_id, name in self.dataset.names.items()}
        return {model_id: by_name.get(name.casefold()) for model_id, name in model_names.items()}

    def set_mapping(self, raw: dict) -> dict:
        with self._lock:
            if not self.backend:
                raise DatasetError("Load a model first")
            mapping: dict[int, int | None] = {}
            for key, value in raw.items():
                model_id = int(key)
                if model_id not in self.backend.names:
                    raise DatasetError(f"Unknown model class {model_id}")
                if value is not None and int(value) not in self.dataset.names:
                    raise DatasetError(f"Unknown dataset class {value}")
                mapping[model_id] = None if value is None else int(value)
            self.mapping.update(mapping)
            for items in self.predictions.values():
                for item in items:
                    item["class_id"] = self.mapping.get(item["model_class_id"])
        return self.payload()

    def run(self, image_ids: list[str] | None = None, split: str | None = None, only_unlabeled: bool = False) -> dict:
        with self._lock:
            if self.status != "ready" or not self.backend:
                raise DatasetError("Load a model first")
            if self.job["state"] == "running":
                raise DatasetError("A prediction run is already in progress")
            records = self._select(image_ids, split, only_unlabeled)
            self.job = self._new_job("running", len(records))
            self._failed_image_ids = []
            self._cancel_event.clear()
            self._thread = threading.Thread(target=self._run_batch, args=(records,), daemon=True)
            self._thread.start()
        return self.payload()

    def cancel(self) -> dict:
        with self._lock:
            if self.job["state"] != "running":
                raise DatasetError("No prediction run is in progress")
            self._cancel_event.set()
            self.job["cancel_requested"] = True
        return self.payload()

    def retry_failed(self) -> dict:
        with self._lock:
            if self.job["state"] == "running":
                raise DatasetError("A prediction run is already in progress")
            image_ids = list(self._failed_image_ids)
        if not image_ids:
            raise DatasetError("There are no failed predictions to retry")
        return self.run(image_ids=image_ids)

    def _select(self, image_ids: list[str] | None, split: str | None, only_unlabeled: bool) -> list[ImageRecord]:
        if image_ids is not None:
            missing = [image_id for image_id in image_ids if image_id not in self.dataset.images]
            if missing:
                raise DatasetError(f"Image not found: {missing[0]}")
            records = [self.dataset.images[image_id] for image_id in image_ids]
        else:
            records = [record for record in self.dataset.images.values() if split in (None, "all") or record.split == split]
        if only_unlabeled:
            records = [record for record in records if not record.annotations]
        if not records:
            raise DatasetError("No images match the requested selection")
        return records

    def _run_batch(self, records: list[ImageRecord]) -> None:
        for index, record in enumerate(records):
            if self._cancel_event.is_set():
                with self._lock:
                    self.job["cancelled"] = len(records) - index
                break
            try:
                raw = self.backend.predict(record.path)
                with self._lock:
                    items = [self._prediction(record, item) for item in raw]
                    self.predictions[record.id] = self._filter_new(record, items)
            except Exception as exc:  # surfaced to the UI, never crashes the server
                with self._lock:
                    failure = {"image_id": record.id, "image_name": record.path.name, "error": str(exc)}
                    self.job["failures"].append(failure)
                    self.job["failed"] += 1
                    self.job["done"] += 1
                    self._failed_image_ids.append(record.id)
                continue
            with self._lock:
                self.job["completed"] += 1
                self.job["done"] += 1
        with self._lock:
            self.job["state"] = "cancelled" if self._cancel_event.is_set() else "done"
            self.job["cancel_requested"] = False

    def _prediction(self, record: ImageRecord, item: dict) -> dict:
        self._counter += 1
        model_class_id = int(item["class_id"])
        return {
            "id": f"pred:{record.id}:{self._counter}",
            "model_class_id": model_class_id,
            "model_class_name": self.backend.names.get(model_class_id, str(model_class_id)),
            "class_id": self.mapping.get(model_class_id),
            "confidence": float(item["confidence"]),
            "points": [float(value) for value in item["points"]],
        }

    def predict_image(self, image_id: str) -> dict:
        """Run the model on one image synchronously — the editor's per-image predict action."""
        record = self.dataset.require_image(image_id)
        with self._lock:
            if self.status != "ready" or not self.backend:
                raise DatasetError("Load a model first")
            if self.job["state"] == "running":
                raise DatasetError("A prediction run is already in progress")
        try:
            raw = self.backend.predict(record.path)
        except Exception as exc:
            raise DatasetError(f"Prediction failed: {exc}") from exc
        with self._lock:
            self.predictions[record.id] = self._filter_new(record, [self._prediction(record, item) for item in raw])
        return self.predictions_for(record.id)

    def _filter_new(self, record: ImageRecord, items: list[dict]) -> list[dict]:
        """Drop predictions that duplicate an existing annotation (same class, overlapping geometry)."""
        return [
            item
            for item in items
            if not _matches_existing(self.dataset.category, item["class_id"], item["points"], record.annotations)
        ]

    def predictions_for(self, image_id: str) -> dict:
        record = self.dataset.require_image(image_id)
        with self._lock:
            return {"image_id": record.id, "items": [dict(item) for item in self.predictions.get(record.id, [])]}

    def _take(self, image_id: str, prediction_ids: list[str] | None) -> tuple[list[dict], list[dict]]:
        """Split pending predictions into (chosen, kept) without mutating state."""
        pending = self.predictions.get(image_id, [])
        if prediction_ids is None:
            return list(pending), []
        wanted = set(prediction_ids)
        chosen = [item for item in pending if item["id"] in wanted]
        if len(chosen) != len(wanted):
            raise DatasetError("Prediction not found; it may already be accepted or rejected")
        return chosen, [item for item in pending if item["id"] not in wanted]

    def accept(self, image_id: str, prediction_ids: list[str] | None = None, min_confidence: float | None = None) -> dict:
        record = self.dataset.require_image(image_id)
        with self._lock:
            chosen, kept = self._take(record.id, prediction_ids)
            if min_confidence is not None:
                below = [item for item in chosen if item["confidence"] < min_confidence]
                kept.extend(below)
                chosen = [item for item in chosen if item["confidence"] >= min_confidence]
            if prediction_ids is None:
                # accept-all keeps unmapped predictions pending instead of failing the batch
                kept.extend(item for item in chosen if item["class_id"] is None)
                chosen = [item for item in chosen if item["class_id"] is not None]
            elif any(item["class_id"] is None for item in chosen):
                raise DatasetError("Cannot accept predictions with unmapped classes; adjust the class mapping first")
            # annotations may have changed since prediction time; never write a duplicate
            chosen = self._filter_new(record, chosen)
            if not chosen:
                raise DatasetError("No predictions to accept")
            annotations = [annotation.as_dict() for annotation in record.annotations]
            annotations.extend({"class_id": item["class_id"], "points": item["points"]} for item in chosen)
        detail = self.dataset.replace_annotations(record.id, annotations)
        with self._lock:
            self.predictions[record.id] = kept
        return {"accepted": len(chosen), "detail": detail, "predictions": kept}

    def reject(self, image_id: str, prediction_ids: list[str] | None = None) -> dict:
        record = self.dataset.require_image(image_id)
        with self._lock:
            chosen, kept = self._take(record.id, prediction_ids)
            self.predictions[record.id] = kept
            return {"rejected": len(chosen), "predictions": kept}
