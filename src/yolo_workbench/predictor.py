"""Model-assisted labeling: run an Ultralytics model, hold predictions in memory until accepted."""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

from .dataset import Dataset, DatasetError
from .models import ImageRecord


def ultralytics_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


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
        self._available = backend_factory is not None or ultralytics_available()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.status = "idle" if self._available else "unavailable"
        self.error: str | None = None
        self.backend = None
        self.model_path: str | None = None
        self.conf = 0.25
        self.iou = 0.45
        self.mapping: dict[int, int | None] = {}  # model class id -> dataset class id (None = unmapped)
        self.predictions: dict[str, list[dict]] = {}  # image_id -> pending predictions
        self.job: dict = {"state": "idle", "done": 0, "total": 0, "error": None}
        self._counter = 0

    @staticmethod
    def _default_factory(path: Path, category: str, conf: float, iou: float):
        return UltralyticsBackend(path, category, conf, iou)

    def payload(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "error": self.error,
                "model": {"path": self.model_path, "names": self.backend.names if self.backend else {}},
                "conf": self.conf,
                "iou": self.iou,
                "mapping": {str(key): value for key, value in self.mapping.items()},
                "job": dict(self.job),
                "pending": {image_id: len(items) for image_id, items in self.predictions.items() if items},
            }

    def load(self, path: str, conf: float | None = None, iou: float | None = None) -> dict:
        if not self._available:
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
            self.job = {"state": "idle", "done": 0, "total": 0, "error": None}
            self.status = "ready"
        return self.payload()

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
            self.job = {"state": "running", "done": 0, "total": len(records), "error": None}
            self._thread = threading.Thread(target=self._run_batch, args=(records,), daemon=True)
            self._thread.start()
        return self.payload()

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
        for record in records:
            try:
                raw = self.backend.predict(record.path)
            except Exception as exc:  # surfaced to the UI, never crashes the server
                with self._lock:
                    self.job["state"] = "error"
                    self.job["error"] = f"{record.path.name}: {exc}"
                return
            with self._lock:
                self.predictions[record.id] = [self._prediction(record, item) for item in raw]
                self.job["done"] += 1
        with self._lock:
            self.job["state"] = "done"

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
            self.predictions[record.id] = [self._prediction(record, item) for item in raw]
        return self.predictions_for(record.id)

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
