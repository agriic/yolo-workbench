"""Model-assisted labeling: run an Ultralytics or RF-DETR model, hold predictions in memory until accepted."""

from __future__ import annotations

import gc
import importlib.util
import json
import logging
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import frameworks, identity, libraries
from .dataset import Dataset, DatasetError
from .imaging import upright_rgb
from .masks import drop_specks, mask_outlines, simplify_polygon
from .models import ImageRecord

log = logging.getLogger(__name__)


def ultralytics_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def rfdetr_available() -> bool:
    return importlib.util.find_spec("rfdetr") is not None


def free_gpu_memory() -> None:
    """Hand the memory of dropped models back to the GPU (only when torch is already loaded), and the freed
    heap back to the operating system: glibc keeps it otherwise, so loading and unloading models (SAM, a
    prediction model, around every training run) would grow a long-running server's memory."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    _trim_heap()


def _trim_heap() -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # not glibc (e.g. musl): nothing to trim


MODEL_EXTENSIONS = {".pt", ".pth"}
UPLOAD_MODEL_EXTENSIONS = MODEL_EXTENSIONS | {".onnx"}
UNLOAD_WAIT = 10.0  # seconds an unload waits for a batch run's current image


def model_upload_target(dataset: Dataset, name: str) -> Path:
    """Where an uploaded model goes: the dataset's models/ folder (a project's models/), under a free name.
    Weights files are pickles: loading one runs code from it, so only trusted users may upload them."""
    if dataset.category == "classification":
        raise DatasetError("Model-assisted labeling is not supported for classification datasets")
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(name).replace("\\", "/").rsplit("/", 1)[-1]).strip(" ._")
    suffix = Path(clean).suffix.lower()
    if not clean or suffix not in UPLOAD_MODEL_EXTENSIONS:
        raise DatasetError(f"Upload a model file ({', '.join(sorted(UPLOAD_MODEL_EXTENSIONS))})")
    directory = (dataset.yaml_path if dataset.yaml_path.is_dir() else dataset.yaml_path.parent) / "models"
    directory.mkdir(parents=True, exist_ok=True)
    target, counter = directory / clean, 2
    while target.exists():
        target = directory / f"{Path(clean).stem}-{counter}{suffix}"
        counter += 1
    return target
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


def _public(items: list[dict]) -> list[dict]:
    """Pending predictions as the API shows them (without the model's unsimplified polygon)."""
    return [{key: value for key, value in item.items() if key != "raw_points"} for item in items]


def _same_file(a: str, b: str) -> bool:
    """Whether two model paths name the same file, whatever separators or relative form they use."""
    try:
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    except (OSError, RuntimeError):  # e.g. a symlink loop
        return a == b


class UltralyticsBackend:
    """Thin wrapper so PredictorManager only depends on names + predict()."""

    framework = "ultralytics"
    default_conf = 0.25

    def __init__(self, path: Path, category: str, conf: float, iou: float):
        libraries.load("ultralytics.models")  # what `from ultralytics import YOLO` imports
        from ultralytics import YOLO

        self.category = category
        self.conf = conf
        self.iou = iou
        self.model = YOLO(str(path))
        self.names = {int(key): str(value) for key, value in self.model.names.items()}

    def predict(self, image_path: Path, conf: float | None = None) -> list[dict]:
        """Predictions at the backend's confidence threshold, or at `conf` (Label QA looks lower)."""
        result = self.model.predict(str(image_path), conf=self.conf if conf is None else conf, iou=self.iou, verbose=False)[0]
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


def _nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float) -> list[int]:
    """Indices kept by greedy per-class non-maximum suppression, highest score first."""
    order = list(np.argsort(-scores))
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kept: list[int] = []
    while order:
        best = order.pop(0)
        kept.append(int(best))
        rest = np.array(order, dtype=int)
        if not len(rest):
            break
        width = np.clip(np.minimum(boxes[rest, 2], boxes[best, 2]) - np.maximum(boxes[rest, 0], boxes[best, 0]), 0, None)
        height = np.clip(np.minimum(boxes[rest, 3], boxes[best, 3]) - np.maximum(boxes[rest, 1], boxes[best, 1]), 0, None)
        overlap = width * height / np.maximum(areas[rest] + areas[best] - width * height, 1e-12)
        order = [int(index) for index, value, same in zip(rest, overlap, classes[rest] == classes[best]) if not (same and value >= iou)]
    return kept


class RFDETRBackend:
    """RF-DETR weights (a training run's checkpoint_best_total.pth, or RF-DETR's pretrained ones) behind
    the same names + predict() as UltralyticsBackend. RF-DETR ignores EXIF orientation, so it is given the
    upright image. A DETR has no NMS of its own; at low thresholds (Label QA) its queries overlap, so the
    predictor's IoU setting is applied as per-class NMS."""

    framework = "rfdetr"
    default_conf = 0.5  # RF-DETR's own default threshold; its scores run lower than YOLO's

    def __init__(self, path: Path, category: str, conf: float, iou: float):
        libraries.load("rfdetr")
        from rfdetr import RFDETR

        self.category = category
        self.conf = conf
        self.iou = iou
        self.model = RFDETR.from_checkpoint(str(path))
        if category == "segmentation" and not self.model.model_config.segmentation_head:
            raise DatasetError("This RF-DETR model only detects boxes; a segmentation dataset needs an RF-DETR segmentation model (rfdetr-seg-…)")
        custom = getattr(self.model.model, "class_names", None)
        if custom:
            names = dict(enumerate(custom))
        else:  # RF-DETR's COCO-pretrained weights predict COCO's own (sparse) category ids
            from rfdetr.assets.coco_classes import COCO_CLASSES

            names = COCO_CLASSES
        self.names = {int(key): str(value) for key, value in names.items()}

    def predict(self, image_path: Path, conf: float | None = None) -> list[dict]:
        with Image.open(image_path) as opened:
            image = upright_rgb(opened)
        width, height = image.size
        detections = self.model.predict(image, threshold=self.conf if conf is None else conf, include_source_image=False)
        if not len(detections):
            return []
        boxes = np.asarray(detections.xyxy, dtype=float)
        scores = np.asarray(detections.confidence, dtype=float)
        classes = np.asarray(detections.class_id, dtype=int)
        masks = detections.mask if self.category == "segmentation" else None
        predictions: list[dict] = []
        for index in _nms(boxes, scores, classes, self.iou):
            if masks is not None:
                outlines = mask_outlines(drop_specks(masks[index]))
                if not outlines or len(outlines[0]) < 6:
                    continue
                points = outlines[0]
            else:
                left, top = _clamp(boxes[index, 0] / width), _clamp(boxes[index, 1] / height)
                right, bottom = _clamp(boxes[index, 2] / width), _clamp(boxes[index, 3] / height)
                if right - left <= 0 or bottom - top <= 0:
                    continue
                points = [(left + right) / 2, (top + bottom) / 2, right - left, bottom - top]
            predictions.append({"class_id": int(classes[index]), "confidence": float(scores[index]), "points": points})
        return predictions


class PredictorManager:
    """Loads a model on demand and keeps per-image predictions in memory until accepted.

    Predictions never touch label files directly: acceptance goes through
    Dataset.replace_annotations, so backups, undo, and conflict detection all apply.
    """

    def __init__(self, dataset: Dataset, backend_factory=None):
        self.dataset = dataset
        self._backend_factory = backend_factory or self._default_factory
        self._available = (backend_factory is not None or ultralytics_available() or rfdetr_available()) and dataset.category != "classification"
        self._lock = threading.Lock()
        # one inference at a time on the shared model (batch runs, the editor's predict, Label QA)
        self.backend_lock = threading.Lock()
        # why the model must not be swapped or run in a batch right now (set by Label QA while it analyzes)
        self.external_busy: str | None = None
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._failed_image_ids: list[str] = []
        self.status = "idle" if self._available else "unavailable"
        self.error: str | None = None
        self.backend = None
        self.model_path: str | None = None
        self.unloaded: str | None = None  # why the model was dropped (a training run took the GPU, or it sat idle)
        self.loaded_by: dict | None = None  # who loaded the model ({name, display_name}, with sign-in)
        self._generation = 0  # bumped by unload, so a load finishing afterwards is discarded
        self.last_used = time.monotonic()  # for the idle unload (and least-recently-used unload across projects)
        self.conf = 0.25
        self.iou = 0.45
        # segmentation: pending polygons are simplified to at most this many points (None = as the model drew them)
        saved = self._settings().get("max_points")
        self.max_points: int | None = saved if isinstance(saved, int) and saved >= 3 else None
        self.mapping: dict[int, int | None] = {}  # model class id -> dataset class id (None = unmapped)
        self.predictions: dict[str, list[dict]] = {}  # image_id -> pending predictions
        self.job: dict = self._new_job()
        self._counter = 0
        dataset.add_schema_listener(self._schema_changed)

    def _schema_changed(self, mapping: dict[int, int | None] | None) -> None:
        """Follow dataset class renumbering so the mapping and pending predictions stay correct."""
        with self._lock:
            names = self.dataset.names
            if mapping is None:
                self.mapping = {model_id: target if target in names else None for model_id, target in self.mapping.items()}
            else:
                self.mapping = {model_id: None if target is None else mapping.get(target) for model_id, target in self.mapping.items()}
            if self.backend:
                # a class added or renamed to match a model class gets mapped automatically
                taken = {target for target in self.mapping.values() if target is not None}
                for model_id, target in self._auto_map(self.backend.names).items():
                    if self.mapping.get(model_id) is None and target is not None and target not in taken:
                        self.mapping[model_id] = target
                        taken.add(target)
            for items in self.predictions.values():
                for item in items:
                    item["class_id"] = self.mapping.get(item["model_class_id"])

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
        if frameworks.weights_framework(path) == "rfdetr":
            if not rfdetr_available():
                raise DatasetError('These are RF-DETR weights, and rfdetr is not installed (uv add "rfdetr[train]")')
            return RFDETRBackend(path, category, conf, iou)
        return UltralyticsBackend(path, category, conf, iou)

    def payload(self) -> dict:
        with self._lock:
            job = {**self.job, "failures": [dict(item) for item in self.job["failures"]]}
            return {
                "status": self.status,
                "error": self.error,
                "model": {"path": self.model_path, "names": self.backend.names if self.backend else {}, "framework": getattr(self.backend, "framework", None)},
                "conf": self.conf,
                "iou": self.iou,
                "max_points": self.max_points,
                "unloaded": self.unloaded,
                "loaded_by": self.loaded_by if self.backend else None,
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
            if self.external_busy:
                raise DatasetError(f"{self.external_busy}; wait for it to finish before loading a model")
            self.status = "loading"
            self.error = None
            generation = self._generation
        previous = getattr(self.backend, "framework", None)
        try:
            backend = self._backend_factory(model_path, self.dataset.category, conf if conf is not None else self.conf, iou if iou is not None else self.iou)
            if conf is None and getattr(backend, "framework", None) != previous and getattr(backend, "default_conf", None) is not None:
                # switching between YOLO and RF-DETR: their scores are calibrated differently, so start from the model's own default
                conf = backend.default_conf
                backend.conf = conf
        except Exception as exc:
            with self._lock:
                self.status = "error"
                self.error = str(exc)
            raise DatasetError(f"Failed to load model: {exc}") from exc
        with self._lock:
            if generation != self._generation:
                raise DatasetError(self.unloaded or "The model was unloaded while loading")
            self.backend = backend
            self.unloaded = None
            self.loaded_by = identity.who()
            self.model_path = model_path.as_posix()
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
            self.last_used = time.monotonic()
        self._remember_model(model_path.as_posix())
        log.info("predictor: loaded %s for %s", model_path.name, self.dataset.root.name)
        return self.payload()

    def unload(self, run: str, reason: str | None = None) -> bool:
        """Drop the model so a training run gets the GPU (or because it sat idle: `reason`); cancels a batch run
        and waits for its current image. Pending predictions, the class mapping and the model path stay.
        False when nothing was loaded."""
        with self._lock:
            if self.backend is None and self.status != "loading":
                return False
            self._generation += 1
            if self.job["state"] == "running":
                self._cancel_event.set()
                self.job["cancel_requested"] = True
            thread = self._thread
            self.backend = None
            self.status = "idle"
            self.unloaded = reason or f"The model was unloaded to free memory for training run '{run}'; load it again"
        if thread is not None:
            thread.join(UNLOAD_WAIT)
        log.info("predictor: unloaded the model of %s (%s)", self.dataset.root.name, reason or f"training run '{run}'")
        return True

    def busy(self) -> bool:
        """In use right now: a batch run, Label QA's analysis, or an inference."""
        with self._lock:
            running = self.job["state"] == "running" or bool(self.external_busy)
        return running or self.backend_lock.locked()

    def unload_if_idle(self, idle_seconds: float) -> bool:
        """Unload a model nobody used for `idle_seconds` (the maintenance thread calls this)."""
        if self.busy():
            self.last_used = time.monotonic()  # the countdown starts when the batch or Label QA ends
            return False
        with self._lock:
            if self.backend is None or self.status != "ready" or time.monotonic() - self.last_used < idle_seconds:
                return False
        minutes = round(idle_seconds / 60)
        return self.unload("", reason=f"The model was unloaded after {minutes} minute{'s' if minutes != 1 else ''} without use to free memory; load it again")

    def close(self, timeout: float) -> None:
        """Shutdown: cancel a batch run, wait for it, and drop the model."""
        with self._lock:
            self._generation += 1
            if self.job["state"] == "running":
                self._cancel_event.set()
                self.job["cancel_requested"] = True
            thread = self._thread
            self.backend = None
            if self.status != "unavailable":
                self.status = "idle"
        if thread is not None:
            thread.join(timeout)

    @property
    def _recents_path(self) -> Path:
        return self.dataset.root / ".yolo-workbench" / "predictor.json"

    def _settings(self) -> dict:
        try:
            data = json.loads(self._recents_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_settings(self, **changes) -> None:
        try:
            self._recents_path.parent.mkdir(parents=True, exist_ok=True)
            self._recents_path.write_text(json.dumps({**self._settings(), **changes}), encoding="utf-8")
        except OSError:
            pass  # recents and options are a convenience; failing to persist them must never break anything

    def _recent_models(self) -> list[str]:
        recents = self._settings().get("recent_models", [])
        return [str(item) for item in recents] if isinstance(recents, list) else []

    def _remember_model(self, path: str) -> None:
        # older recents may name the same file differently (backslashes on Windows, relative paths)
        recents = [path] + [item for item in self._recent_models() if not _same_file(item, path)]
        self._save_settings(recent_models=recents[:10])

    def set_options(self, max_points: int | None) -> dict:
        """Change the polygon point limit; pending predictions are re-simplified from the model's own polygons."""
        with self._lock:
            self.max_points = max_points
            for image_id, items in self.predictions.items():
                record = self.dataset.images.get(image_id)
                for item in items:
                    item["points"] = self.shape(record, item["raw_points"])
        self._save_settings(max_points=max_points)
        return self.payload()

    def shape(self, record: ImageRecord | None, points: list[float]) -> list[float]:
        """A model's polygon as it is shown and accepted: simplified to max_points (segmentation only)."""
        if self.dataset.category != "segmentation" or not self.max_points:
            return points
        width, height = (record.width, record.height) if record else (1, 1)
        return simplify_polygon(points, self.max_points, width, height)

    def _trained_models(self) -> dict[Path, str]:
        """The best weights of the dataset's training runs (newest first) -> "<run name> · best.pt" (· best (RF-DETR))."""
        from .training import runs_root_for

        root = runs_root_for(self.dataset.yaml_path)
        found: list[tuple[float, Path, str]] = []
        if root.is_dir():
            # workbench runs sit directly in runs/; `yolo train` runs one level deeper (runs/detect/train2)
            for run_dir in [*root.glob("*/"), *root.glob("*/*/")]:
                if run_dir.name.startswith(".") or not run_dir.is_dir():
                    continue
                framework = frameworks.of_run(run_dir)
                weights = framework.weights(run_dir).get("best")
                if weights is None:
                    continue
                try:
                    name = json.loads((run_dir / "workbench" / "run.json").read_text(encoding="utf-8")).get("name") or run_dir.name
                    mtime = weights.stat().st_mtime
                except (OSError, ValueError, AttributeError):
                    name, mtime = run_dir.name, 0.0
                found.append((mtime, weights.resolve(), f"{name} · {'best (RF-DETR)' if framework is frameworks.RFDETR else 'best.pt'}"))
        return {path: label for _, path, label in sorted(found, key=lambda item: item[0], reverse=True)}

    def discover_models(self) -> dict:
        """Recently used models first, then this dataset's trained runs, then .pt files found near the dataset, cwd, and recents' directories."""
        items: list[dict] = []
        seen: set[Path] = set()
        trained = self._trained_models()
        with self._lock:
            model_path = self.model_path
        # matched here as resolved paths: the client may have loaded it as "D:/x/best.pt" or "D:\x\best.pt"
        loaded = Path(model_path).resolve() if model_path else None

        def add(path: Path, source: str) -> None:
            resolved = path.resolve()
            if resolved in seen or resolved.suffix.lower() not in MODEL_EXTENSIONS:
                return
            if source == "found" and (resolved.parent / "training_config.json").is_file():
                return  # an RF-DETR run's other weights (EMA, last): its best is listed as the run's
            try:
                stat = resolved.stat()
            except OSError:
                return
            seen.add(resolved)
            items.append({"path": resolved.as_posix(), "name": trained.get(resolved, resolved.name), "size": stat.st_size, "mtime": int(stat.st_mtime), "source": source, "trained": resolved in trained, "loaded": resolved == loaded})

        recents = self._recent_models()
        for recent in recents:
            add(Path(recent), "recent")
        for path in trained:
            add(path, "run")
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
                raise DatasetError(self.unloaded or "Load a model first")
            if self.job["state"] == "running":
                raise DatasetError("A prediction run is already in progress")
            if self.external_busy:
                raise DatasetError(f"{self.external_busy}; wait for it to finish")
            records = self._select(image_ids, split, only_unlabeled)
            self.job = self._new_job("running", len(records))
            self.job["started_by"] = identity.who()  # the predictions are shared: everyone sees who ran the batch
            self._failed_image_ids = []
            self._cancel_event.clear()
            self.last_used = time.monotonic()
            self._thread = threading.Thread(target=self._run_batch, args=(records, self.backend), daemon=True)
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

    def _run_batch(self, records: list[ImageRecord], backend) -> None:
        for index, record in enumerate(records):
            if self._cancel_event.is_set():
                with self._lock:
                    self.job["cancelled"] = len(records) - index
                break
            try:
                with self.backend_lock:
                    raw = backend.predict(record.path)
                with self._lock:
                    items = [self._prediction(record, item, backend.names) for item in raw]
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
            self.last_used = time.monotonic()

    def _prediction(self, record: ImageRecord, item: dict, names: dict) -> dict:
        self._counter += 1
        model_class_id = int(item["class_id"])
        points = [float(value) for value in item["points"]]
        return {
            "id": f"pred:{record.id}:{self._counter}",
            "model_class_id": model_class_id,
            "model_class_name": names.get(model_class_id, str(model_class_id)),
            "class_id": self.mapping.get(model_class_id),
            "confidence": float(item["confidence"]),
            "points": self.shape(record, points),
            "raw_points": points,  # kept server-side so a changed point limit starts from the model's polygon
            "raw_count": len(points) // 2,
        }

    def predict_image(self, image_id: str) -> dict:
        """Run the model on one image synchronously — the editor's per-image predict action."""
        record = self.dataset.require_image(image_id)
        with self._lock:
            if self.status != "ready" or not self.backend:
                raise DatasetError(self.unloaded or "Load a model first")
            if self.job["state"] == "running":
                raise DatasetError("A prediction run is already in progress")
            backend = self.backend
            self.last_used = time.monotonic()
        try:
            with self.backend_lock:
                raw = backend.predict(record.path)
        except Exception as exc:
            raise DatasetError(f"Prediction failed: {exc}") from exc
        with self._lock:
            self.predictions[record.id] = self._filter_new(record, [self._prediction(record, item, backend.names) for item in raw])
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
            return {"image_id": record.id, "items": _public(self.predictions.get(record.id, []))}

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
        return {"accepted": len(chosen), "detail": detail, "predictions": _public(kept)}

    def reject(self, image_id: str, prediction_ids: list[str] | None = None) -> dict:
        record = self.dataset.require_image(image_id)
        with self._lock:
            chosen, kept = self._take(record.id, prediction_ids)
            self.predictions[record.id] = kept
            return {"rejected": len(chosen), "predictions": _public(kept)}
