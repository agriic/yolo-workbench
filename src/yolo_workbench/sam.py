"""Smart annotation: Segment Anything (SAM 2.1) turns clicked points or a rough box into an annotation.

One SamManager serves the whole app (both modes): one model for every user and project. Loading is
started explicitly (the editor's Smart annotation button) and runs in a background thread,
since the first use downloads the weights. It is unloaded when nobody used it for the idle time
(the server's maintenance), after which the editor loads it again by itself on the next use, or for
a training run (the desktop; then it stays off until turned on again). Each image is encoded once (the slow step) and its
features kept in a small LRU, so every further click only runs the prompt decoder.
Results are never written here: the editor adds them as annotations through the normal save path.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import libraries
from .dataset import DatasetError
from .masks import drop_specks, mask_box, mask_outlines, mask_trace
from .media_cache import default_weights_dir
from .predictor import ultralytics_available

log = logging.getLogger(__name__)

LARGE, BASE = "sam2.1_l.pt", "sam2.1_b.pt"
FEATURE_CACHE = 8  # encoded images kept (shared by everyone using SAM); SAM 2.1 features are ~16 MB each
MIN_AREA = 4  # px; a smaller mask counts as "nothing found"


def load_upright(path: Path) -> np.ndarray:
    """The image as BGR, EXIF orientation applied — how the browser shows it and Ultralytics loads it."""
    with Image.open(path) as opened:
        rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    return np.ascontiguousarray(rgb[..., ::-1])


class UltralyticsSamBackend:
    """SAM 2.1 through Ultralytics' SAM2Predictor: encode() once per image, segment() per prompt."""

    def __init__(self, path: Path):
        libraries.load("ultralytics.models.sam")
        from ultralytics.models.sam import SAM2Predictor

        self.predictor = SAM2Predictor(overrides={"model": str(path), "imgsz": 1024, "conf": 0.0, "save": False, "verbose": False})
        self.predictor.setup_model(verbose=False)
        self.device = str(self.predictor.device)

    def encode(self, image: np.ndarray):
        import torch

        with torch.inference_mode():
            self.predictor.set_image(image)
            features = self.predictor.features
        self.predictor.reset_image()
        return features

    def segment(self, features, shape: tuple[int, int], points: list[list[float]], labels: list[int], box: list[float] | None) -> tuple[np.ndarray, float]:
        """Best mask (bool, H×W) and its score; points and box are in pixels."""
        import torch

        # one object prompted by all points: shapes (1, N, 2) / (1, N)
        kwargs = {"points": [points], "labels": [labels]} if points else {}
        if box is not None:
            kwargs["bboxes"] = [box]
        # a lone click is ambiguous (part or whole?): let SAM propose three masks and keep the best
        multimask = len(points) == 1 and box is None
        with torch.inference_mode():
            masks, boxes = self.predictor.inference_features(features, shape, multimask_output=multimask, **kwargs)
        if masks is None or not len(masks):
            return np.zeros(shape, dtype=bool), 0.0
        scores = boxes[:, 4]
        best = int(scores.argmax())
        return masks[best].cpu().numpy(), float(scores[best])


def _gpu_available() -> bool:
    try:
        libraries.load("torch")
        import torch

        return torch.cuda.is_available() or torch.backends.mps.is_available()
    except Exception:
        return False


def weights_path(name: str) -> Path:
    """An existing copy (ours, or Ultralytics' weights dir) or where to download it."""
    libraries.load("ultralytics.utils")
    from ultralytics.utils import SETTINGS

    weights_dir = default_weights_dir()  # shared with training's pretrained weights
    for directory in (weights_dir, Path(SETTINGS["weights_dir"])):
        if (directory / name).is_file():
            return directory / name
    return weights_dir / name


def _default_factory(name: str):
    libraries.load("ultralytics.utils.downloads")
    from ultralytics.utils.downloads import attempt_download_asset

    # an absolute path keeps the download out of the server's cwd
    target = weights_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    return UltralyticsSamBackend(Path(attempt_download_asset(str(target))))


class SamManager:
    def __init__(self, backend_factory=None, gpu: bool | None = None):
        self._factory = backend_factory or _default_factory
        self._gpu = gpu  # None: detect when loading
        self._available = backend_factory is not None or ultralytics_available()
        self._lock = threading.Lock()  # state fields
        self._run_lock = threading.Lock()  # the model itself: one encode/decode at a time
        self._thread: threading.Thread | None = None
        self.status = "idle" if self._available else "unavailable"
        self.message: str | None = None
        self.error: str | None = None
        self.model: str | None = None
        self.device: str | None = None
        self.backend = None
        self.unloaded: str | None = None  # why the model was dropped when it must be turned on again (a training run took the GPU)
        self.idle_unloaded = False  # dropped because nobody used it: the editor loads it again by itself
        self._generation = 0  # bumped by unload, so a load finishing afterwards is discarded
        self.last_used = time.monotonic()
        self._features: OrderedDict[tuple, tuple] = OrderedDict()  # (path, mtime_ns, size) -> (features, (h, w))

    def payload(self) -> dict:
        with self._lock:
            return {"status": self.status, "message": self.message, "error": self.error, "model": self.model, "device": self.device, "unloaded": self.unloaded, "idle_unloaded": self.idle_unloaded}

    def candidates(self) -> list[str]:
        """Large when a GPU can run it at interactive speed, base as the fallback."""
        gpu = self._gpu if self._gpu is not None else _gpu_available()
        return [LARGE, BASE] if gpu else [BASE]

    def start(self) -> dict:
        """Begin loading the model in the background (no-op while loading or once ready)."""
        with self._lock:
            if self.status == "unavailable":
                raise DatasetError("Smart annotation needs ultralytics. Run 'uv sync' to install all dependencies")
            if self.status in ("idle", "error"):
                self.status = "loading"
                self.error = None
                self.unloaded = None
                self.idle_unloaded = False
                self._thread = threading.Thread(target=self._load, args=(self._generation,), name="sam-load", daemon=True)
                self._thread.start()
        return self.payload()

    def _load(self, generation: int) -> None:
        errors = []
        for name in self.candidates():
            with self._lock:
                if generation != self._generation:
                    return
                self.message = f"Loading {name} (downloaded on first use)…"
            try:
                backend = self._factory(name)
            except Exception as exc:  # e.g. no network for the download, or out of GPU memory: try the next one
                errors.append(f"{name}: {exc}")
                continue
            with self._lock:
                if generation != self._generation:
                    return  # unloaded meanwhile: the backend is dropped with this thread
                self.backend = backend
                self.model = name
                self.device = getattr(backend, "device", None)
                self.status = "ready"
                self.message = None
                self.last_used = time.monotonic()
            log.info("sam: loaded %s on %s", name, self.device)
            return
        with self._lock:
            if generation != self._generation:
                return
            self.status = "error"
            self.message = None
            self.error = "Failed to load SAM — " + "; ".join(errors)

    def unload(self, run: str, reason: str | None = None, idle: bool = False) -> bool:
        """Drop the model (and the encoded images) so a training run gets the GPU (it stays off until turned on
        again), or because nobody used it (`idle`: the editor loads it again on the next use). False when
        nothing was loaded."""
        with self._lock:
            if self.status not in ("ready", "loading"):
                return False
            self._generation += 1
            self.backend = None
            self.model = None
            self.device = None
            self.status = "idle"
            self.message = None
            self.idle_unloaded = idle
            self.unloaded = None if idle else reason or f"Smart annotation was unloaded to free memory for training run '{run}'; turn it on again to load it"
        with self._run_lock:  # an in-flight encode or decode finishes first
            self._features.clear()
        log.info("sam: unloaded (%s)", reason or f"training run '{run}'")
        return True

    def unload_if_idle(self, idle_seconds: float) -> bool:
        """Unload a model nobody prompted for `idle_seconds` (the maintenance thread calls this); it loads
        again with the next use."""
        with self._lock:
            if self.status != "ready" or self._run_lock.locked() or time.monotonic() - self.last_used < idle_seconds:
                return False
        minutes = idle_seconds / 60
        return self.unload("", reason=f"{minutes:g} minute{'s' if minutes != 1 else ''} without use; loads again when used", idle=True)

    def close(self, timeout: float) -> None:
        """Shutdown: drop the model; a load in progress is discarded when it finishes."""
        with self._lock:
            self._generation += 1
            self.backend = None
            if self.status in ("ready", "loading"):
                self.status = "idle"
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    def _require_backend(self):
        with self._lock:
            if self.status != "ready":
                raise DatasetError({
                    "loading": "Smart annotation is still loading",
                    "error": self.error or "Smart annotation failed to load",
                    "unavailable": "Smart annotation needs ultralytics",
                }.get(self.status, self.unloaded or "Start smart annotation first"))
            return self.backend

    def _encoded(self, path: Path) -> tuple:
        """(features, (h, w)) for the image, encoding it unless cached. Caller holds _run_lock."""
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        cached = self._features.get(key)
        if cached is not None:
            self._features.move_to_end(key)
            return cached
        backend = self._require_backend()
        image = load_upright(path)
        entry = (backend.encode(image), image.shape[:2])
        self._features[key] = entry
        while len(self._features) > FEATURE_CACHE:
            self._features.popitem(last=False)
        return entry

    def encode(self, path: Path) -> dict:
        self._require_backend()
        self.last_used = time.monotonic()
        with self._run_lock:
            self._encoded(path)
        return {"encoded": True}

    def segment(self, path: Path, category: str, points: list[dict], box: list[float] | None, max_points: int | None = None) -> dict:
        """Prompt with normalized points ({x, y, positive}) and/or a normalized x1,y1,x2,y2 box. A segmentation
        polygon gets at most `max_points` vertices, fitted to the mask outline (None: the usual light simplification)."""
        if category == "classification":
            raise DatasetError("Smart annotation is not available for classification datasets")
        if not points and box is None:
            raise DatasetError("Click the object or drag a box around it first")
        if points and not any(point["positive"] for point in points) and box is None:
            raise DatasetError("Click at least once on the object")
        if box is not None:
            x1, y1, x2, y2 = (min(max(float(value), 0.0), 1.0) for value in box)
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        backend = self._require_backend()
        self.last_used = time.monotonic()
        with self._run_lock:
            features, (height, width) = self._encoded(path)
            pixels = [[point["x"] * width, point["y"] * height] for point in points]
            labels = [1 if point["positive"] else 0 for point in points]
            box_px = None if box is None else [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
            mask, score = backend.segment(features, (height, width), pixels, labels, box_px)
        mask = drop_specks(mask)
        # `mask` traces SAM's mask (every part) for the editor to fill, so the next click can steer toward
        # the object's shape; `points` is what ⏎ adds: a box around every part, or the largest part as the polygon
        parts, points = [], []
        if mask.sum() >= MIN_AREA:
            parts = mask_trace(mask)
            points = mask_box(mask) if category == "detection" else next(iter(mask_outlines(mask, max_points)), [])
        if not parts or not points:
            return {"found": False, "score": score, "mask": [], "points": []}
        return {"found": True, "score": round(score, 4), "mask": parts, "points": points}
