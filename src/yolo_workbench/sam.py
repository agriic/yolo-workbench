"""Smart annotation: Segment Anything (SAM 2.1) turns clicked points or a rough box into an annotation.

One SamManager serves the whole app (both modes), so the model is loaded once. Loading is
started explicitly (the editor's Smart annotation button) and runs in a background thread,
since the first use downloads the weights. Each image is encoded once (the slow step) and its
features kept in a small LRU, so every further click only runs the prompt decoder.
Results are never written here: the editor adds them as annotations through the normal save path.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .dataset import DatasetError
from .predictor import ultralytics_available

LARGE, BASE = "sam2.1_l.pt", "sam2.1_b.pt"
FEATURE_CACHE = 4  # encoded images kept; SAM 2.1 features are ~16 MB each
MIN_AREA = 4  # px; a smaller mask counts as "nothing found"


def load_upright(path: Path) -> np.ndarray:
    """The image as BGR, EXIF orientation applied — how the browser shows it and Ultralytics loads it."""
    with Image.open(path) as opened:
        rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    return np.ascontiguousarray(rgb[..., ::-1])


class UltralyticsSamBackend:
    """SAM 2.1 through Ultralytics' SAM2Predictor: encode() once per image, segment() per prompt."""

    def __init__(self, path: Path):
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
        import torch

        return torch.cuda.is_available() or torch.backends.mps.is_available()
    except Exception:
        return False


WEIGHTS_DIR = Path.home() / ".cache" / "yolo-workbench" / "weights"  # shared with training's pretrained weights


def weights_path(name: str) -> Path:
    """An existing copy (ours, or Ultralytics' weights dir) or where to download it."""
    from ultralytics.utils import SETTINGS

    for directory in (WEIGHTS_DIR, Path(SETTINGS["weights_dir"])):
        if (directory / name).is_file():
            return directory / name
    return WEIGHTS_DIR / name


def _default_factory(name: str):
    from ultralytics.utils.downloads import attempt_download_asset

    # an absolute path keeps the download out of the server's cwd
    target = weights_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    return UltralyticsSamBackend(Path(attempt_download_asset(str(target))))


def drop_specks(mask: np.ndarray) -> np.ndarray:
    """The mask without the specks SAM scatters around an object, often far from it (its 256×256 low-res
    mask is up-sampled, leaving islands of a few cells). A part stays when it is at least 4×4 low-res cells
    and 1% of the largest part; the largest part always stays."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    speck = max((max(mask.shape) / 64) ** 2, areas.max() / 100)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = (areas >= speck) | (areas == areas.max())
    return keep[labels]


def mask_outlines(mask: np.ndarray) -> list[list[float]]:
    """External contour of each part of the mask, simplified, as normalized x,y pairs (≥3 points), largest first."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    outlines = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        # ~1 px of tolerance on small objects, proportionally more on large ones: few enough vertices to edit
        epsilon = max(1.0, 0.0015 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) >= 3:
            outlines.append([round(min(max(value, 0.0), 1.0), 6) for x, y in polygon.tolist() for value in (x / width, y / height)])
    return outlines


def mask_box(mask: np.ndarray) -> list[float]:
    """Normalized cx, cy, w, h around every mask pixel (all parts, e.g. an object split by an occluder)."""
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    left, right = xs.min() / width, (xs.max() + 1) / width
    top, bottom = ys.min() / height, (ys.max() + 1) / height
    return [round(float(value), 6) for value in ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)]


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
        self._features: OrderedDict[tuple, tuple] = OrderedDict()  # (path, mtime_ns, size) -> (features, (h, w))

    def payload(self) -> dict:
        with self._lock:
            return {"status": self.status, "message": self.message, "error": self.error, "model": self.model, "device": self.device}

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
                self._thread = threading.Thread(target=self._load, name="sam-load", daemon=True)
                self._thread.start()
        return self.payload()

    def _load(self) -> None:
        errors = []
        for name in self.candidates():
            with self._lock:
                self.message = f"Loading {name} (downloaded on first use)…"
            try:
                backend = self._factory(name)
            except Exception as exc:  # e.g. no network for the download, or out of GPU memory: try the next one
                errors.append(f"{name}: {exc}")
                continue
            with self._lock:
                self.backend = backend
                self.model = name
                self.device = getattr(backend, "device", None)
                self.status = "ready"
                self.message = None
            return
        with self._lock:
            self.status = "error"
            self.message = None
            self.error = "Failed to load SAM — " + "; ".join(errors)

    def _require_backend(self):
        with self._lock:
            if self.status != "ready":
                raise DatasetError({
                    "loading": "Smart annotation is still loading",
                    "error": self.error or "Smart annotation failed to load",
                    "unavailable": "Smart annotation needs ultralytics",
                }.get(self.status, "Start smart annotation first"))
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
        with self._run_lock:
            self._encoded(path)
        return {"encoded": True}

    def segment(self, path: Path, category: str, points: list[dict], box: list[float] | None) -> dict:
        """Prompt with normalized points ({x, y, positive}) and/or a normalized x1,y1,x2,y2 box."""
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
        with self._run_lock:
            features, (height, width) = self._encoded(path)
            pixels = [[point["x"] * width, point["y"] * height] for point in points]
            labels = [1 if point["positive"] else 0 for point in points]
            box_px = None if box is None else [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
            mask, score = backend.segment(features, (height, width), pixels, labels, box_px)
        mask = drop_specks(mask)
        outlines = mask_outlines(mask) if mask.sum() >= MIN_AREA else []
        if not outlines:
            return {"found": False, "score": score, "outlines": [], "points": []}
        # outlines show exactly what is added: a box around every part, or the largest part as the polygon
        if category == "detection":
            return {"found": True, "score": round(score, 4), "outlines": outlines, "points": mask_box(mask)}
        return {"found": True, "score": round(score, 4), "outlines": outlines[:1], "points": outlines[0]}
