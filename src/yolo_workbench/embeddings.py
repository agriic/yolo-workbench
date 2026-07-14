"""FiftyOne (Voxel51) embeddings for ground-truth object patches (brain key "gt_viz")."""

from __future__ import annotations

import importlib.util
import threading

from .dataset import Dataset


def fiftyone_available() -> bool:
    return importlib.util.find_spec("fiftyone") is not None


class EmbeddingsManager:
    """Computes and caches a 3D patch-embedding visualization of the dataset."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.status = "idle" if fiftyone_available() else "unavailable"
        self.error: str | None = None
        self.items: list[dict] | None = None

    def payload(self) -> dict:
        return {
            "status": self.status,
            "error": self.error,
            "brain_key": "gt_viz",
            "dimensions": 3,
            "items": self.items or [],
        }

    def start(self) -> dict:
        with self._lock:
            if self.status == "unavailable":
                self.error = "fiftyone is not installed. Run 'uv sync' to install all dependencies"
            elif self.status != "computing":
                self.status = "computing"
                self.error = None
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
        return self.payload()

    def _run(self) -> None:
        try:
            items = compute_gt_viz(self.dataset)
            with self._lock:
                self.items = items
                self.status = "ready"
        except Exception as exc:  # surfaced to the UI, never crashes the server
            with self._lock:
                self.status = "error"
                self.error = str(exc)


def compute_gt_viz(dataset: Dataset) -> list[dict]:
    import fiftyone as fo
    import fiftyone.brain as fob

    name = f"yolo-workbench-{dataset.session_id}"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    ds = fo.Dataset(name)
    try:
        samples = []
        for record in dataset.images.values():
            detections = []
            for annotation in record.annotations:
                if dataset.category == "detection":
                    cx, cy, w, h = annotation.points
                    box = [cx - w / 2, cy - h / 2, w, h]
                else:
                    xs, ys = annotation.points[::2], annotation.points[1::2]
                    if not xs:
                        continue
                    box = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                detection = fo.Detection(label=dataset.names.get(annotation.class_id, str(annotation.class_id)), bounding_box=box)
                detection["workbench"] = "\x1f".join([record.id, annotation.id, str(annotation.class_id), record.split, record.path.name])
                detections.append(detection)
            if detections:
                samples.append(fo.Sample(filepath=str(record.path), ground_truth=fo.Detections(detections=detections)))
        if not samples:
            raise RuntimeError("The dataset has no annotations to embed")
        ds.add_samples(samples, progress=False)

        label_ids = ds.values("ground_truth.detections.id", unwind=True)
        workbench = ds.values("ground_truth.detections.workbench", unwind=True)
        meta = dict(zip(label_ids, workbench))
        # PCA cannot return more components than there are input samples. UMAP
        # also becomes unstable for very small datasets, so compute as many
        # real components as possible and pad the remainder below.
        num_dims = min(3, len(label_ids))
        method = "umap" if len(label_ids) > num_dims + 1 and importlib.util.find_spec("umap") else "pca"
        results = fob.compute_visualization(
            ds,
            patches_field="ground_truth",
            brain_key="gt_viz",
            method=method,
            num_dims=num_dims,
            progress=False,
        )

        result_ids = getattr(results, "label_ids", None)
        result_ids = list(result_ids) if result_ids is not None and len(result_ids) else list(label_ids)
        points = _normalize_3d_points(results.points)

        items = []
        for label_id, (x, y, z) in zip(result_ids, points):
            raw = meta.get(label_id)
            if not raw:
                continue
            image_id, annotation_id, class_id, split, image_name = raw.split("\x1f")
            items.append({
                "x": x,
                "y": y,
                "z": z,
                "image_id": image_id,
                "annotation_id": annotation_id,
                "class_id": int(class_id),
                "split": split,
                "image_name": image_name,
            })
        return items
    finally:
        fo.delete_dataset(name)


def _normalize_3d_points(points) -> list[tuple[float, float, float]]:
    """Normalizes each component to [0, 1] and pads points to three axes."""
    rows = [tuple(float(value) for value in point) for point in points]
    if not rows:
        return []

    dimensions = min(3, max(len(row) for row in rows))
    columns = [[row[index] if index < len(row) else 0.0 for row in rows] for index in range(dimensions)]
    normalized = []
    for column in columns:
        low, high = min(column), max(column)
        span = high - low
        normalized.append([(value - low) / span if span else 0.5 for value in column])

    while len(normalized) < 3:
        normalized.append([0.5] * len(rows))

    return list(zip(*normalized))
