"""FiftyOne (Voxel51) embeddings, brain key "gt_viz": ground-truth object patches for
detection/segmentation, whole images for classification."""

from __future__ import annotations

import hashlib
import importlib.util
import threading

from . import identity, libraries
from .dataset import Dataset, object_key


def fiftyone_available() -> bool:
    return importlib.util.find_spec("fiftyone") is not None


class EmbeddingsManager:
    """Computes and caches a 3D patch-embedding visualization of the dataset. The computed objects follow the labels
    by shape (`_follow`), so the plot outlives edits: what is deleted or reshaped drops out until an undo brings it
    back, and what is added is in it after the next computation."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        from .runtime import RUNTIME

        self.enabled = RUNTIME.embeddings_enabled
        self.status = "idle" if self.enabled and fiftyone_available() else "unavailable"
        self.error: str | None = None
        self.items: list[dict] | None = None  # as computed: point, image id and shape (compute_gt_viz)
        self._view: tuple[int, list[dict]] | None = None  # (dataset revision, the items as its labels were then)
        self.source_generation: int | None = None
        self.started_by: dict | None = None

    def payload(self) -> dict:
        with self.dataset._lock, self._lock:  # every change bumps the revision holding the dataset lock
            if self.items is not None and (self._view is None or self._view[0] != self.dataset.revision):
                self._view = (self.dataset.revision, self._follow())
            return {
                "status": self.status,
                "error": self.error,
                "brain_key": "gt_viz",
                "dimensions": 3,
                "source_generation": self.source_generation,
                "started_by": self.started_by,
                "items": self._view[1] if self.items is not None else [],
            }

    def _follow(self) -> list[dict]:
        """The computed objects whose shape is still in their image's labels, with that annotation's current id,
        class and key (ids are line positions: deleting a line renumbers the ones after it). `uid` names an object
        for as long as these items stand. Identical shapes in one image pair up in order."""
        view, unmatched = [], {}
        for uid, item in enumerate(self.items):
            record = self.dataset.images.get(item["image_id"])
            if record is None:
                continue
            if record.id not in unmatched:
                shapes = unmatched[record.id] = {}
                for annotation in record.annotations:
                    shapes.setdefault(tuple(annotation.points), []).append(annotation)
            same = unmatched[record.id].get(item["shape"])
            if not same:
                continue
            annotation = same.pop(0)
            view.append({
                "uid": uid,
                "x": item["x"],
                "y": item["y"],
                "z": item["z"],
                "image_id": record.id,
                "annotation_id": annotation.id,
                "class_id": annotation.class_id,
                "key": object_key(annotation),
                "split": record.split,
                "image_name": record.path.name,
            })
        return view

    def start(self) -> dict:
        with self._lock:
            if self.status == "unavailable":
                self.error = "fiftyone is not installed. Run 'uv sync' to install all dependencies" if self.enabled else "Embeddings are turned off on this server ([embeddings] enabled = false)"
            elif self.status != "computing":
                self.status = "computing"
                self.error = None
                self.started_by = identity.who()
                generation = self.dataset.annotation_generation
                self._thread = threading.Thread(target=self._run, args=(generation,), daemon=True)
                self._thread.start()
        return self.payload()

    def close(self, timeout: float) -> None:
        """Wait (up to `timeout` seconds) for a running computation; FiftyOne/UMAP cannot be interrupted."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    def _run(self, generation: int) -> None:
        try:
            items = compute_gt_viz(self.dataset)
            with self._lock:
                self._view = None
                if generation == self.dataset.annotation_generation:
                    self.items = items
                    self.source_generation = generation
                    self.status = "ready"
                else:
                    self.items = None
                    self.source_generation = None
                    self.status = "idle"
        except Exception as exc:  # surfaced to the UI, never crashes the server
            with self._lock:
                if generation == self.dataset.annotation_generation:
                    self.status = "error"
                    self.error = str(exc)
                else:
                    self.status = "idle"
                    self.error = None


def compute_gt_viz(dataset: Dataset) -> list[dict]:
    """One item per embedded object: its 3D point, image id and annotation shape (the points, as a tuple)."""
    libraries.load("fiftyone", "fiftyone.brain")
    import fiftyone as fo
    import fiftyone.brain as fob

    # several projects can be opened in one server process; keep their FiftyOne datasets apart
    root_hash = hashlib.sha1(str(dataset.root).encode()).hexdigest()[:8]
    name = f"yolo-workbench-{root_hash}-{dataset.session_id}"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    ds = fo.Dataset(name)
    objects = []  # (record, annotation), numbered by each sample's or detection's "workbench" field
    try:
        if dataset.category == "classification":
            # whole-image embeddings: the class covers the entire image, so there are no patches
            samples = []
            for record in dataset.images.values():
                annotation = record.annotations[0]
                sample = fo.Sample(filepath=str(record.path), ground_truth=fo.Classification(label=dataset.names.get(annotation.class_id, str(annotation.class_id))))
                sample["workbench"] = str(len(objects))
                objects.append((record, annotation))
                samples.append(sample)
            if not samples:
                raise RuntimeError("The dataset has no images to embed")
            ds.add_samples(samples, progress=False)
            sample_ids = ds.values("id")
            meta = dict(zip(sample_ids, ds.values("workbench")))
            num_dims = min(3, len(sample_ids))
            method = "umap" if len(sample_ids) > num_dims + 1 and importlib.util.find_spec("umap") else "pca"
            results = fob.compute_visualization(ds, brain_key="gt_viz", method=method, num_dims=num_dims, progress=False)
            result_ids = getattr(results, "sample_ids", None)
            result_ids = list(result_ids) if result_ids is not None and len(result_ids) else list(sample_ids)
            return _build_items(meta, objects, result_ids, _normalize_3d_points(results.points))
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
                detection["workbench"] = str(len(objects))
                objects.append((record, annotation))
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
        return _build_items(meta, objects, result_ids, _normalize_3d_points(results.points))
    finally:
        fo.delete_dataset(name)


def _build_items(meta: dict, objects: list, result_ids: list, points: list[tuple[float, float, float]]) -> list[dict]:
    items = []
    for result_id, (x, y, z) in zip(result_ids, points):
        number = meta.get(result_id)
        if number is None:
            continue
        record, annotation = objects[int(number)]
        items.append({"x": x, "y": y, "z": z, "image_id": record.id, "shape": tuple(annotation.points)})
    return items


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
