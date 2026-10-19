import asyncio
import os
import threading

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from test_dataset import make_dataset
from yolo_workbench.dataset import DatasetError
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.sam import BASE, LARGE, SamManager, drop_specks, mask_box, mask_outlines
from yolo_workbench.web import create_app


class FakeSam:
    """Stands in for UltralyticsSamBackend: the mask is the rectangle around the positive points (±10 px), or the box."""

    def __init__(self, name):
        self.name = name
        self.encoded = []

    def encode(self, image):
        self.encoded.append(image.shape)
        return {"shape": image.shape}

    def segment(self, features, shape, points, labels, box):
        mask = np.zeros(shape, dtype=bool)
        positive = [point for point, label in zip(points, labels) if label == 1]
        if box is not None:
            x1, y1, x2, y2 = (int(round(value)) for value in box)
        elif positive:
            xs, ys = [p[0] for p in positive], [p[1] for p in positive]
            x1, y1, x2, y2 = int(min(xs)) - 10, int(min(ys)) - 10, int(max(xs)) + 10, int(max(ys)) + 10
        else:
            return mask, 0.0
        mask[max(0, y1):y2, max(0, x1):x2] = True
        return mask, 0.9


def ready_manager(factory=FakeSam, gpu=True) -> SamManager:
    manager = SamManager(backend_factory=factory, gpu=gpu)
    manager.start()
    manager._thread.join(5)
    return manager


def test_prefers_large_on_gpu_and_base_without(tmp_path):
    assert ready_manager().model == LARGE
    assert ready_manager(gpu=False).model == BASE


def test_falls_back_to_base_when_large_fails():
    def factory(name):
        if name == LARGE:
            raise RuntimeError("CUDA out of memory")
        return FakeSam(name)

    manager = ready_manager(factory)
    assert manager.status == "ready" and manager.model == BASE


def test_load_failure_reports_error_and_can_retry():
    attempts = []

    def factory(name):
        attempts.append(name)
        if len(attempts) <= 2:
            raise RuntimeError("no network")
        return FakeSam(name)

    manager = ready_manager(factory)
    assert manager.status == "error" and "no network" in manager.error
    manager.start()
    manager._thread.join(5)
    assert manager.status == "ready" and manager.model == LARGE


def test_requires_ready_model(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    manager = SamManager(backend_factory=FakeSam, gpu=True)
    with pytest.raises(DatasetError, match="Start smart annotation"):
        manager.segment(record.path, "detection", [{"x": 0.5, "y": 0.5, "positive": True}], None)


def test_detection_returns_box_around_mask(tmp_path):
    dataset, _ = make_dataset(tmp_path)  # 100×80 image
    record = next(iter(dataset.images.values()))
    manager = ready_manager()
    result = manager.segment(record.path, "detection", [{"x": 0.3, "y": 0.5, "positive": True}, {"x": 0.5, "y": 0.5, "positive": True}], None)
    assert result["found"]
    # positive points at x=30,50 y=40 → mask x 20..60, y 30..50
    assert result["points"] == pytest.approx([0.4, 0.5, 0.4, 0.25])
    assert len(result["outlines"]) == 1 and len(result["outlines"][0]) >= 6


class SpeckledSam(FakeSam):
    """FakeSam plus the kind of speck real SAM leaves far from the object (a 2×2 island in a corner)."""

    def segment(self, features, shape, points, labels, box):
        mask, score = super().segment(features, shape, points, labels, box)
        mask[-3:-1, -3:-1] = True
        return mask, score


def test_specks_do_not_stretch_the_box(tmp_path):
    dataset, _ = make_dataset(tmp_path)  # 100×80 image
    record = next(iter(dataset.images.values()))
    manager = ready_manager(SpeckledSam)
    result = manager.segment(record.path, "detection", [{"x": 0.3, "y": 0.5, "positive": True}, {"x": 0.5, "y": 0.5, "positive": True}], None)
    assert result["points"] == pytest.approx([0.4, 0.5, 0.4, 0.25])  # as without the speck
    assert len(result["outlines"]) == 1


def test_segmentation_returns_polygon_and_box_prompt_works(tmp_path):
    dataset, _ = make_dataset(tmp_path, "segmentation")
    record = next(iter(dataset.images.values()))
    manager = ready_manager()
    result = manager.segment(record.path, "segmentation", [], [0.6, 0.5, 0.1, 0.25])  # reversed corners are ordered
    assert result["found"]
    xs, ys = result["points"][0::2], result["points"][1::2]
    assert len(xs) >= 3 and len(result["points"]) % 2 == 0
    assert min(xs) == pytest.approx(0.1) and max(xs) == pytest.approx(0.59, abs=0.011)
    assert min(ys) == pytest.approx(0.25) and max(ys) == pytest.approx(0.4875, abs=0.013)


def test_prompt_validation(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    manager = ready_manager()
    with pytest.raises(DatasetError, match="first"):
        manager.segment(record.path, "detection", [], None)
    with pytest.raises(DatasetError, match="at least once"):
        manager.segment(record.path, "detection", [{"x": 0.5, "y": 0.5, "positive": False}], None)
    with pytest.raises(DatasetError, match="classification"):
        manager.segment(record.path, "classification", [{"x": 0.5, "y": 0.5, "positive": True}], None)


def test_nothing_found_is_not_an_error():
    assert mask_outlines(np.zeros((10, 10), dtype=bool)) == []


def test_mask_helpers_cover_every_part():
    mask = np.zeros((10, 20), dtype=bool)
    mask[1:4, 1:4] = True
    mask[5:9, 10:18] = True
    assert mask_box(mask) == pytest.approx([0.475, 0.5, 0.85, 0.8])
    largest, small = mask_outlines(mask)  # every part, largest first
    assert min(largest[0::2]) == pytest.approx(0.5) and min(largest[1::2]) == pytest.approx(0.5)
    assert max(small[0::2]) == pytest.approx(0.15) and max(small[1::2]) == pytest.approx(0.3)


def test_drop_specks_keeps_object_parts():
    mask = np.zeros((640, 640), dtype=bool)  # specks: under 10×10 px here, or under 1% of the largest part
    mask[100:300, 100:200] = True  # an object split by an occluder: both halves stay
    mask[100:300, 220:260] = True
    mask[600:609, 600:609] = True  # 81 px: a speck
    mask[10:12, 400:402] = True
    cleaned = drop_specks(mask)
    assert cleaned.sum() == 200 * 100 + 200 * 40
    assert mask_box(cleaned) == pytest.approx([180 / 640, 200 / 640, 160 / 640, 200 / 640])
    tiny = np.zeros((640, 640), dtype=bool)  # the largest part stays however small
    tiny[5:8, 5:8] = True
    tiny[50:52, 50:52] = True
    assert drop_specks(tiny).sum() == 9


def test_encodes_each_image_once_until_it_changes(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    manager = ready_manager()
    point = [{"x": 0.5, "y": 0.5, "positive": True}]
    manager.encode(record.path)
    manager.segment(record.path, "detection", point, None)
    manager.segment(record.path, "detection", point, None)
    assert len(manager.backend.encoded) == 1
    Image.new("RGB", (100, 80), "black").save(record.path)
    stat = record.path.stat()
    os.utime(record.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    manager.segment(record.path, "detection", point, None)
    assert len(manager.backend.encoded) == 2


def test_encodes_the_exif_upright_image(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    image = Image.new("RGB", (100, 80), "white")
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90°: shown as 80×100
    image.save(record.path, exif=exif.tobytes())
    manager = ready_manager()
    manager.encode(record.path)
    assert manager.backend.encoded == [(100, 80, 3)]


def test_api_flow(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    sam = SamManager(backend_factory=FakeSam, gpu=True)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"), sam=sam)
    image_id = next(iter(dataset.images))

    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/api/v1/sam")).json()["status"] == "idle"
            body = {"points": [{"x": 0.5, "y": 0.5}]}
            response = await client.post(f"/api/v1/images/{image_id}/sam/segment", json=body)
            assert response.status_code == 400
            assert (await client.post("/api/v1/sam/start")).status_code == 200
            sam._thread.join(5)
            state = (await client.get("/api/v1/sam")).json()
            assert state["status"] == "ready" and state["model"] == LARGE
            assert (await client.post(f"/api/v1/images/{image_id}/sam/encode")).json() == {"encoded": True}
            response = await client.post(f"/api/v1/images/{image_id}/sam/segment", json=body)
            assert response.status_code == 200
            assert response.json()["found"] is True
            assert (await client.post(f"/api/v1/images/{image_id}/sam/segment", json={"points": [{"x": 1.5, "y": 0.5}]})).status_code == 422
            assert (await client.post("/api/v1/images/unknown/sam/segment", json=body)).status_code == 404

    asyncio.run(run())


def rounded_rectangle_mask(shape=(480, 640)):
    """A tilted 360×220 rectangle whose corners the mask rounds off (as a model's mask does) and its true corners."""
    import cv2

    corners = cv2.boxPoints(((320, 240), (360, 220), 12)).astype(np.float32)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    return mask.astype(bool), corners


def test_point_limit_fits_vertices_into_rounded_corners():
    mask, corners = rounded_rectangle_mask()
    height, width = mask.shape
    assert len(mask_outlines(mask)[0]) // 2 > 8  # lightly simplified, the rounded corners take several points
    polygon = np.array(mask_outlines(mask, max_points=4)[0]).reshape(-1, 2) * [width, height]
    assert len(polygon) == 4
    # the vertices leave the outline for the corners it rounded off (Douglas–Peucker alone stays ~5 px inside)
    for corner in corners:
        assert np.hypot(*(polygon - corner).T).min() < 2.5
    assert mask_outlines(mask, max_points=1000) == [mask_outlines(mask, max_points=1000)[0]]  # under the limit: untouched


def test_segment_applies_the_point_limit(tmp_path):
    image = tmp_path / "large.png"
    Image.new("RGB", (640, 480)).save(image)

    class RoundedSam(FakeSam):
        def segment(self, features, shape, points, labels, box):
            return rounded_rectangle_mask(shape)[0], 0.9

    manager = ready_manager(RoundedSam)
    prompt = [{"x": 0.5, "y": 0.5, "positive": True}]
    assert len(manager.segment(image, "segmentation", prompt, None)["points"]) > 16
    limited = manager.segment(image, "segmentation", prompt, None, max_points=5)
    assert len(limited["points"]) // 2 <= 5 and limited["outlines"] == [limited["points"]]
    box = manager.segment(image, "detection", prompt, None, max_points=5)["points"]
    assert len(box) == 4  # boxes ignore the limit


def test_unload_drops_the_model_until_started_again(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    path = next(iter(dataset.images.values())).path
    manager = ready_manager()
    manager.encode(path)
    assert manager.unload("run 1") is True
    state = manager.payload()
    assert state["status"] == "idle" and state["model"] is None and "training run 'run 1'" in state["unloaded"]
    assert manager.backend is None and not manager._features
    with pytest.raises(DatasetError, match="training run 'run 1'"):
        manager.segment(path, "detection", [{"x": 0.5, "y": 0.5, "positive": True}], None)
    assert manager.unload("run 2") is False  # nothing loaded
    manager.start()
    manager._thread.join(5)
    assert manager.payload()["status"] == "ready" and manager.payload()["unloaded"] is None


def test_unload_while_loading_discards_the_model(tmp_path):
    release = threading.Event()

    class SlowSam(FakeSam):
        def __init__(self, name):
            release.wait(5)
            super().__init__(name)

    manager = SamManager(backend_factory=SlowSam, gpu=True)
    manager.start()
    assert manager.unload("run") is True
    release.set()
    manager._thread.join(5)
    assert manager.payload()["status"] == "idle" and manager.backend is None
