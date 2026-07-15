import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from test_dataset import make_dataset
from yolo_workbench.dataset import DatasetError
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.predictor import PredictorManager
from yolo_workbench.web import create_app


class FakeBackend:
    """Stands in for UltralyticsBackend: model classes ONE/zero/extra."""

    def __init__(self, path, category, conf, iou):
        self.category = category
        self.conf = conf
        self.iou = iou
        self.names = {0: "ONE", 1: "zero", 2: "extra"}
        self.calls = []

    def predict(self, image_path):
        self.calls.append(image_path)
        # placed away from the sample's existing annotation so dedup filtering doesn't drop them
        points = [0.85, 0.85, 0.2, 0.2] if self.category == "detection" else [0.05, 0.05, 0.15, 0.05, 0.1, 0.15]
        return [
            {"class_id": 0, "confidence": 0.9, "points": points},
            {"class_id": 2, "confidence": 0.4, "points": points},
        ]


def make_predictor(tmp_path, category="detection"):
    dataset, label = make_dataset(tmp_path, category)
    manager = PredictorManager(dataset, backend_factory=FakeBackend)
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake")
    return dataset, label, manager, model


def wait_job(manager):
    for _ in range(200):
        if manager.job["state"] in {"done", "error"}:
            return
        import time

        time.sleep(0.01)
    raise AssertionError("prediction job did not finish")


def test_load_auto_maps_classes_case_insensitively(tmp_path):
    _, _, manager, model = make_predictor(tmp_path)
    payload = manager.load(str(model))
    assert payload["status"] == "ready"
    # model "ONE" -> dataset "one" (1), model "zero" -> 0, "extra" unmapped
    assert manager.mapping == {0: 1, 1: 0, 2: None}


def test_load_missing_model_errors(tmp_path):
    _, _, manager, _ = make_predictor(tmp_path)
    with pytest.raises(DatasetError):
        manager.load(str(tmp_path / "nope.pt"))


def test_run_stores_predictions_in_memory_only(tmp_path):
    dataset, label, manager, model = make_predictor(tmp_path)
    manager.load(str(model))
    before = label.read_text()
    manager.run()
    wait_job(manager)
    record = next(iter(dataset.images.values()))
    items = manager.predictions_for(record.id)["items"]
    assert len(items) == 2
    assert items[0]["class_id"] == 1  # mapped via ONE -> one
    assert items[1]["class_id"] is None  # "extra" unmapped
    assert label.read_text() == before  # nothing written


def test_only_unlabeled_skips_annotated_images(tmp_path):
    _, _, manager, model = make_predictor(tmp_path)
    manager.load(str(model))
    with pytest.raises(DatasetError):
        manager.run(only_unlabeled=True)  # the sample image already has annotations


def test_accept_merges_through_replace_annotations(tmp_path):
    dataset, label, manager, model = make_predictor(tmp_path)
    manager.load(str(model))
    record = next(iter(dataset.images.values()))
    manager.run(image_ids=[record.id])
    wait_job(manager)
    result = manager.accept(record.id)  # accept-all skips the unmapped prediction
    assert result["accepted"] == 1
    assert len(record.annotations) == 2
    assert "1 0.85 0.85 0.2 0.2" in label.read_text()
    assert len(manager.predictions_for(record.id)["items"]) == 1  # unmapped stays pending
    dataset.history("undo")
    assert "0.2 0.2" not in label.read_text()


def test_accept_unmapped_by_id_is_rejected(tmp_path):
    dataset, _, manager, model = make_predictor(tmp_path)
    manager.load(str(model))
    record = next(iter(dataset.images.values()))
    manager.run(image_ids=[record.id])
    wait_job(manager)
    unmapped = manager.predictions_for(record.id)["items"][1]
    with pytest.raises(DatasetError):
        manager.accept(record.id, prediction_ids=[unmapped["id"]])


def test_accept_min_confidence_filters(tmp_path):
    dataset, _, manager, model = make_predictor(tmp_path)
    manager.load(str(model))
    record = next(iter(dataset.images.values()))
    manager.run(image_ids=[record.id])
    wait_job(manager)
    manager.set_mapping({"2": 0})  # map "extra" so only confidence decides
    result = manager.accept(record.id, min_confidence=0.5)
    assert result["accepted"] == 1
    assert len(manager.predictions_for(record.id)["items"]) == 1


def test_reject_never_touches_disk(tmp_path):
    dataset, label, manager, model = make_predictor(tmp_path)
    manager.load(str(model))
    record = next(iter(dataset.images.values()))
    manager.run(image_ids=[record.id])
    wait_job(manager)
    before = label.read_text()
    mtime = label.stat().st_mtime_ns
    result = manager.reject(record.id)
    assert result["rejected"] == 2
    assert manager.predictions_for(record.id)["items"] == []
    assert label.read_text() == before
    assert label.stat().st_mtime_ns == mtime


def test_set_mapping_updates_pending_predictions(tmp_path):
    dataset, _, manager, model = make_predictor(tmp_path)
    manager.load(str(model))
    record = next(iter(dataset.images.values()))
    manager.run(image_ids=[record.id])
    wait_job(manager)
    manager.set_mapping({"2": 0})
    assert all(item["class_id"] is not None for item in manager.predictions_for(record.id)["items"])
    with pytest.raises(DatasetError):
        manager.set_mapping({"2": 99})


def test_segmentation_predictions_accept(tmp_path):
    dataset, label, manager, model = make_predictor(tmp_path, "segmentation")
    manager.load(str(model))
    record = next(iter(dataset.images.values()))
    manager.run(image_ids=[record.id])
    wait_job(manager)
    manager.accept(record.id)
    assert len(record.annotations) == 2


def overlapping_backend(points, model_class_id=0):
    """Backend emitting one prediction with the given geometry (model class 0 maps to dataset class 1)."""

    class Backend(FakeBackend):
        def predict(self, image_path):
            return [{"class_id": model_class_id, "confidence": 0.9, "points": points}]

    return Backend


def test_prediction_matching_existing_annotation_is_filtered(tmp_path):
    # the sample image already has class 1 at [0.5, 0.5, 0.2, 0.4]; model class 0 maps to dataset class 1
    dataset, _ = make_dataset(tmp_path)
    manager = PredictorManager(dataset, backend_factory=overlapping_backend([0.5, 0.5, 0.2, 0.4]))
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake")
    manager.load(str(model))
    record = next(iter(dataset.images.values()))
    assert manager.predict_image(record.id)["items"] == []
    manager.run(image_ids=[record.id])
    wait_job(manager)
    assert manager.predictions_for(record.id)["items"] == []


def test_overlapping_prediction_with_different_class_is_kept(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    # model class 1 maps to dataset class 0, which differs from the existing annotation's class 1
    manager = PredictorManager(dataset, backend_factory=overlapping_backend([0.5, 0.5, 0.2, 0.4], model_class_id=1))
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake")
    manager.load(str(model))
    record = next(iter(dataset.images.values()))
    assert len(manager.predict_image(record.id)["items"]) == 1


def test_non_overlapping_same_class_prediction_is_kept(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    manager = PredictorManager(dataset, backend_factory=overlapping_backend([0.85, 0.85, 0.1, 0.1]))
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake")
    manager.load(str(model))
    record = next(iter(dataset.images.values()))
    assert len(manager.predict_image(record.id)["items"]) == 1


def test_accept_recheck_drops_predictions_duplicated_meanwhile(tmp_path):
    dataset, label, manager, model = make_predictor(tmp_path)
    manager.load(str(model))
    record = next(iter(dataset.images.values()))
    manager.predict_image(record.id)
    # the user manually draws the same box the model predicted before accepting
    annotations = [annotation.as_dict() for annotation in record.annotations]
    annotations.append({"class_id": 1, "points": [0.85, 0.85, 0.2, 0.2]})
    dataset.replace_annotations(record.id, annotations)
    with pytest.raises(DatasetError):
        manager.accept(record.id)  # only the now-duplicate mapped prediction was acceptable
    assert label.read_text().count("0.85") == 2  # no duplicate row written


def test_segmentation_prediction_matching_existing_polygon_is_filtered(tmp_path):
    dataset, _ = make_dataset(tmp_path, "segmentation")
    # same polygon as the existing class-1 annotation
    manager = PredictorManager(dataset, backend_factory=overlapping_backend([0.2, 0.2, 0.8, 0.2, 0.5, 0.8]))
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake")
    manager.load(str(model))
    record = next(iter(dataset.images.values()))
    assert manager.predict_image(record.id)["items"] == []


def test_predict_single_image_synchronously(tmp_path):
    dataset, label, manager, model = make_predictor(tmp_path)
    record = next(iter(dataset.images.values()))
    with pytest.raises(DatasetError):
        manager.predict_image(record.id)  # no model loaded yet
    manager.load(str(model))
    before = label.read_text()
    result = manager.predict_image(record.id)
    assert len(result["items"]) == 2
    assert manager.job["state"] == "idle"  # no background job involved
    assert label.read_text() == before
    # re-running replaces the pending set instead of accumulating
    assert len(manager.predict_image(record.id)["items"]) == 2


def test_predict_single_image_endpoint(tmp_path):
    dataset, _, manager, model = make_predictor(tmp_path)
    manager.load(str(model))
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"), predictor=manager)

    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            image_id = next(iter(dataset.images))
            response = await client.post(f"/api/v1/images/{image_id}/predictions/compute")
            assert response.status_code == 200
            assert len(response.json()["items"]) == 2
            assert (await client.post("/api/v1/images/unknown/predictions/compute")).status_code == 404

    asyncio.run(run())


def test_discover_models_scans_and_prioritizes_recents(tmp_path):
    dataset, _, manager, model = make_predictor(tmp_path)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "best.pt").write_bytes(b"fake")
    (tmp_path / "weights" / "exported.onnx").write_bytes(b"fake")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "skipped.pt").write_bytes(b"fake")
    (tmp_path / "notes.txt").write_text("not a model")

    found = manager.discover_models()["items"]
    names = [item["name"] for item in found if str(tmp_path) in item["path"]]
    assert "model.pt" in names and "best.pt" in names
    assert "exported.onnx" not in names  # only .pt models are discovered
    assert "skipped.pt" not in names and "notes.txt" not in names
    assert all(item["source"] == "found" for item in found if str(tmp_path) in item["path"])

    manager.load(str(model))
    assert (tmp_path / ".yolo-workbench" / "predictor.json").exists()
    found = manager.discover_models()["items"]
    assert found[0]["path"] == str(model.resolve())
    assert found[0]["source"] == "recent"
    assert [item["path"] for item in found].count(str(model.resolve())) == 1  # no duplicate from the scan


def test_discover_scans_directories_of_recent_models(tmp_path, tmp_path_factory):
    dataset, _, manager, _ = make_predictor(tmp_path)
    outside = tmp_path_factory.mktemp("models-elsewhere")
    loaded = outside / "loaded.pt"
    loaded.write_bytes(b"fake")
    (outside / "sibling.pt").write_bytes(b"fake")
    manager.load(str(loaded))
    found = manager.discover_models()["items"]
    assert found[0]["path"] == str(loaded.resolve())
    assert any(item["path"] == str((outside / "sibling.pt").resolve()) for item in found)


def test_recent_models_skip_deleted_files(tmp_path):
    dataset, _, manager, model = make_predictor(tmp_path)
    manager.load(str(model))
    model.unlink()
    assert all(item["path"] != str(model.resolve()) for item in manager.discover_models()["items"])


def test_models_endpoint(tmp_path):
    dataset, _, manager, model = make_predictor(tmp_path)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"), predictor=manager)

    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            items = (await client.get("/api/v1/predictor/models")).json()["items"]
            assert any(item["name"] == "model.pt" for item in items)

    asyncio.run(run())


def test_predictor_api_endpoints(tmp_path):
    dataset, label, manager, model = make_predictor(tmp_path)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"), predictor=manager)

    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            state = (await client.get("/api/v1/predictor")).json()
            assert state["status"] == "idle"
            loaded = await client.post("/api/v1/predictor/load", json={"path": str(model), "conf": 0.3})
            assert loaded.status_code == 200
            assert loaded.json()["status"] == "ready"
            assert loaded.json()["mapping"] == {"0": 1, "1": 0, "2": None}
            image_id = next(iter(dataset.images))
            run_response = await client.post("/api/v1/predictor/run", json={"image_ids": [image_id]})
            assert run_response.status_code == 200
            wait_job(manager)
            items = (await client.get(f"/api/v1/images/{image_id}/predictions")).json()["items"]
            assert len(items) == 2
            remap = await client.put("/api/v1/predictor/mapping", json={"mapping": {"2": None}})
            assert remap.status_code == 200
            accepted = await client.post(f"/api/v1/images/{image_id}/predictions/accept", json={})
            assert accepted.json()["accepted"] == 1
            rejected = await client.post(f"/api/v1/images/{image_id}/predictions/reject", json={})
            assert rejected.json()["rejected"] == 1
            assert (await client.get("/api/v1/predictor")).json()["pending"] == {}
            missing = await client.post("/api/v1/predictor/load", json={"path": str(tmp_path / "nope.pt")})
            assert missing.status_code == 400

    asyncio.run(run())
