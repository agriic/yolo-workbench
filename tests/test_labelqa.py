import asyncio
import io
import time

import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from PIL import Image

from yolo_workbench.dataset import Dataset, DatasetError
from yolo_workbench.labelqa import LabelQAManager, compute_findings
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.models import Annotation
from yolo_workbench.predictor import PredictorManager
from yolo_workbench.web import create_app

LABELS = (
    "0 0.2 0.2 0.1 0.1\n"   # A: the model agrees
    "1 0.6 0.6 0.2 0.2\n"   # B: the model says class 0 here  -> wrong_class
    "0 0.85 0.15 0.1 0.1\n"  # C: the model sees nothing here  -> unmatched_label
    "1 0.5 0.85 0.2 0.1\n"   # D: the model's box is shifted   -> box_mismatch
)
PREDICTIONS = [
    {"class_id": 0, "confidence": 0.9, "points": [0.2, 0.2, 0.1, 0.1]},
    {"class_id": 0, "confidence": 0.8, "points": [0.6, 0.6, 0.2, 0.2]},
    {"class_id": 1, "confidence": 0.7, "points": [0.2, 0.8, 0.1, 0.1]},   # no label here -> missing_label
    {"class_id": 1, "confidence": 0.8, "points": [0.6, 0.85, 0.2, 0.1]},
    {"class_id": 1, "confidence": 0.2, "points": [0.4, 0.4, 0.05, 0.05]},  # weak: only below the default threshold
]


class FakeBackend:
    def __init__(self, path, category, conf, iou):
        self.names = {0: "zero", 1: "one"}
        self.confs = []

    def predict(self, image_path, conf=None):
        self.confs.append(conf)
        return [dict(item) for item in PREDICTIONS]


def make_qa(tmp_path):
    (tmp_path / "train").mkdir()
    Image.new("RGB", (200, 160), "white").save(tmp_path / "train" / "a.jpg")
    (tmp_path / "train" / "a.txt").write_text(LABELS)
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"path": ".", "train": "train", "names": {0: "zero", 1: "one"}}))
    dataset = Dataset(tmp_path / "data.yaml", "detection")
    predictor = PredictorManager(dataset, backend_factory=FakeBackend)
    (tmp_path / "model.pt").write_bytes(b"fake")
    predictor.load(str(tmp_path / "model.pt"))
    return dataset, predictor, LabelQAManager(dataset, predictor)


def wait(qa):
    for _ in range(200):
        if qa.job["state"] != "running":
            return
        time.sleep(0.01)
    raise AssertionError("Label QA did not finish")


def kinds(result):
    return sorted(item["kind"] for item in result["items"])


def test_analysis_finds_each_kind_of_disagreement(tmp_path):
    dataset, predictor, qa = make_qa(tmp_path)
    qa.start()
    wait(qa)
    assert qa.job["state"] == "done" and qa.job["done"] == 1
    assert predictor.backend.confs == [0.05]  # QA asks the model for weak detections too
    assert predictor.external_busy is None

    result = qa.findings()
    assert kinds(result) == ["box_mismatch", "missing_label", "unmatched_label", "wrong_class"]
    assert result["counts"] == {"missing_label": 1, "wrong_class": 1, "box_mismatch": 1, "unmatched_label": 1}
    by_kind = {item["kind"]: item for item in result["items"]}
    assert by_kind["wrong_class"]["label"]["class_id"] == 1 and by_kind["wrong_class"]["prediction"]["class_id"] == 0
    assert by_kind["unmatched_label"]["label"]["points"] == [0.85, 0.15, 0.1, 0.1]
    assert 0.1 < by_kind["box_mismatch"]["iou"] < 0.5
    # the weak prediction shows up only at a lower threshold
    assert qa.findings(min_conf=0.1)["counts"]["missing_label"] == 2
    # most suspicious first
    assert [item["score"] for item in result["items"]] == sorted((item["score"] for item in result["items"]), reverse=True)


def test_applying_fixes_goes_through_the_label_file_and_undo(tmp_path):
    dataset, predictor, qa = make_qa(tmp_path)
    qa.start()
    wait(qa)
    label = tmp_path / "train" / "a.txt"
    image_id = next(iter(dataset.images))
    by_kind = {item["kind"]: item for item in qa.findings()["items"]}

    qa.apply(image_id, by_kind["missing_label"]["key"])
    assert label.read_text().endswith("1 0.2 0.8 0.1 0.1\n")
    qa.apply(image_id, by_kind["wrong_class"]["key"])
    assert "0 0.6 0.6 0.2 0.2" in label.read_text()
    remaining = {item["kind"]: item for item in qa.findings()["items"]}
    assert set(remaining) == {"box_mismatch", "unmatched_label"}  # recomputed from the new labels
    qa.apply(image_id, remaining["box_mismatch"]["key"])
    assert "1 0.6 0.85 0.2 0.1" in label.read_text()
    qa.apply(image_id, qa.findings()["items"][0]["key"])  # unmatched label: deleted
    assert "0.85 0.15" not in label.read_text()
    assert qa.findings()["total"] == 0

    dataset.history("undo")
    assert "0.85 0.15" in label.read_text()
    with pytest.raises(DatasetError):
        qa.apply(image_id, "no-such-finding")


def test_dismissals_persist_and_results_survive_a_restart(tmp_path):
    dataset, predictor, qa = make_qa(tmp_path)
    qa.start()
    wait(qa)
    image_id = next(iter(dataset.images))
    unmatched = next(item for item in qa.findings()["items"] if item["kind"] == "unmatched_label")
    qa.dismiss(image_id, unmatched["key"])
    assert qa.findings()["counts"]["unmatched_label"] == 0

    reopened = LabelQAManager(dataset, predictor)
    assert reopened.payload()["result"]["images"] == 1
    assert reopened.findings()["counts"] == {"missing_label": 1, "wrong_class": 1, "box_mismatch": 1, "unmatched_label": 0}


def test_changed_image_is_stale_and_classes_follow_renumbering(tmp_path):
    from yolo_workbench import classes

    dataset, predictor, qa = make_qa(tmp_path)
    qa.start()
    wait(qa)
    classes.reorder_classes(dataset, [1, 0])  # zero <-> one swap ids
    result = qa.findings()
    by_kind = {item["kind"]: item for item in result["items"]}
    assert by_kind["wrong_class"]["prediction"]["class_id"] == 1  # still "zero", now id 1
    assert by_kind["wrong_class"]["label"]["class_id"] == 0

    record = next(iter(dataset.images.values()))
    Image.new("RGB", (200, 160), "black").save(record.path)
    assert qa.findings() == {**qa.findings(), "total": 0, "stale": 1}


def test_qa_blocks_model_swaps_while_running_and_needs_a_model(tmp_path):
    dataset, predictor, qa = make_qa(tmp_path)
    predictor.external_busy = "Label QA is analyzing images"
    with pytest.raises(DatasetError, match="Label QA"):
        predictor.load(str(tmp_path / "model.pt"))
    predictor.external_busy = None

    fresh = PredictorManager(dataset, backend_factory=FakeBackend)
    with pytest.raises(DatasetError, match="Load a model"):
        LabelQAManager(dataset, fresh).start()


def test_segmentation_uses_polygon_boxes():
    label = Annotation("x:1", 0, [0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 0.1, 0.3])
    same = {"class_id": 0, "model_class_id": 0, "model_class_name": "zero", "confidence": 0.9, "points": [0.1, 0.1, 0.3, 0.1, 0.3, 0.3]}
    assert compute_findings("segmentation", [label], [same], {0: "zero"}) == []


def test_label_qa_api(tmp_path):
    dataset, predictor, _ = make_qa(tmp_path)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"), predictor=predictor)

    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/api/v1/qa")).json()["available"]
            assert (await client.post("/api/v1/qa/run", json={"split": "train"})).status_code == 200
            for _ in range(200):
                if (await client.get("/api/v1/qa")).json()["job"]["state"] != "running":
                    break
                await asyncio.sleep(0.01)
            data = (await client.get("/api/v1/qa/findings", params={"kind": "missing_label"})).json()
            assert data["total"] == 1 and data["counts"]["wrong_class"] == 1
            item = data["items"][0]
            crop = await client.get(f"/api/v1/qa/images/{item['image_id']}/findings/{item['key']}/crop")
            assert crop.status_code == 200 and Image.open(io.BytesIO(crop.content)).size[0] > 0
            applied = await client.post(f"/api/v1/qa/images/{item['image_id']}/findings/{item['key']}/apply")
            assert applied.json()["kind"] == "missing_label"
            again = await client.post(f"/api/v1/qa/images/{item['image_id']}/findings/{item['key']}/apply")
            assert again.status_code == 400
            assert (await client.get("/api/v1/qa/findings", params={"kind": "bogus"})).status_code == 422
            assert (await client.delete("/api/v1/qa")).json()["result"] is None

    asyncio.run(run())
