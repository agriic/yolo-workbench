import asyncio
import io
import math
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


# ---------- pairing on real-world near misses (640×480 image, names car / truck) ----------

NAMES = {0: "car", 1: "truck"}
SIZE = (640, 480)


def box(x, y, w, h, class_id=0, conf=None):
    """A label (no conf) or a prediction from a pixel box."""
    points = [(x + w / 2) / 640, (y + h / 2) / 480, w / 640, h / 480]
    if conf is None:
        return Annotation(f"a:{x}:{y}", class_id, points)
    return {"class_id": class_id, "model_class_id": class_id, "model_class_name": NAMES[class_id], "confidence": conf, "points": points}


def findings_of(labels, predictions, tolerance=3.0, size=SIZE):
    return compute_findings("detection", labels, predictions, NAMES, size, tolerance)


def test_a_couple_of_pixels_off_is_the_same_box_even_on_small_objects():
    # a 10 px object 2 px off on two sides: IoU 0.47, which used to read as a loose box
    assert findings_of([box(100, 100, 10, 10)], [box(102, 102, 10, 10, conf=0.9)]) == []


def test_weak_other_class_guess_does_not_steal_the_label():
    # the right-class box is a few pixels off; a 7% "truck" fits the label exactly. Pairing by
    # IoU alone gave the label to the truck and reported the confident car as a missing label.
    found = findings_of([box(100, 100, 10, 10)], [box(102, 102, 10, 10, conf=0.9), box(100, 100, 10, 10, class_id=1, conf=0.07)])
    assert found == []


def test_box_off_by_more_than_the_tolerance_is_a_loose_box_even_with_good_overlap():
    # 30 px off on a 100 px object: IoU 0.54, which used to pass silently
    [finding] = findings_of([box(100, 100, 100, 100)], [box(130, 100, 100, 100, conf=0.8)])
    assert finding["kind"] == "box_mismatch" and finding["deviation_px"] == 30
    assert findings_of([box(100, 100, 100, 100)], [box(130, 100, 100, 100, conf=0.8)], tolerance=30) == []


def test_tolerance_grows_with_the_object():
    # 15 px on a 400 px object is within 5% of its size
    assert findings_of([box(100, 40, 400, 400)], [box(115, 40, 400, 400, conf=0.9)]) == []


def test_confident_other_class_still_beats_a_stray_same_class_box():
    labels = [box(100, 100, 100, 100)]
    found = findings_of(labels, [box(100, 100, 100, 100, class_id=1, conf=0.9), box(170, 170, 100, 100, conf=0.3)])
    assert [finding["kind"] for finding in found] == ["wrong_class", "missing_label"]


def test_without_pixel_sizes_only_poor_overlap_is_a_loose_box():
    labels = [box(100, 100, 10, 10)]
    assert findings_of(labels, [box(102, 102, 10, 10, conf=0.9)], size=(0, 0))[0]["kind"] == "box_mismatch"
    assert findings_of(labels, [box(100, 100, 10, 10, conf=0.9)], size=(0, 0)) == []


def test_tolerance_is_a_listing_option(tmp_path):
    dataset, predictor, qa = make_qa(tmp_path)
    qa.start()
    wait(qa)
    assert qa.findings()["counts"]["box_mismatch"] == 1  # label D is 20 px off on a 200 px wide image
    assert qa.findings(tolerance=25)["counts"]["box_mismatch"] == 0
    # a finding listed at any tolerance can still be applied
    image_id = next(iter(dataset.images))
    loose = next(item for item in qa.findings()["items"] if item["kind"] == "box_mismatch")
    qa.findings(tolerance=25)
    assert qa.apply(image_id, loose["key"])["kind"] == "box_mismatch"


# ---------- segmentation: the predictor's polygon point limit ----------

CIRCLE = [round(value, 6) for i in range(64) for value in (0.5 + 0.2 * math.cos(i * math.pi / 32), 0.5 + 0.2 * math.sin(i * math.pi / 32))]


class FakeSegBackend(FakeBackend):
    def predict(self, image_path, conf=None):
        return [{"class_id": 0, "confidence": 0.9, "points": list(CIRCLE)}]


def make_seg_qa(tmp_path):
    (tmp_path / "train").mkdir()
    Image.new("RGB", (200, 160), "white").save(tmp_path / "train" / "a.jpg")
    (tmp_path / "train" / "a.txt").write_text("")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"path": ".", "train": "train", "names": {0: "zero", 1: "one"}}))
    dataset = Dataset(tmp_path / "data.yaml", "segmentation")
    predictor = PredictorManager(dataset, backend_factory=FakeSegBackend)
    (tmp_path / "model.pt").write_bytes(b"fake")
    predictor.load(str(tmp_path / "model.pt"))
    qa = LabelQAManager(dataset, predictor)
    qa.start()
    wait(qa)
    return dataset, predictor, qa


def test_segmentation_findings_follow_the_point_limit(tmp_path):
    dataset, predictor, qa = make_seg_qa(tmp_path)
    image_id = next(iter(dataset.images))
    [found] = qa.findings()["items"]
    assert found["kind"] == "missing_label" and len(found["prediction"]["points"]) == 128

    predictor.set_options(8)
    [limited] = qa.findings()["items"]
    assert limited["key"] == found["key"]  # keyed as the model drew it: dismissals survive a new limit
    assert len(limited["prediction"]["points"]) <= 16
    assert qa.payload()["predictor"]["max_points"] == 8

    qa.apply(image_id, found["key"])
    [annotation] = dataset.images[image_id].annotations
    assert annotation.points == limited["prediction"]["points"]


def test_segmentation_loose_outline_takes_the_limited_polygon(tmp_path):
    dataset, predictor, qa = make_seg_qa(tmp_path)
    label = tmp_path / "train" / "a.txt"
    label.write_text("0 0.25 0.25 0.65 0.25 0.65 0.65 0.25 0.65\n")  # a square, off from the model's circle
    dataset.rescan()
    image_id = next(iter(dataset.images))
    predictor.set_options(6)
    [found] = qa.findings()["items"]
    assert found["kind"] == "box_mismatch"
    qa.apply(image_id, found["key"])
    assert len(dataset.images[image_id].annotations[0].points) <= 12
