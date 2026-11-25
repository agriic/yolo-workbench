import asyncio
import io
from pathlib import Path

import numpy as np
import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageDraw

from test_dataset import make_dataset
from yolo_workbench import boxfit
from yolo_workbench.boxfit import BoxFitManager, crop_plan, geometry_key
from yolo_workbench.dataset import Dataset, DatasetError, object_key
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.sam import SamManager
from yolo_workbench.web import create_app

WIDTH, HEIGHT = 1600, 1200
CARD = (400, 300, 600, 450)  # the dark rectangle's pixels: x 400..599, y 300..449


class DarkSam:
    """Stands in for SAM on synthetic images: the mask is the dark part of the prompt box connected to the positive
    click, without the parts holding a background click. Records what it was asked."""

    score = 0.95

    def __init__(self, name="fake"):
        self.name = name
        self.encoded = []
        self.prompts = []

    def encode(self, image):
        self.encoded.append(image.shape[:2])
        return {"image": image.copy()}

    def candidates(self, features, shape, points, labels, box):
        import cv2

        self.prompts.append({"shape": shape, "points": points, "labels": labels, "box": box})
        image = features["image"]
        assert image.shape[:2] == tuple(shape)
        dark = image.mean(axis=2) < 128
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        inside = np.zeros_like(dark)
        inside[max(0, y1):y2, max(0, x1):x2] = True
        _, parts = cv2.connectedComponents((dark & inside).astype(np.uint8))
        positive = [point for point, label in zip(points, labels) if label == 1][0]
        part = parts[int(positive[1]), int(positive[0])]
        if part == 0:
            return [(np.zeros(shape, bool), 0.1)]
        mask = parts == part
        for point, label in zip(points, labels):
            if label == 0 and parts[int(point[1]), int(point[0])] == part:
                return [(np.zeros(shape, bool), 0.1)]
        return [(mask, self.score)]


class UnsureSam(DarkSam):
    score = 0.4


def ready(factory=DarkSam) -> SamManager:
    manager = SamManager(backend_factory=factory, gpu=True)
    manager.start()
    manager._thread.join(5)
    return manager


def norm(box, width=WIDTH, height=HEIGHT):
    return [box[0] / width, box[1] / height, box[2] / width, box[3] / height]


def yolo(box, width=WIDTH, height=HEIGHT):
    x1, y1, x2, y2 = norm(box, width, height)
    return f"{(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def card_dataset(tmp_path: Path, labels: list[str], cards=(CARD,), name="cards.png") -> Dataset:
    split = tmp_path / "train"
    split.mkdir(exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    for card in cards:
        draw.rectangle([card[0], card[1], card[2] - 1, card[3] - 1], fill="black")
    image.save(split / name)
    (split / name).with_suffix(".txt").write_text("".join(f"{line}\n" for line in labels))
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": ".", "train": "train", "names": {0: "card", 1: "ball"}}))
    return Dataset(yaml_path, "detection")


def run(manager: BoxFitManager, sam, **kwargs):
    manager.start(sam, **kwargs)
    manager._thread.join(10)
    assert manager.job["state"] == "done", manager.job


# ---------- geometry ----------

def test_small_boxes_get_regions_of_their_own_and_close_ones_share():
    boxes = [[100, 100, 150, 140], [160, 100, 210, 140], [1300, 900, 1350, 950]]
    plan = crop_plan(boxes, WIDTH, HEIGHT)
    assert sorted(members for _, members in plan) == [[0, 1], [2]]
    for region, _ in plan:
        assert max(region[2] - region[0], region[3] - region[1]) < 600  # far finer than the whole 1600 px image


def test_a_box_nearly_as_large_as_the_image_uses_the_whole_image():
    plan = crop_plan([[100, 100, 1500, 1100], [700, 500, 760, 560]], WIDTH, HEIGHT)
    assert plan[0][0] == (0, 0, WIDTH, HEIGHT) or plan[1][0] == (0, 0, WIDTH, HEIGHT)


def test_background_clicks_on_neighbours_inside_the_search_area_only():
    box = [100, 100, 200, 200]
    prompt = boxfit.grown(box, WIDTH, HEIGHT)
    assert prompt == (85, 85, 215, 215)
    others = [[195, 120, 225, 180], [150, 150, 170, 170], [205, 100, 260, 180], [600, 600, 700, 700]]  # beside, inside the label, centre past the search area, far
    assert boxfit._negatives(box, prompt, others) == [(210.0, 150.0)]


# ---------- fitting ----------

def test_fits_a_loose_box_to_the_object(tmp_path):
    label = (392, 306, 610, 444)  # 8 px too far left, 6 px too low on top, 10 px too far right, 6 px too high below
    dataset = card_dataset(tmp_path, [f"0 {yolo(label)}"])
    record = next(iter(dataset.images.values()))
    sam = ready()
    [fit] = sam.fit_boxes(record.path, [norm(label)], [0])
    assert fit["status"] == "fitted" and fit["reasons"] == []
    assert fit["box"] == pytest.approx(norm(CARD))
    assert fit["image_size"] == [WIDTH, HEIGHT]
    backend = sam.backend
    assert backend.encoded and max(backend.encoded[0]) < 600  # a crop around the box, not the whole image


def test_takes_the_mask_that_agrees_with_the_label():
    class ThreeMasks:
        def candidates(self, features, shape, points, labels, box):
            part, card, stack = (np.zeros(shape, bool) for _ in range(3))
            part[40:60, 40:60] = True     # the suit symbol, very sure
            card[20:120, 20:100] = True   # the card
            stack[0:200, 0:200] = True    # the whole stack
            return [(part, 0.99), (card, 0.9), (stack, 0.95)]

    fit = boxfit.fit_one(ThreeMasks(), None, (0, 0, 200, 200), [22, 18, 98, 122], [], 200, 200)
    assert fit["box"] == pytest.approx([0.1, 0.1, 0.5, 0.6])


def test_prefers_the_mask_whose_outline_follows_the_image_edges():
    # a ball on plain felt: SAM's surest mask fills the prompt box with felt; the ball's own outline is the edge
    import cv2

    image = np.full((200, 200, 3), (60, 110, 35), np.uint8)
    cv2.circle(image, (100, 100), 30, (40, 200, 240), -1)
    ball, filled = np.zeros((200, 200), bool), np.zeros((200, 200), bool)
    cv2.circle(ball.view(np.uint8), (100, 100), 30, 1, -1)
    filled[62:139, 62:139] = True

    class BoxFilling:
        def candidates(self, features, shape, points, labels, box):
            return [(ball, 0.75), (filled, 0.92)]

    label = [64, 66, 138, 136]  # a few pixels off the ball (70..131)
    without = boxfit.fit_one(BoxFilling(), None, (0, 0, 200, 200), label, [], 200, 200)
    assert without["box"] == pytest.approx([0.31, 0.31, 0.695, 0.695])  # score × overlap alone takes the felt
    fit = boxfit.fit_one(BoxFilling(), None, (0, 0, 200, 200), label, [], 200, 200, boxfit.edge_strength(image))
    assert fit["box"] == pytest.approx([0.35, 0.35, 0.655, 0.655])


def test_edge_fringe_does_not_widen_the_box():
    mask = np.zeros((300, 300), bool)
    mask[100:200, 80:180] = True               # the card
    mask[100:200:3, 180:192:2] = True           # SAM's dotted fringe along its right edge, joined to it
    mask[96:100, 80:180:3] = True               # ... and above it
    extent, _ = boxfit._mask_extent(mask, [75, 95, 195, 205])
    assert extent == pytest.approx((80, 100, 180, 200), abs=1)  # fringe touching the edge may stay: a pixel, not 12
    tilted = np.zeros((300, 300), np.uint8)    # a rotated card keeps its corners
    import cv2

    corners = cv2.boxPoints(((150, 150), (120, 80), 30)).astype(np.int32)
    cv2.fillPoly(tilted, [corners], 1)
    ys, xs = np.nonzero(tilted)
    extent, _ = boxfit._mask_extent(tilted.astype(bool), [60, 60, 240, 240])
    assert extent == pytest.approx((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), abs=1)


def test_unsure_and_failed_fits(tmp_path):
    labels = [
        f"0 {yolo((480, 360, 520, 390))}",   # far too small: the outline stops at the search area's edge
        f"0 {yolo((1000, 800, 1100, 900))}",  # nothing there
    ]
    dataset = card_dataset(tmp_path, labels)
    record = next(iter(dataset.images.values()))
    small, empty = ready().fit_boxes(record.path, [norm((480, 360, 520, 390)), norm((1000, 800, 1100, 900))])
    assert small["status"] == "uncertain" and "edge" in small["reasons"] and "different" in small["reasons"]
    assert empty["status"] == "failed" and empty["box"] is None
    [low] = ready(UnsureSam).fit_boxes(record.path, [norm((395, 296, 604, 455))], [0])
    assert low["status"] == "uncertain" and low["reasons"] == ["low_score"]


def test_overlapping_labels_are_uncertain_and_neighbours_are_background(tmp_path):
    other = (560, 280, 700, 470)  # covers the card's right part
    dataset = card_dataset(tmp_path, [f"0 {yolo((395, 296, 604, 455))}", f"0 {yolo(other)}"])
    record = next(iter(dataset.images.values()))
    sam = ready()
    [fit] = sam.fit_boxes(record.path, [norm((395, 296, 604, 455)), norm(other)], [0])
    assert fit["reasons"] == ["overlap"] and fit["box"] == pytest.approx(norm(CARD))  # fitted, but to be looked at
    prompt = sam.backend.prompts[0]
    # the other label's centre is past this label but inside its search area: a background click there
    assert prompt["labels"] == [1, 0]
    [(region, _)] = crop_plan([[395, 296, 604, 455]], WIDTH, HEIGHT)
    assert prompt["points"][1] == pytest.approx([630 - region[0], 375 - region[1]])


def test_whole_image_encoding_is_cached_for_the_editor(tmp_path):
    dataset, _ = make_dataset(tmp_path)  # a 100×80 white image
    record = next(iter(dataset.images.values()))
    sam = ready()
    sam.fit_boxes(record.path, [[0.4, 0.3, 0.6, 0.7]])
    sam.fit_boxes(record.path, [[0.4, 0.3, 0.6, 0.7]])
    assert sam.backend.encoded == [(80, 100)]  # encoded once, kept like smart annotation's
    sam.fit_boxes(record.path, [[0.4, 0.3, 0.6, 0.7]], cache=False)
    assert len(sam.backend.encoded) == 2


def test_fitting_needs_sam_loaded(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    with pytest.raises(DatasetError, match="Start smart annotation"):
        SamManager(backend_factory=DarkSam, gpu=True).fit_boxes(record.path, [[0.4, 0.3, 0.6, 0.7]])


# ---------- the batch and its findings ----------

def test_run_lists_fits_and_applies_one_with_undo(tmp_path):
    label = (392, 306, 610, 444)
    dataset = card_dataset(tmp_path, [f"0 {yolo(label)}", f"1 {yolo((1000, 800, 1100, 900))}"])
    manager = BoxFitManager(dataset)
    sam = ready()
    run(manager, sam, class_ids=[0])
    assert manager.payload()["result"]["objects"] == 1
    listing = manager.findings()
    assert listing["counts"] == {"fitted": 1, "uncertain": 0, "failed": 0}
    [finding] = listing["items"]
    assert finding["edges"] == pytest.approx([8, -6, -10, 6], abs=0.1)
    assert finding["deviation_px"] == pytest.approx(10, abs=0.1)
    assert manager.findings(tolerance=10)["total"] == 0  # within the tolerance: tight enough

    result = manager.apply(finding["image_id"], finding["key"])
    assert result["applied"] == 1
    record = dataset.images[finding["image_id"]]
    assert record.annotations[0].points == pytest.approx([0.3125, 0.3125, 0.125, 0.125])
    assert record.annotations[1].class_id == 1  # the other class was not fitted or touched
    assert manager.findings()["total"] == 0  # the label changed: its fit no longer applies
    dataset.history("undo")
    assert manager.findings()["total"] == 1  # back to the fitted geometry: the fit applies again


def test_apply_all_is_one_transaction_and_skips_uncertain(tmp_path):
    cards = [(100, 100, 300, 250), (700, 100, 900, 250), (1200, 700, 1400, 850)]
    labels = [f"0 {yolo((92, 106, 310, 244))}", f"0 {yolo((705, 94, 890, 258))}", f"0 {yolo((1280, 760, 1320, 790))}"]
    dataset = card_dataset(tmp_path, labels, cards)
    manager = BoxFitManager(dataset)
    run(manager, ready())
    counts = manager.findings()["counts"]
    assert counts["fitted"] == 2 and counts["uncertain"] == 1
    before = [list(annotation.points) for annotation in next(iter(dataset.images.values())).annotations]
    result = manager.apply_all("fitted", None, "all", 2.0)
    assert result["applied"] == 2 and result["files"] == 1
    record = next(iter(dataset.images.values()))
    assert record.annotations[2].points == pytest.approx(before[2])  # the uncertain one stays
    assert manager.findings()["counts"]["fitted"] == 0
    dataset.history("undo")  # one undo for the whole batch
    after = [list(annotation.points) for annotation in next(iter(dataset.images.values())).annotations]
    assert sum(after, []) == pytest.approx(sum(before, []))


def test_run_on_chosen_objects_adds_to_earlier_fits(tmp_path):
    cards = [(100, 100, 300, 250), (700, 100, 900, 250)]
    dataset = card_dataset(tmp_path, [f"0 {yolo((92, 106, 310, 244))}", f"1 {yolo((705, 94, 890, 258))}"], cards)
    manager = BoxFitManager(dataset)
    sam = ready()
    record = next(iter(dataset.images.values()))
    run(manager, sam, objects=[{"image_id": record.id, "key": object_key(record.annotations[0])}])
    assert [item["class_id"] for item in manager.findings()["items"]] == [0]
    run(manager, sam, class_ids=[1])
    assert sorted(item["class_id"] for item in manager.findings()["items"]) == [0, 1]
    assert [item["class_id"] for item in manager.findings(class_id=1)["items"]] == [1]


def test_dismiss_stale_image_and_saved_results(tmp_path):
    dataset = card_dataset(tmp_path, [f"0 {yolo((392, 306, 610, 444))}"])
    manager = BoxFitManager(dataset)
    run(manager, ready())
    [finding] = manager.findings()["items"]
    assert BoxFitManager(dataset).findings()["total"] == 1  # kept in .yolo-workbench/boxfit.json
    manager.dismiss(finding["image_id"], finding["key"])
    assert manager.findings()["total"] == 0
    with pytest.raises(DatasetError, match="no longer applies"):
        manager.apply(finding["image_id"], finding["key"])

    manager.clear()
    run(manager, ready())
    record = next(iter(dataset.images.values()))
    record.image_mtime += 1  # the file changed (as a rescan would see it)
    assert manager.findings()["stale"] == 1 and manager.findings()["total"] == 0


def test_runs_only_on_detection(tmp_path):
    dataset, _ = make_dataset(tmp_path, "segmentation")
    with pytest.raises(DatasetError, match="detection"):
        BoxFitManager(dataset).start(ready(), class_ids=[0])


def test_cancel_stops_the_run(tmp_path):
    dataset = card_dataset(tmp_path, [f"0 {yolo((392, 306, 610, 444))}"])
    manager = BoxFitManager(dataset)
    sam = SamManager(backend_factory=DarkSam, gpu=True)
    sam.status = "loading"  # the run waits for the model
    manager.start(sam, class_ids=[0])
    manager.cancel()
    manager._thread.join(5)
    assert manager.job["state"] == "cancelled"


def test_render_shows_both_boxes(tmp_path):
    dataset = card_dataset(tmp_path, [f"0 {yolo((392, 306, 610, 444))}"])
    manager = BoxFitManager(dataset)
    run(manager, ready())
    [finding] = manager.findings()["items"]
    content = manager.crop(finding["image_id"], finding["key"], lambda class_id: "#ff0000")
    with Image.open(io.BytesIO(content)) as image:
        assert image.format == "JPEG" and max(image.size) >= 400  # scaled up so a few pixels show


# ---------- reshape through the object bulk edit ----------

def test_bulk_reshape_validates_points(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    annotation = record.annotations[0]
    bad = dataset.bulk_edit_objects([{"image_id": record.id, "annotation_id": annotation.id, "action": "reshape", "points": [0.5, 0.5, 2.0, 0.1], "key": object_key(annotation)}])
    assert bad["applied"] == 0 and bad["skipped"]
    missing = dataset.bulk_edit_objects([{"image_id": record.id, "annotation_id": annotation.id, "action": "reshape", "points": None}])
    assert missing["skipped"][0]["reason"] == "Reshape requires points"
    good = dataset.bulk_edit_objects([{"image_id": record.id, "annotation_id": annotation.id, "action": "reshape", "points": [0.5, 0.5, 0.3, 0.3]}])
    assert good["applied"] == 1 and dataset.images[record.id].annotations[0].points == [0.5, 0.5, 0.3, 0.3]


def test_geometry_key_ignores_the_class(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    annotation = next(iter(dataset.images.values())).annotations[0]
    assert geometry_key(annotation.points) == geometry_key(list(annotation.points))


# ---------- API ----------

def test_api_flow(tmp_path):
    label = (392, 306, 610, 444)
    dataset = card_dataset(tmp_path, [f"0 {yolo(label)}"])
    sam = SamManager(backend_factory=DarkSam, gpu=True)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"), sam=sam)
    image_id = next(iter(dataset.images))
    box = [float(value) for value in yolo(label).split()]

    async def flow():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            state = (await client.get("/api/v1/boxfit")).json()
            assert state["available"] and state["result"] is None and state["sam"]["status"] == "idle"
            # the editor's fit needs SAM on (the editor starts it first)
            assert (await client.post(f"/api/v1/images/{image_id}/sam/fit", json={"boxes": [box], "targets": [0]})).status_code == 400
            started = (await client.post("/api/v1/boxfit/run", json={"class_ids": [0]})).json()
            assert started["job"]["state"] == "running"  # it waits for SAM, which the run turned on
            sam._thread.join(5)
            for _ in range(200):
                state = (await client.get("/api/v1/boxfit")).json()
                if state["job"]["state"] != "running":
                    break
                await asyncio.sleep(0.02)
            assert state["job"]["state"] == "done" and state["result"]["objects"] == 1

            fit = (await client.post(f"/api/v1/images/{image_id}/sam/fit", json={"boxes": [box], "targets": [0]})).json()
            assert fit["items"][0]["status"] == "fitted"
            assert fit["items"][0]["points"] == pytest.approx([0.3125, 0.3125, 0.125, 0.125])
            assert (await client.post(f"/api/v1/images/{image_id}/sam/fit", json={"boxes": [box], "targets": [3]})).status_code == 400

            listing = (await client.get("/api/v1/boxfit/findings", params={"tolerance": 2})).json()
            assert listing["total"] == 1
            key = listing["items"][0]["key"]
            crop = await client.get(f"/api/v1/boxfit/images/{image_id}/findings/{key}/crop")
            assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"
            again = await client.get(f"/api/v1/boxfit/images/{image_id}/findings/{key}/crop", headers={"if-none-match": crop.headers["etag"]})
            assert again.status_code == 304

            applied = (await client.post("/api/v1/boxfit/apply-all", json={"kind": "fitted", "tolerance": 2})).json()
            assert applied["applied"] == 1 and applied["image_ids"] == [image_id]
            assert (await client.get("/api/v1/boxfit/findings")).json()["total"] == 0
            assert (await client.post("/api/v1/boxfit/apply-all", json={"kind": "fitted"})).status_code == 400
            assert (await client.post("/api/v1/history/undo")).status_code == 200
            assert (await client.get("/api/v1/boxfit/findings")).json()["total"] == 1
            assert (await client.delete("/api/v1/boxfit")).json()["result"] is None

    asyncio.run(flow())
