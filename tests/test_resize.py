import asyncio

import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from PIL import Image

from test_imaging import make_rotated_dataset, red
from test_workspace import make_workspace
from yolo_workbench import importer, storage
from yolo_workbench.dataset import Dataset, DatasetError, WriteConflict
from yolo_workbench.imaging import ORIENTATION
from yolo_workbench.jobs import JobCancelled, JobManager
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.resize import fit_size, resize_images, resize_options, summary
from yolo_workbench.web import create_app

FIT = {"mode": "fit", "width": 100, "height": 100, "enlarge": False}


def make_dataset(tmp_path):
    """train: big.jpg 400×200 (labelled), small.png 50×40; val: tall.jpg 120×300."""
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    Image.new("RGB", (400, 200), "red").save(tmp_path / "train" / "big.jpg")
    (tmp_path / "train" / "big.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    Image.new("RGB", (50, 40), "blue").save(tmp_path / "train" / "small.png")
    Image.new("RGB", (120, 300), "green").save(tmp_path / "val" / "tall.jpg")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"path": ".", "train": "train", "val": "val", "names": {0: "thing"}}))
    return Dataset(tmp_path / "data.yaml", "detection")


def record_named(dataset, name):
    return next(record for record in dataset.images.values() if record.path.name == name)


def test_fit_size_keeps_the_aspect_ratio():
    assert fit_size(400, 200, FIT) == (100, 50)
    assert fit_size(120, 300, FIT) == (40, 100)
    assert fit_size(1920, 1080, {**FIT, "width": 1280, "height": 1280}) == (1280, 720)
    assert fit_size(50, 40, FIT) is None  # fits already
    assert fit_size(100, 100, FIT) is None
    assert fit_size(50, 40, {**FIT, "enlarge": True}) == (100, 80)
    assert fit_size(100, 60, {**FIT, "enlarge": True}) is None  # already as large as it fits
    assert fit_size(10000, 1, FIT) == (100, 1)  # never 0 px


def test_resize_options_are_validated():
    assert resize_options(None) is None
    assert resize_options({"mode": "none", "width": 5}) is None
    assert resize_options({"mode": "fit", "width": "640", "height": 480}) == {"mode": "fit", "width": 640, "height": 480, "enlarge": False}
    for bad in ({"mode": "fit", "width": 0, "height": 10}, {"mode": "fit", "width": 10, "height": 99999}, {"mode": "stretch", "width": 10, "height": 10}, {"mode": "fit", "width": "x", "height": 10}):
        with pytest.raises(DatasetError):
            resize_options(bad)


def test_resizes_the_images_that_do_not_fit_as_one_undoable_change(tmp_path):
    dataset = make_dataset(tmp_path)
    big, small, tall = (record_named(dataset, name) for name in ("big.jpg", "small.png", "tall.jpg"))
    original = big.path.read_bytes()
    assert summary(dataset, FIT) == {"images": 3, "resize": 2, "unknown": 0, "bytes": big.image_size + tall.image_size}
    assert summary(dataset, FIT, "train")["resize"] == 1

    result = resize_images(dataset, FIT)
    assert (result["resized"], result["unchanged"], result["failed"], result["skipped"]) == (2, 1, 0, 0)
    with Image.open(big.path) as image:
        assert image.size == (100, 50)
    with Image.open(tall.path) as image:
        assert image.size == (40, 100)
    assert (big.width, big.height, tall.width, tall.height) == (100, 50, 40, 100)
    assert big.image_size == big.path.stat().st_size
    assert (tmp_path / "train" / "big.txt").read_text() == "0 0.5 0.5 0.2 0.4\n"  # relative coordinates: unchanged
    assert big.annotations[0].points == [0.5, 0.5, 0.2, 0.4]
    assert small.path.stat().st_size == small.image_size  # fits: untouched
    # the originals wait in the trash, which the undo history protects
    trash = tmp_path / ".yolo-workbench" / "trash" / dataset.session_id
    assert (trash / "train" / "big.jpg").read_bytes() == original
    assert not list((trash / storage.VERSION_TRASH).rglob("*.*"))  # the new versions are in place
    assert dataset.referenced_trash_sessions() == {dataset.session_id}
    # the size index knows the new sizes: opening again probes nothing
    assert record_named(Dataset(dataset.yaml_path, "detection"), "big.jpg").width == 100

    undo = dataset.history("undo")
    assert set(undo["image_ids"]) == {big.id, tall.id}
    assert big.path.read_bytes() == original and (big.width, big.height) == (400, 200)
    assert not (trash / "train" / "big.jpg").exists()
    assert len(list((trash / storage.VERSION_TRASH).rglob("*.*"))) == 2  # kept for a redo
    # restoring the trash leaves the versions to the undo history
    assert storage.restore_trash(dataset, dataset.session_id)["restored"] == 0
    assert len(dataset.images) == 3

    dataset.history("redo")
    with Image.open(big.path) as image:
        assert image.size == (100, 50)
    assert (big.width, big.height) == (100, 50)


def test_undo_is_refused_once_the_image_was_replaced_on_disk(tmp_path):
    dataset = make_dataset(tmp_path)
    big = record_named(dataset, "big.jpg")
    resize_images(dataset, FIT, "train")
    Image.new("RGB", (300, 300), "white").save(big.path)  # replaced by hand
    with pytest.raises(WriteConflict, match="big.jpg"):
        dataset.history("undo")
    with Image.open(big.path) as image:
        assert image.size == (300, 300)


def test_resizes_the_upright_image_and_drops_the_orientation(tmp_path):
    dataset = make_rotated_dataset(tmp_path)  # 400×200 pixels stored, 200×400 upright, a red block at x 0.6–0.9, y 0.1–0.3
    record = next(iter(dataset.images.values()))
    resize_images(dataset, {**FIT, "width": 50})
    with Image.open(record.path) as image:
        assert image.size == (50, 100)
        assert image.getexif().get(ORIENTATION) is None
        assert red(image.convert("RGB").getpixel((37, 20)))  # the block is where the label says
    assert (record.width, record.height) == (50, 100)


def test_a_cancelled_resize_changes_nothing(tmp_path):
    dataset = make_dataset(tmp_path)
    before = {record.path: record.path.read_bytes() for record in dataset.images.values()}

    class CancelOnReplace:
        def progress(self, done=None, total=None, message=None):
            if message == "Replacing the images…":
                raise JobCancelled()

    with pytest.raises(JobCancelled):
        resize_images(dataset, FIT, job=CancelOnReplace())
    assert {path: path.read_bytes() for path in before} == before
    assert not (tmp_path / ".yolo-workbench" / "trash").exists() or not list((tmp_path / ".yolo-workbench" / "trash").rglob("*.*"))
    assert not dataset.history_state()["can_undo"]


def test_an_image_removed_while_resizing_is_left_out(tmp_path):
    dataset = make_dataset(tmp_path)
    tall = record_named(dataset, "tall.jpg")

    class DeleteMeanwhile:
        def progress(self, done=None, total=None, message=None):
            if message == "Replacing the images…":
                dataset.delete_images([tall.id])

    result = resize_images(dataset, FIT, job=DeleteMeanwhile())
    assert (result["resized"], result["skipped"]) == (1, 1)
    assert not list((tmp_path / ".yolo-workbench" / "trash").rglob(f"{tall.id}*"))  # its new version is removed


def test_unreadable_images_are_counted(tmp_path):
    dataset = make_dataset(tmp_path)
    record = record_named(dataset, "tall.jpg")
    record.width = record.height = 0  # as if not probed yet
    record.path.write_bytes(b"not an image")
    result = resize_images(dataset, FIT)
    assert result["failed"] == 1 and result["errors"][0].startswith("tall.jpg")
    assert result["resized"] == 1


def test_import_resizes_after_the_crop(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Fit", "detection", ["x"])
    dataset = workspace.session("fit").dataset
    source = tmp_path / "wide"
    source.mkdir()
    Image.new("RGB", (400, 200), "red").save(source / "a.jpg")
    (source / "a.txt").write_text("0 0.5 0.5 0.1 0.2\n0 0.95 0.5 0.1 0.2\n")  # the second one reaches the right edge
    Image.new("RGB", (60, 30), "blue").save(source / "b.png")
    options = {"split_mode": "fixed", "split": "train", "resize": {"mode": "fit", "width": 100, "height": 100}}
    result = importer.run_import(dataset, source, options)
    assert (result["imported"], result["resized"], result["cropped"]) == (2, 1, 0)
    with Image.open(dataset.root / "images" / "train" / "a.jpg") as image:
        assert image.size == (100, 50)
    assert (dataset.root / "labels" / "train" / "a.txt").read_text() == (source / "a.txt").read_text()  # a resize alone keeps them
    assert (dataset.root / "images" / "train" / "b.png").read_bytes() == (source / "b.png").read_bytes()  # fits: copied as it is
    assert importer.run_import(dataset, source, options)["duplicates"] == 2

    cropped = {**options, "crop": {"mode": "center", "unit": "px", "width": 200, "height": 200}, "resize": {"mode": "fit", "width": 50, "height": 50}}
    result = importer.run_import(dataset, source, cropped)
    assert (result["imported"], result["cropped"], result["resized"]) == (2, 1, 2)
    with Image.open(dataset.root / "images" / "train" / "a_1.jpg") as image:
        assert image.size == (50, 50)
    with Image.open(dataset.root / "images" / "train" / "b_1.png") as image:
        assert image.size == (50, 25)  # within the crop already, then resized
    assert (dataset.root / "labels" / "train" / "a_1.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"  # clipped to the crop


def test_resize_api(tmp_path):
    dataset = make_dataset(tmp_path)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"), jobs=JobManager(synchronous=True))

    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            planned = (await client.get("/api/v1/resize", params={"width": 100, "height": 100, "split": "val"})).json()
            assert (planned["images"], planned["resize"]) == (1, 1)
            assert (await client.get("/api/v1/resize", params={"width": 0, "height": 100})).status_code == 422
            dataset.write_guard = lambda: "a training run uses this dataset"
            assert (await client.post("/api/v1/resize", json={"width": 100, "height": 100})).status_code == 409
            dataset.write_guard = None
            job = (await client.post("/api/v1/resize", json={"width": 100, "height": 100})).json()
            assert job["state"] == "done" and job["kind"] == "resize", job
            assert job["result"]["resized"] == 2
            assert (await client.get("/api/v1/dataset")).json()["can_undo"]
            image_id = record_named(dataset, "big.jpg").id
            assert (await client.get(f"/api/v1/images/{image_id}")).json()["width"] == 100
            assert (await client.post("/api/v1/history/undo")).json()["image_ids"]
            assert (await client.get(f"/api/v1/images/{image_id}")).json()["width"] == 400

    asyncio.run(run())


def test_running_takes_several_job_kinds():
    jobs = JobManager()
    seen = []

    def work(job):
        seen.append(jobs.running("p", ("import", "resize")))
        return {}

    jobs.synchronous = True
    jobs.start("resize", "p", "Resize", work)
    assert [job["kind"] for job in seen[0]] == ["resize"]
    assert jobs.running("p", ("import", "resize")) == []
