import asyncio
import errno
import json
import os
import time

import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from PIL import Image

from test_training import ARGS, FakeLauncher, make_dataset, runner_finishes
from yolo_workbench import download, snapshots, splitstats
from yolo_workbench.dataset import Dataset, DatasetLocked
from yolo_workbench.fsutil import InsufficientSpace
from yolo_workbench.jobs import JobCancelled, JobManager
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.resize import resize_images
from yolo_workbench.runtime import RUNTIME
from yolo_workbench.training import TrainingManager
from yolo_workbench.web import create_app


def record_named(dataset, name):
    return next(record for record in dataset.images.values() if record.path.name == name)


def snapshot_files(dataset_root, snapshot_id):
    path = snapshots.snapshot_dir(dataset_root, snapshot_id)
    return sorted(file.relative_to(path).as_posix() for file in path.rglob("*") if file.is_file() and ".yolo-workbench" not in file.relative_to(path).parts)


class StubJob:
    """A job whose progress reports run `hook` once (to act while a snapshot is half taken)."""

    def __init__(self, hook=None, cancel_at=None):
        self.hook = hook
        self.cancel_at = cancel_at
        self.messages = []

    def progress(self, done=None, total=None, message=None):
        if message:
            self.messages.append(message)
        if message == "Linking the images…" and self.hook:
            hook, self.hook = self.hook, None
            hook()
        if self.cancel_at is not None and message == self.cancel_at:
            raise JobCancelled()


def test_a_snapshot_links_the_images_copies_the_labels_and_opens_read_only(tmp_path):
    dataset = make_dataset(tmp_path)
    meta = snapshots.take(dataset, "First", "before cleaning")
    root = dataset.root
    path = snapshots.snapshot_dir(root, meta["id"])
    assert (meta["name"], meta["note"], meta["kind"], meta["images"]) == ("First", "before cleaning", "manual", 6)
    assert (meta["linked"], meta["copied"], meta["missing"]) == (6, 0, 0)
    assert meta["splits"] == {"train": 2, "val": 2, "test": 2}
    assert meta["table"]["train"]["classes"] == {"0": {"images": 1, "objects": 1}, "1": {"images": 1, "objects": 1}}
    assert meta["names"] == ["cat", "dog"] and meta["fingerprint"] == snapshots.fingerprint(dataset)
    assert snapshot_files(root, meta["id"]) == ["data.yaml"] + [f"images/{split}/{split}{index}.jpg" for split in ("test", "train", "val") for index in (0, 1)] + [f"labels/{split}/{split}{index}.txt" for split in ("test", "train", "val") for index in (0, 1)]
    image = root / "images" / "train" / "train0.jpg"
    assert os.stat(path / "images" / "train" / "train0.jpg").st_ino == os.stat(image).st_ino  # linked
    assert os.stat(path / "labels" / "train" / "train0.txt").st_ino != os.stat(root / "labels" / "train" / "train0.txt").st_ino  # copied
    rows = snapshots.manifest(root, meta["id"])
    row = next(row for row in rows if row["path"] == "images/train/train0.jpg")
    assert (row["split"], row["folder"], row["label"], row["classes"], row["size"]) == ("train", "images/train", "labels/train/train0.txt", [0], [32, 32])
    assert (path / ".yolo-workbench" / "source.yaml").read_text() == dataset.yaml_path.read_text()
    assert "path" not in yaml.safe_load((path / "data.yaml").read_text())

    copy = snapshots.open_dataset(root, meta["id"], "detection")
    assert copy.read_only and len(copy.images) == 6 and copy.indexing == {"done": 6, "total": 6}  # sizes from its index
    assert sorted((record.split, record.path.name, record.annotations[0].class_id) for record in copy.images.values()) == sorted((record.split, record.path.name, record.annotations[0].class_id) for record in dataset.images.values())
    with pytest.raises(DatasetLocked):
        copy.replace_annotations(next(iter(copy.images)), [])
    assert not (path / ".yolo-workbench" / "review.json").exists()


def test_the_dataset_changing_later_leaves_the_snapshot_as_it_was(tmp_path):
    dataset = make_dataset(tmp_path)
    meta = snapshots.take(dataset)
    path = snapshots.snapshot_dir(dataset.root, meta["id"])
    record = record_named(dataset, "train0.jpg")
    dataset.replace_annotations(record.id, [{"class_id": 1, "points": [0.1, 0.1, 0.1, 0.1]}])
    resize_images(dataset, {"mode": "fit", "width": 16, "height": 16, "enlarge": False})
    dataset.delete_images([record_named(dataset, "val0.jpg").id])
    assert (path / "labels" / "train" / "train0.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
    with Image.open(path / "images" / "train" / "train0.jpg") as image:
        assert image.size == (32, 32)
    assert (path / "images" / "val" / "val0.jpg").is_file()
    changes = splitstats.diff(dataset, meta, snapshots.manifest(dataset.root, meta["id"]))
    assert changes["counts"] == {"added": 0, "removed": 1, "moved": 0, "changed": 1, "unchanged": 4}
    assert changes["removed"] == ["images/val/val0.jpg"] and changes["changed"] == [record.id]


def test_without_hardlinks_the_images_are_copied_and_the_space_is_checked(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)

    def no_links(source, target):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(snapshots.os, "link", no_links)
    check = snapshots.link_check(dataset)
    assert (check["mode"], check["copy_images"]) == ("copy", 6)
    assert check["copy_bytes"] == sum(record.image_size for record in dataset.images.values())
    meta = snapshots.take(dataset)
    assert (meta["linked"], meta["copied"]) == (0, 6) and meta["copied_bytes"] == check["copy_bytes"]
    path = snapshots.snapshot_dir(dataset.root, meta["id"])
    assert (path / "images" / "val" / "val1.jpg").read_bytes() == (dataset.root / "images" / "val" / "val1.jpg").read_bytes()
    monkeypatch.setattr(RUNTIME, "min_free_bytes", 1 << 62)
    with pytest.raises(InsufficientSpace):
        snapshots.take(dataset, "too big")
    assert [item["id"] for item in snapshots.list_snapshots(dataset.root)] == [meta["id"]]
    assert not [path for path in snapshots.folder(dataset.root).iterdir() if path.name.startswith(".")]


def test_image_moves_wait_while_the_images_are_linked_and_label_edits_go_on(tmp_path):
    dataset = make_dataset(tmp_path)
    record = record_named(dataset, "train0.jpg")
    seen = {}

    def meanwhile():
        with pytest.raises(DatasetLocked, match="a snapshot is being taken"):
            dataset.move_images([record.id], "val")
        with pytest.raises(DatasetLocked):
            dataset.delete_images([record.id])
        dataset.replace_annotations(record.id, [{"class_id": 1, "points": [0.5, 0.5, 0.4, 0.4]}])
        seen["edited"] = True

    meta = snapshots.take(dataset, job=StubJob(meanwhile))
    assert seen == {"edited": True}
    path = snapshots.snapshot_dir(dataset.root, meta["id"])
    assert (path / "labels" / "train" / "train0.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"  # as it was when taken
    dataset.move_images([record.id], "val")  # the hold is gone


def test_a_cancelled_snapshot_leaves_nothing(tmp_path):
    dataset = make_dataset(tmp_path)
    with pytest.raises(JobCancelled):
        snapshots.take(dataset, job=StubJob(cancel_at="Linking the images…"))
    assert snapshots.list_snapshots(dataset.root) == []
    assert list(snapshots.folder(dataset.root).iterdir()) == []
    dataset.move_images([record_named(dataset, "train0.jpg").id], "val")


def test_a_snapshot_downloads_like_the_dataset_did(tmp_path):
    dataset = make_dataset(tmp_path)
    expected = sorted(entry.name for entry in download.plan(dataset, "full"))
    meta = snapshots.take(dataset)
    dataset.delete_images([record_named(dataset, "test0.jpg").id])
    assert sorted(entry.name for entry in snapshots.download_entries(dataset.root, meta["id"], "full")) == expected
    labels = [entry.name for entry in snapshots.download_entries(dataset.root, meta["id"], "labels")]
    assert "data.yaml" in labels and not [name for name in labels if name.endswith(".jpg")]


def test_a_proposed_split_lists_the_images_by_their_new_splits(tmp_path):
    dataset = make_dataset(tmp_path)
    moved = record_named(dataset, "train0.jpg")
    meta = snapshots.take(dataset, "Proposed", kind="auto", assignment={moved.id: "val"}, extra={"rebalance": {"digest": "abc"}})
    path = snapshots.snapshot_dir(dataset.root, meta["id"])
    assert meta["splits"] == {"train": 1, "val": 3, "test": 2} and meta["fingerprint"] is None
    config = yaml.safe_load((path / "data.yaml").read_text())
    assert (config["train"], config["val"], config["test"]) == ("train.txt", "val.txt", "test.txt")
    copy = snapshots.open_dataset(dataset.root, meta["id"], "detection")
    assert sorted(record.path.name for record in copy.images.values() if record.split == "val") == ["train0.jpg", "val0.jpg", "val1.jpg"]
    assert record_named(dataset, "train0.jpg").split == "train"  # nothing moved in the dataset


def test_classification_snapshots(tmp_path):
    root = tmp_path / "cls"
    for split in ("train", "val"):
        for name in ("cat", "dog"):
            (root / split / name).mkdir(parents=True)
            Image.new("RGB", (16, 16), "white").save(root / split / name / f"{split}-{name}.jpg")
    dataset = Dataset(root, "classification")
    meta = snapshots.take(dataset)
    assert snapshot_files(root, meta["id"]) == [f"{split}/{name}/{split}-{name}.jpg" for split in ("train", "val") for name in ("cat", "dog")]
    copy = snapshots.open_dataset(root, meta["id"], "classification")
    assert sorted((record.split, record.annotations[0].class_id) for record in copy.images.values()) == [("train", 0), ("train", 1), ("val", 0), ("val", 1)]
    cat = next(record for record in dataset.images.values() if record.path.name == "train-cat.jpg")
    dataset.replace_annotations(cat.id, [{"class_id": 1, "points": []}])  # a class change moves it to dog/
    dataset.move_images([next(record.id for record in dataset.images.values() if record.path.name == "val-dog.jpg")], "train")
    changes = splitstats.diff(dataset, meta, snapshots.manifest(root, meta["id"]))
    assert (changes["counts"]["added"], changes["counts"]["removed"], changes["counts"]["moved"], changes["counts"]["changed"]) == (0, 0, 1, 1)


def test_retention_keeps_manual_snapshots_and_those_runs_use(tmp_path):
    dataset = make_dataset(tmp_path)
    root = dataset.root
    manual = snapshots.take(dataset, "mine")
    auto = snapshots.take(dataset, "before rebalance", kind="auto")
    used = snapshots.take(dataset, "proposed", kind="auto")
    snapshots.use(root, used["id"], "run-1")
    kept = snapshots.take(dataset, "kept", kind="auto")
    snapshots.update(root, kept["id"], keep=True)
    for item in (manual, auto, used, kept):
        meta = snapshots.get(root, item["id"])
        meta["created"] = "2020-01-01T00:00:00"
        snapshots._write_json(snapshots.snapshot_dir(root, item["id"]) / ".yolo-workbench" / "snapshot.json", meta)
    assert snapshots.purge_expired(root) == [auto["id"]]
    assert {item["id"] for item in snapshots.list_snapshots(root)} == {manual["id"], used["id"], kept["id"]}
    assert snapshots.get(root, kept["id"])["kind"] == "manual"
    run = snapshots.take(dataset, "run", kind="run")
    snapshots.use(root, run["id"], "a")
    snapshots.use(root, run["id"], "b")
    assert not snapshots.release(root, run["id"], "a")
    assert snapshots.release(root, run["id"], "b")
    with pytest.raises(KeyError):
        snapshots.get(root, run["id"])


def test_recount_tells_what_deleting_would_free(tmp_path):
    dataset = make_dataset(tmp_path)
    meta = snapshots.take(dataset)
    assert meta["own_bytes"] == sum(len((dataset.root / "labels" / split / f"{split}{index}.txt").read_text()) for split in ("train", "val", "test") for index in (0, 1))
    image_bytes = sum(record.image_size for record in dataset.images.values())
    dataset.delete_images([record_named(dataset, "test0.jpg").id])
    for path in (dataset.root / ".yolo-workbench" / "trash").rglob("*.jpg"):
        path.unlink()  # the trash emptied: the snapshot is all that holds that image now
    counted = snapshots.recount(dataset.root, meta["id"])
    test0 = (snapshots.snapshot_dir(dataset.root, meta["id"]) / "images" / "test" / "test0.jpg").stat().st_size
    assert counted["own_bytes"] >= meta["own_bytes"] + test0
    assert counted["shared_bytes"] == image_bytes - test0


# ---------- training on snapshots ----------


def make_manager(dataset, tmp_path, launcher=None):
    manager = TrainingManager.for_dataset(dataset, jobs=JobManager(synchronous=True), launcher=launcher or FakeLauncher(), weights_dir=tmp_path / "weights")
    manager.attach(dataset)
    return manager


def test_a_run_on_a_snapshot_taken_at_launch_leaves_the_dataset_writable(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    run = manager.enqueue(None, dataset, "On a snapshot", "", ARGS, snapshot="launch")["run"]
    run_dir = launcher.launches[-1][0]
    spec = json.loads((run_dir / "workbench" / "spec.json").read_text())
    taken = snapshots.get(dataset.root, spec["snapshot"])
    assert (taken["kind"], taken["used_by"], taken["name"]) == ("run", [run_dir.name], "Run 'On a snapshot'")
    data = yaml.safe_load((run_dir / "workbench" / "data.yaml").read_text())
    train = (run_dir / "workbench" / "train.txt").read_text().split()
    assert data["train"].endswith("train.txt") and all(str(snapshots.snapshot_dir(dataset.root, taken["id"])) in path for path in train)
    assert manager.lock_info(dataset) is None
    record = record_named(dataset, "train0.jpg")
    dataset.replace_annotations(record.id, [{"class_id": 1, "points": [0.5, 0.5, 0.3, 0.3]}])  # writable while it trains
    summary = manager.run_detail(None, run["id"])
    assert summary["dataset"]["snapshot"]["id"] == taken["id"] and summary["snapshot"] == taken["id"]

    runner_finishes(run_dir)
    launcher.handle.code = 0
    manager.tick()
    # unchanged data reuses a snapshot; an edit since takes a new one
    manager.enqueue(None, dataset, "Second", "", ARGS, confirm=True, snapshot="launch")
    second = json.loads((launcher.launches[-1][0] / "workbench" / "spec.json").read_text())["snapshot"]
    assert second != taken["id"]
    runner_finishes(launcher.launches[-1][0])
    launcher.handle.code = 0
    manager.tick()
    manager.enqueue(None, dataset, "Third", "", ARGS, confirm=True, snapshot="launch")
    third = json.loads((launcher.launches[-1][0] / "workbench" / "spec.json").read_text())["snapshot"]
    assert third == second
    assert snapshots.get(dataset.root, second)["used_by"] == sorted([launcher.launches[1][0].name, launcher.launches[2][0].name])

    # deleting the run deletes the snapshot only it used
    manager.delete_run(None, run["id"])
    with pytest.raises(KeyError):
        snapshots.get(dataset.root, taken["id"])


def test_a_resumed_run_trains_on_its_snapshot_and_the_snapshot_stays_while_it_can_resume(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    manual = snapshots.take(dataset, "Frozen")
    run = manager.enqueue(None, dataset, "", "", ARGS, snapshot=manual["id"])["run"]
    run_dir = launcher.launches[-1][0]
    assert snapshots.get(dataset.root, manual["id"])["used_by"] == [run_dir.name]
    assert manager.snapshot_users(None, manual["id"]) == [run["name"]]
    runner_finishes(run_dir, status="paused", epochs=1)
    launcher.handle.code = 0
    manager.tick()
    assert manager.snapshot_users(None, manual["id"]) == [run["name"]]  # resumable
    dataset.delete_images([record_named(dataset, "train0.jpg").id])
    manager.resume(None, run["id"])
    train = (run_dir / "workbench" / "train.txt").read_text()
    assert "train0.jpg" in train  # the snapshot still has it
    assert len(launcher.launches) == 2


def test_preflight_and_runs_on_a_snapshot_that_is_gone(tmp_path):
    dataset = make_dataset(tmp_path)
    manager = make_manager(dataset, tmp_path)
    meta = snapshots.take(dataset)
    dataset.delete_images([record_named(dataset, "val0.jpg").id, record_named(dataset, "val1.jpg").id])
    assert manager.preflight(None, dataset, {"model": "yolo11n.pt"}, snapshot=meta["id"])["splits"]["val"] == 2
    assert manager.preflight(None, dataset, {"model": "yolo11n.pt"})["errors"]
    snapshots.delete(dataset.root, meta["id"])
    with pytest.raises(Exception, match="no longer exists"):
        manager.preflight(None, dataset, {"model": "yolo11n.pt"}, snapshot=meta["id"])


# ---------- API ----------


def client_for(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1", headers={"X-Workbench": "1"})


def test_snapshot_api(tmp_path):
    dataset = make_dataset(tmp_path)
    jobs = JobManager(synchronous=True)
    launcher = FakeLauncher()
    training = TrainingManager.for_dataset(dataset, jobs=jobs, launcher=launcher, weights_dir=tmp_path / "weights")
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media"), jobs=jobs, training=training)

    async def scenario():
        async with client_for(app) as client:
            assert (await client.get("/api/v1/snapshots/link-check")).json()["mode"] == "link"
            job = (await client.post("/api/v1/snapshots", json={"name": "Before QA", "note": "n"})).json()
            assert job["state"] == "done" and job["kind"] == "snapshot"
            snapshot_id = job["result"]["id"]
            listing = (await client.get("/api/v1/snapshots")).json()
            assert [item["name"] for item in listing["items"]] == ["Before QA"] and listing["retention_days"] == 30
            renamed = (await client.patch(f"/api/v1/snapshots/{snapshot_id}", json={"name": "Renamed"})).json()
            assert renamed["name"] == "Renamed"

            record = record_named(dataset, "train0.jpg")
            await client.post("/api/v1/images/move", json={"image_ids": [record.id], "split": "val"})
            diff = (await client.get(f"/api/v1/snapshots/{snapshot_id}/diff")).json()
            assert diff["counts"]["moved"] == 1 and set(diff["subsets"]) == {"moved"}
            gallery = (await client.get("/api/v1/images", params={"subset": diff["subsets"]["moved"]})).json()
            assert [item["id"] for item in gallery["items"]] == [record.id]
            assert (await client.get("/api/v1/images", params={"subset": "gone"})).status_code == 410

            archive = await client.get(f"/api/v1/snapshots/{snapshot_id}/download", params={"kind": "labels"})
            assert archive.status_code == 200 and archive.headers["content-type"] == "application/zip"

            options = (await client.get("/api/v1/training/options")).json()
            assert options["snapshot_link"] == "link" and [item["id"] for item in options["snapshots"]] == [snapshot_id]
            # the folders as they were in the snapshot, with their counts (train0 moved to val since)
            assert options["snapshots"][0]["folder_counts"] == {"train": {"images/train": 2}, "val": {"images/val": 2}, "test": {"images/test": 2}}
            started = (await client.post("/api/v1/training/runs", json={"yaml": ARGS, "snapshot": snapshot_id, "confirm": True})).json()
            assert started["queued"]
            refused = await client.delete(f"/api/v1/snapshots/{snapshot_id}")
            assert refused.status_code == 400 and "train" in refused.json()["detail"]
            assert (await client.get("/api/v1/dataset")).json()["training_lock"] is None

            assert (await client.post(f"/api/v1/snapshots/{snapshot_id}/recount")).json()["state"] == "done"
            assert (await client.get("/api/v1/snapshots/20990101-000000")).status_code == 404

    asyncio.run(scenario())
