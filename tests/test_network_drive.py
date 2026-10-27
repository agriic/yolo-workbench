"""Datasets on network drives (PLAN-network-drives.md): no request waits on a filesystem call per image.

On a network drive every stat, exists or open is a round trip to the file server, milliseconds instead of
microseconds, so these count the calls made under the dataset folder (fscount) where they must not grow with
the number of images.
"""

import asyncio
import errno
import io
import os
import sys
import time
import zipfile

import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from PIL import Image

from fscount import counting
from test_labelqa import wait
from yolo_workbench import download, fsscan, ownership
from yolo_workbench import classes as class_ops
from yolo_workbench.dataset import Dataset, WriteConflict
from yolo_workbench.labelqa import LabelQAManager
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.predictor import PredictorManager
from yolo_workbench.web import create_app

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="pathlib on 3.10 bypasses the call counter")

IMAGES = 40


def make_share(root, images=IMAGES):
    """A dataset with parallel images/ and labels/ folders; every tenth image has no label, plus one orphan label."""
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    picture = Image.new("RGB", (64, 48), "white")
    for index in range(images):
        split = "val" if index % 4 == 0 else "train"
        picture.save(root / "images" / split / f"img{index:03d}.jpg")
        if index % 10:
            (root / "labels" / split / f"img{index:03d}.txt").write_text(f"{index % 2} 0.5 0.5 0.2 0.2\n")
    (root / "labels" / "train" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (root / "data.yaml").write_text(yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val", "names": {0: "zero", 1: "one"}}))
    return Dataset(root / "data.yaml", "detection")  # the first opening probes the image sizes


def labelled(dataset):
    return [record for record in dataset.images.values() if dataset._mtimes[record.label_path] is not None]


class Backend:
    def __init__(self, path, category, conf, iou):
        self.names = {0: "zero", 1: "one"}

    def predict(self, image_path, conf=None):
        return [{"class_id": 1, "confidence": 0.9, "points": [0.5, 0.5, 0.2, 0.2]}, {"class_id": 0, "confidence": 0.7, "points": [0.2, 0.2, 0.1, 0.1]}]


def test_opening_lists_folders_instead_of_asking_about_each_file(tmp_path):
    def opened(share, images):
        make_share(share, images)
        with counting(share) as opening:
            dataset = Dataset(share / "data.yaml", "detection")
        with counting(share) as rescanning:
            assert dataset.rescan() == {"added": 0, "removed": 0, "updated": 0, "names_changed": False}
        return dataset, opening, rescanning

    small, small_opening, small_rescan = opened(tmp_path / "small", IMAGES // 2)
    dataset, opening, rescanning = opened(tmp_path / "share", IMAGES)
    assert len(dataset.images) == IMAGES and len(labelled(dataset)) == IMAGES - IMAGES // 10
    assert dataset.orphan_labels == [tmp_path / "share" / "labels" / "train" / "orphan.txt"]
    assert all(record.width == 64 and record.image_mtime for record in dataset.images.values())
    # the YAML and the folders, whatever the number of images; each label file is opened once, images not at all
    for calls, fewer in ((opening, small_opening), (rescanning, small_rescan)):
        assert calls["stat"] + calls["lstat"] + calls["scandir"] == fewer["stat"] + fewer["lstat"] + fewer["scandir"], (calls, fewer)
    assert opening["open"] - small_opening["open"] == len(labelled(dataset)) - len(labelled(small))
    assert rescanning["open"] == small_rescan["open"]


def test_label_qa_listings_ask_the_disk_nothing(tmp_path):
    share = tmp_path / "share"
    dataset = make_share(share)
    predictor = PredictorManager(dataset, backend_factory=Backend)
    (tmp_path / "model.pt").write_bytes(b"fake")
    predictor.load(str(tmp_path / "model.pt"))
    qa = LabelQAManager(dataset, predictor)
    qa.start()
    wait(qa)
    with counting(share) as calls:
        results = [qa.findings(**params) for params in ({}, {"kind": "wrong_class"}, {"min_conf": 0.1}, {"tolerance": 9.0}, {"split": "val"})]
    assert results[0]["total"] > 0 and results[0]["stale"] == 0
    assert sum(calls.values()) == 0, calls


def test_model_list_does_not_walk_the_image_folders(tmp_path):
    share = tmp_path / "share"
    dataset = make_share(share)
    (share / "models").mkdir()
    (share / "models" / "mine.pt").write_bytes(b"weights")
    predictor = PredictorManager(dataset, backend_factory=Backend)
    with counting(share / "images" / "train") as image_calls, counting(share / "labels" / "train") as label_calls:
        items = predictor.discover_models()["items"]
    assert "mine.pt" in [item["name"] for item in items]
    # never listed (each holds a file per image); asking whether labels/ exists is once per split
    assert image_calls["scandir"] + image_calls["listdir"] + label_calls["scandir"] + label_calls["listdir"] == 0, (image_calls, label_calls)


def test_cached_thumbnails_ask_the_disk_nothing(tmp_path):
    share = tmp_path / "share"
    dataset = make_share(share)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media"))
    image_ids = [record.id for record in labelled(dataset)[:5]]

    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for image_id in image_ids:
                assert (await client.get(f"/api/v1/images/{image_id}/thumbnail", params={"annotated": "true"})).status_code == 200
            with counting(share) as calls:
                for image_id in image_ids:
                    assert (await client.get(f"/api/v1/images/{image_id}/thumbnail", params={"annotated": "true"})).status_code == 200
            assert sum(calls.values()) == 0, calls
            # a label saved here changes the annotated thumbnail's key without anyone asking the disk
            detail = (await client.get(f"/api/v1/images/{image_ids[0]}")).json()
            before = await client.get(f"/api/v1/images/{image_ids[0]}/thumbnail", params={"annotated": "true"})
            await client.put(f"/api/v1/images/{image_ids[0]}/annotations", json={"revision": detail["revision"], "annotations": []})
            after = await client.get(f"/api/v1/images/{image_ids[0]}/thumbnail", params={"annotated": "true"})
            assert after.headers["etag"] != before.headers["etag"]

    asyncio.run(run())


def test_a_new_time_alone_is_no_conflict_but_new_content_is(tmp_path):
    dataset = make_share(tmp_path / "share")
    record = labelled(dataset)[0]
    label = record.label_path
    # a share reporting a file's time late, or a sync tool touching it: the content is what the app wrote
    later = time.time_ns() + 5_000_000_000
    os.utime(label, ns=(later, later))
    dataset.replace_annotations(record.id, [{"class_id": 0, "points": [0.5, 0.5, 0.3, 0.3]}])
    assert label.read_text() == "0 0.5 0.5 0.3 0.3\n"
    os.utime(label, ns=(later + 10**9, later + 10**9))
    dataset.history("undo")
    assert label.read_text() == "1 0.5 0.5 0.2 0.2\n"

    # an edit elsewhere within the share's time resolution: the same time, other content
    stat = label.stat()
    label.write_text("0 0.1 0.1 0.1 0.1\n")
    os.utime(label, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(WriteConflict, match="changed outside"):
        dataset.replace_annotations(record.id, [])
    assert label.read_text() == "0 0.1 0.1 0.1 0.1\n"
    assert record.annotations[0].points == [0.1, 0.1, 0.1, 0.1]  # read again


def test_rescan_notices_a_replaced_image(tmp_path):
    dataset = make_share(tmp_path / "share")
    record = labelled(dataset)[0]
    Image.new("RGB", (80, 60), "black").save(record.path)
    assert (record.width, record.height) == (64, 48)
    assert dataset.rescan()["updated"] == 1
    assert (record.width, record.height) == (80, 60) and record.image_size == record.path.stat().st_size


def test_many_label_files_are_written_and_undone_together(tmp_path):
    share = tmp_path / "share"
    dataset = make_share(share)
    original = {path: path.read_text() for path in (share / "labels").rglob("img*.txt")}
    class_ops.reorder_classes(dataset, [1, 0])
    swapped = {path: path.read_text() for path in original}
    assert all(swapped[path][0] == str(1 - int(text[0])) and swapped[path][1:] == text[1:] for path, text in original.items())
    assert all(record.annotations[0].class_id == 1 - int(original[record.label_path][0]) for record in labelled(dataset))
    dataset.history("undo")
    assert {path: path.read_text() for path in original} == original
    # a bulk edit of many images: one transaction, undone as one
    operations = [{"image_id": record.id, "annotation_id": record.annotations[0].id, "action": "delete", "class_id": None} for record in labelled(dataset)]
    assert dataset.bulk_edit_objects(operations)["files"] == len(original)
    assert all(path.read_text() == "" for path in original)
    dataset.history("undo")
    assert {path: path.read_text() for path in original} == original


def test_labels_download_reads_ahead_and_keeps_the_order(tmp_path):
    dataset = make_share(tmp_path / "share", images=3 * download.READ_AHEAD)
    entries = download.plan(dataset, "labels")
    archive = zipfile.ZipFile(io.BytesIO(b"".join(download.stream(entries))))
    assert archive.namelist() == [entry.name for entry in entries]
    assert archive.read("labels/train/img001.txt") == b"1 0.5 0.5 0.2 0.2\n"


def test_owner_lock_on_a_filesystem_without_locks(tmp_path, monkeypatch, caplog):
    def unsupported(stream):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(ownership, "_lock", unsupported)
    lock = ownership.OwnerLock(tmp_path / ".workbench").acquire()
    assert lock.held and lock.unlocked and "does not support file locks" in caplog.text
    lock.release()

    def held(stream):
        raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")

    monkeypatch.setattr(ownership, "_lock", held)
    with pytest.raises(ownership.AlreadyOwned):
        ownership.OwnerLock(tmp_path / ".workbench").acquire()


def test_listing_answers_like_the_filesystem(tmp_path):
    (tmp_path / "a" / ".yolo-workbench").mkdir(parents=True)
    (tmp_path / "a" / "x.txt").write_text("1")
    (tmp_path / "a" / ".yolo-workbench" / "y.txt").write_text("")
    listing = fsscan.Listing()
    assert listing.stat(tmp_path / "a" / "x.txt")[1] == 1
    assert listing.is_dir(tmp_path / "a") and not listing.exists(tmp_path / "b")
    upper = tmp_path / "a" / "X.TXT"
    assert listing.exists(upper) == upper.exists()  # case rules as the filesystem has them
    assert listing.files(tmp_path, lambda path, name: name == ".yolo-workbench") == [str(tmp_path / "a" / "x.txt")]
    assert fsscan.read_text(tmp_path / "a" / "x.txt")[0] == "1" and fsscan.read_text(tmp_path / "none.txt") == (None, None, None)
    if hasattr(os, "symlink"):
        try:
            os.symlink(tmp_path / "a", tmp_path / "link", target_is_directory=True)
        except OSError:
            return  # Windows without the privilege
        listing = fsscan.Listing()
        assert listing.is_dir(tmp_path / "link") and listing.is_symlink(tmp_path / "link")
        assert str(tmp_path / "link" / "x.txt") not in listing.files(tmp_path)  # like rglob: a linked folder is not entered


def test_a_label_that_is_not_utf8_is_an_issue_not_a_failure(tmp_path):
    share = tmp_path / "share"
    make_share(share)
    (share / "labels" / "train" / "img001.txt").write_bytes(b"\xff\xfe broken")
    dataset = Dataset(share / "data.yaml", "detection")
    record = next(record for record in dataset.images.values() if record.label_path.name == "img001.txt")
    assert [issue["kind"] for issue in record.issues] == ["unreadable_label"]
