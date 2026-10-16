import asyncio

import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from PIL import Image

from test_dataset import make_dataset
from yolo_workbench import classes
from yolo_workbench.dataset import Dataset, DatasetError, WriteConflict
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.web import create_app


def issues_of(dataset, kind):
    return [issue for issue in dataset.issues() if issue["kind"] == kind]


def test_malformed_line_fix_removes_only_that_line_and_undoes(tmp_path):
    dataset, label = make_dataset(tmp_path)
    original = "1 0.5 0.5 0.2 0.4\nnot a label\n0 0.2 0.2 0.1 0.1\n"
    label.write_text(original)
    dataset = Dataset(dataset.yaml_path, "detection")
    [issue] = issues_of(dataset, "malformed_label")
    assert issue["fixable"] and issue["line"] == 2

    dataset.fix_issue(issue["id"])
    assert label.read_text() == "1 0.5 0.5 0.2 0.4\n0 0.2 0.2 0.1 0.1\n"
    assert not issues_of(dataset, "malformed_label")
    record = next(iter(dataset.images.values()))
    assert len(record.annotations) == 2

    dataset.history("undo")
    assert label.read_text() == original
    assert issues_of(dataset, "malformed_label")


def test_bulk_malformed_fix_covers_every_file(tmp_path):
    dataset, label = make_dataset(tmp_path)
    label.write_text("1 0.5 0.5 0.2 0.4\n1 2 3\n")
    Image.new("RGB", (100, 80)).save(tmp_path / "train" / "other.jpg")
    (tmp_path / "train" / "other.txt").write_text("x\ny\n0 0.5 0.5 0.1 0.1\n")
    dataset = Dataset(dataset.yaml_path, "detection")
    assert len(issues_of(dataset, "malformed_label")) == 3

    result = dataset.fix_issues_bulk("malformed_label")
    assert (result["fixed"], result["files"]) == (3, 2)
    assert (tmp_path / "train" / "other.txt").read_text() == "0 0.5 0.5 0.1 0.1\n"
    dataset.history("undo")
    assert (tmp_path / "train" / "other.txt").read_text() == "x\ny\n0 0.5 0.5 0.1 0.1\n"


def test_raw_label_text_round_trip_conflict_and_undo(tmp_path):
    dataset, label = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    raw = dataset.label_text(record.id)
    assert raw["text"] == "1 0.5 0.5 0.2 0.4\n" and raw["exists"]

    detail = dataset.write_label_text(record.id, "0 0.1 0.1 0.1 0.1\r\nbroken\r\n\r\n", raw["revision"])
    assert label.read_text() == "0 0.1 0.1 0.1 0.1\nbroken\n"
    assert [issue["kind"] for issue in detail["issues"]] == ["malformed_label"]
    # the old revision is stale now
    with pytest.raises(WriteConflict):
        dataset.write_label_text(record.id, "", raw["revision"])

    dataset.history("undo")
    assert label.read_text() == "1 0.5 0.5 0.2 0.4\n"


def test_orphan_label_moves_to_trash_and_back(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    orphan = tmp_path / "train" / "ghost.txt"
    orphan.write_text("0 0.5 0.5 0.1 0.1\n")
    dataset = Dataset(dataset.yaml_path, "detection")
    [issue] = issues_of(dataset, "orphan_label")
    assert issue["fixable"] and issue["path"] == "train/ghost.txt"

    dataset.fix_issue(issue["id"])
    assert not orphan.exists()
    assert list((tmp_path / ".yolo-workbench" / "trash").rglob("ghost.txt"))
    assert not issues_of(dataset, "orphan_label")

    dataset.history("undo")
    assert orphan.read_text() == "0 0.5 0.5 0.1 0.1\n"
    assert issues_of(dataset, "orphan_label")
    dataset.history("redo")
    assert not orphan.exists()


def test_bulk_orphan_fix(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    for name in ("a", "b"):
        (tmp_path / "train" / f"{name}.txt").write_text("")
    dataset = Dataset(dataset.yaml_path, "detection")
    assert dataset.fix_issues_bulk("orphan_label")["fixed"] == 2
    assert not issues_of(dataset, "orphan_label")


def unknown_dataset(tmp_path):
    dataset, label = make_dataset(tmp_path)  # names 0 zero, 1 one
    label.write_text("1 0.5 0.5 0.2 0.4\n5 0.2 0.2 0.1 0.1\n5 0.7 0.7 0.1 0.1\n")
    return Dataset(dataset.yaml_path, "detection"), label


def test_unknown_class_map_and_delete(tmp_path):
    dataset, label = unknown_dataset(tmp_path)
    assert classes.unknown_classes(dataset) == [{"id": 5, "annotations": 2, "images": 1}]

    classes.resolve_unknown(dataset, 5, "map", target_id=0)
    assert label.read_text() == "1 0.5 0.5 0.2 0.4\n0 0.2 0.2 0.1 0.1\n0 0.7 0.7 0.1 0.1\n"
    assert not issues_of(dataset, "unknown_class")
    dataset.history("undo")
    assert classes.unknown_classes(dataset)

    classes.resolve_unknown(dataset, 5, "delete")
    assert label.read_text() == "1 0.5 0.5 0.2 0.4\n"


def test_unknown_class_added_as_next_contiguous_id(tmp_path):
    dataset, label = unknown_dataset(tmp_path)
    classes.resolve_unknown(dataset, 5, "add", name="five")
    assert dataset.names == {0: "zero", 1: "one", 2: "five"}
    assert label.read_text() == "1 0.5 0.5 0.2 0.4\n2 0.2 0.2 0.1 0.1\n2 0.7 0.7 0.1 0.1\n"
    assert yaml.safe_load(dataset.yaml_path.read_text())["names"] == {0: "zero", 1: "one", 2: "five"}
    dataset.history("undo")
    assert dataset.names == {0: "zero", 1: "one"}
    assert classes.unknown_classes(dataset)[0]["id"] == 5


def test_unknown_class_add_refuses_to_swallow_another_unknown_id(tmp_path):
    dataset, label = make_dataset(tmp_path)
    label.write_text("2 0.5 0.5 0.2 0.4\n5 0.2 0.2 0.1 0.1\n")
    dataset = Dataset(dataset.yaml_path, "detection")
    with pytest.raises(DatasetError, match="resolve class 2 first"):
        classes.resolve_unknown(dataset, 5, "add", name="five")
    with pytest.raises(DatasetError):
        classes.resolve_unknown(dataset, 1, "delete")  # a known class


def test_repair_api(tmp_path):
    dataset, label = unknown_dataset(tmp_path)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"))

    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            data = (await client.get("/api/v1/issues")).json()
            assert data["unknown_classes"] == [{"id": 5, "annotations": 2, "images": 1}]
            image_id = next(iter(dataset.images))
            raw = (await client.get(f"/api/v1/images/{image_id}/label-text")).json()
            response = await client.put(f"/api/v1/images/{image_id}/label-text", json={"revision": raw["revision"], "text": "0 0.5 0.5 0.1 0.1\n"})
            assert response.status_code == 200
            assert label.read_text() == "0 0.5 0.5 0.1 0.1\n"
            response = await client.post("/api/v1/classes/unknown/5", json={"action": "delete"})
            assert response.status_code == 400  # no label uses class 5 any more

    asyncio.run(run())
