import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from test_dataset import make_dataset
from yolo_workbench.dataset import Dataset, DatasetError
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.web import create_app


def three_images(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    for name in ("b", "c"):
        Image.new("RGB", (100, 80)).save(tmp_path / "train" / f"{name}.jpg")
        (tmp_path / "train" / f"{name}.txt").write_text("")
    (tmp_path / "train" / "c.txt").write_text("0 0.5 0.5 2 2\n")  # out of range
    dataset = Dataset(dataset.yaml_path, "detection")
    ids = {record.path.name: record.id for record in dataset.images.values()}
    return dataset, ids


def names(dataset, **filters):
    return [item["name"] for item in dataset.list_images(**filters)["items"]]


def test_review_status_filters_persists_and_counts(tmp_path):
    dataset, ids = three_images(tmp_path)
    dataset.set_review([ids["b.jpg"]], "done")
    dataset.set_review([ids["c.jpg"]], "needs_work")

    assert names(dataset, review="done") == ["b.jpg"]
    assert names(dataset, review="needs_work") == ["c.jpg"]
    assert names(dataset, review="unreviewed") == ["sample.jpg"]
    assert dataset.review_counts() == {"done": 1, "needs_work": 1, "unreviewed": 1}
    assert dataset.detail(ids["b.jpg"])["review"] == "done"

    stored = json.loads((tmp_path / ".yolo-workbench" / "review.json").read_text())
    assert stored["images"]["train/b.jpg"]["status"] == "done"
    reloaded = Dataset(dataset.yaml_path, "detection")
    assert names(reloaded, review="done") == ["b.jpg"]

    dataset.set_review([ids["b.jpg"]], None)
    assert names(dataset, review="done") == []
    with pytest.raises(DatasetError):
        dataset.set_review([ids["b.jpg"]], "maybe")


def test_review_status_follows_moves_and_survives_undone_deletes(tmp_path):
    dataset, ids = three_images(tmp_path)
    (tmp_path / "val").mkdir()
    dataset.yaml_path.write_text(dataset.yaml_path.read_text() + "val: val\n")
    dataset = Dataset(dataset.yaml_path, "detection")
    ids = {record.path.name: record.id for record in dataset.images.values()}
    dataset.set_review([ids["b.jpg"]], "done")

    dataset.move_images([ids["b.jpg"]], "val")
    assert names(dataset, split="val", review="done") == ["b.jpg"]
    assert "val/b.jpg" in json.loads((tmp_path / ".yolo-workbench" / "review.json").read_text())["images"]

    dataset.delete_images([ids["b.jpg"]])
    dataset.history("undo")
    assert names(dataset, review="done") == ["b.jpg"]


def test_issue_filter(tmp_path):
    dataset, _ = three_images(tmp_path)
    assert names(dataset, issue="any") == ["c.jpg"]
    assert names(dataset, issue="out_of_range") == ["c.jpg"]
    assert names(dataset, issue="none") == ["b.jpg", "sample.jpg"]
    assert names(dataset, issue="zero_area") == []


def test_next_unreviewed_wraps_and_follows_filters(tmp_path):
    dataset, ids = three_images(tmp_path)  # order: b, c, sample
    dataset.set_review([ids["c.jpg"]], "done")
    assert dataset.next_unreviewed(1)["item"]["name"] == "sample.jpg"
    assert dataset.next_unreviewed(1)["index"] == 2
    dataset.set_review([ids["sample.jpg"]], "done")
    assert dataset.next_unreviewed(1)["item"]["name"] == "b.jpg"  # wrapped
    assert dataset.next_unreviewed(0, issue="any")["index"] is None


def test_review_api(tmp_path):
    dataset, ids = three_images(tmp_path)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"))

    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/images/review", json={"image_ids": [ids["b.jpg"]], "status": "done"})
            assert response.json()["review_counts"]["done"] == 1
            listed = (await client.get("/api/v1/images", params={"review": "done"})).json()
            assert [item["name"] for item in listed["items"]] == ["b.jpg"] and listed["items"][0]["review"] == "done"
            assert (await client.get("/api/v1/images", params={"review": "bogus"})).status_code == 422
            assert (await client.get("/api/v1/images", params={"issue": "any"})).json()["total"] == 1
            following = (await client.get("/api/v1/images/next-unreviewed", params={"start": 0})).json()
            assert following["item"]["name"] == "c.jpg" and following["index"] == 1
            assert (await client.get("/api/v1/dataset")).json()["review_counts"]["unreviewed"] == 2
            assert (await client.post("/api/v1/images/review", json={"image_ids": ["missing"], "status": "done"})).status_code == 404

    asyncio.run(run())
