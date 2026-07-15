import asyncio
import json
import time

from httpx import ASGITransport, AsyncClient
from PIL import Image

from test_dataset import make_dataset
from yolo_workbench.dataset import Dataset
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.web import create_app


def test_size_index_written_and_reused(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    index_path = tmp_path / ".yolo-workbench" / "index.json"
    assert index_path.exists()
    entry = next(iter(json.loads(index_path.read_text()).values()))
    assert (entry["width"], entry["height"]) == (100, 80)
    # poison the cache: a matching (mtime, size) entry must be trusted without re-probing
    poisoned = {path: {**value, "width": 123, "height": 45} for path, value in json.loads(index_path.read_text()).items()}
    index_path.write_text(json.dumps(poisoned))
    dataset = Dataset(dataset.yaml_path, "detection")
    record = next(iter(dataset.images.values()))
    assert (record.width, record.height) == (123, 45)
    assert dataset.indexing == {"done": 1, "total": 1}


def test_size_index_invalidated_on_mtime_change(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    Image.new("RGB", (50, 40), "black").save(record.path)
    future = time.time_ns() + 10_000_000_000
    import os

    os.utime(record.path, ns=(future, future))
    dataset = Dataset(dataset.yaml_path, "detection")
    record = next(iter(dataset.images.values()))
    assert (record.width, record.height) == (50, 40)


def test_unreadable_image_issue_survives_reread(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    record.path.write_bytes(b"not an image")
    dataset = Dataset(dataset.yaml_path, "detection")
    record = next(iter(dataset.images.values()))
    assert any(issue["kind"] == "unreadable_image" for issue in record.issues)
    dataset._read_record(record)
    assert any(issue["kind"] == "unreadable_image" for issue in record.issues)


def test_thumbnail_cache_and_invalidation_after_edit(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    cache_dir = tmp_path / "media-cache"
    app = create_app(dataset, media_cache=MediaCache(cache_dir))

    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            image_id = next(iter(dataset.images))
            first = await client.get(f"/api/v1/images/{image_id}/thumbnail", params={"annotated": "true"})
            etag = first.headers["etag"]
            assert list(cache_dir.glob("*.jpg"))
            revalidated = await client.get(f"/api/v1/images/{image_id}/thumbnail", params={"annotated": "true"}, headers={"If-None-Match": etag})
            assert revalidated.status_code == 304
            # editing annotations bumps the label mtime, so the annotated thumbnail key changes
            time.sleep(0.01)
            detail = (await client.get(f"/api/v1/images/{image_id}")).json()
            await client.put(f"/api/v1/images/{image_id}/annotations", json={"revision": detail["revision"], "annotations": [{"class_id": 0, "points": [0.5, 0.5, 0.1, 0.1]}]})
            after = await client.get(f"/api/v1/images/{image_id}/thumbnail", params={"annotated": "true"}, headers={"If-None-Match": etag})
            assert after.status_code == 200
            assert after.headers["etag"] != etag

    asyncio.run(run())


def test_crop_served_from_cache_with_etag(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"))

    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            image_id = next(iter(dataset.images))
            annotation_id = dataset.images[image_id].annotations[0].id
            first = await client.get(f"/api/v1/objects/{image_id}/{annotation_id}/crop")
            assert first.status_code == 200
            again = await client.get(f"/api/v1/objects/{image_id}/{annotation_id}/crop", headers={"If-None-Match": first.headers["etag"]})
            assert again.status_code == 304

    asyncio.run(run())


def test_media_cache_prunes_oldest(tmp_path):
    cache = MediaCache(tmp_path / "cache", max_bytes=25, prune_every=1000)
    cache.put("aaa", b"0" * 10)
    time.sleep(0.01)
    cache.put("bbb", b"0" * 10)
    time.sleep(0.01)
    cache.put("ccc", b"0" * 10)
    cache.prune()
    assert cache.get("aaa") is None
    assert cache.get("ccc") == b"0" * 10
