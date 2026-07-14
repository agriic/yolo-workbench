import asyncio

from httpx import ASGITransport, AsyncClient

from test_dataset import make_dataset
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.web import create_app


def make_app(dataset, tmp_path):
    return create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"))


def test_metadata_images_and_edit_api(tmp_path):
    dataset, label = make_dataset(tmp_path)
    async def run():
        async with AsyncClient(transport=ASGITransport(app=make_app(dataset, tmp_path)), base_url="http://test") as client:
            metadata = (await client.get("/api/v1/dataset")).json()
            assert metadata["image_count"] == 1
            image_id = (await client.get("/api/v1/images")).json()["items"][0]["id"]
            response = await client.put(f"/api/v1/images/{image_id}/annotations", json={"annotations": [{"class_id": 0, "points": [0.5, 0.5, 0.1, 0.1]}]})
            assert response.status_code == 200
            assert label.read_text().startswith("0 ")
    asyncio.run(run())


def test_media_and_validation_endpoints(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    async def run():
        async with AsyncClient(transport=ASGITransport(app=make_app(dataset, tmp_path)), base_url="http://test") as client:
            image_id = next(iter(dataset.images))
            assert (await client.get(f"/api/v1/images/{image_id}/thumbnail")).headers["content-type"] == "image/jpeg"
            assert (await client.get("/api/v1/issues")).status_code == 200
    asyncio.run(run())


def test_embeddings_endpoints_report_state(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    async def run():
        async with AsyncClient(transport=ASGITransport(app=make_app(dataset, tmp_path)), base_url="http://test") as client:
            state = (await client.get("/api/v1/embeddings")).json()
            assert state["brain_key"] == "gt_viz"
            assert state["dimensions"] == 3
            assert state["status"] in {"idle", "unavailable"}
            started = (await client.post("/api/v1/embeddings/compute")).json()
            assert started["status"] in {"computing", "ready", "unavailable"}
    asyncio.run(run())


def test_bulk_endpoints(tmp_path):
    dataset, label = make_dataset(tmp_path)
    label.write_text("1 0.5 0.5 0.2 0.4\n1 0.5 0.5 0.2 0.4\n")
    dataset = type(dataset)(dataset.yaml_path, "detection")
    async def run():
        async with AsyncClient(transport=ASGITransport(app=make_app(dataset, tmp_path)), base_url="http://test") as client:
            fixed = await client.post("/api/v1/issues/fix-bulk", json={"kind": "duplicate"})
            assert fixed.status_code == 200
            assert fixed.json()["fixed"] == 1
            image_id = next(iter(dataset.images))
            annotation_id = dataset.images[image_id].annotations[0].id
            bulk = await client.post("/api/v1/objects/bulk", json={"operations": [{"image_id": image_id, "annotation_id": annotation_id, "action": "relabel", "class_id": 0}]})
            assert bulk.status_code == 200
            assert bulk.json()["applied"] == 1
            assert label.read_text().startswith("0 ")
            undo = (await client.post("/api/v1/history/undo")).json()
            assert undo["image_ids"] == [image_id]
    asyncio.run(run())
