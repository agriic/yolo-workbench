import asyncio

from httpx import ASGITransport, AsyncClient
from PIL import Image

from test_dataset import make_dataset
from yolo_workbench.dataset import Dataset
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.web import create_app


def make_app(dataset, tmp_path):
    return create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"))


def test_frontend_es_modules_are_served(tmp_path):
    dataset, _ = make_dataset(tmp_path)

    async def run():
        async with AsyncClient(transport=ASGITransport(app=make_app(dataset, tmp_path)), base_url="http://test") as client:
            index = await client.get("/")
            assert '<script type="module" src="/app.js"></script>' in index.text

            app_script = await client.get("/app.js")
            assert app_script.status_code == 200
            assert 'from "./state.js"' in app_script.text

            for name in ("api.js", "state.js", "grid.js", "canvas.js", "predictor.js", "embeddings.js", "manage.js", "repair.js", "labelqa.js", "projects.js", "projects.html"):
                module = await client.get(f"/{name}")
                assert module.status_code == 200
                assert module.headers["cache-control"] == "no-cache"

    asyncio.run(run())


def test_metadata_images_and_edit_api(tmp_path):
    dataset, label = make_dataset(tmp_path)
    async def run():
        async with AsyncClient(transport=ASGITransport(app=make_app(dataset, tmp_path)), base_url="http://test") as client:
            metadata = (await client.get("/api/v1/dataset")).json()
            assert metadata["image_count"] == 1
            image_id = (await client.get("/api/v1/images")).json()["items"][0]["id"]
            revision = (await client.get(f"/api/v1/images/{image_id}")).json()["revision"]
            response = await client.put(f"/api/v1/images/{image_id}/annotations", json={"revision": revision, "annotations": [{"class_id": 0, "points": [0.5, 0.5, 0.1, 0.1]}]})
            assert response.status_code == 200
            assert response.json()["revision"] > revision
            assert label.read_text().startswith("0 ")
    asyncio.run(run())


def test_server_side_prediction_filter_has_correct_total(tmp_path):
    dataset, _ = make_dataset(tmp_path)

    class PendingPredictor:
        def pending_image_ids(self):
            return set(dataset.images)

    async def run():
        app = create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"), predictor=PendingPredictor())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            filtered = (await client.get("/api/v1/images", params={"has_predictions": "true"})).json()
            assert filtered["total"] == 1
            assert len(filtered["items"]) == 1

    asyncio.run(run())



def test_images_endpoint_filters_by_label_count(tmp_path):
    dataset, _ = make_dataset(tmp_path)

    async def run():
        app = create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/api/v1/images", params={"min_labels": 1})).json()["total"] == 1
            assert (await client.get("/api/v1/images", params={"max_labels": 0})).json()["total"] == 0
            assert (await client.get("/api/v1/images", params={"min_labels": -1})).status_code == 422

    asyncio.run(run())

def test_images_endpoint_filters_by_folder(tmp_path):
    root = tmp_path / "ds"
    for folder, count in (("images/a", 2), ("images/b", 3), ("images/val", 1), ("extra", 1)):
        (root / folder).mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (16, 16)).save(root / folder / f"{folder.replace('/', '-')}-{index}.jpg")
    (root / "listed.txt").write_text("extra/extra-0.jpg\n")
    (root / "data.yaml").write_text("train: [images/a, images/b, listed.txt]\nval: images/val\nnames: [cat]\n")
    dataset = Dataset(root / "data.yaml", "detection")
    assert dataset.capabilities()["folders"] == {"train": ["images/a", "images/b"], "val": ["images/val"]}

    async def run():
        async with AsyncClient(transport=ASGITransport(app=make_app(dataset, tmp_path)), base_url="http://test") as client:
            async def names(**params):
                return sorted(item["name"] for item in (await client.get("/api/v1/images", params={"limit": 50, **params})).json()["items"])

            assert await names(folder="images/b") == ["images-b-0.jpg", "images-b-1.jpg", "images-b-2.jpg"]
            assert await names(folder="images/b", search="-1") == ["images-b-1.jpg"]
            assert await names(folder="images/b", split="val") == []
            assert await names(folder="images/missing") == []
            assert len(await names(folder="")) == 7  # a .txt list's image is in no folder, but listed without the filter
            following = (await client.get("/api/v1/images/next-unreviewed", params={"folder": "images/a"})).json()
            assert following["total"] == 2 and following["item"]["name"] == "images-a-0.jpg"

    asyncio.run(run())


def test_image_and_object_endpoints_support_random_access_slices(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    split = tmp_path / "train"
    for index in range(12):
        Image.new("RGB", (24, 24), "white").save(split / f"image-{index:02}.jpg")
        (split / f"image-{index:02}.txt").write_text(f"{index % 2} 0.5 0.5 0.25 0.25\n")
    dataset = Dataset(dataset.yaml_path, "detection")

    async def run():
        async with AsyncClient(transport=ASGITransport(app=make_app(dataset, tmp_path)), base_url="http://test") as client:
            middle = (await client.get("/api/v1/images", params={"offset": 4, "limit": 3})).json()
            assert middle["total"] == 13
            assert [item["name"] for item in middle["items"]] == ["image-04.jpg", "image-05.jpg", "image-06.jpg"]

            filtered_end = (await client.get("/api/v1/images", params={"search": "image-", "offset": 10, "limit": 5})).json()
            assert filtered_end["total"] == 12
            assert [item["name"] for item in filtered_end["items"]] == ["image-10.jpg", "image-11.jpg"]

            beyond_end = (await client.get("/api/v1/images", params={"offset": 100, "limit": 5})).json()
            assert beyond_end == {"total": 13, "items": []}

            class_slice = (await client.get("/api/v1/images", params={"class_id": 0, "split": "train", "offset": 4, "limit": 2})).json()
            assert class_slice["total"] == 6
            assert [item["name"] for item in class_slice["items"]] == ["image-08.jpg", "image-10.jpg"]

            objects = (await client.get("/api/v1/objects", params={"class_id": 0, "offset": 4, "limit": 2})).json()
            assert objects["total"] == 6
            assert [item["image_name"] for item in objects["items"]] == ["image-08.jpg", "image-10.jpg"]

            objects_end = (await client.get("/api/v1/objects", params={"class_id": 0, "offset": 6, "limit": 2})).json()
            assert objects_end == {"total": 6, "items": []}

            sized = (await client.get("/api/v1/objects", params={"class_id": 0, "unit": "percent", "max_width": 100, "min_aspect": 0, "sort": "largest"})).json()
            assert sized["total"] == 6
            assert (await client.get("/api/v1/objects", params={"class_id": 0, "min_width": 101, "unit": "percent"})).json()["total"] == 0
            assert (await client.get("/api/v1/objects", params={"class_id": 0, "sort": "random"})).status_code == 422
            assert (await client.get("/api/v1/objects", params={"class_id": 0, "min_width": -1})).status_code == 422

    asyncio.run(run())


def test_annotation_api_rejects_stale_revision_and_non_finite_values(tmp_path):
    dataset, label = make_dataset(tmp_path)

    async def run():
        async with AsyncClient(transport=ASGITransport(app=make_app(dataset, tmp_path)), base_url="http://test") as client:
            image_id = next(iter(dataset.images))
            revision = (await client.get(f"/api/v1/images/{image_id}")).json()["revision"]
            first = await client.put(
                f"/api/v1/images/{image_id}/annotations",
                json={"revision": revision, "annotations": [{"class_id": 0, "points": [0.5, 0.5, 0.1, 0.1]}]},
            )
            assert first.status_code == 200

            stale = await client.put(
                f"/api/v1/images/{image_id}/annotations",
                json={"revision": revision, "annotations": [{"class_id": 1, "points": [0.2, 0.2, 0.1, 0.1]}]},
            )
            assert stale.status_code == 409
            assert label.read_text().startswith("0 ")

            non_finite = await client.put(
                f"/api/v1/images/{image_id}/annotations",
                content=f'{{"revision":{first.json()["revision"]},"annotations":[{{"class_id":0,"points":[0.5,0.5,NaN,0.1]}}]}}',
                headers={"Content-Type": "application/json"},
            )
            assert non_finite.status_code == 422

    asyncio.run(run())


def test_media_and_validation_endpoints(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    async def run():
        async with AsyncClient(transport=ASGITransport(app=make_app(dataset, tmp_path)), base_url="http://test") as client:
            image_id = next(iter(dataset.images))
            assert (await client.get(f"/api/v1/images/{image_id}/thumbnail")).headers["content-type"] == "image/jpeg"
            assert (await client.get("/api/v1/issues")).status_code == 200
            statistics = (await client.get("/api/v1/statistics")).json()
            assert statistics["summary"]["annotations"] == 1
            assert statistics["class_balance"][1]["name"] == "one"
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


def test_image_file_is_cacheable_only_under_its_current_version(tmp_path):
    dataset, _ = make_dataset(tmp_path)

    async def run():
        async with AsyncClient(transport=ASGITransport(app=make_app(dataset, tmp_path)), base_url="http://test") as client:
            image_id = next(iter(dataset.images))
            version = (await client.get(f"/api/v1/images/{image_id}")).json()["image_version"]
            fresh = await client.get(f"/api/v1/images/{image_id}/file", params={"v": version})
            assert fresh.status_code == 200 and "immutable" in fresh.headers["cache-control"]
            stale = await client.get(f"/api/v1/images/{image_id}/file", params={"v": "old"})
            assert stale.headers["cache-control"] == "no-cache"
            # editing the image changes its version
            Image.new("RGB", (100, 80), "black").save(dataset.images[image_id].path)
            assert (await client.get(f"/api/v1/images/{image_id}")).json()["image_version"] != version

    asyncio.run(run())
