import asyncio
from pathlib import Path

import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from PIL import Image

from yolo_workbench import importer
from yolo_workbench.dataset import DatasetError, WriteConflict
from yolo_workbench.jobs import JobManager
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.web import create_workspace_app
from yolo_workbench.workspace import Workspace


def make_workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "projects", jobs=JobManager(synchronous=True), background_probe=False)


def client_for(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1", headers={"X-Workbench": "1"})


def make_app(workspace: Workspace, tmp_path: Path, **kwargs):
    return create_workspace_app(workspace, media_cache=MediaCache(tmp_path / "media"), **kwargs)


def make_source(tmp_path: Path) -> Path:
    """A YOLO export with its own class order: 0=dog, 1=cat, plus one unlabeled image."""
    source = tmp_path / "export"
    for split, name, label in [("train", "a", "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"), ("train", "b", "1 0.5 0.5 0.4 0.4\n"), ("valid", "c", None)]:
        (source / split / "images").mkdir(parents=True, exist_ok=True)
        (source / split / "labels").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), {"a": "red", "b": "green", "c": "blue"}[name]).save(source / split / "images" / f"{name}.jpg")
        if label:
            (source / split / "labels" / f"{name}.txt").write_text(label)
    (source / "data.yaml").write_text(yaml.safe_dump({"names": ["dog", "cat"]}))
    return source


def test_create_list_rename_trash_restore(tmp_path):
    workspace = make_workspace(tmp_path)
    project = workspace.create("Street Scenes!", "detection", ["car", "bus"])
    assert project["slug"] == "street-scenes"
    directory = workspace.root / "street-scenes"
    config = yaml.safe_load((directory / "data.yaml").read_text())
    assert "path" not in config and config["names"] == {0: "car", 1: "bus"}
    assert (directory / "images" / "test").is_dir() and (directory / "labels" / "val").is_dir() and (directory / "models").is_dir()
    assert workspace.create("Street Scenes", "segmentation")["slug"] == "street-scenes-2"

    session = workspace.session("street-scenes")
    assert session.dataset.images == {} and session.dataset.names == {0: "car", 1: "bus"}
    assert workspace.update("street-scenes", name="Streets")["name"] == "Streets"
    listed = workspace.list_projects()
    assert [item["slug"] for item in listed["projects"]] == ["street-scenes", "street-scenes-2"] or len(listed["projects"]) == 2

    entry = workspace.trash("street-scenes")
    assert workspace.active is None and not directory.exists()
    assert workspace.list_trash()[0]["name"] == "Streets"
    restored = workspace.restore(entry["entry"])
    assert restored["slug"] == "street-scenes" and directory.exists()
    workspace.trash("street-scenes-2")
    workspace.purge(workspace.list_trash()[0]["entry"])
    assert workspace.list_trash() == []
    with pytest.raises(KeyError):
        workspace.session("../etc")


def test_classification_project_and_adopt(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Pets", "classification", ["dog", "cat"])
    config = yaml.safe_load((workspace.root / "pets" / "data.yaml").read_text())
    assert config["names"] == ["cat", "dog"] and (workspace.root / "pets" / "val" / "dog").is_dir()
    assert workspace.session("pets").dataset.names == {0: "cat", 1: "dog"}

    loose = workspace.root / "loose"
    (loose / "train" / "ant").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(loose / "train" / "ant" / "x.png")
    listed = workspace.list_projects()
    assert listed["unmanaged"][0]["slug"] == "loose" and listed["unmanaged"][0]["adoptable"]
    workspace.adopt("loose", "classification")
    assert (loose / "data.yaml").is_file()
    assert workspace.session("loose").dataset.names == {0: "ant"}


def test_single_open_project_refuses_switch_while_busy(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("One", "detection", ["a"])
    workspace.create("Two", "detection", ["a"])
    first = workspace.session("one")
    first.predictor.job = {"state": "running"}
    with pytest.raises(WriteConflict):
        workspace.session("two")
    with pytest.raises(WriteConflict):
        workspace.trash("one")
    first.predictor.job = {"state": "done"}
    assert workspace.session("two").slug == "two"
    assert workspace.active.slug == "two"


def test_scan_suggests_mapping_by_name(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Pets", "detection", ["cat"])
    dataset = workspace.session("pets").dataset
    scanned = importer.scan(dataset, make_source(tmp_path))
    assert scanned["images"] == 3 and scanned["labeled"] == 2
    assert scanned["names_source"] == "data.yaml"
    assert scanned["splits"] == {"train": 2, "val": 1}
    assert [(item["key"], item["name"], item["count"], item["suggested"]) for item in scanned["classes"]] == [("0", "dog", 1, "new"), ("1", "cat", 2, 0)]


def test_import_maps_classes_splits_and_skips_duplicates(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Pets", "detection", ["cat"])
    dataset = workspace.session("pets").dataset
    source = make_source(tmp_path)
    result = importer.run_import(dataset, source, {"split_mode": "source", "split": "train", "mapping": {"0": "new", "1": 0}})
    assert result["imported"] == 3 and result["labels"] == 2 and result["by_split"] == {"train": 2, "val": 1}
    assert dataset.names == {0: "cat", 1: "dog"}
    labels = dataset.root / "labels" / "train"
    assert (labels / "a.txt").read_text() == "1 0.5 0.5 0.2 0.2\n0 0.3 0.3 0.1 0.1\n"
    assert dataset.list_images(class_id=0)["total"] == 2
    assert dataset.list_images(split="val")["total"] == 1
    # the class addition is one undoable schema change
    assert dataset.history_state()["can_undo"]

    again = importer.run_import(dataset, source, {"split_mode": "fixed", "split": "train"})
    assert again["duplicates"] == 3 and again["imported"] == 0
    assert (dataset.root / WORKBENCH_LOG).read_text().count("\n") == 2


WORKBENCH_LOG = Path(".yolo-workbench") / "imports.jsonl"


def test_import_auto_split_is_deterministic_and_renames_collisions(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Flat", "detection", ["x"])
    dataset = workspace.session("flat").dataset
    source = tmp_path / "flat"
    source.mkdir()
    for index in range(10):
        Image.new("RGB", (8, 8), (index * 20, 0, 0)).save(source / f"img{index}.png")
    Image.new("RGB", (8, 8), (1, 2, 3)).save(dataset.root / "images" / "train" / "img0.png")
    existing = (dataset.root / "images" / "train" / "img0.png").read_bytes()
    dataset.rescan()
    options = {"split_mode": "auto", "ratios": {"train": 0.7, "val": 0.2, "test": 0.1}, "seed": 7}
    result = importer.run_import(dataset, source, options)
    assert result["by_split"] == {"train": 7, "val": 2, "test": 1}
    assert result["renamed"] == 1
    first = {record.path.name.replace("img0_1", "img0"): record.split for record in dataset.images.values() if record.path.read_bytes() != existing}

    workspace.create("Flat 2", "detection", ["x"])
    other = workspace.session("flat-2").dataset
    importer.run_import(other, source, options)
    second = {record.path.name: record.split for record in other.images.values()}
    assert first == second


def test_classification_import_by_folder_and_unassigned(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Animals", "classification", ["cat"])
    dataset = workspace.session("animals").dataset
    source = tmp_path / "animals"
    (source / "Cat").mkdir(parents=True)
    (source / "owl").mkdir()
    Image.new("RGB", (8, 8), "red").save(source / "Cat" / "1.jpg")
    Image.new("RGB", (8, 8), "blue").save(source / "owl" / "2.jpg")
    Image.new("RGB", (8, 8), "green").save(source / "3.jpg")
    scanned = importer.scan(dataset, source)
    assert {item["key"]: item["suggested"] for item in scanned["classes"]} == {"": None, "Cat": 0, "owl": "new"}
    result = importer.run_import(dataset, source, {"split_mode": "fixed", "split": "val", "mapping": {"": 0}})
    assert result["imported"] == 3
    assert dataset.names == {0: "cat", 1: "owl"}
    assert (dataset.root / "val" / "cat" / "3.jpg").exists() and (dataset.root / "val" / "owl" / "2.jpg").exists()


def test_upload_paths_are_sanitized(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Up", "detection", ["x"])
    dataset = workspace.session("up").dataset
    staging = importer.create_staging(dataset)["staging_id"]
    for bad in ["../x.jpg", "/etc/x.jpg", "a/../../x.jpg", ".hidden/x.jpg", "x.exe", "C:/x.jpg"]:
        with pytest.raises(DatasetError):
            importer.write_staged_file(dataset, staging, bad, b"1")
    importer.write_staged_file(dataset, staging, "batch/images/x.jpg", b"1")
    assert (importer.staging_dir(dataset, staging) / "batch" / "images" / "x.jpg").exists()
    with pytest.raises(DatasetError):
        importer.resolve_source(dataset, str(dataset.root), None)


def test_workspace_api_end_to_end(tmp_path):
    workspace = make_workspace(tmp_path)
    source = make_source(tmp_path)

    async def run():
        app = make_app(workspace, tmp_path, allowed_hosts={"127.0.0.1"}, require_header=True)
        async with client_for(app) as client:
            assert (await client.get("/api/v1/app")).json()["mode"] == "projects"
            created = (await client.post("/api/v1/projects", json={"name": "Demo", "category": "detection", "classes": ["cat"]})).json()
            base = f"/api/v1/projects/{created['slug']}"
            meta = (await client.get(f"{base}/dataset")).json()
            assert meta["image_count"] == 0 and meta["project"]["name"] == "Demo"
            assert meta["capabilities"]["splits"] == ["train", "val", "test"]

            staging = (await client.post(f"{base}/import/staging")).json()["staging_id"]
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    response = await client.put(f"{base}/import/staging/{staging}/files", params={"path": str(path.relative_to(source))}, content=path.read_bytes())
                    assert response.status_code == 200, response.text
            scanned = (await client.post(f"{base}/import/scan", json={"staging_id": staging})).json()
            assert scanned["images"] == 3
            job = (await client.post(f"{base}/import/commit", json={"staging_id": staging, "split_mode": "fixed", "split": "train"})).json()
            assert job["state"] == "done", job
            assert job["result"]["imported"] == 3
            assert not (workspace.root / "demo" / ".yolo-workbench" / "staging" / staging).exists()

            classes = (await client.get(f"{base}/classes")).json()["items"]
            assert [(item["name"], item["annotations"]) for item in classes] == [("cat", 2), ("dog", 1)]
            renamed = await client.patch(f"{base}/classes/1", json={"name": "puppy"})
            assert renamed.status_code == 200
            removed = (await client.post(f"{base}/classes/0/remove", json={"mode": "merge", "target_id": 1})).json()
            assert [item["name"] for item in removed["items"]] == ["puppy"]
            undo = (await client.post(f"{base}/history/undo")).json()
            assert undo["schema_changed"] is True

            images = (await client.get(f"{base}/images")).json()["items"]
            moved = (await client.post(f"{base}/images/move", json={"image_ids": [images[0]["id"]], "split": "val"})).json()
            assert moved["moved"] == 1
            deleted = (await client.post(f"{base}/images/delete", json={"image_ids": [images[1]["id"]]})).json()
            assert deleted["deleted"] == 1
            assert (await client.get(f"{base}/dataset")).json()["image_count"] == 2

            listed = (await client.get("/api/v1/projects")).json()
            assert listed["projects"][0]["summary"]["images"] == 2 and listed["projects"][0]["has_cover"]
            assert (await client.get(f"/api/v1/projects/{created['slug']}/cover")).status_code == 200

            assert (await client.delete(f"/api/v1/projects/{created['slug']}")).status_code == 200
            assert (await client.get(f"{base}/dataset")).status_code == 404
            trash = (await client.get("/api/v1/trash")).json()["items"]
            assert (await client.post(f"/api/v1/trash/{trash[0]['entry']}/restore")).status_code == 200

            # guards: no header on a mutating request, or a foreign Host
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as bare:
                assert (await bare.post("/api/v1/projects", json={"name": "x", "category": "detection"})).status_code == 403
                assert (await bare.get("/api/v1/projects")).status_code == 200
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://evil.example") as evil:
                assert (await evil.get("/api/v1/projects")).status_code == 403

    asyncio.run(run())
