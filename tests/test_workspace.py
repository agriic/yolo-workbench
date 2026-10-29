import asyncio
import io
import json
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


def make_workspace(tmp_path: Path, **kwargs) -> Workspace:
    return Workspace(tmp_path / "projects", jobs=JobManager(synchronous=True), background_probe=False, **kwargs)


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


def test_full_pool_refuses_opening_while_busy(tmp_path):
    workspace = make_workspace(tmp_path, max_open=1)
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
    assert workspace.active.slug == "two" and list(workspace.sessions) == ["two"]


def test_several_projects_stay_open_and_the_least_recently_used_idle_one_closes(tmp_path):
    workspace = make_workspace(tmp_path, max_open=2)
    for name in ("One", "Two", "Three"):
        slug = workspace.create(name, "detection", ["a"])["slug"]
        Image.new("RGB", (8, 8)).save(workspace.root / slug / "images" / "train" / "a.jpg")
    one, two = workspace.session("one"), workspace.session("two")
    image = next(iter(one.dataset.images))
    one.dataset.replace_annotations(image, [{"class_id": 0, "points": [0.5, 0.5, 0.2, 0.2]}])
    # alternating between two open projects keeps both, with their undo history
    assert workspace.session("one") is one and workspace.session("two") is two
    assert one.dataset.history_state()["can_undo"]
    assert workspace.info("one")["active"] and workspace.info("two")["active"]
    workspace.session("one")  # two is now the least recently used
    two.predictor.job = {"state": "running"}
    workspace.session("three")  # two is busy, so one closes instead
    assert list(workspace.sessions) == ["two", "three"] and not workspace.info("one")["active"]
    two.predictor.job = {"state": "done"}
    # idle projects close by themselves; busy ones stay
    workspace.sessions["three"].last_used -= 3600
    two.last_used -= 3600
    two.predictor.job = {"state": "running"}
    assert workspace.close_idle(1800) == ["three"] and list(workspace.sessions) == ["two"]
    two.predictor.job = {"state": "done"}
    workspace.close_all()
    assert workspace.sessions == {}


def test_trashed_projects_expire(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Old", "detection", ["a"])
    workspace.create("New", "detection", ["a"])
    old = workspace.trash("old")["entry"]
    workspace.trash("new")
    aged = workspace.trash_root / "old--20000101-000000"
    (workspace.trash_root / old).rename(aged)
    assert workspace.purge_expired_trash(0) == []
    assert workspace.purge_expired_trash(30) == [aged.name]
    assert [item["slug"] for item in workspace.list_trash()] == ["new"]


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


def test_split_with_several_folders_import_move_and_add_folder(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Multi", "detection", ["dog", "cat"])
    root = workspace.root / "multi"
    yaml_path = root / "data.yaml"
    # a hand-written layout: train is two folders, with a comment to keep
    text = yaml_path.read_text().replace("train: images/train\n", "# training data\ntrain:\n- images/train\n- images/extra  # second batch\n")
    yaml_path.write_text(text)
    (root / "images" / "extra").mkdir()
    (root / "labels" / "extra").mkdir()
    dataset = workspace.session("multi").dataset
    assert dataset.capabilities()["folders"]["train"] == ["images/train", "images/extra"]

    result = importer.run_import(dataset, make_source(tmp_path), {"split_mode": "source", "folders": {"train": "images/extra"}})
    assert result["by_split"] == {"train": 2, "val": 1}
    assert (root / "images" / "extra" / "a.jpg").exists() and (root / "labels" / "extra" / "a.txt").exists()
    assert (root / "images" / "val" / "c.jpg").exists()
    counts = {(item["split"], item["path"]): item["images"] for item in dataset.folders()["items"]}
    assert counts[("train", "images/extra")] == 2 and counts[("train", "images/train")] == 0

    # move between folders of the same split, then to another split's folder
    record = next(record for record in dataset.images.values() if record.path.name == "a.jpg")
    assert dataset.move_images([record.id], "train", "images/train")["moved"] == 1
    assert (root / "labels" / "train" / "a.txt").exists()
    assert dataset.move_images([record.id], "train", "images/train")["skipped"] == 1
    with pytest.raises(DatasetError):
        dataset.move_images([record.id], "val", "images/nope")

    # a new folder is created (with its labels twin) and recorded in data.yaml, keeping comments
    added = dataset.add_folder("val", "images/val_2")
    assert added["path"] == "images/val_2"
    assert (root / "images" / "val_2").is_dir() and (root / "labels" / "val_2").is_dir()
    content = yaml_path.read_text()
    assert "# training data" in content and "# second batch" in content
    assert yaml.safe_load(content)["val"] == ["images/val", "images/val_2"]
    assert dataset.capabilities()["folders"]["val"] == ["images/val", "images/val_2"]
    for bad in ["images/train/sub", "images", "../outside", ".hidden"]:
        with pytest.raises(DatasetError):
            dataset.add_folder("val", bad)
    dataset.history("undo")
    assert not (root / "images" / "val_2").exists()
    assert yaml.safe_load(yaml_path.read_text())["val"] == "images/val"
    assert dataset.capabilities()["folders"]["val"] == ["images/val"]


def test_classification_split_with_several_folders(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Animals", "classification", ["cat", "dog"])
    root = workspace.root / "animals"
    dataset = workspace.session("animals").dataset
    dataset.add_folder("train", "train_more")
    assert yaml.safe_load((root / "data.yaml").read_text())["train"] == ["train", "train_more"]
    from yolo_workbench import classes
    classes.add_classes(dataset, ["owl"])
    assert (root / "train_more" / "owl").is_dir() and (root / "train" / "owl").is_dir()
    source = tmp_path / "src"
    (source / "dog").mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(source / "dog" / "1.jpg")
    importer.run_import(dataset, source, {"split_mode": "fixed", "split": "train", "folders": {"train": "train_more"}})
    assert (root / "train_more" / "dog" / "1.jpg").exists()
    record = next(iter(dataset.images.values()))
    assert record.annotations[0].class_id == 1
    dataset.replace_annotations(record.id, [{"class_id": 2, "points": []}])
    assert (root / "train_more" / "owl" / "1.jpg").exists()
    classes.rename_class(dataset, 2, "eagle")
    assert (root / "train_more" / "eagle" / "1.jpg").exists() and dataset.names == {0: "cat", 1: "dog", 2: "eagle"}


def test_crop_box_center_and_region():
    center = {"mode": "center", "unit": "px", "x": 0, "y": 0, "width": 40, "height": 30}
    assert importer.crop_box(100, 50, center) == (30, 10, 70, 40)
    assert importer.crop_box(20, 50, center) == (0, 10, 20, 40)  # narrower image: that side is kept whole
    assert importer.crop_box(100, 50, {**center, "unit": "percent", "width": 50, "height": 50}) == (25, 12, 75, 37)
    region = {"mode": "box", "unit": "px", "x": 80, "y": 10, "width": 40, "height": 20}
    assert importer.crop_box(100, 50, region) == (80, 10, 100, 30)  # clipped to the image
    assert importer.crop_box(60, 50, region) is None
    with pytest.raises(DatasetError):
        importer.crop_options({"mode": "center", "width": 0, "height": 10})
    with pytest.raises(DatasetError):
        importer.crop_options({"mode": "box", "unit": "percent", "width": 120, "height": 10})
    assert importer.crop_options({"mode": "none"}) is None


def test_crop_label_text_clips_boxes_and_polygons():
    box = (50, 0, 100, 100)  # right half of a 100×100 image
    text = (
        "0 0.75 0.5 0.2 0.2\n"  # fully inside
        "1 0.5 0.5 0.2 0.2\n"  # half inside: clipped
        "2 0.2 0.2 0.1 0.1\n"  # outside: dropped
        "3 0.48 0.5 0.1 0.1\n"  # 30% visible... kept; 1 px: dropped below
        "4 0.455 0.5 0.1 0.1\n"  # 5% visible: dropped
        "0 0.4 0.4 0.8 0.4 0.8 0.8 0.4 0.8\n"  # square polygon, right 3/4 inside
        "bad line\n"
    )
    cropped, dropped = importer.crop_label_text(text, box, 100, 100)
    lines = cropped.splitlines()
    assert dropped == 2
    assert lines[0] == "0 0.5 0.5 0.4 0.2"
    assert lines[1] == "1 0.1 0.5 0.2 0.2"
    assert lines[2] == "3 0.03 0.5 0.06 0.1"
    assert lines[3].split()[0] == "0" and sorted(set(map(float, lines[3].split()[1::2]))) == [0.0, 0.6]
    assert lines[4] == "bad line"


def test_import_with_center_crop(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Crop", "detection", ["x"])
    dataset = workspace.session("crop").dataset
    source = tmp_path / "wide"
    source.mkdir()
    Image.new("RGB", (200, 100), "red").save(source / "a.jpg")
    (source / "a.txt").write_text("0 0.5 0.5 0.1 0.2\n0 0.05 0.5 0.1 0.2\n")
    Image.new("RGB", (100, 100), "blue").save(source / "b.png")
    options = {"split_mode": "fixed", "split": "train", "crop": {"mode": "center", "unit": "px", "width": 100, "height": 100}}
    result = importer.run_import(dataset, source, options)
    assert result["imported"] == 2 and result["cropped"] == 1 and result["objects_cropped_out"] == 1
    with Image.open(dataset.root / "images" / "train" / "a.jpg") as image:
        assert image.size == (100, 100)
    assert (dataset.root / "labels" / "train" / "a.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
    # b.png already fits: copied unchanged
    assert (dataset.root / "images" / "train" / "b.png").read_bytes() == (source / "b.png").read_bytes()
    # the same crop again produces identical bytes, so it is caught as a duplicate
    assert importer.run_import(dataset, source, options)["duplicates"] == 2
    region = {**options, "crop": {"mode": "box", "unit": "px", "x": 150, "y": 0, "width": 40, "height": 40}}
    assert importer.run_import(dataset, source, region)["crop_skipped"] == 1  # misses the 100×100 image


def test_import_sample_preview(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Pets", "detection", ["cat"])
    dataset = workspace.session("pets").dataset
    source = make_source(tmp_path)
    scanned = importer.scan(dataset, source)
    assert scanned["samples"] == ["train/images/a.jpg", "train/images/b.jpg", "valid/images/c.jpg"]

    async def run():
        app = make_app(workspace, tmp_path)
        async with client_for(app) as client:
            response = await client.get("/api/v1/projects/pets/import/sample", params={"path": str(source), "file": "train/images/a.jpg"})
            assert response.status_code == 200 and response.headers["x-image-size"] == "32x24"
            bad = await client.get("/api/v1/projects/pets/import/sample", params={"path": str(source), "file": "../export/data.yaml"})
            assert bad.status_code == 400

    asyncio.run(run())


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


def make_coco_source(tmp_path: Path) -> Path:
    """A Roboflow COCO export: <split>/_annotations.coco.json next to the images; the project itself is an unused
    category, cat has another id in valid, and train has a stray YOLO label that must not be used."""
    source = tmp_path / "coco"
    for index, (split, name) in enumerate([("train", "a"), ("train", "b"), ("valid", "c"), (".", "d")]):
        (source / split).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 50), (0, 0, index * 40)).save(source / split / f"{name}.jpg")
    (source / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    animals = {"id": 0, "name": "animals", "supercategory": "none"}
    write_json(source / "train" / "_annotations.coco.json", {
        "categories": [animals, {"id": 1, "name": "dog", "supercategory": "animals"}, {"id": 2, "name": "cat", "supercategory": "animals"}],
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 50}, {"id": 2, "file_name": "b.jpg", "width": 100, "height": 50}, {"id": 3, "file_name": "gone.jpg", "width": 100, "height": 50}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "segmentation": [[10, 10, 30, 10, 30, 30]], "iscrowd": 0},
            # one cat in two parts, the second turned the other way round
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 10, 30, 10], "segmentation": [[50, 10, 60, 10, 60, 20, 50, 20], [70, 10, 70, 20, 80, 20, 80, 10]], "iscrowd": 0},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [40, 30, 20, 10], "segmentation": [], "iscrowd": 0},
            {"id": 4, "image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "segmentation": {"size": [50, 100], "counts": [0, 10]}, "iscrowd": 1},
            {"id": 5, "image_id": 1, "category_id": 9, "bbox": [0, 0, 10, 10], "segmentation": [], "iscrowd": 0},
            {"id": 6, "image_id": 3, "category_id": 1, "bbox": [0, 0, 10, 10], "segmentation": [], "iscrowd": 0},
        ],
    })
    write_json(source / "valid" / "_annotations.coco.json", {
        "categories": [{"id": 1, "name": "cat"}],
        "images": [{"id": 7, "file_name": "c.jpg"}],  # no size: the image's own
        "annotations": [{"id": 1, "image_id": 7, "category_id": 1, "bbox": [-10, 0, 60, 25], "iscrowd": 0}],  # sticks out on the left
    })
    write_json(source / "notes.json", {"categories": [{"id": 0, "name": "zebra"}]})  # not COCO
    return source


def test_coco_scan(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Pets", "segmentation", ["cat"])
    scanned = importer.scan(workspace.session("pets").dataset, make_coco_source(tmp_path))
    assert scanned["format"] == "coco" and scanned["images"] == 4 and scanned["labeled"] == 3
    assert scanned["splits"] == {"train": 2, "val": 1}
    assert scanned["coco"] == {"files": ["train/_annotations.coco.json", "valid/_annotations.coco.json"], "missing_images": 1, "skipped": {"crowd": 1, "unknown_class": 1, "no_shape": 0}}
    # cat is one class although its id differs between the files; the unused parent category is not suggested
    classes = [(item["key"], item["name"], item["count"], item["suggested"]) for item in scanned["classes"]]
    assert classes == [("0", "animals", 0, None), ("1", "dog", 2, "new"), ("2", "cat", 2, 0)]


def test_coco_import_into_segmentation_and_detection(tmp_path):
    workspace = make_workspace(tmp_path)
    source = make_coco_source(tmp_path)
    workspace.create("Seg", "segmentation", ["cat"])
    dataset = workspace.session("seg").dataset
    result = importer.run_import(dataset, source, {"split_mode": "source", "split": "train"})
    assert result["imported"] == 4 and result["labels"] == 3 and result["by_split"] == {"train": 3, "val": 1}
    assert dataset.names == {0: "cat", 1: "dog"}
    labels = dataset.root / "labels"
    assert (labels / "train" / "a.txt").read_text() == (
        "1 0.1 0.2 0.3 0.2 0.3 0.6\n"
        # the parts joined by a zero-width bridge between their closest corners
        "0 0.5 0.2 0.6 0.2 0.7 0.2 0.8 0.2 0.8 0.4 0.7 0.4 0.7 0.2 0.6 0.2 0.6 0.4 0.5 0.4\n"
        "1 0.4 0.6 0.6 0.6 0.6 0.8 0.4 0.8\n"  # only a box: its rectangle
    )
    assert (labels / "train" / "b.txt").read_text() == ""  # listed without objects: a background image
    assert (labels / "val" / "c.txt").read_text() == "0 0 0 0.5 0 0.5 0.5 0 0.5\n"
    assert not (labels / "train" / "d.txt").exists()
    assert not any(issue["kind"] != "missing_label" for record in dataset.images.values() for issue in record.issues)

    workspace.create("Det", "detection", ["cat", "dog"])
    dataset = workspace.session("det").dataset
    importer.run_import(dataset, source, {"split_mode": "source", "mapping": {"1": None}})
    labels = dataset.root / "labels"
    assert (labels / "train" / "a.txt").read_text() == "0 0.65 0.3 0.3 0.2\n"  # dogs dropped by the mapping
    assert (labels / "val" / "c.txt").read_text() == "0 0.25 0.25 0.5 0.5\n"  # clipped to the image


def rle_string(counts: list[int]) -> str:
    """pycocotools' rleToString, to check the decoder against."""
    text = []
    for index, value in enumerate(counts):
        if index > 2:
            value -= counts[index - 2]
        more = True
        while more:
            char = value & 0x1F
            value >>= 5
            more = value != -1 if char & 0x10 else value != 0
            text.append(chr((char | 0x20 if more else char) + 48))
    return "".join(text)


def test_coco_official_layout_with_rle_masks(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Rle", "segmentation")
    dataset = workspace.session("rle").dataset
    source = tmp_path / "coco2017"
    (source / "val2017").mkdir(parents=True)
    Image.new("RGB", (20, 10)).save(source / "val2017" / "x.jpg")
    # a 10×5 block at x 5..14, y 2..6, in column-major runs; and a 4×4 block at x 0..3, y 0..3
    block = [52] + [5, 5] * 9 + [5, 53]
    image = {"id": 1, "file_name": "x.jpg", "width": 20, "height": 10}
    write_json(source / "annotations" / "instances_val2017.json", {
        "categories": [{"id": 1, "name": "sign"}, {"id": 2, "name": "dot"}], "images": [image],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": {"size": [10, 20], "counts": rle_string(block)}, "iscrowd": 0, "bbox": [5, 2, 10, 5]},
            {"id": 2, "image_id": 1, "category_id": 2, "segmentation": {"size": [10, 20], "counts": [0, 4, 6, 4, 6, 4, 6, 4, 166]}, "iscrowd": 0, "bbox": [0, 0, 4, 4]},
        ],
    })
    # the people again, with keypoints: left out, or each person would come in twice
    write_json(source / "annotations" / "person_keypoints_val2017.json", {
        "categories": [{"id": 1, "name": "person", "keypoints": ["nose"]}], "images": [image],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "segmentation": [], "iscrowd": 0}],
    })
    scanned = importer.scan(dataset, source)
    assert scanned["coco"]["files"] == ["annotations/instances_val2017.json"]
    assert scanned["splits"] == {"val": 1}  # from the file's name
    assert [(item["name"], item["count"]) for item in scanned["classes"]] == [("sign", 1), ("dot", 1)]
    importer.run_import(dataset, source, {"split_mode": "source"})
    lines = [list(map(float, line.split())) for line in (dataset.root / "labels" / "val" / "x.txt").read_text().splitlines()]
    assert [line[0] for line in lines] == [0, 1]
    assert sorted(set(lines[0][1::2])) == [0.25, 0.7] and sorted(set(lines[0][2::2])) == [0.2, 0.6]
    assert sorted(set(lines[1][1::2])) == [0.0, 0.15] and sorted(set(lines[1][2::2])) == [0.0, 0.3]


def test_coco_sample_preview_draws_the_objects(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.create("Pets", "segmentation", ["cat"])
    source = make_coco_source(tmp_path)

    async def run():
        app = make_app(workspace, tmp_path)
        async with client_for(app) as client:
            response = await client.get("/api/v1/projects/pets/import/sample", params={"path": str(source), "file": "train/a.jpg"})
            assert response.status_code == 200 and response.headers["x-image-size"] == "100x50"
            with Image.open(io.BytesIO(response.content)) as preview:
                red, green, blue = preview.convert("RGB").getpixel((20, 10))  # on the dog's top edge
            assert red > 150 and green > 120 and blue < 100

    asyncio.run(run())
