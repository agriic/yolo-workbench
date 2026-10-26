import asyncio
import io
import zipfile
from pathlib import Path

import yaml
from httpx import ASGITransport, AsyncClient
from PIL import Image

from test_dataset import make_classification_dataset
from yolo_workbench import download
from yolo_workbench.dataset import Dataset
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.web import create_app
from yolo_workbench.yamledit import remove_key


def make_parallel(root: Path, yaml_text: str | None = None) -> Dataset:
    for split, names in {"train": ["a.jpg", "b.png"], "val": ["c.jpg"]}.items():
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for name in names:
            Image.new("RGB", (40, 30), "white").save(root / "images" / split / name)
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "labels" / "val" / "c.txt").write_text("1 0.4 0.4 0.1 0.1\n")
    (root / "data.yaml").write_text(yaml_text or "# my dataset\npath: .\ntrain: images/train\nval: images/val\nnames:\n  0: cat\n  1: dog\n")
    return Dataset(root / "data.yaml", "detection")


def unzip(entries) -> zipfile.ZipFile:
    archive = zipfile.ZipFile(io.BytesIO(b"".join(download.stream(entries))))
    assert archive.testzip() is None
    return archive


def test_labels_only_keeps_the_layout_and_drops_path(tmp_path):
    dataset = make_parallel(tmp_path / "ds")
    archive = unzip(download.plan(dataset, "labels"))
    assert sorted(archive.namelist()) == ["data.yaml", "labels/train/a.txt", "labels/val/c.txt"]
    assert archive.read("labels/val/c.txt") == b"1 0.4 0.4 0.1 0.1\n"
    data = archive.read("data.yaml").decode()
    assert data.startswith("# my dataset\n")  # comments kept
    assert yaml.safe_load(data) == {"train": "images/train", "val": "images/val", "names": {0: "cat", 1: "dog"}}


def test_full_download_has_images_and_labels(tmp_path):
    dataset = make_parallel(tmp_path / "ds")
    archive = unzip(download.plan(dataset, "full"))
    assert sorted(archive.namelist()) == ["data.yaml", "images/train/a.jpg", "images/train/b.png", "images/val/c.jpg", "labels/train/a.txt", "labels/val/c.txt"]
    assert archive.read("images/train/b.png") == (tmp_path / "ds" / "images" / "train" / "b.png").read_bytes()
    assert archive.getinfo("images/train/a.jpg").compress_type == zipfile.ZIP_STORED
    assert archive.getinfo("labels/train/a.txt").compress_type == zipfile.ZIP_DEFLATED


def test_a_file_gone_since_the_plan_is_left_out(tmp_path):
    dataset = make_parallel(tmp_path / "ds")
    entries = download.plan(dataset, "full")
    (tmp_path / "ds" / "images" / "train" / "b.png").unlink()
    assert "images/train/b.png" not in unzip(entries).namelist()


def test_outside_files_and_lists_are_rewritten(tmp_path):
    root = tmp_path / "ds"
    outside = tmp_path / "elsewhere" / "images"
    outside.mkdir(parents=True)
    Image.new("RGB", (40, 30), "white").save(outside / "x.jpg")
    (tmp_path / "elsewhere" / "labels").mkdir()
    (tmp_path / "elsewhere" / "labels" / "x.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    root.mkdir()
    (root / "images" / "val").mkdir(parents=True)
    Image.new("RGB", (40, 30), "white").save(root / "images" / "val" / "c.jpg")
    (root / "val.txt").write_text(f"{root / 'images' / 'val' / 'c.jpg'}\n")
    (root / "data.yaml").write_text(f"train: {outside}\nval: val.txt\nnames: [cat]\n")
    dataset = Dataset(root / "data.yaml", "detection")
    archive = unzip(download.plan(dataset, "full"))
    names = set(archive.namelist())
    assert {"external/images/train/x.jpg", "external/labels/train/x.txt", "images/val/c.jpg"} <= names
    data = yaml.safe_load(archive.read("data.yaml"))
    assert data["train"] == "external/images/train"
    list_name = data["val"]
    assert archive.read(list_name).decode() == "./images/val/c.jpg\n"


def test_classification_downloads_class_folders_only_with_images(tmp_path):
    dataset = make_classification_dataset(tmp_path)
    archive = unzip(download.plan(dataset, "full"))
    assert sorted(archive.namelist()) == ["train/cat/train_cat.png", "train/dog/train_dog.png", "val/cat/val_cat.png", "val/dog/val_dog.png"]


def test_remove_key_keeps_everything_else():
    text = "# head\npath: /data/x  # where\ntrain: images/train\nnames:\n  0: a\n"
    assert remove_key(text, "path") == "# head\ntrain: images/train\nnames:\n  0: a\n"
    assert remove_key(text, "missing") == text


def test_download_endpoint(tmp_path):
    dataset = make_parallel(tmp_path / "ds")
    (tmp_path / "cls").mkdir()
    classification = make_classification_dataset(tmp_path / "cls")

    async def run():
        async with AsyncClient(transport=ASGITransport(app=create_app(dataset, media_cache=MediaCache(tmp_path / "media"))), base_url="http://test") as client:
            response = await client.get("/api/v1/download", params={"kind": "labels"})
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/zip"
            assert response.headers["content-disposition"] == "attachment; filename*=UTF-8''ds-labels.zip"
            assert "labels/train/a.txt" in zipfile.ZipFile(io.BytesIO(response.content)).namelist()
            full = await client.get("/api/v1/download")
            assert "images/train/a.jpg" in zipfile.ZipFile(io.BytesIO(full.content)).namelist()
            assert (await client.get("/api/v1/download", params={"kind": "other"})).status_code == 422
        async with AsyncClient(transport=ASGITransport(app=create_app(classification, media_cache=MediaCache(tmp_path / "media2"))), base_url="http://test") as client:
            response = await client.get("/api/v1/download", params={"kind": "labels"})
            assert response.status_code == 400

    asyncio.run(run())
