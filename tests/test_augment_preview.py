import asyncio
import io
import json
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from yolo_workbench import augment_preview
from yolo_workbench.dataset import Dataset
from yolo_workbench.jobs import JobManager
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.training import TrainingManager, ultralytics_defaults, validate_args
from yolo_workbench.web import class_color, create_app

from test_training import client_for

DEFAULTS = ultralytics_defaults()[0]


def make_dataset(tmp_path: Path, category: str) -> Dataset:
    """Two train images and one val image with an off-centre object, so geometry changes are visible."""
    root = tmp_path / "ds"
    if category == "classification":
        for split in ("train", "val"):
            for name in ("cat", "dog"):
                (root / split / name).mkdir(parents=True)
                image = Image.new("RGB", (80, 60), "navy")
                ImageDraw.Draw(image).rectangle((5, 5, 30, 25), fill="yellow")
                image.save(root / split / name / f"{name}.jpg")
        return Dataset(root, category)
    for split, count in (("train", 2), ("val", 1)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(count):
            image = Image.new("RGB", (80, 60), "navy")
            ImageDraw.Draw(image).rectangle((8, 6, 32, 30), fill="yellow")
            image.save(root / "images" / split / f"{split}{index}.png")
            label = "0 0.25 0.3 0.3 0.4" if category == "detection" else "0 0.1 0.1 0.4 0.1 0.4 0.5 0.1 0.5"
            (root / "labels" / split / f"{split}{index}.txt").write_text(label + "\n")
    (root / "data.yaml").write_text(yaml.safe_dump({"train": "images/train", "val": "images/val", "names": {0: "box"}}))
    return Dataset(root / "data.yaml", category)


def pixels(preview) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(preview.jpeg)).convert("RGB"), dtype=np.int16)


def render(dataset, key, side, **values):
    return augment_preview.render_augmentation(dataset, key, side, values, DEFAULTS, 0, 64, 64, True, class_color)


def test_range_ends_differ_and_match_their_direction(tmp_path):
    dataset = make_dataset(tmp_path, "detection")
    original = pixels(augment_preview.render_original(dataset, 0, 64, 64, True, class_color))
    low, high = pixels(render(dataset, "degrees", "min", degrees=40)), pixels(render(dataset, "degrees", "max", degrees=40))
    assert np.abs(low - high).mean() > 5 and np.abs(low - original).mean() > 5
    darker, brighter = pixels(render(dataset, "hsv_v", "min", hsv_v=0.8)), pixels(render(dataset, "hsv_v", "max", hsv_v=0.8))
    assert darker.mean() < original.mean() < brighter.mean()
    # zero strength leaves the image as the identity letterbox
    assert np.abs(pixels(render(dataset, "translate", "max", translate=0)) - original).mean() < 2


def test_labels_dropped_at_the_extremes_are_reported(tmp_path):
    dataset = make_dataset(tmp_path, "detection")
    assert render(dataset, "scale", "min", scale=0.99).note == "1 of 1 labels dropped"
    assert render(dataset, "scale", "max", scale=0.99).note == ""


def test_chance_augmentations_and_pipeline_samples_render(tmp_path):
    dataset = make_dataset(tmp_path, "segmentation")
    for key in sorted(augment_preview.preview_keys("segmentation")):
        assert render(dataset, key, "max", copy_paste=0.5, mixup=0.5).jpeg.startswith(b"\xff\xd8")
    assert "pasted" in render(dataset, "copy_paste", "max", copy_paste=1.0).note
    flipped = pixels(render(dataset, "fliplr", "max"))
    assert np.abs(flipped - pixels(augment_preview.render_original(dataset, 0, 64, 64, True, class_color))[:, ::-1]).mean() < 6
    values = {"degrees": 10, "mixup": 0.5, "copy_paste": 0.5}
    first = augment_preview.render_pipeline(dataset, values, DEFAULTS, 0, 3, 64, 64, True, class_color)
    again = augment_preview.render_pipeline(dataset, values, DEFAULTS, 0, 3, 64, 64, True, class_color)
    assert first.jpeg == again.jpeg  # seeded


def test_renders_leave_ultralytics_and_global_random_state_alone(tmp_path):
    from ultralytics.data import augment

    dataset = make_dataset(tmp_path, "detection")
    random.seed(7)
    np.random.seed(7)
    expected = (random.random(), np.random.random())
    random.seed(7)
    np.random.seed(7)
    render(dataset, "hsv_h", "max", hsv_h=0.5)
    augment_preview.render_pipeline(dataset, {}, DEFAULTS, 0, 1, 64, 64, True, class_color)
    assert (random.random(), np.random.random()) == expected
    assert augment.random is random and augment.np is np


def test_classification_previews(tmp_path):
    dataset = make_dataset(tmp_path, "classification")
    for key in sorted(augment_preview.preview_keys("classification")):
        for side in ("min", "max"):
            assert render(dataset, key, side, hsv_h=0.9, erasing=1.0).jpeg.startswith(b"\xff\xd8")
    assert render(dataset, "auto_augment", "min", auto_augment=None).note == "Off"
    assert augment_preview.render_pipeline(dataset, {"auto_augment": None}, DEFAULTS, 0, 0, 64, 64, False, class_color).jpeg
    assert "hsv_h" in validate_args("classification", "model: yolo11n-cls.pt\nhsv_h: 0.6\n")["errors"][0]["message"]
    assert not validate_args("detection", "model: yolo11n.pt\nhsv_h: 0.6\n")["errors"]


def test_augment_preview_api(tmp_path):
    dataset = make_dataset(tmp_path, "detection")
    manager = TrainingManager.for_dataset(dataset, jobs=JobManager(synchronous=True), weights_dir=tmp_path / "weights")
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media"), training=manager)
    args = json.dumps({"degrees": 15, "imgsz": 64})

    async def run():
        async with client_for(app) as client:
            response = await client.get("/api/v1/training/augment/preview", params={"key": "degrees", "side": "min", "args": args, "size": 64})
            assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
            response = await client.get("/api/v1/training/augment/original", params={"sample": 3, "imgsz": 64})
            # samples are ordered by record id, a hash of the (temporary) path: either train image can be second
            expected = augment_preview.sample_pool(dataset, 3)[0][0].path.name
            assert response.status_code == 200 and response.headers["x-preview-note"] == f"{expected}%20%C2%B7%202%20of%202"
            response = await client.get("/api/v1/training/augment/sample", params={"args": args, "seed": 2})
            assert response.status_code == 200
            assert (await client.get("/api/v1/training/augment/preview", params={"key": "erasing", "side": "min"})).status_code == 400
            assert (await client.get("/api/v1/training/augment/preview", params={"key": "degrees", "side": "min", "args": "[1]"})).status_code == 400
            assert (await client.get("/api/v1/training/augment/sample", params={"args": json.dumps({"mosaic": 2})})).status_code == 400

    asyncio.run(run())
