import asyncio
from pathlib import Path

import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from PIL import Image

from test_dataset import make_classification_dataset
from yolo_workbench import classes
from yolo_workbench.dataset import Dataset, DatasetError, SchemaChanged
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.overlap import same_place
from yolo_workbench.web import create_app

BOX = {"class_id": 0, "points": [0.5, 0.5, 0.2, 0.2]}
TRIANGLE = [0.2, 0.2, 0.8, 0.2, 0.5, 0.8]


def make(tmp_path: Path, labels: dict[str, str], category: str = "detection") -> tuple[Dataset, dict[str, str]]:
    split = tmp_path / "train"
    split.mkdir()
    for name, content in labels.items():
        Image.new("RGB", (100, 80), "white").save(split / f"{name}.jpg")
        if content is not None:
            (split / f"{name}.txt").write_text(content)
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": ".", "train": "train", "names": {0: "zero", 1: "one"}}))
    dataset = Dataset(yaml_path, category)
    return dataset, {record.path.stem: record.id for record in dataset.images.values()}


def label(tmp_path: Path, name: str) -> str:
    return (tmp_path / "train" / f"{name}.txt").read_text()


def test_same_place_compares_boxes_and_filled_polygons():
    assert same_place("detection", [0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2])
    assert same_place("detection", [0.5, 0.5, 0.2, 0.2], [0.505, 0.5, 0.2, 0.2])  # IoU 0.95
    assert not same_place("detection", [0.5, 0.5, 0.2, 0.2], [0.55, 0.5, 0.2, 0.2])  # IoU 0.6
    assert not same_place("detection", [0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1])
    assert same_place("segmentation", TRIANGLE, [0.21, 0.2, 0.81, 0.2, 0.51, 0.8])
    # same bounds, other half of the box: the filled areas barely overlap
    assert not same_place("segmentation", TRIANGLE, [0.2, 0.8, 0.8, 0.8, 0.5, 0.2])


def test_propagate_copies_to_the_given_images_as_one_undoable_step(tmp_path):
    dataset, ids = make(tmp_path, {
        "src": "0 0.5 0.5 0.2 0.2\n",
        "empty": "",
        "unlabeled": None,
        "other": "1 0.1 0.1 0.05 0.05\n",
        "untouched": "",
    })
    result = dataset.propagate_annotations([ids["empty"], ids["unlabeled"], ids["other"]], [BOX])
    assert result == {"image_ids": [ids["empty"], ids["unlabeled"], ids["other"]], "added": 3, "unchanged": [], "skipped": []}
    assert label(tmp_path, "empty") == "0 0.5 0.5 0.2 0.2\n"
    assert label(tmp_path, "unlabeled") == "0 0.5 0.5 0.2 0.2\n"
    assert label(tmp_path, "other") == "1 0.1 0.1 0.05 0.05\n0 0.5 0.5 0.2 0.2\n"  # appended after what was there
    assert label(tmp_path, "untouched") == ""
    assert [a.points for a in dataset.images[ids["empty"]].annotations] == [BOX["points"]]

    undone = dataset.history("undo")
    assert sorted(undone["image_ids"]) == sorted([ids["empty"], ids["unlabeled"], ids["other"]])
    assert label(tmp_path, "empty") == ""
    assert not (tmp_path / "train" / "unlabeled.txt").exists()
    assert label(tmp_path, "other") == "1 0.1 0.1 0.05 0.05\n"
    dataset.history("redo")
    assert label(tmp_path, "other") == "1 0.1 0.1 0.05 0.05\n0 0.5 0.5 0.2 0.2\n"


def test_propagate_skips_shapes_an_image_already_has(tmp_path):
    dataset, ids = make(tmp_path, {
        "same": "0 0.5 0.5 0.2 0.2\n",
        "nearly": "0 0.505 0.5 0.2 0.2\n",  # what Validation would report as an overlap
        "moved": "0 0.55 0.5 0.2 0.2\n",  # IoU 0.6: another object
        "other_class": "1 0.5 0.5 0.2 0.2\n",
        "half": "0 0.5 0.5 0.2 0.2\n",
    })
    second = {"class_id": 1, "points": [0.2, 0.2, 0.1, 0.1]}
    targets = [ids["same"], ids["nearly"], ids["moved"], ids["other_class"]]
    result = dataset.propagate_annotations(targets, [BOX])
    assert result["image_ids"] == [ids["moved"], ids["other_class"]] and result["added"] == 2
    assert result["unchanged"] == [ids["same"], ids["nearly"]]
    assert label(tmp_path, "same") == "0 0.5 0.5 0.2 0.2\n"
    assert label(tmp_path, "nearly") == "0 0.505 0.5 0.2 0.2\n"
    assert label(tmp_path, "moved").count("\n") == 2
    assert label(tmp_path, "other_class").count("\n") == 2

    # several shapes: each one an image lacks
    result = dataset.propagate_annotations([ids["half"]], [BOX, second])
    assert result["added"] == 1
    assert label(tmp_path, "half") == "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n"
    # propagating again adds nothing
    result = dataset.propagate_annotations(targets + [ids["half"]], [BOX])
    assert result["image_ids"] == [] and result["added"] == 0 and len(result["unchanged"]) == 5


def test_propagate_reports_images_it_cannot_change(tmp_path):
    dataset, ids = make(tmp_path, {"good": "", "malformed": "0 0.5\n"})
    result = dataset.propagate_annotations([ids["good"], ids["malformed"], "missing", ids["good"]], [BOX])
    assert result["image_ids"] == [ids["good"]] and result["added"] == 1
    assert [item["image_id"] for item in result["skipped"]] == [ids["malformed"], "missing"]
    assert "malformed" in result["skipped"][0]["reason"] and result["skipped"][1]["reason"] == "Image not found"
    assert label(tmp_path, "good") == "0 0.5 0.5 0.2 0.2\n"  # once, though listed twice
    assert label(tmp_path, "malformed") == "0 0.5\n"


def test_propagate_refuses_invalid_shapes_and_outdated_class_lists(tmp_path):
    dataset, ids = make(tmp_path, {"a": ""})
    for shapes in ([{"class_id": 7, "points": [0.5, 0.5, 0.2, 0.2]}], [{"class_id": 0, "points": [0.5, 0.5, 1.2, 0.2]}], [{"class_id": 0, "points": TRIANGLE}], []):
        with pytest.raises(DatasetError):
            dataset.propagate_annotations([ids["a"]], shapes)
    generation = dataset.schema_generation
    classes.add_classes(dataset, ["two"])
    with pytest.raises(SchemaChanged):
        dataset.propagate_annotations([ids["a"]], [BOX], schema_generation=generation)
    assert label(tmp_path, "a") == ""


def test_propagate_polygons(tmp_path):
    dataset, ids = make(tmp_path, {"a": "", "b": f"0 {' '.join(map(str, TRIANGLE))}\n"}, "segmentation")
    result = dataset.propagate_annotations([ids["a"], ids["b"]], [{"class_id": 0, "points": TRIANGLE}])
    assert result["image_ids"] == [ids["a"]] and result["unchanged"] == [ids["b"]]
    assert [a.points for a in dataset.images[ids["a"]].annotations] == [TRIANGLE]


def test_propagate_sets_the_class_of_classification_images(tmp_path):
    dataset = make_classification_dataset(tmp_path)
    by_name = {record.path.name: record for record in dataset.images.values()}
    targets = [by_name["train_cat.png"].id, by_name["val_cat.png"].id, by_name["val_dog.png"].id]
    result = dataset.propagate_annotations(targets, [{"class_id": 1, "points": []}])
    assert result["image_ids"] == targets[:2] and result["added"] == 2
    assert result["unchanged"] == [by_name["val_dog.png"].id] and result["skipped"] == []
    assert (tmp_path / "train" / "dog" / "train_cat.png").exists()
    assert (tmp_path / "val" / "dog" / "val_cat.png").exists()
    dataset.history("undo")
    assert (tmp_path / "train" / "cat" / "train_cat.png").exists()
    assert (tmp_path / "val" / "cat" / "val_cat.png").exists()
    with pytest.raises(DatasetError):
        dataset.propagate_annotations(targets, [{"class_id": 1, "points": [0.5, 0.5, 0.2, 0.2]}])


def test_propagate_api(tmp_path):
    dataset, ids = make(tmp_path, {"a": "", "b": "0 0.5 0.5 0.2 0.2\n"})

    async def run():
        app = create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            body = {"image_ids": [ids["a"], ids["b"]], "annotations": [BOX], "schema_generation": dataset.schema_generation}
            response = await client.post("/api/v1/images/propagate", json=body)
            assert response.status_code == 200
            assert response.json() == {"image_ids": [ids["a"]], "added": 1, "unchanged": [ids["b"]], "skipped": []}
            assert dataset.history_state()["can_undo"]
            response = await client.post("/api/v1/images/propagate", json={**body, "annotations": [{"class_id": 5, "points": BOX["points"]}]})
            assert response.status_code == 400
            response = await client.post("/api/v1/images/propagate", json={**body, "annotations": []})
            assert response.status_code == 422
            response = await client.post("/api/v1/images/propagate", json={**body, "schema_generation": dataset.schema_generation - 1})
            assert response.status_code == 409

    asyncio.run(run())
