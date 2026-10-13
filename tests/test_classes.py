from pathlib import Path

import pytest
import yaml
from PIL import Image

from yolo_workbench import classes
from yolo_workbench.dataset import Dataset, DatasetError, WriteConflict


def make_detection(tmp_path: Path, names=None, yaml_text: str | None = None) -> Dataset:
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)
    for name, text in {"a": "0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n", "b": "2 0.500000 0.5 0.3 0.3\n", "c": "1 0.1 0.1 0.1 0.1\n"}.items():
        Image.new("RGB", (40, 30), "white").save(tmp_path / "images" / "train" / f"{name}.jpg")
        (tmp_path / "labels" / "train" / f"{name}.txt").write_text(text)
    yaml_path = tmp_path / "data.yaml"
    if yaml_text is None:
        yaml_text = yaml.safe_dump({"train": "images/train", "val": "images/val", "names": names or {0: "cat", 1: "dog", 2: "bird"}}, sort_keys=False)
    yaml_path.write_text(yaml_text)
    return Dataset(yaml_path, "detection")


def labels(tmp_path: Path) -> dict[str, str]:
    return {path.stem: path.read_text() for path in sorted((tmp_path / "labels" / "train").glob("*.txt"))}


def test_remove_class_delete_renumbers_and_undo_restores_bytes(tmp_path):
    dataset = make_detection(tmp_path)
    before_labels, before_yaml = labels(tmp_path), (tmp_path / "data.yaml").read_text()
    result = classes.remove_class(dataset, 0, "delete")
    assert result["mapping"] == {"0": None, "1": 0, "2": 1}
    assert dataset.names == {0: "dog", 1: "bird"}
    # geometry text is untouched, only the class token changes
    assert labels(tmp_path) == {"a": "0 0.25 0.25 0.1 0.1\n", "b": "1 0.500000 0.5 0.3 0.3\n", "c": "0 0.1 0.1 0.1 0.1\n"}
    assert yaml.safe_load((tmp_path / "data.yaml").read_text())["names"] == {0: "dog", 1: "bird"}
    assert dataset.list_objects(0)["total"] == 2

    history = dataset.history("undo")
    assert history["schema_changed"]
    assert labels(tmp_path) == before_labels
    assert (tmp_path / "data.yaml").read_text() == before_yaml
    assert dataset.names == {0: "cat", 1: "dog", 2: "bird"}
    dataset.history("redo")
    assert dataset.names == {0: "dog", 1: "bird"}


def test_merge_class_into_target(tmp_path):
    dataset = make_detection(tmp_path)
    classes.remove_class(dataset, 2, "merge", target_id=0)
    assert dataset.names == {0: "cat", 1: "dog"}
    assert labels(tmp_path)["b"] == "0 0.500000 0.5 0.3 0.3\n"
    classes.merge_classes(dataset, [1], 0)
    assert dataset.names == {0: "cat"}
    assert set(labels(tmp_path).values()) == {"0 0.5 0.5 0.2 0.2\n0 0.25 0.25 0.1 0.1\n", "0 0.500000 0.5 0.3 0.3\n", "0 0.1 0.1 0.1 0.1\n"}


def test_add_rename_reorder_keep_list_style_and_comments(tmp_path):
    text = "# my dataset\npath: .\ntrain: images/train  # training images\nval: images/val\nnames:\n  - cat\n  - dog\n  - bird\n# trailing note\nnc: 3\n"
    dataset = make_detection(tmp_path, yaml_text=text)
    classes.add_classes(dataset, ["fish"])
    content = (tmp_path / "data.yaml").read_text()
    assert content.startswith("# my dataset\npath: .\ntrain: images/train  # training images\n")
    assert "# trailing note\nnc: 3\n" in content
    assert yaml.safe_load(content)["names"] == ["cat", "dog", "bird", "fish"]
    classes.rename_class(dataset, 3, "shark")
    assert dataset.names[3] == "shark"
    classes.reorder_classes(dataset, [2, 0, 1, 3])
    assert dataset.names == {0: "bird", 1: "cat", 2: "dog", 3: "shark"}
    assert labels(tmp_path)["a"] == "1 0.5 0.5 0.2 0.2\n2 0.25 0.25 0.1 0.1\n"
    with pytest.raises(DatasetError):
        classes.add_classes(dataset, ["Cat"])


def test_schema_change_aborts_on_external_edit(tmp_path):
    dataset = make_detection(tmp_path)
    label = tmp_path / "labels" / "train" / "b.txt"
    label.write_text("2 0.4 0.4 0.1 0.1\n")
    import os
    os.utime(label, ns=(1, 1))
    with pytest.raises(WriteConflict):
        classes.remove_class(dataset, 0, "delete")
    # nothing was written
    assert labels(tmp_path)["a"].startswith("0 ")
    assert yaml.safe_load((tmp_path / "data.yaml").read_text())["names"][0] == "cat"


def test_schema_listeners_receive_mapping(tmp_path):
    dataset = make_detection(tmp_path)
    seen = []
    dataset.add_schema_listener(seen.append)
    classes.remove_class(dataset, 1, "delete")
    dataset.history("undo")
    assert seen == [{0: 0, 1: None, 2: 1}, {0: 0, 1: 2}]


def test_write_names_falls_back_and_appends():
    assert classes.write_names("train: x\n", {0: "a"}) == "train: x\nnames:\n  0: a\n"
    assert yaml.safe_load(classes.write_names("names: [a, b]\ntrain: x\n", {0: "a"}, "list")) == {"names": ["a"], "train": "x"}


def make_classification(tmp_path: Path) -> Dataset:
    for split, cls, name in [("train", "cat", "1.jpg"), ("train", "dog", "2.jpg"), ("val", "cat", "3.jpg"), ("val", "dog", "4.jpg")]:
        (tmp_path / split / cls).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), "white").save(tmp_path / split / cls / name)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({"train": "train", "val": "val", "names": ["cat", "dog"]}))
    return Dataset(yaml_path, "classification")


def test_classification_add_rename_delete_merge_with_undo(tmp_path):
    dataset = make_classification(tmp_path)
    classes.add_classes(dataset, ["ant"])
    assert dataset.names == {0: "ant", 1: "cat", 2: "dog"}
    assert (tmp_path / "train" / "ant").is_dir() and (tmp_path / "val" / "ant").is_dir()
    assert yaml.safe_load((tmp_path / "data.yaml").read_text())["names"] == ["ant", "cat", "dog"]

    classes.rename_class(dataset, 1, "zebra")
    assert dataset.names == {0: "ant", 1: "dog", 2: "zebra"}
    assert (tmp_path / "train" / "zebra" / "1.jpg").exists()
    assert sorted(item["name"] for item in dataset.list_images(class_id=2)["items"]) == ["1.jpg", "3.jpg"]

    classes.remove_class(dataset, 2, "merge", target_id=1)
    assert dataset.names == {0: "ant", 1: "dog"}
    assert (tmp_path / "train" / "dog" / "1.jpg").exists() and not (tmp_path / "train" / "zebra").exists()
    assert dataset.list_images(class_id=1)["total"] == 4

    dataset.history("undo")
    assert dataset.names == {0: "ant", 1: "dog", 2: "zebra"}
    assert (tmp_path / "train" / "zebra" / "1.jpg").exists()

    classes.remove_class(dataset, 2, "delete")
    assert dataset.names == {0: "ant", 1: "dog"}
    assert len(dataset.images) == 2
    dataset.history("undo")
    assert len(dataset.images) == 4
    assert (tmp_path / "val" / "zebra" / "3.jpg").exists()
    with pytest.raises(DatasetError):
        classes.reorder_classes(dataset, [1, 0, 2])


def test_delete_and_move_images_are_undoable(tmp_path):
    dataset = make_detection(tmp_path)
    ids = {record.path.stem: record.id for record in dataset.images.values()}
    result = dataset.move_images([ids["a"]], "val")
    assert result["moved"] == 1
    assert (tmp_path / "images" / "val" / "a.jpg").exists() and (tmp_path / "labels" / "val" / "a.txt").exists()
    assert dataset.list_images(split="val")["total"] == 1
    assert dataset.images[ids["a"]].annotations[0].class_id == 0

    dataset.delete_images([ids["b"], ids["a"]])
    assert len(dataset.images) == 1
    assert not (tmp_path / "labels" / "train" / "b.txt").exists()
    assert dataset.list_objects(2)["total"] == 0

    dataset.history("undo")
    assert len(dataset.images) == 3
    assert dataset.list_objects(2)["total"] == 1
    assert (tmp_path / "labels" / "val" / "a.txt").exists()
    dataset.history("undo")
    assert (tmp_path / "images" / "train" / "a.jpg").exists()
    assert dataset.list_images(split="val")["total"] == 0


def test_empty_dataset_and_rescan(tmp_path):
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
    (tmp_path / "data.yaml").write_text("train: images/train\nval: images/val\nnames: {}\n")
    with pytest.raises(DatasetError):
        Dataset(tmp_path / "data.yaml", "detection")
    dataset = Dataset(tmp_path / "data.yaml", "detection", allow_empty=True)
    assert dataset.images == {} and dataset.names == {}
    assert dataset.statistics()["summary"]["images"] == 0
    classes.add_classes(dataset, ["car", "bus"])
    Image.new("RGB", (10, 10)).save(tmp_path / "images" / "val" / "x.png")
    (tmp_path / "labels" / "val" / "x.txt").write_text("1 0.5 0.5 0.5 0.5\n")
    assert dataset.rescan()["added"] == 1
    assert dataset.list_images(class_id=1)["total"] == 1
    assert dataset.images[next(iter(dataset.images))].width == 10
    (tmp_path / "images" / "val" / "x.png").unlink()
    assert dataset.rescan()["removed"] == 1
    assert dataset.list_images()["total"] == 0
