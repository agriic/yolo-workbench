import math
from pathlib import Path

import pytest
import yaml
from PIL import Image

from yolo_workbench.dataset import Dataset, DatasetError, WriteConflict


def make_dataset(tmp_path: Path, category="detection") -> tuple[Dataset, Path]:
    split = tmp_path / "train"
    split.mkdir()
    Image.new("RGB", (100, 80), "white").save(split / "sample.jpg")
    label = split / "sample.txt"
    label.write_text("1 0.5 0.5 0.2 0.4\n" if category == "detection" else "1 0.2 0.2 0.8 0.2 0.5 0.8\n")
    config = {"path": ".", "train": "train", "names": {0: "zero", 1: "one"}}
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(config))
    return Dataset(yaml_path, category), label


def test_loads_colocated_detection_dataset(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    assert record.width == 100
    assert record.annotations[0].points == [0.5, 0.5, 0.2, 0.4]
    assert dataset.list_images(class_id=1)["total"] == 1


def test_annotation_indexes_and_cached_aggregates_refresh_after_write(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))

    assert dataset.metadata()["issue_count"] == 0
    assert dataset.list_objects(1)["total"] == 1
    dataset.replace_annotations(record.id, [{"class_id": 0, "points": [0.4, 0.4, 0.1, 0.1]}])

    assert dataset.list_images(class_id=1)["total"] == 0
    assert dataset.list_images(class_id=0)["total"] == 1
    assert dataset.list_objects(1)["total"] == 0
    assert dataset.list_objects(0)["items"][0]["image_id"] == record.id
    assert dataset.metadata()["issue_count"] == 0


def test_statistics_cover_balance_geometry_splits_cooccurrence_and_refresh(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))

    stats = dataset.statistics()

    assert stats["summary"] == {
        "images": 1,
        "annotated_images": 1,
        "unlabeled_images": 0,
        "annotations": 1,
        "average_annotations": 1.0,
    }
    assert next(item for item in stats["class_balance"] if item["class_id"] == 1)["annotations"] == 1
    assert stats["annotations_per_image"]["median"] == 1
    assert stats["box_size"]["median"] == pytest.approx(0.08)
    assert stats["aspect_ratio"]["median"] == pytest.approx(0.5)
    assert stats["split_comparison"][0]["class_annotations"]["1"] == 1
    assert stats["cooccurrence"]["matrix"] == [[0, 0], [0, 1]]
    assert dataset.statistics() is stats

    dataset.replace_annotations(record.id, [
        {"class_id": 0, "points": [0.4, 0.4, 0.1, 0.1]},
        {"class_id": 1, "points": [0.6, 0.6, 0.2, 0.2]},
    ])
    refreshed = dataset.statistics()

    assert refreshed is not stats
    assert refreshed["summary"]["annotations"] == 2
    assert refreshed["cooccurrence"]["matrix"] == [[1, 1], [1, 1]]
    assert refreshed["source_generation"] == 1


def test_statistics_flag_clear_geometry_outlier(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    for index in range(9):
        Image.new("RGB", (40, 40), "white").save(split / f"{index}.jpg")
        size = 0.9 if index == 8 else 0.1
        (split / f"{index}.txt").write_text(f"0 0.5 0.5 {size} {size}\n")
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": ".", "train": "train", "names": ["object"]}))

    outliers = Dataset(yaml_path, "detection").statistics()["outliers"]

    assert any(item["image_name"] == "8.jpg" and item["kind"] == "box_size" for item in outliers)


def test_writes_backup_and_supports_history(tmp_path):
    dataset, label = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    original = label.read_text()
    dataset.replace_annotations(record.id, [{"class_id": 0, "points": [0.4, 0.4, 0.1, 0.1]}])
    assert label.read_text().startswith("0 ")
    backups = list((tmp_path / ".yolo-workbench" / "backups").rglob("sample.txt"))
    assert len(backups) == 1
    assert backups[0].read_text() == original
    dataset.history("undo")
    assert label.read_text() == original
    dataset.history("redo")
    assert label.read_text().startswith("0 ")


def test_history_restores_missing_label_file_state(tmp_path):
    dataset, label = make_dataset(tmp_path)
    label.unlink()
    dataset = Dataset(dataset.yaml_path, "detection")
    record = next(iter(dataset.images.values()))

    dataset.replace_annotations(record.id, [])
    assert label.exists()

    dataset.history("undo")
    assert not label.exists()
    assert any(issue["kind"] == "missing_label" for issue in record.issues)

    dataset.history("redo")
    assert label.exists()
    assert label.read_text() == ""


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite_annotation_geometry(tmp_path, value):
    dataset, label = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    original = label.read_text()

    with pytest.raises(DatasetError, match="Invalid annotation geometry"):
        dataset.replace_annotations(record.id, [{"class_id": 1, "points": [0.5, 0.5, value, 0.2]}])

    assert label.read_text() == original


def test_detects_external_write_conflict(tmp_path):
    dataset, label = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    label.write_text("0 0.1 0.1 0.1 0.1\n")
    with pytest.raises(WriteConflict):
        dataset.replace_annotations(record.id, [])


def test_reports_and_fixes_out_of_range(tmp_path):
    dataset, label = make_dataset(tmp_path)
    label.write_text("1 1.01 0.5 0.2 0.4\n")
    dataset = Dataset(dataset.yaml_path, "detection")
    issue = next(issue for issue in dataset.issues() if issue["kind"] == "out_of_range")
    dataset.fix_issue(issue["id"])
    assert label.read_text().startswith("1 1 ")


def test_segmentation_requires_three_points(tmp_path):
    dataset, _ = make_dataset(tmp_path, "segmentation")
    record = next(iter(dataset.images.values()))
    with pytest.raises(DatasetError):
        dataset.replace_annotations(record.id, [{"class_id": 1, "points": [0.1, 0.1, 0.2, 0.2]}])


def test_parallel_images_and_labels_layout(tmp_path):
    images = tmp_path / "images" / "train"
    labels = tmp_path / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (20, 20)).save(images / "a.png")
    (labels / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": ".", "train": "images/train", "names": ["object"]}))
    dataset = Dataset(yaml_path, "detection")
    assert next(iter(dataset.images.values())).label_path == labels / "a.txt"


def test_file_list_entries_with_stale_prefix_relocate_under_root(tmp_path):
    images = tmp_path / "images" / "train" / "cam0"
    labels = tmp_path / "labels" / "train" / "cam0"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (20, 20)).save(images / "a.png")
    (labels / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (tmp_path / "train.txt").write_text("data/images/train/cam0/a.png\n")
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": ".", "train": "train.txt", "names": ["object"]}))
    dataset = Dataset(yaml_path, "detection")
    record = next(iter(dataset.images.values()))
    assert record.path == images / "a.png"
    assert record.label_path == labels / "a.txt"
    assert not record.issues


def test_reports_nested_orphan_labels_recursively(tmp_path):
    images = tmp_path / "images" / "train"
    labels = tmp_path / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (20, 20)).save(images / "a.png")
    (labels / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    orphan = labels / "nested" / "orphan.txt"
    orphan.parent.mkdir()
    orphan.write_text("0 0.5 0.5 0.5 0.5\n")
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": ".", "train": "images/train", "names": ["object"]}))

    dataset = Dataset(yaml_path, "detection")

    assert dataset.orphan_labels == [orphan.resolve()]
    assert any(issue["kind"] == "orphan_label" and issue["image_name"] == "orphan.txt" for issue in dataset.issues())
