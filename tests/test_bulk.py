from pathlib import Path

import pytest
import yaml
from PIL import Image

from yolo_workbench.dataset import Dataset, WriteConflict


def make_multi_dataset(tmp_path: Path, labels: dict[str, str]) -> Dataset:
    split = tmp_path / "train"
    split.mkdir()
    for name, content in labels.items():
        Image.new("RGB", (100, 80), "white").save(split / f"{name}.jpg")
        (split / f"{name}.txt").write_text(content)
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": ".", "train": "train", "names": {0: "zero", 1: "one"}}))
    return Dataset(yaml_path, "detection")


def record_by_name(dataset: Dataset, name: str):
    return next(r for r in dataset.images.values() if r.path.stem == name)


def test_bulk_fix_duplicates_across_files_single_undo(tmp_path):
    dataset = make_multi_dataset(tmp_path, {
        "a": "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n",
        "b": "1 0.4 0.4 0.1 0.1\n1 0.4 0.4 0.1 0.1\n",
    })
    originals = {name: (tmp_path / "train" / f"{name}.txt").read_text() for name in ("a", "b")}
    result = dataset.fix_issues_bulk("duplicate")
    assert result == {"fixed": 2, "files": 2, "skipped": []}
    for name in ("a", "b"):
        assert len((tmp_path / "train" / f"{name}.txt").read_text().splitlines()) == 1
    outcome = dataset.history("undo")
    assert len(outcome["image_ids"]) == 2
    for name, original in originals.items():
        assert (tmp_path / "train" / f"{name}.txt").read_text() == original
    dataset.history("redo")
    assert len((tmp_path / "train" / "a.txt").read_text().splitlines()) == 1


def test_bulk_fix_skips_invalid_records(tmp_path):
    dataset = make_multi_dataset(tmp_path, {
        "good": "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n",
        "bad": "7 0.5 0.5 0.2 0.2\n7 0.5 0.5 0.2 0.2\n",  # unknown class blocks validation
    })
    result = dataset.fix_issues_bulk("duplicate")
    assert result["fixed"] == 1 and result["files"] == 1
    assert len(result["skipped"]) == 1
    assert "Unknown class" in result["skipped"][0]["reason"]
    assert (tmp_path / "train" / "bad.txt").read_text().count("\n") == 2


def test_bulk_conflict_aborts_whole_batch(tmp_path):
    dataset = make_multi_dataset(tmp_path, {
        "a": "0 0.5 0.5 0.2 0.2\n",
        "b": "1 0.4 0.4 0.1 0.1\n",
    })
    (tmp_path / "train" / "b.txt").write_text("0 0.1 0.1 0.05 0.05\n")
    operations = [
        {"image_id": record_by_name(dataset, name).id, "annotation_id": record_by_name(dataset, name).annotations[0].id, "action": "delete", "class_id": None}
        for name in ("a", "b")
    ]
    with pytest.raises(WriteConflict):
        dataset.bulk_edit_objects(operations)
    assert (tmp_path / "train" / "a.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"


def test_bulk_object_relabel_and_delete(tmp_path):
    dataset = make_multi_dataset(tmp_path, {
        "a": "0 0.5 0.5 0.2 0.2\n",
        "b": "1 0.4 0.4 0.1 0.1\n0 0.2 0.2 0.1 0.1\n",
    })
    a, b = record_by_name(dataset, "a"), record_by_name(dataset, "b")
    result = dataset.bulk_edit_objects([
        {"image_id": a.id, "annotation_id": a.annotations[0].id, "action": "relabel", "class_id": 1},
        {"image_id": b.id, "annotation_id": b.annotations[0].id, "action": "delete", "class_id": None},
        {"image_id": b.id, "annotation_id": "missing", "action": "delete", "class_id": None},
    ])
    assert result["applied"] == 2 and result["files"] == 2
    assert result["skipped"][0]["reason"] == "Annotation not found"
    assert (tmp_path / "train" / "a.txt").read_text().startswith("1 ")
    assert (tmp_path / "train" / "b.txt").read_text() == "0 0.2 0.2 0.1 0.1\n"
    dataset.history("undo")
    assert (tmp_path / "train" / "a.txt").read_text().startswith("0 ")
    assert len((tmp_path / "train" / "b.txt").read_text().splitlines()) == 2


def test_bulk_fix_missing_labels(tmp_path):
    dataset = make_multi_dataset(tmp_path, {"a": "0 0.5 0.5 0.2 0.2\n"})
    label = tmp_path / "train" / "a.txt"
    label.unlink()
    dataset = Dataset(dataset.yaml_path, "detection")
    result = dataset.fix_issues_bulk("missing_label")
    assert result["fixed"] == 1
    assert label.read_text() == ""
