from pathlib import Path

import pytest
import yaml
from PIL import Image

from yolo_workbench.dataset import Dataset, DatasetError
from yolo_workbench.models import Annotation
from yolo_workbench.overlap import find_overlaps


def make(tmp_path: Path, text: str, category: str = "detection") -> tuple[Dataset, Path]:
    split = tmp_path / "train"
    split.mkdir()
    Image.new("RGB", (100, 80), "white").save(split / "a.jpg")
    label = split / "a.txt"
    label.write_text(text)
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": ".", "train": "train", "names": {0: "zero", 1: "one"}}))
    return Dataset(yaml_path, category), label


def overlaps(dataset: Dataset) -> list[dict]:
    return [issue for issue in dataset.issues() if issue["kind"] == "overlap"]


def annotations(*rows) -> list[Annotation]:
    return [Annotation(f"x:{index}", class_id, list(points)) for index, (class_id, points) in enumerate(rows, 1)]


def test_boxes_in_the_same_place_are_reported_on_the_later_one(tmp_path):
    dataset, _ = make(tmp_path, "0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.05 0.05\n0 0.505 0.5 0.21 0.2\n")
    [issue] = overlaps(dataset)
    assert issue["annotation_id"].endswith(":3") and issue["other_annotation_id"].endswith(":1")
    assert issue["same_class"] and issue["fixable"] and issue["iou"] > 0.9
    assert "line 1" in issue["message"]


def test_different_class_overlap_is_reported_but_not_fixable(tmp_path):
    dataset, _ = make(tmp_path, "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.19 0.2\n")
    [issue] = overlaps(dataset)
    assert not issue["same_class"] and not issue["fixable"]
    assert "another class" in issue["message"]


def test_exact_copies_stay_duplicates_and_neighbours_are_fine(tmp_path):
    dataset, _ = make(tmp_path, "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n0 0.6 0.5 0.2 0.2\n")
    kinds = sorted(issue["kind"] for issue in dataset.issues())
    assert kinds == ["duplicate"]  # the third box is shifted by half its width: IoU 1/3


def test_fix_deletes_the_later_box_and_undo_restores_it(tmp_path):
    dataset, label = make(tmp_path, "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.21 0.2\n")
    original = label.read_text()
    [issue] = overlaps(dataset)
    dataset.fix_issue(issue["id"])
    assert label.read_text() == "0 0.5 0.5 0.2 0.2\n"
    assert overlaps(dataset) == []
    dataset.history("undo")
    assert label.read_text() == original


def test_fix_refuses_different_classes(tmp_path):
    dataset, _ = make(tmp_path, "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.21\n")
    [issue] = overlaps(dataset)
    with pytest.raises(DatasetError):
        dataset.fix_issue(issue["id"])


def test_bulk_fix_keeps_the_end_of_a_chain(tmp_path):
    # 1~2 and 2~3 overlap, 1 and 3 do not: deleting 2 settles both, 3 stays
    dataset, label = make(tmp_path, "0 0.5 0.5 0.2 0.2\n0 0.51 0.5 0.2 0.2\n0 0.52 0.5 0.2 0.2\n")  # IoU 0.90, 0.90; 0.82
    pairs = [(issue["other_annotation_id"][-1], issue["annotation_id"][-1]) for issue in overlaps(dataset)]
    assert pairs == [("1", "2"), ("2", "3")]
    result = dataset.fix_issues_bulk("overlap")
    assert result["files"] == 1
    assert label.read_text().splitlines() == ["0 0.5 0.5 0.2 0.2", "0 0.52 0.5 0.2 0.2"]


def test_segmentation_compares_filled_polygons(tmp_path):
    square = "0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6"
    nudged = "0.21 0.2 0.6 0.21 0.6 0.6 0.2 0.6"
    dataset, _ = make(tmp_path, f"0 {square}\n0 {nudged}\n", "segmentation")
    [issue] = overlaps(dataset)
    assert issue["iou"] > 0.95 and "polygon" in issue["message"]


def test_polygons_sharing_a_box_but_not_an_area_do_not_overlap():
    # two triangles splitting one square along its diagonal: identical boxes, no shared area
    pairs = find_overlaps("segmentation", annotations(
        (0, [0.2, 0.2, 0.6, 0.2, 0.2, 0.6]),
        (0, [0.6, 0.2, 0.6, 0.6, 0.2, 0.6]),
    ))
    assert pairs == []


def test_threshold_and_zero_area_boxes():
    rows = annotations((0, [0.5, 0.5, 0.2, 0.2]), (0, [0.55, 0.5, 0.2, 0.2]), (0, [0.5, 0.5, 0.0, 0.2]))
    [(first, second, iou)] = find_overlaps("detection", rows, threshold=0.5)
    assert (first, second) == (0, 1) and iou == pytest.approx(0.6)
    assert find_overlaps("detection", rows) == []
