from pathlib import Path

import random

import pytest
import yaml
from PIL import Image

from yolo_workbench.dataset import Dataset, DatasetError
from yolo_workbench.models import Annotation
from yolo_workbench.overlap import RASTER_FLOOR, _polygon_iou, find_overlaps, max_ious


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


def test_max_ious_per_class_counts_exact_copies():
    rows = annotations(
        (0, [0.5, 0.5, 0.2, 0.2]), (1, [0.55, 0.5, 0.2, 0.2]),  # IoU 0.6
        (2, [0.1, 0.1, 0.1, 0.1]), (2, [0.1, 0.1, 0.1, 0.1]),  # an exact copy: IoU 1
        (3, [0.9, 0.9, 0.05, 0.05]),  # overlaps nothing
    )
    best = max_ious("detection", rows)
    assert best[None] == 1.0 and best[2] == 1.0
    assert best[0] == pytest.approx(0.6) and best[1] == pytest.approx(0.6)
    assert 3 not in best
    assert max_ious("detection", annotations((0, [0.2, 0.2, 0.1, 0.1]), (0, [0.25, 0.2, 0.1, 0.1]))) == {None: pytest.approx(1 / 3), 0: pytest.approx(1 / 3)}
    assert max_ious("detection", annotations((0, [0.2, 0.2, 0.1, 0.1]), (0, [0.7, 0.7, 0.1, 0.1]))) == {}


def test_max_ious_of_polygons_match_every_pair_rasterized():
    # the pruning skips pairs that cannot raise a class: the result must equal comparing every pair
    generator = random.Random(7)
    for _ in range(20):
        rows = []
        for _ in range(8):
            x, y, w, h = generator.uniform(0.1, 0.6), generator.uniform(0.1, 0.6), generator.uniform(0.05, 0.3), generator.uniform(0.05, 0.3)
            rows.append((generator.randrange(3), [x, y, x + w, y, x + w * generator.uniform(0.3, 1), y + h, x, y + h]))
        shapes = annotations(*rows)
        expected: dict = {}
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                a, b = shapes[i].points, shapes[j].points
                xs, ys = a[0::2] + b[0::2], a[1::2] + b[1::2]
                iou = _polygon_iou(a, b, (min(xs), min(ys), max(xs), max(ys)))
                if iou > RASTER_FLOOR:
                    for key in (None, shapes[i].class_id, shapes[j].class_id):
                        expected[key] = max(expected.get(key, 0.0), iou)
        assert max_ious("segmentation", shapes) == pytest.approx(expected)


def test_max_ious_of_polygons_sharing_only_a_box_or_an_edge_is_nothing():
    left, right = [0.2, 0.2, 0.4, 0.2, 0.4, 0.6, 0.2, 0.6], [0.4, 0.2, 0.6, 0.2, 0.6, 0.6, 0.4, 0.6]
    assert max_ious("segmentation", annotations((0, left), (1, right))) == {}
    assert max_ious("segmentation", annotations((0, [0.2, 0.2, 0.6, 0.2, 0.2, 0.6]), (1, [0.6, 0.2, 0.6, 0.6, 0.2, 0.6]))) == {}


def test_images_filtered_by_overlap_iou_follow_edits(tmp_path):
    dataset, _ = make(tmp_path, "0 0.5 0.5 0.2 0.2\n1 0.55 0.5 0.2 0.2\n")  # IoU 0.6 between classes 0 and 1
    split = tmp_path / "train"
    Image.new("RGB", (100, 80), "white").save(split / "b.jpg")
    (split / "b.txt").write_text("0 0.2 0.2 0.1 0.1\n0 0.25 0.2 0.1 0.1\n1 0.8 0.8 0.1 0.1\n")  # IoU 1/3, class 0 only
    Image.new("RGB", (100, 80), "white").save(split / "c.jpg")
    (split / "c.txt").write_text("0 0.2 0.2 0.1 0.1\n")
    dataset = Dataset(tmp_path / "dataset.yaml", "detection")

    def names(**kwargs):
        return [item["name"] for item in dataset.list_images(**kwargs)["items"]]

    assert names(iou_above=0) == ["a.jpg", "b.jpg"]
    assert names(iou_above=0.5) == ["a.jpg"]
    assert names(iou_above=0.6) == []  # strictly above
    assert names(iou_above=0.2, class_id=1) == ["a.jpg"]  # b's class 1 box overlaps nothing
    assert names(iou_above=0.2, class_id=0) == ["a.jpg", "b.jpg"]
    record = next(record for record in dataset.images.values() if record.path.name == "b.jpg")
    dataset.replace_annotations(record.id, [{"class_id": 0, "points": [0.2, 0.2, 0.1, 0.1]}, {"class_id": 0, "points": [0.2, 0.2, 0.1, 0.1]}])
    assert names(iou_above=0.5) == ["a.jpg", "b.jpg"]
    dataset.history("undo")
    assert names(iou_above=0.5) == ["a.jpg"]
    assert dataset.next_unreviewed(0, iou_above=0.5)["item"]["name"] == "a.jpg"
