from test_dataset import make_dataset
from yolo_workbench.dataset import Dataset, object_key
from yolo_workbench.embeddings import EmbeddingsManager, _normalize_3d_points


def test_normalize_3d_points_normalizes_all_axes():
    points = _normalize_3d_points([(10, -2, 100), (20, 2, 300)])

    assert points == [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]


def test_normalize_3d_points_pads_missing_and_constant_axes():
    points = _normalize_3d_points([(4,), (4,)])

    assert points == [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]


def computed(tmp_path):
    """A dataset of one image with three boxes and its embeddings as compute_gt_viz leaves them (without FiftyOne)."""
    dataset, label = make_dataset(tmp_path)
    label.write_text("0 0.2 0.2 0.1 0.1\n1 0.5 0.5 0.2 0.4\n0 0.8 0.8 0.1 0.1\n")
    dataset = Dataset(dataset.yaml_path, "detection")
    record = next(iter(dataset.images.values()))
    embeddings = EmbeddingsManager(dataset)
    embeddings.items = [
        {"x": index / 2, "y": 0.5, "z": 0.5, "image_id": record.id, "shape": tuple(annotation.points)}
        for index, annotation in enumerate(record.annotations)
    ]
    embeddings.status = "ready"
    return dataset, record, embeddings


def shown(embeddings):
    return [(item["uid"], item["annotation_id"].split(":")[-1], item["class_id"]) for item in embeddings.payload()["items"]]


def test_embeddings_follow_deletes_and_relabels_until_undone(tmp_path):
    dataset, record, embeddings = computed(tmp_path)
    assert shown(embeddings) == [(0, "1", 0), (1, "2", 1), (2, "3", 0)]
    first = embeddings.payload()["items"][0]
    assert first["key"] == object_key(record.annotations[0])
    assert (first["split"], first["image_name"]) == ("train", "sample.jpg")

    # deleting the first line renumbers the others: the plot keeps them under their new ids and keys
    dataset.bulk_edit_objects([{"image_id": record.id, "annotation_id": first["annotation_id"], "action": "delete", "key": first["key"]}])
    assert shown(embeddings) == [(1, "1", 1), (2, "2", 0)]
    last = embeddings.payload()["items"][-1]
    assert last["key"] == object_key(record.annotations[1])

    dataset.bulk_edit_objects([{"image_id": record.id, "annotation_id": last["annotation_id"], "action": "relabel", "class_id": 1, "key": last["key"]}])
    assert shown(embeddings) == [(1, "1", 1), (2, "2", 1)]
    assert embeddings.payload()["items"][-1]["key"] == object_key(record.annotations[1])

    dataset.history("undo")
    dataset.history("undo")
    assert shown(embeddings) == [(0, "1", 0), (1, "2", 1), (2, "3", 0)]


def test_embeddings_leave_out_reshaped_objects_and_deleted_images(tmp_path):
    dataset, record, embeddings = computed(tmp_path)
    dataset.replace_annotations(record.id, [
        {"class_id": 0, "points": [0.2, 0.2, 0.1, 0.1]},
        {"class_id": 1, "points": [0.5, 0.5, 0.3, 0.4]},  # moved: its embedding was of another patch
        {"class_id": 0, "points": [0.8, 0.8, 0.1, 0.1]},
        {"class_id": 1, "points": [0.1, 0.9, 0.1, 0.1]},  # added: not embedded yet
    ])
    assert shown(embeddings) == [(0, "1", 0), (2, "3", 0)]

    dataset.delete_images([record.id])
    assert shown(embeddings) == []
    dataset.history("undo")
    assert shown(embeddings) == [(0, "1", 0), (2, "3", 0)]
