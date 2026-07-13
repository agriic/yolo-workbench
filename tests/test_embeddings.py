from yolo_workbench.embeddings import _normalize_3d_points


def test_normalize_3d_points_normalizes_all_axes():
    points = _normalize_3d_points([(10, -2, 100), (20, 2, 300)])

    assert points == [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]


def test_normalize_3d_points_pads_missing_and_constant_axes():
    points = _normalize_3d_points([(4,), (4,)])

    assert points == [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
