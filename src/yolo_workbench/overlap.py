"""Near-duplicate annotations: pairs of objects in one image that cover nearly the same place.

Boxes compare by IoU; polygons by the IoU of their filled areas (rasterized over the pair's bounds),
reached only when a cheap upper bound from their boxes and areas can still pass the threshold.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

OVERLAP_IOU = 0.85  # two objects at or above this IoU are reported as covering the same place
RASTER = 128  # cells across each side of a polygon pair's bounds


def _bounds(category: str, points: list[float]) -> tuple[float, float, float, float]:
    if category == "segmentation":
        xs, ys = points[0::2], points[1::2]
        return min(xs), min(ys), max(xs), max(ys)
    cx, cy, w, h = points
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _polygon_area(points: list[float]) -> float:
    xs, ys = points[0::2], points[1::2]
    return abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(xs)))) / 2


def _polygon_iou(a: list[float], b: list[float], box: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = box
    sx, sy = RASTER / max(x1 - x0, 1e-12), RASTER / max(y1 - y0, 1e-12)

    def mask(points: list[float]) -> np.ndarray:
        image = Image.new("1", (RASTER, RASTER))
        ImageDraw.Draw(image).polygon([((x - x0) * sx, (y - y0) * sy) for x, y in zip(points[0::2], points[1::2])], fill=1)
        return np.asarray(image)

    first, second = mask(a), mask(b)
    union = np.count_nonzero(first | second)
    return np.count_nonzero(first & second) / union if union else 0.0


def find_overlaps(category: str, annotations, threshold: float = OVERLAP_IOU) -> list[tuple[int, int, float]]:
    """(earlier index, later index, IoU) for every pair at or above `threshold`; exact copies are skipped."""
    shapes = []
    for index, annotation in enumerate(annotations):
        x0, y0, x1, y1 = _bounds(category, annotation.points)
        area = (x1 - x0) * (y1 - y0) if category == "detection" else _polygon_area(annotation.points)
        if area > 0:
            shapes.append((x0, y0, x1, y1, area, index))
    shapes.sort()
    pairs = []
    for position, (ax0, ay0, ax1, ay1, a_area, i) in enumerate(shapes):
        for bx0, by0, bx1, by1, b_area, j in shapes[position + 1:]:
            if bx0 >= ax1:
                break  # sorted by left edge: nothing further right can intersect
            width, height = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
            if width <= 0 or height <= 0:
                continue
            first, second = sorted((i, j))
            if annotations[first].class_id == annotations[second].class_id and annotations[first].points == annotations[second].points:
                continue  # reported as an exact duplicate
            intersection = width * height
            if category == "detection":
                iou = intersection / (a_area + b_area - intersection)
            elif min(intersection, a_area, b_area) / max(a_area, b_area) < threshold:
                continue  # the polygons' overlap is within their boxes' and their union at least the larger one
            else:
                iou = _polygon_iou(annotations[i].points, annotations[j].points, (min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)))
            if iou >= threshold:
                pairs.append((first, second, iou))
    return sorted(pairs, key=lambda pair: (pair[1], pair[0]))
