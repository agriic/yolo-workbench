"""Masks to annotations: SAM's smart annotation and RF-DETR's segmentation predictions."""

from __future__ import annotations

import numpy as np


def drop_specks(mask: np.ndarray) -> np.ndarray:
    """The mask without the specks SAM scatters around an object, often far from it (its 256×256 low-res
    mask is up-sampled, leaving islands of a few cells). A part stays when it is at least 4×4 low-res cells
    and 1% of the largest part; the largest part always stays."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    speck = max((max(mask.shape) / 64) ** 2, areas.max() / 100)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = (areas >= speck) | (areas == areas.max())
    return keep[labels]


def mask_outlines(mask: np.ndarray) -> list[list[float]]:
    """External contour of each part of the mask, simplified, as normalized x,y pairs (≥3 points), largest first."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    outlines = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        # ~1 px of tolerance on small objects, proportionally more on large ones: few enough vertices to edit
        epsilon = max(1.0, 0.0015 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) >= 3:
            outlines.append([round(min(max(value, 0.0), 1.0), 6) for x, y in polygon.tolist() for value in (x / width, y / height)])
    return outlines


def mask_box(mask: np.ndarray) -> list[float]:
    """Normalized cx, cy, w, h around every mask pixel (all parts, e.g. an object split by an occluder)."""
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    left, right = xs.min() / width, (xs.max() + 1) / width
    top, bottom = ys.min() / height, (ys.max() + 1) / height
    return [round(float(value), 6) for value in ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)]
