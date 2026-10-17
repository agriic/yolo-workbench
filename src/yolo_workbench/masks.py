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


def mask_outlines(mask: np.ndarray, max_points: int | None = None) -> list[list[float]]:
    """External contour of each part of the mask, simplified, as normalized x,y pairs (≥3 points), largest first.
    With `max_points`, a polygon that would have more vertices is fitted to the full contour with that many."""
    import cv2

    method = cv2.CHAIN_APPROX_NONE if max_points else cv2.CHAIN_APPROX_SIMPLE
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, method)
    height, width = mask.shape
    outlines = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        # ~1 px of tolerance on small objects, proportionally more on large ones: few enough vertices to edit
        epsilon = max(1.0, 0.0015 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        if max_points and len(polygon) > max_points:
            raw = [value for x, y in contour.reshape(-1, 2).tolist() for value in (x / width, y / height)]
            outlines.append(simplify_polygon(raw, max_points, width, height))
        else:
            outlines.append([round(min(max(value, 0.0), 1.0), 6) for x, y in polygon.tolist() for value in (x / width, y / height)])
    return outlines


FIT_ROUNDS = 6  # line-fit / re-split rounds after Douglas–Peucker picks the vertices
PARALLEL_SIN = float(np.sin(np.radians(4)))  # edges closer to parallel than this do not meet at a usable corner


def simplify_polygon(points: list[float], max_points: int, width: int, height: int) -> list[float]:
    """The polygon (normalized x,y pairs) with at most `max_points` vertices, fitted to the original outline.

    Works in pixels of a width×height image (both axes weigh the same) on the outline resampled at ~1 px.
    Douglas–Peucker at the smallest tolerance that fits picks which stretch of outline each edge covers;
    then each edge becomes the least-squares line through its stretch and each vertex moves to where its
    two edge lines meet — off the outline, e.g. into a corner the mask rounded off — the stretches are
    re-split at the outline points nearest the new vertices, and so on while the fit improves. The polygon
    is returned unchanged when it already has few enough vertices or cannot keep 3."""
    if max_points < 3 or len(points) // 2 <= max_points:
        return points
    import cv2

    if width <= 0 or height <= 0:
        width = height = 1000  # size not probed yet: any isotropic pixel scale works
    scale = np.array([width, height], dtype=float)
    outline = _resample(np.asarray(points, dtype=float).reshape(-1, 2) * scale)
    # Douglas–Peucker keeps its starting vertex: start at the one farthest from the centre, a corner
    outline = np.roll(outline, -int(np.argmax(((outline - outline.mean(axis=0)) ** 2).sum(axis=1))), axis=0)
    contour = outline.astype(np.float32).reshape(-1, 1, 2)
    low, high = 0.0, float(cv2.arcLength(contour, True))
    best = None
    while high - low > 0.01:
        epsilon = (low + high) / 2
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) <= max_points:
            best, high = approx, epsilon
        else:
            low = epsilon
    if best is None or len(best) < 3:
        return points
    # approxPolyDP returns a subset of its input: find each vertex's position along the outline
    positions: dict[tuple[float, float], int] = {}
    for index, point in enumerate(map(tuple, contour.reshape(-1, 2).tolist())):
        positions.setdefault(point, index)
    breaks = sorted({positions[tuple(point)] for point in best.tolist()})
    if len(breaks) < 3:
        return points
    vertices = _fit_vertices(outline, breaks)
    return [round(min(max(float(value), 0.0), 1.0), 6) for x, y in (vertices / scale).tolist() for value in (x, y)]


def _resample(outline: np.ndarray) -> np.ndarray:
    """The closed outline with points added along its edges so neighbours are at most ~1 px apart (at most
    ~4000 added in all); every original point stays. Line fits then weigh each stretch by its length."""
    following = np.roll(outline, -1, axis=0)
    lengths = np.hypot(*(following - outline).T)
    spacing = max(1.0, float(lengths.sum()) / 4000)
    counts = np.maximum(np.ceil(lengths / spacing).astype(int), 1)
    owner = np.repeat(np.arange(len(outline)), counts)
    step = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)) / np.repeat(counts, counts)
    return outline[owner] + (following[owner] - outline[owner]) * step[:, None]


def _stretch(outline: np.ndarray, start: int, end: int) -> np.ndarray:
    """Outline points from position `start` to `end` inclusive; positions may run past either end and wrap."""
    return outline[np.arange(start, end + 1) % len(outline)]


def _line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total least-squares line through the points (≥2): centroid and unit direction."""
    centre = points.mean(axis=0)
    if len(points) < 3:
        direction = points[-1] - points[0]
        return centre, direction / max(float(np.hypot(*direction)), 1e-12)
    _, vectors = np.linalg.eigh(np.cov((points - centre).T))
    return centre, vectors[:, -1]


def _fit_error(outline: np.ndarray, breaks: list[int], vertices: np.ndarray) -> float:
    """Sum of squared distances from each outline point to the polygon edge covering it."""
    total = 0.0
    count = len(breaks)
    for index in range(count):
        end = breaks[index + 1] if index + 1 < count else breaks[0] + len(outline)
        stretch = _stretch(outline, breaks[index], end)
        a, b = vertices[index], vertices[(index + 1) % count]
        edge = b - a
        t = np.clip(((stretch - a) @ edge) / max(float(edge @ edge), 1e-12), 0, 1)
        total += float((((a + t[:, None] * edge) - stretch) ** 2).sum())
    return total


def _fit_vertices(outline: np.ndarray, breaks: list[int]) -> np.ndarray:
    """Vertices of the polygon whose edges cover the outline between consecutive `breaks` (increasing
    positions spanning less than one lap), fitted as simplify_polygon describes."""
    count, size = len(breaks), len(outline)
    vertices = outline[breaks]
    error = _fit_error(outline, breaks, vertices)
    for _ in range(FIT_ROUNDS):
        ends = breaks[1:] + [breaks[0] + size]
        lines = [_line(_stretch(outline, start, end)) for start, end in zip(breaks, ends)]
        moved = vertices.copy()
        for index in range(count):
            (c1, d1), (c2, d2) = lines[index - 1], lines[index]  # vertex `index` joins the edge before it and its own
            old = vertices[index]
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            # how far a vertex may move: half its shorter edge, so a shallow intersection cannot run away
            reach = max(1.0, 0.5 * min(np.hypot(*(old - vertices[index - 1])), np.hypot(*(old - vertices[(index + 1) % count]))))
            point = None
            if abs(cross) > PARALLEL_SIN:
                offset = c2 - c1
                point = c1 + d1 * (offset[0] * d2[1] - offset[1] * d2[0]) / cross
            if point is None or np.hypot(*(point - old)) > reach:
                # (nearly) straight through this vertex: the mean of its projections onto both lines
                point = (c1 + d1 * float((old - c1) @ d1) + c2 + d2 * float((old - c2) @ d2)) / 2
            moved[index] = point
        # re-split: each stretch now starts at the outline point nearest its new vertex, keeping the order
        resplit = list(breaks)
        for index in range(count):
            low = resplit[index - 1] if index else breaks[-1] - size
            high = breaks[index + 1] if index + 1 < count else resplit[0] + size
            candidates = np.arange(low + 1, high)
            if len(candidates):
                resplit[index] = int(candidates[np.argmin(((outline[candidates % size] - moved[index]) ** 2).sum(axis=1))])
        moved_error = _fit_error(outline, resplit, moved)
        if moved_error >= error:
            break
        improved = moved_error < error * 0.999
        breaks, vertices, error = resplit, moved, moved_error
        if not improved:
            break
    return vertices


def mask_box(mask: np.ndarray) -> list[float]:
    """Normalized cx, cy, w, h around every mask pixel (all parts, e.g. an object split by an occluder)."""
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    left, right = xs.min() / width, (xs.max() + 1) / width
    top, bottom = ys.min() / height, (ys.max() + 1) / height
    return [round(float(value), 6) for value in ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)]
