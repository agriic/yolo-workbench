"""Augmentation previews for the new-run form: each augmentation at the two ends of its range, and whole-pipeline samples.

Detection and segmentation run Ultralytics' own transforms (`ultralytics.data.augment`) on dataset images, so a preview
is what training does, labels included (clipping, the box filter, copy-paste selection). For the range ends, every random
draw inside that module is pinned to the low or the high end of its range for the duration of one render; pipeline samples
seed the real random generators instead. Classification runs the torchvision operations Ultralytics composes
(`classify_augmentations`). Imports of ultralytics/torch happen on first use.

Images are rendered at `min(imgsz, WORK_SIZE)`. Every augmentation is scale-free except `perspective`, whose coefficient is
in pixels; it is rescaled so a preview looks like the same value at the real `imgsz`.
"""
from __future__ import annotations

import io
import math
import random
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from .dataset import Dataset, DatasetError
from .models import Annotation

WORK_SIZE = 640
SEGMENT_POINTS = 1000  # Ultralytics resamples every polygon to this many points
RANGE_KEYS = {"hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective"}
DET_SEG_KEYS = RANGE_KEYS | {"flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix", "copy_paste"}
CLS_KEYS = {"hsv_h", "hsv_s", "hsv_v", "scale", "flipud", "fliplr", "erasing", "auto_augment"}
CLS_RATIO = (3 / 4, 4 / 3)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_render_lock = threading.Lock()  # renders patch module globals and reseed global generators
_image_cache: OrderedDict = OrderedDict()
_IMAGE_CACHE_SIZE = 24


@dataclass
class Sample:
    path: Path
    annotations: list[Annotation] = field(default_factory=list)


@dataclass
class Preview:
    jpeg: bytes
    note: str = ""


def preview_keys(category: str) -> set[str]:
    return CLS_KEYS if category == "classification" else DET_SEG_KEYS


# ---------- samples ----------


def sample_pool(dataset: Dataset, index: int, partners: int = 8) -> tuple[list[Sample], int]:
    """The `index`-th preview image (train split first, labelled images first) plus the images it is mixed with."""
    with dataset._lock:
        records = list(dataset.images.values())
        train = [record for record in records if record.split == "train"] or records
        if dataset.category != "classification":
            labelled = [record for record in train if any(_valid(annotation, dataset.category) for annotation in record.annotations)]
            train = labelled or train
        pool = [Sample(record.path, list(record.annotations)) for record in sorted(train, key=lambda record: record.id)]
    if not pool:
        raise DatasetError("No images to preview augmentations on")
    start = index % len(pool)
    return [pool[(start + offset) % len(pool)] for offset in range(1 + partners)], len(pool)


def _valid(annotation: Annotation, category: str) -> bool:
    points = annotation.points
    if category == "detection":
        return len(points) == 4 and points[2] > 0 and points[3] > 0
    return len(points) >= 6 and len(points) % 2 == 0


def _read_bgr(path: Path, size: int) -> np.ndarray:
    """The image as Ultralytics' loader keeps it for training: BGR, long side resized to `size`."""
    import cv2

    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, size)
    cached = _image_cache.get(key)
    if cached is not None:
        _image_cache.move_to_end(key)
        return cached
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DatasetError(f"Cannot read image {path.name}")
    height, width = image.shape[:2]
    ratio = size / max(height, width)
    if ratio != 1:
        image = cv2.resize(image, (min(math.ceil(width * ratio), size), min(math.ceil(height * ratio), size)), interpolation=cv2.INTER_LINEAR)
    _image_cache[key] = image
    while len(_image_cache) > _IMAGE_CACHE_SIZE:
        _image_cache.popitem(last=False)
    return image


def _labels(sample: Sample, category: str, size: int) -> dict:
    """A labels dict shaped like YOLODataset.get_image_and_label()."""
    from ultralytics.utils.instance import Instances
    from ultralytics.utils.ops import resample_segments, segments2boxes

    image = _read_bgr(sample.path, size).copy()
    annotations = [annotation for annotation in sample.annotations if _valid(annotation, category)]
    cls = np.array([[annotation.class_id] for annotation in annotations], dtype=np.float32).reshape(-1, 1)
    if category == "segmentation" and annotations:
        polygons = [np.array(annotation.points, dtype=np.float32).reshape(-1, 2) for annotation in annotations]
        bboxes = segments2boxes(polygons)
        count = max(SEGMENT_POINTS, max(len(polygon) for polygon in polygons) + 1)
        segments = np.stack(resample_segments(polygons, n=count), axis=0)
    else:
        bboxes = np.array([annotation.points for annotation in annotations], dtype=np.float32).reshape(-1, 4)
        segments = np.zeros((0, SEGMENT_POINTS, 2), dtype=np.float32)
    return {
        "im_file": str(sample.path),
        "ori_shape": image.shape[:2],
        "resized_shape": image.shape[:2],
        "img": image,
        "cls": cls,
        "instances": Instances(bboxes, segments, None, bbox_format="xywh", normalized=True),
    }


class _PreviewDataset:
    """What Ultralytics' mix transforms need from a dataset: partner images, in order, whatever index they draw."""

    cache = "ram"  # Mosaic draws from the whole dataset instead of its buffer
    data: dict = {}
    use_keypoints = False

    def __init__(self, partners: list[Sample], category: str, size: int):
        self.partners = partners or []
        self.category = category
        self.size = size
        self.taken = 0

    def __len__(self) -> int:
        return max(1, len(self.partners))

    def get_image_and_label(self, index: int) -> dict:
        sample = self.partners[self.taken % len(self.partners)]
        self.taken += 1
        return _labels(sample, self.category, self.size)


# ---------- pinned and seeded randomness ----------


class _Pinned:
    """Stands in for `random` / `np.random`: every draw lands `at` of the way through its range (0 low end, 1 high end)."""

    def __init__(self, at: float, fallback):
        self.at = at
        self.fallback = fallback

    def uniform(self, low=0.0, high=1.0, size=None):
        value = low + (high - low) * self.at
        return np.full(size, value) if size is not None else value

    def beta(self, a, b, size=None):
        return a / (a + b)

    def __getattr__(self, name):
        return getattr(self.fallback, name)


class _NumpyWithPinnedRandom:
    def __init__(self, at: float):
        self.random = _Pinned(at, np.random)

    def __getattr__(self, name):
        return getattr(np, name)


class _pinned_draws:
    def __init__(self, at: float):
        self.at = at

    def __enter__(self):
        from ultralytics.data import augment

        self.module = augment
        self.saved = (augment.random, augment.np)
        augment.random = _Pinned(self.at, random)
        augment.np = _NumpyWithPinnedRandom(self.at)

    def __exit__(self, *exc):
        self.module.random, self.module.np = self.saved


class _seeded:
    """Seed the global generators Ultralytics and torchvision draw from, and put their state back afterwards."""

    def __init__(self, seed: int):
        self.seed = seed

    def __enter__(self):
        import torch

        self.saved = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def __exit__(self, *exc):
        import torch

        python_state, numpy_state, torch_state = self.saved
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


# ---------- detection / segmentation ----------


def _hyp(values: dict, defaults: dict, imgsz: int, work: int) -> SimpleNamespace:
    keys = DET_SEG_KEYS | {"copy_paste_mode"}
    hyp = {key: values.get(key, defaults.get(key)) for key in keys}
    hyp["perspective"] = float(hyp["perspective"] or 0) * imgsz / work
    hyp["augmentations"] = None
    return SimpleNamespace(**hyp)


def _letterbox(labels: dict, work: int) -> dict:
    """The non-mosaic path: an identity RandomPerspective centres the image on a square canvas."""
    from ultralytics.data.augment import RandomPerspective

    return RandomPerspective(degrees=0, translate=0, scale=0, shear=0, perspective=0, size=(work, work))(labels)


def _instance_count(labels: dict) -> int:
    return len(labels["cls"])


def _detect_range(key: str, value: float, side: str, samples: list[Sample], category: str, imgsz: int, work: int) -> tuple[dict, str]:
    from ultralytics.data.augment import RandomHSV, RandomPerspective

    labels = _labels(samples[0], category, work)
    before = _instance_count(labels)
    with _pinned_draws(0.0 if side == "min" else 1.0):
        if key.startswith("hsv_"):
            labels = _letterbox(labels, work)
            gains = {name: value if name == key else 0.0 for name in ("hsv_h", "hsv_s", "hsv_v")}
            labels = RandomHSV(hgain=gains["hsv_h"], sgain=gains["hsv_s"], vgain=gains["hsv_v"])(labels)
        else:
            params = {"degrees": 0, "translate": 0, "scale": 0, "shear": 0, "perspective": 0}
            params[key] = value * imgsz / work if key == "perspective" else value
            labels = RandomPerspective(**params, size=(work, work))(labels)
    after = _instance_count(labels)
    return labels, f"{before - after} of {before} labels dropped" if after < before else ""


def _detect_applied(key: str, values: dict, defaults: dict, samples: list[Sample], category: str, imgsz: int, work: int, seed: int) -> tuple[dict, str]:
    """The chance augmentations, applied: what an image looks like when the coin lands on this augmentation."""
    from ultralytics.data.augment import CopyPaste, CutMix, Mosaic, MixUp, RandomFlip

    value = float(values.get(key, defaults.get(key)) or 0)
    labels = _labels(samples[0], category, work)
    partners = _PreviewDataset(samples[1:], category, work)
    note = ""
    if key in ("fliplr", "flipud"):
        labels = RandomFlip(p=1.0, direction="horizontal" if key == "fliplr" else "vertical")(_letterbox(labels, work))
    elif key == "bgr":
        labels = _letterbox(labels, work)
        labels["img"] = np.ascontiguousarray(labels["img"][..., ::-1])
    elif key == "mosaic":
        with _pinned_draws(0.5):  # mosaic centre in the middle of its range
            labels = _letterbox(Mosaic(partners, imgsz=work, p=1.0)(labels), work)
    elif key == "mixup":
        with _pinned_draws(0.5):  # mix ratio at the mean of Beta(32, 32)
            labels = MixUp(partners, pre_transform=lambda item: _letterbox(item, work), p=1.0)(_letterbox(labels, work))
    elif key == "cutmix":
        labels, note = _cutmix(labels, partners, work, seed)
    elif key == "copy_paste":
        mode = values.get("copy_paste_mode", defaults.get("copy_paste_mode")) or "flip"
        labels, note = _copy_paste(CopyPaste, labels, partners, mode, value, work)
    return labels, note


def _cutmix(labels: dict, partners: _PreviewDataset, work: int, seed: int) -> tuple[dict, str]:
    """CutMix only pastes into an area free of this image's objects; try a few draws to find one."""
    from ultralytics.data.augment import CutMix

    base = _letterbox(labels, work)
    transform = CutMix(partners, pre_transform=lambda item: _letterbox(item, work), p=1.0)
    for attempt in range(24):
        candidate = {**base, "img": base["img"].copy(), "instances": base["instances"][np.arange(len(base["cls"]))]}
        partners.taken = 0
        with _seeded(seed * 100 + attempt):
            params = transform.get_params(candidate)
        if not params.get("skip"):
            candidate = transform.apply_instances(transform.apply_image(candidate, params), params)
            candidate.pop("mix_labels", None)
            return candidate, ""
    return base, "No object-free area here: CutMix leaves this image unchanged"


def _copy_paste(CopyPaste, labels: dict, partners: _PreviewDataset, mode: str, value: float, work: int) -> tuple[dict, str]:
    if not len(labels["instances"].segments):
        return _letterbox(labels, work), "Copy-paste needs polygons; this image has none"
    before = _instance_count(labels)
    transform = CopyPaste(partners, pre_transform=lambda item: _letterbox(item, work), p=value, mode=mode)
    if mode == "flip":  # pastes flipped copies of this image's objects, before the affine step
        labels = _letterbox(transform(labels), work)
    else:  # pastes objects of another image, after it; the chance check is skipped to show the effect
        labels = _letterbox(labels, work)
        params = transform.get_params(labels)
        labels = transform.apply_instances(transform.apply_image(labels, params), params)
        labels.pop("mix_labels", None)
    pasted = _instance_count(labels) - before
    return labels, f"{pasted} object{'s' if pasted != 1 else ''} pasted" + ("" if mode == "flip" else " from another image")


def _detect_pipeline(values: dict, defaults: dict, samples: list[Sample], category: str, imgsz: int, work: int, seed: int) -> dict:
    """One training sample: Ultralytics' v8_transforms with these arguments, seeded."""
    from ultralytics.data.augment import v8_transforms

    hyp = _hyp(values, defaults, imgsz, work)
    partners = _PreviewDataset(samples, category, work)
    with _seeded(seed):
        transforms = v8_transforms(partners, work, hyp)
        labels = transforms(_labels(samples[0], category, work))
        labels.pop("mix_labels", None)
        if hyp.bgr and random.uniform(0, 1) < float(hyp.bgr):  # Format() swaps channels after the transforms
            labels["img"] = np.ascontiguousarray(labels["img"][..., ::-1])
    return labels


def _detect_image(labels: dict, category: str, size: int, draw_labels: bool, color: Callable[[int], str]) -> Image.Image:
    import cv2

    image = labels["img"]
    height, width = image.shape[:2]
    rgb = Image.fromarray(cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2RGB))
    factor = size / max(height, width)
    rgb = rgb.resize((max(1, round(width * factor)), max(1, round(height * factor))), Image.Resampling.LANCZOS)
    if draw_labels and _instance_count(labels):
        instances = labels["instances"]
        instances.convert_bbox("xyxy")
        if instances.normalized:
            instances.denormalize(width, height)
        draw = ImageDraw.Draw(rgb)
        classes = labels["cls"].reshape(-1).astype(int)
        line = max(1, round(size / 160))
        if category == "segmentation" and len(instances.segments):
            for class_id, segment in zip(classes, instances.segments):
                points = [(float(x) * factor, float(y) * factor) for x, y in segment]
                draw.line(points + points[:1], fill=color(int(class_id)), width=line)
        else:
            for class_id, box in zip(classes, instances.bboxes):
                draw.rectangle(tuple(float(value) * factor for value in box), outline=color(int(class_id)), width=line)
    return rgb


# ---------- classification (torchvision) ----------


def _cls_base(sample: Sample, work: int) -> Image.Image:
    import cv2

    return Image.fromarray(cv2.cvtColor(_read_bgr(sample.path, work * 2), cv2.COLOR_BGR2RGB))


def _crop_box(width: int, height: int, area_fraction: float) -> tuple[int, int, int, int]:
    """A centred RandomResizedCrop box (top, left, h, w) of this share of the area at aspect 1, else torchvision's fallback."""
    side = round(math.sqrt(area_fraction * width * height))
    if area_fraction < 1 and 0 < side <= min(width, height):
        return (height - side) // 2, (width - side) // 2, side, side
    ratio = width / height
    if ratio < CLS_RATIO[0]:
        w, h = width, round(width / CLS_RATIO[0])
    elif ratio > CLS_RATIO[1]:
        h, w = height, round(height * CLS_RATIO[1])
    else:
        w, h = width, height
    return (height - h) // 2, (width - w) // 2, h, w


def _cls_square(image: Image.Image, work: int, area_fraction: float = 1.0) -> Image.Image:
    import torchvision.transforms.functional as F

    top, left, h, w = _crop_box(image.width, image.height, area_fraction)
    return F.resized_crop(image, top, left, h, w, [work, work], antialias=True)


def _erase(image: Image.Image, seed: int) -> Image.Image:
    """RandomErasing as Ultralytics places it: after normalisation, so erased pixels become the mean colour."""
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as F

    tensor = F.normalize(F.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
    with _seeded(seed):
        tensor = T.RandomErasing(p=1.0)(tensor)
    tensor = tensor * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    return F.to_pil_image(tensor.clamp(0, 1))


def _auto_augment(policy: str | None):
    import torchvision.transforms as T

    return {"randaugment": T.RandAugment, "autoaugment": T.AutoAugment, "augmix": T.AugMix}.get(policy or "")


def _cls_range(key: str, value, side: str, sample: Sample, work: int) -> tuple[Image.Image, str]:
    import torchvision.transforms.functional as F

    low = side == "min"
    image = _cls_base(sample, work)
    if key == "scale":
        fraction = max(1.0 - float(value), 0.01) if low else 1.0
        return _cls_square(image, work, fraction), ""
    square = _cls_square(image, work)
    if key == "hsv_h":
        return F.adjust_hue(square, -min(float(value), 0.5) if low else min(float(value), 0.5)), ""
    factor = max(0.0, 1 - float(value)) if low else 1 + float(value)
    if key == "hsv_s":
        return F.adjust_saturation(square, factor), ""
    return F.adjust_contrast(F.adjust_brightness(square, factor), factor), ""  # hsv_v drives brightness and contrast


def _cls_applied(key: str, values: dict, defaults: dict, sample: Sample, work: int, seed: int, side: str) -> tuple[Image.Image, str]:
    import torchvision.transforms.functional as F

    square = _cls_square(_cls_base(sample, work), work)
    if key == "fliplr":
        return F.hflip(square), ""
    if key == "flipud":
        return F.vflip(square), ""
    if key == "erasing":  # erased area 2–33 % of the image: two draws
        return _erase(square, seed * 2 + (side == "max")), ""
    policy = values.get("auto_augment", defaults.get("auto_augment"))
    transform = _auto_augment(policy)
    if transform is None:
        return square, "Off"
    with _seeded(seed * 2 + (side == "max")):  # two different random draws of the policy
        return transform()(square), ""


def _cls_pipeline(values: dict, defaults: dict, sample: Sample, work: int, seed: int) -> Image.Image:
    """One training sample: Ultralytics' classify_augmentations with these arguments, seeded."""
    import torch
    import torchvision.transforms.functional as F
    from ultralytics.data.augment import classify_augmentations

    value = lambda key: values.get(key, defaults.get(key))  # noqa: E731
    transforms = classify_augmentations(
        size=work,
        scale=(1.0 - float(value("scale")), 1.0),
        hflip=float(value("fliplr")),
        vflip=float(value("flipud")),
        erasing=float(value("erasing")),
        auto_augment=value("auto_augment"),
        hsv_h=min(float(value("hsv_h")), 0.5),
        hsv_s=float(value("hsv_s")),
        hsv_v=float(value("hsv_v")),
    )
    with _seeded(seed):
        tensor = transforms(_cls_base(sample, work))
    tensor = tensor * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    return F.to_pil_image(tensor.clamp(0, 1))


# ---------- entry points ----------


def _jpeg(image: Image.Image, size: int) -> bytes:
    if max(image.size) != size:
        factor = size / max(image.size)
        image = image.resize((max(1, round(image.width * factor)), max(1, round(image.height * factor))), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.convert("RGB").save(output, "JPEG", quality=86)
    return output.getvalue()


def _work_size(imgsz: int) -> int:
    return max(32, min(int(imgsz), WORK_SIZE))


def render_original(dataset: Dataset, sample: int, imgsz: int, size: int, draw_labels: bool, color: Callable[[int], str]) -> Preview:
    samples, total = sample_pool(dataset, sample, partners=0)
    work = _work_size(imgsz)
    with _render_lock:
        if dataset.category == "classification":
            image = _cls_square(_cls_base(samples[0], work), work)
        else:
            image = _detect_image(_letterbox(_labels(samples[0], dataset.category, work), work), dataset.category, size, draw_labels, color)
    return Preview(_jpeg(image, size), f"{samples[0].path.name} · {sample % total + 1} of {total}")


def render_augmentation(dataset: Dataset, key: str, side: str, values: dict, defaults: dict, sample: int, imgsz: int, size: int, draw_labels: bool, color: Callable[[int], str]) -> Preview:
    """`key` alone, at the low (`side="min"`) or high end of its range; chance augmentations always show their effect."""
    category = dataset.category
    if key not in preview_keys(category):
        raise DatasetError(f"No preview for `{key}` in {category} training")
    samples, _ = sample_pool(dataset, sample)
    work = _work_size(imgsz)
    value = values.get(key, defaults.get(key))
    with _render_lock:
        if category == "classification":
            if key in RANGE_KEYS:
                image, note = _cls_range(key, value or 0, side, samples[0], work)
            else:
                image, note = _cls_applied(key, values, defaults, samples[0], work, sample, side)
            return Preview(_jpeg(image, size), note)
        if key in RANGE_KEYS:
            labels, note = _detect_range(key, float(value or 0), side, samples, category, imgsz, work)
        else:
            labels, note = _detect_applied(key, values, defaults, samples, category, imgsz, work, sample)
        return Preview(_jpeg(_detect_image(labels, category, size, draw_labels, color), size), note)


def render_pipeline(dataset: Dataset, values: dict, defaults: dict, sample: int, seed: int, imgsz: int, size: int, draw_labels: bool, color: Callable[[int], str]) -> Preview:
    """A training sample with every augmentation at once, as the dataloader would produce it."""
    samples, _ = sample_pool(dataset, sample)
    work = _work_size(imgsz)
    with _render_lock:
        if dataset.category == "classification":
            return Preview(_jpeg(_cls_pipeline(values, defaults, samples[0], work, seed), size))
        labels = _detect_pipeline(values, defaults, samples, dataset.category, imgsz, work, seed)
        return Preview(_jpeg(_detect_image(labels, dataset.category, size, draw_labels, color), size))
