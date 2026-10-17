"""RF-DETR's custom augmentations (`aug_config`): the Albumentations transforms on offer, their validation, and previews.

`aug_config` maps Albumentations transform names to their parameters (`{"HorizontalFlip": {"p": 0.5}, ...}`); RF-DETR
builds each one on its own, in order, after its resize step. A transform RF-DETR counts as geometric (`GEOMETRIC`, a copy
of `rfdetr.datasets.transforms.GEOMETRIC_TRANSFORMS`) moves the boxes and masks with the image; any other one only
changes pixels. RF-DETR skips a transform it cannot build with nothing but a log line, so the server builds every one
first (`check`) and reports what would be dropped.

No aug_config at all is RF-DETR's default (torchvision: a horizontal flip half the time), `{}` turns augmentation off.
The presets are copies of `rfdetr.datasets.aug_configs` (importing rfdetr takes seconds and loads torch, so the server
never does; a test keeps them in sync). Previews build the same transforms with Albumentations directly, on a dataset
image resized to the square training input, at `min(resolution, WORK_SIZE)`.
"""
from __future__ import annotations

import copy
import difflib
import inspect
import json
import os
import typing
import warnings
from collections.abc import Callable
from functools import cache

import numpy as np
from PIL import Image, ImageDraw

from . import libraries
from .augment_preview import Preview, Sample, _jpeg, _read_bgr, _valid, sample_pool
from .dataset import Dataset, DatasetError

WORK_SIZE = 640
CONTAINERS = ("OneOf", "SomeOf", "Sequential")

# rfdetr.datasets.aug_configs (AUG_CONFIG is what Albumentations-era RF-DETR did by default)
PRESETS: dict[str, dict] = {
    "conservative": {
        "HorizontalFlip": {"p": 0.5},
        "RandomBrightnessContrast": {"brightness_limit": 0.1, "contrast_limit": 0.1, "p": 0.3},
    },
    "aggressive": {
        "HorizontalFlip": {"p": 0.5},
        "VerticalFlip": {"p": 0.5},
        "Rotate": {"limit": 45, "p": 0.5},
        "Affine": {"scale": [0.8, 1.2], "translate_percent": [-0.1, 0.1], "rotate": [-15, 15], "shear": [-5, 5], "p": 0.5},
        "ColorJitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1, "p": 0.5},
    },
    "aerial": {
        "HorizontalFlip": {"p": 0.5},
        "VerticalFlip": {"p": 0.5},
        "Rotate": {"limit": [90, 90], "p": 0.5},
        "RandomBrightnessContrast": {"brightness_limit": 0.15, "contrast_limit": 0.15, "p": 0.4},
    },
    "industrial": {
        "HorizontalFlip": {"p": 0.3},
        "RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5},
        "GaussianBlur": {"blur_limit": 3, "p": 0.3},
        "GaussNoise": {"std_range": [0.01, 0.05], "p": 0.3},
    },
}
DEFAULT_EQUIVALENT = {"HorizontalFlip": {"p": 0.5}}  # what RF-DETR does without an aug_config

# rfdetr.datasets.transforms.GEOMETRIC_TRANSFORMS (with the horizontal-flip aliases): boxes follow these
GEOMETRIC = frozenset({
    "VerticalFlip", "Transpose", "Rotate", "RandomRotate90", "Affine", "ShiftScaleRotate", "SafeRotate", "RandomCrop",
    "RandomSizedCrop", "CenterCrop", "Crop", "CropNonEmptyMaskIfExists", "RandomCropNearBBox", "RandomCropFromBorders",
    "RandomSizedBBoxSafeCrop", "BBoxSafeRandomCrop", "AtLeastOneBBoxRandomCrop", "RandomResizedCrop", "CropAndPad",
    "Perspective", "ElasticTransform", "GridDistortion", "GridElasticDeform", "OpticalDistortion", "PiecewiseAffine",
    "ThinPlateSpline", "RandomGridShuffle", "Resize", "SmallestMaxSize", "LongestMaxSize", "CappedLongestMaxSize",
    "RandomScale", "Downscale", "PadIfNeeded", "Pad",
    "HorizontalFlip", "TimeReverse", "Flip", "D4", "SquareSymmetry",
})
# rfdetr.datasets.kornia_transforms._REGISTRY: what the GPU (Kornia) augmentation backend can run
KORNIA = frozenset({
    "HorizontalFlip", "VerticalFlip", "Rotate", "Affine", "ShiftScaleRotate", "ColorJitter", "ToGray", "RandomBrightnessContrast",
    "GaussianBlur", "GaussNoise", "Blur", "Sharpen", "Equalize", "CLAHE", "Perspective",
})
# Albumentations modules offered in the form, as its groups
GROUPS = {"geometric": "Flips, rotation & geometry", "crops": "Crops", "pixel": "Colour, light & noise", "blur": "Blur", "dropout": "Dropout & erasing"}
# not for single-image detection/segmentation training: they need extra inputs or change the data type
EXCLUDED = frozenset({"Lambda", "Normalize", "ToFloat", "FromFloat", "ToRGB", "NoOp", "FDA", "HistogramMatching", "PixelDistributionAdaptation", "Mosaic", "OverlayElements", "TextImage", "RandomCropNearBBox", "CropNonEmptyMaskIfExists", "MaskDropout"})
# parameters most runs leave alone: folded away in the form
ADVANCED = frozenset({"interpolation", "mask_interpolation", "border_mode", "fill", "fill_mask", "rotate_method", "area_for_downscale", "keep_ratio", "balanced_scale", "fit_output", "noise_scale_factor", "approximate", "same_dxdy", "normalized", "pad_mode", "pad_position", "pad_if_needed", "position"})
# the OpenCV constants behind some choices
_CV_NAMES = {
    "interpolation": {0: "nearest", 1: "linear", 2: "cubic", 3: "area", 4: "lanczos4"},
    "mask_interpolation": {0: "nearest", 1: "linear", 2: "cubic", 3: "area", 4: "lanczos4"},
    "border_mode": {0: "constant", 1: "replicate", 2: "reflect", 3: "wrap", 4: "reflect 101"},
}


def _albumentations():
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # no version check over the network on import
    libraries.load("cv2", "albumentations")
    import albumentations

    return albumentations


def available() -> bool:
    import importlib.util

    return importlib.util.find_spec("albumentations") is not None


def _jsonable(value):
    try:
        return json.loads(json.dumps(value, default=lambda item: list(item) if isinstance(item, tuple) else str(item)))
    except (TypeError, ValueError):
        return str(value)


def _choices(cls, name: str) -> list | None:
    schema = getattr(cls, "InitSchema", None)
    info = getattr(schema, "model_fields", {}).get(name) if schema else None
    if info is None:
        return None
    options: list = []
    pending = [info.annotation]
    while pending:
        annotation = pending.pop()
        origin = typing.get_origin(annotation)
        if origin is typing.Literal:
            options.extend(typing.get_args(annotation))
        elif origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
            pending.extend(typing.get_args(annotation))
        elif annotation is type(None):
            options.append(None)
        else:
            return None
    return options or None


def _summary(cls) -> str:
    text = inspect.getdoc(cls) or ""
    return " ".join(text.split("\n\n")[0].split())


@cache
def catalog() -> dict:
    """The transforms the form offers: [{name, group, kind, summary, params: [{name, default, choices?, required?, advanced?}]}].

    kind: `geometric` (RF-DETR moves boxes and masks), `pixel` (only pixels change), or `unmoved` (it moves the
    image but RF-DETR leaves the boxes where they were)."""
    A = _albumentations()
    items = []
    for name in sorted(dir(A)):
        cls = getattr(A, name)
        if not inspect.isclass(cls) or not issubclass(cls, A.BasicTransform) or inspect.isabstract(cls) or name in EXCLUDED:
            continue
        group = cls.__module__.split(".")[2] if cls.__module__.count(".") >= 2 else ""
        if group not in GROUPS:
            continue
        params = []
        for key, parameter in inspect.signature(cls).parameters.items():
            if key in ("p", "strict", "always_apply") or parameter.kind in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL):
                continue
            entry: dict = {"name": key, "default": None if parameter.default is inspect.Parameter.empty else _jsonable(parameter.default)}
            if parameter.default is inspect.Parameter.empty:
                entry["required"] = True
            choices = _choices(cls, key)
            if choices and all(isinstance(choice, (str, int, float, bool)) or choice is None for choice in choices):
                labels = _CV_NAMES.get(key, {})
                entry["choices"] = [[choice, labels.get(choice, "none" if choice is None else str(choice))] for choice in choices]
            if key in ADVANCED:
                entry["advanced"] = True
            params.append(entry)
        moves = issubclass(cls, A.DualTransform) and group in ("geometric", "crops") and name != "Morphological"
        kind = "geometric" if name in GEOMETRIC else "unmoved" if moves else "pixel"
        default_p = inspect.signature(cls).parameters.get("p")
        items.append({"name": name, "group": group, "kind": kind, "summary": _summary(cls), "p": default_p.default if default_p is not None else 0.5, "params": params, "kornia": name in KORNIA})
    return {"groups": GROUPS, "transforms": items}


def _known() -> dict[str, dict]:
    return {item["name"]: item for item in catalog()["transforms"]}


# ---------- building (as rfdetr.datasets.transforms._build_albu_transform) ----------


class _UnknownParameter(ValueError):
    pass


def _build(name: str, params, strict: bool = True):
    A = _albumentations()
    if name in CONTAINERS:
        if isinstance(params, list):
            params = {"transforms": params}
        if not isinstance(params, dict):
            raise ValueError(f"the parameters must be a mapping, not {type(params).__name__}")
        nested = params.get("transforms", [])
        if not isinstance(nested, list):
            raise ValueError("`transforms` must be a list of single-key mappings")
        built = []
        for entry in nested:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(f"each nested transform must be a single-key mapping, not {entry!r}")
            nested_name, nested_params = next(iter(entry.items()))
            if not isinstance(nested_params, dict):
                raise ValueError(f"the parameters of {nested_name} must be a mapping")
            built.append(_build(str(nested_name), nested_params, strict))
        other = {key: value for key, value in params.items() if key != "transforms"}
        if name in ("OneOf", "Sequential"):
            other.setdefault("p", 1.0)
            if name == "OneOf" and not built:
                raise ValueError("OneOf needs at least one transform")
        return getattr(A, name)(transforms=built, **other)
    if not isinstance(params, dict):
        raise ValueError(f"the parameters must be a mapping, not {type(params).__name__}")
    cls = getattr(A, name, None)
    if not inspect.isclass(cls) or not issubclass(cls, A.BasicTransform):
        raise KeyError(name)
    params = dict(params)
    if name == "RandomSizedCrop" and "size" not in params and {"height", "width"} <= set(params) and "size" in inspect.signature(cls).parameters:
        params["size"] = (params.pop("height"), params.pop("width"))
    if strict:
        # Albumentations only warns about an argument it does not know, and then ignores it
        signature = inspect.signature(cls).parameters
        unknown = [key for key in params if key not in signature and key not in ("p", "always_apply")]
        if unknown and not any(parameter.kind is parameter.VAR_KEYWORD for parameter in signature.values()):
            close = difflib.get_close_matches(unknown[0], [key for key in signature if key != "strict"], n=1)
            raise _UnknownParameter(f"{name} has no parameter `{unknown[0]}`" + (f" (did you mean `{close[0]}`?)" if close else "") + "; Albumentations would ignore it")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(**params)


def _error_text(exc: Exception) -> str:
    """One line from an Albumentations error; its pydantic report (`1 validation error for InitSchema`, then a field
    line and an indented message per problem, one per alternative of a union) becomes `field: message; …`."""
    lines = str(exc).splitlines()
    if len(lines) > 1 and "validation error" in lines[0]:
        problems: dict[str, str] = {}
        for where, message in zip(lines[1:], lines[2:]):
            if where.startswith(" ") or not message.startswith(" "):
                continue
            field = where.split(".")[0].strip()
            problems.setdefault(field, message.strip().split(" [type=")[0])
        if problems:
            return "; ".join(f"`{field}`: {message}" for field, message in problems.items())
    return lines[0] if lines else type(exc).__name__


def _geometric(name: str, params) -> bool:
    if name in CONTAINERS:
        nested = params if isinstance(params, list) else params.get("transforms", []) if isinstance(params, dict) else []
        return any(isinstance(entry, dict) and any(_geometric(str(key), value) for key, value in entry.items()) for entry in nested)
    return name in GEOMETRIC


def _names(name: str, params) -> list[str]:
    """The leaf transforms of an entry."""
    if name in CONTAINERS:
        nested = params if isinstance(params, list) else params.get("transforms", []) if isinstance(params, dict) else []
        return [leaf for entry in nested if isinstance(entry, dict) for key, value in entry.items() for leaf in _names(str(key), value)]
    return [name]


def check(aug_config: dict, backend: str | None = None) -> tuple[list[tuple[str | None, str]], list[tuple[str | None, str]]]:
    """Errors and warnings for an aug_config, each (transform name or None, message): what RF-DETR would skip
    (an unknown transform, bad parameters), transforms whose boxes would not follow the image, and transforms the
    GPU (Kornia) augmentation backend cannot run."""
    errors: list[tuple[str | None, str]] = []
    warns: list[tuple[str | None, str]] = []
    known = _known()
    for name, params in aug_config.items():
        name = str(name)
        try:
            _build(name, params)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else name
            close = difflib.get_close_matches(str(missing), list(known) + list(CONTAINERS), n=1)
            errors.append((name, f"aug_config: `{missing}` is not an Albumentations transform" + (f" (did you mean `{close[0]}`?)" if close else "")))
            continue
        except _UnknownParameter as exc:
            errors.append((name, f"aug_config: {exc}"))
            continue
        except Exception as exc:  # noqa: BLE001 - whatever the transform rejects, RF-DETR would skip it
            errors.append((name, f"aug_config: {name} — {_error_text(exc)} (RF-DETR would skip it)"))
            continue
        for leaf in _names(name, params):
            if known.get(leaf, {}).get("kind") == "unmoved":
                warns.append((name, f"aug_config: {leaf} moves the image, but RF-DETR does not move the labels with it"))
    if aug_config and backend in ("kornia", "gpu", "auto"):
        unsupported = sorted({leaf for name, params in aug_config.items() for leaf in _names(str(name), params) if leaf not in KORNIA and (leaf in known or leaf in CONTAINERS)})
        if unsupported:
            message = f"augmentation_backend: {backend} runs augmentations with Kornia on the GPU, which cannot run {', '.join(unsupported)}"
            if backend == "auto":
                warns.append((None, message + " (when a CUDA GPU is present); use cpu"))
            else:
                errors.append((None, message + "; use cpu"))
    return errors, warns


# ---------- previews ----------


def _work_size(resolution: int) -> int:
    return max(32, min(int(resolution), WORK_SIZE))


def _square(sample: Sample, category: str, work: int):
    """The image as RF-DETR's square resize gives it (RGB, work×work) with its boxes (pixels, x1y1x2y2), classes and masks."""
    import cv2

    image = _read_bgr(sample.path, max(work, 1))
    image = cv2.cvtColor(cv2.resize(image, (work, work), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    boxes, classes, masks = [], [], []
    for annotation in sample.annotations:
        if not _valid(annotation, category):
            continue
        points = annotation.points
        if category == "segmentation":
            polygon = np.array(points, dtype=np.float64).reshape(-1, 2) * work
            mask = np.zeros((work, work), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
            box = [polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max()]
            masks.append(mask)
        else:
            cx, cy, w, h = points
            box = [(cx - w / 2) * work, (cy - h / 2) * work, (cx + w / 2) * work, (cy + h / 2) * work]
        box = [min(max(value, 0.0), float(work)) for value in box]
        if box[2] - box[0] < 1e-3 or box[3] - box[1] < 1e-3:
            if category == "segmentation":
                masks.pop()
            continue
        boxes.append(box)
        classes.append(annotation.class_id)
    return image, boxes, classes, masks if category == "segmentation" else None


def _apply(entries: list[tuple[str, object]], image, boxes, classes, masks, seed: int):
    """The transforms one after another, as RF-DETR wraps them: geometric ones carry boxes (and masks), others only pixels."""
    A = _albumentations()
    for step, (name, params) in enumerate(entries):
        transform = _build(name, params, strict=False)
        if _geometric(name, params):
            compose = A.Compose([transform], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["idxs"], min_visibility=0.0, clip=True), seed=seed * 101 + step)
            data = {"image": image, "bboxes": np.array(boxes, dtype=np.float32).reshape(-1, 4), "idxs": list(range(len(boxes)))}
            if masks:
                data["masks"] = masks
            result = compose(**data)
            keep = [int(index) for index in result["idxs"]]
            image = result["image"]
            boxes = [list(map(float, box)) for box in result["bboxes"]]
            classes = [classes[index] for index in keep]
            if masks:
                masks = [np.asarray(result["masks"][index]) for index in keep]
        else:
            image = A.Compose([transform], seed=seed * 101 + step)(image=image)["image"]
    return image, boxes, classes, masks


def _draw(image, boxes, classes, masks, size: int, draw_labels: bool, color: Callable[[int], str]) -> Image.Image:
    import cv2

    picture = Image.fromarray(np.ascontiguousarray(image))
    height, width = image.shape[:2]
    factor = size / max(height, width)
    picture = picture.resize((max(1, round(width * factor)), max(1, round(height * factor))), Image.Resampling.LANCZOS)
    if draw_labels:
        draw = ImageDraw.Draw(picture)
        line = max(1, round(size / 160))
        if masks is not None:
            for class_id, mask in zip(classes, masks):
                contours, _ = cv2.findContours((np.asarray(mask) > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    points = [(float(x) * factor, float(y) * factor) for x, y in contour.reshape(-1, 2)]
                    if len(points) >= 2:
                        draw.line(points + points[:1], fill=color(int(class_id)), width=line)
        else:
            for class_id, box in zip(classes, boxes):
                draw.rectangle(tuple(value * factor for value in box), outline=color(int(class_id)), width=line)
    return picture


def parse_config(text: str) -> dict:
    """An aug_config from a preview request (JSON)."""
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise DatasetError("aug_config must be a JSON object") from exc
    if not isinstance(value, dict):
        raise DatasetError("aug_config must map transform names to their parameters")
    return value


def render(dataset: Dataset, aug_config: dict | None, index: int | None, sample: int, seed: int, resolution: int, size: int, draw_labels: bool, color: Callable[[int], str]) -> Preview:
    """The `sample`-th preview image after one transform (`index`, always applied) or the whole pipeline (`index`
    None, each transform at its own chance); no aug_config is RF-DETR's default flip. index -1 is the image as is."""
    category = dataset.category
    samples, total = sample_pool(dataset, sample, partners=0)
    work = _work_size(resolution)
    config = DEFAULT_EQUIVALENT if aug_config is None else aug_config
    entries = [(str(name), params) for name, params in config.items()]
    if index is not None and index >= 0:
        if index >= len(entries):
            raise DatasetError("No such augmentation")
        name, params = entries[index]
        params = copy.deepcopy(params)
        if isinstance(params, dict):
            params["p"] = 1.0
        entries = [(name, params)]
    elif index is not None:
        entries = []
    image, boxes, classes, masks = _square(samples[0], category, work)
    before = len(classes)
    try:
        image, boxes, classes, masks = _apply(entries, image, boxes, classes, masks, seed)
    except KeyError as exc:
        raise DatasetError(f"`{exc.args[0] if exc.args else ''}` is not an Albumentations transform") from exc
    except Exception as exc:  # noqa: BLE001 - shown on the tile
        raise DatasetError(_error_text(exc)) from exc
    if index == -1:
        note = f"{samples[0].path.name} · {sample % total + 1} of {total}"
    else:
        note = f"{len(classes)} of {before} objects kept" if len(classes) != before else ""
        if image.shape[0] != work or image.shape[1] != work:
            note = " · ".join(part for part in (note, f"{image.shape[1]}×{image.shape[0]} px") if part)
    return Preview(_jpeg(_draw(image, boxes, classes, masks, size, draw_labels, color), size), note)
