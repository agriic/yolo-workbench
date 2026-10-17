"""Training frameworks: what differs between an Ultralytics YOLO run and an RF-DETR run.

`training.py` owns runs, the queue and the dataset lock; it asks a `Framework` for everything that
depends on the trainer: the arguments (schema, defaults, validation), the pretrained models, where a
run keeps its weights, how to resume it, and how to read its results. A run's framework is in its
`run.json`/`spec.json` (`framework`; absent = Ultralytics, so older runs read as before); for new
arguments it follows `model:` (an RF-DETR variant such as `rfdetr-medium`, else Ultralytics).

Nothing here imports torch, ultralytics or rfdetr: the server validates RF-DETR arguments against
the schema below (rfdetr 1.11's TrainConfig, checked by a test), and the runner validates them again
with RF-DETR's own pydantic model before training.
"""

from __future__ import annotations

import csv
import difflib
import importlib.util
import json
import os
import re
import zipfile
from pathlib import Path

import yaml

from . import libraries
from .dataset import DatasetError
from .fsutil import replace

TASKS = {"detection": "detect", "segmentation": "segment", "classification": "classify"}


# ---------- small file helpers (shared with training.py) ----------


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    replace(temporary, path)


def _read_yaml(path: Path):
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _key_line(text: str, key: str) -> int | None:
    for number, line in enumerate(text.splitlines(), 1):
        if re.match(rf"^\s*{re.escape(key)}\s*:", line):
            return number
    return None


def args_to_yaml(args: dict) -> str:
    """One `key: value` line per argument (lists inline, like args.yaml)."""
    lines = []
    for key, value in args.items():
        if isinstance(value, list) and all(not isinstance(item, (list, dict)) for item in value):
            lines.append(f"{key}: {json.dumps(value)}\n")
        else:
            lines.append(yaml.safe_dump({key: value}, sort_keys=False, default_flow_style=False, allow_unicode=True))
    return "".join(lines)


def _parse_yaml_args(text: str) -> tuple[dict | None, list[dict]]:
    """The raw YAML as a mapping, or the errors that stop it from being one."""
    try:
        data = yaml.safe_load(text) if text.strip() else {}
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        return None, [{"line": mark.line + 1 if mark else None, "message": f"Invalid YAML: {getattr(exc, 'problem', None) or exc}"}]
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return None, [{"line": 1, "message": "Arguments must be `key: value` lines"}]
    return data, []


def _device_problem(device) -> str | None:
    if isinstance(device, list) and len(device) > 1 or isinstance(device, str) and "," in device:
        return "Multi-GPU training is not supported here (no live progress with DDP); pick one device"
    return None


def _ordered(args: dict, first: tuple[str, ...]) -> dict:
    ordered = {key: args[key] for key in first if key in args}
    ordered.update((key, value) for key, value in args.items() if key not in ordered)
    return ordered


def _inside(path_text, directory: Path) -> bool:
    """Whether a model/weights path names a file inside `directory` (whatever separators it uses)."""
    if not path_text:
        return False
    try:
        return Path(str(path_text)).expanduser().resolve().is_relative_to(directory.resolve())
    except (OSError, RuntimeError, ValueError):
        return False


class Framework:
    id = ""
    label = ""
    categories: tuple[str, ...] = ()
    managed: tuple[str, ...] = ()
    weight_keys: tuple[str, ...] = ("model",)  # arguments that may name a previous run's weights

    def unavailable(self) -> str | None:
        """Why this framework cannot train here (None: it can)."""
        return None

    def owns(self, model) -> bool:
        raise NotImplementedError

    def defaults(self, category: str) -> dict:
        raise NotImplementedError

    def help(self) -> dict:
        raise NotImplementedError

    def validate(self, category: str, data: dict, text: str) -> dict:
        raise NotImplementedError

    def pretrained(self, category: str, weights_dir: Path) -> list[dict]:
        raise NotImplementedError

    def weights(self, run_dir: Path) -> dict[str, Path]:
        """The run's weights files that exist: {"best": …, "last": …}."""
        raise NotImplementedError

    def resume_checkpoint(self, run_dir: Path) -> Path | None:
        raise NotImplementedError

    def fitness(self, row: dict, category: str) -> float | None:
        raise NotImplementedError

    def results(self, run_dir: Path) -> list[dict]:
        """Epoch rows from the framework's own results file (runs without the runner's epochs.jsonl)."""
        raise NotImplementedError

    def run_args(self, run_dir: Path, spec: dict) -> dict:
        raise NotImplementedError

    def copy_args(self, args: dict, category: str) -> dict:
        """A run's arguments for a new run: managed keys and default values dropped."""
        raise NotImplementedError

    def base_model(self, args: dict) -> str:
        """What a run starts from, for run.json: the model, or the previous run's weights it fine-tunes."""
        return str(args.get("model") or "")

    def started_from(self, args: dict, run_dir: Path) -> bool:
        """Whether these arguments load weights from `run_dir` (fine-tuning from it)."""
        return any(_inside(args.get(key), run_dir) for key in self.weight_keys)


# ---------- Ultralytics ----------

MANAGED_KEYS = ("data", "project", "name", "exist_ok", "resume", "mode", "task", "save_dir")
MODEL_SUFFIX = {"detection": "", "segmentation": "-seg", "classification": "-cls"}
PRETRAINED_FAMILIES = (("yolo26", "YOLO26"), ("yolo11", "YOLO11"), ("yolov8", "YOLOv8"))
PRETRAINED_SIZES = "nsmlx"
PRETRAINED_PATTERN = re.compile(r"^(yolo(?:v8|11|26))([nsmlx])(-seg|-cls|-pose|-obb)?\.(pt|yaml)$")


def read_results_csv(path: Path) -> list[dict]:
    """Ultralytics' results.csv as rows of floats (older versions pad the header with spaces)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for raw in csv.DictReader(text.splitlines()):
        row = {}
        for key, value in raw.items():
            if key is None:
                continue
            number = _number(value)
            if number is not None:
                row[key.strip()] = number
        if "epoch" in row:
            row["epoch"] = int(row["epoch"])
            rows.append(row)
    return rows


def fitness(row: dict, category: str) -> float | None:
    """Ultralytics 8.4's fitness per task (used for runs without the runner's own fitness column)."""
    if category == "classification":
        top1, top5 = row.get("metrics/accuracy_top1"), row.get("metrics/accuracy_top5")
        return None if top1 is None or top5 is None else (top1 + top5) / 2
    box = row.get("metrics/mAP50-95(B)")
    if category == "segmentation":
        mask = row.get("metrics/mAP50-95(M)")
        return None if box is None or mask is None else box + mask
    return box


_defaults_cache: tuple[dict, dict] | None = None


def ultralytics_defaults() -> tuple[dict, dict]:
    """(defaults, help) from Ultralytics' cfg/default.yaml, read without importing ultralytics (or torch)."""
    global _defaults_cache
    if _defaults_cache is None:
        spec = importlib.util.find_spec("ultralytics")
        if spec is None or not spec.origin:
            raise DatasetError("ultralytics is not installed")
        path = Path(spec.origin).parent / "cfg" / "default.yaml"
        text = path.read_text(encoding="utf-8")
        defaults = yaml.safe_load(text) or {}
        help_text = {}
        for line in text.splitlines():
            match = re.match(r"^([a-z_0-9]+):[^#]*#\s*(.*)$", line)
            if match:
                help_text[match.group(1)] = match.group(2).strip()
        _defaults_cache = (defaults, help_text)
    return _defaults_cache


def _check_value(key: str, value) -> str | None:
    """Ultralytics' own type/range check for one argument (imported lazily; ~1 s once per process)."""
    try:
        libraries.load("ultralytics.cfg")
        from ultralytics.cfg import check_cfg
    except Exception:  # ultralytics unavailable or broken: the runner reports problems instead
        return None
    try:
        check_cfg({key: value}, hard=True)
    except (TypeError, ValueError) as exc:
        return str(exc).strip()
    return None


def validate_args(category: str, text: str) -> dict:
    """Parse raw YAML Ultralytics training arguments: {args, errors: [{line, message}], warnings}."""
    data, errors = _parse_yaml_args(text)
    if data is None:
        return {"args": {}, "errors": errors, "warnings": []}
    return ULTRALYTICS.validate(category, data, text)


class UltralyticsFramework(Framework):
    id = "ultralytics"
    label = "Ultralytics YOLO"
    categories = ("detection", "segmentation", "classification")
    managed = MANAGED_KEYS

    def unavailable(self) -> str | None:
        return None if importlib.util.find_spec("ultralytics") else "ultralytics is not installed"

    def owns(self, model) -> bool:
        return not RFDETR.owns(model)

    def defaults(self, category: str) -> dict:
        defaults, _ = ultralytics_defaults()
        return {key: value for key, value in defaults.items() if key not in MANAGED_KEYS}

    def help(self) -> dict:
        return ultralytics_defaults()[1]

    def validate(self, category: str, data: dict, text: str) -> dict:
        errors: list[dict] = []
        warnings: list[dict] = []
        defaults, _ = ultralytics_defaults()
        args: dict = {}
        for key, value in data.items():
            key = str(key)
            line = _key_line(text, key)
            if key in MANAGED_KEYS:
                errors.append({"line": line, "message": f"`{key}` is set by the workbench and cannot be changed"})
                continue
            if key not in defaults:
                close = difflib.get_close_matches(key, list(defaults), n=1)
                errors.append({"line": line, "message": f"Unknown argument `{key}`" + (f" (did you mean `{close[0]}`?)" if close else "")})
                continue
            problem = _check_value(key, value)
            if problem:
                errors.append({"line": line, "message": problem})
                continue
            args[key] = value
        cache = args.get("cache")
        if isinstance(cache, str) and cache.lower() == "disk":
            errors.append({"line": _key_line(text, "cache"), "message": "cache: disk writes .npy files next to every image in the dataset; use `ram` or `false`"})
        hue = args.get("hsv_h")
        if category == "classification" and isinstance(hue, (int, float)) and hue > 0.5:
            errors.append({"line": _key_line(text, "hsv_h"), "message": "hsv_h must be at most 0.5 for classification (torchvision's ColorJitter hue range)"})
        problem = _device_problem(args.get("device"))
        if problem:
            errors.append({"line": _key_line(text, "device"), "message": problem})
        model = args.get("model")
        if not model:
            errors.append({"line": _key_line(text, "model") or 1, "message": "`model` is required (a pretrained name such as yolo11n.pt, or a previous run's weights)"})
        else:
            model = str(model)
            args["model"] = model
            match = PRETRAINED_PATTERN.match(model)
            if match:
                suffix = match.group(3) or ""
                if suffix != MODEL_SUFFIX[category]:
                    errors.append({"line": _key_line(text, "model"), "message": f"{model} is not a {category} model; use a name like {match.group(1)}{match.group(2)}{MODEL_SUFFIX[category]}.pt"})
            elif "/" in model or os.sep in model:
                if not Path(model).expanduser().is_file():
                    errors.append({"line": _key_line(text, "model"), "message": f"Model file not found: {model}"})
                elif weights_framework(Path(model).expanduser()) == "rfdetr":
                    errors.append({"line": _key_line(text, "model"), "message": "These are RF-DETR weights: set `model` to their RF-DETR variant and `pretrain_weights` to the file"})
            elif not model.endswith((".pt", ".yaml", ".yml")):
                errors.append({"line": _key_line(text, "model"), "message": "`model` must be a .pt weights file or a model .yaml"})
        if args.get("time"):
            warnings.append({"line": _key_line(text, "time"), "message": "`time` overrides `epochs`: the epoch count changes while training"})
        return {"args": _ordered(args, ("model",)), "errors": errors, "warnings": warnings}

    def pretrained(self, category: str, weights_dir: Path) -> list[dict]:
        suffix = MODEL_SUFFIX[category]
        items = []
        for family, label in PRETRAINED_FAMILIES:
            for size in PRETRAINED_SIZES:
                name = f"{family}{size}{suffix}.pt"
                items.append({"name": name, "family": label, "size": size, "cached": (weights_dir / name).is_file()})
        return items

    def weights(self, run_dir: Path) -> dict[str, Path]:
        found = {kind: run_dir / "weights" / f"{kind}.pt" for kind in ("best", "last")}
        return {kind: path for kind, path in found.items() if path.is_file()}

    def resume_checkpoint(self, run_dir: Path) -> Path | None:
        path = run_dir / "weights" / "last.pt"
        return path if path.is_file() else None

    def fitness(self, row: dict, category: str) -> float | None:
        return fitness(row, category)

    def results(self, run_dir: Path) -> list[dict]:
        return read_results_csv(run_dir / "results.csv")

    def run_args(self, run_dir: Path, spec: dict) -> dict:
        return _read_yaml(run_dir / "args.yaml") or spec.get("args") or {}

    def copy_args(self, args: dict, category: str) -> dict:
        defaults, _ = ultralytics_defaults()
        kept = {key: value for key, value in args.items() if key not in MANAGED_KEYS and (key == "model" or key not in defaults or (defaults[key] != value and not (defaults[key] is None and value == "")))}
        kept = {key: value for key, value in kept.items() if key in defaults}  # keys from other Ultralytics versions
        return _ordered(kept, ("model",))


# ---------- RF-DETR ----------

# variant id (RF-DETR's own `size` names) -> class, default resolution, resolution step, pretrained weights
RFDETR_VARIANTS: dict[str, dict] = {
    "rfdetr-nano": {"class": "RFDETRNano", "label": "Nano", "category": "detection", "resolution": 384, "step": 32, "weights": "rf-detr-nano.pth"},
    "rfdetr-small": {"class": "RFDETRSmall", "label": "Small", "category": "detection", "resolution": 512, "step": 32, "weights": "rf-detr-small.pth"},
    "rfdetr-medium": {"class": "RFDETRMedium", "label": "Medium", "category": "detection", "resolution": 576, "step": 32, "weights": "rf-detr-medium.pth"},
    "rfdetr-large": {"class": "RFDETRLarge", "label": "Large", "category": "detection", "resolution": 704, "step": 32, "weights": "rf-detr-large-2026.pth"},
    "rfdetr-seg-nano": {"class": "RFDETRSegNano", "label": "Nano", "category": "segmentation", "resolution": 312, "step": 12, "weights": "rf-detr-seg-nano.pt"},
    "rfdetr-seg-small": {"class": "RFDETRSegSmall", "label": "Small", "category": "segmentation", "resolution": 384, "step": 24, "weights": "rf-detr-seg-small.pt"},
    "rfdetr-seg-medium": {"class": "RFDETRSegMedium", "label": "Medium", "category": "segmentation", "resolution": 432, "step": 24, "weights": "rf-detr-seg-medium.pt"},
    "rfdetr-seg-large": {"class": "RFDETRSegLarge", "label": "Large", "category": "segmentation", "resolution": 504, "step": 24, "weights": "rf-detr-seg-large.pt"},
    "rfdetr-seg-xlarge": {"class": "RFDETRSegXLarge", "label": "XLarge", "category": "segmentation", "resolution": 624, "step": 24, "weights": "rf-detr-seg-xlarge.pt"},
    "rfdetr-seg-2xlarge": {"class": "RFDETRSeg2XLarge", "label": "2XLarge", "category": "segmentation", "resolution": 768, "step": 24, "weights": "rf-detr-seg-xxlarge.pt"},
}
RFDETR_WEIGHTS_SUBDIR = "rfdetr"  # RF_HOME under the workbench weights cache

# The TrainConfig fields of rfdetr 1.11 (plus the segmentation extras), as (type, default, help).
# Types: int, float, bool, str, path, dict, list, "int?" (or null), "float?", "bool?", "batch" (int or auto),
# "choice:a|b|…" (a null choice is written `null`), "any".
_RF_FIELDS: dict[str, tuple[str, object, str]] = {
    "epochs": ("int", 100, "Number of epochs to train"),
    "batch_size": ("batch", 4, "Images per step; `auto` probes the largest batch that fits (CUDA only)"),
    "grad_accum_steps": ("int", 1, "Gradient accumulation steps; effective batch = batch_size × grad_accum_steps"),
    "lr": ("float", 1e-4, "Learning rate of the decoder and heads (AdamW)"),
    "lr_encoder": ("float", 1.5e-4, "Learning rate of the DINOv2 backbone"),
    "weight_decay": ("float", 1e-4, "AdamW weight decay"),
    "warmup_epochs": ("float", 0.0, "Linear learning-rate warmup, in epochs"),
    "lr_vit_layer_decay": ("float", 0.8, "Per-layer learning-rate decay inside the backbone"),
    "lr_component_decay": ("float", 0.7, "Learning-rate decay between model components"),
    "lr_scheduler": ("str", "step", "`step` or `cosine` (or a dotted path to a scheduler class)"),
    "lr_scheduler_kwargs": ("dict", {}, "Scheduler options (`lr_drop`, `min_factor` for the built-in ones)"),
    "lr_scheduler_interval": ("choice:step|epoch", "step", "Step the scheduler every optimizer step or every epoch"),
    "lr_scheduler_monitor": ("str", "val/loss", "Metric a ReduceLROnPlateau scheduler watches"),
    "optimizer": ("str", "adamw", "A torch.optim class name, or a dotted path"),
    "optimizer_kwargs": ("dict", {}, "Extra optimizer options"),
    "clip_max_norm": ("float", 0.1, "Gradient clipping norm"),
    "drop_path": ("float", 0.0, "Stochastic depth rate in the backbone"),
    "cls_loss_coef": ("float", 1.0, "Weight of the classification loss"),
    "mask_point_sample_ratio": ("int", 16, "Points sampled per mask for the mask losses (segmentation)"),
    "mask_ce_loss_coef": ("float", 5.0, "Weight of the mask cross-entropy loss (segmentation)"),
    "mask_dice_loss_coef": ("float", 5.0, "Weight of the mask dice loss (segmentation)"),
    "use_ema": ("bool", True, "Keep an exponential moving average of the weights (validated and saved as best)"),
    "ema_decay": ("float", 0.993, "EMA decay"),
    "ema_tau": ("int", 100, "EMA warmup steps"),
    "ema_update_interval": ("int", 1, "Optimizer steps between EMA updates"),
    "eval_base_model": ("bool", False, "Also validate the raw (non-EMA) weights every epoch"),
    "eval_ema_only": ("bool", False, "Deprecated: use eval_base_model"),
    "early_stopping": ("bool", False, "Stop when the validation mAP stops improving"),
    "early_stopping_patience": ("int", 10, "Epochs without improvement before early stopping"),
    "early_stopping_min_delta": ("float", 0.001, "Smallest mAP gain that counts as an improvement"),
    "early_stopping_use_ema": ("bool", False, "Early stopping watches the EMA metric"),
    "skip_best_epochs": ("int", 0, "Ignore the first N epochs when picking the best weights and for early stopping"),
    "smooth_alpha": ("float", 0.0, "Smooth the metric that picks the best weights (0 = off)"),
    "best_model_metric": ("choice:map|mar", "map", "Metric that picks the best weights: mAP or mAR"),
    "checkpoint_interval": ("int", 10, "Save a full checkpoint every N epochs"),
    "eval_interval": ("int", 1, "Validate every N epochs (the last epoch always validates)"),
    "eval_batch_size": ("int?", None, "Validation batch size (default: batch_size)"),
    "eval_max_dets": ("int", 500, "Detections per image kept for evaluation"),
    "eval_backend": ("choice:vernier|hotcoco|faster_coco_eval|ufcoco", "vernier", "COCO evaluation backend"),
    "eval_masks_head_resolution": ("bool", False, "Evaluate masks at the mask head's resolution (faster, less exact)"),
    "log_per_class_metrics": ("bool", False, "Log AP per class every epoch (the final evaluation always does)"),
    "compute_val_loss": ("choice:auto|true|false", "auto", "Compute the validation loss (auto: when something uses it)"),
    "compute_test_loss": ("bool", True, "Compute the test loss in the test evaluation"),
    "compute_train_metrics": ("bool", False, "Compute mAP on training batches too (slow)"),
    "compact_train_metrics": ("bool", True, "Fold per-layer auxiliary losses into one value each"),
    "train_log_on_step": ("bool", False, "Log the train loss every step, not only per epoch"),
    "multi_scale": ("choice:per-batch|per-sample|off|true|false", "per-batch", "Train at random scales around the resolution"),
    "expanded_scales": ("bool", True, "Use the wider set of training scales"),
    "scale_jitter": ("bool", True, "Jitter the training scales"),
    "square_resize_div_64": ("bool", True, "Resize to a square divisible by 64"),
    "aug_config": ("dict", None, "Custom Albumentations augmentations (needs rfdetr[augment])"),
    "augmentation_backend": ("choice:cpu|auto|torchvision|albumentations|kornia|gpu|tv|albu", "cpu", "Where augmentations run"),
    "save_dataset_grids": ("bool", False, "Save image grids of the augmented train/val batches into the run folder"),
    "amp_dtype": ("choice:auto|bf16|fp16|fp8|null", "auto", "Mixed precision (null = full FP32; fp8 needs an NVIDIA GPU)"),
    "fp16_eval": ("bool", False, "Deprecated, no effect"),
    "num_workers": ("int", 2, "Dataloader workers"),
    "pin_memory": ("bool?", None, "Pin dataloader memory (default: on with CUDA)"),
    "persistent_workers": ("bool?", None, "Keep dataloader workers between epochs"),
    "prefetch_factor": ("int?", None, "Batches each worker loads ahead"),
    "pack_targets": ("bool", True, "Pack targets into one tensor per batch"),
    "pad_targets_to": ("int?", None, "Pad targets to a fixed count per image"),
    "auto_batch_target_effective": ("int", 16, "Effective batch that batch_size: auto aims for"),
    "auto_batch_max_targets_per_image": ("int", 100, "Objects per image assumed by batch_size: auto"),
    "auto_batch_ema_headroom": ("float", 0.7, "Share of GPU memory batch_size: auto may use"),
    "num_sanity_val_steps": ("int", 0, "Validation batches run before training starts"),
    "seed": ("int?", None, "Random seed"),
}
# Model (architecture) arguments the workbench accepts besides `model` and `device`
_RF_MODEL_FIELDS: dict[str, tuple[str, object, str]] = {
    "pretrain_weights": ("path", None, "Start from these weights (a previous RF-DETR run) instead of the variant's pretrained ones"),
    "resolution": ("int?", None, "Input size; a multiple of the variant's step (default: the variant's own)"),
    "device": ("any", None, "cpu, mps, or a GPU index"),
}
_RF_SEGMENTATION_ONLY = {"mask_point_sample_ratio", "mask_ce_loss_coef", "mask_dice_loss_coef"}
# set by the runner (or meaningless here: keypoints, loggers, multi-GPU)
RFDETR_MANAGED = (
    "dataset_dir", "output_dir", "dataset_file", "resume", "class_names", "run_test", "devices", "strategy", "num_nodes",
    "accelerator", "sync_bn", "train_log_sync_dist", "progress_bar", "tensorboard", "wandb", "mlflow", "clearml",
    "project", "run", "notes", "dont_save_weights",
)
_RF_UNSUPPORTED = {"keypoint_flip_pairs", "keypoint_l1_loss_coef", "keypoint_findable_loss_coef", "keypoint_visible_loss_coef", "keypoint_nll_loss_coef", "keypoint_oks_sigmas"}

# RF-DETR's logged names -> the workbench's epoch-row names (the Ultralytics ones where they mean the same)
_RF_METRICS = {"mAP_50_95": "mAP50-95", "mAP_50": "mAP50", "mAP_75": "mAP75", "mAR": "mAR", "precision": "precision", "recall": "recall", "F1": "F1"}
_RF_LOSSES = {"loss": "loss", "loss_ce": "ce_loss", "loss_bbox": "bbox_loss", "loss_giou": "giou_loss", "loss_mask_ce": "mask_ce_loss", "loss_mask_dice": "mask_dice_loss"}
# the progress bar's running values during an epoch
RFDETR_RUNNING_LOSSES = {"loss": "train/loss", "loss_cls": "train/ce_loss", "loss_box": "train/bbox_loss", "loss_giou": "train/giou_loss", "mask_ce": "train/mask_ce_loss", "mask_dice": "train/mask_dice_loss"}
RFDETR_BEST = ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth")  # total exists once the fit ended


def normalize_rfdetr_metrics(raw: dict, prefix: str = "val") -> dict:
    """RF-DETR's logged metrics as epoch-row keys: `metrics/mAP50-95(B)`, `train/ce_loss`, `lr/pg0`, …

    `<prefix>/` holds the evaluation metrics (val, or test for the test evaluation). EMA values
    (`ema/…`) are kept only when they differ from the plain ones, which mirror EMA by default.
    Diagnostics (cardinality, auxiliary per-layer losses) are left out; metrics.csv keeps them.
    """
    out: dict = {}
    for key, value in raw.items():
        number = _number(value)
        if number is None:
            continue
        group, _, name = str(key).partition("/")
        if group == prefix:
            ema = name.startswith("ema_")
            name = name[4:] if ema else name
            mask = name.startswith("segm_")
            name = name[5:] if mask else name
            if name in _RF_METRICS:
                out[f"{'ema' if ema else 'metrics'}/{_RF_METRICS[name]}({'M' if mask else 'B'})"] = number
            elif name in _RF_LOSSES and not ema and not mask and prefix == "val":
                out[f"val/{_RF_LOSSES[name]}"] = number
        elif group == "train":
            if name in _RF_LOSSES:
                out[f"train/{_RF_LOSSES[name]}"] = number
            elif name == "lr":
                out["lr/pg0"] = number
    for key in [key for key in out if key.startswith("ema/")]:
        plain = out.get("metrics/" + key[4:])
        if plain is not None and abs(plain - out[key]) < 1e-9:
            del out[key]
    return out


def rfdetr_class_rows(raw: dict, prefix: str, instances: dict[str, int] | None = None) -> list[dict]:
    """Per-class AP (`<prefix>/AP/<class>`) as result-table rows."""
    rows = []
    marker = f"{prefix}/AP/"
    for key, value in raw.items():
        if str(key).startswith(marker):
            name = str(key)[len(marker):]
            row = {"Class": name}
            if instances is not None:
                row["Instances"] = instances.get(name, 0)
            row["mAP50-95"] = _number(value)
            rows.append(row)
    return rows


def rfdetr_fitness(row: dict, category: str, metric: str = "map") -> float | None:
    """What picks RF-DETR's best weights: box (detection) or mask (segmentation) mAP50-95, or mAR; the better of plain and EMA."""
    name = "mAR(B)" if metric == "mar" else "mAP50-95(M)" if category == "segmentation" else "mAP50-95(B)"
    values = [value for value in (row.get(f"metrics/{name}"), row.get(f"ema/{name}")) if value is not None]
    return max(values) if values else None


def read_rfdetr_metrics_csv(path: Path, category: str) -> list[dict]:
    """Lightning's metrics.csv (one row per log call) folded into one row per epoch, 1-based like Ultralytics."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    by_epoch: dict[int, dict] = {}
    for raw in csv.DictReader(text.splitlines()):
        epoch = _number(raw.get("epoch"))
        if epoch is None:
            continue
        merged = by_epoch.setdefault(int(epoch), {})
        merged.update({key: value for key, value in raw.items() if key and value not in (None, "")})
    rows = []
    for epoch in sorted(by_epoch):
        row = {"epoch": epoch + 1, **normalize_rfdetr_metrics(by_epoch[epoch])}
        row["fitness"] = rfdetr_fitness(row, category)
        rows.append(row)
    return rows


def _rf_check(kind: str, value) -> str | None:
    """None when `value` fits the schema type, else what was expected."""
    if kind == "any":
        return None
    if kind.endswith("?"):
        if value is None:
            return None
        kind = kind[:-1]
    if kind.startswith("choice:"):
        choices = kind[7:].split("|")
        text = "null" if value is None else str(value).lower() if isinstance(value, bool) else str(value)
        return None if text in choices else f"one of {', '.join(choices)}"
    if kind == "int":
        return None if isinstance(value, int) and not isinstance(value, bool) else "an integer"
    if kind == "float":
        return None if isinstance(value, (int, float)) and not isinstance(value, bool) else "a number"
    if kind == "bool":
        return None if isinstance(value, bool) else "true or false"
    if kind in ("str", "path"):
        return None if isinstance(value, str) and value else "a text value"
    if kind == "dict":
        return None if isinstance(value, dict) or value is None else "a mapping (`key: value` pairs)"
    if kind == "list":
        return None if isinstance(value, list) else "a list"
    if kind == "batch":
        return None if value == "auto" or isinstance(value, int) and not isinstance(value, bool) and value >= 1 else "a positive integer or `auto`"
    return None


_RF_POSITIVE = {"epochs", "grad_accum_steps", "checkpoint_interval", "eval_interval", "ema_update_interval", "eval_batch_size", "auto_batch_target_effective", "auto_batch_max_targets_per_image", "prefetch_factor", "pad_targets_to", "eval_max_dets", "resolution"}
_RF_NON_NEGATIVE = {"lr", "lr_encoder", "weight_decay", "warmup_epochs", "clip_max_norm", "drop_path", "skip_best_epochs", "early_stopping_patience", "num_workers", "ema_tau", "num_sanity_val_steps", "early_stopping_min_delta"}


def rfdetr_device(device) -> str | None:
    """The workbench's device value as RF-DETR's: a GPU index becomes cuda:N."""
    if device is None or device == "":
        return None
    if isinstance(device, int) and not isinstance(device, bool) or isinstance(device, str) and device.isdigit():
        return f"cuda:{device}"
    return str(device)


class RFDETRFramework(Framework):
    id = "rfdetr"
    label = "RF-DETR"
    categories = ("detection", "segmentation")
    managed = RFDETR_MANAGED
    weight_keys = ("pretrain_weights",)

    def unavailable(self) -> str | None:
        if importlib.util.find_spec("rfdetr") is None:
            return 'rfdetr is not installed (uv add "rfdetr[train]")'
        if importlib.util.find_spec("pytorch_lightning") is None:
            return 'RF-DETR training needs its train extra (uv add "rfdetr[train]")'
        return None

    def owns(self, model) -> bool:
        # any rfdetr-… name, so a mistyped variant gets RF-DETR's error rather than Ultralytics'
        return isinstance(model, str) and model.strip().lower().startswith("rfdetr-")

    def fields(self, category: str) -> dict[str, tuple[str, object, str]]:
        fields = {key: value for key, value in _RF_FIELDS.items() if category == "segmentation" or key not in _RF_SEGMENTATION_ONLY}
        return {"model": ("str", None, "RF-DETR variant"), **_RF_MODEL_FIELDS, **fields}

    def defaults(self, category: str) -> dict:
        defaults = {key: default for key, (_, default, _) in self.fields(category).items()}
        # every periodic checkpoint is a full Lightning checkpoint (weights + optimizer, ~4× the weights):
        # the workbench writes none unless asked; last.ckpt (resume) is always kept
        defaults["checkpoint_interval"] = None
        return defaults

    def help(self) -> dict:
        help_text = {key: text for key, (_, _, text) in {**_RF_FIELDS, **_RF_MODEL_FIELDS}.items()}
        help_text["checkpoint_interval"] = "Keep a full checkpoint (weights + optimizer state) every N epochs; empty = only last.ckpt"
        return help_text

    def validate(self, category: str, data: dict, text: str) -> dict:
        errors: list[dict] = []
        warnings: list[dict] = []
        fields = self.fields(category)
        args: dict = {}
        for key, value in data.items():
            key = str(key)
            line = _key_line(text, key)
            if key in RFDETR_MANAGED:
                errors.append({"line": line, "message": f"`{key}` is set by the workbench and cannot be changed"})
                continue
            if key in _RF_UNSUPPORTED:
                errors.append({"line": line, "message": f"`{key}` is for keypoint models, which the workbench does not train"})
                continue
            if key in _RF_SEGMENTATION_ONLY and category != "segmentation":
                errors.append({"line": line, "message": f"`{key}` only applies to segmentation models"})
                continue
            if key not in fields:
                close = difflib.get_close_matches(key, list(fields), n=1)
                errors.append({"line": line, "message": f"Unknown RF-DETR argument `{key}`" + (f" (did you mean `{close[0]}`?)" if close else "")})
                continue
            kind = fields[key][0]
            expected = _rf_check(kind, value)
            if expected:
                errors.append({"line": line, "message": f"`{key}` must be {expected}"})
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key in _RF_POSITIVE and value < 1 or key in _RF_NON_NEGATIVE and value < 0:
                    errors.append({"line": line, "message": f"`{key}` must be {'at least 1' if key in _RF_POSITIVE else 'zero or more'}"})
                    continue
            args[key] = value
        model = args.get("model")
        variant = RFDETR_VARIANTS.get(str(model).strip().lower()) if model else None
        if not model:
            errors.append({"line": _key_line(text, "model") or 1, "message": "`model` is required (an RF-DETR variant such as rfdetr-medium)"})
        elif variant is None:
            errors.append({"line": _key_line(text, "model"), "message": f"Unknown RF-DETR variant `{model}`; use one of {', '.join(name for name, item in RFDETR_VARIANTS.items() if item['category'] == category)}"})
        else:
            args["model"] = str(model).strip().lower()
            if variant["category"] != category:
                errors.append({"line": _key_line(text, "model"), "message": f"{model} is a {variant['category']} model; this dataset needs a {category} variant ({'rfdetr-seg-…' if category == 'segmentation' else 'rfdetr-nano … rfdetr-large'})"})
            resolution = args.get("resolution")
            if isinstance(resolution, int) and resolution % variant["step"]:
                errors.append({"line": _key_line(text, "resolution"), "message": f"resolution must be a multiple of {variant['step']} for {model} (e.g. {resolution // variant['step'] * variant['step'] or variant['step']})"})
        weights = args.get("pretrain_weights")
        if weights is not None:
            if not Path(str(weights)).expanduser().is_file():
                errors.append({"line": _key_line(text, "pretrain_weights"), "message": f"Weights file not found: {weights}"})
            elif weights_framework(Path(str(weights)).expanduser()) != "rfdetr":
                errors.append({"line": _key_line(text, "pretrain_weights"), "message": "These are Ultralytics weights; RF-DETR can only start from RF-DETR weights"})
        device = args.get("device")
        problem = _device_problem(device)
        if problem:
            errors.append({"line": _key_line(text, "device"), "message": problem})
        on_cpu = str(device).lower() in ("cpu", "mps")
        if args.get("batch_size") == "auto" and on_cpu:
            errors.append({"line": _key_line(text, "batch_size"), "message": "batch_size: auto needs a CUDA GPU; set a number for CPU/MPS training"})
        if args.get("amp_dtype") == "fp8":
            if args.get("batch_size") == "auto":
                errors.append({"line": _key_line(text, "amp_dtype"), "message": "amp_dtype: fp8 cannot be combined with batch_size: auto"})
            if on_cpu:
                errors.append({"line": _key_line(text, "amp_dtype"), "message": "amp_dtype: fp8 needs an NVIDIA GPU"})
        if args.get("aug_config") is not None and importlib.util.find_spec("albumentations") is None:
            errors.append({"line": _key_line(text, "aug_config"), "message": 'aug_config needs Albumentations: install rfdetr[augment] (uv add "rfdetr[train,augment]")'})
        if args.get("eval_ema_only"):
            warnings.append({"line": _key_line(text, "eval_ema_only"), "message": "eval_ema_only is deprecated; use eval_base_model"})
        if args.get("fp16_eval"):
            warnings.append({"line": _key_line(text, "fp16_eval"), "message": "fp16_eval has no effect; use amp_dtype: fp16"})
        if args.get("checkpoint_interval") == 1:
            warnings.append({"line": _key_line(text, "checkpoint_interval"), "message": "checkpoint_interval: 1 keeps a full checkpoint of every epoch (several hundred MB each)"})
        return {"args": _ordered(args, ("model", "pretrain_weights")), "errors": errors, "warnings": warnings}

    def pretrained(self, category: str, weights_dir: Path) -> list[dict]:
        # where the runner downloads them: RF_HOME when the server was started with it, else the workbench cache
        root = Path(os.environ["RF_HOME"]).expanduser() if os.environ.get("RF_HOME") else weights_dir / RFDETR_WEIGHTS_SUBDIR
        return [
            {"name": name, "label": item["label"], "resolution": item["resolution"], "step": item["step"], "file": item["weights"], "cached": (root / item["weights"]).is_file()}
            for name, item in RFDETR_VARIANTS.items()
            if item["category"] == category
        ]

    def weights(self, run_dir: Path) -> dict[str, Path]:
        found: dict[str, Path] = {}
        best = next((run_dir / name for name in RFDETR_BEST if (run_dir / name).is_file()), None)
        if best is not None:
            found["best"] = best
        if (run_dir / "last_ema.pth").is_file():
            found["last"] = run_dir / "last_ema.pth"
        return found

    def resume_checkpoint(self, run_dir: Path) -> Path | None:
        if (run_dir / "last.ckpt").is_file():
            return run_dir / "last.ckpt"
        numbered = []
        for path in run_dir.glob("checkpoint_*.ckpt"):
            match = re.search(r"(\d+)", path.stem)
            if match:
                numbered.append((int(match.group(1)), path))
        return max(numbered)[1] if numbered else None

    def fitness(self, row: dict, category: str) -> float | None:
        return rfdetr_fitness(row, category)

    def results(self, run_dir: Path) -> list[dict]:
        meta = _read_json(run_dir / "training_config.json") or {}
        segmentation = bool((meta.get("model_config") or {}).get("segmentation_head"))
        return read_rfdetr_metrics_csv(run_dir / "metrics.csv", "segmentation" if segmentation else "detection")

    def run_args(self, run_dir: Path, spec: dict) -> dict:
        if spec.get("args"):
            return spec["args"]
        # a run made outside the app: its training_config.json, minus what is the default anyway
        meta = _read_json(run_dir / "training_config.json") or {}
        config = meta.get("train_config") or {}
        model_config = meta.get("model_config") or {}
        args = {key: value for key, value in config.items() if key in _RF_FIELDS and _RF_FIELDS[key][1] != value}
        name = str(model_config.get("model_name") or "")
        variant = next((key for key, item in RFDETR_VARIANTS.items() if item["class"] == name), None)
        if variant:
            args = {"model": variant, **args}
            if model_config.get("resolution") and model_config["resolution"] != RFDETR_VARIANTS[variant]["resolution"]:
                args["resolution"] = model_config["resolution"]
        return args

    def copy_args(self, args: dict, category: str) -> dict:
        defaults = self.defaults(category)
        kept = {key: value for key, value in args.items() if key not in RFDETR_MANAGED and key in defaults and (key == "model" or defaults[key] != value)}
        return _ordered(kept, ("model", "pretrain_weights"))

    def base_model(self, args: dict) -> str:
        return str(args.get("pretrain_weights") or args.get("model") or "")


ULTRALYTICS = UltralyticsFramework()
RFDETR = RFDETRFramework()
FRAMEWORKS: dict[str, Framework] = {framework.id: framework for framework in (ULTRALYTICS, RFDETR)}


def get(framework_id: str | None) -> Framework:
    return FRAMEWORKS.get(framework_id or "ultralytics", ULTRALYTICS)


def for_args(data: dict) -> Framework:
    return RFDETR if RFDETR.owns(data.get("model")) else ULTRALYTICS


def validate(category: str, text: str) -> dict:
    """Raw YAML training arguments, validated by the framework `model:` picks: {framework, args, errors, warnings}."""
    data, errors = _parse_yaml_args(text)
    if data is None:
        return {"framework": ULTRALYTICS.id, "args": {}, "errors": errors, "warnings": []}
    framework = for_args(data)
    if category not in framework.categories:
        result = {"args": {}, "errors": [{"line": _key_line(text, "model"), "message": f"{framework.label} does not train {category} models"}], "warnings": []}
    else:
        reason = framework.unavailable()
        result = framework.validate(category, data, text)
        if reason:
            result["errors"].insert(0, {"line": _key_line(text, "model"), "message": reason})
    return {"framework": framework.id, **result}


def of_run(run_dir: Path) -> Framework:
    """A run's framework: recorded by the workbench, else recognized from its files (runs made outside the app)."""
    for name in ("run.json", "spec.json"):
        recorded = (_read_json(run_dir / "workbench" / name) or {}).get("framework")
        if recorded:
            return get(recorded)
    if not (run_dir / "args.yaml").is_file() and (run_dir / "training_config.json").is_file():
        return RFDETR
    return ULTRALYTICS


def is_external_run(directory: Path) -> bool:
    """A run folder made outside the app: Ultralytics' args.yaml, or RF-DETR's training_config.json + metrics.csv."""
    return (directory / "args.yaml").is_file() or (directory / "training_config.json").is_file() and (directory / "metrics.csv").is_file()


def weights_framework(path: Path) -> str:
    """Which framework wrote a weights file, without unpickling it: Ultralytics pickles its model classes
    (`ultralytics.nn.tasks…`) into the checkpoint, RF-DETR only state dicts and plain metadata."""
    path = Path(path)
    for parent in (path.parent, path.parent.parent):
        recorded = (_read_json(parent / "workbench" / "run.json") or {}).get("framework")
        if recorded:
            return get(recorded).id
    if path.suffix.lower() == ".pth":
        return RFDETR.id
    try:
        with zipfile.ZipFile(path) as archive:
            pickled = next((name for name in archive.namelist() if name.endswith("data.pkl")), None)
            if pickled is not None and b"ultralytics" not in archive.read(pickled):
                return RFDETR.id
    except (OSError, zipfile.BadZipFile, KeyError):
        pass
    return ULTRALYTICS.id
