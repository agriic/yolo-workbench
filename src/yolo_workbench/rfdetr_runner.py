"""The RF-DETR side of the training subprocess: train_runner.py hands over runs whose spec.json says
`"framework": "rfdetr"`. It reports through the same workbench/ files, with the same `Reporter`.

1. **Dataset folder.** RF-DETR reads YOLO data only as split *folders* under one root (no file lists),
   pairs labels by file stem, and ignores EXIF orientation. So the runner builds one in
   `<run>/workbench/data/` from `splits.json`, the server's snapshot of the dataset taken at launch:
   images are hardlinked (symlinked or copied where that fails) under unique names, EXIF-rotated ones
   are re-encoded upright (workbench labels refer to the upright image), and labels are written from
   the parsed annotations. The folder is removed when the runner exits and rebuilt on resume.
2. **Training** goes through `RFDETR.train()`, so RF-DETR's own dataset-driven setup (class count,
   auto batch size) applies. `WorkbenchCallback` is appended to each Lightning trainer it builds:
   `rfdetr.training.build_trainer` is wrapped in this process.
3. **Final evaluation** of the best weights on val (per-class AP) and test, after a completed or
   early-finished fit.

Pause and Finish early set Lightning's `trainer.should_stop` at an epoch end; Lightning still saves
that epoch's last.ckpt, which a resume continues from. A kill (SIGTERM, or control.json on Windows)
only sets a flag; the callback raises at the next batch boundary, never while a checkpoint is being
written (Lightning's own SIGTERM handling also stops only at batch boundaries).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from pytorch_lightning import Callback

from .frameworks import (
    RFDETR,
    RFDETR_BEST,
    RFDETR_RUNNING_LOSSES,
    RFDETR_VARIANTS,
    RFDETR_WEIGHTS_SUBDIR,
    TASKS,
    normalize_rfdetr_metrics,
    rfdetr_class_rows,
    rfdetr_device,
    rfdetr_fitness,
)
from .fsutil import WINDOWS
from .imaging import orientation
from .train_runner import STATE_INTERVAL, Reporter, RunKilled, _number, write_json

VIEW_DIR = "data"
SPLIT_FOLDERS = {"train": "train", "val": "valid", "test": "test"}  # RF-DETR's Roboflow-style names
DEFAULT_EVAL_BATCH = 4


# ---------- the dataset folder ----------


def place_image(source: Path, target: Path) -> str:
    """Put one dataset image into the folder: "rotated" (re-encoded upright), "linked", or "copied"."""
    try:
        with Image.open(source) as image:
            if orientation(image) not in (0, 1):
                upright = ImageOps.exif_transpose(image)
                options = {"quality": 95} if image.format in ("JPEG", "WEBP") else {}
                upright.save(target, format=image.format, **options)  # saved without EXIF: upright as stored
                return "rotated"
    except (OSError, UnidentifiedImageError, ValueError):
        pass  # unreadable here: link it anyway, RF-DETR reports what it cannot read
    try:
        os.link(source, target)
        return "linked"
    except OSError:
        pass  # another volume, or a filesystem without hardlinks
    if not WINDOWS:  # Windows needs extra privileges for symlinks
        try:
            target.symlink_to(source)
            return "linked"
        except OSError:
            pass
    shutil.copy2(source, target)
    return "copied"


def label_text(annotations: list[list[float]]) -> str:
    lines = [" ".join([str(int(item[0]))] + [f"{value:.6f}" for value in item[1:]]) for item in annotations]
    return "".join(f"{line}\n" for line in lines)


def build_view(splits_path: Path, view: Path, reporter: Reporter | None = None) -> dict:
    """Write the RF-DETR dataset folder; returns {names, category, has_test, instances: {split: {class: n}}, placed: {how: n}}."""
    data = json.loads(splits_path.read_text(encoding="utf-8"))
    names: list[str] = data["names"]
    splits: dict = data["splits"]
    if view.exists():
        shutil.rmtree(view)
    total = sum(len(items) for items in splits.values())
    done = 0
    placed: Counter = Counter()
    instances: dict[str, Counter] = {}
    for split, items in splits.items():
        folder = view / SPLIT_FOLDERS[split]
        (folder / "images").mkdir(parents=True)
        (folder / "labels").mkdir(parents=True)
        counts: Counter = Counter()
        for index, (image, annotations) in enumerate(items):
            if reporter is not None and reporter.kill_requested:
                raise RunKilled()
            source = Path(image)
            # numbered: a split may span folders whose images share a name, and RF-DETR pairs labels by stem
            stem = f"{index:06d}_{source.stem}"
            placed[place_image(source, folder / "images" / f"{stem}{source.suffix.lower()}")] += 1
            (folder / "labels" / f"{stem}.txt").write_text(label_text(annotations), encoding="utf-8", newline="\n")
            counts.update(names[int(item[0])] for item in annotations if 0 <= int(item[0]) < len(names))
            done += 1
            if reporter is not None:
                reporter.write(prepared=done, prepare_total=total)
        instances[split] = counts
    config = {"path": ".", **{split: f"{SPLIT_FOLDERS[split]}/images" for split in splits}, "nc": len(names), "names": names}
    (view / "data.yaml").write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8", newline="\n")  # JSON is YAML
    return {"names": names, "category": data.get("category", "detection"), "has_test": bool(splits.get("test")), "instances": {split: dict(counts) for split, counts in instances.items()}, "placed": dict(placed)}


# ---------- progress ----------


def _gpu_memory() -> float | None:
    try:
        import torch

        return torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else None
    except Exception:
        return None


def _batches(trainer) -> int:
    count = getattr(trainer, "num_training_batches", 0)
    return int(count) if isinstance(count, (int, float)) and math.isfinite(count) else 0


class WorkbenchCallback(Callback):
    """Mirrors an RF-DETR (Lightning) fit into state.json and epochs.jsonl, and applies pause/finish/kill."""

    def __init__(self, reporter: Reporter, category: str, metric: str = "map", skip_best_epochs: int = 0, ema: bool = True):
        self.reporter = reporter
        self.category = category
        self.metric = metric
        self.skip_best_epochs = skip_best_epochs
        # RF-DETR's EMA track saves only above its initial 0.0; the plain track from any value
        self.best: float | None = None
        self.floor = 0.0 if ema else float("-inf")
        self.action: str | None = None  # "pause" or "finish", applied at the next epoch end
        self.validated = False
        self.train_started = time.time()

    def check_kill(self) -> None:
        if self.reporter.kill_requested:
            raise RunKilled()

    def on_train_start(self, trainer, module) -> None:
        self.train_started = time.time()
        self.reporter.write(force=True, status="training", phase="training", epochs=trainer.max_epochs, epoch=trainer.current_epoch, batches=_batches(trainer), task=TASKS[self.category])

    def on_train_epoch_start(self, trainer, module) -> None:
        self.check_kill()  # the previous epoch's checkpoints are written by now
        reporter = self.reporter
        reporter.batch = 0
        reporter.epoch_started = time.time()
        self.validated = False
        reporter.write(force=True, status="training", phase="training", epoch=trainer.current_epoch + 1, epochs=trainer.max_epochs, batch=0, batches=_batches(trainer))

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        reporter = self.reporter
        reporter.batch += 1
        self.check_kill()
        now = time.time()
        if now - reporter.last_write < STATE_INTERVAL:
            return
        batches = _batches(trainer)
        per_batch = (now - reporter.epoch_started) / max(reporter.batch, 1)
        epochs_left = max(trainer.max_epochs - trainer.current_epoch - 1, 0)
        epoch_time = reporter.mean_epoch_time() or per_batch * batches
        running = {RFDETR_RUNNING_LOSSES[key]: _number(value) for key, value in trainer.progress_bar_metrics.items() if key in RFDETR_RUNNING_LOSSES}
        reporter.write(batch=reporter.batch, batches=batches, running_losses=running, gpu_mem=_gpu_memory(), eta=per_batch * max(batches - reporter.batch, 0) + epochs_left * epoch_time)
        action = reporter.control()
        if action in ("pause", "finish"):
            self.action = action

    def on_validation_epoch_start(self, trainer, module) -> None:
        if getattr(trainer.state.fn, "value", trainer.state.fn) == "fit":
            self.reporter.write(force=True, phase="validating-epoch")

    def on_validation_epoch_end(self, trainer, module) -> None:
        self.validated = True

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self.check_kill()

    def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self.check_kill()

    def on_train_epoch_end(self, trainer, module) -> None:
        reporter = self.reporter
        now = time.time()
        epoch = trainer.current_epoch + 1
        metrics = normalize_rfdetr_metrics({key: _number(value) for key, value in trainer.callback_metrics.items()})
        if not self.validated:  # an epoch eval_interval skips: callback_metrics still hold the last validation
            metrics = {key: value for key, value in metrics.items() if not key.startswith(("metrics/", "ema/", "val/"))}
        fitness = rfdetr_fitness(metrics, self.category, self.metric)
        threshold = self.floor if self.best is None else self.best
        is_best = fitness is not None and epoch > self.skip_best_epochs and fitness > threshold
        if is_best:
            self.best = fitness
        epoch_time = now - reporter.epoch_started
        row = {"epoch": epoch, "time": now - self.train_started, **metrics, "fitness": fitness, "is_best": is_best, "epoch_time": epoch_time, "gpu_mem": _gpu_memory()}
        with reporter.epochs_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        reporter.epoch_times.append(epoch_time)
        changes = {"epoch": epoch, "epochs": trainer.max_epochs, "epoch_time": epoch_time, "running_losses": None, "batch": 0, "eta": (reporter.mean_epoch_time() or 0) * max(trainer.max_epochs - epoch, 0)}
        if is_best:
            changes.update(best_epoch=epoch, best_fitness=fitness)
        action = reporter.control()
        if action in ("pause", "finish"):
            self.action = action
        if self.action in ("pause", "finish"):
            trainer.should_stop = True  # Lightning saves this epoch's checkpoints, then leaves the fit loop
            if self.action == "finish":
                changes["stopped_early"] = True
        reporter.write(force=True, **changes)
        primary = "metrics/mAP50-95(M)" if self.category == "segmentation" else "metrics/mAP50-95(B)"
        score = f" · {primary[8:]} {metrics[primary]:.4f}" if primary in metrics else ""
        loss = f" · loss {metrics['train/loss']:.4f}" if "train/loss" in metrics else ""
        print(f"[workbench] epoch {epoch}/{trainer.max_epochs}{loss}{score}{' ★' if is_best else ''} ({epoch_time:.0f}s)", flush=True)


def install_callback(callback: Callback) -> None:
    """Append `callback` to every Lightning trainer RF-DETR builds in this process (train and evaluate)."""
    import rfdetr.training as training

    build = getattr(training.build_trainer, "__wrapped__", training.build_trainer)

    def build_trainer(*args, **kwargs):
        trainer = build(*args, **kwargs)
        trainer.callbacks.append(callback)
        return trainer

    build_trainer.__wrapped__ = build
    training.build_trainer = build_trainer


# ---------- the run ----------


def train_arguments(args: dict, view: Path, run_dir: Path) -> dict:
    """RF-DETR train() keyword arguments from the run's arguments."""
    kwargs = {key: value for key, value in args.items() if key not in ("model", "pretrain_weights", "device")}
    kwargs.update(dataset_dir=str(view), output_dir=str(run_dir), tensorboard=False, progress_bar=None, run_test=False)
    device = rfdetr_device(args.get("device"))
    if device:
        kwargs["device"] = device
    if "checkpoint_interval" not in args:
        # every periodic checkpoint is a full one (weights + optimizer): none unless asked; last.ckpt is kept anyway
        kwargs["checkpoint_interval"] = max(int(args.get("epochs", 100)), 1) + 1
    return kwargs


def _previous_best(epochs_path: Path) -> float | None:
    best = None
    try:
        lines = epochs_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("is_best") and isinstance(row.get("fitness"), (int, float)):
            best = row["fitness"] if best is None else max(best, row["fitness"])
    return best


def final_evaluations(args: dict, info: dict, view: Path, run_dir: Path, workdir: Path, reporter: Reporter, test: bool) -> None:
    """val.json (overall + per-class AP) and test.json for the best weights. Failures are warnings: the run itself trained."""
    from rfdetr import RFDETR as RFDETRModel

    best = next((run_dir / name for name in RFDETR_BEST if (run_dir / name).is_file()), None)
    if best is None:
        return
    batch = args.get("batch_size")
    common = {
        "dataset_dir": str(view),
        "output_dir": str(run_dir),
        "batch_size": batch if isinstance(batch, int) else DEFAULT_EVAL_BATCH,
        "num_workers": args.get("num_workers", 2),
        "log_per_class_metrics": True,
    }
    device = rfdetr_device(args.get("device"))
    if device:
        common["device"] = device
    model = RFDETRModel.from_checkpoint(str(best))
    for split, path, phase in (("val", "val.json", "final-validation"), ("test", "test.json", "testing")):
        if split == "test" and not test:
            continue
        reporter.write(force=True, status="validating" if split == "val" else "testing", phase=phase)
        try:
            results = model.evaluate(split=split, **common)
        except RunKilled:
            raise
        except Exception as exc:
            if reporter.kill_requested:
                raise
            traceback.print_exc()
            reporter.write(force=True, test_error=f"{'Final validation' if split == 'val' else 'Test evaluation'} failed: {exc}")
            continue
        metrics = normalize_rfdetr_metrics(results, split)
        write_json(workdir / path, {"metrics": metrics, "classes": rfdetr_class_rows(results, split, info["instances"].get(split, {})), "speed": None, "weights": best.name})


def run(spec_path: Path, spec: dict) -> int:
    workdir = spec_path.parent
    run_dir = workdir.parent
    resume = bool(spec.get("resume"))
    args = dict(spec.get("args") or {})
    reporter = Reporter(workdir, resume, raise_on_kill=False)
    signal.signal(signal.SIGTERM, reporter.on_sigterm)
    stop_watching = threading.Event()
    threading.Thread(target=reporter.watch_kill, args=(stop_watching,), daemon=True).start()
    # pretrained weights download into the workbench's weights cache (the runner's cwd), once
    os.environ.setdefault("RF_HOME", str(Path.cwd() / RFDETR_WEIGHTS_SUBDIR))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # aug_config: no version check over the network
    view = workdir / VIEW_DIR
    try:
        reporter.write(force=True, phase="preparing-data")
        info = build_view(workdir / spec.get("data", "splits.json"), view, reporter)
        print(f"[workbench] dataset folder: {sum(info['placed'].values())} images ({', '.join(f'{count} {how}' for how, count in sorted(info['placed'].items()))})", flush=True)
        if reporter.kill_requested:
            raise RunKilled()
        reporter.write(force=True, phase="preparing")
        import rfdetr

        callback = WorkbenchCallback(reporter, info["category"], str(args.get("best_model_metric", "map")), int(args.get("skip_best_epochs", 0)), bool(args.get("use_ema", True)))
        install_callback(callback)
        variant = RFDETR_VARIANTS[args["model"]]
        model_kwargs = {"pretrain_weights": str(Path(args["pretrain_weights"]).expanduser())} if args.get("pretrain_weights") else {}
        model = getattr(rfdetr, variant["class"])(**model_kwargs)
        kwargs = train_arguments(args, view, run_dir)
        if resume:
            checkpoint = RFDETR.resume_checkpoint(run_dir)
            if checkpoint is None:
                raise FileNotFoundError("No checkpoint to resume from (last.ckpt is missing)")
            kwargs["resume"] = str(checkpoint)
            callback.best = _previous_best(reporter.epochs_path)
        model.train(**kwargs)
        if reporter.kill_requested:
            raise RunKilled()
        if callback.action == "pause":
            stop_watching.set()
            reporter.write(force=True, status="paused", phase="paused", finished=time.time(), eta=None)
            return 0
        # from here the fit is over: a later Resume would have nothing to continue
        reporter.write(force=True, final_eval=True, eta=None)
        final_evaluations(args, info, view, run_dir, workdir, reporter, bool(spec.get("test_eval")))
        stop_watching.set()
        reporter.write(force=True, status="completed", phase="done", finished=time.time(), eta=0)
        return 0
    except BaseException as exc:  # includes KeyboardInterrupt/SystemExit (Lightning's SIGTERM exit), so the state never stays "training"
        stop_watching.set()
        if reporter.kill_requested:
            reporter.write(force=True, status="stopped", phase="stopped", error="Stopped by user", finished=time.time(), eta=None)
            return 1
        traceback.print_exc()
        message = str(exc) or exc.__class__.__name__
        reporter.write(force=True, status="failed", phase="failed", error=message, finished=time.time(), eta=None)
        return 1
    finally:
        shutil.rmtree(view, ignore_errors=True)
