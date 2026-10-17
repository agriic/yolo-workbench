"""RF-DETR as a second training framework: arguments, results, the runner's dataset folder and
callback, the training manager's handling of RF-DETR runs, and RF-DETR predictions.

No GPU and no training here except the opt-in smoke test; the schema test imports rfdetr (several
seconds) to check the server's static copy of its TrainConfig."""

import enum
import json
import os
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from yolo_workbench import frameworks
from yolo_workbench.dataset import Dataset
from yolo_workbench.frameworks import (
    RFDETR,
    RFDETR_MANAGED,
    RFDETR_VARIANTS,
    normalize_rfdetr_metrics,
    read_rfdetr_metrics_csv,
    rfdetr_class_rows,
    rfdetr_fitness,
    weights_framework,
)
from yolo_workbench.jobs import JobManager
from yolo_workbench.predictor import PredictorManager, RFDETRBackend, _nms
from yolo_workbench.rfdetr_runner import WorkbenchCallback, build_view, train_arguments
from yolo_workbench.train_runner import Reporter, RunKilled
from yolo_workbench.training import TrainingManager, best_epoch, read_epochs_jsonl

from test_training import FakeLauncher, make_dataset, make_manager

RF_ARGS = "model: rfdetr-nano\nepochs: 5\nbatch_size: 2\n"


def validate(text: str, category: str = "detection") -> dict:
    return frameworks.validate(category, text)


def messages(result: dict) -> str:
    return " | ".join(item["message"] for item in result["errors"])


def fake_weights(path: Path, ultralytics: bool) -> Path:
    """A torch-style zip checkpoint whose pickle does (or does not) name Ultralytics classes."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"\x80\x02cultralytics.nn.tasks\nDetectionModel\n" if ultralytics else b"\x80\x02}q\x00X\x04\x00\x00\x00argsq\x01")
    return path


# ---------- arguments ----------


def test_the_framework_follows_the_model():
    result = validate(RF_ARGS)
    assert result["framework"] == "rfdetr" and result["errors"] == []
    assert list(result["args"]) == ["model", "epochs", "batch_size"]
    assert validate("model: yolo11n.pt\nepochs: 5\n")["framework"] == "ultralytics"
    # RF-DETR names its arguments its own way: an Ultralytics key is unknown, with a suggestion
    result = validate("model: rfdetr-medium\nimgsz: 640\n")
    assert "Unknown RF-DETR argument `imgsz`" in messages(result)


def test_rfdetr_argument_errors_have_lines(tmp_path):
    result = validate("model: rfdetr-medium\ndataset_dir: /x\nepochs: many\nresolution: 600\nkeypoint_oks_sigmas: [1]\nmask_ce_loss_coef: 2\nlrr: 0.1\n")
    by_line = {item["line"]: item["message"] for item in result["errors"]}
    assert "set by the workbench" in by_line[2]
    assert "`epochs` must be an integer" in by_line[3]
    assert "multiple of 32" in by_line[4]
    assert "keypoint" in by_line[5]
    assert "only applies to segmentation" in by_line[6]
    assert "did you mean `lr`" in by_line[7]
    assert "seg" in messages(validate("model: rfdetr-medium\n", "segmentation"))  # a box variant on a segmentation dataset
    assert "multiple of 24" in messages(validate("model: rfdetr-seg-medium\nresolution: 440\n", "segmentation"))
    assert validate("model: rfdetr-seg-nano\nresolution: 324\nmask_ce_loss_coef: 2\n", "segmentation")["errors"] == []
    assert "does not train classification" in messages(validate("model: rfdetr-nano\n", "classification"))
    assert "needs a CUDA GPU" in messages(validate("model: rfdetr-nano\nbatch_size: auto\ndevice: cpu\n"))
    assert validate("model: rfdetr-nano\nbatch_size: auto\ndevice: 0\n")["errors"] == []
    assert "Multi-GPU" in messages(validate("model: rfdetr-nano\ndevice: 0,1\n"))
    assert "Unknown RF-DETR variant" in messages(validate("model: rfdetr-xxl\n"))
    assert "rfdetr[augment]" in messages(validate("model: rfdetr-nano\naug_config: {HorizontalFlip: {p: 0.5}}\n"))
    warnings = validate("model: rfdetr-nano\ncheckpoint_interval: 1\n")["warnings"]
    assert warnings and "every epoch" in warnings[0]["message"]


def test_weights_must_match_the_framework(tmp_path):
    yolo = fake_weights(tmp_path / "yolo.pt", ultralytics=True)
    rfdetr_seg = fake_weights(tmp_path / "rf-detr-seg-nano.pt", ultralytics=False)
    rfdetr_pth = tmp_path / "checkpoint_best_total.pth"
    rfdetr_pth.write_bytes(b"x")
    assert weights_framework(yolo) == "ultralytics"
    assert weights_framework(rfdetr_seg) == "rfdetr"
    assert weights_framework(rfdetr_pth) == "rfdetr"
    assert "Ultralytics weights" in messages(validate(f"model: rfdetr-nano\npretrain_weights: {yolo}\n"))
    assert validate(f"model: rfdetr-nano\npretrain_weights: {rfdetr_pth}\n")["errors"] == []
    assert "Weights file not found" in messages(validate(f"model: rfdetr-nano\npretrain_weights: {tmp_path / 'gone.pth'}\n"))
    assert "RF-DETR weights" in messages(validate(f"model: {rfdetr_seg}\n"))


def test_schema_matches_the_installed_rfdetr():
    """The server validates against a static copy of RF-DETR's TrainConfig (importing rfdetr takes seconds)."""
    import rfdetr
    from rfdetr.config import SegmentationTrainConfig, TrainConfig

    fields = SegmentationTrainConfig.model_fields
    for key, (_, default, _) in frameworks._RF_FIELDS.items():
        assert key in fields, key
        actual = fields[key].default
        actual = actual.value if isinstance(actual, enum.Enum) else actual
        if key not in ("lr_scheduler_kwargs", "optimizer_kwargs"):  # default factories
            assert actual == default, (key, actual, default)
    covered = set(frameworks._RF_FIELDS) | set(RFDETR_MANAGED) | frameworks._RF_UNSUPPORTED
    assert set(TrainConfig.model_fields) <= covered, set(TrainConfig.model_fields) - covered
    for name, variant in RFDETR_VARIANTS.items():
        model_class = getattr(rfdetr, variant["class"])
        config = model_class.get_model_config(model_class)  # what the variant builds (Large overrides it)
        assert model_class.size == name
        assert config.resolution == variant["resolution"]
        assert config.patch_size * config.num_windows == variant["step"]
        assert Path(config.pretrain_weights).name == variant["weights"]


# ---------- results ----------

LOGGED = {
    "loss": 4.5, "loss_cls": 0.8, "train/lr": 1e-4, "train/lr_min": 3e-6,
    "train/loss": 4.7, "train/loss_ce": 0.97, "train/loss_bbox": 0.03, "train/loss_giou": 0.09, "train/loss_ce_aux": 2.0, "train/cardinality_error": 8.8,
    "val/mAP_50_95": 0.36, "val/mAP_50": 0.38, "val/mAP_75": 0.38, "val/mAR": 0.48, "val/F1": 0.4, "val/precision": 0.5, "val/recall": 0.33,
    "val/ema_mAP_50_95": 0.36, "val/ema_mAR": 0.48,
}


def test_rfdetr_metrics_become_workbench_keys():
    row = normalize_rfdetr_metrics(LOGGED)
    assert row == {
        "lr/pg0": 1e-4, "train/loss": 4.7, "train/ce_loss": 0.97, "train/bbox_loss": 0.03, "train/giou_loss": 0.09,
        "metrics/mAP50-95(B)": 0.36, "metrics/mAP50(B)": 0.38, "metrics/mAP75(B)": 0.38, "metrics/mAR(B)": 0.48,
        "metrics/F1(B)": 0.4, "metrics/precision(B)": 0.5, "metrics/recall(B)": 0.33,
    }  # EMA mirrors the plain values by default: dropped; diagnostics and aux losses too
    seg = normalize_rfdetr_metrics({"val/segm_mAP_50_95": 0.45, "val/ema_segm_mAP_50_95": 0.5, "train/loss_mask_dice": 0.2})
    assert seg == {"metrics/mAP50-95(M)": 0.45, "ema/mAP50-95(M)": 0.5, "train/mask_dice_loss": 0.2}
    assert rfdetr_fitness(seg, "segmentation") == 0.5  # the better of plain and EMA, as RF-DETR picks its best weights
    assert rfdetr_fitness(row, "detection", "mar") == 0.48
    test = {"test/loss": 4.2, "test/mAP_50_95": 0.36, "test/AP/red": 0.1, "test/AP/green": 0.7}
    assert normalize_rfdetr_metrics(test, "test") == {"metrics/mAP50-95(B)": 0.36}
    assert rfdetr_class_rows(test, "test", {"red": 3}) == [{"Class": "red", "Instances": 3, "mAP50-95": 0.1}, {"Class": "green", "Instances": 0, "mAP50-95": 0.7}]


def test_metrics_csv_is_folded_per_epoch(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(
        "epoch,step,train/loss,train/loss_ce,val/mAP_50_95,val/mAP_50\n"
        "0,4,,,0.25,0.3\n"
        "0,4,6.1,1.1,,\n"
        "1,9,,,0.36,0.4\n"
        "1,9,4.7,0.9,,\n"
        "2,14,,"
    )
    rows = read_rfdetr_metrics_csv(path, "detection")
    assert [(row["epoch"], row.get("train/loss"), row.get("metrics/mAP50-95(B)"), row.get("fitness")) for row in rows] == [(1, 6.1, 0.25, 0.25), (2, 4.7, 0.36, 0.36), (3, None, None, None)]


def test_best_epoch_follows_rfdetr_strict_improvement():
    rows = [{"epoch": 1, "fitness": 0.3, "is_best": True}, {"epoch": 2, "fitness": 0.3, "is_best": False}, {"epoch": 3, "fitness": 0.2, "is_best": False}]
    assert best_epoch(rows, "detection", RFDETR)["epoch"] == 1  # RF-DETR keeps the first of tied epochs
    assert best_epoch(rows, "detection")["epoch"] == 2  # Ultralytics rewrites best.pt on ties
    unmarked = [{key: value for key, value in row.items() if key != "is_best"} for row in rows]
    assert best_epoch(unmarked, "detection", RFDETR)["epoch"] == 1


def test_rfdetr_weights_and_resume_checkpoint(tmp_path):
    assert RFDETR.weights(tmp_path) == {} and RFDETR.resume_checkpoint(tmp_path) is None
    (tmp_path / "checkpoint_best_ema.pth").write_bytes(b"x")  # a paused or killed run: no total yet
    assert RFDETR.weights(tmp_path) == {"best": tmp_path / "checkpoint_best_ema.pth"}
    (tmp_path / "checkpoint_best_total.pth").write_bytes(b"x")
    (tmp_path / "last_ema.pth").write_bytes(b"x")
    assert RFDETR.weights(tmp_path) == {"best": tmp_path / "checkpoint_best_total.pth", "last": tmp_path / "last_ema.pth"}
    for name in ("checkpoint_9.ckpt", "checkpoint_19.ckpt"):
        (tmp_path / name).write_bytes(b"x")
    assert RFDETR.resume_checkpoint(tmp_path) == tmp_path / "checkpoint_19.ckpt"
    (tmp_path / "last.ckpt").write_bytes(b"x")
    assert RFDETR.resume_checkpoint(tmp_path) == tmp_path / "last.ckpt"


# ---------- the runner's dataset folder ----------


def write_splits(tmp_path: Path) -> Path:
    images = tmp_path / "src"
    (images / "a").mkdir(parents=True)
    (images / "b").mkdir()
    Image.new("RGB", (40, 20), "red").save(images / "a" / "same.jpg")
    Image.new("RGB", (40, 20), "blue").save(images / "b" / "same.jpg")  # same name in another folder of the split
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways: upright it is 40 wide, 20 high
    Image.new("RGB", (20, 40), "green").save(images / "a" / "phone.jpg", exif=exif)
    Image.new("RGB", (16, 16)).save(images / "a" / "empty.png")
    splits = {
        "version": 1, "category": "detection", "names": ["cat", "dog"],
        "splits": {
            "train": [[str(images / "a" / "same.jpg"), [[0, 0.5, 0.5, 0.2, 0.2]]], [str(images / "b" / "same.jpg"), [[1, 0.5, 0.5, 0.4, 0.4], [1, 0.1, 0.1, 0.1, 0.1]]], [str(images / "a" / "phone.jpg"), [[0, 0.25, 0.5, 0.5, 1.0]]]],
            "val": [[str(images / "a" / "empty.png"), []]],
        },
    }
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(splits))
    return path


def test_build_view_writes_an_rfdetr_yolo_folder(tmp_path):
    view = tmp_path / "view"
    info = build_view(write_splits(tmp_path), view)
    assert sorted(path.name for path in (view / "train" / "images").iterdir()) == ["000000_same.jpg", "000001_same.jpg", "000002_phone.jpg"]
    assert (view / "train" / "labels" / "000001_same.txt").read_text() == "1 0.500000 0.500000 0.400000 0.400000\n1 0.100000 0.100000 0.100000 0.100000\n"
    assert (view / "valid" / "labels" / "000000_empty.txt").read_text() == ""  # a background image
    assert info["instances"] == {"train": {"cat": 2, "dog": 2}, "val": {}} and info["has_test"] is False
    assert info["placed"] == {"linked": 3, "rotated": 1}
    assert os.path.samefile(view / "train" / "images" / "000000_same.jpg", tmp_path / "src" / "a" / "same.jpg")
    with Image.open(view / "train" / "images" / "000002_phone.jpg") as upright:
        assert upright.size == (40, 20) and upright.getexif().get(0x0112) in (None, 1)
    config = yaml.safe_load((view / "data.yaml").read_text())
    assert config == {"path": ".", "train": "train/images", "val": "valid/images", "nc": 2, "names": ["cat", "dog"]}
    build_view(tmp_path / "splits.json", view)  # rebuilt from scratch (a resume)
    assert len(list((view / "train" / "images").iterdir())) == 3


def test_build_view_copies_when_links_fail(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("not here")

    monkeypatch.setattr(os, "link", refuse)
    monkeypatch.setattr(Path, "symlink_to", refuse)
    info = build_view(write_splits(tmp_path), tmp_path / "view")
    assert info["placed"] == {"copied": 3, "rotated": 1}
    assert not (tmp_path / "view" / "train" / "images" / "000000_same.jpg").is_symlink()


def test_train_arguments_set_the_managed_keys(tmp_path):
    kwargs = train_arguments({"model": "rfdetr-nano", "pretrain_weights": "/w.pth", "epochs": 7, "device": 1, "lr": 2e-4}, tmp_path / "view", tmp_path)
    assert kwargs == {"epochs": 7, "lr": 2e-4, "dataset_dir": str(tmp_path / "view"), "output_dir": str(tmp_path), "tensorboard": False, "progress_bar": None, "run_test": False, "device": "cuda:1", "checkpoint_interval": 8}
    assert train_arguments({"model": "rfdetr-nano", "checkpoint_interval": 5, "device": "cpu"}, tmp_path, tmp_path)["checkpoint_interval"] == 5


# ---------- the runner's Lightning callback ----------


class FakeLightningTrainer:
    def __init__(self, max_epochs=5):
        self.max_epochs = max_epochs
        self.current_epoch = 0
        self.num_training_batches = 4
        self.callback_metrics = {}
        self.progress_bar_metrics = {"loss": 4.0, "loss_cls": 1.0, "train/lr": 1e-4}
        self.should_stop = False
        self.state = SimpleNamespace(fn="fit")


def run_epoch(callback: WorkbenchCallback, trainer: FakeLightningTrainer, metrics: dict, validate: bool = True) -> None:
    callback.on_train_epoch_start(trainer, None)
    for index in range(trainer.num_training_batches):
        callback.on_train_batch_end(trainer, None, None, None, index)
    if validate:
        callback.on_validation_epoch_start(trainer, None)
        callback.on_validation_epoch_end(trainer, None)
    trainer.callback_metrics = metrics
    callback.on_train_epoch_end(trainer, None)
    trainer.current_epoch += 1


def test_callback_records_epochs_and_the_best_one(tmp_path):
    reporter = Reporter(tmp_path, resume=False, raise_on_kill=False)
    callback = WorkbenchCallback(reporter, "detection", skip_best_epochs=1)
    trainer = FakeLightningTrainer()
    callback.on_train_start(trainer, None)
    run_epoch(callback, trainer, {"train/loss": 5.0, "val/mAP_50_95": 0.5})  # skipped for the best weights
    run_epoch(callback, trainer, {"train/loss": 4.0, "val/mAP_50_95": 0.3})
    run_epoch(callback, trainer, {"train/loss": 3.5, "val/mAP_50_95": 0.3})  # a tie is not an improvement
    run_epoch(callback, trainer, {"train/loss": 3.0, "val/mAP_50_95": 0.3}, validate=False)  # eval_interval skipped it
    rows = read_epochs_jsonl(tmp_path / "epochs.jsonl")
    assert [(row["epoch"], row["is_best"], row.get("metrics/mAP50-95(B)")) for row in rows] == [(1, False, 0.5), (2, True, 0.3), (3, False, 0.3), (4, False, None)]
    assert best_epoch(rows, "detection", RFDETR)["epoch"] == 2
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["best_epoch"] == 2 and state["epoch"] == 4 and state["task"] == "detect"
    assert not trainer.should_stop


def test_callback_pause_finish_and_kill(tmp_path):
    reporter = Reporter(tmp_path, resume=False, raise_on_kill=False)
    callback = WorkbenchCallback(reporter, "detection")
    trainer = FakeLightningTrainer()
    (tmp_path / "control.json").write_text(json.dumps({"action": "pause"}))
    run_epoch(callback, trainer, {"val/mAP_50_95": 0.2})
    assert trainer.should_stop and callback.action == "pause"  # Lightning saves this epoch's checkpoint, then stops
    assert len(read_epochs_jsonl(tmp_path / "epochs.jsonl")) == 1

    (tmp_path / "f").mkdir()
    reporter = Reporter(tmp_path / "f", resume=False, raise_on_kill=False)
    callback = WorkbenchCallback(reporter, "detection")
    trainer = FakeLightningTrainer()
    (tmp_path / "f" / "control.json").write_text(json.dumps({"action": "finish"}))
    run_epoch(callback, trainer, {"val/mAP_50_95": 0.2})
    assert trainer.should_stop and json.loads((tmp_path / "f" / "state.json").read_text())["stopped_early"]

    (tmp_path / "k").mkdir()
    reporter = Reporter(tmp_path / "k", resume=False, raise_on_kill=False)
    callback = WorkbenchCallback(reporter, "detection")
    trainer = FakeLightningTrainer()
    reporter.on_sigterm(15, None)  # only a flag: the signal may land while a checkpoint is written
    assert reporter.kill_requested
    with pytest.raises(RunKilled):
        callback.on_train_batch_end(trainer, None, None, None, 0)  # a batch boundary: safe to leave
    with pytest.raises(RunKilled):
        callback.on_validation_batch_end(trainer, None, None, None, 0)


def test_flag_only_reporter_hears_a_kill_through_control_json(tmp_path):
    import threading

    reporter = Reporter(tmp_path, resume=False, raise_on_kill=False)
    stop = threading.Event()
    (tmp_path / "control.json").write_text(json.dumps({"action": "kill"}))
    watcher = threading.Thread(target=reporter.watch_kill, args=(stop, 0.01))
    watcher.start()
    watcher.join(5)
    stop.set()
    assert reporter.kill_requested


# ---------- the training manager ----------


def runner_trains_rfdetr(run_dir: Path, status="completed", epochs=2, **state):
    """What the RF-DETR runner leaves behind."""
    workdir = run_dir / "workbench"
    with (workdir / "epochs.jsonl").open("a") as stream:
        for epoch in range(1, epochs + 1):
            stream.write(json.dumps({"epoch": epoch, "train/loss": 5.0 / epoch, "metrics/mAP50-95(B)": 0.2 * epoch, "fitness": 0.2 * epoch, "is_best": True}) + "\n")
    for name in ("checkpoint_best_total.pth", "last_ema.pth", "last.ckpt"):
        (run_dir / name).write_bytes(b"x")
    current = json.loads((workdir / "state.json").read_text())
    current.update(status=status, **state)
    (workdir / "state.json").write_text(json.dumps(current))


def test_rfdetr_run_lifecycle(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    result = manager.enqueue(None, dataset, "rf", "first", RF_ARGS + "resolution: 448\n")
    run = result["run"]
    run_dir = Path(run["path"])
    assert run["framework"] == "rfdetr" and run["variant"] == "rfdetr-nano" and run["imgsz"] == 448 and run["batch"] == 2
    meta = json.loads((run_dir / "workbench" / "run.json").read_text())
    spec = json.loads((run_dir / "workbench" / "spec.json").read_text())
    assert meta["framework"] == spec["framework"] == "rfdetr" and meta["base_model"] == "rfdetr-nano"
    # the RF-DETR runner gets a snapshot of the images and their parsed labels, not Ultralytics' data.yaml
    assert spec["data"] == "splits.json" and spec["test_eval"] is True and "project" not in spec["args"]
    assert not (run_dir / "workbench" / "data.yaml").exists()
    splits = json.loads((run_dir / "workbench" / "splits.json").read_text())
    assert splits["names"] == ["cat", "dog"] and len(splits["splits"]["train"]) == 2
    assert splits["splits"]["train"][0][1] == [[0, 0.5, 0.5, 0.2, 0.2]]

    runner_trains_rfdetr(run_dir, status="paused")
    launcher.handle.code = 0
    summary = manager.list_runs(None)["items"][0]
    assert summary["status"] == "paused" and summary["resumable"] and summary["best_epoch"] == 2
    assert summary["weights"]["best"].endswith("checkpoint_best_total.pth") and summary["has_weights"] == {"best": True, "last": True}
    path, name = manager.best_weights(None, summary["id"])
    assert path.name == "checkpoint_best_total.pth" and name.endswith("-rfdetr-nano-" + time.strftime("%Y.%m.%d") + ".pth")
    copied = manager.copy_args(None, summary["id"])
    assert copied["framework"] == "rfdetr" and copied["args"] == {"model": "rfdetr-nano", "epochs": 5, "batch_size": 2, "resolution": 448}

    # fine-tuning from it: the class check and the delete guard follow pretrain_weights
    finetune = f"model: rfdetr-nano\npretrain_weights: {summary['weights']['best']}\n"
    assert manager.preflight(None, dataset, frameworks.validate("detection", finetune)["args"])["warnings"] == []
    child = manager.enqueue(None, dataset, "rf-ft", "", finetune)["run"]
    assert child["model"] == summary["weights"]["best"]
    with pytest.raises(Exception, match="starts from this run"):
        manager.delete_run(None, summary["id"])
    manager.cancel(None, child["id"]) if child["status"] == "queued" else None


def test_rfdetr_trains_colocated_labels_that_ultralytics_cannot(tmp_path):
    root = tmp_path / "flat"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        Image.new("RGB", (16, 16)).save(root / "images" / split / "a.jpg")
        (root / "images" / split / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "data.yaml").write_text(yaml.safe_dump({"train": "images/train", "val": "images/val", "names": ["a"]}))
    dataset = Dataset(root / "data.yaml", "detection")
    manager = make_manager(dataset, tmp_path)
    assert any("would not find the labels" in error for error in manager.preflight(None, dataset, {"model": "yolo11n.pt"})["errors"])
    assert manager.preflight(None, dataset, {"model": "rfdetr-nano"})["errors"] == []


def test_auto_batch_needs_a_gpu(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    manager = make_manager(dataset, tmp_path)
    monkeypatch.setattr(manager, "devices", lambda: [])
    assert any("needs a CUDA GPU" in error for error in manager.preflight(None, dataset, {"model": "rfdetr-nano", "batch_size": "auto"})["errors"])
    monkeypatch.setattr(manager, "devices", lambda: [{"id": "0", "name": "GPU"}])
    assert manager.preflight(None, dataset, {"model": "rfdetr-nano", "batch_size": "auto"})["errors"] == []


def test_options_offer_rfdetr_for_detection_only(tmp_path, monkeypatch):
    monkeypatch.delenv("RF_HOME", raising=False)
    dataset = make_dataset(tmp_path)
    manager = make_manager(dataset, tmp_path)
    (tmp_path / "weights" / "rfdetr").mkdir(parents=True)
    (tmp_path / "weights" / "rfdetr" / "rf-detr-small.pth").write_bytes(b"x")
    options = manager.options(None, dataset)
    entries = {item["id"]: item for item in options["frameworks"]}
    assert set(entries) == {"ultralytics", "rfdetr"} and entries["rfdetr"]["available"]
    assert entries["rfdetr"]["defaults"]["epochs"] == 100 and entries["rfdetr"]["defaults"]["checkpoint_interval"] is None
    assert "dataset_dir" in entries["rfdetr"]["managed"] and "mask_ce_loss_coef" not in entries["rfdetr"]["defaults"]
    assert [(item["name"], item["cached"]) for item in entries["rfdetr"]["pretrained"]] == [("rfdetr-nano", False), ("rfdetr-small", True), ("rfdetr-medium", False), ("rfdetr-large", False)]
    assert options["defaults"] == entries["ultralytics"]["defaults"]  # the Ultralytics fields stay at the top level

    classification = make_dataset(tmp_path / "cls", splits=("train", "val"))
    classification.category = "classification"
    manager = make_manager(classification, tmp_path / "cls")
    assert [item["id"] for item in manager.options(None, classification)["frameworks"]] == ["ultralytics"]


def test_external_rfdetr_runs_are_listed(tmp_path):
    dataset = make_dataset(tmp_path)
    manager = make_manager(dataset, tmp_path)
    run_dir = dataset.root / "runs" / "rfdetr-cli"
    run_dir.mkdir(parents=True)
    (run_dir / "training_config.json").write_text(json.dumps({"train_config": {"epochs": 3, "lr": 2e-4, "dataset_dir": str(dataset.root)}, "model_config": {"model_name": "RFDETRSmall", "resolution": 512}}))
    (run_dir / "metrics.csv").write_text("epoch,step,train/loss,val/mAP_50_95\n0,4,,0.2\n0,4,6.0,\n1,9,,0.4\n1,9,5.0,\n")
    (run_dir / "checkpoint_best_total.pth").write_bytes(b"x")
    summary = manager.list_runs(None)["items"][0]
    assert summary["source"] == "external" and summary["framework"] == "rfdetr" and summary["variant"] == "rfdetr-small"
    assert summary["epochs_done"] == 2 and summary["best_epoch"] == 2 and summary["best_metrics"]["metrics/mAP50-95(B)"] == 0.4
    assert summary["imgsz"] == 512 and summary["epochs"] == 3
    detail = manager.run_detail(None, summary["id"])
    assert detail["args"] == {"model": "rfdetr-small", "epochs": 3, "lr": 2e-4}


# ---------- predictions ----------


def test_nms_keeps_the_best_box_per_class():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    classes = np.array([0, 0, 1, 0])
    assert _nms(boxes, scores, classes, 0.5) == [0, 2, 3]  # the other class's box overlaps but stays


class FakeDetections:
    def __init__(self, xyxy, confidence, class_id, mask=None):
        self.xyxy, self.confidence, self.class_id, self.mask = np.array(xyxy, dtype=float), np.array(confidence), np.array(class_id), mask

    def __len__(self):
        return len(self.xyxy)


class FakeRFDETR:
    def __init__(self, segmentation=False, names=("cat", "dog")):
        self.model_config = SimpleNamespace(segmentation_head=segmentation)
        self.model = SimpleNamespace(class_names=list(names) if names else None)
        self.seen: list = []
        self.segmentation = segmentation

    def predict(self, image, threshold=0.5, include_source_image=True):
        self.seen.append((image.size, threshold))
        mask = None
        if self.segmentation:
            mask = np.zeros((3, 20, 40), dtype=bool)
            mask[0, 5:15, 10:30] = True
            mask[1, 5:15, 10:29] = True
            mask[2, 0:2, 0:2] = True
        return FakeDetections([[10, 5, 30, 15], [10, 5, 29, 15], [-5, -5, 2, 2]], [0.9, 0.8, 0.6], [1, 1, 0], mask)


def test_rfdetr_backend_predictions(tmp_path, monkeypatch):
    import rfdetr

    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (20, 40)).save(tmp_path / "phone.jpg", exif=exif)
    fake = FakeRFDETR()
    monkeypatch.setattr(rfdetr.RFDETR, "from_checkpoint", classmethod(lambda cls, path: fake))
    backend = RFDETRBackend(tmp_path / "w.pth", "detection", 0.5, 0.45)
    assert backend.names == {0: "cat", 1: "dog"} and backend.framework == "rfdetr"
    predictions = backend.predict(tmp_path / "phone.jpg", conf=0.05)
    assert fake.seen == [((40, 20), 0.05)]  # the upright image: RF-DETR ignores EXIF orientation
    assert [(item["class_id"], [round(value, 3) for value in item["points"]]) for item in predictions] == [(1, [0.5, 0.5, 0.5, 0.5]), (0, [0.025, 0.05, 0.05, 0.1])]  # NMS, then clipped to the image

    fake = FakeRFDETR(segmentation=True)
    backend = RFDETRBackend(tmp_path / "w.pth", "segmentation", 0.5, 0.45)
    polygon = backend.predict(tmp_path / "phone.jpg")[0]["points"]
    assert len(polygon) >= 8 and min(polygon[0::2]) == pytest.approx(0.25) and max(polygon[1::2]) <= 0.75

    fake = FakeRFDETR(segmentation=False, names=None)  # RF-DETR's COCO-pretrained weights: COCO's own ids
    assert RFDETRBackend(tmp_path / "w.pth", "detection", 0.5, 0.45).names[1] == "person"
    with pytest.raises(Exception, match="segmentation model"):
        RFDETRBackend(tmp_path / "w.pth", "segmentation", 0.5, 0.45)


def test_predictor_picks_the_backend_and_its_threshold(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)

    class Backend:
        default_conf = 0.25
        framework = "ultralytics"

        def __init__(self, path, category, conf, iou):
            self.names, self.conf = {0: "cat"}, conf

    class RFBackend(Backend):
        default_conf = 0.5
        framework = "rfdetr"

    monkeypatch.setattr("yolo_workbench.predictor.UltralyticsBackend", Backend)
    monkeypatch.setattr("yolo_workbench.predictor.RFDETRBackend", RFBackend)
    yolo = fake_weights(tmp_path / "yolo.pt", ultralytics=True)
    rf = tmp_path / "checkpoint_best_total.pth"
    rf.write_bytes(b"x")
    manager = PredictorManager(dataset)
    assert manager.load(str(yolo))["conf"] == 0.25
    state = manager.load(str(rf))
    assert state["conf"] == 0.5 and state["model"]["framework"] == "rfdetr" and manager.backend.conf == 0.5
    assert manager.load(str(rf), conf=0.3)["conf"] == 0.3  # an explicit threshold wins
    assert manager.load(str(yolo))["conf"] == 0.25


def test_discovery_lists_rfdetr_runs_once(tmp_path):
    dataset = make_dataset(tmp_path)
    run_dir = dataset.root / "runs" / "rf"
    (run_dir / "workbench").mkdir(parents=True)
    (run_dir / "workbench" / "run.json").write_text(json.dumps({"name": "RF one", "framework": "rfdetr"}))
    (run_dir / "training_config.json").write_text("{}")
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "last_ema.pth"):
        (run_dir / name).write_bytes(b"x")
    items = PredictorManager(dataset).discover_models()["items"]
    assert [(item["name"], item["source"]) for item in items] == [("RF one · best (RF-DETR)", "run")]


# ---------- real training (opt-in) ----------


@pytest.mark.skipif(not os.environ.get("WORKBENCH_SLOW_TESTS"), reason="set WORKBENCH_SLOW_TESTS=1: trains real RF-DETR models on CPU (downloads their weights)")
@pytest.mark.parametrize("category, model", [("detection", "rfdetr-nano"), ("segmentation", "rfdetr-seg-nano")])
def test_real_rfdetr_training_smoke(tmp_path, category, model):
    dataset = make_dataset(tmp_path)
    if category == "segmentation":
        for label in dataset.root.rglob("labels/*/*.txt"):
            class_id = label.read_text().split()[0]
            label.write_text(f"{class_id} 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n")
        dataset = Dataset(dataset.yaml_path, "segmentation")
    manager = TrainingManager.for_dataset(dataset, jobs=JobManager(synchronous=True), weights_dir=Path.home() / ".cache" / "yolo-workbench" / "weights")
    manager.attach(dataset)
    run = manager.enqueue(None, dataset, "smoke", "", f"model: {model}\nepochs: 2\nbatch_size: 2\nnum_workers: 0\ndevice: cpu\n", confirm=True)["run"]
    deadline = time.time() + 900
    while time.time() < deadline and manager.run_detail(None, run["id"])["active"]:
        time.sleep(1)
    detail = manager.run_detail(None, run["id"])
    assert detail["status"] == "completed", manager.log(None, run["id"])["text"][-3000:]
    assert detail["epochs_done"] == 2 and detail["val"]["metrics"] and detail["test"]["metrics"] and detail["val"]["classes"]
    assert detail["weights"]["best"].endswith("checkpoint_best_total.pth")
    assert not (Path(detail["path"]) / "workbench" / "data").exists()
    assert not list(Path(detail["path"]).glob("checkpoint_*.ckpt"))  # no periodic full checkpoints unless asked
