import asyncio
import json
import os
import signal
import threading
import time
from pathlib import Path

import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from PIL import Image

from yolo_workbench import classes, importer
from yolo_workbench.dataset import Dataset, DatasetError, DatasetLocked
from yolo_workbench.jobs import JobManager
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.train_runner import Reporter, RunKilled, RunPaused
from yolo_workbench.training import (
    TrainingManager,
    args_to_yaml,
    best_epoch,
    collapse_progress,
    fitness,
    read_epochs_jsonl,
    read_results_csv,
    validate_args,
)
from yolo_workbench.web import create_app, create_workspace_app
from yolo_workbench.workspace import Workspace

ARGS = "model: yolo11n.pt\nepochs: 5\nimgsz: 64\n"


# ---------- fixtures ----------


def make_dataset(tmp_path: Path, splits=("train", "val", "test"), category="detection") -> Dataset:
    """Parallel images/ + labels/ layout; every split has one image of each class."""
    root = tmp_path / "ds"
    for split in splits:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for class_id in (0, 1):
            Image.new("RGB", (32, 32), "white").save(root / "images" / split / f"{split}{class_id}.jpg")
            (root / "labels" / split / f"{split}{class_id}.txt").write_text(f"{class_id} 0.5 0.5 0.2 0.2\n")
    config = {split: f"images/{split}" for split in splits}
    config["names"] = {0: "cat", 1: "dog"}
    (root / "data.yaml").write_text(yaml.safe_dump(config))
    return Dataset(root / "data.yaml", category)


class FakeHandle:
    def __init__(self):
        self.pid = os.getpid()
        self.code = None
        self.calls = []

    def poll(self):
        return self.code

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")


class FakeLauncher:
    """Stands in for the subprocess: records launches; tests write the runner's files themselves."""

    def __init__(self):
        self.launches: list[tuple[Path, FakeHandle]] = []

    def __call__(self, spec_path: Path, run_dir: Path, cwd: Path):
        handle = FakeHandle()
        self.launches.append((run_dir, handle))
        return handle

    @property
    def handle(self) -> FakeHandle:
        return self.launches[-1][1]


def make_manager(dataset: Dataset, tmp_path: Path, launcher=None) -> TrainingManager:
    manager = TrainingManager.for_dataset(dataset, jobs=JobManager(synchronous=True), launcher=launcher or FakeLauncher(), weights_dir=tmp_path / "weights")
    manager.attach(dataset)
    return manager


def runner_finishes(run_dir: Path, status="completed", epochs=3, **state):
    """Write what the runner leaves behind, then let the process 'exit'."""
    workdir = run_dir / "workbench"
    with (workdir / "epochs.jsonl").open("a") as stream:
        for epoch in range(1, epochs + 1):
            stream.write(json.dumps({"epoch": epoch, "train/box_loss": 1.0 / epoch, "metrics/mAP50-95(B)": 0.1 * epoch, "fitness": 0.1 * epoch}) + "\n")
    (run_dir / "weights").mkdir(exist_ok=True)
    (run_dir / "weights" / "last.pt").write_bytes(b"x")
    (run_dir / "weights" / "best.pt").write_bytes(b"x")
    current = json.loads((workdir / "state.json").read_text())
    current.update(status=status, **state)
    (workdir / "state.json").write_text(json.dumps(current))


# ---------- parsing ----------


def test_results_csv_parsing_fitness_and_best_epoch(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "                  epoch,      train/box_loss,  metrics/mAP50(B),  metrics/mAP50-95(B),  metrics/mAP50-95(M)\n"
        "                      1,             1.5,          0.2,         0.1,        0.05\n"
        "                      2,             1.2,          0.4,         0.3,        0.2\n"
        "                      3,             1.1,          0.4,         0.3,        0.2\n"
        "                      4,             1.0,          0.3,         0.2,  \n"
    )
    rows = read_results_csv(csv_path)
    assert [row["epoch"] for row in rows] == [1, 2, 3, 4]
    assert rows[1]["metrics/mAP50-95(B)"] == 0.3 and "metrics/mAP50-95(M)" not in rows[3]
    assert fitness(rows[1], "detection") == 0.3
    assert fitness(rows[1], "segmentation") == pytest.approx(0.5)
    assert fitness({"metrics/accuracy_top1": 0.8, "metrics/accuracy_top5": 1.0}, "classification") == pytest.approx(0.9)
    # Ultralytics rewrites best.pt on ties, so the last tied epoch is the best one
    assert best_epoch(rows, "detection")["epoch"] == 3
    assert best_epoch([], "detection") is None


def test_epochs_jsonl_dedupes_resumed_epochs_and_skips_partial_lines(tmp_path):
    path = tmp_path / "epochs.jsonl"
    path.write_text(json.dumps({"epoch": 1, "a": 1}) + "\n" + json.dumps({"epoch": 2, "a": 2}) + "\n" + json.dumps({"epoch": 2, "a": 3}) + "\n" + '{"epoch": 3, "a"')
    assert read_epochs_jsonl(path) == [{"epoch": 1, "a": 1}, {"epoch": 2, "a": 3}]


def test_collapse_progress_keeps_last_redraw_and_drops_ansi():
    text = "\x1b[34mtrain:\x1b[0m start\n 10% bar\r 50% bar\r100% bar\nnext\n"
    assert collapse_progress(text) == "train: start\n100% bar\nnext\n"


# ---------- argument validation ----------


def test_validate_args_reports_problems_with_line_numbers():
    text = "model: yolo11n-seg.pt\nepochs: ten\ndata: other.yaml\nepoch: 3\ncache: disk\ndevice: 0,1\n"
    result = validate_args("detection", text)
    messages = {error["line"]: error["message"] for error in result["errors"]}
    assert "not a detection model" in messages[1]
    assert "epochs" in messages[2]
    assert "set by the workbench" in messages[3]
    assert "did you mean `epochs`" in messages[4]
    assert "disk" in messages[5]
    assert "Multi-GPU" in messages[6]


def test_validate_args_accepts_common_options_and_puts_model_first():
    result = validate_args("segmentation", "epochs: 50\nbatch: -1\nimgsz: 640\ncache: ram\nmodel: yolo26s-seg.pt\nlr0: 0.002\nfreeze: [0, 1]\n")
    assert result["errors"] == []
    assert list(result["args"])[0] == "model" and result["args"]["batch"] == -1
    assert args_to_yaml(result["args"]) == "model: yolo26s-seg.pt\nepochs: 50\nbatch: -1\nimgsz: 640\ncache: ram\nlr0: 0.002\nfreeze: [0, 1]\n"
    assert validate_args("detection", "epochs: 3\n")["errors"][0]["message"].startswith("`model` is required")
    assert validate_args("detection", "model: /does/not/exist.pt\n")["errors"][0]["message"].startswith("Model file not found")
    assert validate_args("detection", "model: [\n")["errors"][0]["message"].startswith("Invalid YAML")


# ---------- queue, lock and lifecycle ----------


def test_enqueue_launches_writes_data_files_and_locks_the_dataset(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    result = manager.enqueue(None, dataset, "First run", "notes", ARGS)
    assert result["queued"] and result["run"]["status"] == "starting" and result["run"]["active"]
    run_dir = launcher.launches[0][0]
    assert run_dir == dataset.root / "runs" / "First-run"
    spec = json.loads((run_dir / "workbench" / "spec.json").read_text())
    assert spec["args"]["project"] == str(dataset.root / "runs") and spec["args"]["name"] == "First-run" and spec["args"]["exist_ok"]
    data = yaml.safe_load(Path(spec["args"]["data"]).read_text())
    assert data["names"] == {0: "cat", 1: "dog"}
    assert Path(data["train"]).read_text().splitlines() == [str(dataset.root / "images" / "train" / name) for name in ("train0.jpg", "train1.jpg")]
    assert spec["test_eval"] == {"data": spec["args"]["data"]}
    assert str(dataset.root / "labels" / "train.cache") in spec["cache_files"]
    meta = json.loads((run_dir / "workbench" / "run.json").read_text())
    assert meta["dataset"]["splits"] == {"train": 2, "val": 2, "test": 2} and len(meta["dataset"]["fingerprint"]) == 12

    # every write path is refused while training; reads still work
    image_id = next(iter(dataset.images))
    with pytest.raises(DatasetLocked, match="First run"):
        dataset.replace_annotations(image_id, [])
    with pytest.raises(DatasetLocked):
        classes.add_classes(dataset, ["bird"])
    with pytest.raises(DatasetLocked):
        dataset.move_images([image_id], "val")
    with pytest.raises(DatasetLocked):
        importer.run_import(dataset, tmp_path, {})
    assert dataset.detail(image_id)["annotations"]
    assert manager.lock_info(dataset)["run"] == "First run"

    runner_finishes(run_dir)
    launcher.handle.code = 0
    manager.tick()
    assert manager.lock_reason(dataset) is None
    dataset.replace_annotations(image_id, [])
    summary = manager.list_runs(None)["items"][0]
    assert summary["status"] == "completed" and summary["best_epoch"] == 3 and summary["epochs_done"] == 3
    assert summary["best_metrics"] == {"metrics/mAP50-95(B)": pytest.approx(0.3)}
    assert not summary["resumable"]  # completed runs went through final_eval


def test_undo_is_blocked_while_training(tmp_path):
    dataset = make_dataset(tmp_path)
    manager = make_manager(dataset, tmp_path)
    dataset.replace_annotations(next(iter(dataset.images)), [])
    assert manager.enqueue(None, dataset, "", "", ARGS, confirm=True)["queued"]
    with pytest.raises(DatasetLocked):
        dataset.history("undo")


def test_queue_is_fifo_reorderable_and_advances_when_the_active_run_exits(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    first = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    second = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    third = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    assert (first["name"], second["name"], third["name"]) == ("exp-1", "exp-2", "exp-3")
    assert second["status"] == "queued" and second["queue_position"] == 1 and third["queue_position"] == 2
    queue = manager.queue()
    assert queue["active"]["id"] == first["id"] and [item["id"] for item in queue["queued"]] == [second["id"], third["id"]]
    with pytest.raises(DatasetError):
        manager.reorder([queue["queued"][0]["key"]])
    manager.reorder([queue["queued"][1]["key"], queue["queued"][0]["key"]])

    launcher.handle.code = -9  # died without writing a terminal state
    manager.tick()
    assert len(launcher.launches) == 2 and launcher.launches[1][0].name == "exp-3"
    runs = {item["id"]: item for item in manager.list_runs(None)["items"]}
    assert runs[first["id"]]["status"] == "interrupted" and "unexpectedly" in runs[first["id"]]["error"]
    assert runs[third["id"]]["active"] and runs[second["id"]]["queue_position"] == 1


def test_queue_waits_for_a_running_import(tmp_path):
    dataset = make_dataset(tmp_path)
    jobs = JobManager()
    release = __import__("threading").Event()
    jobs.start("import", None, "Import", lambda job: release.wait(5))
    launcher = FakeLauncher()
    manager = TrainingManager.for_dataset(dataset, jobs=jobs, launcher=launcher, weights_dir=tmp_path / "w")
    run = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    assert not launcher.launches and run["status"] == "queued" and "import" in run["waiting"]
    release.set()
    for _ in range(50):
        if not jobs.running(None):
            break
        time.sleep(0.02)
    manager.tick()
    assert len(launcher.launches) == 1


def test_queue_survives_restart_and_dead_or_reused_pids_are_interrupted(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    active = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    queued = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    process_file = launcher.launches[0][0] / "workbench" / "process.json"
    info = json.loads(process_file.read_text())
    assert info["pid"] == os.getpid() and info["create_time"] is not None

    # a restarted server has no handle: our own PID with another create_time is a reused PID, not the runner
    process_file.write_text(json.dumps({"pid": os.getpid(), "create_time": info["create_time"] - 1000}))
    relaunch = FakeLauncher()
    restarted = TrainingManager.for_dataset(dataset, jobs=JobManager(synchronous=True), launcher=relaunch, weights_dir=tmp_path / "w")
    runs = {item["id"]: item for item in restarted.list_runs(None)["items"]}
    assert runs[active["id"]]["status"] == "interrupted"
    assert runs[queued["id"]]["active"] and len(relaunch.launches) == 1


def test_reattached_live_process_keeps_the_lock(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    manager = make_manager(dataset, tmp_path)
    run = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    monkeypatch.setattr("yolo_workbench.training._process_alive", lambda info: True)
    restarted = TrainingManager.for_dataset(dataset, jobs=JobManager(synchronous=True), launcher=FakeLauncher(), weights_dir=tmp_path / "w")
    restarted.attach(dataset)
    assert restarted.queue()["active"]["id"] == run["id"]
    with pytest.raises(DatasetLocked):
        dataset.replace_annotations(next(iter(dataset.images)), [])


def test_stop_modes(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    run = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    run_dir = Path(run["path"])
    manager.stop(None, run["id"], "pause")
    assert json.loads((run_dir / "workbench" / "control.json").read_text())["action"] == "pause"
    manager.stop(None, run["id"], "finish")
    assert json.loads((run_dir / "workbench" / "control.json").read_text())["action"] == "finish"
    manager.stop(None, run["id"], "kill")
    assert launcher.handle.calls == ["terminate"]  # the runner alone; it defers past checkpoint writes
    assert json.loads((run_dir / "workbench" / "control.json").read_text())["action"] == "kill"  # how Windows runners hear it
    launcher.handle.code = -15
    manager.tick()
    summary = manager.run_detail(None, run["id"])
    assert summary["status"] == "stopped" and summary["error"] == "Stopped by user"
    assert not (run_dir / "workbench" / "control.json").exists()
    with pytest.raises(DatasetError, match="not running"):
        manager.stop(None, run["id"], "pause")


def test_force_kill_spares_a_resumed_process_in_the_same_run(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    run = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    run_dir = Path(run["path"])
    old = json.loads((run_dir / "workbench" / "process.json").read_text())
    runner_finishes(run_dir, status="stopped", epochs=2)
    launcher.handle.code = -15
    manager.tick()
    manager.resume(None, run["id"])
    # the resumed runner is a new process: give it a different identity
    (run_dir / "workbench" / "process.json").write_text(json.dumps({"pid": old["pid"], "create_time": (old["create_time"] or 0) + 5}))
    manager._force_kill(run_dir, old["pid"], old["create_time"])
    assert launcher.handle.calls == []
    manager._force_kill(run_dir, old["pid"], (old["create_time"] or 0) + 5)
    assert launcher.handle.calls == ["kill"]


def test_paused_runs_resume_and_final_eval_ends_resumability(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    run = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    run_dir = Path(run["path"])
    runner_finishes(run_dir, status="paused", epochs=2)
    launcher.handle.code = 0
    manager.tick()
    assert manager.run_detail(None, run["id"])["resumable"]

    resumed = manager.resume(None, run["id"])
    assert resumed["active"] and json.loads((run_dir / "workbench" / "spec.json").read_text())["resume"] is True
    assert len(launcher.launches) == 2
    with pytest.raises(DatasetError):
        manager.resume(None, run["id"])
    runner_finishes(run_dir, status="failed", epochs=4, final_eval=True, error="boom")
    launcher.handle.code = 1
    manager.tick()
    detail = manager.run_detail(None, run["id"])
    assert detail["status"] == "failed" and detail["epochs_done"] == 4 and not detail["resumable"]


def test_cancel_removes_unstarted_runs_and_returns_resumes_to_paused(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    active = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    queued = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    with pytest.raises(DatasetError, match="Only queued"):
        manager.cancel(None, active["id"])
    manager.cancel(None, queued["id"])
    assert not Path(queued["path"]).exists()

    runner_finishes(Path(active["path"]), status="paused", epochs=1)
    launcher.handle.code = 0
    blocker = manager.enqueue(None, dataset, "", "", ARGS)["run"]  # takes the slot so the resume stays queued
    manager.resume(None, active["id"])
    manager.cancel(None, active["id"])
    detail = manager.run_detail(None, active["id"])
    assert detail["status"] == "paused" and detail["resumable"] and blocker["id"] != active["id"]


def test_deleted_run_folders_disappear_and_queued_ones_are_dropped(tmp_path):
    import shutil

    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    manager.enqueue(None, dataset, "", "", ARGS)
    queued = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    shutil.rmtree(queued["path"])
    assert [item["name"] for item in manager.list_runs(None)["items"]] == ["exp-1"]
    with pytest.raises(KeyError):
        manager.run_detail(None, queued["id"])
    launcher.handle.code = 0
    manager.tick()
    assert manager.queue() == {"active": None, "queued": []} and len(launcher.launches) == 1


def test_delete_run_removes_the_folder_but_not_while_running_queued_or_a_base(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    first = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    with pytest.raises(DatasetError, match="in progress"):
        manager.delete_run(None, first["id"])
    runner_finishes(Path(first["path"]))
    (Path(first["path"]) / "results.png").write_bytes(b"png")
    launcher.handle.code = 0

    # a run fine-tuning from first's weights keeps them until it ends
    finetune = f"model: {Path(first['path']).as_posix()}/weights/best.pt\nepochs: 5\nimgsz: 64\n"
    second = manager.enqueue(None, dataset, "", "", finetune)["run"]
    third = manager.enqueue(None, dataset, "", "", finetune)["run"]
    assert second["active"] and third["queue_position"] == 1
    with pytest.raises(DatasetError, match="queued"):
        manager.delete_run(None, third["id"])
    manager.cancel(None, third["id"])
    with pytest.raises(DatasetError, match="'exp-2' starts from this run's weights"):
        manager.delete_run(None, first["id"])
    runner_finishes(Path(second["path"]))
    launcher.handle.code = 0

    size =manager.run_detail(None, first["id"])["bytes"]
    assert size == sum(path.stat().st_size for path in Path(first["path"]).rglob("*") if path.is_file())
    assert manager.delete_run(None, first["id"]) == {"deleted": first["id"], "bytes": size}
    assert not Path(first["path"]).exists() and [path.name for path in (dataset.root / "runs").iterdir()] == ["exp-2"]
    assert [item["id"] for item in manager.list_runs(None)["items"]] == [second["id"]]
    with pytest.raises(KeyError):
        manager.delete_run(None, first["id"])


def test_delete_external_run(tmp_path):
    dataset = make_dataset(tmp_path)
    external = dataset.root / "runs" / "detect" / "train"
    (external / "weights").mkdir(parents=True)
    (external / "args.yaml").write_text(yaml.safe_dump({"model": "yolo11n.pt", "data": str(dataset.yaml_path)}))
    (external / "weights" / "best.pt").write_bytes(b"x")
    manager = make_manager(dataset, tmp_path)
    run = manager.list_runs(None)["items"][0]
    manager.delete_run(None, run["id"])
    assert not external.exists() and manager.list_runs(None)["items"] == []


def test_preflight_errors_warnings_and_confirmation(tmp_path):
    dataset = make_dataset(tmp_path, splits=("train",))
    manager = make_manager(dataset, tmp_path)
    with pytest.raises(DatasetError, match="val split has no images"):
        manager.enqueue(None, dataset, "", "", ARGS)

    dataset = make_dataset(tmp_path / "two")
    for path in (dataset.root / "labels" / "train").glob("*.txt"):
        path.write_text("0 0.5 0.5 0.2 0.2\n")
    dataset.rescan()
    manager = make_manager(dataset, tmp_path)
    result = manager.enqueue(None, dataset, "", "", ARGS)
    assert result == {"queued": False, "warnings": ["No training examples for: dog"], "notes": []}
    assert manager.list_runs(None)["items"] == []
    assert manager.enqueue(None, dataset, "", "", ARGS, confirm=True)["queued"]


def test_preflight_rejects_labels_ultralytics_cannot_find(tmp_path):
    # labels next to the images inside an images/ folder: the workbench reads them, Ultralytics would not
    root = tmp_path / "ds"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        Image.new("RGB", (16, 16)).save(root / "images" / split / "a.jpg")
        (root / "images" / split / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "data.yaml").write_text(yaml.safe_dump({"train": "images/train", "val": "images/val", "names": ["a"]}))
    dataset = Dataset(root / "data.yaml", "detection")
    check = make_manager(dataset, tmp_path).preflight(None, dataset)
    assert any("would not find the labels of 2" in error for error in check["errors"])


def test_classification_trains_on_the_root_folder(tmp_path):
    root = tmp_path / "cls"
    for split in ("train", "val"):
        for name in ("a", "b"):
            (root / split / name).mkdir(parents=True)
            Image.new("RGB", (16, 16)).save(root / split / name / f"{split}{name}.jpg")
    dataset = Dataset(root, "classification")
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    with pytest.raises(DatasetError, match="classification model"):
        manager.enqueue(None, dataset, "", "", ARGS)
    manager.enqueue(None, dataset, "", "", "model: yolo11n-cls.pt\n")
    run_dir = launcher.launches[0][0]
    assert run_dir.parent == root / "runs"
    spec = json.loads((run_dir / "workbench" / "spec.json").read_text())
    assert spec["args"]["data"] == str(root) and spec["test_eval"] is None and spec["cache_files"] == []
    assert len(Dataset(root, "classification").images) == 4  # the runs folder is not a class


def test_external_runs_are_listed_and_extra_roots_filtered_by_dataset(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    external = dataset.root / "runs" / "detect" / "train2"
    external.mkdir(parents=True)
    (external / "args.yaml").write_text(yaml.safe_dump({"model": "yolo11s.pt", "data": "whatever.yaml", "epochs": 2, "imgsz": 640, "batch": 16}))
    (external / "results.csv").write_text("epoch,metrics/mAP50-95(B)\n1,0.2\n2,0.4\n")
    (external / "weights").mkdir()
    (external / "weights" / "best.pt").write_bytes(b"x")
    cwd = tmp_path / "elsewhere"
    for name, data in (("mine", str(dataset.yaml_path)), ("other", "/some/other/data.yaml")):
        (cwd / "runs" / "detect" / name).mkdir(parents=True)
        (cwd / "runs" / "detect" / name / "args.yaml").write_text(yaml.safe_dump({"model": "yolo11n.pt", "data": data}))
    monkeypatch.chdir(cwd)
    manager = make_manager(dataset, tmp_path)
    items = {item["name"]: item for item in manager.list_runs(None)["items"]}
    assert set(items) == {"train2", "mine"}
    run = items["train2"]
    assert run["source"] == "external" and run["status"] == "external" and run["best_epoch"] == 2 and run["dir"] == "detect/train2"
    assert not run["resumable"] and run["has_weights"] == {"best": True, "last": False}

    renamed = manager.update_run(None, run["id"], name="Baseline", notes="from the CLI")
    assert renamed["name"] == "Baseline" and renamed["notes"] == "from the CLI" and renamed["source"] == "external"
    copied = manager.copy_args(None, run["id"])
    assert copied["args"] == {"model": "yolo11s.pt", "epochs": 2}  # data dropped, defaults (imgsz/batch) stripped
    assert copied["name"] == "Baseline-2" and copied["yaml"].startswith("model: yolo11s.pt")


def test_run_files_logs_and_epochs(tmp_path):
    dataset = make_dataset(tmp_path)
    manager = make_manager(dataset, tmp_path)
    run = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    run_dir = Path(run["path"])
    (run_dir / "results.png").write_bytes(b"png")
    (run_dir / "test").mkdir()
    (run_dir / "test" / "confusion_matrix.png").write_bytes(b"png")
    assert manager.run_file(None, run["id"], "test/confusion_matrix.png") == (run_dir / "test" / "confusion_matrix.png").resolve()
    for bad in ("../../data.yaml", "workbench/spec.json", "missing.png"):
        with pytest.raises(KeyError):
            manager.run_file(None, run["id"], bad)
    assert manager.run_detail(None, run["id"])["plots"] == ["results.png", "test/confusion_matrix.png"]

    log = run_dir / "workbench" / "train.log"
    log.write_bytes(b"line one\nprogress 1\rprogress 2\npartial")
    first = manager.log(None, run["id"])
    assert first["text"] == "line one\nprogress 2\n" and first["offset"] == len(b"line one\nprogress 1\rprogress 2\n")
    with log.open("ab") as stream:
        stream.write(b" line\n")
    assert manager.log(None, run["id"], first["offset"])["text"] == "partial line\n"

    runner_finishes(run_dir, status="training", epochs=3)
    epochs = manager.epochs(None, run["id"], since=1)
    assert [row["epoch"] for row in epochs["items"]] == [2, 3] and epochs["best_epoch"] == 3 and epochs["state"]["status"] == "training"


def test_best_weights_download_name(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    run = manager.enqueue(None, dataset, "Night shift: v2", "", ARGS)["run"]
    with pytest.raises(KeyError):
        manager.best_weights(None, run["id"])  # no best.pt yet
    runner_finishes(Path(run["path"]))
    launcher.handle.code = 0
    best = Path(run["path"]) / "weights" / "best.pt"
    written = time.mktime((2026, 3, 4, 12, 0, 0, 0, 0, -1))
    os.utime(best, (written, written))
    assert manager.best_weights(None, run["id"]) == (best, "ds-Night_shift_v2-64-yolo11n-2026.03.04.pt")

    # a fine-tune names the architecture it descends from, not "best"
    finetune = manager.enqueue(None, dataset, "tuned", "", f"model: {best.as_posix()}\nepochs: 5\nimgsz: [96, 64]\n")["run"]
    runner_finishes(Path(finetune["path"]))
    os.utime(Path(finetune["path"]) / "weights" / "best.pt", (written, written))
    assert manager.best_weights(None, finetune["id"])[1] == "ds-tuned-96x64-yolo11n-2026.03.04.pt"


def test_best_weights_download_api(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media"), training=manager)
    run = manager.enqueue(None, dataset, "baseline", "", ARGS)["run"]

    async def check():
        async with client_for(app) as client:
            url = f"/api/v1/training/runs/{run['id']}/weights/best"
            assert (await client.get(url)).status_code == 404
            runner_finishes(Path(run["path"]))
            launcher.handle.code = 0
            response = await client.get(url)
            assert response.status_code == 200 and response.content == b"x"
            assert response.headers["content-disposition"] == f'attachment; filename="ds-baseline-64-yolo11n-{time.strftime("%Y.%m.%d")}.pt"'
            assert (await client.get("/api/v1/training/runs/nope/weights/best")).status_code == 404

    asyncio.run(check())


def test_options_list_pretrained_models_and_previous_weights(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    monkeypatch.setattr(manager, "devices", lambda: [{"id": "0", "name": "GPU"}])
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "yolo11n.pt").write_bytes(b"x")
    run = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    runner_finishes(Path(run["path"]))
    launcher.handle.code = 0
    options = manager.options(None, dataset)
    names = {item["name"]: item for item in options["pretrained"]}
    assert len(names) == 15 and names["yolo11n.pt"]["cached"] and not names["yolo26x.pt"]["cached"]
    assert options["defaults"]["epochs"] == 100 and "data" not in options["defaults"] and options["help"]["epochs"]
    assert options["runs"][0]["names_match"] is True and options["runs"][0]["weights"]["best"].endswith("weights/best.pt")
    assert options["next_name"] == "exp-2"


def test_preflight_warns_when_base_run_has_other_classes(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    run = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    run_dir = Path(run["path"])
    runner_finishes(run_dir)
    launcher.handle.code = 0
    best = manager.options(None, dataset)["runs"][0]["weights"]["best"]
    assert best == run_dir.as_posix() + "/weights/best.pt"
    assert manager.preflight(None, dataset, {"model": best})["warnings"] == []
    meta = json.loads((run_dir / "workbench" / "run.json").read_text())
    meta["dataset"]["names"] = ["other"]
    (run_dir / "workbench" / "run.json").write_text(json.dumps(meta))
    # the POSIX path options() hands out and the OS-native one both point into the run's weights
    for model in (best, str(run_dir / "weights" / "best.pt")):
        assert any("different classes" in warning for warning in manager.preflight(None, dataset, {"model": model})["warnings"])
    assert manager.preflight(None, dataset, {"model": "yolo11n.pt"})["warnings"] == []


# ---------- dataset scans ignore runs/ ----------


def test_runs_folder_is_not_indexed_or_reported_as_orphan_labels(tmp_path):
    # a colocated dataset whose split is the dataset folder itself
    root = tmp_path / "flat"
    root.mkdir()
    Image.new("RGB", (16, 16)).save(root / "a.jpg")
    (root / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "data.yaml").write_text(yaml.safe_dump({"train": ".", "names": ["a"]}))
    (root / "runs" / "exp").mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(root / "runs" / "exp" / "results.png")
    (root / "runs" / "exp" / "notes.txt").write_text("not a label")
    dataset = Dataset(root / "data.yaml", "detection")
    assert [record.path.name for record in dataset.images.values()] == ["a.jpg"]
    assert dataset.orphan_labels == []


# ---------- runner callbacks ----------


class FakeTrainer:
    def __init__(self, epochs=3):
        self.epochs = epochs
        self.start_epoch = 0
        self.epoch = 0
        self.train_loader = [None] * 4
        self.args = type("Args", (), {"task": "detect"})()
        self.tloss = None
        self.metrics = {"metrics/mAP50-95(B)": 0.5}
        self.lr = {"lr/pg0": 0.01}
        self.fitness = 0.5
        self.best_fitness = 0.5
        self.train_time_start = time.time()
        self.epoch_time = 1.5
        self.stop = False
        self.validator = None

    def label_loss_items(self, loss_items=None, prefix="train"):
        return {f"{prefix}/box_loss": 1.25}

    def _get_memory(self):
        return 0.5


def test_reporter_records_epochs_but_not_the_final_eval(tmp_path):
    reporter = Reporter(tmp_path, resume=False)
    trainer = FakeTrainer()
    reporter.on_pretrain_routine_end(trainer)
    reporter.on_train_epoch_start(trainer)
    for _ in range(4):
        reporter.on_train_batch_end(trainer)
    reporter.on_train_epoch_end(trainer)
    reporter.on_fit_epoch_end(trainer)
    trainer.epoch, trainer.fitness = 1, 0.4
    reporter.on_train_epoch_start(trainer)
    reporter.on_train_epoch_end(trainer)
    trainer.stop = True  # Ultralytics' last epoch
    reporter.on_fit_epoch_end(trainer)
    trainer.epoch = 2
    reporter.on_fit_epoch_end(trainer)  # the call from final_eval(): not an epoch
    rows = read_epochs_jsonl(tmp_path / "epochs.jsonl")
    assert [(row["epoch"], row["is_best"]) for row in rows] == [(1, True), (2, False)]
    assert rows[0]["train/box_loss"] == 1.25 and rows[0]["lr/pg0"] == 0.01 and rows[0]["epoch_time"] == 1.5
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["best_epoch"] == 1 and state["status"] == "validating" and state["final_eval"] is True


def test_reporter_pause_and_finish_requests(tmp_path):
    reporter = Reporter(tmp_path, resume=False)
    trainer = FakeTrainer(epochs=10)
    (tmp_path / "control.json").write_text(json.dumps({"action": "finish"}))
    reporter.on_train_epoch_end(trainer)
    reporter.on_fit_epoch_end(trainer)
    assert trainer.stop and json.loads((tmp_path / "state.json").read_text())["stopped_early"]

    (tmp_path / "pause").mkdir()
    reporter = Reporter(tmp_path / "pause", resume=False)
    trainer = FakeTrainer(epochs=10)
    (tmp_path / "pause" / "control.json").write_text(json.dumps({"action": "pause"}))
    reporter.on_train_epoch_end(trainer)
    with pytest.raises(RunPaused):
        reporter.on_fit_epoch_end(trainer)
    assert len(read_epochs_jsonl(tmp_path / "pause" / "epochs.jsonl")) == 1  # the epoch was recorded first


def test_reporter_defers_a_kill_until_the_checkpoint_is_written(tmp_path):
    reporter = Reporter(tmp_path, resume=False)
    trainer = FakeTrainer(epochs=10)
    with pytest.raises(RunKilled):
        reporter.on_sigterm(signal.SIGTERM, None)  # during batches: leave right away
    (tmp_path / "x").mkdir()
    reporter = Reporter(tmp_path / "x", resume=False)
    reporter.on_train_epoch_end(trainer)  # validation + checkpoint write follow: defer
    reporter.on_sigterm(signal.SIGTERM, None)
    assert reporter.kill_requested
    with pytest.raises(RunKilled):
        reporter.on_fit_epoch_end(trainer)  # called after save_model(): now it is safe
    assert len(read_epochs_jsonl(tmp_path / "x" / "epochs.jsonl")) == 1

    (tmp_path / "y").mkdir()
    reporter = Reporter(tmp_path / "y", resume=False)
    trainer = FakeTrainer(epochs=10)
    trainer.stop = True  # last epoch: final_eval() rewrites the weights after this callback
    reporter.on_train_epoch_end(trainer)
    reporter.on_sigterm(signal.SIGTERM, None)
    reporter.on_fit_epoch_end(trainer)  # still critical
    with pytest.raises(RunKilled):
        reporter.on_train_end(trainer)


def test_reporter_hears_a_kill_through_control_json(tmp_path):
    """Windows has no catchable SIGTERM: the watcher raises it in the main thread instead."""
    reporter = Reporter(tmp_path, resume=False)
    previous = signal.signal(signal.SIGTERM, reporter.on_sigterm)
    stop = threading.Event()
    watcher = threading.Thread(target=reporter.watch_kill, args=(stop, 0.01))
    try:
        (tmp_path / "control.json").write_text(json.dumps({"action": "kill"}))
        watcher.start()
        with pytest.raises(RunKilled):
            for _ in range(500):  # the simulated signal is handled between bytecodes of this thread
                time.sleep(0.01)
        assert reporter.kill_requested
        reporter.on_sigterm(signal.SIGTERM, None)  # a second delivery (the real SIGTERM on POSIX) is ignored
    finally:
        stop.set()
        signal.signal(signal.SIGTERM, previous)


# ---------- API ----------


def client_for(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1", headers={"X-Workbench": "1"})


def test_single_mode_training_api(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = make_manager(dataset, tmp_path, launcher)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media"), training=manager)

    async def run():
        async with client_for(app) as client:
            response = await client.post("/api/v1/training/args/validate", json={"yaml": "model: yolo11n.pt\nepohcs: 3\n"})
            assert response.status_code == 200 and response.json()["errors"][0]["line"] == 2
            response = await client.post("/api/v1/training/runs", json={"name": "api run", "yaml": "epochs: 3\n"})
            assert response.status_code == 400 and "model" in response.json()["detail"]
            response = await client.post("/api/v1/training/runs", json={"name": "api run", "yaml": ARGS})
            run = response.json()["run"]
            assert response.status_code == 200 and run["active"]
            meta = (await client.get("/api/v1/dataset")).json()
            assert meta["training_lock"]["run_id"] == run["id"]
            image_id = next(iter(dataset.images))
            response = await client.put(f"/api/v1/images/{image_id}/annotations", json={"revision": 1, "annotations": []})
            assert response.status_code == 409 and "read-only" in response.json()["detail"]
            assert (await client.get("/api/v1/training/queue")).json()["active"]["id"] == run["id"]
            assert (await client.get(f"/api/v1/training/runs/{run['id']}")).json()["name"] == "api run"
            assert (await client.get("/api/v1/training/runs/nope")).status_code == 404
            (Path(run["path"]) / "results.png").write_bytes(b"\x89PNG")
            response = await client.get(f"/api/v1/training/runs/{run['id']}/files/results.png")
            assert response.status_code == 200 and response.content == b"\x89PNG"
            assert (await client.get(f"/api/v1/training/runs/{run['id']}/files/workbench/spec.json")).status_code == 404
            response = await client.post(f"/api/v1/training/runs/{run['id']}/stop", json={"mode": "finish"})
            assert response.status_code == 200
            runner_finishes(Path(run["path"]))
            launcher.handle.code = 0
            listed = (await client.get("/api/v1/training/runs")).json()
            assert listed["items"][0]["status"] == "completed" and listed["runs_root"] == (dataset.root / "runs").as_posix()
            assert (await client.get(f"/api/v1/training/runs/{run['id']}/epochs?since=2")).json()["items"][0]["epoch"] == 3
            assert (await client.post(f"/api/v1/training/runs/{run['id']}/stop", json={"mode": "pause"})).status_code == 400
            response = await client.patch(f"/api/v1/training/runs/{run['id']}", json={"notes": "good"})
            assert response.json()["notes"] == "good"
            assert (await client.get(f"/api/v1/training/runs/{run['id']}/args")).json()["args"] == {"model": "yolo11n.pt", "epochs": 5, "imgsz": 64}
            assert (await client.get("/api/v1/dataset")).json()["training_lock"] is None
            response = await client.delete(f"/api/v1/training/runs/{run['id']}")
            assert response.status_code == 200 and response.json()["deleted"] == run["id"] and not Path(run["path"]).exists()
            assert (await client.get(f"/api/v1/training/runs/{run['id']}")).status_code == 404
            assert (await client.delete(f"/api/v1/training/runs/{run['id']}")).status_code == 404

    asyncio.run(run())


def test_projects_mode_shared_queue_lock_and_trash_guard(tmp_path):
    launcher = FakeLauncher()
    workspace = Workspace(tmp_path / "projects", jobs=JobManager(synchronous=True), background_probe=False, training_options={"launcher": launcher, "weights_dir": tmp_path / "w"})
    for name in ("Alpha", "Beta"):
        slug = workspace.create(name, "detection", ["cat"])["slug"]
        for split in ("train", "val"):
            Image.new("RGB", (16, 16)).save(workspace.root / slug / "images" / split / "a.jpg")
            (workspace.root / slug / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    app = create_workspace_app(workspace, media_cache=MediaCache(tmp_path / "media"))

    async def run():
        async with client_for(app) as client:
            response = await client.post("/api/v1/projects/alpha/training/runs", json={"yaml": ARGS})
            alpha_run = response.json()["run"]
            assert alpha_run["active"] and Path(alpha_run["path"]).parent == workspace.root / "alpha" / "runs"
            response = await client.post("/api/v1/projects/beta/training/runs", json={"yaml": ARGS})
            beta_run = response.json()["run"]
            assert beta_run["queue_position"] == 1
            queue = (await client.get("/api/v1/training/queue")).json()
            assert queue["active"]["project"] == "alpha" and queue["queued"][0]["project_name"] == "Beta"
            assert (await client.delete("/api/v1/projects/beta")).status_code == 409

            # beta is open now; alpha (training) opens read-only
            response = await client.put(f"/api/v1/projects/alpha/images/{next(iter(workspace.session('alpha').dataset.images))}/annotations", json={"revision": 1, "annotations": []})
            assert response.status_code == 409
            assert (await client.get("/api/v1/projects/alpha/dataset")).json()["training_lock"]["run_id"] == alpha_run["id"]
            response = await client.post("/api/v1/projects/alpha/import/commit", json={"path": str(tmp_path)})
            assert response.status_code == 409

            runner_finishes(Path(alpha_run["path"]))
            launcher.handle.code = 0
            queue = (await client.get("/api/v1/training/queue")).json()
            assert queue["active"]["id"] == beta_run["id"] and queue["queued"] == []
            assert (await client.get("/api/v1/projects/alpha/dataset")).json()["training_lock"] is None
            # beta's data was prepared while alpha was the open project: a temporary Dataset was loaded
            spec = json.loads((Path(beta_run["path"]) / "workbench" / "spec.json").read_text())
            assert Path(spec["args"]["data"]).is_file()

    asyncio.run(run())


# ---------- real training (opt-in) ----------


@pytest.mark.skipif(not os.environ.get("WORKBENCH_SLOW_TESTS"), reason="set WORKBENCH_SLOW_TESTS=1: trains a real model (downloads yolo11n.pt)")
def test_real_training_smoke(tmp_path):
    dataset = make_dataset(tmp_path)
    manager = TrainingManager.for_dataset(dataset, jobs=JobManager(synchronous=True), weights_dir=Path.home() / ".cache" / "yolo-workbench" / "weights")
    manager.attach(dataset)
    run = manager.enqueue(None, dataset, "smoke", "", "model: yolo11n.pt\nepochs: 2\nimgsz: 64\nbatch: 2\nworkers: 0\n", confirm=True)["run"]
    deadline = time.time() + 600
    while time.time() < deadline and manager.run_detail(None, run["id"])["active"]:
        time.sleep(1)
    detail = manager.run_detail(None, run["id"])
    assert detail["status"] == "completed", manager.log(None, run["id"])["text"][-3000:]
    assert detail["epochs_done"] == 2 and detail["val"]["metrics"] and detail["test"]["metrics"]
    assert not list(dataset.root.rglob("*.cache"))
