"""Training runs (Ultralytics YOLO or RF-DETR): run history, a persistent single-GPU queue, and the dataset lock.

A run is a directory under the dataset's runs root (`<project>/runs/` or `<dataset dir>/runs/`):
the trainer's own output plus `workbench/` (see train_runner.py for the files the subprocess writes).
What depends on the trainer (arguments, weights, results files) is asked of its `frameworks.Framework`.
Runs made outside the app are listed too: `yolo train` runs from their args.yaml/results.csv, RF-DETR
runs from their training_config.json/metrics.csv.

`TrainingManager` owns the queue (`queue.json` under its state dir): one active run, launched as a
detached subprocess (own session, so it survives server restarts and Ctrl+C), the rest FIFO.
`tick()` advances it: it notices when the active process is gone, releases the dataset lock,
and launches the next entry. It runs on API reads and from a supervisor thread (CLI only).

While a run is active its dataset is read-only: `attach(dataset)` installs a write guard that
makes every write path raise `DatasetLocked` (HTTP 409).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from . import frameworks
from .dataset import Dataset, DatasetError, DatasetLocked  # noqa: F401  (DatasetLocked is part of this module's API)
from .frameworks import (  # noqa: F401  (re-exported: the Ultralytics argument helpers used to live here)
    MANAGED_KEYS,
    TASKS,
    _read_json,
    _read_yaml,
    _write_json,
    args_to_yaml,
    fitness,
    read_results_csv,
    ultralytics_defaults,
    validate_args,
)
from .fsutil import WINDOWS, replace
from .jobs import Job, JobCancelled, JobManager

META_DIR = "workbench"
ACTIVE_STATES = {"starting", "preparing", "training", "validating", "testing"}
TERMINAL_STATES = {"completed", "paused", "failed", "stopped", "interrupted"}
RESUMABLE_STATES = {"paused", "failed", "stopped", "interrupted"}
PLOT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
LOG_TAIL = 64 * 1024
SCAN_DEPTH = 3
KILL_GRACE = 30.0  # the runner may first finish a checkpoint write; then it and its workers are killed
EXPORT_POLL = 0.5


@dataclass
class TrainingTarget:
    """Where one dataset's runs live and how to get at the dataset itself."""

    project: str | None  # project slug; None in single-dataset mode
    name: str
    category: str
    yaml_path: Path  # dataset YAML (classification: the root directory when there is no YAML)
    runs_root: Path
    dataset: Callable[[], Dataset]  # the open Dataset, or a freshly loaded one
    extra_roots: list[Path] = field(default_factory=list)  # also scanned, keeping only runs of this dataset

    @property
    def dataset_root(self) -> Path:
        return self.yaml_path if self.yaml_path.is_dir() else self.yaml_path.parent


def runs_root_for(yaml_path: Path) -> Path:
    """`<dataset dir>/runs`: next to data.yaml, or inside the classification root."""
    yaml_path = yaml_path.resolve()
    return (yaml_path if yaml_path.is_dir() else yaml_path.parent) / "runs"


# ---------- results ----------


def read_epochs_jsonl(path: Path) -> list[dict]:
    """The runner's epoch rows; a resumed run may repeat an epoch, the last row wins."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    by_epoch: dict[int, dict] = {}
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a line being written right now
        if isinstance(row, dict) and isinstance(row.get("epoch"), int):
            by_epoch[row["epoch"]] = row
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def best_epoch(rows: list[dict], category: str, framework: frameworks.Framework = frameworks.ULTRALYTICS) -> dict | None:
    """The epoch the best weights hold. Ultralytics rewrites best.pt whenever fitness ties the best, so the
    last tie wins; RF-DETR saves only on a strict improvement (and not in its skip_best_epochs), so its
    runner marks the epochs (`is_best`) and the last marked one wins."""
    if framework is frameworks.RFDETR:
        marked = [row for row in rows if row.get("is_best")]
        if marked:
            return marked[-1]
    best, best_value = None, None
    for row in rows:
        value = row.get("fitness") if row.get("fitness") is not None else framework.fitness(row, category)
        if value is not None and (best_value is None or value > best_value or value == best_value and framework is not frameworks.RFDETR):
            best, best_value = row, value
    return best


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def collapse_progress(text: str) -> str:
    """Plain text of a console log: ANSI colors dropped, only the last carriage-return segment of each line (tqdm redraws)."""
    text = ANSI_ESCAPE.sub("", text)
    return "\n".join(line.rsplit("\r", 1)[-1] if "\r" in line.rstrip("\r") else line.rstrip("\r") for line in text.split("\n"))


# ---------- processes ----------


class SubprocessHandle:
    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.pid = process.pid

    def poll(self):
        return self.process.poll()

    def terminate(self) -> None:
        terminate_process(self.pid)

    def kill(self) -> None:
        kill_process_tree(self.pid)


def terminate_process(pid: int) -> None:
    """Ask a runner to stop: SIGTERM to the runner alone, which defers it past checkpoint writes.

    Windows has no SIGTERM a process can catch (os.kill there is TerminateProcess), so nothing is sent:
    the runner watches control.json for "kill", which the server writes first on every platform.
    """
    if WINDOWS:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def kill_process_tree(pid: int) -> None:
    """Force-kill a runner and its dataloader workers."""
    if not WINDOWS:
        try:
            os.killpg(pid, signal.SIGKILL)  # start_new_session: the runner leads its own process group
        except ProcessLookupError:
            pass
        return
    import psutil

    try:
        root = psutil.Process(pid)
        # the tree also holds the interpreter a venv's python.exe launcher starts
        processes = root.children(recursive=True) + [root]
    except psutil.Error:
        return
    for process in processes:
        try:
            process.kill()
        except psutil.Error:
            pass


def launch_subprocess(spec_path: Path, run_dir: Path, cwd: Path) -> SubprocessHandle:
    return _spawn("yolo_workbench.train_runner", spec_path, spec_path.parent / "train.log", cwd)


def launch_export(spec_path: Path, log_path: Path, cwd: Path) -> SubprocessHandle:
    """The ONNX exporter (export_runner.py); its log is rewritten by every export."""
    log_path.unlink(missing_ok=True)
    return _spawn("yolo_workbench.export_runner", spec_path, log_path, cwd)


def _spawn(module: str, spec_path: Path, log_path: Path, cwd: Path) -> SubprocessHandle:
    cwd.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    if WINDOWS:
        # no sessions on Windows: a new process group with its own hidden console keeps the run out of
        # the server console's Ctrl+C and close events, and its dataloader workers open no windows
        detach = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    else:
        detach = {"start_new_session": True}
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", module, str(spec_path)],
            cwd=cwd,  # Ultralytics downloads weights (and its AMP-check model) into the cwd
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            # UTF-8 log also from the dataloader worker processes (train_runner reconfigures its own streams)
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8:backslashreplace"},
            **detach,
        )
    finally:
        log.close()
    return SubprocessHandle(process)


def _process_create_time(pid: int) -> float | None:
    try:
        import psutil

        return psutil.Process(pid).create_time()
    except Exception:
        return None


def _process_alive(info: dict | None) -> bool:
    """Is the recorded runner process still running (and not a different process reusing its PID)?"""
    if not info or not isinstance(info.get("pid"), int):
        return False
    try:
        import psutil

        process = psutil.Process(info["pid"])
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
        if info.get("create_time") is not None and abs(process.create_time() - info["create_time"]) > 1:
            return False
        return any("train_runner" in part for part in process.cmdline())
    except Exception:
        return False


# ---------- the manager ----------


class TrainingManager:
    def __init__(self, state_dir: Path, resolve: Callable[[str | None], TrainingTarget], jobs: JobManager | None = None, launcher=None, weights_dir: Path | None = None, exporter=None):
        self.state_dir = state_dir
        self.resolve = resolve
        self.jobs = jobs or JobManager()
        self.launcher = launcher or launch_subprocess
        self.exporter = exporter or launch_export
        self._exports: dict[Path, str] = {}  # run dir -> its latest ONNX export job
        self.weights_dir = weights_dir or Path.home() / ".cache" / "yolo-workbench" / "weights"
        self._lock = threading.RLock()
        self._handle = None  # process handle of a run this server launched (None after a restart)
        self._locked: dict[Path, dict] = {}  # dataset yaml -> lock info; replaced whole, read without the lock
        self._file_cache: dict[tuple, object] = {}
        self._devices: tuple[float, list] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._queue = self._load_queue()
        self.tick()

    @classmethod
    def for_dataset(cls, dataset: Dataset, **kwargs) -> "TrainingManager":
        """Single-dataset mode: one target, state under the dataset's .yolo-workbench/."""
        yaml_path = dataset.yaml_path
        runs_root = runs_root_for(yaml_path)
        cwd_runs = (Path.cwd() / "runs").resolve()
        target = TrainingTarget(None, dataset.root.name, dataset.category, yaml_path, runs_root, lambda: dataset, [cwd_runs] if cwd_runs != runs_root else [])

        def resolve(project):
            if project is not None:
                raise KeyError("Project not found")
            return target

        return cls(dataset.root / ".yolo-workbench" / "training", resolve, **kwargs)

    # ---------- supervisor ----------

    def start(self, interval: float = 2.0) -> None:
        if self._thread is not None:
            return

        def loop():
            while not self._stop.wait(interval):
                try:
                    self.tick()
                except Exception as exc:  # the supervisor must keep going
                    print(f"training supervisor: {exc}", file=sys.stderr)

        self._thread = threading.Thread(target=loop, daemon=True, name="training-supervisor")
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()

    # ---------- queue file ----------

    @property
    def _queue_path(self) -> Path:
        return self.state_dir / "queue.json"

    def _load_queue(self) -> dict:
        data = _read_json(self._queue_path) or {}
        queue = [entry for entry in data.get("queue", []) if isinstance(entry, dict) and entry.get("dir")]
        active = data.get("active") if isinstance(data.get("active"), dict) else None
        return {"active": active, "queue": queue}

    def _save_queue(self) -> None:
        if self._queue["active"] is None and not self._queue["queue"] and not self._queue_path.exists():
            return
        _write_json(self._queue_path, {"version": 1, **self._queue})

    @staticmethod
    def _key(entry: dict) -> str:
        return f"{entry.get('project') or ''}/{entry['dir']}"

    def _entry_dir(self, entry: dict) -> tuple[TrainingTarget, Path]:
        target = self.resolve(entry.get("project"))
        return target, target.runs_root / entry["dir"]

    # ---------- dataset lock ----------

    def attach(self, dataset: Dataset) -> None:
        dataset.write_guard = lambda: self.lock_reason(dataset)

    def lock_info(self, dataset: Dataset) -> dict | None:
        return self._locked.get(dataset.yaml_path)

    def lock_reason(self, dataset: Dataset) -> str | None:
        info = self._locked.get(dataset.yaml_path)
        return info["reason"] if info else None

    def _refresh_lock(self) -> None:
        active = self._queue["active"]
        locked = {}
        if active is not None:
            try:
                target, run_dir = self._entry_dir(active)
                meta = _read_json(run_dir / META_DIR / "run.json") or {}
                locked[target.yaml_path.resolve()] = {"reason": f"training run '{meta.get('name', active['dir'])}' is in progress", "run_id": self.run_id(run_dir), "run": meta.get("name", active["dir"])}
            except KeyError:
                pass
        self._locked = locked

    def project_busy(self, project: str | None) -> str | None:
        """Why a project cannot be moved away right now (it has an active or queued run)."""
        with self._lock:
            if self._queue["active"] and self._queue["active"].get("project") == project:
                return "a training run is in progress"
            if any(entry.get("project") == project for entry in self._queue["queue"]):
                return "it has queued training runs"
        if self.jobs.running(project, "export"):
            return "an ONNX export is running"
        return None

    # ---------- lifecycle ----------

    def tick(self) -> None:
        """Reap a finished active run and launch the next queued one."""
        with self._lock:
            active = self._queue["active"]
            if active is not None and not self._active_alive(active):
                self._finish_active(active)
            while self._queue["active"] is None and self._queue["queue"]:
                entry = self._queue["queue"][0]
                try:
                    target, run_dir = self._entry_dir(entry)
                except KeyError:
                    self._queue["queue"].pop(0)  # the project is gone
                    self._save_queue()
                    continue
                if not (run_dir / META_DIR / "spec.json").is_file():
                    self._queue["queue"].pop(0)  # the run folder was deleted by hand
                    self._save_queue()
                    continue
                if self.jobs.running(entry.get("project"), "import"):
                    self._write_state(run_dir, status="queued", waiting="an import into this dataset is running")
                    break
                self._launch(entry, target, run_dir)
            self._refresh_lock()

    def _active_alive(self, active: dict) -> bool:
        if self._handle is not None:
            return self._handle.poll() is None
        try:
            _, run_dir = self._entry_dir(active)
        except KeyError:
            return False
        return _process_alive(_read_json(run_dir / META_DIR / "process.json"))

    def _write_state(self, run_dir: Path, replace: bool = False, **changes) -> dict:
        path = run_dir / META_DIR / "state.json"
        state = {} if replace else (_read_json(path) or {})
        state.update(changes)
        state["updated"] = time.time()
        _write_json(path, state)
        return state

    def _finish_active(self, active: dict) -> None:
        self._handle = None
        try:
            _, run_dir = self._entry_dir(active)
        except KeyError:
            run_dir = None
        if run_dir is not None and (run_dir / META_DIR).is_dir():
            workdir = run_dir / META_DIR
            state = _read_json(workdir / "state.json") or {}
            control = _read_json(workdir / "control.json") or {}
            if state.get("status") not in TERMINAL_STATES:
                killed = control.get("action") == "kill"
                self._write_state(
                    run_dir,
                    status="stopped" if killed else "interrupted",
                    phase="stopped" if killed else "interrupted",
                    finished=time.time(),
                    eta=None,
                    error=state.get("error") or ("Stopped by user" if killed else "The training process exited unexpectedly; see the log"),
                )
            (workdir / "control.json").unlink(missing_ok=True)
        self._queue["active"] = None
        self._save_queue()

    def _launch(self, entry: dict, target: TrainingTarget, run_dir: Path) -> None:
        workdir = run_dir / META_DIR
        spec = _read_json(workdir / "spec.json") or {}
        framework = frameworks.get(spec.get("framework"))
        try:
            dataset = target.dataset()
            with dataset._lock:  # in-flight writes finish before the guard flips
                plan = self._data_plan(target, dataset, framework)
                if plan["errors"]:
                    raise DatasetError("; ".join(plan["errors"]))
                if framework is frameworks.RFDETR:
                    # the runner builds an RF-DETR dataset folder from this snapshot (see rfdetr_runner.py)
                    _write_json(workdir / "splits.json", self._rfdetr_splits(dataset, plan))
                    spec["data"] = "splits.json"
                    spec["test_eval"] = plan["has_test"]
                else:
                    data = self._write_data(target, plan, workdir)
                    spec["args"] = {**spec.get("args", {}), "data": data["data"], "project": str(target.runs_root), "name": run_dir.name, "exist_ok": True}
                    spec["test_eval"] = {"data": data["data"]} if plan["has_test"] else None
                    spec["cache_files"] = plan["cache_files"]
                _write_json(workdir / "spec.json", spec)
                meta = _read_json(workdir / "run.json") or {}
                meta["dataset"] = {"fingerprint": plan["fingerprint"], "splits": plan["counts"], "names": plan["names"], "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                _write_json(workdir / "run.json", meta)
                (workdir / "control.json").unlink(missing_ok=True)
                self._write_state(run_dir, replace=True, status="starting", phase="starting", resumed=bool(spec.get("resume")), queued_at=None)
                self._queue["queue"].pop(0)
                self._queue["active"] = entry
                self._refresh_lock()
                self._handle = self.launcher(workdir / "spec.json", run_dir, self.weights_dir)
            _write_json(workdir / "process.json", {"pid": self._handle.pid, "create_time": _process_create_time(self._handle.pid), "started": time.time()})
            self._save_queue()
        except Exception as exc:
            if self._queue["active"] is entry:
                self._queue["active"] = None
            elif self._queue["queue"] and self._queue["queue"][0] is entry:
                self._queue["queue"].pop(0)
            self._handle = None
            self._write_state(run_dir, status="failed", phase="failed", error=f"Could not start: {exc}", finished=time.time())
            self._save_queue()
            self._refresh_lock()

    # ---------- dataset → training data ----------

    def _data_plan(self, target: TrainingTarget, dataset: Dataset, framework: frameworks.Framework = frameworks.ULTRALYTICS) -> dict:
        """What the run will train on, plus blocking errors and confirmable warnings."""
        errors: list[str] = []
        warnings: list[str] = []
        notes: list[str] = []
        names = dataset.names
        records = list(dataset.images.values())
        splits = {split: sorted((record for record in records if record.split == split), key=lambda record: str(record.path)) for split in ("train", "val", "test")}
        counts = {split: len(items) for split, items in splits.items()}
        if not names:
            errors.append("The dataset has no classes")
        elif sorted(names) != list(range(len(names))):
            errors.append(f"Class IDs must be contiguous from 0 ({framework.label} requirement); reorder or remove classes first")
        if not counts["train"]:
            errors.append("The train split has no images")
        if target.category == "classification":
            root = dataset.root
            folders = dataset._split_folders
            for split in ("train", "val", "test"):
                listed = folders.get(split, [])
                if listed and listed != [root / split]:
                    errors.append(f"Classification training needs the {split} images in {root / split} (one folder per split)")
            if not counts["val"] and not counts["test"]:
                errors.append("Classification training needs a val (or test) split with images")
            has_test = bool(counts["test"]) and bool(counts["val"])
        else:
            if not counts["val"]:
                errors.append(f"The val split has no images; {framework.label} validates on it every epoch")
            has_test = bool(counts["test"])
            # RF-DETR trains on a folder the runner builds, so only Ultralytics needs the labels where it looks
            mismatched = 0 if framework is frameworks.RFDETR else sum(1 for record in records if Path(_ultralytics_label_path(str(record.path))) != record.label_path)
            if mismatched:
                errors.append(f"Ultralytics would not find the labels of {mismatched} image(s): their labels are next to the images inside an images/ folder; move them to the parallel labels/ folder")
        if not has_test:
            notes.append("No test split: the run is evaluated on val only")
        issue_errors = sum(1 for issue in dataset.issues() if issue.get("severity") == "error")
        if issue_errors:
            skipped = "broken lines are left out" if framework is frameworks.RFDETR else "Ultralytics skips or repairs broken labels"
            warnings.append(f"{issue_errors} validation error(s) in the dataset ({skipped}; see the Validation tab)")
        if target.category == "classification":
            present = {record.annotations[0].class_id for record in splits["train"] if record.annotations}
        else:
            present = {annotation.class_id for record in splits["train"] for annotation in record.annotations}
        missing = [names[class_id] for class_id in sorted(names) if class_id not in present]
        if missing and counts["train"]:
            warnings.append(f"No training examples for: {', '.join(missing[:8])}{' …' if len(missing) > 8 else ''}")
        digest = hashlib.sha1()
        for split in ("train", "val", "test"):
            for record in splits[split]:
                digest.update(f"{split}\0{record.path}\0{dataset._mtimes.get(record.label_path)}\n".encode())
        cache_files = sorted({str(record.label_path.parent.with_suffix(".cache")) for record in records}) if target.category != "classification" and framework is frameworks.ULTRALYTICS else []
        return {
            "errors": errors,
            "warnings": warnings,
            "notes": notes,
            "splits": {split: [str(record.path) for record in items] for split, items in splits.items()},
            "counts": counts,
            "names": [names[class_id] for class_id in sorted(names)],
            "has_test": has_test,
            "fingerprint": digest.hexdigest()[:12],
            "cache_files": cache_files,
            "root": str(dataset.root),
        }

    def _write_data(self, target: TrainingTarget, plan: dict, workdir: Path) -> dict:
        if target.category == "classification":
            return {"data": plan["root"]}
        config: dict = {"path": str(workdir)}
        for split in ("train", "val", "test"):
            if plan["splits"][split]:
                list_path = workdir / f"{split}.txt"
                list_path.write_text("".join(f"{path}\n" for path in plan["splits"][split]), encoding="utf-8", newline="\n")
                config[split] = str(list_path)
        config["names"] = dict(enumerate(plan["names"]))
        data_path = workdir / "data.yaml"
        header = "# Generated by YOLO Workbench for this run: the dataset's images per split, as absolute file lists.\n"
        data_path.write_text(header + yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
        return {"data": str(data_path)}

    @staticmethod
    def _rfdetr_splits(dataset: Dataset, plan: dict) -> dict:
        """The images per split with their parsed labels (broken lines already left out), for the RF-DETR runner."""
        records = {str(record.path): record for record in dataset.images.values()}
        splits = {
            split: [[path, [[annotation.class_id, *annotation.points] for annotation in records[path].annotations]] for path in paths]
            for split, paths in plan["splits"].items()
            if paths
        }
        return {"version": 1, "category": dataset.category, "names": plan["names"], "splits": splits}

    # ---------- runs ----------

    @staticmethod
    def run_id(run_dir: Path) -> str:
        return hashlib.sha1(str(run_dir.resolve()).encode()).hexdigest()[:12]

    def _scan(self, target: TrainingTarget) -> list[Path]:
        found: list[Path] = []

        def walk(directory: Path, depth: int, keep) -> None:
            if (directory / META_DIR / "run.json").is_file() or frameworks.is_external_run(directory):
                if keep(directory):
                    found.append(directory)
                return
            if depth >= SCAN_DEPTH:
                return
            try:
                children = sorted(child for child in directory.iterdir() if child.is_dir() and not child.name.startswith(".") and child.name != "weights")
            except OSError:
                return
            for child in children:
                walk(child, depth + 1, keep)

        if target.runs_root.is_dir():
            walk(target.runs_root, 0, lambda _: True)
        dataset_paths = {target.yaml_path.resolve(), target.dataset_root.resolve()}

        def belongs(directory: Path) -> bool:
            data = (_read_yaml(directory / "args.yaml") or {}).get("data") or ((_read_json(directory / "training_config.json") or {}).get("train_config") or {}).get("dataset_dir")
            if not data:
                return False
            path = Path(str(data))
            return (path if path.is_absolute() else (directory / path)).resolve() in dataset_paths or path.resolve() in dataset_paths

        for extra in target.extra_roots:
            if extra.is_dir():
                walk(extra, 0, belongs)
        return found

    def _runs(self, target: TrainingTarget) -> dict[str, Path]:
        return {self.run_id(path): path for path in self._scan(target)}

    def _require_run(self, target: TrainingTarget, run_id: str) -> Path:
        run_dir = self._runs(target).get(run_id)
        if run_dir is None:
            raise KeyError("Run not found")
        return run_dir

    def _cached(self, path: Path, loader):
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key not in self._file_cache:
            if len(self._file_cache) > 512:
                self._file_cache.clear()
            self._file_cache[key] = loader(path)
        return self._file_cache[key]

    def _epochs(self, run_dir: Path, category: str, framework: frameworks.Framework | None = None) -> list[dict]:
        framework = framework or frameworks.of_run(run_dir)
        rows = self._cached(run_dir / META_DIR / "epochs.jsonl", read_epochs_jsonl)
        if not rows:
            results = run_dir / ("metrics.csv" if framework is frameworks.RFDETR else "results.csv")
            rows = self._cached(results, lambda _: framework.results(run_dir)) or []
        for row in rows:
            if row.get("fitness") is None:
                row["fitness"] = framework.fitness(row, category)
        return rows

    def _queue_position(self, target: TrainingTarget, run_dir: Path) -> int | None:
        for position, entry in enumerate(self._queue["queue"], 1):
            if entry.get("project") == target.project and entry["dir"] == run_dir.name and target.runs_root / entry["dir"] == run_dir:
                return position
        return None

    def _is_active(self, target: TrainingTarget, run_dir: Path) -> bool:
        active = self._queue["active"]
        return active is not None and active.get("project") == target.project and target.runs_root / active["dir"] == run_dir

    def _summary(self, target: TrainingTarget, run_dir: Path) -> dict:
        workdir = run_dir / META_DIR
        meta = _read_json(workdir / "run.json") or {}
        source = meta.get("source") or ("workbench" if (workdir / "spec.json").is_file() else "external")
        state = _read_json(workdir / "state.json") or {}
        spec = _read_json(workdir / "spec.json") or {}
        recorded = meta.get("framework") or spec.get("framework")
        framework = frameworks.get(recorded) if recorded else frameworks.of_run(run_dir)
        args = self._run_args(framework, run_dir, spec)
        rows = self._epochs(run_dir, target.category, framework)
        best = best_epoch(rows, target.category, framework)
        status = state.get("status")
        active = self._is_active(target, run_dir)
        position = self._queue_position(target, run_dir)
        if source == "external":
            status = "external"
        elif status in ACTIVE_STATES and not active:
            status = "interrupted"  # a run the queue no longer tracks (e.g. a crash while the server was down)
        elif status == "queued" and position is None:
            status = "interrupted"
        test = _read_json(workdir / "test.json")
        val = _read_json(workdir / "val.json")
        epochs = state.get("epochs") or args.get("epochs")
        created = meta.get("created") or time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_dir.stat().st_mtime))
        weights = framework.weights(run_dir)
        if framework is frameworks.RFDETR:
            variant = frameworks.RFDETR_VARIANTS.get(str(args.get("model") or ""))
            imgsz, batch = args.get("resolution") or (variant or {}).get("resolution"), args.get("batch_size", 4)
            if not epochs and source == "external":
                epochs = framework.defaults(target.category)["epochs"]
        else:
            imgsz, batch = args.get("imgsz"), args.get("batch")
        return {
            "id": self.run_id(run_dir),
            "dir": (run_dir.relative_to(target.runs_root) if run_dir.is_relative_to(target.runs_root) else run_dir).as_posix(),
            "path": run_dir.as_posix(),
            "name": meta.get("name") or run_dir.name,
            "notes": meta.get("notes", ""),
            "source": source,
            "framework": framework.id,
            "status": status,
            "phase": state.get("phase"),
            "active": active,
            "queue_position": position,
            "waiting": state.get("waiting") if status == "queued" else None,
            "created": created,
            # a resume rewrites args.yaml's model to the run's own last.pt; show what the run started from
            "model": meta.get("base_model") or args.get("model"),
            "variant": args.get("model") if framework is frameworks.RFDETR else None,
            "imgsz": imgsz,
            "batch": batch,
            "epochs": epochs,
            "epochs_done": rows[-1]["epoch"] if rows else 0,
            "epoch": state.get("epoch") if active else (rows[-1]["epoch"] if rows else 0),
            "best_epoch": best["epoch"] if best else None,
            "best_metrics": {key: value for key, value in best.items() if key.startswith("metrics/")} if best else {},
            "best_fitness": best.get("fitness") if best else None,
            "val_metrics": (val or {}).get("metrics") or None,
            "test_metrics": (test or {}).get("metrics") or None,
            "stopped_early": bool(state.get("stopped_early")),
            "error": state.get("error"),
            "test_error": state.get("test_error"),
            "resumable": self._resumable(framework, source, status, state, run_dir, rows, active, position),
            "weights": {kind: path.as_posix() for kind, path in weights.items()},
            "has_weights": {kind: kind in weights for kind in ("best", "last")},
            "onnx": self._onnx_info(run_dir, weights.get("best")),
            "dataset": meta.get("dataset"),
        }

    @staticmethod
    def _resumable(framework, source, status, state, run_dir, rows, active, position) -> bool:
        # final_eval() strips the optimizer from last.pt, after which Ultralytics refuses to resume
        return (
            source == "workbench"
            and status in RESUMABLE_STATES
            and not active
            and position is None
            and not state.get("final_eval")
            and bool(rows)
            and framework.resume_checkpoint(run_dir) is not None
        )

    def _run_args(self, framework: frameworks.Framework, run_dir: Path, spec: dict) -> dict:
        if framework is frameworks.ULTRALYTICS:
            return self._cached(run_dir / "args.yaml", _read_yaml) or spec.get("args") or {}
        return framework.run_args(run_dir, spec)

    def list_runs(self, project: str | None) -> dict:
        self.tick()
        target = self.resolve(project)
        with self._lock:
            items = []
            for run_dir in self._scan(target):
                try:
                    items.append(self._summary(target, run_dir))
                except OSError:
                    continue  # deleted while listing
        items.sort(key=lambda item: item["created"] or "", reverse=True)
        return {"items": items, "category": target.category, "runs_root": target.runs_root.as_posix()}

    def run_detail(self, project: str | None, run_id: str) -> dict:
        self.tick()
        target = self.resolve(project)
        with self._lock:
            run_dir = self._require_run(target, run_id)
            summary = self._summary(target, run_dir)
        workdir = run_dir / META_DIR
        args = frameworks.get(summary["framework"]).run_args(run_dir, _read_json(workdir / "spec.json") or {})
        plots = []
        for directory, prefix in ((run_dir, ""), (run_dir / "test", "test/")):
            if directory.is_dir():
                plots.extend(f"{prefix}{path.name}" for path in sorted(directory.iterdir()) if path.suffix.lower() in PLOT_EXTENSIONS and path.is_file())
        return {
            **summary,
            "bytes": _tree_size(run_dir),
            "args": args,
            "state": _read_json(workdir / "state.json") or {},
            "val": _read_json(workdir / "val.json"),
            "test": _read_json(workdir / "test.json"),
            "plots": plots,
        }

    def epochs(self, project: str | None, run_id: str, since: int = 0) -> dict:
        target = self.resolve(project)
        run_dir = self._require_run(target, run_id)
        framework = frameworks.of_run(run_dir)
        rows = self._epochs(run_dir, target.category, framework)
        best = best_epoch(rows, target.category, framework)
        state = _read_json(run_dir / META_DIR / "state.json") or {}
        return {"items": [row for row in rows if row["epoch"] > since], "best_epoch": best["epoch"] if best else None, "state": state if self._is_active(target, run_dir) else None}

    def log(self, project: str | None, run_id: str, offset: int | None = None) -> dict:
        target = self.resolve(project)
        path = self._require_run(target, run_id) / META_DIR / "train.log"
        try:
            size = path.stat().st_size
        except OSError:
            return {"text": "", "offset": 0, "size": 0}
        start = max(size - LOG_TAIL, 0) if offset is None or offset > size else offset
        with path.open("rb") as stream:
            stream.seek(start)
            data = stream.read(4 * LOG_TAIL)
        end = start + len(data)
        cut = data.rfind(b"\n")
        if cut >= 0 and not data.endswith(b"\n"):
            data, end = data[: cut + 1], start + cut + 1  # a partial last line waits for the next poll
        return {"text": collapse_progress(data.decode("utf-8", "replace")), "offset": end, "size": size, "truncated": start > 0 and offset is None}

    def run_file(self, project: str | None, run_id: str, name: str) -> Path:
        target = self.resolve(project)
        run_dir = self._require_run(target, run_id).resolve()
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir) or path.suffix.lower() not in PLOT_EXTENSIONS or not path.is_file():
            raise KeyError("File not found")
        return path

    def best_weights(self, project: str | None, run_id: str, onnx: bool = False) -> tuple[Path, str]:
        """The best weights (or their ONNX export) and the download name:
        <dataset>-<run>-<imgsz>-<model>-<YYYY.MM.DD best weights written>.pt (.pth for RF-DETR, .onnx)."""
        target = self.resolve(project)
        with self._lock:
            run_dir = self._require_run(target, run_id)
            summary = self._summary(target, run_dir)
        path = Path(summary["weights"].get("best") or run_dir / "weights" / "best.pt")
        try:
            written = path.stat().st_mtime
        except OSError:
            raise KeyError("This run has no best weights") from None
        if onnx:
            if not summary["onnx"]["available"]:
                raise KeyError("This run's best weights have not been exported to ONNX")
            path = path.with_suffix(".onnx")
        imgsz = summary["imgsz"]
        if isinstance(imgsz, (list, tuple)):
            imgsz = "x".join(str(side) for side in imgsz)
        parts = [target.name, summary["name"], imgsz, _model_label(summary["model"]), time.strftime("%Y.%m.%d", time.localtime(written))]
        parts = [_filename_part(part) for part in parts if part not in (None, "")]
        return path, "-".join(part for part in parts if part) + path.suffix

    # ---------- ONNX export ----------

    def _export_job(self, run_dir: Path) -> dict | None:
        job_id = self._exports.get(run_dir)
        if job_id is None:
            return None
        try:
            return self.jobs.get(job_id)
        except KeyError:  # pruned from the job list
            return None

    def _onnx_info(self, run_dir: Path, best: Path | None) -> dict | None:
        """The ONNX export next to the best weights (best.onnx, checkpoint_best_total.onnx); stale once the best
        weights changed after it (a run still training, a resume). Carries a running or failed export job."""
        if best is None:
            return None
        job = self._export_job(run_dir)
        output = best.with_suffix(".onnx")
        info = {"available": False, "file": output.name, "job": job if job and job["state"] in ("running", "failed") else None}
        try:
            stat, weights = output.stat(), best.stat()
        except OSError:
            return info
        record = _read_json(run_dir / META_DIR / "export.json") or {}
        if record.get("output") != output.name or not record.get("weights_mtime_ns"):
            record = {}  # exported outside the workbench: compare the files' times
        stale = record["weights_mtime_ns"] != weights.st_mtime_ns if record else stat.st_mtime_ns < weights.st_mtime_ns
        return {**info, "available": True, "bytes": stat.st_size, "exported": stat.st_mtime, "stale": stale, "options": record.get("options")}

    def export_onnx(self, project: str | None, run_id: str, imgsz=None, dynamic: bool = False, simplify: bool = True, nms: bool = False) -> dict:
        """Export the run's best weights to ONNX next to them: a background job running export_runner.py.

        Ultralytics takes the input size (default: the run's imgsz), a dynamic batch and image size, onnxslim
        simplification, and NMS inside the graph (not for classification). RF-DETR exports at its trained
        resolution, and `dynamic` makes only its batch dynamic."""
        target = self.resolve(project)
        with self._lock:
            run_dir = self._require_run(target, run_id)
            summary = self._summary(target, run_dir)
            if not summary["weights"].get("best"):
                raise DatasetError("This run has no best weights to export yet")
            previous = self._export_job(run_dir)
            if previous and previous["state"] == "running":
                raise DatasetError("This run's best weights are already being exported")
            framework = frameworks.get(summary["framework"])
            best = Path(summary["weights"]["best"])
            if framework is frameworks.RFDETR:
                options = {"dynamic": bool(dynamic)}
            else:
                options = {"imgsz": _export_imgsz(imgsz or summary["imgsz"]), "dynamic": bool(dynamic), "simplify": bool(simplify), "nms": bool(nms) and target.category != "classification"}
            workdir = run_dir / META_DIR
            workdir.mkdir(exist_ok=True)
            output = best.with_suffix(".onnx")
            spec_path = workdir / "export-spec.json"
            scratch = workdir / "export-tmp"
            _write_json(spec_path, {"framework": framework.id, "weights": best.as_posix(), "output": output.as_posix(), "scratch": scratch.as_posix(), **options})

            def work(job: Job) -> dict:
                written = best.stat().st_mtime_ns  # before the runner copies it: weights rewritten meanwhile leave the export stale
                log = workdir / "export.log"
                handle = self.exporter(spec_path, log, workdir)
                try:
                    while handle.poll() is None:
                        job.progress(message=_last_line(log))
                        time.sleep(EXPORT_POLL)
                except JobCancelled:
                    handle.kill()
                    shutil.rmtree(scratch, ignore_errors=True)
                    raise
                if handle.poll() != 0 or not output.is_file():
                    raise DatasetError(f"ONNX export failed: {_last_line(log, error=True) or f'the exporter exited with code {handle.poll()}'}")
                _write_json(workdir / "export.json", {"version": 1, "output": output.name, "weights_mtime_ns": written, "options": options, "exported": time.time()})
                return {"path": output.as_posix(), "bytes": output.stat().st_size}

            job = self.jobs.start("export", project, f"Export '{summary['name']}' to ONNX", work)
            self._exports[run_dir] = job["id"]
        return job

    def update_run(self, project: str | None, run_id: str, name: str | None = None, notes: str | None = None) -> dict:
        target = self.resolve(project)
        with self._lock:
            run_dir = self._require_run(target, run_id)
            path = run_dir / META_DIR / "run.json"
            meta = _read_json(path) or {"version": 1, "source": "external", "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_dir.stat().st_mtime))}
            if name is not None:
                if not name.strip():
                    raise DatasetError("Run name cannot be empty")
                meta["name"] = name.strip()
            if notes is not None:
                meta["notes"] = notes
            _write_json(path, meta)
            self._refresh_lock()
            return self._summary(target, run_dir)

    def delete_run(self, project: str | None, run_id: str) -> dict:
        """Delete a run's folder for good: weights, plots, logs. Not while it runs or waits in the queue,
        nor while an active or queued run is to start from its weights."""
        self.tick()
        target = self.resolve(project)
        with self._lock:
            run_dir = self._require_run(target, run_id)
            if self._is_active(target, run_dir):
                raise DatasetError("This run is in progress; stop it before deleting it")
            if self._queue_position(target, run_dir) is not None:
                raise DatasetError("This run is queued; remove it from the queue before deleting it")
            if (self._export_job(run_dir) or {}).get("state") == "running":
                raise DatasetError("This run's best weights are being exported to ONNX; wait for the export or cancel it first")
            user = self._started_from(run_dir)
            if user:
                raise DatasetError(f"Run '{user}' starts from this run's weights; delete this run once '{user}' has finished or left the queue")
            size = _tree_size(run_dir)
            # out of sight first (_scan skips dot folders): on Windows a file open elsewhere fails the
            # rename, before anything is lost
            doomed = run_dir.with_name(f".deleting-{run_dir.name}-{time.time_ns()}")
            try:
                replace(run_dir, doomed)
            except OSError as exc:
                raise DatasetError(f"Could not delete the run folder ({exc.strerror or exc}); is one of its files open in another program?") from exc
        try:
            shutil.rmtree(doomed)
        except OSError as exc:
            raise DatasetError(f"The run is gone from the list, but some of its files could not be deleted: {doomed.as_posix()} ({exc.strerror or exc})") from exc
        return {"deleted": run_id, "bytes": size}

    def _started_from(self, run_dir: Path) -> str | None:
        """The name of the active or a queued run that fine-tunes from this run's weights (a resume loads its own checkpoint)."""
        entries = ([self._queue["active"]] if self._queue["active"] else []) + self._queue["queue"]
        for entry in entries:
            try:
                _, other = self._entry_dir(entry)
            except KeyError:
                continue
            spec = _read_json(other / META_DIR / "spec.json") or {}
            if other != run_dir and not spec.get("resume") and frameworks.get(spec.get("framework")).started_from(spec.get("args") or {}, run_dir):
                return (_read_json(other / META_DIR / "run.json") or {}).get("name") or other.name
        return None

    def copy_args(self, project: str | None, run_id: str) -> dict:
        """A run's arguments for a new run: managed keys and Ultralytics defaults stripped."""
        target = self.resolve(project)
        run_dir = self._require_run(target, run_id)
        framework = frameworks.of_run(run_dir)
        args = self._run_args(framework, run_dir, _read_json(run_dir / META_DIR / "spec.json") or {})
        ordered = framework.copy_args(args, target.category)
        return {"args": ordered, "framework": framework.id, "yaml": args_to_yaml(ordered), "name": _next_name(self._summary(target, run_dir)["name"], {path.name for path in target.runs_root.iterdir()} if target.runs_root.is_dir() else set())}

    # ---------- options and validation ----------

    def options(self, project: str | None, dataset: Dataset) -> dict:
        target = self.resolve(project)
        names = [dataset.names[class_id] for class_id in sorted(dataset.names)]
        previous = []
        for summary in self.list_runs(project)["items"]:
            if summary["weights"]:
                run_names = (summary.get("dataset") or {}).get("names")
                previous.append({"id": summary["id"], "name": summary["name"], "framework": summary["framework"], "variant": summary["variant"], "weights": summary["weights"], "best_metrics": summary["best_metrics"], "names_match": None if run_names is None else run_names == names})
        available = {}
        for framework in frameworks.FRAMEWORKS.values():
            if target.category not in framework.categories:
                continue
            reason = framework.unavailable()
            entry = {"id": framework.id, "label": framework.label, "available": reason is None, "reason": reason}
            if reason is None:
                entry.update(defaults=framework.defaults(target.category), help=framework.help(), managed=list(framework.managed), pretrained=framework.pretrained(target.category, self.weights_dir))
            available[framework.id] = entry
        ultralytics = available.get("ultralytics", {})
        return {
            "category": target.category,
            "task": TASKS[target.category],
            # the Ultralytics entry's fields at the top level, as before RF-DETR
            "defaults": ultralytics.get("defaults", {}),
            "help": ultralytics.get("help", {}),
            "managed": ultralytics.get("managed", list(MANAGED_KEYS)),
            "pretrained": ultralytics.get("pretrained", []),
            "frameworks": list(available.values()),
            "runs": previous,
            "devices": self.devices(),
            "weights_dir": self.weights_dir.as_posix(),
            "next_name": _next_name(None, {path.name for path in target.runs_root.iterdir()} if target.runs_root.is_dir() else set()),
        }

    def devices(self) -> list[dict]:
        """GPUs from nvidia-smi (cached); the server never initializes CUDA itself."""
        now = time.time()
        if self._devices is not None and now - self._devices[0] < 30:
            return self._devices[1]
        items = []
        try:
            output = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True).stdout
            for line in output.strip().splitlines():
                index, name, total, used = (part.strip() for part in line.split(","))
                items.append({"id": index, "name": name, "memory_total": int(float(total)), "memory_used": int(float(used))})
        except (OSError, subprocess.SubprocessError, ValueError):
            items = []
        self._devices = (now, items)
        return items

    def validate(self, project: str | None, text: str) -> dict:
        target = self.resolve(project)
        result = frameworks.validate(target.category, text)
        result["yaml"] = args_to_yaml(result["args"]) if not result["errors"] else text
        return result

    def preflight(self, project: str | None, dataset: Dataset, args: dict | None = None) -> dict:
        target = self.resolve(project)
        framework = frameworks.for_args(args or {})
        with dataset._lock:
            plan = self._data_plan(target, dataset, framework)
        warnings = list(plan["warnings"])
        errors = list(plan["errors"])
        if framework is frameworks.RFDETR and (args or {}).get("batch_size") == "auto" and frameworks.rfdetr_device((args or {}).get("device")) in (None, "cuda") and not self.devices():
            errors.append("batch_size: auto needs a CUDA GPU, and nvidia-smi found none; set a number")
        if args:
            # resolved paths: the weights path comes back POSIX-style from options(), run folders are OS-native
            for run_dir in self._scan(target):
                if framework.started_from(args, run_dir):
                    run_names = ((_read_json(run_dir / META_DIR / "run.json") or {}).get("dataset") or {}).get("names")
                    if run_names is not None and run_names != plan["names"]:
                        warnings.append("The base run was trained with different classes; its class head is re-initialized")
        return {"errors": errors, "warnings": warnings, "notes": plan["notes"], "splits": plan["counts"]}

    # ---------- queue operations ----------

    def enqueue(self, project: str | None, dataset: Dataset, name: str, notes: str, text: str, confirm: bool = False) -> dict:
        target = self.resolve(project)
        parsed = frameworks.validate(target.category, text)
        framework = frameworks.get(parsed["framework"])
        if parsed["errors"]:
            raise DatasetError("; ".join(f"line {error['line']}: {error['message']}" if error.get("line") else error["message"] for error in parsed["errors"]))
        check = self.preflight(project, dataset, parsed["args"])
        if check["errors"]:
            raise DatasetError("; ".join(check["errors"]))
        if check["warnings"] and not confirm:
            return {"queued": False, "warnings": check["warnings"], "notes": check["notes"]}
        with self._lock:
            target.runs_root.mkdir(parents=True, exist_ok=True)
            existing = {path.name for path in target.runs_root.iterdir()}
            display = (name or "").strip() or _next_name(None, existing)
            directory = _unique_dir(display, existing)
            run_dir = target.runs_root / directory
            workdir = run_dir / META_DIR
            workdir.mkdir(parents=True)
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            _write_json(workdir / "run.json", {"version": 1, "name": display, "notes": notes or "", "created": now, "source": "workbench", "framework": framework.id, "category": target.category, "task": TASKS[target.category], "base_model": framework.base_model(parsed["args"])})
            _write_json(workdir / "spec.json", {"version": 1, "framework": framework.id, "args": parsed["args"], "resume": False})
            self._write_state(run_dir, replace=True, status="queued", phase="queued", queued_at=time.time())
            self._queue["queue"].append({"project": project, "dir": directory})
            self._save_queue()
        self.tick()
        with self._lock:
            return {"queued": True, "run": self._summary(target, run_dir), "warnings": check["warnings"], "notes": check["notes"]}

    def resume(self, project: str | None, run_id: str) -> dict:
        target = self.resolve(project)
        with self._lock:
            run_dir = self._require_run(target, run_id)
            summary = self._summary(target, run_dir)
            if not summary["resumable"]:
                raise DatasetError("This run cannot be resumed (it finished, is queued or running, or has no resumable last.pt)")
            workdir = run_dir / META_DIR
            spec = _read_json(workdir / "spec.json") or {}
            spec["resume"] = True
            _write_json(workdir / "spec.json", spec)
            (workdir / "control.json").unlink(missing_ok=True)
            self._write_state(run_dir, replace=True, status="queued", phase="queued", resumed=True, queued_at=time.time())
            self._queue["queue"].append({"project": project, "dir": run_dir.name})
            self._save_queue()
        self.tick()
        with self._lock:
            return self._summary(target, run_dir)

    def stop(self, project: str | None, run_id: str, mode: str) -> dict:
        if mode not in {"pause", "finish", "kill"}:
            raise DatasetError("Stop mode must be pause, finish, or kill")
        target = self.resolve(project)
        with self._lock:
            run_dir = self._require_run(target, run_id)
            if not self._is_active(target, run_dir):
                raise DatasetError("This run is not running")
            workdir = run_dir / META_DIR
            _write_json(workdir / "control.json", {"action": mode, "at": time.time()})
            if mode == "kill":
                self._stop_process(run_dir, force=False)  # the runner defers it past checkpoint writes
                # the force-kill targets this process, not the run: a quick Resume starts a new one in the same folder
                process = _read_json(workdir / "process.json") or {}
                timer = threading.Timer(KILL_GRACE, self._force_kill, args=(run_dir, process.get("pid"), process.get("create_time")))
                timer.daemon = True
                timer.start()
            return self._summary(target, run_dir)

    def _stop_process(self, run_dir: Path, force: bool) -> None:
        """Ask the active runner to stop (force: kill it and its workers)."""
        if self._handle is not None:
            (self._handle.kill if force else self._handle.terminate)()
            return
        info = _read_json(run_dir / META_DIR / "process.json")
        if _process_alive(info):
            (kill_process_tree if force else terminate_process)(info["pid"])

    def _force_kill(self, run_dir: Path, pid: int | None, create_time: float | None) -> None:
        with self._lock:
            active = self._queue["active"]
            if active is None or not self._active_alive(active):
                return
            try:
                _, active_dir = self._entry_dir(active)
            except KeyError:
                return
            current = _read_json(run_dir / META_DIR / "process.json") or {}
            if active_dir == run_dir and pid is not None and current.get("pid") == pid and current.get("create_time") == create_time:
                self._stop_process(run_dir, force=True)

    def cancel(self, project: str | None, run_id: str) -> dict:
        """Remove a queued run; its folder only holds the workbench files written at enqueue."""
        target = self.resolve(project)
        with self._lock:
            run_dir = self._require_run(target, run_id)
            position = self._queue_position(target, run_dir)
            if position is None:
                raise DatasetError("Only queued runs can be cancelled; stop a running one instead")
            self._queue["queue"].pop(position - 1)
            self._save_queue()
            state = _read_json(run_dir / META_DIR / "state.json") or {}
            if state.get("resumed"):
                self._write_state(run_dir, status="paused", phase="paused", resumed=False)  # back to resumable
                spec = _read_json(run_dir / META_DIR / "spec.json") or {}
                spec["resume"] = False
                _write_json(run_dir / META_DIR / "spec.json", spec)
            elif [child.name for child in run_dir.iterdir()] == [META_DIR]:
                shutil.rmtree(run_dir)
            return {"cancelled": run_id}

    def queue(self) -> dict:
        """The whole queue (all projects): the active run and the waiting ones."""
        self.tick()
        with self._lock:
            def describe(entry: dict) -> dict | None:
                try:
                    target, run_dir = self._entry_dir(entry)
                    return {"key": self._key(entry), "project": entry.get("project"), "project_name": target.name, **self._summary(target, run_dir)}
                except (KeyError, OSError):
                    return None

            active = describe(self._queue["active"]) if self._queue["active"] else None
            queued = [item for item in map(describe, self._queue["queue"]) if item is not None]
            return {"active": active, "queued": queued}

    def reorder(self, keys: list[str]) -> dict:
        with self._lock:
            by_key = {self._key(entry): entry for entry in self._queue["queue"]}
            if sorted(keys) != sorted(by_key):
                raise DatasetError("The order must list every queued run exactly once")
            self._queue["queue"] = [by_key[key] for key in keys]
            self._save_queue()
        return self.queue()

    def active_summary(self) -> dict | None:
        return self.queue()["active"]


def _ultralytics_label_path(image_path: str) -> str:
    """Ultralytics' img2label_paths for one image (the last /images/ becomes /labels/)."""
    marker, replacement = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return replacement.join(image_path.rsplit(marker, 1)).rsplit(".", 1)[0] + ".txt"


def _export_imgsz(imgsz):
    """An export input size: one side, or [height, width]; None leaves Ultralytics' choice (the model's)."""
    if isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)[:2]
        imgsz = imgsz[0] if len(imgsz) == 1 else imgsz
    sides = imgsz if isinstance(imgsz, list) else [imgsz]
    if imgsz is not None and not all(isinstance(side, int) and not isinstance(side, bool) and 32 <= side <= 8192 for side in sides):
        raise DatasetError("The export image size must be whole pixels between 32 and 8192")
    return imgsz


def _last_line(log: Path, error: bool = False) -> str:
    """A log's last line (export progress), or with `error` the line that says why it failed: Ultralytics'
    `export failure` line or a traceback's last line."""
    try:
        with log.open("rb") as stream:
            stream.seek(max(log.stat().st_size - LOG_TAIL, 0))
            lines = [line.strip() for line in collapse_progress(stream.read().decode("utf-8", "replace")).splitlines() if line.strip()]
    except OSError:
        return ""
    if error:
        lines = [line for line in lines if "export failure" in line or re.match(r"[\w.]*(Error|Exception)\b", line)] or lines
    return lines[-1][:300] if lines else ""


def _tree_size(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            pass  # written or removed meanwhile
    return total


def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")[:64] or "run"


def _filename_part(value) -> str:
    """One piece of a download name: path separators, characters Windows rejects and whitespace become `_`."""
    return re.sub(r'[\s<>:"/\\|?*\x00-\x1f]+', "_", str(value)).strip("_.")


def _model_label(model) -> str:
    """What a run started from, by name: yolo11n.pt -> yolo11n. A previous run's weights stand for
    that run's own base model (followed back through fine-tunes), or else that run's folder name."""
    seen: set[Path] = set()
    fallback = ""
    while model:
        path = Path(str(model).replace("\\", "/"))
        if path.parent.name == "weights":
            run_dir = path.parent.parent
        elif (path.parent / META_DIR / "run.json").is_file() or (path.parent / "training_config.json").is_file():
            run_dir = path.parent  # RF-DETR keeps its weights in the run folder itself
        else:
            return path.stem
        fallback = fallback or run_dir.name
        if run_dir in seen:
            break
        seen.add(run_dir)
        meta = _read_json(run_dir / META_DIR / "run.json") or {}
        model = meta.get("base_model") or (_read_yaml(run_dir / "args.yaml") or {}).get("model")
    return fallback


def _unique_dir(name: str, existing: set[str]) -> str:
    base = _slugify(name)
    candidate, counter = base, 2
    while candidate in existing:
        candidate = f"{base}-{counter}"
        counter += 1
    return candidate


def _next_name(previous: str | None, existing: set[str]) -> str:
    """exp-3 -> exp-4 (skipping taken folder names); a fresh dataset starts at exp-1."""
    match = re.match(r"^(.*?)(\d+)$", previous or "")
    base, number = (match.group(1), int(match.group(2)) + 1) if match else ((previous + "-") if previous else "exp-", 2 if previous else 1)
    if not previous:
        numbers = [int(m.group(1)) for m in (re.match(r"^exp-(\d+)$", name) for name in existing) if m]
        number = max(numbers, default=0) + 1
    while _slugify(f"{base}{number}") in existing:
        number += 1
    return f"{base}{number}"
