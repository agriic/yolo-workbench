"""The training subprocess: `python -m yolo_workbench.train_runner <run>/workbench/spec.json`.

Runs one Ultralytics training (or resume) and reports through files in <run>/workbench/, which the
server polls; nothing here talks to the server directly, so a run survives server restarts.
RF-DETR runs (`spec.json` framework "rfdetr") are handed to rfdetr_runner.py, which reports through
the same files with the same `Reporter`.

  state.json    rewritten atomically (≤1/s): status, phase, epoch/batch progress, best epoch, error
  epochs.jsonl  one row per completed epoch (the results.csv columns plus fitness/is_best/timing)
  val.json      final validation of best.pt; test.json the same on the test split
  control.json  written by the server: {"action": "pause" | "finish" | "kill"}

Kill: on POSIX the server also sends SIGTERM. Windows cannot deliver a catchable SIGTERM to another
process, so a watcher thread turns control.json's "kill" into a SIGTERM raised in the main thread;
either way `Reporter.on_sigterm` handles it and defers it past checkpoint writes.

Only the standard library and ultralytics (or rfdetr) are imported, so the server process never pays for torch.
"""

from __future__ import annotations

import json
import math
import os
import _thread
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

from .fsutil import replace

STATE_INTERVAL = 1.0  # seconds between state.json rewrites during an epoch
CONTROL_INTERVAL = 0.5  # seconds between control.json checks of the kill watcher


class RunPaused(Exception):
    """Raised from a callback to leave training after an epoch's checkpoint was saved."""


class RunKilled(Exception):
    """SIGTERM from the server (Kill). Raised only where no checkpoint is being written."""


def write_json(path: Path, data) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, default=_json_default), encoding="utf-8")
    replace(temporary, path)


def _json_default(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _number(value):
    """Plain floats for JSON (tensors, numpy scalars); non-finite values become None."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class Reporter:
    """Ultralytics callbacks that mirror training progress into the run's workbench/ files.

    `raise_on_kill=False` (RF-DETR): a kill only sets `kill_requested`; the Lightning callback raises at
    the next batch boundary, where no checkpoint is being written.
    """

    def __init__(self, workdir: Path, resume: bool, raise_on_kill: bool = True):
        self.workdir = workdir
        self.state_path = workdir / "state.json"
        self.epochs_path = workdir / "epochs.jsonl"
        self.control_path = workdir / "control.json"
        self.state: dict = {"status": "preparing", "phase": "preparing", "pid": os.getpid(), "resumed": resume, "started": time.time()}
        self.last_write = 0.0
        self.batch = 0
        self.in_epoch = False  # on_fit_epoch_end also fires from final_eval(), without an epoch before it
        self.finish_requested = False
        # Ultralytics writes last.pt/best.pt in place: a SIGTERM while they are written corrupts them.
        # From the end of an epoch's batches until its checkpoint is saved (and through final_eval(),
        # which rewrites both), a kill only sets a flag and is honored right after the write.
        self.critical = False
        self.kill_requested = False
        self.raise_on_kill = raise_on_kill
        self.epoch_started = time.time()
        self.epoch_times: list[float] = []  # recent epoch durations (train + val) for the ETA
        self.write(force=True)

    # ---------- files ----------

    def write(self, force: bool = False, **changes) -> None:
        self.state.update(changes)
        now = time.time()
        if force or now - self.last_write >= STATE_INTERVAL:
            self.state["updated"] = now
            write_json(self.state_path, self.state)
            self.last_write = now

    def control(self) -> str | None:
        try:
            action = json.loads(self.control_path.read_text(encoding="utf-8")).get("action")
        except (OSError, ValueError, AttributeError):
            return None
        return action if action in {"pause", "finish", "kill"} else None

    def watch_kill(self, stop: threading.Event, interval: float = CONTROL_INTERVAL) -> None:
        """Thread body: a "kill" in control.json becomes a SIGTERM in the main thread (see on_sigterm)."""
        while not stop.wait(interval):
            if self.control() == "kill":
                if self.raise_on_kill:
                    _thread.interrupt_main(signal.SIGTERM)
                else:
                    self.kill_requested = True
                return

    # ---------- callbacks ----------

    def on_pretrain_routine_end(self, trainer) -> None:
        self.write(force=True, status="training", phase="training", epochs=trainer.epochs, epoch=trainer.start_epoch, batches=len(trainer.train_loader), task=trainer.args.task)

    def mean_epoch_time(self) -> float | None:
        recent = self.epoch_times[-5:]  # the first epochs (warm-up, caching) are slower than the rest
        return sum(recent) / len(recent) if recent else None

    def on_train_epoch_start(self, trainer) -> None:
        self.batch = 0
        self.epoch_started = time.time()
        self.write(force=True, status="training", phase="training", epoch=trainer.epoch + 1, epochs=trainer.epochs, batch=0, batches=len(trainer.train_loader))

    def on_train_batch_end(self, trainer) -> None:
        self.batch += 1
        now = time.time()
        if now - self.last_write < STATE_INTERVAL:
            return
        batches = len(trainer.train_loader)
        per_batch = (now - self.epoch_started) / max(self.batch, 1)
        losses = trainer.label_loss_items(trainer.tloss, prefix="train") if trainer.tloss is not None else {}
        epochs_left = max(trainer.epochs - trainer.epoch - 1, 0)
        epoch_time = self.mean_epoch_time() or per_batch * batches
        self.write(
            batch=self.batch,
            batches=batches,
            running_losses={key: _number(value) for key, value in losses.items()},
            gpu_mem=_number(trainer._get_memory()),
            eta=per_batch * (batches - self.batch) + epochs_left * epoch_time,
        )
        if self.control() == "finish":
            self.finish_requested = True

    def on_sigterm(self, signum, frame) -> None:
        if self.kill_requested:  # both the signal and the control.json watcher deliver a kill on POSIX
            return
        self.kill_requested = True
        if self.raise_on_kill and not self.critical:
            raise RunKilled()

    def on_train_epoch_end(self, trainer) -> None:
        self.critical = True
        self.in_epoch = True
        self.write(force=True, phase="validating-epoch")

    def on_fit_epoch_end(self, trainer) -> None:
        if not self.in_epoch:  # the extra call from final_eval(): the final validation of best.pt
            return
        self.in_epoch = False
        epoch = trainer.epoch + 1
        fitness = _number(trainer.fitness)
        is_best = fitness is not None and trainer.best_fitness is not None and trainer.fitness == trainer.best_fitness
        row = {"epoch": epoch, "time": _number(time.time() - trainer.train_time_start)}
        row.update({key: _number(value) for key, value in trainer.label_loss_items(trainer.tloss, prefix="train").items()})
        row.update({key: _number(value) for key, value in (trainer.metrics or {}).items()})
        row.update({key: _number(value) for key, value in (trainer.lr or {}).items()})
        row.update(fitness=fitness, is_best=is_best, epoch_time=_number(trainer.epoch_time), gpu_mem=_number(trainer._get_memory()))
        with self.epochs_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if row["epoch_time"]:
            self.epoch_times.append(row["epoch_time"])
        changes = {"epoch": epoch, "epochs": trainer.epochs, "epoch_time": row["epoch_time"], "running_losses": None, "batch": 0, "eta": (self.mean_epoch_time() or 0) * max(trainer.epochs - epoch, 0)}
        if is_best:
            changes.update(best_epoch=epoch, best_fitness=fitness)
        action = self.control()
        if action == "finish" or self.finish_requested:
            trainer.stop = True
            changes["stopped_early"] = True
        if trainer.stop:
            # final_eval() strips the optimizer from last.pt: from here on the run cannot be resumed
            changes.update(status="validating", phase="final-validation", final_eval=True)
        self.write(force=True, **changes)
        if trainer.stop:
            return  # final_eval() follows and rewrites the weights; stay critical until on_train_end
        self.critical = False
        if self.kill_requested:
            raise RunKilled()
        if action == "pause":
            raise RunPaused()

    def on_train_end(self, trainer) -> None:
        validator = getattr(trainer, "validator", None)
        metrics = getattr(validator, "metrics", None)
        write_json(self.workdir / "val.json", {"metrics": {key: _number(value) for key, value in (trainer.metrics or {}).items()}, "classes": _summary(metrics), "speed": getattr(metrics, "speed", None)})
        self.critical = False
        if self.kill_requested:
            raise RunKilled()


def _summary(metrics) -> list[dict]:
    """Per-class metrics (Ultralytics' summary(); empty when it is not available)."""
    if metrics is None or not hasattr(metrics, "summary"):
        return []
    try:
        rows = metrics.summary()
    except Exception:  # summary() differs between tasks and versions; per-class rows are optional
        return []
    return [{key: (_number(value) if not isinstance(value, str) else value) for key, value in row.items()} for row in rows]


def _test_eval(spec: dict, run_dir: Path, workdir: Path, reporter: Reporter) -> None:
    from ultralytics import YOLO

    test = spec.get("test_eval")
    best = run_dir / "weights" / "best.pt"
    if not test or not best.exists():
        return
    reporter.write(force=True, status="testing", phase="testing")
    args = spec["args"]
    batch = args.get("batch")
    options = {
        "data": test["data"],
        "split": "test",
        "imgsz": args.get("imgsz", 640),
        "batch": batch if isinstance(batch, int) and batch > 0 else 16,
        "project": str(run_dir),
        "name": "test",
        "exist_ok": True,
        "plots": True,
    }
    if args.get("device") is not None:
        options["device"] = args["device"]
    try:
        metrics = YOLO(str(best)).val(**options)
        write_json(workdir / "test.json", {"metrics": {key: _number(value) for key, value in metrics.results_dict.items() if key != "fitness"}, "classes": _summary(metrics), "speed": getattr(metrics, "speed", None)})
    except Exception as exc:  # the run itself succeeded; a failing test eval is only a warning
        traceback.print_exc()
        reporter.write(force=True, test_error=str(exc))


def run(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("framework") == "rfdetr":
        from .rfdetr_runner import run as run_rfdetr

        return run_rfdetr(spec_path, spec)
    from ultralytics import YOLO

    workdir = spec_path.parent
    run_dir = workdir.parent
    resume = bool(spec.get("resume"))
    reporter = Reporter(workdir, resume)
    signal.signal(signal.SIGTERM, reporter.on_sigterm)
    stop_watching = threading.Event()
    threading.Thread(target=reporter.watch_kill, args=(stop_watching,), daemon=True).start()
    existing_caches = {path for path in spec.get("cache_files", []) if Path(path).exists()}
    status = "failed"
    try:
        if resume:
            model = YOLO(str(run_dir / "weights" / "last.pt"))
        else:
            model = YOLO(spec["args"]["model"])
        for event in ("on_pretrain_routine_end", "on_train_epoch_start", "on_train_batch_end", "on_train_epoch_end", "on_fit_epoch_end", "on_train_end"):
            model.add_callback(event, getattr(reporter, event))
        if resume:
            model.train(resume=True)
        else:
            model.train(**{key: value for key, value in spec["args"].items() if key != "model"})
        _test_eval(spec, run_dir, workdir, reporter)
        stop_watching.set()
        reporter.critical = True  # a kill arriving now must not interrupt the final state write
        status = "completed"
        reporter.write(force=True, status=status, phase="done", finished=time.time(), eta=0)
        return 0
    except RunPaused:
        stop_watching.set()
        reporter.critical = True
        status = "paused"
        reporter.write(force=True, status=status, phase="paused", finished=time.time(), eta=None)
        return 0
    except BaseException as exc:  # includes KeyboardInterrupt/SystemExit, so the state never stays "training"
        stop_watching.set()
        reporter.critical = True
        if reporter.kill_requested:  # RunKilled, or an error caused by the kill (e.g. a dataloader worker gone)
            reporter.write(force=True, status="stopped", phase="stopped", error="Stopped by user", finished=time.time(), eta=None)
            return 1
        traceback.print_exc()
        message = str(exc) or exc.__class__.__name__
        reporter.write(force=True, status="failed", phase="failed", error=message, finished=time.time(), eta=None)
        return 1
    finally:
        for path in spec.get("cache_files", []):
            if path not in existing_caches:
                Path(path).unlink(missing_ok=True)


def utf8_output() -> None:
    """Write stdout/stderr (train.log, read back as UTF-8) as UTF-8. Redirected to a file on Windows
    they default to the ANSI code page, where rich's box-drawing tables (RF-DETR's metrics summary)
    raise UnicodeEncodeError and fail the run."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def main(argv: list[str] | None = None) -> int:
    utf8_output()
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m yolo_workbench.train_runner <run>/workbench/spec.json", file=sys.stderr)
        return 2
    return run(Path(argv[0]))


if __name__ == "__main__":
    sys.exit(main())
