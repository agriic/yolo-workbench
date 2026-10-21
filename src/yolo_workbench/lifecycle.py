"""What runs around the HTTP app: startup, an ordered shutdown, and the server's maintenance thread.

Both app factories install a FastAPI lifespan that calls `Lifecycle.startup()` / `shutdown()`. The
desktop CLI leaves training runs alone on exit (they outlive the server); the server profile pauses
the active run at its next checkpoint so a restart resumes it (`pause_training_on_stop`).

Maintenance (server only) wakes every `tick` seconds: unloads idle models, closes idle projects,
starts a new backup/trash session daily, checks free disk space, and every `interval` seconds
removes abandoned uploads and applies the retention of dataset trash/backups and deleted projects.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import storage
from .dataset import WORKBENCH_DIR
from .predictor import free_gpu_memory
from .runtime import RUNTIME

log = logging.getLogger(__name__)


@dataclass
class LifecycleOptions:
    pause_training_on_stop: bool = False
    training_pause_deadline: float = 600.0
    shutdown_deadline: float = 10.0
    maintenance: bool = False
    tick: float = 60.0  # maintenance wake-up
    interval: float = 3600.0  # staging cleanup and retention
    idle_unload: float = 0.0  # seconds; 0 keeps models loaded
    project_idle_close: float = 0.0  # seconds; 0 keeps idle projects open
    session_rotate: float = 0.0  # seconds; 0 keeps one backup/trash session per opening
    staging_max_age: float = 24 * 3600.0
    project_trash_retention_days: float = 0.0
    max_loaded_models: int = 0  # prediction models loaded across open projects; 0 = no limit


class Lifecycle:
    def __init__(self, training, jobs, sam, sessions: Callable[[], list], data_root: Path, options: LifecycleOptions | None = None, workspace=None):
        self.training = training
        self.jobs = jobs
        self.sam = sam
        self.sessions = sessions  # the open DatasetSessions
        self.data_root = Path(data_root)
        self.options = options or LifecycleOptions()
        self.workspace = workspace
        self.stopping = False
        self.disk: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_heavy = 0.0
        self._model_lock = threading.Lock()

    # ---------- startup / shutdown ----------

    def startup(self) -> None:
        self.stopping = False
        self._stop.clear()
        self.check_disk()
        self.training.start()
        if self.options.maintenance and self._thread is None:
            self._last_heavy = time.monotonic() - self.options.interval + 60  # first cleanup a minute after start
            self._thread = threading.Thread(target=self._loop, daemon=True, name="maintenance")
            self._thread.start()
        log.info("workbench ready")

    def shutdown(self) -> None:
        """Ordered shutdown: no new work; the active run asked to pause (server); jobs cancelled and joined;
        open datasets closed (indexes and summaries saved, models dropped); then wait for the pause."""
        self.stopping = True
        started = time.monotonic()
        options = self.options
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(options.shutdown_deadline)
        pausing = self.training.begin_shutdown(pause=options.pause_training_on_stop)
        leftover = self.jobs.close(options.shutdown_deadline)
        if leftover:
            log.warning("shutdown: still running after %.0f s: %s", options.shutdown_deadline, ", ".join(leftover))
        if self.workspace is not None:
            self.workspace.close_all(options.shutdown_deadline)
        else:
            for session in self.sessions():
                session.close(options.shutdown_deadline)
        close = getattr(self.sam, "close", None)
        if close is not None:
            close(options.shutdown_deadline)
        if pausing:
            remaining = max(0.0, options.training_pause_deadline - (time.monotonic() - started))
            self.training.wait_for_pause(remaining)
        self.training.shutdown()
        log.info("shutdown complete after %.1f s", time.monotonic() - started)

    # ---------- models across projects ----------

    def enforce_model_limit(self, keep) -> list[str]:
        """At most `max_loaded_models` prediction models stay loaded across open projects: the least recently used
        others are unloaded after `keep` (a session) loaded one."""
        limit = self.options.max_loaded_models
        if limit <= 0:
            return []
        with self._model_lock:
            loaded = [session for session in self.sessions() if session is not keep and getattr(session.predictor, "backend", None) is not None]
            excess = len(loaded) + (1 if getattr(keep.predictor, "backend", None) is not None else 0) - limit
            unloaded = []
            for session in sorted(loaded, key=lambda item: item.predictor.last_used)[: max(0, excess)]:
                if session.predictor.busy():
                    continue
                name = keep.project.get("name") or keep.slug or "another project"
                if session.predictor.unload("", reason=f"The model was unloaded because project '{name}' loaded one (at most {limit} stay loaded); load it again"):
                    unloaded.append(session.slug)
            if unloaded:
                free_gpu_memory()
            return unloaded

    def gpu_warning(self, loaded: bool) -> str | None:
        """A model loaded while a run trains on the same GPU may run either of them out of memory."""
        run = self.training.active_run_name()
        if loaded and run:
            return f"Training run '{run}' is using the GPU; this model shares its memory and may slow it down or run either out of memory"
        return None

    def refuse_gpu_load(self) -> str | None:
        """Why a model must not load now (server setting gpu_models_during_training = refuse), or None."""
        run = self.training.active_run_name()
        if RUNTIME.gpu_models_during_training == "refuse" and run:
            return f"Training run '{run}' is using the GPU; models can be loaded once it finishes"
        return None

    # ---------- maintenance ----------

    def _loop(self) -> None:
        while not self._stop.wait(self.options.tick):
            try:
                self.maintenance_tick()
            except Exception:  # maintenance must keep going
                log.exception("maintenance failed")

    def maintenance_tick(self, heavy: bool | None = None) -> None:
        options = self.options
        if options.idle_unload > 0:
            freed = self.sam.unload_if_idle(options.idle_unload) if hasattr(self.sam, "unload_if_idle") else False
            for session in self.sessions():
                unload = getattr(session.predictor, "unload_if_idle", None)
                if unload is not None and unload(options.idle_unload):
                    freed = True
            if freed:
                free_gpu_memory()
        if self.workspace is not None and options.project_idle_close > 0:
            self.workspace.close_idle(options.project_idle_close)
        if options.session_rotate > 0:
            for session in self.sessions():
                dataset = session.dataset
                if time.time() - dataset.session_started >= options.session_rotate:
                    log.info("storage: new backup/trash session %s for %s", dataset.rotate_session(), dataset.root.name)
        self.check_disk()
        if heavy is None:
            heavy = time.monotonic() - self._last_heavy >= options.interval
        if heavy:
            self._last_heavy = time.monotonic()
            self.clean_and_retain()

    def clean_and_retain(self) -> None:
        """Abandoned uploads, expired dataset trash/backups (every project, open or not), expired deleted projects."""
        from .importer import clean_staging

        open_datasets = {Path(session.dataset.root).resolve(): session.dataset for session in self.sessions()}
        if self.workspace is not None:
            roots = [directory for directory in self.workspace.root.iterdir() if directory.is_dir() and not directory.name.startswith(".") and (directory / "project.json").is_file()]
        else:
            roots = list(open_datasets)
        for root in roots:
            dataset = open_datasets.get(root.resolve())
            clean_staging(root / WORKBENCH_DIR / "staging", self.options.staging_max_age)
            try:
                storage.apply_retention(root, dataset)
            except OSError as exc:
                log.warning("storage: retention failed for %s: %s", root.name, exc)
        if self.workspace is not None:
            self.workspace.purge_expired_trash(self.options.project_trash_retention_days)

    def check_disk(self) -> dict | None:
        try:
            usage = shutil.disk_usage(self.data_root)
        except OSError:
            self.disk = None
            return None
        low = bool(RUNTIME.min_free_bytes) and usage.free < RUNTIME.min_free_bytes
        if low and not (self.disk or {}).get("low"):
            log.warning("disk: only %.1f GB free on %s (minimum %.1f GB)", usage.free / 2**30, self.data_root, RUNTIME.min_free_bytes / 2**30)
        self.disk = {"free": usage.free, "total": usage.total, "minimum": RUNTIME.min_free_bytes, "low": low}
        return self.disk

    def status(self) -> dict:
        warnings = []
        if self.disk and self.disk["low"]:
            warnings.append(f"Low disk space: {self.disk['free'] / 2**30:.1f} GB free (at least {self.disk['minimum'] / 2**30:.1f} GB should stay free); uploads, imports and new training runs are refused until space is freed")
        return {"disk": self.disk, "warnings": warnings, "stopping": self.stopping}
