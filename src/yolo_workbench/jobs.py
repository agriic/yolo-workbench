"""Background jobs with progress, polled by the UI via /api/v1/jobs.

Imports and image resizing use it. The record shape (project, kind, progress, message,
result/error, cancel) is meant to also fit future long-running work such as
training runs executed in a subprocess.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from typing import Callable

from . import identity

log = logging.getLogger(__name__)

MAX_FINISHED = 50
# jobs that write a dataset's image files: a queued training run and closing the project wait for them
IMAGE_JOBS = {"import": "an import is running", "resize": "images are being resized"}


class JobCancelled(Exception):
    pass


class Job:
    def __init__(self, job_id: str, kind: str, project: str | None, title: str):
        self.id = job_id
        self.kind = kind
        self.project = project
        self.title = title
        self.state = "running"
        self.done = 0
        self.total = 0
        self.message = ""
        self.error: str | None = None
        self.result: dict | None = None
        self.started = time.time()
        self.finished: float | None = None
        self.started_by = identity.who()  # {name, display_name} with sign-in
        self.identity = identity.snapshot()  # the work runs as whoever started it
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    def progress(self, done: int | None = None, total: int | None = None, message: str | None = None) -> None:
        """Report progress; raises JobCancelled once cancellation was requested."""
        with self._lock:
            if done is not None:
                self.done = done
            if total is not None:
                self.total = total
            if message is not None:
                self.message = message
        if self._cancel.is_set():
            raise JobCancelled()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def payload(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "project": self.project,
                "title": self.title,
                "state": self.state,
                "done": self.done,
                "total": self.total,
                "message": self.message,
                "error": self.error,
                "result": self.result,
                "cancel_requested": self._cancel.is_set(),
                "started": self.started,
                "finished": self.finished,
                "started_by": self.started_by,
            }


class JobManager:
    def __init__(self, synchronous: bool = False):
        # synchronous=True runs jobs inline; tests use it to avoid polling threads
        self.synchronous = synchronous
        self._jobs: dict[str, Job] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self.closed = False  # set at shutdown: no new jobs

    def start(self, kind: str, project: str | None, title: str, work: Callable[[Job], dict | None]) -> dict:
        job = Job(f"job-{int(time.time())}-{next(self._counter)}", kind, project, title)
        with self._lock:
            if self.closed:
                from .dataset import DatasetError

                raise DatasetError("The server is shutting down; try again once it is back")
            self._jobs[job.id] = job
            self._prune()
        log.info("job %s started: %s%s", job.id, title, f" (by {job.started_by['name']})" if job.started_by else "")
        if self.synchronous:
            self._run(job, work)
        else:
            thread = threading.Thread(target=self._run, args=(job, work), daemon=True, name=f"job-{kind}")
            with self._lock:
                self._threads[job.id] = thread
            thread.start()
        return job.payload()

    def _run(self, job: Job, work: Callable[[Job], dict | None]) -> None:
        try:
            with identity.restored(job.identity), identity.action(job.kind):
                result = work(job)
            state, error = ("cancelled" if job.cancel_requested else "done"), None
        except JobCancelled:
            result, state, error = None, "cancelled", None
        except Exception as exc:  # surfaced in the UI; never takes the server down
            result, state, error = None, "failed", str(exc)
        with job._lock:
            job.result = result
            job.state = state
            job.error = error
            job.finished = time.time()
        with self._lock:
            self._threads.pop(job.id, None)
        (log.warning if state == "failed" else log.info)("job %s %s%s", job.id, state, f": {error}" if error else "")

    def _prune(self) -> None:
        finished = [job for job in self._jobs.values() if job.state != "running"]
        for job in sorted(finished, key=lambda item: item.started)[:-MAX_FINISHED]:
            self._jobs.pop(job.id, None)

    def get(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError("Job not found")
        return job.payload()

    def list(self, project: str | None = None) -> list[dict]:
        with self._lock:
            jobs = [job for job in self._jobs.values() if project is None or job.project == project]
        return [job.payload() for job in sorted(jobs, key=lambda item: item.started, reverse=True)]

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError("Job not found")
        job._cancel.set()
        return job.payload()

    def close(self, timeout: float) -> list[str]:
        """Shutdown: refuse new jobs, ask the running ones to cancel, and wait up to `timeout` seconds.
        Returns the titles of jobs still running afterwards."""
        with self._lock:
            self.closed = True
            running = [job for job in self._jobs.values() if job.state == "running"]
            threads = dict(self._threads)
        for job in running:
            job._cancel.set()
        end = time.monotonic() + timeout
        for job in running:
            thread = threads.get(job.id)
            if thread is not None:
                thread.join(max(0.0, end - time.monotonic()))
        return [job.title for job in running if job.state == "running"]

    def running(self, project: str | None, kind: str | tuple[str, ...] | None = None) -> list[dict]:
        kinds = (kind,) if isinstance(kind, str) else kind
        with self._lock:
            jobs = [job for job in self._jobs.values() if job.state == "running" and job.project == project and (kinds is None or job.kind in kinds)]
        return [job.payload() for job in jobs]
