"""Projects mode: a folder of projects, each one YOLO dataset plus room for models and runs.

<root>/
  .trash/<slug>--<timestamp>/   deleted projects, restorable
  <slug>/
    project.json                 name, category, description, summary (cards never load a Dataset)
    data.yaml                    always written; Ultralytics-compatible (no `path:` key)
    images/, labels/             detection / segmentation
    train/, val/, test/          classification (<split>/<class>/<images>)
    models/                      reserved for trained/exported models
    runs/                        training runs (see training.py)
  .training/queue.json           the training queue shared by all projects
    .yolo-workbench/             backups, size index, trash, staging, imports.jsonl, cover.jpg

Several projects can be open at once (`max_open`): a request for a project not yet open opens it, and
when the pool is full the least recently used idle project closes. Projects with nothing running close
by themselves after `idle_close` seconds without requests (the maintenance thread calls `close_idle`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

import yaml
from PIL import Image

from .dataset import WORKBENCH_DIR, Dataset, DatasetError, WriteConflict
from .fsutil import replace
from .imaging import upright_rgb
from .jobs import IMAGE_JOBS, JobManager
from .session import DatasetSession
from .training import TrainingManager, TrainingTarget

PROJECT_FILE = "project.json"
TRASH_DIR = ".trash"
SPLITS = ("train", "val", "test")
CATEGORIES = ("detection", "segmentation", "classification")
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
COVER_SIZE = 480
COVER_VERSION = 2  # bump when cover rendering changes; 2: EXIF-upright
TRASH_STAMP = "%Y%m%d-%H%M%S"

log = logging.getLogger(__name__)


class Workspace:
    def __init__(self, root: Path, jobs: JobManager | None = None, background_probe: bool = True, session_factory=None, training_options: dict | None = None, max_open: int = 4):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = jobs or JobManager()
        self.background_probe = background_probe
        self._session_factory = session_factory or self._default_session
        self._lock = threading.RLock()
        self.max_open = max(1, int(max_open))
        self.sessions: OrderedDict[str, DatasetSession] = OrderedDict()  # open projects, least recently used first
        self.staging_max_age = 24 * 3600.0
        self.session_close_timeout = 5.0
        # one queue for all projects (one GPU); runs outlive project switches, so it is not per session
        self.training = TrainingManager(self.root / ".training", self.training_target, jobs=self.jobs, **(training_options or {}))

    # ---------- paths ----------

    @property
    def trash_root(self) -> Path:
        return self.root / TRASH_DIR

    def project_dir(self, slug: str) -> Path:
        if not SLUG_PATTERN.match(slug or ""):
            raise KeyError("Project not found")
        path = (self.root / slug).resolve()
        if path.parent != self.root:
            raise KeyError("Project not found")
        return path

    def _read_project(self, directory: Path) -> dict:
        try:
            data = json.loads((directory / PROJECT_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise KeyError("Project not found") from exc
        if not isinstance(data, dict) or data.get("category") not in CATEGORIES:
            raise KeyError("Project not found")
        return data

    def _write_project(self, directory: Path, data: dict) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".project.", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(data, stream, indent=2)
                stream.write("\n")
            replace(temporary, directory / PROJECT_FILE)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def info(self, slug: str) -> dict:
        directory = self.project_dir(slug)
        data = self._read_project(directory)
        return self._card(slug, directory, data)

    def _card(self, slug: str, directory: Path, data: dict) -> dict:
        return {
            "slug": slug,
            "name": data.get("name") or slug,
            "category": data["category"],
            "description": data.get("description", ""),
            "created": data.get("created"),
            "updated": data.get("updated"),
            "yaml": data.get("yaml", "data.yaml"),
            "summary": data.get("summary") or {},
            "has_cover": (directory / WORKBENCH_DIR / "cover.jpg").is_file(),
            "active": slug in self.sessions,  # open in this server
        }

    # ---------- listing ----------

    def list_projects(self) -> dict:
        with self._lock:
            for session in list(self.sessions.values()):
                self.refresh_summary(session)
            projects, unmanaged = [], []
            for directory in sorted(self.root.iterdir(), key=lambda item: item.name):
                if not directory.is_dir() or directory.name.startswith("."):
                    continue
                if (directory / PROJECT_FILE).is_file():
                    try:
                        projects.append(self._card(directory.name, directory, self._read_project(directory)))
                    except KeyError:
                        unmanaged.append({"slug": directory.name, "reason": "project.json is unreadable", "category": None, "adoptable": False})
                else:
                    guess = self._guess_unmanaged(directory)
                    if guess is not None:
                        adoptable = bool(SLUG_PATTERN.match(directory.name))
                        if not adoptable:
                            guess["reason"] = "rename the folder to lowercase letters, digits, '-' or '_' to adopt it"
                        unmanaged.append({"slug": directory.name, "adoptable": adoptable, **guess})
            projects.sort(key=lambda item: item.get("updated") or item.get("created") or "", reverse=True)
            return {"root": str(self.root), "projects": projects, "unmanaged": unmanaged}

    @staticmethod
    def _dataset_yaml(directory: Path) -> Path | None:
        preferred = directory / "data.yaml"
        if preferred.is_file():
            return preferred
        for candidate in sorted(directory.glob("*.y*ml")):
            try:
                config = yaml.safe_load(candidate.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(config, dict) and ("names" in config or "train" in config):
                return candidate
        return None

    def _guess_unmanaged(self, directory: Path) -> dict | None:
        """A folder that looks like a dataset but has no project.json; it can be adopted."""
        yaml_path = self._dataset_yaml(directory)
        if yaml_path is not None:
            return {"yaml": yaml_path.name, "category": None, "reason": "dataset without project.json"}
        split_dirs = [directory / split for split in SPLITS if (directory / split).is_dir()]
        if split_dirs and any(child.is_dir() for split in split_dirs for child in split.iterdir()):
            return {"yaml": None, "category": "classification", "reason": "classification folders without project.json"}
        return None

    # ---------- create / update / adopt ----------

    def _slug_for(self, name: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:48] or "project"
        if not base[0].isalnum():
            base = f"p-{base}"
        slug, counter = base, 2
        while (self.root / slug).exists():
            slug = f"{base}-{counter}"
            counter += 1
        return slug

    def create(self, name: str, category: str, classes: list[str] | None = None, description: str = "") -> dict:
        name = str(name).strip()
        if not name:
            raise DatasetError("Project name cannot be empty")
        if category not in CATEGORIES:
            raise DatasetError(f"Category must be one of {', '.join(CATEGORIES)}")
        clean: list[str] = []
        for raw in classes or []:
            class_name = str(raw).strip()
            if not class_name:
                continue
            if any(ch in class_name for ch in "/\\") or class_name in {".", ".."} or (category == "classification" and class_name.startswith(".")):
                raise DatasetError(f"Class name {class_name!r} contains characters that are not allowed")
            if class_name.casefold() in {item.casefold() for item in clean}:
                raise DatasetError(f"Duplicate class name {class_name!r}")
            clean.append(class_name)
        with self._lock:
            slug = self._slug_for(name)
            directory = self.root / slug
            directory.mkdir()
            try:
                if category == "classification":
                    clean.sort()
                    for split in SPLITS:
                        for class_name in clean or [None]:
                            (directory / split / class_name if class_name else directory / split).mkdir(parents=True, exist_ok=True)
                    config = {"train": "train", "val": "val", "test": "test", "names": clean}
                else:
                    for split in SPLITS:
                        (directory / "images" / split).mkdir(parents=True)
                        (directory / "labels" / split).mkdir(parents=True)
                    config = {"train": "images/train", "val": "images/val", "test": "images/test", "names": dict(enumerate(clean))}
                (directory / "models").mkdir()
                (directory / "runs").mkdir()
                header = "# Created by YOLO Workbench. Paths are relative to this file (no `path:` key, so Ultralytics resolves them here too).\n"
                (directory / "data.yaml").write_text(header + yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
                now = _now()
                self._write_project(directory, {
                    "version": 1,
                    "name": name,
                    "category": category,
                    "description": str(description or ""),
                    "yaml": "data.yaml",
                    "created": now,
                    "updated": now,
                    "summary": {"images": 0, "annotated_images": 0, "annotations": 0, "classes": len(clean), "splits": {}},
                })
            except Exception:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            return self.info(slug)

    def update(self, slug: str, name: str | None = None, description: str | None = None) -> dict:
        with self._lock:
            directory = self.project_dir(slug)
            data = self._read_project(directory)
            if name is not None:
                if not str(name).strip():
                    raise DatasetError("Project name cannot be empty")
                data["name"] = str(name).strip()
            if description is not None:
                data["description"] = str(description)
            data["updated"] = _now()
            self._write_project(directory, data)
            if slug in self.sessions:
                self.sessions[slug].project = self._card(slug, directory, data)
            return self.info(slug)

    def adopt(self, slug: str, category: str, name: str | None = None) -> dict:
        if category not in CATEGORIES:
            raise DatasetError(f"Category must be one of {', '.join(CATEGORIES)}")
        with self._lock:
            directory = self.project_dir(slug)
            if not directory.is_dir():
                raise KeyError("Project not found")
            if (directory / PROJECT_FILE).exists():
                raise DatasetError("This folder is already a project")
            yaml_path = self._dataset_yaml(directory)
            if yaml_path is None:
                if category != "classification":
                    raise DatasetError("Detection and segmentation datasets need a dataset YAML to adopt")
                # always keep a data.yaml next to the dataset, also for classification
                splits = {split: split for split in SPLITS if (directory / split).is_dir()}
                yaml_path = directory / "data.yaml"
                names = sorted({child.name for split in splits for child in (directory / split).iterdir() if child.is_dir() and not child.name.startswith(".")})
                yaml_path.write_text(yaml.safe_dump({**splits, "names": names}, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
            Dataset(yaml_path, category, allow_empty=True)  # validates the layout before committing to it
            (directory / "models").mkdir(exist_ok=True)
            (directory / "runs").mkdir(exist_ok=True)
            now = _now()
            self._write_project(directory, {
                "version": 1,
                "name": (name or slug).strip() or slug,
                "category": category,
                "description": "",
                "yaml": yaml_path.name,
                "created": now,
                "updated": now,
                "summary": {},
            })
            return self.info(slug)

    # ---------- trash ----------

    def trash(self, slug: str) -> dict:
        with self._lock:
            directory = self.project_dir(slug)
            data = self._read_project(directory)
            reason = self.training.project_busy(slug)
            if reason:
                raise WriteConflict(f"Project '{data.get('name', slug)}' cannot be deleted: {reason}")
            if slug in self.sessions:
                self.close(slug)
            self.trash_root.mkdir(exist_ok=True)
            entry = f"{slug}--{time.strftime(TRASH_STAMP)}"
            target = self.trash_root / entry
            counter = 2
            while target.exists():
                target = self.trash_root / f"{entry}-{counter}"
                counter += 1
            replace(directory, target)
            return {"entry": target.name, "slug": slug, "name": data.get("name", slug)}

    def list_trash(self) -> list[dict]:
        if not self.trash_root.is_dir():
            return []
        items = []
        for directory in sorted(self.trash_root.iterdir(), key=lambda item: item.name, reverse=True):
            if not directory.is_dir():
                continue
            try:
                data = self._read_project(directory)
            except KeyError:
                data = {}
            slug = directory.name.split("--", 1)[0]
            items.append({"entry": directory.name, "slug": slug, "name": data.get("name", slug), "category": data.get("category"), "deleted": directory.name.split("--", 1)[-1], "summary": data.get("summary") or {}})
        return items

    def _trash_entry(self, entry: str) -> Path:
        path = (self.trash_root / entry).resolve()
        if path.parent != self.trash_root.resolve() or not path.is_dir():
            raise KeyError("Trash entry not found")
        return path

    def restore(self, entry: str) -> dict:
        with self._lock:
            source = self._trash_entry(entry)
            base = source.name.split("--", 1)[0]
            slug = base if SLUG_PATTERN.match(base) else "restored"
            counter = 2
            while (self.root / slug).exists():
                slug = f"{base}-restored-{counter}" if counter > 2 else f"{base}-restored"
                counter += 1
            replace(source, self.root / slug)
            try:
                return self.info(slug)
            except KeyError:
                return {"slug": slug, "name": slug, "category": None}

    def purge(self, entry: str) -> dict:
        with self._lock:
            path = self._trash_entry(entry)
            shutil.rmtree(path)
            return {"purged": entry}

    def purge_expired_trash(self, days: float) -> list[str]:
        """Delete trashed projects older than `days` for good (0: keep them)."""
        if days <= 0 or not self.trash_root.is_dir():
            return []
        cutoff = time.time() - days * 86400
        purged = []
        for item in self.list_trash():
            try:
                deleted = time.mktime(time.strptime(item["deleted"][:15], TRASH_STAMP))
            except ValueError:
                continue
            if deleted < cutoff:
                try:
                    self.purge(item["entry"])
                    purged.append(item["entry"])
                    log.info("workspace: deleted trashed project %s for good (older than %g days)", item["entry"], days)
                except (KeyError, OSError) as exc:
                    log.warning("workspace: could not delete trashed project %s: %s", item["entry"], exc)
        return purged

    # ---------- sessions ----------

    def _default_session(self, slug: str, directory: Path, data: dict) -> DatasetSession:
        yaml_path = directory / data.get("yaml", "data.yaml")
        dataset = Dataset(yaml_path, data["category"], background_probe=self.background_probe, allow_empty=True)
        return DatasetSession.open(dataset, slug=slug)

    @property
    def active(self) -> DatasetSession | None:
        """The most recently used open project."""
        with self._lock:
            return next(reversed(self.sessions.values()), None)

    def open_sessions(self) -> list[DatasetSession]:
        """A snapshot of the open projects. Lock-free on purpose: the training manager calls it (to unload models
        before a run) while holding its own lock, and the workspace calls into training while holding this one."""
        return list(self.sessions.values())

    def busy_reason(self, session: DatasetSession) -> str | None:
        busy = self.jobs.running(session.slug, tuple(IMAGE_JOBS))
        if busy:
            return IMAGE_JOBS[busy[0]["kind"]]
        return session.busy_reason()

    def session(self, slug: str) -> DatasetSession:
        """The open project `slug`, opening it (and closing the least recently used idle one when the pool is full)."""
        with self._lock:
            session = self.sessions.get(slug)
            if session is not None:
                self.sessions.move_to_end(slug)
                session.touch()
                return session
            directory = self.project_dir(slug)
            data = self._read_project(directory)
            while len(self.sessions) >= self.max_open:
                idle = next((open_slug for open_slug, other in self.sessions.items() if not self.busy_reason(other)), None)
                if idle is None:
                    names = ", ".join(f"'{other.project.get('name', other.slug)}'" for other in self.sessions.values())
                    raise WriteConflict(f"{len(self.sessions)} projects are open and all are busy ({names}); wait for one to finish before opening another project")
                self.close(idle)
            _clean_staging(directory, self.staging_max_age)
            session = self._session_factory(slug, directory, data)
            self.training.attach(session.dataset)  # read-only from the start if one of its runs is training
            self.sessions[slug] = session
            session.project = self._card(slug, directory, data)
            self.refresh_summary(session)
            log.info("workspace: opened project %s (%d open)", slug, len(self.sessions))
            return session

    def training_target(self, slug: str | None) -> TrainingTarget:
        """Where a project's runs live. Never takes the workspace lock (the training manager calls it under its own)."""
        if slug is None:
            raise KeyError("Project not found")
        directory = self.project_dir(slug)
        data = self._read_project(directory)
        yaml_path = (directory / data.get("yaml", "data.yaml")).resolve()

        def dataset() -> Dataset:
            session = self.sessions.get(slug)
            if session is not None:
                return session.dataset
            return Dataset(yaml_path, data["category"], background_probe=True, allow_empty=True)

        return TrainingTarget(slug, data.get("name") or slug, data["category"], yaml_path, directory / "runs", dataset)

    def close(self, slug: str) -> None:
        """Close an open project (refused while something runs in it)."""
        with self._lock:
            session = self.sessions.get(slug)
            if session is None:
                return
            reason = self.busy_reason(session)
            if reason:
                raise WriteConflict(f"Project '{session.project.get('name', slug)}' is busy ({reason}); wait or cancel it first")
            del self.sessions[slug]
            self.refresh_summary(session)
        session.close(self.session_close_timeout)
        log.info("workspace: closed project %s", slug)

    def close_active(self) -> None:
        """Close the most recently used project."""
        session = self.active
        if session is not None:
            self.close(session.slug)

    def close_idle(self, idle_seconds: float) -> list[str]:
        """Close projects without requests for `idle_seconds` that have nothing running (0: never)."""
        if idle_seconds <= 0:
            return []
        now = time.monotonic()
        closed = []
        for session in self.open_sessions():
            if now - session.last_used >= idle_seconds and not self.busy_reason(session):
                try:
                    self.close(session.slug)
                    closed.append(session.slug)
                except WriteConflict:
                    pass  # became busy meanwhile
        return closed

    def close_all(self, timeout: float = 5.0) -> None:
        """Shutdown: persist every open project's summary and stop its background work (busy or not)."""
        with self._lock:
            sessions, self.sessions = list(self.sessions.values()), OrderedDict()
        for session in sessions:
            try:
                self.refresh_summary(session)
            except Exception:
                log.exception("workspace: could not save the summary of %s", session.slug)
            session.close(timeout)

    def clean_staging(self) -> None:
        """Remove abandoned browser uploads (untouched for `staging_max_age`) in every open project."""
        for session in self.open_sessions():
            _clean_staging(self.project_dir(session.slug), self.staging_max_age)

    def refresh_summary(self, session: DatasetSession) -> None:
        """Persist the counts shown on project cards, and the cover image (the first image)."""
        if session.slug is None:
            return
        directory = self.root / session.slug
        try:
            data = self._read_project(directory)
        except KeyError:
            return
        dataset = session.dataset
        meta = dataset.metadata()
        with dataset._lock:
            annotated = sum(1 for record in dataset.images.values() if record.annotations)
            annotations = sum(len(record.annotations) for record in dataset.images.values())
            first = dataset.images[dataset._sorted_image_ids[0]] if dataset._sorted_image_ids else None
        summary = {"images": meta["image_count"], "annotated_images": annotated, "annotations": annotations, "classes": len(meta["names"]), "splits": meta["split_counts"]}
        cover_source = str(first.path) if first else None
        cover = directory / WORKBENCH_DIR / "cover.jpg"
        stale = data.get("cover_version") != COVER_VERSION
        if cover_source != data.get("cover_source") or (cover_source and (stale or not cover.exists())):
            if first is None:
                cover.unlink(missing_ok=True)
            else:
                _write_cover(first.path, cover)
            data["cover_source"] = cover_source
            data["cover_version"] = COVER_VERSION
            data["summary"] = None  # force a write below
        if summary != data.get("summary"):
            data["summary"] = summary
            data["updated"] = _now()
            self._write_project(directory, data)

    def cover(self, slug: str) -> Path:
        path = self.project_dir(slug) / WORKBENCH_DIR / "cover.jpg"
        if not path.is_file():
            raise KeyError("No cover image")
        return path


def _write_cover(image_path: Path, cover: Path) -> None:
    try:
        with Image.open(image_path) as source:
            image = upright_rgb(source)
            image.thumbnail((COVER_SIZE, COVER_SIZE))
            cover.parent.mkdir(parents=True, exist_ok=True)
            image.save(cover, "JPEG", quality=82)
    except OSError:
        cover.unlink(missing_ok=True)  # an unreadable first image just means no cover


def _clean_staging(directory: Path, max_age: float = 24 * 3600) -> None:
    from .importer import clean_staging

    clean_staging(directory / WORKBENCH_DIR / "staging", max_age)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

