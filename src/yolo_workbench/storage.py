"""What the workbench keeps inside a dataset: per-session trash and backups under .yolo-workbench/.

Deleting images (or a classification class) moves files into trash/<session>/, and the first
write to each label file in a session copies it into backups/<session>/. Both mirror the
dataset-relative paths, so a session's trash can be put back where it came from. Undo covers
the current session; older sessions are restored or deleted for good from here.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from .dataset import WORKBENCH_DIR, Dataset, DatasetError, IMAGE_EXTENSIONS

KINDS = ("trash", "backups")
SESSION_PATTERN = re.compile(r"^\d{8}-\d{6}$")
FOLDER_TRASH = "_folders"  # classification class folders keep their own subtree (see classes._trash_dir)


def _session_dir(dataset: Dataset, kind: str, session: str) -> Path:
    if kind not in KINDS:
        raise DatasetError("Unknown storage kind")
    if not SESSION_PATTERN.match(session or ""):
        raise DatasetError("Invalid session id")
    path = dataset.root / WORKBENCH_DIR / kind / session
    if not path.is_dir():
        raise KeyError(f"No {kind} for session {session}")
    return path


def _files(directory: Path) -> list[Path]:
    return [path for path in directory.rglob("*") if path.is_file()]


def _started(session: str) -> str | None:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.strptime(session, "%Y%m%d-%H%M%S"))
    except ValueError:
        return None


def summary(dataset: Dataset) -> dict:
    """Every session's trash and backups with file counts and sizes, newest first."""
    result: dict = {"current_session": dataset.session_id, "total_bytes": 0}
    for kind in KINDS:
        base = dataset.root / WORKBENCH_DIR / kind
        sessions = []
        if base.is_dir():
            for directory in sorted((entry for entry in base.iterdir() if entry.is_dir() and SESSION_PATTERN.match(entry.name)), key=lambda entry: entry.name, reverse=True):
                files = _files(directory)
                size = sum(path.stat().st_size for path in files)
                sessions.append({
                    "session": directory.name,
                    "started": _started(directory.name),
                    "files": len(files),
                    "images": sum(1 for path in files if path.suffix.lower() in IMAGE_EXTENSIONS),
                    "bytes": size,
                    "current": directory.name == dataset.session_id,
                })
                result["total_bytes"] += size
        result[kind] = sessions
    return result


def restore_trash(dataset: Dataset, session: str) -> dict:
    """Put a session's trashed files back at their dataset paths (files whose place is taken are kept in the trash)."""
    with dataset._lock:
        dataset.check_writable()
        directory = _session_dir(dataset, "trash", session)
        restored, skipped = 0, []
        for path in sorted(_files(directory)):
            relative = path.relative_to(directory)
            if relative.parts and relative.parts[0] == FOLDER_TRASH:
                relative = Path(*relative.parts[1:])
            target = dataset.root / relative
            if target.exists():
                skipped.append(relative.as_posix())
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            dataset._move_file(path, target)
            restored += 1
        _prune_empty(directory)
    rescan = dataset.rescan()
    return {"restored": restored, "skipped": skipped, "added": rescan["added"], **summary(dataset)}


def purge(dataset: Dataset, kind: str, session: str) -> dict:
    """Delete a session's trash or backups for good. The current session's stay: its undo history refers to them."""
    directory = _session_dir(dataset, kind, session)
    if session == dataset.session_id:
        raise DatasetError(f"The current session's {kind} backs its undo history; it can be deleted after a restart")
    shutil.rmtree(directory)
    return summary(dataset)


def _prune_empty(directory: Path) -> None:
    for path in sorted((entry for entry in directory.rglob("*") if entry.is_dir()), key=lambda entry: len(entry.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass
    try:
        directory.rmdir()
    except OSError:
        pass
