"""What the workbench keeps inside a dataset: per-session trash and backups under .yolo-workbench/.

Deleting images (or a classification class) moves files into trash/<session>/ (resizing images keeps the
originals there too), and the first
write to each label file in a session copies it into backups/<session>/. Both mirror the
dataset-relative paths, so a session's trash can be put back where it came from. Undo covers
the current session (and older trash it still points into); older sessions are restored or
deleted for good from here, or by the server's retention (`purge_expired`).

A session starts when the dataset is opened; a server also starts a new one daily (Dataset.rotate_session).
"""

from __future__ import annotations

import logging
import re
import shutil
import time
from pathlib import Path

from . import fsscan
from .dataset import WORKBENCH_DIR, Dataset, DatasetError, IMAGE_EXTENSIONS
from .runtime import RUNTIME

log = logging.getLogger(__name__)

KINDS = ("trash", "backups")
SESSION_PATTERN = re.compile(r"^\d{8}-\d{6}$")
FOLDER_TRASH = "_folders"  # classification class folders keep their own subtree (see classes._trash_dir)
# image versions the undo history swaps in and out (resize.py): they have no dataset path of their own to go back to
VERSION_TRASH = "_versions"


def _session_dir(dataset: Dataset, kind: str, session: str) -> Path:
    if kind not in KINDS:
        raise DatasetError("Unknown storage kind")
    if not SESSION_PATTERN.match(session or ""):
        raise DatasetError("Invalid session id")
    path = dataset.root / WORKBENCH_DIR / kind / session
    if not path.is_dir():
        raise KeyError(f"No {kind} for session {session}")
    return path


def _files(directory: Path, listing: fsscan.Listing | None = None) -> list[Path]:
    # listed folder by folder (with sizes and times): a session holds a file per edited label or deleted image
    return [Path(path) for path in (listing or fsscan.Listing(stat=False)).files(directory)]


def _started(session: str) -> str | None:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.strptime(session, "%Y%m%d-%H%M%S"))
    except ValueError:
        return None


def summary(dataset: Dataset) -> dict:
    """Every session's trash and backups with file counts and sizes, newest first."""
    referenced = dataset.referenced_trash_sessions()
    result: dict = {"current_session": dataset.session_id, "total_bytes": 0, "retention_days": {"trash": RUNTIME.trash_retention_days, "backups": RUNTIME.backup_retention_days}}
    for kind in KINDS:
        base = dataset.root / WORKBENCH_DIR / kind
        sessions = []
        listing = fsscan.Listing()
        if listing.is_dir(base):
            for directory in sorted((entry for entry in listing.subfolders(base) if SESSION_PATTERN.match(entry.name)), key=lambda entry: entry.name, reverse=True):
                files = _files(directory, listing)
                size = sum((listing.stat(path) or (0, 0))[1] for path in files)
                sessions.append({
                    "session": directory.name,
                    "started": _started(directory.name),
                    "files": len(files),
                    "images": sum(1 for path in files if path.suffix.lower() in IMAGE_EXTENSIONS),
                    "bytes": size,
                    "current": directory.name == dataset.session_id,
                    "in_undo": kind == "trash" and directory.name in referenced,
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
            if relative.parts and relative.parts[0] == VERSION_TRASH:
                continue
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
    """Delete a session's trash or backups for good. The current session's stay, and so does trash the undo
    history can still put back."""
    directory = _session_dir(dataset, kind, session)
    if session == dataset.session_id:
        raise DatasetError(f"The current session's {kind} backs its undo history; it can be deleted once a new session starts (after a restart, or daily on a server)")
    if kind == "trash" and session in dataset.referenced_trash_sessions():
        raise DatasetError("Undo can still put files back from this session's trash; it can be deleted once those edits leave the undo history")
    shutil.rmtree(directory)
    return summary(dataset)


def purge_expired(root: Path, kind: str, days: float, protected: set[str] = frozenset()) -> list[str]:
    """Delete the `kind` sessions of the dataset at `root` whose newest file is older than `days` (0: keep all),
    except the `protected` ones (the open dataset's current session and trash its undo history points into)."""
    base = root / WORKBENCH_DIR / kind
    if days <= 0 or not base.is_dir():
        return []
    cutoff = time.time() - days * 86400
    purged = []
    for directory in base.iterdir():
        if not directory.is_dir() or not SESSION_PATTERN.match(directory.name) or directory.name in protected:
            continue
        listing = fsscan.Listing()
        try:
            newest = max((listing.stat(path) or (0, 0))[0] / 1e9 for path in _files(directory, listing))
        except ValueError:  # an empty session: its folder's time
            try:
                newest = directory.stat().st_mtime
            except OSError:
                continue
        if newest < cutoff:
            shutil.rmtree(directory, ignore_errors=True)
            purged.append(directory.name)
            log.info("storage: deleted %s session %s of %s (older than %g days)", kind, directory.name, root.name, days)
    return purged


def apply_retention(root: Path, dataset: Dataset | None = None) -> dict:
    """The server's retention for one dataset (open or not): expired trash and backup sessions go."""
    current: set[str] = {dataset.session_id} if dataset is not None else set()
    in_undo = dataset.referenced_trash_sessions() if dataset is not None else set()
    return {
        "trash": purge_expired(root, "trash", RUNTIME.trash_retention_days, current | in_undo),
        "backups": purge_expired(root, "backups", RUNTIME.backup_retention_days, current),
    }


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
