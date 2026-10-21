"""Which installed release this process runs (standard library only).

deploy/install.sh puts each release in its own folder, `/opt/yolo-workbench/releases/<name>/`, with the
virtual environment the process runs from (`.venv`) and `release.json` ({name, commit, installed}) beside it.
A source checkout or a pip install has no such file: only the package version is known there.
"""

from __future__ import annotations

import json
import sys
from functools import cache
from pathlib import Path

from . import __version__

RELEASE_FILE = "release.json"


@cache
def release_info() -> dict | None:
    """{name, commit, installed} of the release, or None outside an installed release. Read once: the
    process keeps running its release's files after `current` points at another one."""
    path = Path(sys.prefix).resolve().parent / RELEASE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("name"), str) or not data["name"]:
        return None
    return {"name": data["name"], "commit": data.get("commit") or None, "installed": data.get("installed") or None}


def release_name() -> str | None:
    info = release_info()
    return info["name"] if info else None


def describe() -> str:
    info = release_info()
    if info is None:
        return f"version {__version__}, not an installed release (environment {sys.prefix})"
    details = [f"commit {info['commit'][:12]}" if info["commit"] else None, f"installed {info['installed']}" if info["installed"] else None]
    return f"{info['name']} ({', '.join(detail for detail in details if detail)})" if any(details) else info["name"]
