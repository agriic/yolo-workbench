import os

import pytest

from yolo_workbench import fsutil


def test_replace_retries_while_windows_holds_the_target(tmp_path, monkeypatch):
    source, target = tmp_path / "new", tmp_path / "old"
    source.write_text("new")
    target.write_text("old")
    real_replace, attempts = os.replace, []

    def held_twice(src, dst):
        attempts.append(dst)
        if len(attempts) <= 2:
            raise PermissionError("in use")
        real_replace(src, dst)

    monkeypatch.setattr(fsutil, "WINDOWS", True)
    monkeypatch.setattr(fsutil, "RETRY_DELAYS", (0, 0, 0))
    monkeypatch.setattr(fsutil.os, "replace", held_twice)
    fsutil.replace(source, target)
    assert target.read_text() == "new" and len(attempts) == 3

    attempts.clear()
    monkeypatch.setattr(fsutil, "RETRY_DELAYS", (0,))
    target.write_text("old")
    (tmp_path / "again").write_text("x")
    with pytest.raises(PermissionError):  # still held after the retries: the error surfaces
        fsutil.replace(tmp_path / "again", target)


def test_occupied_ignores_the_same_folder_under_another_case(tmp_path):
    (tmp_path / "Cat").mkdir()
    assert not fsutil.occupied(tmp_path / "dog", tmp_path / "Cat")
    assert not fsutil.occupied(tmp_path / "Cat", tmp_path / "Cat")  # what cat/ is on a case-insensitive file system
    (tmp_path / "dog").mkdir()
    assert fsutil.occupied(tmp_path / "dog", tmp_path / "Cat")
