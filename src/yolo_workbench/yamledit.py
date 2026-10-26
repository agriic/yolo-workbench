"""Minimal edits of dataset YAML files that keep comments and every other line intact."""

from __future__ import annotations

import re

import yaml


def write_key(text: str, key: str, value) -> str:
    """Replace (or append) only the top-level `key:` block of a YAML document.

    Falls back to a full dump when the block cannot be located or the edit would not parse back.
    """
    block = yaml.safe_dump({key: value}, sort_keys=False, allow_unicode=True, default_flow_style=False)
    lines = text.splitlines(keepends=True)
    span = _block(lines, key)
    if span is None:
        candidate = text + ("" if not text or text.endswith("\n") else "\n") + block
    else:
        candidate = "".join(lines[: span[0]]) + block + "".join(lines[span[1] :])
    expected = yaml.safe_load(block)[key]
    try:
        original = (yaml.safe_load(text) or {}) if text.strip() else {}
        parsed = yaml.safe_load(candidate) or {}
        if isinstance(original, dict) and isinstance(parsed, dict) and parsed.get(key) == expected and {k: v for k, v in parsed.items() if k != key} == {k: v for k, v in original.items() if k != key}:
            return candidate
    except yaml.YAMLError:
        pass
    config = yaml.safe_load(text) if text.strip() else {}
    config = config if isinstance(config, dict) else {}
    config[key] = expected
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def remove_key(text: str, key: str) -> str:
    """Drop only the top-level `key:` block (falls back to a full dump when that would not parse back)."""
    lines = text.splitlines(keepends=True)
    span = _block(lines, key)
    if span is None:
        return text
    candidate = "".join(lines[: span[0]]) + "".join(lines[span[1] :])
    try:
        original = yaml.safe_load(text)
        parsed = yaml.safe_load(candidate) or {}
        if isinstance(original, dict) and isinstance(parsed, dict) and parsed == {k: v for k, v in original.items() if k != key}:
            return candidate
        config = dict(original) if isinstance(original, dict) else {}
    except yaml.YAMLError:
        return text
    config.pop(key, None)
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def _block(lines: list[str], key: str) -> tuple[int, int] | None:
    """The line range of the top-level `key:` block, or None."""
    pattern = re.compile(rf"^{re.escape(key)}\s*:")
    start = next((index for index, line in enumerate(lines) if pattern.match(line)), None)
    if start is None:
        return None
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and not line[0].isspace() and not line.startswith("-") and not line.lstrip().startswith("#"):
            break
        end += 1
    while end > start + 1 and (not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")):
        end -= 1  # trailing blank lines and comments belong to whatever follows
    return start, end
