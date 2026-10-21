"""Server configuration (`yolo-workbench-server --config config.toml`).

Every setting has a default; a TOML file, `WORKBENCH_*` environment variables and CLI options override
it, strongest last: CLI > environment > file > default. The file groups settings in tables
([server], [storage], ...); the environment uses the flat field name (`WORKBENCH_PORT`,
`WORKBENCH_TRAINING_PAUSE_DEADLINE_S`). Relative paths in the file are relative to the file, elsewhere
to the working directory. Unknown keys are errors, so a typo does not silently keep a default.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

ENV_PREFIX = "WORKBENCH_"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}
WILDCARD_HOSTS = {"0.0.0.0", "::", ""}
LOG_LEVELS = ("debug", "info", "warning", "error")
LOG_FORMATS = ("text", "json")
GPU_POLICIES = ("allow", "refuse")


class ConfigError(ValueError):
    pass


def _setting(default, section: str, kind: str, key: str | None = None, choices: tuple | None = None, minimum: float | None = 0):
    meta = {"section": section, "kind": kind, "key": key, "choices": choices, "minimum": minimum}
    if isinstance(default, list):
        return field(default_factory=lambda: list(default), metadata=meta)
    return field(default=default, metadata=meta)


@dataclass
class ServerConfig:
    # [server]
    projects_root: Path | None = _setting(None, "server", "path")
    create_projects_root: bool = _setting(False, "server", "bool")  # a missing mount must not become an empty workspace
    host: str = _setting("0.0.0.0", "server", "str")
    port: int = _setting(8765, "server", "int", minimum=1)
    allowed_hosts: list = _setting([], "server", "list")  # names and addresses people type; "*" turns the check off
    shutdown_deadline_s: float = _setting(60.0, "server", "float")
    # [storage]
    cache_dir: Path | None = _setting(None, "storage", "path")  # media, weights and library caches; default ~/.cache
    media_cache_mb: int = _setting(512, "storage", "int", minimum=16)
    min_free_disk_gb: float = _setting(5.0, "storage", "float")
    trash_retention_days: int = _setting(30, "storage", "int")  # 0 keeps forever
    backup_retention_days: int = _setting(30, "storage", "int")
    project_trash_retention_days: int = _setting(30, "storage", "int")
    session_rotate_hours: float = _setting(24.0, "storage", "float")  # 0: one session per opening
    staging_max_age_hours: float = _setting(24.0, "storage", "float", minimum=0.1)
    maintenance_interval_min: float = _setting(60.0, "storage", "float", minimum=0.1)
    # [limits]
    max_upload_file_mb: int = _setting(1024, "limits", "int", minimum=1)
    max_upload_files: int = _setting(100_000, "limits", "int", minimum=1)
    max_upload_total_mb: int = _setting(102_400, "limits", "int", minimum=1)
    max_model_upload_mb: int = _setting(4096, "limits", "int", minimum=1)
    undo_max_transactions: int = _setting(200, "limits", "int", minimum=1)
    undo_max_mb: int = _setting(256, "limits", "int", minimum=1)
    # [training]
    training_pause_deadline_s: float = _setting(600.0, "training", "float", key="pause_deadline_s")
    resume_after_service_stop: bool = _setting(True, "training", "bool")
    hold_queue_after_crash: bool = _setting(False, "training", "bool")
    # [models]
    idle_unload_minutes: float = _setting(2.0, "models", "float")  # SAM / prediction models nobody used; 0 keeps them loaded
    gpu_models_during_training: str = _setting("allow", "models", "choice", choices=GPU_POLICIES)
    max_loaded_models: int = _setting(2, "models", "int", minimum=1)  # prediction models across open projects
    # [projects]
    max_open_projects: int = _setting(4, "projects", "int", minimum=1)
    project_idle_close_min: float = _setting(60.0, "projects", "float")  # 0 keeps idle projects open
    # [auth]
    auth_enabled: bool = _setting(True, "auth", "bool", key="enabled")  # sign-in with the accounts of `yolo-workbench-server user`
    session_days: float = _setting(30.0, "auth", "float", minimum=0.01)  # a sign-in lasts this long after its last use
    # [embeddings]
    embeddings_enabled: bool = _setting(True, "embeddings", "bool", key="enabled")
    mongodb_uri: str | None = _setting(None, "embeddings", "str")
    # [logging]
    log_level: str = _setting("info", "logging", "choice", key="level", choices=LOG_LEVELS)
    log_format: str = _setting("text", "logging", "choice", key="format", choices=LOG_FORMATS)
    access_log: bool = _setting(False, "logging", "bool")  # one line per HTTP request (the UI polls every few seconds)

    # ---------- derived ----------

    def allowed_host_set(self) -> set[str] | None:
        """Host header values the guard accepts (lowercase, no port); None turns the check off."""
        hosts = {_host_only(host) for host in self.allowed_hosts}
        if "*" in hosts:
            return None
        if self.host not in WILDCARD_HOSTS:
            hosts.add(_host_only(self.host))
        return hosts | LOCAL_HOSTS

    @property
    def wildcard_bind(self) -> bool:
        return self.host in WILDCARD_HOSTS


def _host_only(value: str) -> str:
    value = str(value).strip().lower()
    if value.startswith("["):
        return value.split("]", 1)[0] + "]"
    if value.count(":") > 1:
        return f"[{value}]"  # a bare IPv6 address
    return value.rsplit(":", 1)[0] if ":" in value else value


def _fields() -> dict[str, object]:
    return {item.name: item for item in fields(ServerConfig)}


def _file_keys() -> dict[tuple[str, str], str]:
    return {(item.metadata["section"], item.metadata["key"] or item.name): item.name for item in fields(ServerConfig)}


def env_name(name: str) -> str:
    return ENV_PREFIX + name.upper()


def _coerce(name: str, value, base: Path | None):
    item = _fields()[name]
    kind, choices, minimum = item.metadata["kind"], item.metadata["choices"], item.metadata["minimum"]
    origin = f"`{name}`"
    if value is None or (isinstance(value, str) and value.strip() == "" and kind in ("path", "str") and item.default is None):
        if item.default is None:
            return None
    if kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"{origin} must be true or false, not {value!r}")
    if kind in ("int", "float"):
        if isinstance(value, bool):
            raise ConfigError(f"{origin} must be a number, not {value!r}")
        try:
            number = int(value) if kind == "int" and not (isinstance(value, float) and not value.is_integer()) else float(value)
            number = int(number) if kind == "int" else float(number)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{origin} must be a{'n integer' if kind == 'int' else ' number'}, not {value!r}") from exc
        if minimum is not None and number < minimum:
            raise ConfigError(f"{origin} must be at least {minimum:g}, not {number:g}")
        return number
    if kind == "list":
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, (list, tuple)) and all(isinstance(part, str) for part in value):
            return [part.strip() for part in value if part.strip()]
        raise ConfigError(f"{origin} must be a list of strings")
    if kind == "path":
        path = Path(str(value)).expanduser()
        if not path.is_absolute() and base is not None:
            path = base / path
        return path.resolve()
    text = str(value).strip()
    if kind == "choice":
        text = text.lower()
        if text not in choices:
            raise ConfigError(f"{origin} must be one of {', '.join(choices)}, not {value!r}")
    return text


def _read_file(path: Path) -> dict[str, object]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc.strerror or exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid TOML: {exc}") from exc
    keys = _file_keys()
    names = _fields()
    values: dict[str, object] = {}
    for section, table in data.items():
        if isinstance(table, dict):
            for key, value in table.items():
                name = keys.get((section, key))
                if name is None:
                    raise ConfigError(f"Unknown setting [{section}] {key} in {path}")
                values[name] = value
        elif section in names:  # a top-level key spelled as the field name
            values[section] = table
        else:
            raise ConfigError(f"Unknown setting {section} in {path}")
    return values


def load_config(path: Path | None = None, overrides: dict | None = None, environ=None, serving: bool = True) -> ServerConfig:
    """The validated configuration from file, environment, and CLI overrides (None values are ignored).
    serving=False (commands that only need the projects folder) skips the checks of how the server listens."""
    environ = os.environ if environ is None else environ
    values: dict[str, object] = {}
    if path is not None:
        path = Path(path).expanduser().resolve()
        base = path.parent
        for name, value in _read_file(path).items():
            values[name] = _coerce(name, value, base)
    for name in _fields():
        if env_name(name) in environ:
            values[name] = _coerce(name, environ[env_name(name)], Path.cwd())
    for name, value in (overrides or {}).items():
        if name not in _fields():
            raise ConfigError(f"Unknown setting {name}")
        if value is not None:
            values[name] = _coerce(name, value, Path.cwd())
    config = ServerConfig(**values)
    validate(config, serving)
    return config


def validate(config: ServerConfig, serving: bool = True) -> None:
    if config.projects_root is None:
        raise ConfigError("Set the projects folder: [server] projects_root in the config file, WORKBENCH_PROJECTS_ROOT, or --projects")
    if not serving:
        return
    if config.port > 65535:
        raise ConfigError("`port` must be at most 65535")
    if config.wildcard_bind and not config.allowed_hosts:
        raise ConfigError(
            f"`host` {config.host or '(all addresses)'} accepts connections on every address, so list the names and addresses "
            "people use to reach the server in [server] allowed_hosts (or [\"*\"] to accept any Host header)"
        )
