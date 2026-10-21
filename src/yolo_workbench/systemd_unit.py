"""The server's systemd unit, written from its configuration (`yolo-workbench-server unit`).

The unit names what the configuration decides: the folders the service may write (its sandbox mounts
everything else read-only), the mounts it waits for, and how long a stop may take. deploy/apply-config.sh
writes it to /etc/systemd/system, and install.sh and rollback.sh run that script, so moving the projects or
the cache needs no edit of the unit: change config.toml, run apply-config.sh, restart. The unit reads the
same environment file as the service (`WORKBENCH_*` settings there override the file).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from .config import ENV_PREFIX, ServerConfig, load_config

UNIT = "yolo-workbench"  # the unit's name, also the service account's
PREFIX = Path("/opt/yolo-workbench")  # install.sh's releases, `current` and `previous`
ENV_FILE = "yolo-workbench.env"  # beside config.toml
HOME_FOLDERS = ("/home", "/root", "/run/user")  # what the unit's ProtectHome= hides
STOP_MARGIN_S = 60  # TimeoutStopSec beyond the run's pause and the shutdown
ACCOUNT_PATTERN = re.compile(r"^[a-z_][a-z0-9_.-]{0,31}$")


class UnitError(ValueError):
    pass


def under(path: Path, folders) -> bool:
    path = Path(path).resolve()
    return any(path.is_relative_to(Path(folder).resolve()) for folder in folders)


def read_env_file(path: Path) -> dict[str, str]:
    """The variables of a systemd EnvironmentFile (KEY=VALUE lines, # and ; comments, quoted values); {}
    when it does not exist."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#;" or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\") if value[0] == '"' else value[1:-1]
        values[key] = value
    return values


def load(config_path: Path, env_file: Path | None = None) -> ServerConfig:
    """The configuration as the service will see it: the file plus the unit's environment file."""
    config_path = Path(config_path).expanduser().resolve()
    environ = {key: value for key, value in read_env_file(env_file or config_path.parent / ENV_FILE).items() if key.startswith(ENV_PREFIX)}
    return load_config(config_path, environ=environ, serving=False)  # how it listens does not change the unit


def service_folders(config: ServerConfig) -> list[Path]:
    """The folders the unit lets the service write: the projects folder, then the cache (apply-config.sh relies
    on the order)."""
    root = config.projects_root
    if config.cache_dir is None:
        raise UnitError("set [storage] cache_dir (e.g. \"/var/cache/yolo-workbench\"): the service may write only the folders its configuration names")
    for setting, path in (("[server] projects_root", root), ("[storage] cache_dir", config.cache_dir)):
        if under(path, HOME_FOLDERS):
            raise UnitError(f"{setting} {path} is under /home, /root or /run/user, which the unit hides from the service (ProtectHome=yes): use a folder elsewhere")
    return [root] if config.cache_dir == root else [root, config.cache_dir]


def stop_timeout(config: ServerConfig) -> int:
    """TimeoutStopSec: the active run's pause, then the shutdown of everything else, then a margin."""
    return math.ceil(config.training_pause_deadline_s + config.shutdown_deadline_s) + STOP_MARGIN_S


def _value(path: Path, setting: str, command: bool = False) -> str:
    """`path` as a unit file value: % escaped (specifiers), and $ in a command line (variables)."""
    text = Path(path).as_posix()
    if not Path(path).is_absolute():
        raise UnitError(f"{setting} must be an absolute path, not {text}")
    if any(char.isspace() or char in "\"'\\" or ord(char) < 32 for char in text):
        raise UnitError(f"{setting} {text}: the unit cannot name a path with spaces, quotes or backslashes; use another folder")
    text = text.replace("%", "%%")
    return text.replace("$", "$$") if command else text


def render(config: ServerConfig, config_path: Path, env_file: Path | None = None, account: str = UNIT, prefix: Path = PREFIX) -> str:
    """The unit file for this configuration (`load`), installed release prefix and service account."""
    config_path = Path(config_path).expanduser().resolve()
    env_file = Path(env_file).expanduser().resolve() if env_file else config_path.parent / ENV_FILE
    if not ACCOUNT_PATTERN.match(account):
        raise UnitError(f"invalid account name {account!r}")
    folders = " ".join(_value(path, "[storage] cache_dir" if index else "[server] projects_root") for index, path in enumerate(service_folders(config)))
    prefix_value = _value(prefix, "--prefix")
    return TEMPLATE.format(
        config=_value(config_path, "the configuration file", command=True),
        config_comment=config_path.as_posix(),
        env_comment=env_file.as_posix(),
        env_file=_value(env_file, "--env-file"),
        prefix=prefix_value,
        executable=_value(Path(prefix) / "current" / ".venv" / "bin" / "yolo-workbench-server", "--prefix", command=True),
        account=account,
        unit=UNIT,
        folders=folders,
        cache_dir=_value(config.cache_dir, "[storage] cache_dir"),
        timeout=stop_timeout(config),
        pause=f"{config.training_pause_deadline_s:g}",
        shutdown=f"{config.shutdown_deadline_s:g}",
        margin=STOP_MARGIN_S,
    )


TEMPLATE = """\
# systemd unit of the YOLO Workbench server, written by `yolo-workbench-server unit` for {config_comment}
# and {env_comment}. Do not edit it: install.sh, rollback.sh and apply-config.sh write it again.
# After changing projects_root, cache_dir or a stop deadline in the configuration:
#   sudo bash {prefix}/current/deploy/apply-config.sh && sudo systemctl restart {unit}
# Settings the configuration does not cover go into a drop-in (`sudo systemctl edit {unit}`), which stays.

[Unit]
Description=YOLO Workbench (dataset annotation and training server)
Documentation=file://{prefix}/current/docs/server.md
After=network-online.target
Wants=network-online.target
# the disks of the projects and cache folders must be mounted first
RequiresMountsFor={folders}
# a crash loop stops after 10 starts within 5 minutes; manual restarts count too (systemctl reset-failed clears it)
StartLimitIntervalSec=300
StartLimitBurst=10

[Service]
Type=exec
User={account}
Group={account}
# what a library downloads into the working directory lands in the cache
WorkingDirectory={cache_dir}
EnvironmentFile=-{env_file}
ExecStart={executable} run --config {config}
# always, not on-failure: the server exits cleanly on SIGTERM (it pauses training first), also when an OOM
# killer (earlyoom, systemd-oomd) or a stray `kill` sends it; it must come back by itself. `systemctl stop`
# and `restart` are not affected (systemd does not restart a unit it stopped).
Restart=always
RestartSec=10

# Stopping: SIGTERM goes to the server only (KillMode=mixed). It asks the active training run to pause
# at its next checkpoint, waits up to [training] pause_deadline_s, closes everything else within
# [server] shutdown_deadline_s, and exits. Whatever is left (a run that did not reach its checkpoint,
# dataloader workers) gets SIGKILL as soon as the server has exited; the run then resumes from its previous
# checkpoint at the next start. TimeoutStopSec bounds the server itself: pause_deadline_s +
# shutdown_deadline_s + {margin} s ({pause} + {shutdown} + {margin}), or systemd would kill everything before the run could pause.
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec={timeout}

# Logs go to the journal: journalctl -u {unit}
StandardOutput=journal
StandardError=journal

# Hardening. The account may write only the configured projects folder and cache; the code is read-only,
# and so is everything else, whatever its owner and mode. Check GPU training after changing any of these
# (PrivateDevices would hide /dev/nvidia*).
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ReadWritePaths={folders}
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictSUIDSGID=yes
LockPersonality=yes
# PyTorch dataloader workers share memory through /dev/shm; PrivateTmp leaves it alone.
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
"""
