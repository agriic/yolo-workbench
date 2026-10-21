"""`yolo-workbench-server`: the workbench as an unattended service in projects mode.

    yolo-workbench-server run --config /etc/yolo-workbench/config.toml
    yolo-workbench-server check --config /etc/yolo-workbench/config.toml
    yolo-workbench-server user add alice --admin --config /etc/yolo-workbench/config.toml
    yolo-workbench-server unit --config /etc/yolo-workbench/config.toml   # the systemd unit (deploy/apply-config.sh)

`run` never prompts and never opens a browser. It validates the configuration, points every library
cache at the configured cache folder (before torch/Ultralytics/FiftyOne are imported), takes the owner
lock of the projects folder, and serves until SIGTERM: then the lifespan pauses the active training run
at its next checkpoint (resumed at the next start) and shuts the rest down in order (lifecycle.py).
With `[auth] enabled` (the default) everyone signs in with an account made by `user add` (auth.py).
The desktop command (`yolo-workbench`, cli.py) is unchanged and has no sign-in.
"""

from __future__ import annotations

import getpass
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from . import systemd_unit
from .config import ConfigError, ServerConfig, load_config
from .ownership import AlreadyOwned, OwnerLock
from .release import describe as describe_release
from .systemd_unit import HOME_FOLDERS, PREFIX, UNIT, UnitError, under

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Run the YOLO Workbench as a server (projects mode).")
user_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Manage the accounts people sign in with (stored in <projects_root>/.workbench/users.json).")
app.add_typer(user_app, name="user")
log = logging.getLogger("yolo_workbench.server")

OWNER_DIR = ".workbench"  # workspace-level server state inside the projects folder
APPLY_CONFIG = f"{PREFIX.as_posix()}/current/deploy/apply-config.sh"  # writes the unit for config.toml


def _load(config: Optional[Path], projects: Optional[Path] = None, host: Optional[str] = None, port: Optional[int] = None) -> ServerConfig:
    try:
        return load_config(config, {"projects_root": projects, "host": host, "port": port})
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(2) from exc


def prepare_environment(config: ServerConfig) -> None:
    """Library caches and settings under the configured cache folder (so the service account needs no home
    folder), no runtime pip installs, and FiftyOne's MongoDB. Must run before torch/Ultralytics/FiftyOne load.
    The training and export subprocesses inherit all of it."""
    if config.cache_dir is not None:
        cache = config.cache_dir
        os.environ["XDG_CACHE_HOME"] = str(cache)  # workbench media + weights (media_cache.cache_root)
        for name, sub in (
            ("YOLO_CONFIG_DIR", "ultralytics"),
            ("TORCH_HOME", "torch"),
            ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
            ("TRITON_HOME", "triton"),  # compiled GPU kernels (RF-DETR training): $TRITON_HOME/.triton
            ("MPLCONFIGDIR", "matplotlib"),
            ("HF_HOME", "huggingface"),
            ("FIFTYONE_MODEL_ZOO_DIR", "fiftyone/models"),
            ("FIFTYONE_DATASET_ZOO_DIR", "fiftyone/datasets"),
        ):
            os.environ.setdefault(name, str(cache / sub))
            _make_folder(Path(os.environ[name]))  # Ultralytics falls back to /tmp for a missing one
        # whatever else writes into ~: the account's home may not exist, and the unit's sandbox mounts it
        # read-only unless it is the projects folder or the cache
        if not _home_writable():
            home = cache / "home"
            _make_folder(home)
            os.environ["HOME"] = str(home)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # Ultralytics would pip-install missing packages into the venv
    if config.mongodb_uri:
        os.environ["FIFTYONE_DATABASE_URI"] = config.mongodb_uri


def _make_folder(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # `check` reports an unwritable cache folder


def _home_writable() -> bool:
    try:
        return os.access(Path.home(), os.W_OK)
    except (RuntimeError, KeyError):  # no HOME and no home folder for the uid
        return False


def _projects_root(config: ServerConfig) -> Path:
    root = config.projects_root
    if not root.is_dir():
        if not config.create_projects_root:
            # INVOCATION_ID: started by systemd, whose unit hides the home folders (ProtectHome=yes)
            hidden = " The service cannot see folders under /home, /root or /run/user (ProtectHome=yes in the unit)." if os.environ.get("INVOCATION_ID") and under(root, HOME_FOLDERS) else ""
            typer.echo(f"Error: the projects folder {root} does not exist (is its disk mounted?). Create it, or set [server] create_projects_root = true.{hidden}", err=True)
            raise typer.Exit(2)
        root.mkdir(parents=True, exist_ok=True)
    if not os.access(root, os.W_OK):
        typer.echo(f"Error: the projects folder {_not_writable(root)}", err=True)
        raise typer.Exit(2)
    return root


def _sandbox_fix(path: Path) -> str:
    return f"the unit is written from config.toml, so with {path.resolve()} set there, write it again: sudo bash {APPLY_CONFIG}, then restart (docs/server.md: Folders)"


def _not_writable(path: Path) -> str:
    """Why this process cannot write the folder `path` (os.access said it cannot)."""
    if hasattr(os, "statvfs") and os.statvfs(path).f_flag & os.ST_RDONLY:
        # the unit's sandbox (ProtectSystem=strict) mounts everything but its ReadWritePaths read-only: no mode helps
        return f"{path} is read-only for this process: a read-only mount, or in the systemd service a folder outside the unit's ReadWritePaths; for that: {_sandbox_fix(path)}"
    stat = path.stat()
    try:
        user, owner = getpass.getuser(), path.owner()
    except Exception:  # no name for the uid
        user, owner = "this user", str(stat.st_uid)
    return f"{path} is not writable by {user} (owner {owner}, mode {stat.st_mode & 0o7777:04o})"


def unit_sandbox(unit: str = UNIT) -> dict[str, str] | None:
    """The installed systemd unit's filesystem sandbox (`systemctl show`), or None without systemd or the unit."""
    try:
        result = subprocess.run(["systemctl", "show", unit, "--property=LoadState,FragmentPath,ProtectSystem,ProtectHome,ReadWritePaths"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    values = dict(line.partition("=")[::2] for line in result.stdout.splitlines())
    return values if result.returncode == 0 and values.get("LoadState") == "loaded" else None


def sandbox_problem(path: Path, sandbox: dict[str, str]) -> str | None:
    """Why the service cannot write `path` under the unit's sandbox, with the fix (None: the sandbox allows it).
    `check` runs outside the sandbox (sudo -u), so its own write test cannot tell."""
    home = sandbox.get("ProtectHome", "no")
    if home in ("yes", "tmpfs") and under(path, HOME_FOLDERS):
        return f"the unit hides /home, /root and /run/user from the service (ProtectHome={home}): move the folder elsewhere"
    writable = [folder.lstrip("-+") for folder in sandbox.get("ReadWritePaths", "").split()]  # "-": may be missing
    if under(path, writable):
        return None
    system = sandbox.get("ProtectSystem", "no")
    read_only = {"yes": ["/usr", "/boot", "/efi"], "full": ["/usr", "/boot", "/efi", "/etc"], "strict": ["/"]}.get(system, [])
    if under(path, read_only + (list(HOME_FOLDERS) if home == "read-only" else [])):
        return f"the unit's sandbox (ProtectSystem={system}, ProtectHome={home}) lets the service write only {', '.join(writable) or 'nothing'}; {_sandbox_fix(path)}"
    return None


def unit_drift(config: Path, sandbox: dict[str, str]) -> str | None:
    """How the installed unit differs from the one apply-config.sh would write for `config` (None: it does not)."""
    fragment = sandbox.get("FragmentPath")
    if not fragment:
        return None
    try:
        installed = Path(fragment).read_text(encoding="utf-8")
        wanted = systemd_unit.render(systemd_unit.load(config), config)
    except (OSError, ConfigError, UnitError) as exc:
        return str(exc)
    if installed == wanted:
        return None
    changed = [line for line in wanted.splitlines() if line and not line.startswith("#") and line not in installed.splitlines()]
    return f"{fragment} is not the unit for this configuration ({'; '.join(changed[:3]) or 'edited by hand'} expected): sudo bash {APPLY_CONFIG}, then restart"


def lifecycle_options(config: ServerConfig):
    from .lifecycle import LifecycleOptions

    return LifecycleOptions(
        pause_training_on_stop=True,
        training_pause_deadline=config.training_pause_deadline_s,
        shutdown_deadline=config.shutdown_deadline_s,
        maintenance=True,
        tick=30.0,  # idle models unload within half a minute of their idle time
        interval=config.maintenance_interval_min * 60,
        idle_unload=config.idle_unload_minutes * 60,
        project_idle_close=config.project_idle_close_min * 60,
        session_rotate=config.session_rotate_hours * 3600,
        staging_max_age=config.staging_max_age_hours * 3600,
        project_trash_retention_days=config.project_trash_retention_days,
        max_loaded_models=config.max_loaded_models,
        # SAM and other projects' models stay loaded while a run trains, unless models must not share the GPU
        unload_for_training="all" if config.gpu_models_during_training == "refuse" else "project",
    )


def build_app(config: ServerConfig, root: Path):
    """The projects-mode app with the server profile (also used by tests)."""
    from . import runtime
    from .auth import Auth
    from .media_cache import MediaCache, default_cache_dir
    from .web import create_workspace_app
    from .workspace import Workspace

    runtime.configure(config)
    workspace = Workspace(root, max_open=config.max_open_projects)
    workspace.staging_max_age = config.staging_max_age_hours * 3600
    workspace.training.resume_after_service_stop = config.resume_after_service_stop
    workspace.training.hold_queue_after_crash = config.hold_queue_after_crash
    media = MediaCache(default_cache_dir(), max_bytes=config.media_cache_mb * 1024 * 1024)
    auth = Auth(root / OWNER_DIR, config.session_days) if config.auth_enabled else None
    if auth is not None and auth.users.count() == 0:
        log.warning("sign-in is on but there are no user accounts yet: add one with `yolo-workbench-server user add <name> --admin`")
    return create_workspace_app(workspace, media_cache=media, allowed_hosts=config.allowed_host_set(), require_header=True, lifecycle_options=lifecycle_options(config), auth=auth)


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, "--config", "-c", envvar="WORKBENCH_CONFIG", dir_okay=False, help="TOML configuration file"),
    projects: Optional[Path] = typer.Option(None, "--projects", file_okay=False, help="Projects folder (overrides [server] projects_root)"),
    host: Optional[str] = typer.Option(None, help="Address to bind (overrides [server] host)"),
    port: Optional[int] = typer.Option(None, min=1, max=65535, help="Port (overrides [server] port)"),
) -> None:
    """Serve the projects folder until SIGTERM (systemctl stop)."""
    settings = _load(config, projects, host, port)
    prepare_environment(settings)
    from .logs import configure_logging

    configure_logging(settings.log_level, settings.log_format)
    root = _projects_root(settings)
    lock = OwnerLock(root / OWNER_DIR)
    try:
        lock.acquire()
    except AlreadyOwned as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    try:
        import uvicorn

        app_instance = build_app(settings, root)
        app_instance.state.owner_lock = lock
        log.info("release %s", describe_release())
        log.info("serving %s on http://%s:%s (allowed hosts: %s; sign-in %s)", root, settings.host or "*", settings.port, ", ".join(sorted(settings.allowed_host_set() or ["any"])), "on" if settings.auth_enabled else "off")
        uvicorn.run(
            app_instance,
            host=settings.host,
            port=settings.port,
            log_config=None,  # logs.configure_logging already routed uvicorn's loggers
            access_log=settings.access_log,
            timeout_graceful_shutdown=int(min(settings.shutdown_deadline_s, 30)),
            server_header=False,
        )
    finally:
        lock.release()


@app.command()
def check(
    config: Optional[Path] = typer.Option(None, "--config", "-c", envvar="WORKBENCH_CONFIG", dir_okay=False, help="TOML configuration file"),
    projects: Optional[Path] = typer.Option(None, "--projects", file_okay=False, help="Projects folder (overrides [server] projects_root)"),
) -> None:
    """Check the configuration, folders, disk space, GPU, libraries and MongoDB without starting the server."""
    settings = _load(config, projects)
    prepare_environment(settings)
    failures = 0

    def report(ok: bool | None, label: str, detail: str = "") -> None:
        nonlocal failures
        mark = {True: "ok  ", False: "FAIL", None: "warn"}[ok]
        failures += ok is False
        typer.echo(f"[{mark}] {label}{f': {detail}' if detail else ''}")

    sandbox = unit_sandbox()

    def folder(label: str, path: Path) -> None:
        if not os.access(path, os.W_OK):
            report(False, label, _not_writable(path))
        elif sandbox is not None and (problem := sandbox_problem(path, sandbox)):
            report(False, label, f"{path} is writable by this user but not by the service: {problem}")
        else:
            report(True, label, f"{path} is writable")

    report(True, "release", describe_release())
    report(True, "configuration", str(config) if config else "defaults and environment")
    root = settings.projects_root
    if not root.is_dir():
        report(None if settings.create_projects_root else False, "projects folder", f"{root} does not exist{' (it will be created)' if settings.create_projects_root else ''}")
    else:
        folder("projects folder", root)
        try:
            usage = shutil.disk_usage(root)
            low = usage.free < settings.min_free_disk_gb * 2**30
            report(None if low else True, "free disk space", f"{usage.free / 2**30:.1f} GB of {usage.total / 2**30:.1f} GB (minimum {settings.min_free_disk_gb:g} GB)")
        except OSError as exc:
            report(False, "free disk space", str(exc))
        lock = OwnerLock(root / OWNER_DIR)
        try:
            lock.acquire()
            lock.release()
            report(True, "owner lock", "no other workbench uses this folder")
        except AlreadyOwned as exc:
            report(None, "owner lock", str(exc))
    cache = settings.cache_dir or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    try:
        cache.mkdir(parents=True, exist_ok=True)
        folder("cache folder", cache)
    except OSError as exc:
        report(False, "cache folder", f"{cache}: {exc}")
    if sandbox is not None and config is not None and sandbox.get("FragmentPath"):
        drift = unit_drift(config, sandbox)
        report(None if drift else True, "systemd unit", drift or f"{sandbox['FragmentPath']} is written for this configuration")
    if settings.auth_enabled:
        if root.is_dir():
            from .auth import Auth, AuthError

            try:
                users = Auth(root / OWNER_DIR, settings.session_days).users.list()
                admins = sum(1 for user in users if user["admin"])
                report(None if not users else True, "sign-in", f"{len(users)} account(s), {admins} administrator(s)" if users else "no accounts yet: nobody can sign in. Add one: yolo-workbench-server user add <name> --admin")
            except AuthError as exc:
                report(False, "sign-in", str(exc))
    else:
        report(None, "sign-in", "turned off: anyone who reaches the port can use every project")
    hosts = settings.allowed_host_set()
    report(True, "listening", f"http://{settings.host or '*'}:{settings.port}, Host header {'not checked' if hosts is None else 'one of ' + ', '.join(sorted(hosts))}")
    try:
        import torch

        if torch.cuda.is_available():
            names = [f"{torch.cuda.get_device_name(index)} ({torch.cuda.get_device_properties(index).total_memory / 2**30:.0f} GB)" for index in range(torch.cuda.device_count())]
            report(True, "GPU", f"{', '.join(names)}; torch {torch.__version__}, CUDA {torch.version.cuda}")
        else:
            report(None, "GPU", f"no CUDA device visible to torch {torch.__version__}; training and SAM run on the CPU")
    except Exception as exc:
        report(False, "torch", str(exc))
    for module, label in (("ultralytics", "Ultralytics"), ("rfdetr", "RF-DETR")):
        try:
            imported = __import__(module)
            report(True, label, getattr(imported, "__version__", None) or _distribution_version(module))
        except Exception as exc:
            report(False, label, str(exc))
    if settings.embeddings_enabled:
        uri = settings.mongodb_uri or os.environ.get("FIFTYONE_DATABASE_URI")
        if not uri:
            report(None, "MongoDB (embeddings)", "no [embeddings] mongodb_uri; FiftyOne uses ~/.fiftyone/config.json or starts its own database")
        else:
            try:
                import pymongo

                pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000).admin.command("ping")
                report(True, "MongoDB (embeddings)", uri)
            except Exception as exc:
                report(None, "MongoDB (embeddings)", f"{uri} is not reachable ({exc}); the Embeddings tab will report it")
    else:
        report(True, "embeddings", "turned off")
    raise typer.Exit(1 if failures else 0)


@app.command()
def unit(
    config: Path = typer.Option(..., "--config", "-c", envvar="WORKBENCH_CONFIG", dir_okay=False, help="TOML configuration file"),
    env_file: Optional[Path] = typer.Option(None, "--env-file", dir_okay=False, help="The unit's environment file, whose WORKBENCH_ settings count too (default: yolo-workbench.env beside the configuration)"),
    account: str = typer.Option(UNIT, "--account", help="Service account"),
    prefix: Path = typer.Option(PREFIX, "--prefix", file_okay=False, help="Installation folder; the unit runs <prefix>/current"),
    folders: bool = typer.Option(False, "--folders", help="Print the folders the unit lets the service write, one per line, instead"),
) -> None:
    """Print the systemd unit for this configuration (deploy/apply-config.sh installs it)."""
    try:
        settings = systemd_unit.load(config, env_file)
        if folders:
            for folder in systemd_unit.service_folders(settings):
                typer.echo(folder.as_posix())
        else:
            typer.echo(systemd_unit.render(settings, config, env_file, account, prefix), nl=False)
    except (ConfigError, UnitError) as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(2) from exc


def _distribution_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "installed"


# ---------- user accounts ----------

ConfigOption = typer.Option(None, "--config", "-c", envvar="WORKBENCH_CONFIG", dir_okay=False, help="TOML configuration file")
ProjectsOption = typer.Option(None, "--projects", file_okay=False, help="Projects folder (overrides [server] projects_root)")


def _users(config: Optional[Path], projects: Optional[Path]):
    from .auth import UserStore

    try:
        settings = load_config(config, {"projects_root": projects}, serving=False)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(2) from exc
    root = settings.projects_root
    if not root.is_dir():
        typer.echo(f"Error: the projects folder {root} does not exist", err=True)
        raise typer.Exit(2)
    return UserStore(root / OWNER_DIR)


def _password(from_stdin: bool) -> str:
    if from_stdin:
        return sys.stdin.readline().rstrip("\r\n")
    return typer.prompt("Password", hide_input=True, confirmation_prompt=True)


def _account_command(action):
    from .auth import AuthError

    try:
        return action()
    except AuthError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@user_app.command("add")
def user_add(
    name: str = typer.Argument(..., help="Sign-in name: lowercase letters, digits, '.', '_', '-'"),
    display_name: Optional[str] = typer.Option(None, "--display-name", help="Name shown in the app (default: the sign-in name)"),
    admin: bool = typer.Option(False, "--admin", help="Mark the account as an administrator"),
    password_stdin: bool = typer.Option(False, "--password-stdin", help="Read the password from the first line of stdin instead of prompting"),
    config: Optional[Path] = ConfigOption,
    projects: Optional[Path] = ProjectsOption,
) -> None:
    """Create an account."""
    from .auth import normalize_name

    store = _users(config, projects)
    name = _account_command(lambda: normalize_name(name))
    if store.get(name) is not None:
        typer.echo(f"Error: user {name} already exists", err=True)
        raise typer.Exit(1)
    user = _account_command(lambda: store.add(name, _password(password_stdin), display_name, admin))
    typer.echo(f"Added {user.name}{' (administrator)' if user.admin else ''}")


@user_app.command("passwd")
def user_passwd(
    name: str = typer.Argument(...),
    password_stdin: bool = typer.Option(False, "--password-stdin", help="Read the password from the first line of stdin instead of prompting"),
    config: Optional[Path] = ConfigOption,
    projects: Optional[Path] = ProjectsOption,
) -> None:
    """Set a new password; the account's sign-ins end."""
    store = _users(config, projects)
    if store.get(name.strip().lower()) is None:
        typer.echo(f"Error: there is no user {name}", err=True)
        raise typer.Exit(1)
    _account_command(lambda: store.set_password(name.strip().lower(), _password(password_stdin)))
    typer.echo(f"Changed the password of {name.strip().lower()}; its sign-ins have ended")


@user_app.command("remove")
def user_remove(name: str = typer.Argument(...), config: Optional[Path] = ConfigOption, projects: Optional[Path] = ProjectsOption) -> None:
    """Delete an account; its sign-ins end. What the person did stays attributed to the name."""
    store = _users(config, projects)
    _account_command(lambda: store.remove(name))
    typer.echo(f"Removed {name.strip().lower()}")


@user_app.command("admin")
def user_admin(name: str = typer.Argument(...), revoke: bool = typer.Option(False, "--revoke", help="Take the administrator mark away"), config: Optional[Path] = ConfigOption, projects: Optional[Path] = ProjectsOption) -> None:
    """Mark an account as an administrator (or --revoke it)."""
    store = _users(config, projects)
    user = _account_command(lambda: store.update(name, admin=not revoke))
    typer.echo(f"{user.name} is {'an administrator' if user.admin else 'no longer an administrator'}")


@user_app.command("list")
def user_list(config: Optional[Path] = ConfigOption, projects: Optional[Path] = ProjectsOption) -> None:
    """List the accounts."""
    users = _account_command(lambda: _users(config, projects).list())
    if not users:
        typer.echo("No accounts yet. Add one: yolo-workbench-server user add <name> --admin")
        return
    width = max(len(user["name"]) for user in users)
    for user in users:
        typer.echo(f"{user['name']:<{width}}  {user['display_name']:<24}  {'admin' if user['admin'] else '     '}  last sign-in {user['last_login'] or 'never'}")


def main() -> None:  # pragma: no cover - console script
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
