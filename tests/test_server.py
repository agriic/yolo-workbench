"""The server profile: configuration, the owner lock, lifespan and health, training across service stops,
queue holds, streamed uploads, and the long-running hygiene (undo budget, sessions, retention, idle unload)."""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import fields, replace as dc_replace
from pathlib import Path

import pytest
from PIL import Image
from httpx import ASGITransport, AsyncClient

from test_dataset import make_dataset as make_small_dataset
from test_training import ARGS, FakeLauncher, make_dataset, runner_finishes
from yolo_workbench import storage
from yolo_workbench.config import ConfigError, load_config
from yolo_workbench.dataset import DatasetError
from yolo_workbench.jobs import JobManager
from yolo_workbench.lifecycle import Lifecycle, LifecycleOptions
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.ownership import AlreadyOwned, OwnerLock
from yolo_workbench.predictor import PredictorManager
from yolo_workbench.runtime import RUNTIME
from yolo_workbench.sam import SamManager
from yolo_workbench.server import build_app
from yolo_workbench.training import TrainingManager
from yolo_workbench.web import create_app, create_workspace_app
from yolo_workbench.workspace import Workspace


@pytest.fixture(autouse=True)
def restore_runtime():
    saved = dc_replace(RUNTIME)
    yield
    for item in fields(RUNTIME):
        setattr(RUNTIME, item.name, getattr(saved, item.name))


def client_for(app, host="127.0.0.1", header=True):
    return AsyncClient(transport=ASGITransport(app=app), base_url=f"http://{host}", headers={"X-Workbench": "1"} if header else {})


def manager_for(dataset, tmp_path, launcher):
    manager = TrainingManager.for_dataset(dataset, jobs=JobManager(synchronous=True), launcher=launcher, weights_dir=tmp_path / "weights")
    manager.attach(dataset)
    return manager


def queue_file(dataset) -> dict:
    return json.loads((dataset.root / ".yolo-workbench" / "training" / "queue.json").read_text())


# ---------- configuration ----------


def test_config_precedence_paths_and_validation(tmp_path):
    config_dir = tmp_path / "etc"
    config_dir.mkdir()
    path = config_dir / "config.toml"
    path.write_text('[server]\nprojects_root = "projects"\nport = 9000\nallowed_hosts = ["gpu-box", "10.0.0.5:8765"]\n[training]\npause_deadline_s = 120\n[logging]\nlevel = "debug"\n')
    config = load_config(path, environ={})
    assert config.projects_root == (config_dir / "projects").resolve() and config.port == 9000
    assert config.training_pause_deadline_s == 120 and config.log_level == "debug"
    assert config.allowed_host_set() == {"gpu-box", "10.0.0.5", "localhost", "127.0.0.1", "[::1]"}
    # environment beats the file, the CLI beats both
    config = load_config(path, {"port": 9100}, environ={"WORKBENCH_PORT": "9050", "WORKBENCH_RESUME_AFTER_SERVICE_STOP": "no", "WORKBENCH_ALLOWED_HOSTS": "a, b"})
    assert config.port == 9100 and config.resume_after_service_stop is False and config.allowed_hosts == ["a", "b"]
    assert load_config(path, environ={"WORKBENCH_PORT": "9050"}).port == 9050

    path.write_text('[server]\nprojects_root = "p"\nallowed_hosts = ["x"]\nprot = 1\n')
    with pytest.raises(ConfigError, match="prot"):
        load_config(path, environ={})
    with pytest.raises(ConfigError, match="allowed_hosts"):
        load_config(None, {"projects_root": tmp_path}, environ={})
    with pytest.raises(ConfigError, match="projects folder"):
        load_config(None, {"host": "127.0.0.1"}, environ={})
    with pytest.raises(ConfigError, match="integer"):
        load_config(None, {"projects_root": tmp_path, "host": "127.0.0.1"}, environ={"WORKBENCH_PORT": "eighty"})
    assert load_config(None, {"projects_root": tmp_path, "allowed_hosts": ["*"]}, environ={}).allowed_host_set() is None


# ---------- owner lock ----------


def test_owner_lock_refuses_a_second_owner_and_frees_on_release(tmp_path):
    first = OwnerLock(tmp_path / ".workbench").acquire()
    with pytest.raises(AlreadyOwned, match=f"process {os.getpid()}"):
        OwnerLock(tmp_path / ".workbench").acquire()
    script = "import sys; from pathlib import Path; from yolo_workbench.ownership import OwnerLock, AlreadyOwned\ntry:\n OwnerLock(Path(sys.argv[1])).acquire()\nexcept AlreadyOwned:\n sys.exit(3)\n"
    other = subprocess.run([sys.executable, "-c", script, str(tmp_path / ".workbench")], env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")})
    assert other.returncode == 3
    first.release()
    second = OwnerLock(tmp_path / ".workbench").acquire()
    assert second.held
    second.release()


# ---------- the server app: lifespan, health, guards ----------


def test_server_app_lifespan_health_and_guards(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    root = tmp_path / "projects"
    root.mkdir()
    config = load_config(None, {"projects_root": root, "allowed_hosts": ["gpu-box"], "idle_unload_minutes": 0, "auth_enabled": False}, environ={})
    app = build_app(config, root)
    lifecycle = app.state.lifecycle
    assert lifecycle.options.pause_training_on_stop and RUNTIME.server_mode and not RUNTIME.allow_path_import

    async def run():
        async with client_for(app) as client:
            assert (await client.get("/healthz")).status_code == 200
            assert (await client.get("/readyz")).json()["problems"] == ["starting"]
        async with app.router.lifespan_context(app):
            assert lifecycle.training._thread is not None  # the supervisor started with the app
            async with client_for(app, "gpu-box:8765") as client:
                assert (await client.get("/readyz")).status_code == 200
                info = (await client.get("/api/v1/app")).json()
                assert info["server"] and info["features"]["path_import"] is False
                created = await client.post("/api/v1/projects", json={"name": "Cars", "category": "detection", "classes": ["car"]})
                assert created.status_code == 200
                assert (await client.get("/api/v1/status")).json()["ready"]
            async with client_for(app, "evil.example") as client:
                assert (await client.get("/api/v1/projects")).status_code == 403
                assert (await client.get("/healthz")).status_code == 200  # health checks skip the Host check
            async with client_for(app, header=False) as client:
                assert (await client.post("/api/v1/projects", json={"name": "X", "category": "detection"})).status_code == 403
            async with client_for(app) as client:
                assert (await client.get("/api/v1/projects/cars/dataset")).status_code == 200
                response = await client.post("/api/v1/projects/cars/import/scan", json={"path": str(tmp_path)})
                assert response.status_code == 400 and "turned off" in response.json()["detail"]
        assert lifecycle.training._thread is None and lifecycle.workspace.sessions == {}
        async with client_for(app) as client:
            assert (await client.post("/api/v1/projects", json={"name": "Y", "category": "detection"})).status_code == 503
            assert (await client.get("/readyz")).status_code == 503

    asyncio.run(run())


# ---------- training across service stops ----------


def test_service_stop_pauses_the_run_and_the_next_start_resumes_it(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = manager_for(dataset, tmp_path, launcher)
    run = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    queued = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    run_dir = Path(run["path"])
    assert manager.begin_shutdown(pause=True) == run["name"]
    control = json.loads((run_dir / "workbench" / "control.json").read_text())
    assert control["action"] == "pause" and control["by"] == "service"
    assert queue_file(dataset)["active"]["auto_resume"]
    # the runner pauses at its checkpoint and exits
    runner_finishes(run_dir, status="paused", epochs=2)
    launcher.handle.code = 0
    assert manager.wait_for_pause(5, poll=0.01) == "paused"
    assert queue_file(dataset)["auto_resume"] == [f"/{run_dir.name}"]
    assert len(launcher.launches) == 1  # draining: the queued run did not start meanwhile
    manager.shutdown()

    relaunch = FakeLauncher()
    restarted = manager_for(dataset, tmp_path, relaunch)
    assert relaunch.launches == []  # constructing the manager launches nothing
    restarted.start(interval=60)
    try:
        assert len(relaunch.launches) == 1 and relaunch.launches[0][0] == run_dir
        spec = json.loads((run_dir / "workbench" / "spec.json").read_text())
        assert spec["resume"] is True
        queue = restarted.queue()
        assert queue["active"]["id"] == run["id"] and queue["queued"][0]["id"] == queued["id"] and queue["auto_resume"] == []
    finally:
        restarted.shutdown()


def test_a_run_killed_past_the_pause_deadline_resumes_from_its_checkpoint(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = manager_for(dataset, tmp_path, launcher)
    run = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    run_dir = Path(run["path"])
    runner_finishes(run_dir, status="training", epochs=2)  # mid-epoch: checkpoints of two epochs exist
    manager.begin_shutdown(pause=True)
    assert manager.wait_for_pause(0.05, poll=0.01) == "timeout"
    launcher.handle.code = -9  # systemd's SIGKILL at TimeoutStopSec
    relaunch = FakeLauncher()
    restarted = manager_for(dataset, tmp_path, relaunch)
    restarted.tick()
    assert len(relaunch.launches) == 1 and json.loads((run_dir / "workbench" / "spec.json").read_text())["resume"]


def test_without_resume_after_service_stop_the_run_stays_paused(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = manager_for(dataset, tmp_path, launcher)
    manager.resume_after_service_stop = False
    run = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    manager.begin_shutdown(pause=True)
    runner_finishes(Path(run["path"]), status="paused", epochs=2)
    launcher.handle.code = 0
    assert manager.wait_for_pause(5, poll=0.01) == "paused"
    relaunch = FakeLauncher()
    restarted = manager_for(dataset, tmp_path, relaunch)
    restarted.tick()
    assert relaunch.launches == [] and restarted.list_runs(None)["items"][0]["status"] == "paused"


def test_held_queue_launches_nothing_until_released_and_survives_restart(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = manager_for(dataset, tmp_path, launcher)
    first = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    second = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    # maintenance: hold and pause the active run
    queue = manager.set_hold(True, pause_active=True)
    assert queue["held"] and queue["pausing"] and queue["held_reason"]
    runner_finishes(Path(first["path"]), status="paused", epochs=1)
    launcher.handle.code = 0
    manager.tick()
    assert manager.queue()["active"] is None and len(launcher.launches) == 1
    relaunch = FakeLauncher()
    restarted = manager_for(dataset, tmp_path, relaunch)
    restarted.tick()
    assert restarted.queue()["held"] and relaunch.launches == []
    queue = restarted.set_hold(False)
    # the paused run goes first, ahead of the one that was waiting
    assert not queue["held"] and queue["active"]["id"] == first["id"] and queue["queued"][0]["id"] == second["id"]


def test_hold_queue_after_crash(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = manager_for(dataset, tmp_path, launcher)
    manager.hold_queue_after_crash = True
    manager.enqueue(None, dataset, "", "", ARGS)
    manager.enqueue(None, dataset, "", "", ARGS)
    launcher.handle.code = 1  # died without reporting
    queue = manager.queue()
    assert queue["held"] and "interrupted" in queue["held_reason"] and queue["active"] is None and len(launcher.launches) == 1


def test_a_launch_the_server_did_not_record_is_found_or_reported(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = manager_for(dataset, tmp_path, launcher)
    run = manager.enqueue(None, dataset, "", "", ARGS)["run"]
    workdir = Path(run["path"]) / "workbench"
    (workdir / "process.json").unlink()  # the server stopped right after Popen
    found = {"pid": os.getpid(), "create_time": None, "started": time.time()}
    monkeypatch.setattr("yolo_workbench.training._find_runner", lambda spec: found)
    monkeypatch.setattr("yolo_workbench.training._process_alive", lambda info: info is not None)
    restarted = manager_for(dataset, tmp_path, FakeLauncher())
    assert restarted.queue()["active"]["id"] == run["id"] and json.loads((workdir / "process.json").read_text())["pid"] == os.getpid()

    (workdir / "process.json").unlink()
    monkeypatch.setattr("yolo_workbench.training._find_runner", lambda spec: None)
    relaunch = FakeLauncher()
    again = manager_for(dataset, tmp_path, relaunch)
    item = again.list_runs(None)["items"][0]
    assert item["status"] == "failed" and "did not start" in item["error"] and relaunch.launches == []


# ---------- jobs and lifecycle ----------


def test_job_manager_close_cancels_and_joins():
    jobs = JobManager()
    started = threading.Event()

    def work(job):
        started.set()
        while True:
            job.progress()
            time.sleep(0.01)

    job = jobs.start("import", None, "endless", work)
    started.wait(2)
    assert jobs.close(2) == []
    assert jobs.get(job["id"])["state"] == "cancelled"
    with pytest.raises(DatasetError, match="shutting down"):
        jobs.start("import", None, "late", work)


def test_lifecycle_shutdown_pauses_training_and_closes_sessions(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = manager_for(dataset, tmp_path, launcher)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media"), training=manager, sam=SamManager(backend_factory=lambda name: object(), gpu=False), lifecycle_options=LifecycleOptions(pause_training_on_stop=True, training_pause_deadline=5, shutdown_deadline=2))
    lifecycle = app.state.lifecycle
    run = manager.enqueue(None, dataset, "", "", ARGS)["run"]

    def runner_pauses():
        workdir = Path(run["path"]) / "workbench"
        for _ in range(500):
            if (workdir / "control.json").exists():
                runner_finishes(Path(run["path"]), status="paused", epochs=1)
                launcher.handle.code = 0
                return
            time.sleep(0.01)

    async def go():
        async with app.router.lifespan_context(app):
            threading.Thread(target=runner_pauses, daemon=True).start()
    asyncio.run(go())
    assert lifecycle.stopping and dataset._closing
    assert queue_file(dataset)["auto_resume"] == [f"/{Path(run['path']).name}"]


def test_max_loaded_models_unloads_the_least_recently_used(tmp_path):
    workspace = Workspace(tmp_path / "projects", jobs=JobManager(synchronous=True), background_probe=False, session_factory=None)
    for name in ("A", "B"):
        workspace.create(name, "detection", ["one"])
    sessions = []
    for slug in ("a", "b"):
        session = workspace.session(slug)
        session.predictor = PredictorManager(session.dataset, backend_factory=lambda *args: type("Backend", (), {"names": {0: "one"}})())
        model = tmp_path / f"{slug}.pt"
        model.write_bytes(b"x")
        session.predictor.load(str(model))
        sessions.append(session)
    lifecycle = Lifecycle(workspace.training, workspace.jobs, SamManager(backend_factory=lambda name: object(), gpu=False), workspace.open_sessions, workspace.root, LifecycleOptions(max_loaded_models=1), workspace=workspace)
    assert lifecycle.enforce_model_limit(sessions[1]) == ["a"]
    assert sessions[0].predictor.backend is None and "project 'B'" in sessions[0].predictor.unloaded
    assert sessions[1].predictor.backend is not None


# ---------- uploads ----------


def make_project_app(tmp_path):
    workspace = Workspace(tmp_path / "projects", jobs=JobManager(synchronous=True), background_probe=False)
    workspace.create("Pets", "detection", ["cat"])
    return workspace, create_workspace_app(workspace, media_cache=MediaCache(tmp_path / "media"))


def test_uploads_stream_to_disk_with_limits(tmp_path):
    workspace, app = make_project_app(tmp_path)
    RUNTIME.max_upload_file_bytes = 1000
    RUNTIME.max_upload_files = 2

    async def chunks(total, size=100):
        for _ in range(total // size):
            yield b"x" * size

    async def run():
        async with client_for(app) as client:
            staging = (await client.post("/api/v1/projects/pets/import/staging")).json()["staging_id"]
            base = f"/api/v1/projects/pets/import/staging/{staging}/files"
            response = await client.put(f"{base}?path=a/one.jpg", content=b"y" * 500)
            assert response.status_code == 200 and response.json() == {"path": "a/one.jpg", "size": 500}
            # chunked (no Content-Length): counted while it streams
            response = await client.put(f"{base}?path=two.jpg", content=chunks(1500))
            assert response.status_code == 413
            response = await client.put(f"{base}?path=two.jpg", content=chunks(300))
            assert response.status_code == 200 and response.json()["size"] == 300
            response = await client.put(f"{base}?path=three.jpg", content=b"z")
            assert response.status_code == 413 and "files" in response.json()["detail"]
            response = await client.put(f"{base}?path=evil.exe", content=b"z")
            assert response.status_code == 400
            directory = workspace.root / "pets" / ".yolo-workbench" / "staging" / staging
            assert sorted(path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()) == ["a/one.jpg", "two.jpg"]
            RUNTIME.min_free_bytes = 1 << 62
            response = await client.put(f"{base}?path=a/one.jpg", content=b"y")
            assert response.status_code == 507 and "disk space" in response.json()["detail"]

    asyncio.run(run())


def test_model_upload_goes_into_the_projects_models_folder(tmp_path):
    workspace, app = make_project_app(tmp_path)

    async def run():
        async with client_for(app) as client:
            response = await client.put("/api/v1/projects/pets/models/upload?name=best.pt", content=b"weights")
            assert response.status_code == 200 and response.json()["name"] == "best.pt"
            assert (workspace.root / "pets" / "models" / "best.pt").read_bytes() == b"weights"
            response = await client.put("/api/v1/projects/pets/models/upload?name=..%2Fbest.pt", content=b"more")
            assert response.json()["name"] == "best-2.pt"
            assert (await client.put("/api/v1/projects/pets/models/upload?name=notes.txt", content=b"x")).status_code == 400
            models = (await client.get("/api/v1/projects/pets/predictor/models")).json()["items"]
            assert {item["name"] for item in models} >= {"best.pt", "best-2.pt"}

    asyncio.run(run())


def test_models_folder_lists_and_deletes_models(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = manager_for(dataset, tmp_path, launcher)
    predictor = PredictorManager(dataset, backend_factory=lambda *args: type("Backend", (), {"names": {0: "cat"}})())
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media"), training=manager, predictor=predictor, sam=SamManager(backend_factory=lambda name: object(), gpu=False))
    folder = dataset.yaml_path.parent / "models"

    async def run():
        async with client_for(app) as client:
            assert (await client.get("/api/v1/models")).json()["items"] == []
            for name in ("old.pt", "new.onnx", "base.pt"):
                assert (await client.put(f"/api/v1/models/upload?name={name}", content=b"weights")).status_code == 200
            os.utime(folder / "old.pt", (1, 1))
            (folder / ".partial.pt").write_bytes(b"x")  # an upload still being received
            (folder / "notes.txt").write_text("x")
            listing = (await client.get("/api/v1/models")).json()
            assert [item["name"] for item in listing["items"]][-1] == "old.pt" and len(listing["items"]) == 3
            assert listing["total_bytes"] == 21 and not any(item["loaded"] for item in listing["items"])

            loaded = (await client.post("/api/v1/predictor/load", json={"path": (folder / "old.pt").as_posix()})).json()
            assert loaded["status"] == "ready"
            assert [item["name"] for item in (await client.get("/api/v1/models")).json()["items"] if item["loaded"]] == ["old.pt"]
            listing = (await client.delete("/api/v1/models/old.pt")).json()
            assert {item["name"] for item in listing["items"]} == {"base.pt", "new.onnx"}
            assert not (folder / "old.pt").exists()
            state = (await client.get("/api/v1/predictor")).json()
            assert state["status"] == "idle" and "old.pt was deleted" in state["unloaded"]
            assert all(item["name"] != "old.pt" for item in (await client.get("/api/v1/predictor/models")).json()["items"])

            for name in ("old.pt", "..%5Cbase.pt", "notes.txt", ".partial.pt"):
                assert (await client.delete(f"/api/v1/models/{name}")).status_code == 404, name
            assert (folder / ".partial.pt").exists() and (folder / "notes.txt").exists()

            manager.enqueue(None, dataset, "", "", f"model: {(folder / 'base.pt').as_posix()}\nepochs: 5\nimgsz: 64\n")
            response = await client.delete("/api/v1/models/base.pt")
            assert response.status_code == 400 and "is queued to start from base.pt" in response.json()["detail"]
            assert (folder / "base.pt").exists()

    asyncio.run(run())


def test_gpu_policy_refuses_model_loads_while_training(tmp_path):
    dataset = make_dataset(tmp_path)
    launcher = FakeLauncher()
    manager = manager_for(dataset, tmp_path, launcher)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media"), training=manager, predictor=PredictorManager(dataset, backend_factory=lambda *args: type("Backend", (), {"names": {0: "cat"}})()), sam=SamManager(backend_factory=lambda name: object(), gpu=False))
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    manager.enqueue(None, dataset, "", "", ARGS)

    async def run():
        async with client_for(app) as client:
            response = await client.post("/api/v1/predictor/load", json={"path": str(model)})
            assert response.status_code == 200 and "is using the GPU" in response.json()["gpu_warning"]
            RUNTIME.gpu_models_during_training = "refuse"
            response = await client.post("/api/v1/predictor/load", json={"path": str(model)})
            assert response.status_code == 400 and "once it finishes" in response.json()["detail"]
            assert (await client.post("/api/v1/sam/start")).status_code == 400

    asyncio.run(run())


# ---------- long-running hygiene ----------


def test_undo_history_is_bounded(tmp_path):
    dataset, _ = make_small_dataset(tmp_path)
    image_id = next(iter(dataset.images))
    RUNTIME.undo_max_transactions = 3
    for index in range(5):
        dataset.replace_annotations(image_id, [{"class_id": 0, "points": [0.5, 0.5, 0.1 + index / 100, 0.1]}])
    assert len(dataset._undo) == 3 and dataset.history_trimmed == 2
    RUNTIME.undo_max_bytes = 1
    dataset.replace_annotations(image_id, [])
    assert len(dataset._undo) == 1  # the newest always stays
    dataset.history("undo")
    with pytest.raises(DatasetError):
        dataset.history("undo")


def test_session_rotation_keeps_trash_undo_needs_and_retention_purges_the_rest(tmp_path):
    dataset, _ = make_small_dataset(tmp_path)
    image_id = next(iter(dataset.images))
    dataset.session_id = first = "20000101-000000"  # an earlier session, so rotating gives a new id
    dataset.replace_annotations(image_id, [])  # backup in the first session
    dataset.delete_images([image_id])  # trash in the first session, still undoable
    assert dataset.rotate_session() != first
    with pytest.raises(DatasetError, match="Undo can still"):
        storage.purge(dataset, "trash", first)
    assert storage.summary(dataset)["trash"][0]["in_undo"]
    old = time.time() - 40 * 86400
    for path in (dataset.root / ".yolo-workbench").rglob("*"):
        os.utime(path, (old, old))
    RUNTIME.trash_retention_days = RUNTIME.backup_retention_days = 30
    purged = storage.apply_retention(dataset.root, dataset)
    assert purged == {"trash": [], "backups": [first]}  # the trash is still needed by undo
    dataset.history("undo")  # the image is back; nothing points into the trash any more
    dataset._redo.clear()
    for path in (dataset.root / ".yolo-workbench" / "trash").rglob("*"):
        os.utime(path, (old, old))  # the (now empty) folders changed when the files moved back
    assert storage.apply_retention(dataset.root, dataset)["trash"] == [first]
    assert storage.apply_retention(dataset.root, None) == {"trash": [], "backups": []}


def test_maintenance_cleans_staging_rotates_sessions_and_expires_project_trash(tmp_path):
    workspace = Workspace(tmp_path / "projects", jobs=JobManager(synchronous=True), background_probe=False)
    workspace.create("Live", "detection", ["a"])
    workspace.create("Gone", "detection", ["a"])
    session = workspace.session("live")
    staging = workspace.root / "live" / ".yolo-workbench" / "staging"
    (staging / "abandoned").mkdir(parents=True)
    (staging / "fresh").mkdir()
    old = time.time() - 2 * 86400
    os.utime(staging / "abandoned", (old, old))
    entry = workspace.trash("gone")["entry"]
    (workspace.trash_root / entry).rename(workspace.trash_root / "gone--20000101-000000")
    session.dataset.session_started -= 2 * 86400
    session.dataset.session_id = "20000101-000000"
    lifecycle = Lifecycle(workspace.training, workspace.jobs, SamManager(backend_factory=lambda name: object(), gpu=False), workspace.open_sessions, workspace.root, LifecycleOptions(maintenance=True, session_rotate=86400, project_trash_retention_days=30), workspace=workspace)
    lifecycle.maintenance_tick(heavy=True)
    assert not (staging / "abandoned").exists() and (staging / "fresh").exists()
    assert workspace.list_trash() == []
    assert session.dataset.session_id != "20000101-000000"  # a day old: a new backup/trash session started
    assert lifecycle.status()["disk"]["free"] > 0


def test_idle_models_unload(tmp_path):
    sam = SamManager(backend_factory=lambda name: object(), gpu=False)
    sam.start()
    sam._thread.join(5)
    assert not sam.unload_if_idle(60)
    sam.last_used -= 120
    assert sam.unload_if_idle(60)
    state = sam.payload()
    assert state["status"] == "idle" and state["idle_unloaded"] and state["unloaded"] is None  # the editor loads it again
    sam.start()
    sam._thread.join(5)
    assert sam.payload()["status"] == "ready" and not sam.payload()["idle_unloaded"]

    dataset, _ = make_small_dataset(tmp_path)
    predictor = PredictorManager(dataset, backend_factory=lambda *args: type("Backend", (), {"names": {0: "zero"}})())
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    predictor.load(str(model))
    predictor.last_used -= 120
    predictor.external_busy = "Label QA is analyzing images"
    assert not predictor.unload_if_idle(60)  # busy: the countdown restarts
    predictor.external_busy = None
    assert not predictor.unload_if_idle(60)
    predictor.last_used -= 120
    assert predictor.unload_if_idle(60) and predictor.backend is None
    state = predictor.payload()
    assert state["status"] == "parked" and "without use" in state["parked"] and state["unloaded"] is None
    assert state["model"]["names"] == {0: "zero"}  # the mapping stays editable while parked


def test_a_parked_model_loads_again_on_use_with_its_mapping_and_predictions(tmp_path):
    from test_predictor import make_predictor

    dataset, _, manager, model = make_predictor(tmp_path)
    manager.load(str(model))
    first = second = next(iter(dataset.images))
    pending = manager.predict_image(first)["items"]
    manager.set_mapping({"0": None})
    manager.last_used -= 600
    assert manager.unload_if_idle(60) and manager.status == "parked"
    loads = []
    factory = manager._backend_factory
    manager._backend_factory = lambda *args: loads.append(args) or factory(*args)
    assert manager.pending_image_ids() == {first}
    manager.set_mapping({"0": None})  # works while parked
    manager.predict_image(second)  # loads it again first
    assert len(loads) == 1 and manager.status == "ready" and manager.mapping[0] is None  # the mapping chosen by hand stayed
    # a run that needs the GPU makes a parked model one that is loaded again only by hand
    manager.last_used -= 600
    manager.unload_if_idle(60)
    assert manager.unload("big run") is False and manager.status == "idle" and "big run" in manager.payload()["unloaded"]
    with pytest.raises(DatasetError, match="big run"):
        manager.predict_image(second)


def test_server_training_unloads_only_the_trained_projects_model(tmp_path):
    from test_training import ARGS, FakeLauncher, FakePredictorBackend, FakeSamBackend

    launcher = FakeLauncher()
    workspace = Workspace(tmp_path / "projects", jobs=JobManager(synchronous=True), background_probe=False, training_options={"launcher": launcher, "weights_dir": tmp_path / "w"})
    slugs = [workspace.create(name, "detection", ["cat"])["slug"] for name in ("Alpha", "Beta")]
    for slug in slugs:
        for split in ("train", "val"):
            Image.new("RGB", (16, 16)).save(workspace.root / slug / "images" / split / "a.jpg")
            (workspace.root / slug / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    sam = SamManager(backend_factory=FakeSamBackend, gpu=True)
    app = create_workspace_app(workspace, media_cache=MediaCache(tmp_path / "media"), sam=sam, lifecycle_options=LifecycleOptions(unload_for_training="project"))
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    predictors = []
    for slug in slugs:
        predictor = workspace.session(slug).predictor
        predictor._backend_factory = FakePredictorBackend
        predictor._available = True
        predictor.load(str(model))
        predictors.append(predictor)
    sam.start()
    sam._thread.join(5)

    async def run():
        async with client_for(app) as client:
            response = await client.post(f"/api/v1/projects/{slugs[0]}/training/runs", json={"name": "p", "yaml": ARGS})
            assert response.json()["run"]["active"]
        assert predictors[0].status == "idle" and "training run" in predictors[0].unloaded
        assert predictors[1].status == "ready" and sam.status == "ready"

    asyncio.run(run())


# ---------- the real server process ----------


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_server_process_refuses_a_second_owner_and_stops_cleanly_on_a_signal(tmp_path):
    import signal
    import urllib.request

    root = tmp_path / "projects"
    root.mkdir()
    port = _free_port()
    config = tmp_path / "config.toml"
    config.write_text(f'[server]\nprojects_root = "{root.as_posix()}"\nhost = "127.0.0.1"\nport = {port}\nshutdown_deadline_s = 10\n[storage]\ncache_dir = "{(tmp_path / "cache").as_posix()}"\n[embeddings]\nenabled = false\n')
    command = [sys.executable, "-m", "yolo_workbench.server", "run", "--config", str(config)]
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    server = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=flags)
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            assert server.poll() is None, server.stdout.read().decode(errors="replace")
            assert time.monotonic() < deadline, "the server did not become ready"
            time.sleep(0.2)
        second = subprocess.run([*command[:-2], "--config", str(config), "--port", str(_free_port())], env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
        assert second.returncode == 1 and b"already in use" in second.stderr
        server.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM)
        # uvicorn re-raises the signal once it has shut down: death by SIGTERM (systemd counts it as a clean
        # exit), or the C runtime's exit code 3 for SIGBREAK on Windows
        code = server.wait(30)
        output = server.stdout.read().decode(errors="replace")
        assert code in (0, 3, -signal.SIGTERM), output
        assert "shutdown complete" in output, output
    finally:
        if server.poll() is None:
            server.kill()
    OwnerLock(root / ".workbench").acquire().release()  # released with the process



# ---------- the installed release ----------

RELEASE = {"name": "2026.10.21", "commit": "279d5d130af5ec1bbae64f9dcf005ad5da193d93", "installed": "2026-10-21T10:00:00Z"}


@pytest.fixture
def installed_release(tmp_path, monkeypatch):
    """The process runs from releases/<name>/.venv with install.sh's release.json beside it."""
    from yolo_workbench import release

    folder = tmp_path / "releases" / RELEASE["name"]
    (folder / ".venv").mkdir(parents=True)
    (folder / "release.json").write_text(json.dumps(RELEASE))
    monkeypatch.setattr(sys, "prefix", str(folder / ".venv"))
    release.release_info.cache_clear()
    yield folder
    release.release_info.cache_clear()


def test_the_release_is_reported_by_readyz_app_info_and_the_owner_lock(tmp_path, monkeypatch, installed_release):
    from yolo_workbench import release

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert release.describe() == "2026.10.21 (commit 279d5d130af5, installed 2026-10-21T10:00:00Z)"
    (installed_release / "release.json").write_text(json.dumps({**RELEASE, "name": "later"}))
    assert release.release_name() == "2026.10.21"  # read once: `current` may point elsewhere by now
    root = tmp_path / "projects"
    root.mkdir()
    config = load_config(None, {"projects_root": root, "allowed_hosts": ["gpu-box"], "idle_unload_minutes": 0, "auth_enabled": False}, environ={})
    app = build_app(config, root)

    async def run():
        async with client_for(app, "gpu-box") as client:
            assert (await client.get("/api/v1/app")).json()["release"] == RELEASE
        async with client_for(app, "10.0.0.20") as client:
            assert (await client.get("/readyz")).json() == {"status": "unavailable", "problems": ["starting"], "release": "2026.10.21"}

    asyncio.run(run())
    lock = OwnerLock(root / ".workbench").acquire()
    try:
        assert json.loads((root / ".workbench" / "owner.json").read_text())["release"] == "2026.10.21"
        with pytest.raises(AlreadyOwned, match=r"since [^(]*, release 2026\.10\.21 \("):
            OwnerLock(root / ".workbench").acquire()
    finally:
        lock.release()


def test_without_an_installed_release_only_the_version_is_known(tmp_path, monkeypatch):
    from yolo_workbench import __version__, release

    venv = tmp_path / "checkout" / ".venv"
    venv.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(venv))
    try:
        for content in (None, "not json", json.dumps({"name": ""}), json.dumps(["2026.10.21"])):
            if content is not None:
                (venv.parent / "release.json").write_text(content)
            release.release_info.cache_clear()
            assert release.release_info() is None and release.release_name() is None, content
            assert release.describe().startswith(f"version {__version__}, not an installed release")
        (venv.parent / "release.json").write_text(json.dumps({"name": "hand-made"}))
        release.release_info.cache_clear()
        assert release.release_info() == {"name": "hand-made", "commit": None, "installed": None} and release.describe() == "hand-made"
    finally:
        release.release_info.cache_clear()


def test_check_names_its_release_and_the_one_holding_the_folder(tmp_path, monkeypatch, installed_release):
    """What an admin runs after install.sh and before the restart: the new code, the old server's lock."""
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from yolo_workbench import server

    torch = SimpleNamespace(__version__="2.9.0", cuda=SimpleNamespace(is_available=lambda: False))
    for name, module in (("torch", torch), ("ultralytics", SimpleNamespace(__version__="8.3.0")), ("rfdetr", SimpleNamespace(__version__="1.11.0"))):
        monkeypatch.setitem(sys.modules, name, module)  # importing the real ones takes seconds
    root = tmp_path / "projects"
    root.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(f'[server]\nprojects_root = "{root.as_posix()}"\nallowed_hosts = ["gpu-box"]\n[storage]\ncache_dir = "{(tmp_path / "cache").as_posix()}"\n[auth]\nenabled = false\n[embeddings]\nenabled = false\n')
    running = tmp_path / "releases" / "2026.10.01" / ".venv"
    running.mkdir(parents=True)
    (running.parent / "release.json").write_text(json.dumps({**RELEASE, "name": "2026.10.01"}))
    from yolo_workbench import release

    monkeypatch.setattr(sys, "prefix", str(running))  # the old server took the lock
    release.release_info.cache_clear()
    lock = OwnerLock(root / ".workbench").acquire()
    monkeypatch.setattr(sys, "prefix", str(installed_release / ".venv"))
    release.release_info.cache_clear()
    monkeypatch.setattr(server, "unit_sandbox", lambda: None)  # not the unit of a server this may run on
    environment = os.environ.copy()  # check points the library caches at cache_dir
    try:
        result = CliRunner().invoke(server.app, ["check", "--config", str(config)])
    finally:
        os.environ.clear()
        os.environ.update(environment)
        lock.release()
    assert result.exit_code == 0, result.output
    assert "[ok  ] release: 2026.10.21 (commit 279d5d130af5, installed 2026-10-21T10:00:00Z)" in result.output
    assert "[warn] owner lock:" in result.output and ", release 2026.10.01 (" in result.output


SANDBOX = {"LoadState": "loaded", "ProtectSystem": "strict", "ProtectHome": "yes"}  # as systemd_unit.TEMPLATE


@pytest.mark.skipif(sys.platform == "win32", reason="systemd")
def test_a_folder_outside_the_units_writable_paths_is_read_only_in_the_service(tmp_path):
    """ProtectSystem=strict mounts everything but ReadWritePaths read-only, whatever a folder's mode."""
    from yolo_workbench import server

    allowed, elsewhere = tmp_path / "srv", tmp_path / "data"
    (allowed / "projects").mkdir(parents=True)
    elsewhere.mkdir(mode=0o777)
    (allowed / "link").symlink_to(elsewhere)
    sandbox = {**SANDBOX, "ReadWritePaths": f"{allowed} -/var/cache/yolo-workbench"}
    assert server.sandbox_problem(allowed / "projects", sandbox) is None
    for path in (elsewhere, allowed / "link"):  # a symlink out of a writable folder leads to a read-only one
        problem = server.sandbox_problem(path, sandbox)
        assert problem and f"write only {allowed}, /var/cache/yolo-workbench;" in problem and f"with {elsewhere.resolve()} set there, write it again: sudo bash /opt/yolo-workbench/current/deploy/apply-config.sh" in problem
    assert "hides /home" in server.sandbox_problem(Path("/home/alice/projects"), sandbox)
    assert server.sandbox_problem(Path("/home/alice/projects"), {**sandbox, "ProtectHome": "read-only"})
    full = {**sandbox, "ProtectSystem": "full"}
    assert server.sandbox_problem(elsewhere, full) is None and server.sandbox_problem(Path("/etc/yolo-workbench/projects"), full)


@pytest.mark.skipif(sys.platform == "win32", reason="systemd")
def test_check_and_the_service_say_why_a_folder_is_not_writable(tmp_path, monkeypatch, capsys):
    """`check` runs as the account but outside the sandbox (sudo -u): the folder is writable there, not in the
    service, which sees it on a read-only mount."""
    from types import SimpleNamespace

    import typer
    from typer.testing import CliRunner

    from yolo_workbench import server
    from yolo_workbench.config import ServerConfig

    torch = SimpleNamespace(__version__="2.9.0", cuda=SimpleNamespace(is_available=lambda: False))
    for name, module in (("torch", torch), ("ultralytics", SimpleNamespace(__version__="8.3.0")), ("rfdetr", SimpleNamespace(__version__="1.11.0"))):
        monkeypatch.setitem(sys.modules, name, module)
    root, cache = tmp_path / "data" / "projects", tmp_path / "srv" / "cache"
    root.mkdir(parents=True)
    root.chmod(0o777)
    config = tmp_path / "config.toml"
    config.write_text(f'[server]\nprojects_root = "{root.as_posix()}"\nallowed_hosts = ["gpu-box"]\n[storage]\ncache_dir = "{cache.as_posix()}"\n[auth]\nenabled = false\n[embeddings]\nenabled = false\n')
    monkeypatch.setattr(server, "unit_sandbox", lambda: {**SANDBOX, "ReadWritePaths": str(tmp_path / "srv")})
    environment = os.environ.copy()
    try:
        result = CliRunner().invoke(server.app, ["check", "--config", str(config)])
    finally:
        os.environ.clear()
        os.environ.update(environment)
    assert result.exit_code == 1, result.output
    assert f"[FAIL] projects folder: {root} is writable by this user but not by the service: the unit's sandbox (ProtectSystem=strict, ProtectHome=yes) lets the service write only {tmp_path / 'srv'}" in result.output
    assert f"with {root.resolve()} set there, write it again: sudo bash /opt/yolo-workbench/current/deploy/apply-config.sh, then restart" in result.output
    assert f"[ok  ] cache folder: {cache} is writable" in result.output

    # the service: os.access fails on the read-only mount
    def startup_error(read_only: bool) -> str:
        monkeypatch.setattr(os, "statvfs", lambda path: SimpleNamespace(f_flag=os.ST_RDONLY if read_only else 0))
        with pytest.raises(typer.Exit):
            server._projects_root(ServerConfig(projects_root=root))
        return capsys.readouterr().err

    monkeypatch.setattr(os, "access", lambda path, mode: False)
    assert f"Error: the projects folder {root} is read-only for this process: a read-only mount, or in the systemd service a folder outside the unit's ReadWritePaths; for that: the unit is written from config.toml, so with {root.resolve()} set there, write it again: sudo bash /opt/yolo-workbench/current/deploy/apply-config.sh" in startup_error(True)
    error = startup_error(False)  # a plain permission problem names the folder's owner and mode
    assert f"Error: the projects folder {root} is not writable by " in error and ", mode 0777)" in error

# ---------- the systemd unit ----------


def unit_settings(text: str) -> dict[str, str]:
    """A unit file's settings (the last of each name), as `systemctl show` would name them."""
    lines = (line for line in text.splitlines() if line and not line.startswith(("#", "[")))
    return dict(line.split("=", 1) for line in lines)


def test_the_unit_is_written_from_the_configuration(tmp_path):
    """Folders, mounts and stop timeout come from config.toml plus the unit's environment file; nothing names
    the example's /srv/yolo-workbench."""
    from typer.testing import CliRunner

    from yolo_workbench import server, systemd_unit

    root, cache = tmp_path / "data" / "projects", tmp_path / "fast" / "cache"
    config = tmp_path / "etc" / "config.toml"
    config.parent.mkdir()
    config.write_text(f'[server]\nprojects_root = "{root.as_posix()}"\nshutdown_deadline_s = 30\n[storage]\ncache_dir = "{(tmp_path / "other").as_posix()}"\n[training]\npause_deadline_s = 1200\n')
    (config.parent / "yolo-workbench.env").write_text(f'# what the service reads too\nWORKBENCH_CACHE_DIR="{cache.as_posix()}"\nHTTPS_PROXY=http://proxy:3128\n')
    text = systemd_unit.render(systemd_unit.load(config), config)
    unit = unit_settings(text)
    assert unit["ReadWritePaths"] == unit["RequiresMountsFor"] == f"{root} {cache}"
    assert unit["WorkingDirectory"] == str(cache) and unit["TimeoutStopSec"] == "1290"  # 1200 + 30 + 60
    assert unit["ExecStart"] == f"/opt/yolo-workbench/current/.venv/bin/yolo-workbench-server run --config {config}"
    assert unit["EnvironmentFile"] == f"-{config.parent / 'yolo-workbench.env'}" and unit["User"] == unit["Group"] == "yolo-workbench"
    assert "/srv/" not in text and "listen" not in text
    sandbox = {"LoadState": "loaded", **{key: unit[key] for key in ("ProtectSystem", "ProtectHome", "ReadWritePaths")}}
    assert sandbox == {**SANDBOX, "ReadWritePaths": f"{root} {cache}"}  # what the sandbox tests assume
    assert server.sandbox_problem(root, sandbox) is None and server.sandbox_problem(cache / "yolo-workbench" / "weights", sandbox) is None

    result = CliRunner().invoke(server.app, ["unit", "--config", str(config), "--account", "wb", "--prefix", "/srv/wb"])
    assert result.exit_code == 0, result.output
    assert unit_settings(result.output)["ExecStart"].startswith("/srv/wb/current/.venv/bin/yolo-workbench-server run") and "User=wb\n" in result.output
    folders = CliRunner().invoke(server.app, ["unit", "--config", str(config), "--folders"])
    assert folders.exit_code == 0 and folders.output == f"{root.as_posix()}\n{cache.as_posix()}\n"


def test_a_configuration_the_unit_cannot_serve_is_refused(tmp_path):
    from typer.testing import CliRunner

    from yolo_workbench import server, systemd_unit
    from yolo_workbench.config import ServerConfig

    config = tmp_path / "config.toml"
    config.write_text(f'[server]\nprojects_root = "{(tmp_path / "projects").as_posix()}"\n')
    result = CliRunner().invoke(server.app, ["unit", "--config", str(config)])
    assert result.exit_code == 2 and "set [storage] cache_dir" in result.output  # ~/.cache of whom?
    for root, message in ((Path("/home/alice/projects"), "hides from the service"), (tmp_path / "my projects", "spaces")):
        with pytest.raises(systemd_unit.UnitError, match=message):
            systemd_unit.render(ServerConfig(projects_root=root, cache_dir=tmp_path / "cache"), config)
    unit = unit_settings(systemd_unit.render(ServerConfig(projects_root=tmp_path / "100%", cache_dir=tmp_path / "cache"), tmp_path / "$etc" / "config.toml"))
    assert unit["ReadWritePaths"].startswith(f"{tmp_path}/100%% ")  # % starts a specifier, $ a variable in a command
    assert unit["ExecStart"].endswith(f"--config {tmp_path}/$$etc/config.toml")


@pytest.mark.skipif(sys.platform == "win32", reason="systemd")
def test_check_warns_when_the_installed_unit_is_not_the_one_for_the_configuration(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from yolo_workbench import server, systemd_unit

    torch = SimpleNamespace(__version__="2.9.0", cuda=SimpleNamespace(is_available=lambda: False))
    for name, module in (("torch", torch), ("ultralytics", SimpleNamespace(__version__="8.3.0")), ("rfdetr", SimpleNamespace(__version__="1.11.0"))):
        monkeypatch.setitem(sys.modules, name, module)
    root, cache = tmp_path / "data" / "projects", tmp_path / "cache"
    root.mkdir(parents=True)
    config = tmp_path / "config.toml"
    config.write_text(f'[server]\nprojects_root = "{root.as_posix()}"\nallowed_hosts = ["gpu-box"]\n[storage]\ncache_dir = "{cache.as_posix()}"\n[auth]\nenabled = false\n[embeddings]\nenabled = false\n')
    installed = tmp_path / "yolo-workbench.service"
    installed.write_text(systemd_unit.render(systemd_unit.load(config), config))
    monkeypatch.setattr(server, "unit_sandbox", lambda: {**SANDBOX, "FragmentPath": str(installed), "ReadWritePaths": f"{root} {cache}"})

    def check() -> str:
        environment = os.environ.copy()
        try:
            return CliRunner().invoke(server.app, ["check", "--config", str(config)]).output
        finally:
            os.environ.clear()
            os.environ.update(environment)

    assert f"[ok  ] systemd unit: {installed} is written for this configuration" in check()
    config.write_text(config.read_text().replace("[auth]", "[training]\npause_deadline_s = 1800\n[auth]"))
    assert f"[warn] systemd unit: {installed} is not the unit for this configuration (TimeoutStopSec=1920 expected): sudo bash /opt/yolo-workbench/current/deploy/apply-config.sh, then restart" in check()


def test_library_caches_and_home_go_into_the_cache(tmp_path):
    """The service account needs no home: its home may not exist, and the unit's sandbox mounts it read-only
    (RF-DETR training failed on `~/.triton`). The training subprocesses inherit the environment."""
    from yolo_workbench import server
    from yolo_workbench.config import ServerConfig

    cache = tmp_path / "cache"
    environment = os.environ.copy()
    try:
        for name in ("TRITON_HOME", "TORCHINDUCTOR_CACHE_DIR", "TORCH_HOME", "HF_HOME"):
            os.environ.pop(name, None)
        os.environ["HOME"] = str(tmp_path / "srv" / "yolo-workbench")  # not there
        server.prepare_environment(ServerConfig(projects_root=tmp_path / "projects", cache_dir=cache))
        assert os.environ["TRITON_HOME"] == str(cache / "triton") and (cache / "triton").is_dir()
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(cache / "torchinductor") and os.environ["TORCH_HOME"] == str(cache / "torch")
        assert os.environ["HOME"] == str(cache / "home") and Path.home() == cache / "home" and Path.home().is_dir()
        child = {**os.environ}
        home = tmp_path / "home"
        home.mkdir()
        os.environ["HOME"] = str(home)
        server.prepare_environment(ServerConfig(projects_root=tmp_path / "projects", cache_dir=cache))
        assert os.environ["HOME"] == str(home)  # a writable home stays
    finally:
        os.environ.clear()
        os.environ.update(environment)
    pytest.importorskip("triton")  # Linux only
    where = subprocess.run([sys.executable, "-c", "from triton import knobs; print(knobs.cache.dir)"], env=child, capture_output=True, text=True, timeout=60)
    assert where.stdout.strip() == str(cache / "triton" / ".triton" / "cache"), where.stderr


def test_example_config_lists_every_setting_and_loads(tmp_path):
    from yolo_workbench.config import ServerConfig

    example = Path(__file__).parents[1] / "deploy" / "config.example.toml"
    text = example.read_text(encoding="utf-8")
    for item in fields(ServerConfig):
        assert f"{item.metadata['key'] or item.name} =" in text, f"{item.name} is missing from deploy/config.example.toml"
    config = load_config(example, environ={})
    assert config.allowed_hosts and config.training_pause_deadline_s == 600 and config.max_open_projects == 4
    defaults = ServerConfig()
    for item in fields(ServerConfig):
        if item.name not in ("projects_root", "allowed_hosts", "cache_dir"):
            assert getattr(config, item.name) == getattr(defaults, item.name), f"the example's {item.name} is not the default"
