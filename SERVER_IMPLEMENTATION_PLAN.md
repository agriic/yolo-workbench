# Server implementation plan

The stages below turn `SERVER_PLAN.md` and `SERVER_PLAN_2.md` into work for this deployment: the workbench runs
unattended on one Linux server and other people on the internal network use it from their browsers. Each stage
can be merged and used on its own. Stages 1–5 make up the **first deployment** (one operator at a time);
stages 6–9 add **multi-user** support; stage 10 is a backlog.

Checked against the repository on 2026-10-21 (after `7b7c4a9`, which unloads models before training).

## Implementation status (2026-10-21)

Stages 1–4 and 6 are implemented (commit `59c003f`); stages 7 and 8 are implemented on top (uncommitted at the
time of writing). Stage 5 was run as far as possible on a Linux workstation with an RTX 3070 Laptop GPU (8 GB),
without root: the service ran as a transient **user** systemd unit with the deploy unit's stop/restart settings.
Operator documentation: `docs/server.md`.

| Stage | State | Where / notes |
| --- | --- | --- |
| 1 | done | `config.py`, `server.py` (`run`, `check`), `ownership.py` (desktop CLI takes the lock too), `logs.py`, `runtime.py`; `/healthz`, `/readyz`; cache dirs and `YOLO_AUTOINSTALL=false` set before heavy imports; path import off and no cwd model scan in server mode. Added `[logging] access_log` (off: the UI polls every few seconds). |
| 2 | done | `lifecycle.py` + FastAPI lifespan in both factories; `TrainingManager.begin_shutdown`/`wait_for_pause`, `set_hold` (persisted), `auto_resume`, active entry saved before `Popen` and `_find_runner` reattach, `hold_queue_after_crash`; `JobManager.close`; `DatasetSession.close`. Interrupted imports are logged with `cancelled`. The desktop keeps its old behavior on exit (runs keep training). |
| 3 | done | streamed uploads (413/507), staging cleanup in every open project, `fsutil.ensure_free_space`, bounded undo, daily session rotation, retention (trash/backups 30 days, deleted projects 30 days as answered), idle unload of SAM and prediction models, GPU warning or refusal, model upload into `models/`, disk warning banner (`status.js`). `free_gpu_memory` also trims the heap (`malloc_trim`) after unloads. |
| 4 | done | `pyproject.toml` (`dev` group, `psutil`, `tomli`, `yolo-workbench-server` script), `deploy/install.sh`, `deploy/yolo-workbench.service`, `deploy/config.example.toml`, `deploy/yolo-workbench.env`, `docs/server.md`, README/CLAUDE.md. Changed after stage 5: `Restart=always` (was `on-failure`), start limit 10 per 5 min (was 5 per 10 min). |
| 5 | partly done here | See [Stage 5 results](#stage-5-results-on-the-workstation) below. Still to do on the real server, as root: `install.sh`, the unit's hardening options, a host reboot, a 24–72 h soak, an upgrade/rollback drill (a restore drill ran here). |
| 6 | done (pulled forward) | `Workspace.sessions` pool (`max_open_projects`, LRU close of idle projects, `close_idle`), unload before training and `max_loaded_models` across open projects. Model unloading revised on the server: SAM (one for everyone) and each project's prediction model unload after `idle_unload_minutes` (now 2) without use and load again by themselves on the next use (a parked prediction model keeps its mapping and pending predictions); a starting run unloads only its own project's prediction model (`LifecycleOptions.unload_for_training = "project"`; everything with `gpu_models_during_training = "refuse"`, and on the desktop). |
| 7 | done | `auth.py` (accounts in `.workbench/users.json` with scrypt hashes, hashed session tokens in `sessions.json` with sliding expiry, `AuthMiddleware`, `require_admin`), `yolo-workbench-server user add\|passwd\|remove\|admin\|list`, `[auth] enabled`/`session_days`, `/api/v1/auth/login\|logout\|me\|password`, `login.html`/`login.js`, `user.js` (header menu, change password). The cookie name carries a hash of the projects folder, so two servers on one host do not sign each other out. |
| 8 | done | `identity.py` (user, browser tab, action label as context variables; jobs run as who started them), per-user undo/redo in `dataset.py` (`Transaction`, exact-content and class-list checks, "changed since by …"), change feed (`revision`, `/changes`, polled by `collab.js`; polling never opens a closed project, a reopened one resets pages), `schema_generation` on saves (409 with the class mapping), presence (`/presence`, editor heartbeat), attribution (`review.json` reviewed/edited by, `activity.py` → `activity.jsonl`, `/activity`, reviewer filter), "started by" on imports, prediction batches and model loads, Label QA, embeddings, training runs and queue holds. Pending predictions stay shared per project (answered); presence only shows, it does not lock (default of question 7). |
| 9 | not started | nothing planned in detail yet |

Tests: `tests/test_server.py` (config, lock, lifespan and guards, service stop/resume, holds, launch reconciliation,
jobs, uploads, GPU policy, undo budget, sessions and retention, maintenance, idle unload, the example config, a real
server process), `tests/test_auth.py` (accounts, sessions, middleware, password changes, the `user` commands),
`tests/test_multiuser.py` (per-user undo and its conflicts, the change feed, presence, class-list revisions,
attribution, polling closed projects), plus updated `test_workspace.py`/`test_training.py`.
`test_augment_preview.py::test_chance_augmentations_and_pipeline_samples_render` fails now and then (2 of 8
full runs on 2026-10-21; it asserts two seeded renders are identical); unrelated to the server work.

### Stage 5 results on the workstation

Verified (service as a user unit with `Type=exec`, `KillMode=mixed`, `TimeoutStopSec=720`, the restart policy):

- `check` passes and lists the GPU, libraries and accounts; a second server or a desktop session on the same
  projects folder is refused; a wrong `Host` gets 403; `/readyz` ready about 2 s after start.
- Browser-upload import (360 images), gallery and annotated thumbnails, Training tab, best-weights download.
- Ultralytics (yolo11n): `systemctl restart` mid-epoch → paused at the epoch end (stop took 4 s) → resumed
  automatically, epoch rows continuous; `kill -9` of the server → systemd killed the runner with it, restarted the
  service after `RestartSec`; the run showed *interrupted*, **Resume** continued from its checkpoint to 40/40.
- RF-DETR (nano): the same restart → paused (11.5 s) → resumed with continuous epochs; `kill -9` → interrupted →
  resumed. A stop with a 1 s pause deadline: the server exited and systemd SIGKILLed the runner and its
  dataloader workers **as soon as the server exited** (docs corrected: not at `TimeoutStopSec`); the run was
  requeued and resumed at the next start.
- ONNX export interrupted by a restart, for both frameworks: no leftover process, no `export-tmp/`, no partial
  `.onnx`; the repeated export succeeded.
- Sign-in, change password, sign out, and two users in one project (presence, change notice and reload, class
  rename while the other's editor is open, per-user undo refused after the other's edit, activity list,
  "started by"), in the browser; Label QA and model upload through the API.

Found and fixed: the unit's start limit counted manual restarts (5 per 10 min stopped the service; now 10 per
5 min); a toast raised while the editor is open was invisible (drawn under the modal dialog); and during the soak
**earlyoom** (the workstation's OOM killer) sent the server SIGTERM at 5% free memory. The server paused the
active run and exited cleanly, so `Restart=on-failure` left the service down; it is `Restart=always` now. The
paused run resumed at the next start.

One RF-DETR run diverged (NaN in the matcher at epoch 10) after a kill and two resumes; resuming it from its last
good checkpoint finished 30/30 at the control run's accuracy, and the checkpoint holds the optimizer, scheduler
and EMA state, so this looks like RF-DETR's own instability on this tiny synthetic set.

Not verifiable here: `install.sh` and the system account (no root), the hardening options (`ProtectSystem` etc.
are silently ignored in user units), a host reboot, a long soak, an upgrade/rollback drill.

Soak (55 min, sign-in on, `gpu_models_during_training = "refuse"`): two signed-in users editing, undoing,
reviewing, polling the change feed and sending presence heartbeats, with 273 browser-upload imports each
followed by deleting the imported images, 549 predictions with prediction-model loads, 58 SAM uses and 37
three-epoch training runs (each unloading the models). 115,502 requests, 0 errors. Threads (24), file
descriptors (50), tasks and `/healthz` (median 14.5 ms, max 24 ms) stayed flat; the server's memory with the
models unloaded went 2,014 → 2,071 MB and with SAM loaded 3,319 → 3,389 MB, flat within ~60 MB over the run
(before the `malloc_trim` change it only grew). After the soak, an external SIGTERM to the server: systemd
restarted it within 5 s (`Restart=always`).

Restore drill (without root): the projects folder copied while the service was stopped and served by a fresh
server process: the existing accounts sign in, projects, runs, the activity log and "last edited by" are
there, editing and undo work, best weights download.

An independent code review of stages 7–8 found six defects, all fixed with regression tests: undoing a class
change over labels others had saved for the new class list, undoing a move leaving a label made since as an
orphan, a removed and re-added account reviving its old sessions, an open redirect on the login page
(`next=/%09/host`), a password change on a session-renewing request signing the person out, and
`user remove` failing on capital letters; plus adding a split folder invalidating saves and a rescan not
reporting removed images.


## Decisions taken

| Topic | Decision | Consequence for the plan |
| --- | --- | --- |
| Environment | Isolated internal network; only trusted people can reach it | No TLS, no reverse proxy, no path sandbox, no CSP/login throttling. Guards that protect against mistakes stay |
| Installation | **systemd on Linux**, dedicated service account | Docker moves to the backlog |
| Access | **Direct `http://host:port`** | App binds `0.0.0.0` with a configured Host allow-list. Passwords travel unencrypted (accepted for this network) |
| Mode | **Projects mode only** on the server | The server entry point refuses single-dataset mode; the desktop CLI stays as it is |
| Training on service stop/restart | **Pause at a checkpoint, then resume** | Stop asks the run to pause within a deadline (`KillMode=mixed`); runs do not have to survive the service |
| GPU | **One GPU** | Builds on the existing unload-before-training; adds idle unload. No multi-GPU scheduling |
| Data | **Local disk, browser upload only** | Import by server path is switched off in server mode; no network-filesystem concerns |
| Internet | **Direct access** | Weights keep downloading on first use; prefetch/offline mode moves to the backlog |
| Identity | **Built-in password login, added with multi-user** | No login in the first deployment |
| Roles | **Everyone equal**, with an `admin` flag reserved for a future superadmin | Stage 7 adds the flag and a `require_admin` check; what admins can do is decided later (stage 9) |
| Multi-user | **Several projects open and several users per project** | Stage 6 (projects open side by side) and stage 8 (same-project collaboration) |

### Dropped or deferred because the network is isolated

From the two source plans, these items are **not** planned: reverse-proxy authentication and forwarded-header
trust, TLS, enforcing allowed roots for every path (`data_roots` sandbox), restricting `.pt`/`.pth` loading to
"trusted" files, Content-Security-Policy, login throttling, and hiding host paths in error messages. The Host
allow-list and the `X-Workbench` header stay because they are free and stop accidents (a page in another tab,
a mistyped hostname).

## Overview

| Stage | Goal | Size | Needs |
| --- | --- | --- | --- |
| 1 | Server entry point, configuration, single owner, health, logging | M | – |
| 2 | Lifespan, clean shutdown, training pause on stop, queue hold | M–L | 1 |
| 3 | Long-running hygiene: uploads, retention, bounded memory, idle unload | M | 2 |
| 4 | systemd packaging, `check` command, docs | M | 1–3 |
| 5 | Acceptance on the real server | S–M | 4 |
| — | **First deployment** | | |
| 6 | Several projects open at once | M | 2 |
| 7 | Users and login (with `admin` flag) | M | 1 |
| 8 | Several users in one project | L | 6, 7 |
| 9 | Superadmin features (to be decided) | S | 7 |
| 10 | Backlog | – | – |

---

## Stage 1 — Server entry point and configuration

Goal: `yolo-workbench-server --config /etc/yolo-workbench/config.toml` starts in projects mode with no terminal,
no browser, and no prompts, and refuses to run twice on the same projects folder.

Files: new `config.py`, new `server.py`, `cli.py`, `web.py`, `training.py`, `workspace.py`, `pyproject.toml`.

- [ ] **Configuration object** (`config.py`): a dataclass validated at startup. Sources, strongest first: CLI option > `WORKBENCH_*` environment variable > TOML file > default. Paths in the file are relative to the file. TOML via `tomllib` (3.11+) and `tomli` on 3.10.
  Settings for now: `projects_root`, `create_projects_root` (default false, so a missing mount fails instead of creating an empty workspace), `host` (default `0.0.0.0`), `port`, `allowed_hosts` (required when `host` is a wildcard), `cache_dir`, `media_cache_mb`, `log_level`, `log_format` (`text`/`json`), `shutdown_deadline_s`, `training_pause_deadline_s`; later stages add their own.
- [ ] **Separate console script** `yolo-workbench-server` (in `server.py`), so the desktop command `yolo-workbench <data.yaml> --category …` keeps its arguments. Subcommands: `run` (default), `check` (stage 4), later `user` (stage 7).
- [ ] **Host allow-list stays on** with a wildcard bind: `allowed_hosts` lists the names and addresses people type (e.g. `gpu-box`, `gpu-box.corp.local`, `10.1.2.3`). The `X-Workbench` header check is always on in server mode. The desktop CLI keeps today's behavior.
- [ ] **Constructors without side effects**: `TrainingManager.__init__` calls `tick()` today (`training.py:270`), which can launch a queued run while the app is still being built. Move that first `tick()` into `start()`.
- [ ] **One owner per projects folder**: an OS lock (`fcntl.flock` on Linux, `msvcrt.locking` on Windows) on `<projects>/.workbench/owner.lock`, taken before any manager is created and held for the process lifetime. The desktop CLI takes it too, so a desktop session and the service cannot share a folder. The lock disappears with the process, so a crash does not leave a stale lock.
- [ ] **One process**: the server calls `uvicorn.run(app)` itself with `workers=1`, no reload. Pass `timeout_graceful_shutdown` from the config.
- [ ] **Writable state in known places**: before importing torch/Ultralytics/FiftyOne, point their directories at `cache_dir` (`XDG_CACHE_HOME`, `YOLO_CONFIG_DIR`, `TORCH_HOME`, `RF_HOME`, `MPLCONFIGDIR`; workbench media and weights already follow `XDG_CACHE_HOME`), so the service user needs no home folder. Switch off Ultralytics' runtime `pip install` (verify the setting for the pinned version, e.g. `YOLO_AUTOINSTALL=false`).
- [ ] **Health endpoints**: `GET /healthz` (process answers) and `GET /readyz` (startup finished, projects root writable, owner lock held). Readiness does not wait for image indexing, the GPU, or MongoDB.
- [ ] **Logging**: one `logging` setup for the app and uvicorn (timestamps, level, logger), text or JSON; replace the `print` in the training supervisor. Log project opens/closes, job start/end, run launches/reaps, model loads/unloads.
- [ ] **Server-mode switches**: import by server path off (`importer.resolve_source` refuses `path`; the UI hides "Import from folder"); model discovery skips the server's working directory.

Tests: config precedence and errors; wildcard host without `allowed_hosts` refused; Host check with a wildcard bind; second owner refused (real lock, two processes); `TrainingManager` construction launches nothing; `/healthz`/`/readyz`.

Done when: the server starts with stdin closed from any working directory, a bad config exits with a clear message, and a second instance (server or desktop) on the same folder refuses to start.

## Stage 2 — Lifespan, shutdown, and training on service stop

Goal: `systemctl stop|restart` leaves every run, job, and file in a known state, and an active training run
continues after the restart from its last epoch checkpoint.

Files: `web.py`, `server.py`, `training.py`, `train_runner.py`, `rfdetr_runner.py`, `jobs.py`, `session.py`, `workspace.py`, `predictor.py`, `labelqa.py`, `embeddings.py`, `sam.py`, `static/training.js`.

- [ ] **Shared FastAPI lifespan** for both app factories. Startup: recover state (below), start the training supervisor (moved from `cli.py`), start the maintenance thread (stage 3). Shutdown, in order:
  1. stop accepting new jobs and mutations (new requests get 503 "shutting down");
  2. hold the training queue so nothing new launches;
  3. ask the active run to **pause** (the existing pause in `control.json`) and wait up to `training_pause_deadline_s`; past the deadline, leave it to systemd's kill (the run becomes `interrupted`, resumable from its previous checkpoint);
  4. cancel imports, predictor batches, Label QA, embeddings, and ONNX exports, and join their threads within `shutdown_deadline_s`;
  5. flush the open dataset's size index and project summary, unload models, close the session;
  6. release the owner lock last.
- [ ] **Stoppable managers**: keep thread and process handles; add idempotent `close()`/`join()` to the training supervisor (`shutdown()` only sets an event today), `JobManager`, background image probing, `SamManager`, `PredictorManager`, `LabelQAManager`, `EmbeddingsManager`. Do not join while holding a lock the worker needs.
- [ ] **Resume after a planned stop**: a run paused by the service is marked `paused_by: "service"` in its run state. At startup such runs go back to the front of the queue and resume automatically (`resume_after_service_stop`, default true). Runs paused by a user stay paused.
- [ ] **Crashes** (OOM, kill, power loss): reattach to live runners as today (PID + create time); dead ones become `interrupted` as today. Keep the queue running unless `hold_queue_after_crash` is set; the UI offers Resume when a checkpoint exists, and says plainly when none does.
- [ ] **Queue hold** (also the "drain" for maintenance): a persisted `held` flag in `queue.json`, set from the Training tab's queue chip or `POST /api/v1/training/queue/hold`. "Hold and pause active" is the maintenance button: it pauses the active run at the next checkpoint and launches nothing until released. It stays set across a restart.
- [ ] **Launch records before launching**: write the run's state as `launching` (with a launch id) before `Popen`, have the runner confirm the id in `state.json`, and on startup reconcile `launching` runs (confirmed and alive → attach; never confirmed → `failed to start`, no duplicate launch).
- [ ] **ONNX exports**: on startup, remove leftover `workbench/export-tmp/` folders and mark exports that were running as failed; refuse deleting a run while its export runs.
- [ ] **Imports interrupted by shutdown**: cancel at a file boundary (already supported by the job), write an `interrupted` entry to `imports.jsonl` with the files copied so far, rescan at next open. Startup cleans staging dirs as today.
- [ ] systemd settings this stage relies on (delivered in stage 4): `KillMode=mixed` (SIGTERM only to the server, which coordinates the runner; SIGKILL to everything left at the timeout), `TimeoutStopSec` ≥ `training_pause_deadline_s` + `shutdown_deadline_s` + margin.

Tests (entering the app's lifespan in the test harness, since `ASGITransport` alone does not run it): shutdown order and thread joins; fake runner paused by shutdown and resumed at startup; `launching` reconciliation; hold flag persisted; interrupted export cleanup. Real-process test: SIGTERM to a server with a fake runner, runner pauses, server exits within the deadline.

Done when: stopping the service during training pauses the run at an epoch checkpoint and the run continues after the start; a `kill -9` of the whole service leaves the run `interrupted` and resumable; nothing launches twice.

## Stage 3 — Long-running hygiene

Goal: weeks of uptime without growing memory or disk use, and without one big upload blocking the server.

Files: `web.py`, `importer.py`, `dataset.py`, `storage.py`, `workspace.py`, `sam.py`, `predictor.py`, `training.py`, `static/manage.js`, `static/predictor.js`.

- [ ] **Streamed uploads**: `staging_upload` reads `request.stream()` into a temporary file in chunks (writes off the event loop), counts real bytes, returns 413 past `max_upload_file_mb`; limits per staging dir (`max_upload_files`, `max_upload_total_mb`); a broken connection removes the partial file.
- [ ] **Maintenance thread** (every `maintenance_interval_min`): staging dirs untouched for 24 h are removed in **every** open project (today only on project open); retention purge (below); a free-disk check that the UI shows as a warning banner under `min_free_disk_gb`.
- [ ] **Free-space checks** before an upload, an import, a run launch, and an ONNX export, with a clear error instead of a half-written file.
- [ ] **Bounded undo**: keep at most `undo_max_transactions` (default 200) and `undo_max_mb` per dataset; the oldest drop off. The UI already handles "nothing to undo".
- [ ] **Backup/trash sessions on a long-lived project**: today a session lasts as long as the project stays open. Start a new session id daily (and on reopen); the "first change backs up the original" bookkeeping resets with it. Trash and backups of earlier sessions become purgeable **unless** an entry still on the undo/redo stack points into them (the purge checks this).
- [ ] **Retention**: `trash_retention_days` and `backup_retention_days` (default 30, 0 = keep forever) purge earlier sessions automatically, skipping referenced ones. Workspace `.trash/` (deleted projects) gets `project_trash_retention_days` (default off). Training runs and models are never deleted automatically.
- [ ] **Idle model unload**: SAM and the predictor's model unload after `idle_unload_minutes` (default 30) without use, through the same path as the existing unload before training (under `backend_lock`, references dropped before `free_gpu_memory()`); they reload on the next use.
- [ ] **GPU during training**: keep today's behavior (models are unloaded when a run starts; the user can turn them on again) and add a warning in the UI when SAM or a prediction model is loaded while a run trains on the same GPU. Config `gpu_models_during_training` = `allow` (default) / `refuse`.
- [ ] **Getting model files in**: with browser-only access, add "Upload model" (`.pt`, `.onnx`, `.pth`) into `<project>/models/`, which model discovery already lists. Acceptable here because everyone on the network is trusted; loading a weights file runs code from it.

Tests: chunked upload without `Content-Length`, over-limit 413, disconnect cleanup; staging age cleanup with an active upload; undo bounds; session rotation plus purge refusing a session referenced by undo; idle unload with a fake backend; model upload.

Done when: repeated imports, project switches, and model loads over a 24 h test show flat memory, and disk use only grows with data the user keeps.

## Stage 4 — systemd packaging and documentation

Goal: a clean Linux host gets a running, boot-enabled service by following `docs/server.md`.

Files: `pyproject.toml`, `uv.lock`, new `deploy/`, new `docs/server.md`, `README.md`, `server.py`.

- [ ] **Dependencies**: move `pytest`/`httpx` to `[dependency-groups] dev`; declare `psutil` (imported by `training.py`) and `tomli` (Python 3.10) directly. Baseline Python 3.12 on Linux (installed by `uv python install`).
- [ ] **Release layout**: `/opt/yolo-workbench/releases/<version>/` (source + `.venv` from `uv sync --frozen --no-dev`), `/opt/yolo-workbench/current` symlink, `/etc/yolo-workbench/config.toml`, `/srv/yolo-workbench/projects/`, `/var/cache/yolo-workbench/`. Service account `yolo-workbench` owns only `/srv/…` and `/var/cache/…`; code is read-only to it.
- [ ] **`deploy/yolo-workbench.service`**: `ExecStart=/opt/yolo-workbench/current/.venv/bin/yolo-workbench-server run --config /etc/yolo-workbench/config.toml`, `User=`, `RequiresMountsFor=/srv/yolo-workbench`, `Restart=on-failure` with `RestartSec` and start-rate limits, `KillMode=mixed`, `TimeoutStopSec` (stage 2), journald output, `NoNewPrivileges=yes`, `ProtectSystem=strict`, `ReadWritePaths=/srv/yolo-workbench /var/cache/yolo-workbench`, `ProtectHome=yes`, `PrivateTmp=yes`. Test each hardening option against GPU access and the training subprocesses before keeping it.
- [ ] **`deploy/config.example.toml`** with every setting and its default, commented.
- [ ] **`yolo-workbench-server check`**: config valid, folders exist with the right owner and free space, owner lock free, torch sees the GPU (name, memory, driver), Ultralytics/RF-DETR import, MongoDB reachable when embeddings are enabled. Exit code for scripts.
- [ ] **Embeddings**: `embeddings.enabled` and `embeddings.mongodb_uri` in the config, passed as `FIFTYONE_DATABASE_URI` before FiftyOne is imported; document MongoDB (Docker container on `127.0.0.1:27017`, as in the README) as optional. Everything else keeps working without it.
- [ ] **`docs/server.md`**: install, config reference, firewall (open only the app port), start/stop/logs (`journalctl -u yolo-workbench`), maintenance (hold queue → stop), **backup** (whole projects root including `.workbench/`, `.training/`, `.trash/`, every project's `.yolo-workbench/` and `runs/`; take it with the queue held, or from a filesystem snapshot), **restore** on a fresh host, **upgrade** (hold queue and wait for the pause, install the new release next to the old one, switch `current`, restart, check `/readyz`, release the queue; rollback = switch back), troubleshooting.

Done when: on a fresh VM (CPU is enough) and on the GPU server, the documented steps give a service that starts at boot, survives logout, restarts after a crash, and upgrades and rolls back without touching the projects folder.

## Stage 5 — Acceptance on the real server

Goal: confidence before people rely on it. No new features; fixes go back into stages 1–4.

- [ ] Install from `docs/server.md` on the target server; `check` passes with the GPU.
- [ ] Browser checks from another machine: project list, editing and a 409 conflict, upload import, thumbnails, SAM, predictions, Label QA, training progress, download of best weights, embeddings on and off.
- [ ] Training lifecycle with a tiny dataset, **for Ultralytics and RF-DETR**: `systemctl restart` during an epoch (pause → resume), `kill -9` of the service (interrupted → resume from checkpoint), host reboot, ONNX export interrupted by a restart.
- [ ] 24–72 h soak with a scripted workload (edits, imports, model loads, a short run): memory, disk, threads, and child processes stay flat; `/healthz` stays fast.
- [ ] Backup/restore drill onto a second machine; upgrade and rollback drill.

### First deployment

After stage 5 the server is ready for **one person at a time**. Known limitation until stage 6: in projects
mode every request opens the project it names and closes the previous one (`workspace.session()`), so two
browsers on **different** projects keep reloading each other's dataset and lose each other's undo history.
Pull stage 6 forward if several people will start using the server right away.

---

## Stage 6 — Several projects open at once

Goal: people work on different projects at the same time without disturbing each other. No identity needed yet.

Files: `workspace.py`, `web.py`, `training.py`, `session.py`, `predictor.py`, `static/app.js`, `static/projects.js`.

- [ ] Replace `Workspace.active` with a pool of open `DatasetSession`s: up to `max_open_projects` (default 4), a project with no requests for `project_idle_close_min` closes when nothing runs in it (`busy_reason()`); when the pool is full, the least recently used idle project closes; when all are busy, opening another one gets a 409 with the reason.
- [ ] Training: `attach()` the write guard on every open dataset of a project with an active run; `training_target()` uses the pool instead of `active`.
- [ ] GPU: the unload before training (`_unload_before_training`) covers every open session, not only the current one; idle unload (stage 3) runs per session; at most `max_loaded_models` prediction models across sessions (least recently used unloads).
- [ ] Project cards show which projects are open; covers/summaries refresh per session.
- [ ] Memory check: measure one open dataset's footprint (records, size index, caches) to pick the default pool size.

Tests: two projects used alternately keep their undo history; pool eviction and busy refusal; training lock on a pooled project; unload before training across sessions.

## Stage 7 — Users and login

Goal: everyone signs in; the app knows who made each request. All users can do everything; one or more can be
marked `admin` for later use.

Files: new `auth.py`, `web.py`, `server.py`, new `static/login.html`/`login.js`, `static/api.js`, `static/app.js`, `static/projects.js`.

- [ ] **Users file** `<projects>/.workbench/users.json`: name, display name, password hash (`hashlib.scrypt` with a per-user salt; no new dependency), `admin` flag, created/last-login times. Managed by `yolo-workbench-server user add|passwd|remove|list [--admin]`; the first user is created at install time.
- [ ] **Sessions**: random tokens stored server-side (hashed, in `.workbench/sessions.json`), so logout and password changes end them; cookie `HttpOnly`, `SameSite=Lax`, no `Secure` (plain HTTP); expiry `session_days` (default 30) with sliding renewal.
- [ ] **Middleware**: every page and `/api/` route needs a session except the login page and its assets, `/healthz`, `/readyz`. API requests without one get 401; pages redirect to the login page and back. `auth.enabled` (default true in server mode, always off in the desktop CLI).
- [ ] **Current user** in a `contextvars.ContextVar` set by the middleware (FastAPI copies it into threadpool endpoints); background jobs capture it at start. Stage 8 reads it for attribution.
- [ ] **UI**: user name and Log out in the header; "Change password" dialog; admins see a Users page later (stage 9).
- [ ] **`require_admin`** dependency exists and is tested, but no route uses it yet.

Tests: login/logout, expired and revoked sessions, 401 vs redirect, password change ends other sessions, CLI user management, health endpoints stay open, desktop mode unaffected.

## Stage 8 — Several users in one project

Goal: annotators split a project's images and work at the same time; nobody silently overwrites or undoes
another person's work, and everyone sees who did what.

Files: `dataset.py`, `review.py`, `classes.py`, `predictor.py`, `labelqa.py`, `web.py`, `static/app.js`, `static/canvas.js`, `static/grid.js`, `static/manage.js`.

- [ ] **Attribution**: every transaction records its user; `<dataset>/.yolo-workbench/activity.jsonl` logs user, time, action, image ids; review status stores `by`/`at`; the editor's inspector shows "last edited by … at …"; the gallery can filter by reviewer.
- [ ] **Per-user undo/redo**: each user undoes only their own transactions. If someone else changed the same files afterwards, the undo is refused with "changed since by …" (409). History bounds from stage 3 apply per dataset.
- [ ] **Change feed**: a dataset revision number bumped by every transaction, rescan, and import; `GET /changes?since=<rev>` returns changed image ids and the schema revision. The UI polls it every few seconds: gallery cells refresh, the editor shows "changed by … — reload" for the open image, and a 409 on save offers reload.
- [ ] **Class schema changes**: renames, merges, removals, and reorders rewrite every label. Clients send their schema revision with saves; a save made against an old schema gets 409 and the client reloads classes before retrying, so no label is written with stale class ids.
- [ ] **Presence**: show who has which image open (heartbeat from the editor) and mark it in the gallery, so people avoid editing the same image.
- [ ] **Shared jobs**: imports, Label QA, embeddings, prediction batches, and training show who started them.
- [ ] **Pending predictions**: decide shared vs per user (see open questions); the default plan keeps them shared per dataset and shows who ran the batch.

Tests: two simulated users editing the same and different images; undo refused across users; change feed; schema change while another client saves; attribution entries.

## Stage 9 — Superadmin features (to be decided)

Goal: give `admin` users the extra powers that turn out to be needed. Nothing is planned in detail yet.
Candidates: user management in the UI, server settings (retention, idle unload, limits) without editing
the config file, permanently deleting projects and purging trash, holding the queue / maintenance mode,
viewing logs and disk usage. Each one becomes a route behind `require_admin` from stage 7.

## Stage 10 — Backlog (not needed for this deployment)

- **Crash consistency for multi-file operations**: a recovery journal so a power loss halfway through a class rename or a bulk move can be rolled back or finished automatically, and an import manifest for resuming imports. Until then, each session's backups hold the original of every changed label file, so recovery is possible by hand.
- Docker image and Compose file (with MongoDB); reverse proxy with TLS and/or company SSO; serving under a sub-path (`/workbench/`), which needs a frontend URL helper because the frontend uses root-absolute URLs.
- Weight prefetch command and offline mode (needed if the server loses internet access).
- Training that keeps running while the web service restarts (runner in its own systemd unit or a worker service).
- Metrics endpoint, alerting recipes, admin dashboard.

## Open questions (with the default this plan assumes)

Answer before the stage starts; the default is what gets built otherwise.

1. **Stage 2 — pause deadline on service stop.** Pausing waits for the end of the current epoch. Default 10 minutes, then the run is killed and resumes from the previous epoch's checkpoint. Long epochs on big datasets may need more (`TimeoutStopSec` grows with it).
2. **Stage 2 — automatic resume.** Default: runs paused by a service stop resume automatically at start; runs interrupted by a crash wait for a manual Resume while the rest of the queue continues.
3. **Stage 3 — retention.** Default: dataset trash and label backups of earlier sessions deleted after 30 days; deleted projects kept (restorable) for 30 days, after that fully deleted - or purgable by hand.
4. **Stage 3 — model upload.** Default: anyone can upload weights into a project's `models/` folder (trusted network). 
5. **Stage 6 — pull forward?** "several projects open" should ship with the first deployment
6. **Stage 8 — pending predictions.** Shared per project (default; simple, everyone sees and can accept them) or per user? Shared per project.
7. **Stage 8 — presence.** Only show who has an image open (default), or also lock that image for others until it is closed?
