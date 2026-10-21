# Plan: running the workbench as a standalone server service

Goal: the workbench can be installed on a (GPU) server, started by systemd or Docker, left running for
weeks, and used from other machines' browsers — safely, without anyone at the server's console.

Today the app is built for **one user on one machine**: it binds `127.0.0.1`, opens a browser, has no
login, lets the client name any path on the host, and keeps per-process state (one open project, one undo
stack, in-memory predictions) that assumes one person. This plan lists what has to change, in the order
it should be done.

## 0. Decisions to make first

| Question | Recommended default | Why it matters |
| --- | --- | --- |
| Who uses it? One operator remotely, or a small team at once? | Design for a **small trusted team** (2–10 people), not a public/multi-tenant service | Decides how much of §4 (multi-user) is needed |
| Mode | **Projects mode** only for server deployments | Projects mode already owns its data under one root; single-dataset mode can stay for local use |
| Packaging | **Docker image** (CUDA base) + a **systemd unit** example for bare-metal | Both are common; the systemd path needs the training-process fix in §5 |
| Auth | **Built-in login** (users file) plus optional **trusted proxy header** | No auth = anyone on the network can run code on the server (see §2.2) |
| TLS | Terminated by a reverse proxy (Caddy/nginx) | Keeps the app simple |
| Target OS | Linux (Windows server stays "works, unsupported") | systemd/Docker, signals, process groups |

## 1. What blocks server use today (inventory)

| Area | Current behaviour | Where |
| --- | --- | --- |
| Startup | `typer.confirm()` when the projects folder does not exist (blocks with no TTY); opens a browser unless `--no-browser` | `cli.py` |
| Config | CLI flags only; no env vars or config file | `cli.py` |
| Host check | Binding `0.0.0.0`/`::` **disables** the Host allow-list entirely; no way to list the server's real hostname | `cli.py`, `web._base_app` guard |
| Auth | None. Only the `X-Workbench: 1` header (CSRF) | `web._base_app` |
| Server paths from the client | Import from any local folder (`importer.resolve_source`), load any model file (`PredictorManager.load`), train from any `model:`/`pretrain_weights` path (`frameworks.validate`), model discovery scans the cwd | `importer.py`, `predictor.py`, `frameworks.py` |
| Code execution | `.pt`/`.pth` weights are pickles; loading one runs arbitrary code. Today "the user's own files", on a server "whatever anyone uploaded or pointed at" | `predictor.py`, `sam.py`, training runners |
| Uploads | `await request.body()` reads each uploaded file fully into RAM; no size or count limit; abandoned staging dirs are never cleaned | `web.staging_upload`, `importer.write_staged_file` |
| One open project | `workspace.session(slug)` closes the previous project; two users on two projects kick each other out (or get 409 while a job runs) | `workspace.py` |
| Shared undo | One undo/redo stack per `Dataset`; user A's Ctrl+Z undoes user B's edit | `dataset.py` |
| Unbounded memory | Undo stack keeps full before/after label contents forever; loaded models (SAM, predictor) stay in RAM/VRAM forever | `dataset.py`, `sam.py`, `predictor.py` |
| "Session" = process | Backups/trash are per `Dataset.session_id` (created at load). In single mode that is the whole process lifetime, so the current session's trash can never be purged on a server that runs for months | `dataset.py`, `storage.py` |
| Shutdown | No lifespan hook: training supervisor and job threads are daemon threads; `index.json` progress and job state are just dropped | `web.py`, `training.py`, `jobs.py` |
| Training under systemd/Docker | Runs are detached processes in their own session, but still in the service's **cgroup**: systemd's default `KillMode=control-group` (and `docker stop`) kills them on every restart | `training.py` launcher |
| GPU sharing | SAM, predictor, Label QA and training can all land on the same GPU; nothing coordinates VRAM | `sam.py`, `predictor.py`, `labelqa.py`, `training.py` |
| Runtime downloads/installs | Pretrained and SAM weights download on first use; Ultralytics may pip-install missing packages at runtime (e.g. pi-heif) | `frameworks.py`, `sam.py`, Ultralytics |
| Home-dir state | `~/.cache/yolo-workbench/{media,weights}`, `~/.fiftyone/config.json`, Ultralytics settings in `~/.config/Ultralytics` | various |
| URLs | Frontend uses absolute `/api/v1/...` and `/projects.html`; cannot be served under a sub-path (`https://host/workbench/`) | `static/*.js` |
| Logging | `print()` to stderr in a few places; no logging config, no request IDs | `training.py`, runners |
| Health | No health/readiness endpoint (only `/api/v1/app`) | `web.py` |
| Dependencies | `pytest`/`httpx` are core dependencies | `pyproject.toml` |

## 2. Phase 1 — headless service basics

### 2.1 Configuration
- Add a config layer: CLI flag > env var (`WORKBENCH_*`) > optional TOML file (`--config /etc/yolo-workbench.toml`) > default.
  Settings: `projects_root`, `host`, `port`, `allowed_hosts` (list), `base_path`, `data_roots` (§3.2),
  `cache_dir`, `media_cache_max_mb` (today fixed 512 MB), `max_upload_mb`, `auth.*`, `log_level`, `log_format`,
  `inference_device`, `trash_retention_days`, `idle_unload_minutes`.
- `--projects` + non-existent folder: create it when `--yes`/non-interactive (no TTY), never prompt.
- Add a `serve` subcommand (or `--service` flag) that implies `--no-browser`, refuses single-dataset mode, and validates config at startup with clear errors.

### 2.2 Host check and proxies
- `allowed_hosts` must be configurable and **stay on** when binding `0.0.0.0` (today it is switched off). Default when binding a wildcard: refuse to start unless `allowed_hosts` is set (or `*` explicitly).
- Behind a proxy: pass `proxy_headers=True`, `forwarded_allow_ips=<proxy IP>` to uvicorn; check the Host the proxy forwards.

### 2.3 Base path
- Support `base_path` (e.g. `/workbench`): uvicorn `root_path` plus a frontend that builds every URL from it.
  Simplest: inject `<base href>` / a `window.WORKBENCH_BASE` into `index.html`/`projects.html` at serve time and make `api.js`, `app.js`, `projects.js`, `sam.js`, `training.js`, `manage.js` use relative paths through one helper. Optional — skip if the service always gets its own (sub)domain.

### 2.4 Lifecycle
- FastAPI `lifespan`: on startup start the training supervisor (now in `cli.py`); on shutdown stop the supervisor, cancel/finish running jobs (imports, predictor batch, Label QA, embeddings) with a timeout, flush the dataset's size index, close the open session. Training runs are **not** stopped (they are designed to outlive the server).
- Handle SIGTERM through uvicorn's normal path; set `timeout_graceful_shutdown`.

### 2.5 Health and logging
- `GET /api/v1/health` (liveness, no auth, cheap) and `GET /api/v1/ready` (projects root writable, open project not still indexing, GPU visible if configured).
- Replace `print` with `logging`; one logging config for app + uvicorn; optional JSON format; request ID middleware; log job start/end, training launches, model loads, auth events.

## 3. Phase 2 — security (required before exposing it on any network)

### 3.1 Authentication
- Built-in minimal login:
  - users file (`users.toml` / JSON) with argon2 or bcrypt hashes, managed by a CLI command (`yolo-workbench user add|passwd|remove`); no account creation over the web.
  - `POST /api/v1/auth/login` → signed session cookie (`HttpOnly`, `Secure` behind TLS, `SameSite=Strict`), logout, `GET /api/v1/auth/me`; login page served without auth.
  - Middleware: every `/api/` route and every static page except the login page and `/api/v1/health` requires a session.
  - Login throttling (per IP + per user) to slow down guessing.
- Alternative mode `auth.mode = "proxy"`: trust a user header (`X-Forwarded-User`, `Remote-User`) **only** from `forwarded_allow_ips` — for oauth2-proxy, Authelia, Tailscale serve, etc.
- `auth.mode = "none"` stays for local use and is refused when binding a non-loopback address unless explicitly forced.
- Keep the `X-Workbench` header check (still useful with cookies) and the Host check.
- Roles (optional, later): `viewer` (read-only), `annotator` (labels, review), `admin` (classes, imports, training, model loading, deleting projects, trash purge). At minimum, model loading and training should be admin-only because of §3.3.

### 3.2 Filesystem sandbox
- New setting `data_roots`: the only host folders the client may name. Everything else is rejected with 400.
  - `importer.resolve_source`: local-folder import only inside `data_roots` (or disabled when empty — browser upload still works).
  - `PredictorManager.load` / `discover_models`: model paths only inside the projects root (runs, `models/`) and `data_roots`; drop the cwd scan in server mode.
  - `frameworks.validate`: `model:`/`pretrain_weights`/other path-valued args only inside those roots; pretrained names keep working.
  - `workspace.adopt` already works inside the projects root — keep it that way.
- Resolve symlinks before checking (`Path.resolve()` then `is_relative_to`), as `run_file` already does.

### 3.3 Weights are code
- Document clearly that uploading or pointing at `.pt`/`.pth` files = running code as the service user.
- Allow model upload (into `<project>/models/`) only for admins; consider `torch.load(weights_only=True)` where the framework allows (RF-DETR checkpoints, SAM) and treat Ultralytics `.pt` as trusted-only.
- Run the service as an unprivileged dedicated user with write access only to the projects root and cache dir (systemd `ProtectSystem=strict`, `ReadWritePaths=`; Docker non-root + volumes).

### 3.4 Uploads
- Stream upload bodies to the staging file in chunks (`request.stream()`), enforce `max_upload_mb` per file, a file-count and total-size quota per staging dir, and free-disk checks before writing.
- Clean staging dirs older than N hours on startup and periodically.

### 3.5 Response hardening
- Security headers on static pages: `Content-Security-Policy` (self only; the frontend has no external assets — verify), `X-Content-Type-Options: nosniff`, `Referrer-Policy: same-origin`, `frame-ancestors 'none'`.
- Make sure error messages do not leak absolute host paths to non-admins (many `DatasetError` messages include paths; acceptable for a trusted team, revisit if roles are added).

## 4. Phase 3 — more than one user at a time

Skip or shrink this phase if the deployment is "one person, remotely". Otherwise:

- **Several open projects.** Replace "exactly one open project" in `Workspace` with a small pool of open `DatasetSession`s (LRU, configurable max, idle timeout closes a session that has no running job). The busy check (`busy_reason()`) stays per session. Memory budget: each open dataset holds its index and records.
- **Undo per user.** Tag every transaction with the user (or client id) that made it; Ctrl+Z only undoes the caller's latest transaction, and only if no later transaction from someone else touched the same files (otherwise 409 with a clear message). Keep the stack bounded (e.g. 200 transactions or N MB) — also needed for a single user on a long-running server.
- **Attribution.** Record who changed what: an append-only `.yolo-workbench/activity.jsonl` (user, time, action, image ids); store `by`/`at` with review statuses (`review.json`); show it in the editor/inspector.
- **Stale views.** Edits are already safe (mtime conflict → 409), but other users only see changes after a reload. Add a dataset revision counter (`GET /dataset` already polled for indexing) and have the UI offer "dataset changed — refresh" and re-fetch the open image on 409.
- **Predictions and Label QA.** Pending predictions are per dataset in memory; decide whether they are shared (simplest, document it) or per user. Label QA and embeddings are dataset-wide jobs — fine as shared.
- **Locks visible in the UI.** Show who is running an import/training that makes the dataset read-only.

## 5. Phase 4 — long-running stability

- **Backup/trash sessions.** Rotate `Dataset.session_id` (e.g. daily, and when the open project changes) so older sessions become purgeable; add `trash_retention_days` / `backup_retention_days` with an automatic purge of earlier sessions (never the current one), plus disk-usage warnings in the Dataset tab.
- **Idle model unload.** Unload SAM and the predictor's model after `idle_unload_minutes` (free VRAM for training); reload on demand. Same for embeddings' FiftyOne resources.
- **Bounded memory.** Cap the undo stack (above); check caches stay bounded (augment image LRU and SAM encode LRU already are; `JobManager._prune` already prunes finished jobs).
- **External changes.** Optional periodic `rescan()` (or `watchfiles`) for datasets edited by other tools on the server.
- **Restart safety.** Imports interrupted by a restart: mark them failed on startup and clean their partial state; the import log already exists.

## 6. Phase 5 — training on a server

- **Keep runs alive across service restarts.**
  - systemd: either `KillMode=process` (only the server is killed; runners keep running and are re-attached via `process.json`), or — cleaner — launch each runner in its own transient unit (`systemd-run --user --scope` / `--unit`) when available. Pick one, document it, test restart during a run.
  - Docker: a runner dies with its container. Either accept it (runs become resumable from `last.pt`/`last.ckpt` — show "interrupted, resume?" on restart instead of a failed run) or run the training runner in a separate long-lived worker container sharing the projects volume. Recommendation: accept + clean auto-resume prompt first; separate worker only if needed.
- **GPU assignment.** Settings for which GPU(s) training uses and which the web process uses for SAM/predictor/Label QA (`inference_device`, `CUDA_VISIBLE_DEVICES` for runners). On a single GPU: optionally refuse or warn before loading SAM/predictor while a run is training, and unload them idle (§5).
- **Disk.** Check free space before launching a run and before ONNX export; show run folder sizes.
- **Upgrades while training.** Replacing the venv under a running runner can break its lazy imports. Add a "drain" admin action (pause queue, wait for/pause the active run) and document the upgrade procedure.

## 7. Phase 6 — packaging

- **Dependencies.** Move `pytest`/`httpx` to a `[dependency-groups] dev` group; keep a lock that installs on Linux CUDA hosts.
- **Offline / pinned behaviour.**
  - `yolo-workbench fetch-weights [--all|names…]` to pre-download pretrained YOLO/RF-DETR and SAM 2.1 weights into the cache dir (for air-gapped servers and the Docker image build).
  - Turn off Ultralytics runtime pip installs (its auto-install setting/env var — verify the exact switch for the pinned version) and make pi-heif a real dependency only.
- **Writable state in one place.** Default every cache (`media`, `weights`, Ultralytics settings dir `YOLO_CONFIG_DIR`, `RF_HOME`, torch hub `TORCH_HOME`, matplotlib `MPLCONFIGDIR`) under `cache_dir` in server mode, so the service user needs no real home. FiftyOne: accept `FIFTYONE_DATABASE_URI` via env instead of `~/.fiftyone/config.json`.
- **Docker image.**
  - `nvidia/cuda` runtime base (or `python:3.12-slim` + PyPI CUDA wheels, which already bundle CUDA), `uv sync --frozen --no-dev`, non-root user, `EXPOSE 8765`, `HEALTHCHECK` on `/api/v1/health`.
  - Volumes: `/data/projects` (projects root), `/data/cache`, optional read-only `/data/import` (a `data_roots` entry).
  - `docker-compose.yml`: workbench (GPU via NVIDIA Container Toolkit), `mongo:7` for embeddings, optional Caddy for TLS; `shm_size` large enough for PyTorch dataloader workers.
- **systemd unit** example (`deploy/yolo-workbench.service`): dedicated user, `EnvironmentFile=`, `Restart=on-failure`, `KillMode=process` (see §6), `NoNewPrivileges=yes`, `ProtectSystem=strict`, `ReadWritePaths=<projects> <cache>`, `PrivateTmp=yes`.
- **Reverse proxy examples** (`deploy/Caddyfile`, `deploy/nginx.conf`): TLS, `client_max_body_size` ≥ `max_upload_mb`, `proxy_read_timeout` long enough for slow synchronous endpoints (SAM encode, single-image predict, augmentation previews, ONNX download), buffering off for large file downloads, forwarded headers.

## 8. Phase 7 — operations

- Docs: a `docs/server.md` (or README section) covering install, config reference, users, proxy, GPU, backups, upgrade/drain, troubleshooting.
- Backups: what to back up — the projects root (data, `project.json`, `.yolo-workbench/` state, `runs/`); what can be skipped (`.yolo-workbench/staging`, media cache, weights cache).
- Optional metrics endpoint (Prometheus): open sessions, job counts, training queue length/state, GPU memory, upload bytes.
- Admin page (optional): users, active sessions, jobs, disk usage per project, drain/resume queue.

## 9. Tests

- Config precedence and validation; refusing unsafe combinations (wildcard bind without `allowed_hosts`, `auth=none` on a public address).
- Auth: unauthenticated API/static requests rejected, login/logout, throttling, proxy-header mode trusts only configured IPs.
- Sandbox: import/model/training paths outside `data_roots` rejected, symlink escapes rejected.
- Uploads: streamed, size limit enforced, stale staging cleanup.
- Multi-user: two sessions open at once; per-user undo refuses to undo over someone else's later edit; activity log entries.
- Lifespan: shutdown stops jobs and flushes the size index; startup marks interrupted imports.
- Training: runner survives a simulated server restart (existing fake launcher), interrupted runs offered for resume.
- All with the existing patterns: temp datasets, `create_app`/`create_workspace_app` via httpx `ASGITransport`, no live server.

## Suggested order and rough size

1. Phase 1 (config, headless, Host check, lifespan, health, logging) — small.
2. Phase 2 (auth, path sandbox, upload streaming/limits, headers) — medium; **minimum bar before network exposure**.
3. Phase 6 packaging (Dockerfile, compose, systemd, proxy examples, fetch-weights) + the training/KillMode fix from Phase 5 — medium.
4. Phase 4 stability (session rotation/retention, idle unload, bounded undo) — small–medium.
5. Phase 3 multi-user (session pool, per-user undo, attribution, refresh hints) — largest; only if several people use it at once.
6. Phase 7 ops extras (metrics, admin page) — optional.

After steps 1–3 the tool is deployable for one trusted user or a small team that coordinates; step 5 makes concurrent team use comfortable.
