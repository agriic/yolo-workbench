# Server deployment plan, revision 2

Status: implementation plan; no server deployment changes have been made.
Reviewed against the repository on 2026-10-18.

The target is a service that starts after a server reboot, runs without a terminal or logged-in desktop user, accepts authenticated browser access, and can be stopped, upgraded, backed up, and recovered predictably. Adding a process supervisor is only part of this: the application also needs explicit ownership of background work and limits on resources that currently live for a desktop session.

## 1. Recommended first deployment

Use one Linux host, a dedicated unprivileged service account, one Uvicorn process, local persistent storage, and systemd. Put an authenticated HTTPS reverse proxy in front of the application. Use a dedicated hostname served at `/`. MongoDB is an optional dependency for embeddings; a GPU is required only for workloads configured to require one.

Assumptions for the first release:

- One trusted operator at a time, with the existing shared workspace and undo behavior. This is remote access to the current workbench, not a concurrent team annotation service.
- Projects mode is the recommended default. Keep single-dataset mode supported; it does not inherently prevent deployment as a service.
- Exactly one web process owns a workspace. Do not enable Uvicorn/Gunicorn multiple workers, development reload, replicas, or rolling overlap against the same data.
- The supervisor owns the entire application process tree. A service restart may interrupt training; recover from the last valid checkpoint when available. Continuous training through a web restart requires the separate execution boundary described in section 5.
- Linux/systemd is the first supported server installation. Docker is an additional packaging option after the same lifecycle behavior passes its tests. Keep the existing Windows desktop workflow working.

```text
Browser
   |
   | HTTPS + authentication
   v
Reverse proxy
   |
   | private/loopback upstream
   v
Workbench: one web process + supervised jobs/runners
   |                 |                    |
   v                 v                    v
Projects/state    Cache/weights       MongoDB (optional)
```

No Node.js build, Redis, Kubernetes, or new database for annotations is necessary for this scope. Retain the file-based datasets and existing training queue.

## 2. What the current implementation already provides

| Area | Verified current behavior | Remaining work |
| --- | --- | --- |
| Headless launch | `cli.py` supports `--no-browser`, `--host`, and `--port`. Missing projects directories trigger `typer.confirm()`. | Deterministic noninteractive startup, validated configuration, installation as a service. |
| HTTP application | `web.py` serves FastAPI and static ES modules from the Python package. | Production app factory, lifecycle hooks, health endpoints, logging configuration. |
| Request guards | CLI enables a Host check and `X-Workbench` on mutations. Wildcard binds disable the Host check; factories default to guards off. | Authenticated access and consistent server-mode security regardless of entry point. The custom header is not authentication. |
| Training | `TrainingManager` persists `queue.json`, records PID/create-time, reattaches to live runners, and marks missing runners `interrupted`. Restart tests already exist. | Real service-manager behavior, durable launch reconciliation, queue drain, and shutdown ownership. |
| Startup side effects | `TrainingManager.__init__()` calls `tick()`, which can launch queued work. Only the CLI starts the periodic supervisor. | Move startup work behind configuration validation, exclusive ownership, recovery, and app lifespan. |
| Shutdown | `TrainingManager.shutdown()` sets an event without joining. Other managers run daemon threads. `Workspace.close_active()` refreshes the summary but does not explicitly dispose of resources. | Cancel/join/close APIs and a bounded shutdown sequence. |
| Writes | Labels use temporary files, `fsync`, replacement, revision/mtime checks, backups, and undo transactions. | Multi-file operations are not crash-atomic merely because individual writes are atomic; add interruption recovery. |
| Uploads | Upload paths and extensions are checked. The endpoint reads the entire file with `request.body()`. | Streaming, quotas, concurrency limits, and partial-upload cleanup. |
| Staging cleanup | `_clean_staging()` already removes old directories when a project is opened. | Periodic cleanup for long-open projects and single-dataset mode, with active-upload protection. |
| State lifetime | Undo/redo and pending predictions are in memory; `JobManager` retains at most 50 finished jobs but has no active-job limit or durable job registry. | Bound live memory and active work; recover interrupted operations explicitly. |
| Storage | Projects mode stores the shared queue under `<projects>/.training/`; projects contain labels, review/QA state, trash, backups, and runs. Media cache is bounded and honors `XDG_CACHE_HOME`. | Explicit state/cache locations, retention, disk monitoring, and complete backups. |
| Collaboration | One active project per `Workspace`, one undo stack per dataset. | Keep this constraint explicit; concurrent multi-user behavior is a separate project. |

This builds on `SERVER_PLAN.md`, with several corrections: staging cleanup and interrupted training handling already exist; readiness should not fail merely because a dataset is indexing; changing a backup session ID alone is unsafe while undo still references old trash; and detached subprocesses do not by themselves survive a service stop.

## 3. Configuration and process ownership

Primary files: `cli.py`, `web.py`, `workspace.py`, `training.py`; proposed `config.py`, `server.py`, and an instance-lock helper.

- [ ] Add a validated server configuration object shared by the CLI and application factory. Precedence: explicit CLI option > `WORKBENCH_*` environment variable > config file > default. Explicitly set CLI values must be distinguishable from parser defaults.
- [ ] Add a server entry point, for example `yolo-workbench-server --config /etc/yolo-workbench/config.toml`. This is a proposed command, not an existing one. Keep the current desktop command compatible.
- [ ] Server mode never opens a browser or prompts. Require an existing data root by default; allow explicit initialization/creation. A missing mount must fail startup rather than silently create an empty workspace on the root disk.
- [ ] Resolve configured paths independently of the current working directory, preferably relative to the config file. Validate mode, category, permissions, directories, bind address, auth boundary, and limits before launching any background work.
- [ ] Acquire an OS-held exclusive lock for the canonical workspace/dataset before constructing managers with side effects. Refuse a second owner even if it uses another port. A PID text file alone is insufficient; stale files must not block recovery after a crash.
- [ ] Enforce one worker and no reload in the supported service entry point. Document that existing `RLock`s and mtime checks do not coordinate independent server processes. Reject duplicate instances sharing state even if launched through another supported entry point.
- [ ] Use local storage with verified lock/rename behavior for the first release. Shared network filesystems and multiple hosts need a separate consistency design.

Required configuration groups:

| Group | Settings |
| --- | --- |
| Dataset | Mode, projects root or dataset path/category, initialization policy. |
| Network | Bind host/port, allowed HTTP hosts, trusted proxy addresses, authentication mode. |
| Storage | Cache/weights/temp directories, allowed import roots, trusted model roots, minimum free space. |
| Limits | Upload bytes/files/aggregate quota, active jobs, queue length, undo count/bytes, cache budgets, retention. |
| ML | Enabled features, inference/training devices, model download policy, dataloader/CPU limits. |
| Operations | Log level/format, shutdown deadline, maintenance/drain state, cleanup interval. |

**Done when:** a fresh installation starts with closed stdin and an arbitrary working directory; invalid configuration exits with a useful error; a second instance cannot modify or schedule work in the same workspace.

## 4. Network access and filesystem boundaries

Primary files: `web.py`, `cli.py`, `importer.py`, `predictor.py`, `frameworks.py`, `workspace.py`, `dataset.py`; proposed `path_policy.py` and proxy examples.

### Authentication and proxy configuration

- [ ] For the first release, provide a complete authenticated reverse-proxy recipe, using an existing identity gateway or password authentication over TLS. A new in-app user database is not required for the single-operator deployment.
- [ ] Keep the upstream reachable only through loopback or a private container network. Protect pages, API routes, images/crops, project covers, model downloads, and `/docs`/`openapi.json`; either protect or disable API documentation in server mode.
- [ ] Require a configured Host allow-list independently of the bind address. Binding `0.0.0.0` inside a container must not silently disable it.
- [ ] Configure trusted forwarded-header sources explicitly. Forwarded transport headers do not authenticate a user. If identity headers are consumed, the proxy must remove incoming copies and set its own; verify the direct peer at the correct middleware boundary, before trusting rewritten client addresses.
- [ ] Retain the mutation header check, test same-origin/Origin handling with the selected authentication mode, and avoid permissive credentialed CORS. Apply suitable content-type, framing, and referrer headers; test a CSP against actual frontend behavior before enforcing it.
- [ ] Refuse accidental unauthenticated network exposure in server mode. A documented private tunnel deployment can be an alternative authenticated boundary.

Expose and configure Uvicorn's proxy trust, concurrency, and graceful-shutdown controls through the service configuration rather than relying on implicit environment defaults. See the [Uvicorn settings reference](https://uvicorn.dev/settings/).

### Paths and trusted model execution

- [ ] Centralize read/write path policy. Import sources, preview sample paths, model discovery/loading, training arguments, exports, and restore/delete operations must enforce the intended roots.
- [ ] Check resolved files, not just their parent input directory. Include symlinks, dataset YAML `path`/split entries, text-file image lists, adopted project metadata, and persisted recent-model paths. Separate read-only import/model roots from writable dataset/state roots.
- [ ] Remove cwd/recent-directory model scanning outside configured roots in server mode. Revalidate persisted paths at use time and when a queued run starts.
- [ ] Review the full training argument surface, including configuration/model paths, downloads, and external integrations. Use a server allow-list where the framework exposes capabilities unsuitable for remote callers; do not validate only the top-level `model` argument.
- [ ] Treat framework checkpoint loading as execution of trusted code. Keep models operator-provisioned or from a controlled pretrained catalog. Do not introduce unrestricted model uploads as part of this work. The current dataset upload allow-list already excludes weight files.
- [ ] Run under a dedicated account with read-only application code and narrowly writable state. OS permissions remain necessary because application path checks do not isolate code loaded from a model.

**Done when:** unauthenticated clients cannot access data, forged proxy/Host headers do not bypass checks, and out-of-root paths—including nested symlinks and YAML references—are rejected in every relevant route.

## 5. Lifecycle, training, and interruption recovery

Primary files: `web.py`, `cli.py`, `session.py`, `dataset.py`, `jobs.py`, `training.py`, `train_runner.py`, `export_runner.py`, `rfdetr_runner.py`, and the inference/embedding managers.

### Application lifecycle

- [ ] Install a shared FastAPI lifespan for both modes. Keep constructors passive: loading queue metadata must not start training. Start reconciliation and scheduling only after configuration and instance ownership succeed. [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/) provides the application startup/shutdown boundary.
- [ ] Retain thread/process handles. Add idempotent stop/join/close operations for the training supervisor, jobs, image probing, SAM loading/encoding, predictions, QA, and embeddings. Dispose of listeners/models when a project closes, rather than relying on garbage collection.
- [ ] Add durable maintenance/drain state: refuse new mutations/jobs and stop dequeuing training while allowing status reads. It must remain set across a restart used for maintenance.
- [ ] Shutdown order: stop admitting new work; stop queue launches; finish or cancel mutations safely; stop/checkpoint managed runners; join jobs within a deadline; flush indexes and summaries; release models/connections; release the ownership lock last.
- [ ] Avoid waiting for joins while holding locks the worker needs. Python threads cannot be safely force-killed; work that cannot cooperate with the deadline should run in a managed subprocess or be explicitly disabled in the supported server profile until it can.

### Default training policy: lifecycle tied to the service

The current detached runner is useful for desktop restarts, but it remains part of its service's control group. systemd's default `KillMode=control-group` stops that group; `KillMode=process` is explicitly discouraged by systemd because children can outlive the manager's ownership. Use managed termination, not this shortcut. See [systemd process termination](https://www.freedesktop.org/software/systemd/man/latest/systemd.kill.html).

- [ ] On planned maintenance, drain the queue and ask the active run to pause at a checkpoint before stopping the service. Pause can take an epoch; show progress and allow an explicit deadline.
- [ ] On ordinary termination, prevent further launches, request a cooperative stop, and wait within the service deadline. Configure signal delivery so the web process has time to coordinate runners, then terminate the remaining process tree on timeout. Evaluate `KillMode=mixed` for this arrangement and test it with both frameworks.
- [ ] On hard kill, OOM, or reboot, classify the run as interrupted and inspect checkpoint validity. Reuse existing interrupted/resumable handling. Do not claim that the last batch is saved or that a run without a checkpoint is resumable.
- [ ] Default to explicit resume of interrupted training; retain queued runs, but hold the queue during unresolved recovery/maintenance so another run does not silently consume the GPU. Persist this policy.
- [ ] Make launch recovery durable: write a launch intent/run identity before spawning, have the runner acknowledge it, and reconcile process identity plus run state on restart. Today a crash between `Popen`, `process.json`, and queue persistence can leave an untracked runner or cause a duplicate launch.
- [ ] Preserve PID/create-time checks, add run/launch identity verification, and fail closed when ownership is ambiguous. Recovery must restore dataset write guards before accepting edits or starting the next run.
- [ ] Give ONNX exports equivalent ownership/recovery. Export subprocesses are detached too, but their job IDs and `_exports` mapping currently live only in memory. Prevent duplicate exports and deletion of a run still being exported after restart.

**Optional stronger guarantee:** if training must continue while the web service restarts, move execution to separately supervised runner units or a persistent worker service. Give that owner the execution lock, durable commands, GPU reservation, runner reconciliation, and dataset-lock state. Define whether queued runs advance while the web process is absent. Do not let two schedulers own the queue or give the web service unrestricted service-manager privileges. This is extra architecture, not a prerequisite for the default service above.

### Durable operations and data recovery

- [ ] Persist minimal job records for imports/exports and other long operations: stable ID, type, project, state, progress/checkpoint, owned temporary paths, and timestamps. Atomic files are sufficient initially; no queue broker is needed.
- [ ] Use a shared durable-write helper for critical metadata and journals: unique temporary file, flush/fsync, atomic replacement, and parent-directory synchronization where supported. Audit queue, project, review, QA, and recovery state as well as labels; rename alone does not establish the promised power-loss durability. Version persistent formats for upgrades.
- [ ] At startup, reconcile stale `running` jobs with actual processes and outputs. Mark interrupted jobs, rescan affected datasets, and make retry behavior explicit. Pending predictions and undo may intentionally reset, but report that behavior to the UI.
- [ ] Make imports safe to interrupt: stage each image/label pair, publish completed files atomically, and keep a durable manifest of committed items. Recover/retry without duplicate files or silently discarding valid partial results.
- [ ] Add a recovery journal for multi-file edits/moves and class-schema changes, or another demonstrated rollback/replay mechanism. Record intent and recovery material before mutation. A crash halfway through relabeling must not leave labels silently interpreted under a mismatched schema.
- [ ] Distinguish corrupt critical state from a missing cache. Preserve and report corrupt queue/project/job metadata instead of silently treating it as an empty queue. Keep affected data read-only until recovery succeeds.

**Done when:** graceful stop, forced kill, launch-time failure, and reboot leave every job classified, no duplicate runners, no orphaned exports, and recoverable dataset state.

## 6. Resource limits for an unattended process

Primary files: `web.py`, `importer.py`, `jobs.py`, `dataset.py`, `storage.py`, `workspace.py`, `media_cache.py`, and ML managers.

- [ ] Stream uploads to temporary files with per-file, per-upload, total staging, file-count, and concurrent-upload limits. Count actual received bytes even without `Content-Length`; return 413 for oversized input and clean incomplete files on disconnect/error. Disk writes must not block the async event loop.
- [ ] Bound expensive requests and background jobs. Add active-job and training-queue limits; serialize conflicting imports/mutations and expensive GPU work. Cap decoded image dimensions, preview sizes, and CPU/thread use as well as compressed upload bytes.
- [ ] Check free space before uploads, imports, exports, and training snapshots/checkpoints; handle disk-full errors during writes too. Size capacity for original data, retained backups/trash, staging, RF-DETR temporary data, checkpoints, and exports.
- [ ] Extend existing staging cleanup into periodic maintenance for both app modes. Track active uploads/imports so age-based cleanup cannot remove files still in use.
- [ ] Bound undo/redo by both entry count and retained bytes. Tie retention to references: never purge trash/backups still needed by undo, redo, active imports, or recovery journals. Rotate sessions only at a safe history boundary, resetting backup bookkeeping and informing the UI when history expires.
- [ ] Add explicit retention policies and usage reporting for dataset backups/trash, workspace `.trash`, staging, logs, and caches. Model weights and completed runs need a separate operator-controlled policy; do not automatically delete training results.
- [ ] Reuse the media-cache budget; inspect other persistent dictionaries/caches and pending prediction collections for bounds. Limit prediction batches or evict/persist results when their budget is reached.
- [ ] Add configurable inference/training device assignment. On a shared GPU, reserve it across training, SAM, prediction, QA, and embeddings; unload idle models before granting it to training. Reject or queue conflicting work with a clear reason.
- [ ] Serialize unload with inference/model loading. Release model references before clearing framework caches; a cache-clear call alone does not free live tensors.
- [ ] Bound or rotate application and per-run logs, preserve useful error context, and keep the polling log endpoint correct after rotation.

**Done when:** the service survives repeated imports/project switches/model loads and a 24–72 hour workload without unbounded memory/disk growth; quotas fail cleanly without corrupting data or stalling health checks.

## 7. Packaging and dependency behavior

Primary files: `pyproject.toml`, `uv.lock`, `README.md`; proposed `deploy/`, `docs/server.md`, and CI configuration.

### Reproducible runtime

- [ ] Select and test a Linux/Python baseline, initially Python 3.12 if the locked dependencies support it. Preserve the declared desktop Python compatibility or document any deliberate change. TOML configuration on Python 3.10 requires a compatible parser dependency.
- [ ] Build/install a versioned artifact using the lockfile; keep package resolution and installation out of service startup. Move `pytest` and `httpx` to a development group and declare directly imported runtime dependencies such as `psutil` explicitly.
- [ ] Test CPU and NVIDIA installations with the actual locked torch/torchvision wheels and host driver. Record a tested compatibility matrix instead of assuming CUDA compatibility from the developer workstation. Retain the existing headless OpenCV resolution rule.
- [ ] Route all writable ML configuration/caches to explicit locations, including workbench media/weights, Ultralytics settings, torch, RF-DETR, FiftyOne model downloads, and temporary files. Audit environment variables against installed versions; some paths are captured during import, so configure them before imports.
- [ ] Add a provisioning/diagnostic command to validate directories, disk, feature dependencies, GPU availability, MongoDB when enabled, and requested model weights without starting a training run.
- [ ] Provide explicit model prefetching and an offline mode. Inventory SAM, pretrained YOLO/RF-DETR, embedding models, and auxiliary framework downloads. Disable runtime package installation for the pinned framework versions; fail with an actionable message if a dependency is missing.
- [ ] Make embeddings an explicit feature setting. Configure its MongoDB URI through environment/config, keep MongoDB private, and test absence/outage behavior. Core editing must remain usable when embeddings are disabled or unavailable.

### systemd deliverables

- [ ] `deploy/yolo-workbench.service`: foreground process, dedicated account, absolute executable/config paths, startup after required mounts, boot enablement, restart on failure with backoff/rate limits, journald output, and the tested shutdown policy from section 5.
- [ ] Apply `NoNewPrivileges`, read-only application/system files, and specific writable paths. Test hardening with GPU devices and ML subprocesses rather than copying options that block required devices.
- [ ] `deploy/config.example.toml` and an environment-file example, with no embedded credentials.
- [ ] One complete proxy example with TLS/auth, matching Host configuration, forwarded-header trust, upload limits, and suitable timeouts for inference/downloads. Document which port is public and firewall the upstream.

Recommended filesystem layout:

| Path | Purpose |
| --- | --- |
| `/opt/yolo-workbench/releases/<version>/` | Immutable code and virtual environment, including runner code. |
| `/etc/yolo-workbench/` | Config and restricted secret references/files. |
| `/srv/yolo-workbench/projects/` | Persistent datasets, `.training/`, `.trash/`, project state, models, and runs. |
| `/var/cache/yolo-workbench/` | Rebuildable media, downloaded weights, and library caches. |
| `/run/yolo-workbench/` | Ephemeral service runtime files; the workspace lock must still identify the canonical data root. |

### Docker deliverables, after systemd acceptance

- [ ] Add a pinned non-root image, `.dockerignore`, and Compose example with persistent project/cache mounts, an optional private MongoDB service, restart policy, health check, and separate CPU/GPU configuration.
- [ ] Use an entry point that forwards termination signals, appropriate child reaping, writable temp space, and sufficient shared memory for dataloaders. Set the stop grace period to match application shutdown. Docker stop sends a termination signal and later force-kills on timeout; verify the actual image's behavior. [Docker stop reference](https://docs.docker.com/reference/cli/docker/container/stop/).
- [ ] Keep runners in the same lifecycle for the default image. A separate worker/container is required for the optional training-continuity guarantee. Persisting volumes does not preserve running processes.
- [ ] Test host UID/GID permissions and container recreation. Do not bake datasets, secrets, or the development `.venv` into the image.

**Done when:** a clean Linux host can install a pinned release and start it at boot from documented steps, with no interactive session, surprise dependency installation, or writes into the application directory.

## 8. Health, operations, backups, and upgrades

- [ ] Add cheap liveness (`/healthz`) and readiness (`/readyz`) endpoints. Liveness checks HTTP/event-loop responsiveness; readiness reflects completed startup/recovery, ownership, usable required storage, and maintenance state. Keep unauthenticated responses minimal and details protected.
- [ ] Do not make readiness depend on finishing image indexing, having an idle GPU, or optional MongoDB. Report feature degradation separately so a slow optional service does not make annotation disappear or trigger restart loops.
- [ ] Configure application/Uvicorn logging together: timestamps with timezone, severity, request ID, project/job/run ID, exceptions, and supervisor/job state transitions. Redact credentials and avoid logging upload bodies.
- [ ] Surface low disk space, failed cleanup, repeated job failures, stalled supervision, and corrupt state. Include a simple external health/alert recipe; a full metrics backend/admin dashboard is optional.
- [ ] Back up the entire projects root, including hidden `.training`, `.trash`, `.yolo-workbench`, review/QA metadata, models, run specs, and checkpoints, plus configuration and separately protected secrets. Include external data referenced by single-dataset/adopted datasets or disallow such references under the chosen path policy.
- [ ] Use a drained application or a consistent snapshot for backup. Document treatment of active checkpoint writes and temporary files. Ordinary copying during a class rewrite is not a consistent backup. Local edit backups on the same disk are not disaster recovery.
- [ ] Decide whether MongoDB contains anything that must be retained; current embeddings use temporary FiftyOne datasets and in-memory visualization results, so recomputation may be sufficient. Do not accidentally omit future persistent DB state.
- [ ] Write and exercise restore instructions on a fresh host. Discard stale process identities, hold scheduling until reconciliation, validate project manifests/paths, and check image editing plus checkpoint loading.
- [ ] Upgrade procedure: enter durable drain; wait/pause jobs; take a consistent backup; install into a new version directory; switch the service; verify readiness and a smoke workflow; resume the queue deliberately. Keep the old release and define metadata-format rollback compatibility.
- [ ] Never replace a virtual environment underneath running training/export processes. If separate workers are later introduced, retain each worker's matching code/runtime until its runs finish.

**Done when:** an operator can diagnose a failure, restore onto a fresh host, upgrade, and roll back using `docs/server.md` without reconstructing the application internals.

## 9. Verification and rollout sequence

Use isolated temporary datasets, as the existing tests do. This plan itself does not require running model downloads or training on real user data.

| Milestone | Deliverables | Required acceptance |
| --- | --- | --- |
| M1: service foundation | Config, server entry point, passive constructors, instance lock, lifespan skeleton. | Noninteractive launch; duplicate-owner refusal; both app modes start/stop through lifespan; desktop command remains compatible. |
| M2: access boundary | Authenticated proxy, Host/proxy settings, filesystem policy. | All data routes protected; forged headers, traversal, symlinks, persisted-path escapes, and unsafe training arguments rejected. |
| M3: lifecycle and recovery | Drain, bounded cancellation/join, durable runner launch/job state, import and transaction recovery. | Forced termination at launch/copy/schema-edit/export boundaries recovers without duplicate execution or silent dataset inconsistency. |
| M4: resource bounds | Streaming uploads, quotas, safe retention, bounded history/jobs, GPU arbitration. | Oversize/chunked uploads, client disconnects, disk full, concurrent jobs, and long-lived history handled within configured budgets. |
| M5: deployment package | Locked Linux install, feature/device preflight, systemd/proxy/config examples, operational docs. | Clean host install, logout, boot, crash restart, planned stop, and cold restore work under the real supervisor. |
| M6: unattended acceptance | Logs/alerts, tested backup/upgrade/rollback, sustained workload. | 24–72 hour soak plus reboot and recovery exercises; no unexplained growth, orphan processes, or data loss. |
| M7: optional packaging/features | Docker, sub-path hosting, separate worker, team collaboration. | Each has its own integration tests and documented guarantees before being advertised. |

Dependencies: M1 precedes lifecycle/recovery work; M2 precedes access by network clients; M3 must exist before automated cleanup and the final supervisor configuration; M1–M5 must pass before M6. First unattended production deployment requires M1–M6. M7 is not a blocker.

Testing work to add:

- Unit/API tests for config precedence, path policy, upload quotas, state recovery, retention references, and job/GPU admission. Reuse injected fake launchers/backends for most coverage.
- Explicit lifespan tests. Existing tests use `httpx.ASGITransport`; do not assume transport construction runs startup/shutdown. Enter the application's lifespan in the test harness and assert threads/resources are closed.
- Real process tests for ownership locking, shutdown deadlines, PID reuse, and crashes between launch intent, spawn, acknowledgement, and queue persistence.
- Linux integration tests under systemd and, if shipped, Docker. Fake-runner restart tests do not prove cgroup/signal behavior. Exercise an ordinary web crash, whole-service restart, forced kill, and host reboot separately.
- Tiny opt-in training and resume smoke tests for both Ultralytics and RF-DETR, plus ONNX export interruption. Test CPU behavior and the supported NVIDIA deployment on actual hardware.
- Browser checks through the authenticated proxy: project list, editing/conflict handling, upload, images, training progress, downloads, logout/session expiry, and degraded embeddings. Keep health responsive under load.
- Backup/restore and upgrade/rollback drills, including hidden workspace state and interrupted runs. Verify the documented loss boundary: only uncommitted work/current checkpoint progress may be lost, not previously committed dataset state.

## 10. Explicitly deferred scope

These improve the product but are not necessary to leave the single-operator service running:

- Concurrent independent users: a session pool, per-user/conflict-aware undo, attribution, project permissions, shared-prediction semantics, and stale-view updates. Existing annotation revision checks should be preserved; they do not by themselves make every operation multi-user safe.
- Built-in accounts/login when a complete external authentication boundary is sufficient.
- Hosting under `/workbench/`: frontend HTML/JS currently has root-absolute asset, API, and navigation URLs. Supporting a prefix requires a shared frontend URL builder and end-to-end testing; Uvicorn `root_path` alone is insufficient.
- Uninterrupted training during web upgrades, multi-GPU scheduling across hosts, high availability, multiple web workers, and distributed storage.
- An admin dashboard, Prometheus integration, automatic filesystem watching, and optional dependency profiles that substantially change packaging.

The first implementation should begin with M1 and an executable lifecycle test, then make restart/recovery behavior concrete before publishing service installation instructions.
