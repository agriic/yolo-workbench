# Running the workbench on a server

`yolo-workbench-server` runs the workbench as an unattended service in projects mode: it starts at boot
under systemd, needs no terminal or logged-in user, and is used from other machines' browsers at
`http://<server>:8765`. The desktop command (`yolo-workbench`) is unchanged.

This guide assumes one Linux host with local storage (and usually one NVIDIA GPU) on a trusted internal
network. Everyone signs in with an account made on the server (`yolo-workbench-server user add`), but over
plain HTTP: passwords and session cookies cross the network unencrypted, every signed-in user can use every
project, and model files (`.pt`/`.pth`) run code when loaded. Keep the port reachable only from the network
that should use it.

## What the server does differently

| | Desktop (`yolo-workbench`) | Server (`yolo-workbench-server run`) |
| --- | --- | --- |
| Start | opens a browser, may ask before creating the projects folder | never prompts; a missing projects folder stops it (unless `create_projects_root`) |
| Network | `127.0.0.1` | `0.0.0.0` with a required Host allow-list (`allowed_hosts`) |
| Sign-in | none | everyone signs in (`[auth]`); changes, reviews and jobs are attributed |
| Open projects | several (up to 4) | `max_open_projects`, idle ones close after `project_idle_close_min` |
| Several people in one project | one undo history | each person undoes only their own changes; others' changes, and who has an image open, show up within seconds |
| Import from a server folder by path | yes | off: images arrive by browser upload |
| Stop with a run training | the run keeps going (it outlives the desktop app) | the run pauses at its next checkpoint and resumes at the next start |
| Maintenance | none | idle models unload, daily backup/trash sessions, retention, abandoned uploads removed, disk checks |
| Logs | console | journald (text or JSON lines) |

Both refuse to run twice on the same folder: a second server, or a desktop session on the server's projects
folder, exits with "already in use by process …" (`<projects>/.workbench/owner.lock`).

## Install

Requirements: Linux with systemd, an NVIDIA driver recent enough for CUDA 13.0 (the PyPI torch wheels bundle
CUDA; check with `nvidia-smi`), internet access (Python packages, and model weights on first use), and
[uv](https://docs.astral.sh/uv/) on root's PATH:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin sh
```

From a checkout of this repository on the server:

```bash
sudo bash deploy/install.sh
```

It creates the `yolo-workbench` system account, `/srv/yolo-workbench/projects` (data), `/var/cache/yolo-workbench`
(caches), installs the code and its virtual environment into `/opt/yolo-workbench/releases/<version>/` with
`/opt/yolo-workbench/current` pointing at it, and installs `/etc/yolo-workbench/config.toml` (first time only)
and the systemd unit. Then:

1. Edit `/etc/yolo-workbench/config.toml`: at least `allowed_hosts` — the names and addresses people type in
   the browser (e.g. `["gpu-server", "gpu-server.example.local", "10.0.0.20"]`).
2. Create your account (and one for each colleague; see [Users and sign-in](#users-and-sign-in)):
   ```bash
   sudo -u yolo-workbench /opt/yolo-workbench/current/.venv/bin/yolo-workbench-server user add alice --admin --display-name "Alice Smith" --config /etc/yolo-workbench/config.toml
   ```
3. Check everything without starting the server:
   ```bash
   sudo -u yolo-workbench /opt/yolo-workbench/current/.venv/bin/yolo-workbench-server check --config /etc/yolo-workbench/config.toml
   ```
   It reports the configuration, the projects folder (writable, free space, not in use), the accounts, the
   cache folder, what it listens on, the GPU torch sees, Ultralytics/RF-DETR, and MongoDB when embeddings
   are enabled. It exits non-zero on a failure.
4. Start it now and at every boot:
   ```bash
   sudo systemctl enable --now yolo-workbench
   ```
5. Open the port for the people who use it (and nothing else), e.g. `sudo ufw allow from 10.0.0.0/24 to any port 8765 proto tcp`.

To bring existing datasets in, copy their folders into `/srv/yolo-workbench/projects/` (owned by
`yolo-workbench`) and use **Adopt** on the projects page, or create a project and upload the images.

### Layout

| Path | Contents |
| --- | --- |
| `/opt/yolo-workbench/releases/<version>/` | code and virtual environment of one version (read-only) |
| `/opt/yolo-workbench/current` | the active version |
| `/opt/yolo-workbench/python/` | the Python uv installed for the releases |
| `/etc/yolo-workbench/config.toml` | configuration |
| `/etc/yolo-workbench/yolo-workbench.env` | environment for the unit (optional overrides, proxy, `CUDA_VISIBLE_DEVICES`) |
| `/srv/yolo-workbench/projects/` | every project, `.training/` (the queue), `.trash/` (deleted projects), `.workbench/` (owner lock, `users.json` accounts, `sessions.json` sign-ins) |
| `/var/cache/yolo-workbench/` | thumbnails, downloaded weights, torch/Ultralytics/matplotlib caches — rebuildable |

## Configuration

`deploy/config.example.toml` lists every setting with its default and an explanation. Precedence:
CLI option (`--projects`, `--host`, `--port`) > environment variable > config file > default. The environment
variable is `WORKBENCH_` plus the name in the first column (`WORKBENCH_PORT=9000`, lists comma-separated:
`WORKBENCH_ALLOWED_HOSTS=a,b`). Unknown keys in the file are errors.

| Name (environment) | File | Default | |
| --- | --- | --- | --- |
| `projects_root` | `[server]` | — (required) | the projects folder |
| `create_projects_root` | `[server]` | `false` | create it when missing |
| `host`, `port` | `[server]` | `0.0.0.0`, `8765` | where to listen |
| `allowed_hosts` | `[server]` | — (required with `0.0.0.0`) | accepted `Host` names; `["*"]` = any |
| `shutdown_deadline_s` | `[server]` | `60` | wait for jobs and projects on stop |
| `cache_dir` | `[storage]` | `~/.cache` | caches (media, weights, libraries) |
| `media_cache_mb` | `[storage]` | `512` | thumbnail cache size |
| `min_free_disk_gb` | `[storage]` | `5` | refuse writes below this; warn on every page |
| `trash_retention_days`, `backup_retention_days` | `[storage]` | `30`, `30` | dataset trash / label backups of earlier sessions (0 = keep) |
| `project_trash_retention_days` | `[storage]` | `30` | deleted projects (0 = keep) |
| `session_rotate_hours` | `[storage]` | `24` | new backup/trash session while a project stays open |
| `staging_max_age_hours` | `[storage]` | `24` | abandoned uploads |
| `maintenance_interval_min` | `[storage]` | `60` | cleanup and retention interval |
| `max_upload_file_mb`, `max_upload_files`, `max_upload_total_mb` | `[limits]` | `1024`, `100000`, `102400` | per file / per import |
| `max_model_upload_mb` | `[limits]` | `4096` | uploaded model files |
| `undo_max_transactions`, `undo_max_mb` | `[limits]` | `200`, `256` | undo history per open project |
| `training_pause_deadline_s` | `[training] pause_deadline_s` | `600` | wait for the active run's checkpoint on stop |
| `resume_after_service_stop` | `[training]` | `true` | resume it at the next start |
| `hold_queue_after_crash` | `[training]` | `false` | hold the queue when a run dies |
| `idle_unload_minutes` | `[models]` | `2` | unload SAM / prediction models (0 = never) |
| `gpu_models_during_training` | `[models]` | `allow` | `allow` (with a warning) or `refuse` |
| `max_loaded_models` | `[models]` | `2` | prediction models across open projects |
| `max_open_projects` | `[projects]` | `4` | projects open at once |
| `project_idle_close_min` | `[projects]` | `60` | close idle projects (0 = never; closing ends their undo history) |
| `auth_enabled` | `[auth] enabled` | `true` | sign-in; off: anyone who reaches the port can use everything, unattributed |
| `session_days` | `[auth]` | `30` | a sign-in stays valid this long after it was last used |
| `embeddings_enabled` | `[embeddings] enabled` | `true` | the Embeddings tab |
| `mongodb_uri` | `[embeddings]` | — | MongoDB for FiftyOne |
| `log_level`, `log_format` | `[logging] level`, `format` | `info`, `text` | `json` for one object per line |
| `access_log` | `[logging]` | `false` | one line per HTTP request |

Configuration changes take effect at the next `systemctl restart yolo-workbench`.

## Operating it

```bash
systemctl status yolo-workbench          # running? since when?
journalctl -u yolo-workbench -f          # follow the log
sudo systemctl restart yolo-workbench    # after a configuration change or an upgrade
curl -s localhost:8765/readyz            # 200 when ready; /healthz answers as long as the process runs
```

`/healthz` (liveness) and `/readyz` (startup finished, projects folder writable, owner lock held) need no
Host name from `allowed_hosts`, so a monitoring system can poll them by IP. Readiness does not wait for image
indexing, the GPU, or MongoDB.

### Training and restarts

One run trains at a time; the queue is shared by all projects. On `systemctl stop` or `restart`:

1. nothing new starts;
2. the active run is asked to pause at the end of its current epoch (Ultralytics saves `last.pt`, RF-DETR `last.ckpt`);
3. imports, exports and prediction batches are cancelled, open projects save their state;
4. the server waits up to `pause_deadline_s` for the run to pause, then exits.

If the epoch takes longer, the server exits anyway and systemd kills what is left of the run right away
(`KillMode=mixed`); the run is **interrupted**: it resumes from the previous epoch's checkpoint, losing the
epoch in progress. `TimeoutStopSec` (720 s in the unit) only bounds the server itself. Either way,
with `resume_after_service_stop` the run goes back to the front of the queue at the next start. A run
interrupted by a crash, a power loss or the OOM killer is shown as interrupted and waits for **Resume** in the
Training tab (or holds the whole queue, with `hold_queue_after_crash`). A run without any checkpoint yet
cannot resume and must start again.

For long epochs, raise `pause_deadline_s` and `TimeoutStopSec` together
(`sudo systemctl edit yolo-workbench` → `[Service]` `TimeoutStopSec=1800`).

### Maintenance: holding the queue

The Training tab's queue panel has **Hold** (nothing new starts, also across restarts) and **Hold & pause**
(also pauses the active run at its next checkpoint; it resumes by itself when the queue is released). The
header shows "Training queue held" while it is held. Use **Hold & pause** before maintenance that needs the
GPU or a reboot, wait for the run to show *paused*, do the work, then **Release**.

The same through the API, signed in with an account (the cookie file keeps the session):

```bash
curl -c /tmp/wb -H 'X-Workbench: 1' -H 'Content-Type: application/json' -d '{"name": "alice", "password": "…"}' localhost:8765/api/v1/auth/login
curl -b /tmp/wb -X POST -H 'X-Workbench: 1' -H 'Content-Type: application/json' -d '{"held": true, "pause_active": true}' localhost:8765/api/v1/training/queue/hold
```

### GPU

SAM (smart annotation) is one model for everyone; each project has its own prediction model, shared by everyone
in it. Both unload after `idle_unload_minutes` (2) during which nobody used them, and load again by themselves
on the next use (the first click or prediction then takes a few seconds; a prediction model keeps its class
mapping and pending predictions). A training run that starts unloads only its own project's prediction model
(that project is read-only while it trains; load it again by hand if needed); SAM and other projects' models
stay. At most `max_loaded_models` prediction models stay loaded across projects. Loading one while a run trains
shows a warning (it shares the GPU's memory); `gpu_models_during_training = "refuse"` refuses it instead, and a
starting run then also unloads SAM and every project's prediction model. To keep the server off a GPU, set `CUDA_VISIBLE_DEVICES` in
`/etc/yolo-workbench/yolo-workbench.env` (it applies to the training runs too).

### Data, uploads and models

Images come in by browser upload (Dataset tab → Import); a server folder cannot be imported by path. Uploads
stream to disk; `max_upload_*` limit them (HTTP 413) and `min_free_disk_gb` stops them before the disk fills
(HTTP 507, with a warning on every page). **Upload…** next to the model field (Annotation → Prediction) puts
a `.pt`, `.pth` or `.onnx` file into the project's `models/` folder and loads it; trained runs' best weights
are listed without uploading.

Deleted images and the original of every edited label file are kept per *session* in
`<project>/.yolo-workbench/trash|backups/<session>/`. A new session starts daily while a project stays open;
earlier sessions are deleted after `trash_retention_days` / `backup_retention_days`, except trash the undo
history can still put back. Deleted projects stay in `<projects>/.trash/` for `project_trash_retention_days`.
Training runs and models are never deleted automatically.

### Users and sign-in

Every page and API request needs a signed-in session, except the login page, `/healthz` and `/readyz`.
Accounts live in `<projects>/.workbench/users.json` (scrypt password hashes) and are managed on the server,
as the service account (or as root: the files keep the projects folder's owner):

```bash
WB="sudo -u yolo-workbench /opt/yolo-workbench/current/.venv/bin/yolo-workbench-server"
$WB user add bob --display-name "Bob Jones" --config /etc/yolo-workbench/config.toml   # asks for the password twice
$WB user list --config /etc/yolo-workbench/config.toml
$WB user passwd bob --config /etc/yolo-workbench/config.toml      # a forgotten password; ends bob's sign-ins
$WB user admin bob [--revoke] --config /etc/yolo-workbench/config.toml
$WB user remove bob --config /etc/yolo-workbench/config.toml      # his changes stay attributed to "bob"
```

`--password-stdin` reads the password from the first line of stdin instead (for scripts). Changes take
effect at once; the running server re-reads the file. Passwords need at least 8 characters. Everyone can
change their own password from the menu under their initials (top right); that ends their other sign-ins.

A sign-in lasts `session_days` after it was last used and survives restarts (`.workbench/sessions.json`
keeps only digests of the session tokens; deleting the file while the service is stopped signs everybody
out). The cookie is `HttpOnly` and `SameSite=Lax` but not `Secure`, as the server speaks plain HTTP.

All accounts can do everything. The `admin` mark is reserved for administrator-only features that do not
exist yet.

### Several people

Several projects can be open at once (`max_open_projects`), and several people can work in the same project:

- **Undo is per person**: Ctrl+Z undoes your own latest change, never someone else's. If someone changed the
  same images (or the class list) after you, your undo is refused with who did it; once they undo theirs,
  yours works again. Likewise a class rename, reorder, merge or removal cannot be undone while others' label
  changes made for the new class list are still in place, nor after an import.
- **Changes show up**: every open page asks the server every few seconds what others changed. The gallery
  refreshes (thumbnails included); an editor showing an image someone else just changed says so and offers
  **Reload** (saving over their change would be refused); a class rename, reorder or removal reloads the
  class list everywhere without closing the editor, and a save made against the old class list is refused
  rather than written with stale class ids.
- **Who is where**: the editor shows who else has its image open, and the gallery marks those images with the
  person's initials. Nothing is locked: two people can still edit the same image; the second save after a
  change is refused and reloads.
- **Who did what**: the editor shows who last edited and who reviewed an image; the gallery filters by
  reviewer ("Reviewed by" under Filters); imports, prediction batches, Label QA analyses, embeddings and
  training runs show who started them; the Dataset tab lists recent activity, kept in
  `<project>/.yolo-workbench/activity.jsonl` (rotated past 16 MB).
- Pending predictions are shared: everyone in the project sees and can accept or reject them.

### Embeddings (optional)

The Embeddings tab needs a MongoDB for FiftyOne. Run one locally, e.g.

```bash
docker run -d --name fiftyone-mongo --restart unless-stopped -p 127.0.0.1:27017:27017 -v fiftyone-mongo-data:/data/db mongo:7
```

and set `[embeddings] mongodb_uri = "mongodb://127.0.0.1:27017"`. It holds nothing that must be backed up
(the visualization is recomputed). With `enabled = false` the tab reports that embeddings are turned off.

## Backup and restore

Back up **the whole projects folder** — images, labels, `data.yaml`, `project.json`, every project's
`.yolo-workbench/` (review status and who edited what, the activity log, Label QA, backups, trash), `models/`
and `runs/`, plus the hidden `.training/` (queue), `.trash/` and `.workbench/` (accounts) — and
`/etc/yolo-workbench/`. The cache folder and
`<project>/.yolo-workbench/staging/` can be skipped.

For a consistent copy, take it from a filesystem snapshot (LVM, ZFS, btrfs), or stop the service while
copying (`systemctl stop` pauses training first): a copy taken while a class rename rewrites thousands of
label files can catch it half done. The per-session label backups on the same disk are not a disaster
recovery copy.

Restore onto a fresh host: install as above, stop the service, put the projects folder back at
`projects_root` (owned by `yolo-workbench`) and the configuration in `/etc/yolo-workbench/`, then start. Runs
that were active in the backup show as interrupted; resume them from the Training tab.

## Upgrade and rollback

```bash
# 1. Training tab: Hold & pause, and wait for the run to pause (or accept that the restart pauses it)
cd /path/to/checkout && git pull
sudo bash deploy/install.sh                  # installs the new release next to the old one, switches `current`
sudo systemctl restart yolo-workbench
curl -s localhost:8765/readyz                # then release the queue in the Training tab
```

Rollback: `sudo ln -sfn /opt/yolo-workbench/releases/<previous> /opt/yolo-workbench/current && sudo systemctl restart yolo-workbench`.
Keep old releases until no run they started is still training (a run's process uses the files of the
release that launched it); delete them by hand afterwards.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `Configuration error: …` in the journal, service fails | a setting is wrong or unknown; `yolo-workbench-server check` names it |
| `the projects folder … does not exist (is its disk mounted?)` | mount missing; the unit also waits for `/srv/yolo-workbench` |
| `… is already in use by process …` | another server or a desktop session uses the folder |
| the service stopped by itself and came back 10 s later, `low memory! … sending SIGTERM` in `journalctl` | an OOM killer (earlyoom, systemd-oomd) asked the server to stop: the active run paused at its checkpoint and resumes after the restart (`Restart=always`). Give the host more memory, or load fewer models during training (`gpu_models_during_training = "refuse"`, a lower `max_loaded_models`) |
| `start request repeated too quickly` / `start of the service was attempted too often` | more than 10 starts within 5 minutes (a crash loop, or many restarts); fix the cause, then `sudo systemctl reset-failed yolo-workbench` and start it |
| browser shows `{"detail":"Host not allowed"}` | add the name you typed to `allowed_hosts` |
| the login page says "No user accounts exist yet" | create one: `yolo-workbench-server user add <name> --admin` (see [Users and sign-in](#users-and-sign-in)) |
| someone forgot their password | `yolo-workbench-server user passwd <name>` sets a new one |
| "Cannot undo your last change: … changed the same images since" | someone else edited those images after you; they can undo theirs first, or edit by hand |
| uploads fail with "Not enough free disk space" | free space below `min_free_disk_gb` |
| a run shows *interrupted* after a restart | its epoch did not finish within `pause_deadline_s`; it resumes from the previous checkpoint |
| GPU not found by `check` | driver/CUDA mismatch, or `CUDA_VISIBLE_DEVICES` hides it; `nvidia-smi` as the service account |
| first SAM / training start is slow | weights download on first use into `/var/cache/yolo-workbench/weights/` (set `HTTPS_PROXY` in the env file behind a proxy) |

## Acceptance checklist

Before people rely on a new installation (and after major upgrades):

- [ ] `check` passes with the GPU listed and at least one administrator account.
- [ ] From another machine: sign in (a wrong password is refused), project list, create a project, upload
      images, annotate and save, a 409 on a conflicting save, thumbnails, smart annotation, a prediction model
      (upload and load), Label QA, training progress, downloading best weights, embeddings on or reported off,
      change password, sign out.
- [ ] Two people in one project (two browsers, two accounts): each sees the other's edits and presence within
      seconds, undo only takes back one's own change and is refused after the other edited the same image, a
      class rename by one reloads the other's classes.
- [ ] With a tiny dataset, for Ultralytics **and** RF-DETR: `systemctl restart` during an epoch (pauses,
      resumes), `kill -9` of the server process (systemd restarts it; the run is interrupted and **Resume** continues from its checkpoint), a host reboot,
      an ONNX export interrupted by a restart (fails cleanly, can be repeated).
- [ ] A 24–72 hour run with regular work: memory and disk use stay flat; `/healthz` stays fast; no leftover
      processes after `systemctl stop`.
- [ ] Restore a backup onto a second machine; upgrade and roll back once.
