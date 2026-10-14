# Training: plan

Add a **Training tab** for Ultralytics YOLO models (detection, segmentation, classification; the task follows the dataset category). The tab has three parts:
- a **history** of runs with their validation and test results;
- a **new-run form** with a GUI for the common options and a **raw YAML** mode, which can be prefilled from any previous run;
- a **live view** of the active run: current epoch, batch progress, the per-epoch metric table that the CLI prints, the best epoch so far, and two live charts (losses, and mAP or accuracy).

Target is Ultralytics 8.4.x (8.4.95 is installed). Callback names and metric keys below were checked against that version.

## Decisions taken

| Topic | Decision |
|---|---|
| Modes | **Both modes.** Projects store runs in `<slug>/runs/<run>/`. Single-dataset mode stores them in a visible `<dataset dir>/runs/<run>/` (the folder holding `data.yaml`, or the classification root), which is where the `yolo` CLI puts them too |
| Dataset during training | **Lock editing.** Ultralytics trains on the live dataset, and every write path rejects with 409 "read-only while training *X*" until the run ends. Undo and redo are blocked too |
| Process model | **Detached subprocess + persistent queue.** A run survives server restarts (the server reattaches by PID). One run at a time; further runs wait in a FIFO queue |
| Queue scope (projects) | **One global queue** across all projects (one GPU). Each project's Training tab shows the whole queue, with its own runs highlighted |
| Raw args format | **YAML**, the same `key: value` shape as Ultralytics' `args.yaml` |
| Base model sources | **Official pretrained** (YOLO26 / YOLO11 / YOLOv8 × n/s/m/l/x, suffix by category) and **a previous run's `best.pt` / `last.pt`** |
| Extras in v1 | Test-split evaluation, resume of interrupted runs, "use for labeling", run comparison, discovery of runs made outside the app, rename + notes |
| Deleting runs | **Manual only.** The app never deletes a run folder; you remove it on disk. The run store copes with folders that disappear (see *Run store*) |
| Not in v1 | Export (ONNX etc.), local `.pt` and architecture-YAML base models in the GUI. Raw mode still accepts any `model:` value |

Decisions I made myself (tell me if you disagree):
- **Three ways to stop a run:**
  - **Pause:** exit after the current epoch's checkpoint is saved, skip the final eval, and keep `last.pt` resumable.
  - **Finish early:** set `trainer.stop`. Ultralytics then runs its normal final validation on `best.pt`, plus our test eval. The run is complete, and it is *not* resumable because Ultralytics strips the optimizer from `last.pt`.
  - **Kill:** SIGTERM, then SIGKILL, to the process group. The run is resumable from the last completed epoch.
- **Resume reuses the run's original args** (confirmed). Ultralytics resume mostly ignores new overrides, so the UI offers no fields for it; to train longer, duplicate the run or fine-tune from its weights. A resumed run goes to the back of the queue, like any new run.
- **Live data comes from Ultralytics callbacks**, not from parsing tqdm output. The runner writes `state.json` and `epochs.jsonl`. The console log is still captured for errors.
- **The queue only advances while the server is running.** The active run itself keeps going without the server. On restart, the server reattaches and, if the run finished in the meantime, starts the next queued run.
- **`labels/*.cache` files that Ultralytics creates next to the labels are deleted when the run ends**, unless they already existed. `cache: disk` is rejected, because it writes a `.npy` next to every image in the locked dataset; `cache: ram` is allowed.
- **Single device only.** A multi-GPU `device: 0,1` uses DDP, which re-spawns the trainer without our callbacks, so there would be no live progress. Rejected in v1.
- **The server never imports torch for training.** GPUs are listed with `nvidia-smi`, so the server doesn't open a CUDA context that holds VRAM. Ultralytics' arg validation (`ultralytics.cfg`) is imported lazily, as the predictor already does.
- **Charts use vendored uPlot** (`static/vendor/uPlot.iife.min.js` + css, MIT, pinned version). It's about 50 KB with no build step, and gives live append, hover readouts, legend toggles and multi-run overlays out of the box.
- **The UI polls** (1 s while the Training tab is visible, 5 s otherwise for the header chip), matching jobs, predictor and embeddings. No SSE or WebSocket.

## On-disk layout

```
<runs root>/<run-dir>/                 # <slug>/runs/ or <dataset dir>/runs/
  args.yaml, results.csv, results.png, confusion_matrix*.png, *_curve.png, labels.jpg, val_batch*_pred.jpg …   (Ultralytics)
  weights/best.pt, weights/last.pt                                                                              (Ultralytics)
  test/                                # plots from the test-split eval (Ultralytics val, split=test)
  workbench/
    run.json        # {version, name, notes, created, source: "workbench"|"external", task, base_model, dataset: {fingerprint, splits{train,val,test}, names}}
    spec.json       # what the runner executes: {args (incl. managed keys), resume, test_eval, cleanup_caches}
    state.json      # runner-owned, atomically rewritten ≤1/s: {pid, pid_create_time, status, phase, epoch, epochs, batch, batches,
                    #   running_losses, eta, best_epoch, best_fitness, gpu_mem, error, updated}
    epochs.jsonl    # one line per completed epoch: all results.csv columns + fitness, is_best, epoch_time, gpu_mem
    val.json        # final val on best.pt: overall + per-class (metrics.summary())
    test.json       # same for the test split, if the dataset has one
    control.json    # server → runner: {"action": "pause" | "finish"}
    train.log       # stdout+stderr

Training state root:  <projects>/.training/queue.json     (projects mode; dot-dir, so never listed as a project)
                      <dataset>/.yolo-workbench/training/queue.json   (single mode)
  queue.json        # ordered [{project, run_dir}], the active entry first
```

- **Run directory names** are the sanitized display name (`exp-3`, `y11s-640-aug`), made unique with a suffix. Renaming a run only changes `run.json.name`, the same rule as project slugs.
- **Run IDs** in the API are `sha1(relative run path)[:12]`, like `record_id`. That way external runs with nested paths (`detect/train2`) work too.
- **The runs folder is never part of the dataset.** `Dataset` skips the runs root in its label/orphan scans (`_find_orphans` rglobs `*.txt` under the label roots, which may include the dataset dir with colocated layouts) and in classification class discovery, so run outputs can't turn into orphan-label issues or a bogus class.
- **Weights download cache:** the runner's working directory is `~/.cache/yolo-workbench/weights/`. Pretrained downloads, and the AMP-check model Ultralytics downloads, land there once and are reused, and nothing gets dumped into the server's cwd. The stray `runs/detect/` in the repo root is that kind of leak. The runner always passes `project=` and `name=` explicitly.
- **External runs:** any directory under the runs root with `args.yaml` + `results.csv`. Runs from the CLI with the default `project` end up nested (`runs/detect/train2`), so the scan goes up to 3 levels deep. In single mode, `./runs/**` (the server's cwd) is also scanned, keeping only runs whose `args.yaml` `data` resolves to this dataset. Their `run.json` is created lazily, on first rename or note.

## Backend architecture

### `train_runner.py` (new): the subprocess
Launched as `sys.executable -m yolo_workbench.train_runner <run-dir>/workbench/spec.json` with `start_new_session=True`, so it has its own process group. Ctrl+C on the server terminal doesn't reach it, and kill can target the group. stdout and stderr go to `train.log`, with `PYTHONUNBUFFERED=1`.

1. Write `state.json` with `status: preparing`. Snapshot which `labels/*.cache` files already exist.
2. Run `YOLO(model).train(**args)`, or `YOLO(last.pt).train(resume=True)`, with callbacks:
   - **`on_pretrain_routine_end`:** `status: training`, `epochs`, `batches = len(train_loader)`.
   - **`on_train_batch_end`** (throttled to 1/s): `batch`, `running_losses = label_loss_items(tloss)`, `gpu_mem`, `eta`. This is what the CLI's live line shows. It also checks `control.json`.
   - **`on_train_epoch_end`:** sets an "in epoch" flag, which is needed to tell the final-eval call apart (below).
   - **`on_fit_epoch_end`:**
     - Append the epoch row (`label_loss_items(tloss)` + `trainer.metrics` + `trainer.lr`). Set `is_best = trainer.fitness == trainer.best_fitness`, the same test Ultralytics uses to write `best.pt`.
     - Update `best_epoch` and `best_fitness`.
     - Apply the control action: **finish** sets `trainer.stop = True`; **pause** raises `RunPaused`. Pause fires here, after `save_model()`, so `last.pt` holds this epoch.
     - Ultralytics fires this callback a second time from `final_eval()` without a preceding `on_train_epoch_end`. That call is recorded as the final val (`status: validating`) rather than as an epoch row.
   - **`on_train_end`:** write `val.json` from `trainer.validator.metrics.summary()` plus the overall metrics.
3. If the dataset has a test split and training completed (or finished early), `status: testing`. Run `YOLO(best.pt).val(data, split="test", imgsz, batch, project=<run>, name="test", exist_ok=True)` and write `test.json`. A failing test eval is recorded as a warning; it doesn't fail the run.
4. Final `status`: `completed` | `paused` | `failed` (exception text plus a log tail). Remove only the `labels/*.cache` files this run created.

Best epoch for runs without `epochs.jsonl` (external runs) is computed from `results.csv` with the 8.4 fitness formulas:
- detection: `mAP50-95(B)`;
- segmentation: `mAP50-95(B) + mAP50-95(M)`;
- classification: `(top1 + top5) / 2`.

### `training.py` (new): `TrainingManager`
One instance per app (`app.state.training`). In projects mode it is workspace-level, not per session, because runs outlive project switches.
- **Run store:**
  - lists runs for a dataset: scans the runs root, joins `run.json` + `state.json`, and reads summary metrics from the last `epochs.jsonl` / `results.csv` row;
  - returns detail, epochs since N, a log tail from an offset, and plot files (whitelisted image extensions, path confined to the run dir);
  - renames and edits notes.
  - **Manually deleted runs:** listing rescans the runs root on every request (cheap: one `stat` per run dir, cached by mtime), so a deleted folder simply disappears. A queued entry whose dir is gone is dropped from `queue.json` and logged. If the *active* run's dir is deleted, the process is left alone and its queue entry is dropped once the process exits. The UI handles 404 on an open run by going back to the list.
- **Args:**
  - `options(category)`: GUI field schema plus the real defaults from `ultralytics.cfg.DEFAULT_CFG_DICT`, the pretrained list with cached flags, previous-run weights (with a class-names-match flag), and devices from `nvidia-smi`.
  - `validate(yaml_text)`: parse, run `get_cfg` checks, and return line-numbered errors.
    - Managed keys (`data`, `project`, `name`, `exist_ok`, `resume`, `mode`, `task`, `save_dir`) are errors.
    - `cache: disk` and multi-device lists are errors.
    - Unknown keys get Ultralytics' "did you mean" suggestion.
- **Enqueue:**
  1. Validate.
  2. Run a preflight check:
     - train and val splits are non-empty, there is at least 1 class, and indexing is finished;
     - warnings for error-level validation issues and for classes with no examples in train;
     - an error if a fine-tune base has a different class count than the dataset (a warning if only the names differ).
  3. Create the run dir with `run.json`, `spec.json` and `state.json {status: queued}`, and append it to `queue.json`.
  - Data source:
    - **projects:** the managed `data.yaml`, which is Ultralytics-compatible by construction;
    - **single mode:** a generated `workbench/data.yaml` with absolute split sources and names from the loaded `Dataset`. This sidesteps `path:`/prefix quirks;
    - **classification:** the dataset root.
- **Supervisor thread** (2 s tick):
  - If nothing is running and the queue has an entry, start it. That means acquiring the dataset lock, writing the dataset fingerprint into `run.json`, and launching. If that dataset has a running import job, it waits, with status "waiting for import".
  - Detect when the active process exits: release the lock and pop the queue.
- **Reattach on startup:** for `queue.json`'s active entry, check `state.json.pid`. It counts as alive only if `psutil` (an Ultralytics dependency) confirms the PID, `create_time` and the `train_runner` cmdline, which guards against PID reuse. If dead and the status isn't terminal, mark it `interrupted`, which is resumable.
- **Control:**
  - `stop(run, mode)` writes `control.json`, or for kill signals the process group;
  - `cancel(run)` removes a queued run. Its dir only holds the `workbench/` files written at enqueue, so the app removes that dir. This is the one exception to manual deletion: nothing was trained;
  - `resume(run)` is allowed for `paused`/`interrupted`/`failed` runs whose `last.pt` still has optimizer state;
  - `reorder(queue)`.
- `active_summary()` feeds the header chip and project cards: project, run name, epoch/epochs, best mAP.
- **Testability:** `TrainingManager(..., launcher=...)`. Tests pass a fake launcher that writes `state.json` / `epochs.jsonl` the way the runner would, so no Ultralytics or GPU is needed.

### Dataset lock (`dataset.py`)
- `Dataset.write_guard: Callable[[], str | None]` defaults to `lambda: None`. When it returns a reason, the low-level write paths raise `DatasetLocked(WriteConflict)`, which maps to 409 with the reason. Those paths are `_commit`, `_commit_moves`, `_run_transaction` (undo/redo, schema, folder and record moves), `fix_issue(s)` and `importer.run_import`.
- The session factory (both modes) wires `write_guard = lambda: training.lock_reason(dataset_root)`. A project opened mid-training is therefore locked from the start.
- When a run starts while its dataset is open, the manager takes the dataset's write lock for a moment to let in-flight writes finish before flipping the guard.
- `GET /dataset` gains `locked: {reason, run_id} | null`, so the UI can disable editing.
- Workspace: `trash`/`purge` of a project with a running or queued run returns 409. Switching projects is allowed during training, since training doesn't hold the session.

### Predictor integration
- `discover_models()` also lists `<runs root>/*/weights/best.pt` as source `"run"`, labelled with the run name. "Use for labeling" on a run calls the existing `/predictor/load` with that path and switches to the Annotation tab. Class auto-mapping already handles names.

## API sketch

Dataset-scoped (in `dataset_router`, so under `/api/v1` or `/api/v1/projects/{slug}`):
```
GET    /training/options                     form schema, defaults, pretrained list, previous-run weights, devices
POST   /training/args/validate   {yaml}      → {args, errors[{line,message}], warnings}
GET    /training/runs                        summaries (status, epochs, best epoch + metrics, test metrics, source, queue position)
POST   /training/runs            {name, notes, args_yaml}   → enqueue (preflight warnings need {confirm: true})
GET    /training/runs/{id}                   detail: run.json, state, args, val/test results, plot file list
GET    /training/runs/{id}/epochs?since=N    new epoch rows
GET    /training/runs/{id}/log?offset=B      log tail (\r-progress collapsed)
GET    /training/runs/{id}/files/{name}      plots (images only, confined to the run dir)
GET    /training/runs/{id}/args?copy=1       YAML with managed keys and default-valued keys stripped ("duplicate")
PATCH  /training/runs/{id}       {name?, notes?}
POST   /training/runs/{id}/stop  {mode: pause|finish|kill}
POST   /training/runs/{id}/resume
POST   /training/runs/{id}/cancel            queued → removed
```
App-scoped (shared queue): `GET /api/v1/training/queue` → `{active, queued[]}` across projects, and `POST /api/v1/training/queue/reorder`.

## Frontend

New `training.js` (the tab), `charts.js` (uPlot wrapper: live append, best-epoch marker, overlay mode), and `static/vendor/uPlot.*`. A **Training** tab is added to `index.html`.

- **Runs list** (left column or top table): name, status pill (queued #n / training 37/100 / paused / completed / failed / interrupted), base model, imgsz, epochs, best epoch, best val mAP50-95 (or top1), test mAP50-95, created, external badge.
  - Row actions: open, duplicate, fine-tune from, resume, use for labeling, rename/notes.
  - Checkboxes select runs for **Compare**.
- **Run detail:**
  - **Header:** status, epoch `37/100`, batch progress bar `112/240`, elapsed and ETA, GPU mem, **★ best epoch 31 (mAP50-95 0.612)**, and the Pause / Finish early / Kill buttons (Kill asks for confirmation).
  - **Charts (2):**
    - **Losses:** train solid and val dashed, per component (box/cls/dfl, plus seg for segmentation, or loss for classification), with an optional log-y toggle;
    - **Metrics:** P, R, mAP50, mAP50-95 (box and mask for segmentation; top1/top5 for classification), with a vertical marker at the best epoch.
  - **Epoch table**, mirroring the CLI output: Epoch, GPU mem, train losses, val losses, P, R, mAP50, mAP50-95, lr, time. The newest epoch is on top and the best row is starred and highlighted. A "current" row shows the running losses of the epoch in progress.
  - **Results:** final val on `best.pt` and test-split results side by side. Overall metrics plus a per-class table (P, R, mAP50, mAP50-95, instances).
  - **Plots** gallery: confusion matrix (normalized), PR/F1 curves, `results.png`, val prediction batches; test-split plots under their own heading.
  - **Args** (read-only YAML, non-default keys highlighted), **Log** tail (collapsible; opens automatically on failure), **Notes**.
- **New run** (form panel/dialog), with a **GUI ⇄ Raw YAML** toggle:
  - GUI groups, filtered by category:
    - **Basics:** name, base model (pretrained family/size, or a previous run's best/last), epochs, imgsz, batch (auto −1 / number), device, patience, seed;
    - **Optimizer:** optimizer, lr0, lrf, momentum, weight_decay, warmup_epochs, cos_lr, close_mosaic, amp;
    - **Augmentation:** hsv_h/s/v, degrees, translate, scale, shear, perspective, flipud, fliplr, mosaic, mixup, copy_paste (segmentation), erasing / auto_augment (classification);
    - **Advanced:** workers, cache (off/ram), fraction, freeze, rect, multi_scale, dropout, single_cls, save_period.
  - Every field shows the Ultralytics default as a placeholder. Only changed values go into the args.
  - **One args object is the source of truth.**
    - GUI → Raw: serialize to YAML, with comments for the group headings.
    - Raw → GUI: `POST /training/args/validate`. Errors are listed with line numbers, and switching back is blocked until they're fixed.
    - Keys the GUI doesn't cover are kept and shown as chips under "Other arguments (edit in raw)", so nothing is silently dropped.
  - **Duplicate** (from a run) opens this form prefilled from `GET /runs/{id}/args?copy=1`, with the name incremented (`exp-3` → `exp-4`).
  - The **form remembers the last-used args** per dataset (localStorage), as a convenience.
  - The **preflight** result is shown before Start. Warnings need an explicit "Start anyway". The button reads "Start" when idle and "Add to queue (#2)" otherwise.
- **Queue panel:** active run plus queued runs (all projects in projects mode; others greyed and labelled with their project), with remove and up/down.
- **Compare view** (2+ selected):
  - the same two charts with one color per run, plus a metric picker per chart;
  - a final metrics table (val and test);
  - an **args diff** showing only keys that differ;
  - a dataset fingerprint/counts row, which flags runs trained on different data.
- **Outside the tab:**
  - a header chip "Training exp-4 · 37/100 · ★0.612" on every tab, which links to the run;
  - a lock banner, with editing controls disabled, on Annotation, Exploration, Validation and Dataset while that dataset is training;
  - a "Training" / "Queued" badge on the cards of `projects.html`.

## Phases

Each phase ships on its own, and the tests stay green between phases.

1. **Backend core:**
   - `train_runner.py` (callbacks, control file, pause/finish, test eval, cache cleanup, resume);
   - `training.py` (run store, results parsing, persisted queue, supervisor, reattach, kill);
   - the `Dataset.write_guard` lock and workspace trash/purge guard;
   - the API.
2. **History and live view:**
   - Training tab, runs list, run detail (header, two uPlot charts with the best marker, epoch table, results, plots, args, log);
   - header chip and lock banners.
3. **New run:**
   - options endpoint, GUI form, raw YAML mode with server validation, model picker (pretrained + previous runs), preflight;
   - duplicate and fine-tune-from.
4. **Extras:**
   - compare view;
   - external-run discovery;
   - rename and notes;
   - use for labeling, plus predictor discovery of run weights;
   - queue panel reorder;
   - project-card badges.

## Tests (temporary dirs only, never a real dataset; no GPU or Ultralytics needed except the opt-in smoke test)

- **Results parsing:**
  - `results.csv` fixtures with real 8.4 headers for detect, segment and classify (column groups, fitness, best epoch);
  - tolerates extra or unknown columns and a partial last line.
- **Args:**
  - managed keys, `cache: disk` and multi-device are rejected with line numbers;
  - unknown keys get a suggestion;
  - the copy strips defaults and managed keys;
  - a GUI-unknown key round-trips.
- **Queue and lifecycle** (fake launcher):
  - FIFO order and reorder;
  - one active run at a time;
  - `queue.json` survives a manager restart;
  - a dead PID becomes `interrupted`, and PID reuse (`create_time` mismatch) is not mistaken for a live run;
  - the queue waits while an import job is running;
  - cancel removes the queued dir.
- **Control:**
  - pause/finish write `control.json`;
  - kill signals the group (mocked);
  - resume is allowed only with a resumable `last.pt` and goes to the back of the queue.
- **Dataset scans:** a `runs/` folder with `.txt` files next to `data.yaml` (colocated layout) and a `runs/` folder in a classification root don't produce orphan labels or classes.
- **Lock:**
  - while running: annotation PUT, bulk edit, fix, class ops, folder add, image move/delete, import commit, undo and redo all return 409 with the reason, and reads still work;
  - the lock is released at the end;
  - a project opened mid-run is locked;
  - trash/purge of a training project returns 409.
- **API:**
  - same endpoints in single and projects mode;
  - file serving is confined to the run dir (path traversal);
  - the epochs `since` cursor;
  - a run dir deleted on disk vanishes from the list, a queued entry with a missing dir is dropped, and an open run returns 404;
  - the log tail collapses `\r`.
- **Runner unit:** callback logic against a fake trainer object: the final-eval call isn't counted as an epoch, `is_best`, throttling, and the cache-cleanup snapshot.
- **Smoke (opt-in, `-m slow`):** a real 1-epoch `yolo11n` train on CPU on a 4-image generated dataset, checking `state.json`, `epochs.jsonl`, `val.json`, `test.json` and the cleaned caches. Skipped if the weights can't be downloaded.

## Known limitations / later
- A disk-usage view per run (weights add up: ~6 MB for n up to ~130 MB for x). Deletion stays manual. Export to ONNX/TensorRT. Local `.pt` / YAML base models in the GUI.
- Multi-GPU (DDP) with live progress would need a file-based progress channel from rank 0. Hyperparameter tuning (`model.tune`).
- An optional frozen snapshot mode (hardlinked images + copied labels) for exact reproducibility without locking. The fingerprint in `run.json` already shows when data changed between runs.

## Implemented — phase 1 (backend core)

Built as planned: `train_runner.py`, `training.py`, the `Dataset.write_guard` lock, workspace wiring (shared queue, trash guard, opened projects locked), the API, and `tests/test_training.py` (fake launcher; real smoke test behind `WORKBENCH_SLOW_TESTS=1`). Verified against a real GPU run through the CLI server: live progress, pause → resume, finish early (final + test eval), lock release, cache cleanup.

Deviations and details settled while building:
- **The training data is a generated `workbench/data.yaml` with absolute per-split file lists** (`train.txt`, …) in both modes, built from the workbench's own index when the run starts. Ultralytics trains on exactly the images the workbench shows, whatever the YAML quirks. Preflight rejects datasets whose labels Ultralytics would not find (labels next to images inside an `images/` folder) and non-contiguous class IDs. Classification trains on the dataset root and needs `<root>/train` and `<root>/val` (or `test`).
- **A queued run starts from a project that isn't open** by loading a temporary `Dataset` for it.
- **The runner's PID lives in `workbench/process.json`** (written by the server), not in `state.json`. **Kill** is recorded as `control.json {"action": "kill"}`, so a restarted server still reports the run as `stopped` rather than `interrupted`.
- **Best epoch = the last epoch with the top fitness.** Ultralytics rewrites `best.pt` on ties; the plan said "first".
- **Arg types are checked with Ultralytics' own `check_cfg`**, imported lazily (~1 s once per server). Unknown keys and help texts come from `cfg/default.yaml`, read without importing torch.
- **Extra endpoint:** `POST /training/preflight {yaml}` for the form to show errors/warnings before Start. **Start with warnings** returns `{"queued": false, "warnings": [...]}` until resent with `confirm: true`.
- **Cancelling a queued resume** puts the run back to `paused` instead of deleting anything.
- **The log endpoint strips ANSI colors** as well as collapsing `\r` redraws.
- **`psutil` is used through Ultralytics' dependency.** It's already in `uv.lock`, so `pyproject.toml` is unchanged.

## Implemented — phase 2 (history and live view)

The Training tab (`training.js`) has:
- the runs list: active first, then queued, then newest;
- the run view: header and actions, live progress, KPI tiles, two charts, the epoch table, val/test per-class results, plots with a lightbox, args, and the log tail.

Outside the tab there's the header chip, plus the read-only banner and disabled write controls on every tab while this dataset trains. Pause, Finish early, Kill (with confirmation), Resume and "Remove from queue" are wired up. Verified in the browser against real GPU runs: live updates, the switch to final results when a run ends, pause → resume → kill, and dark mode.

Deviations:
- **No uPlot.** `charts.js` is a small SVG line chart, so there's no third-party download or vendored file. It has:
  - one y axis;
  - a crosshair tooltip that lists every series at the hovered epoch;
  - a legend that toggles series groups;
  - a best-epoch marker;
  - keyboard stepping with ←/→.

  Colors are fixed categorical slots (`--series-1..4`, checked for color-blind separation and contrast in both themes). Train/val (and box/mask) are told apart by solid vs dashed lines, and the epoch table doubles as the table view.
- **The x axis ends at the last recorded epoch**, not at `epochs`, so early epochs and early-finished runs stay readable. Progress against the total is in the progress bar.
- **The ETA is computed by the runner** from the mean of the last 5 epoch times, updated at every epoch end. The batch callback is throttled to 1/s, so it alone gave stale values on fast epochs.
- **Result tables leave out the F1 and Images columns** so they fit a half-width panel. The overall metrics row is above each table.
- **Deep links:** `?tab=training&run=<id>`. The chip uses them to open a run in another project.

## Implemented — phase 3 (new run)

- **The form** (`trainform.js`) has a Form ⇄ Raw YAML toggle over one args object. A field left empty uses the Ultralytics default, shown greyed. A value equal to the default is dropped, so only real choices end up in the args and the YAML. Raw → Form goes through the server validator; errors are listed with line numbers and keep you in raw mode. Keys without a field are kept and shown under "Other arguments".
- **Fields** come in Basics, Optimizer, Augmentation and Advanced groups, filtered by category. Sections with changes open automatically and show a "N changed" count, and help text comes from Ultralytics' `cfg/default.yaml`.
- **Model:** Pretrained (YOLO26 / YOLO11 / YOLOv8 × n–x, the suffix follows the category, shows whether it's already downloaded), or Previous run (best.pt / last.pt, with a warning when that run had different classes). A model set in raw YAML that matches neither shows as "Custom".
- **Before starting:** a live validation + preflight panel shows the split sizes, errors (Start is disabled), warnings (the button turns into "Start anyway") and notes. The button reads "Add to queue (#n)" when runs are ahead.
- **Duplicate and Fine-tune** on every run prefill the form from `GET /runs/{id}/args`. Fine-tune also sets the model to the run's best.pt (or last.pt) and names the run `<name>-ft`. The last-started args are remembered per dataset.
- **The run's Arguments panel** shows only the values chosen for that run, with all of `args.yaml` (changes highlighted) behind a toggle.

Bugs found and fixed while testing with real runs (backend):
- **Kill could corrupt checkpoints.** Ultralytics writes `last.pt`/`best.pt` in place, and Kill used to SIGTERM the whole process group, so a kill during a save left a broken file. It happened once ("PytorchStreamReader failed reading zip archive" when fine-tuning from a killed run). Now:
  - Kill sends SIGTERM to the runner process only.
  - The runner exits right away during batches. From the end of an epoch's batches until its checkpoint is saved, and through `final_eval()`, it waits and exits right after the write.
  - The group is SIGKILLed only after a 30 s grace period (was 10 s).
- **The force-kill timer** targeted the run folder, so a Resume within the grace period got its new process SIGKILLed. It now targets the exact process (PID + create time).

## Implemented — phase 4 (extras)

- **Compare** (`compare.js`): tick up to 8 runs in the list, then press Compare.
  - Two charts, each with a metric picker (validation metrics, val/train losses, fitness, learning rate).
  - Each run keeps its color slot (`--series-1..8`, validated as a set in both themes) for as long as it stays ticked.
  - A results table (best epoch, best/final val, test, dataset split counts, flagged when the fingerprint differs).
  - The arguments that differ, compared against the Ultralytics defaults, so `''`/missing count as default.
- **Queue panel** in the sidebar (shown while runs are waiting):
  - covers all projects; other projects' runs are greyed and labelled;
  - ↑/↓ reorder through `POST /api/v1/training/queue/reorder`;
  - × removes a queued run, for this project's runs only. Removing another project's run would need its dataset router, which opens (switches to) that project.
- **Rename / notes** dialog (PATCH), also for external runs. It saves on the form's `submit`: in testing, the dialog's `close` event never reached its listener.
- **Use for labeling:** loads the run's `best.pt` (or `last.pt`) through the existing predictor and switches to Annotation. `PredictorManager.discover_models()` now lists the dataset's run weights by run name ("Baseline · best.pt", tagged "trained here") right after recent models.
- **Project cards** on `projects.html` show "● Training <run> · n/N" or "Queued #n", polled from the shared queue.
- **Backend tweak:** a run's `model` in the API is the base model from `run.json`. A resume rewrites `args.yaml`'s model to the run's own `last.pt`, which made resumed runs look like fine-tunes of themselves.

Verified in the browser: comparing 4 real runs, queue reorder/remove with 3 runs, rename/notes, use for labeling, and projects mode (badges on cards, the chip in another project linking across, the lock only in the training project).

