# RF-DETR: plan

Add **RF-DETR** (Roboflow's DETR-family detector, `rfdetr` package) as a second framework in the Training tab. Use it for **detection and segmentation datasets**; RF-DETR has no classification models. Same queue, dataset lock, live view, compare, and "use for labeling" as the Ultralytics runs.

Target is **rfdetr 1.11.2** (2026-10-04), whose training is built on PyTorch Lightning. Everything under *Findings* was checked against the source at tag `1.11.2` and the official docs. Items still marked "verify" need a real run.

## What it takes (short answer)

- **Training: medium effort, mostly plumbing.** The queue, the detached runner, the lock, the live view and compare are already framework-neutral in shape. What is Ultralytics-specific today:
  - the args schema and validation;
  - how the dataset is handed to the trainer;
  - the runner's callbacks;
  - how results files are read;
  - where weights live.

  RF-DETR needs its own version of each, behind a small framework layer. It also needs two things Ultralytics didn't:
  - **a generated dataset view**, because RF-DETR can't read `.txt` split lists and needs every split folder inside one root;
  - **EXIF handling**, because RF-DETR doesn't apply EXIF orientation.
- **Prediction: smaller than it looks.** `PredictorManager` and Label QA already go through a backend with `names` + `predict(path, conf=)`. An `RFDETRBackend` (about 100 lines), detection of which framework a weights file belongs to, and `.pth` discovery cover it. "Use for labeling" and Label QA then work on RF-DETR runs. Recommended, as phase 3.
- **Dependencies resolve cleanly.** Adding `rfdetr[train]` to this project and running `uv lock` (on a scratch copy) only adds packages. The one change to an existing package is `typer` 0.26.8 → 0.25.1. torch, numpy, opencv and pillow stay put. Every compiled dependency ships Python 3.14 wheels for macOS arm64, Linux x86_64/aarch64 and Windows.

## Findings that shape the design

| Topic | Finding (rfdetr 1.11.2) | Consequence here |
|---|---|---|
| Training API | `RFDETR.train(**kwargs)` builds `RFDETRModelModule`, `RFDETRDataModule` and `build_trainer(...)`, then runs `trainer.fit`. It has no parameter for extra Lightning callbacks. The documented **Custom Training API** is "call `build_trainer`, then `trainer.callbacks.extend([...])`" | The runner uses the Custom Training API with one Lightning callback of ours (see *Spike*) |
| Dataset format | Auto-detects YOLO from `data.yaml` plus split **directories**. **`.txt` file lists are ignored.** Split paths must resolve inside the dataset root (`resolve()`, so a symlinked *directory* fails). Individual image files are only checked with `is_file()`. Labels are found by image stem in the sibling `labels/` folder. Class IDs must be contiguous from 0 | Our `workbench/data.yaml` with absolute file lists can't be used. Build a per-run **view**: `train/images` etc. with hardlinked images and copied labels |
| EXIF | Neither `decode_image` nor `predict()` applies EXIF orientation. Sizes come from the raw `Image.open().size` | Workbench labels refer to the **upright** image, so phone photos would train with misplaced boxes. The view writes upright copies of rotated images. The predictor backend passes upright PIL images |
| Outputs (`output_dir`) | <ul><li>`metrics.csv`: Lightning `CSVLogger`, always written</li><li>`last.ckpt`: every epoch, unless `checkpoint_interval == 1`</li><li>`checkpoint_{epoch}.ckpt`: every `checkpoint_interval` epochs (default 10), full checkpoints with optimizer state</li><li>`checkpoint_best_regular.pth` / `checkpoint_best_ema.pth`: written on improvement with plain `torch.save`, not atomic</li><li>`checkpoint_best_total.pth`: **only at fit end**</li><li>`last_ema.pth`, `training_config.json`</li></ul> | Weights paths become per-framework. Kill must not land during the best-checkpoint writes. A paused run has no `_total` yet, so it falls back to `best_ema`/`best_regular` |
| Logged metrics | <ul><li>`train/loss` plus `train/loss_ce`, `loss_bbox`, `loss_giou`, and for segmentation `loss_mask_ce`, `loss_mask_dice` (aux terms folded into `*_aux`)</li><li>`train/lr`</li><li>`val/mAP_50_95`, `val/mAP_50`, `val/mAP_75`, `val/mAR`, `val/F1`, `val/precision`, `val/recall`</li><li>`val/segm_mAP_50_95` / `_50`</li><li>`val/ema_mAP_50_95`</li><li>`val/AP/<class>` with `log_per_class_metrics`</li><li>`val/loss` only when computed</li></ul> | The runner normalizes the keys (table below), so charts, the runs list and compare work unchanged |
| Best model | Box mAP50-95 for detection, **mask** mAP50-95 for segmentation. Total = the better of EMA and regular | Fitness per framework. Ultralytics segmentation fitness is box+mask; RF-DETR's is mask only |
| Test eval | `TrainConfig.run_test` defaults to False. `model.evaluate(split="test")` returns `test/*` keys. 1.9.3 fixed test silently evaluating val on YOLO data | Runner runs final val (per-class) and test itself, on the best weights, after a finished fit |
| Resume | `trainer.fit(ckpt_path=last.ckpt)`. Optimizer and best-tracking state are restored when `output_dir` is the same | Pause/resume maps cleanly. Resume from `last.ckpt`, else the newest `checkpoint_epoch=*.ckpt` |
| Constraints | `resolution` divisible by `patch_size × num_windows`: 32 for detection, 24 for segmentation, 12 for Seg Nano. `batch_size: auto` is CUDA only. Multi-GPU is Lightning DDP | Validator checks resolution per variant. Single device only, as for Ultralytics |
| Weights cache | `RF_HOME` (alias `ROBOFLOW_HOME`), default `~/.roboflow/models` | Runner sets `RF_HOME=~/.cache/yolo-workbench/weights/rfdetr`, so `options()` can show "downloaded" |
| License | Package and these weights: Apache 2.0. **Detection XL/2XL need `rfdetr_plus` (PML 1.0)**. Seg XL/2XL are in core | Offer detection N/S/M/L and Seg N/S/M/L/XL/2XL |

Variants offered (default resolution):

| Detection | | Segmentation | |
|---|---|---|---|
| `rfdetr-nano` | 384 | `rfdetr-seg-nano` | 312 |
| `rfdetr-small` | 512 | `rfdetr-seg-small` | 384 |
| `rfdetr-medium` | 576 | `rfdetr-seg-medium` | 432 |
| `rfdetr-large` | 704 | `rfdetr-seg-large` | 504 |
| | | `rfdetr-seg-xlarge` | 624 |
| | | `rfdetr-seg-2xlarge` | 768 |

`RFDETRBase` and the old `RFDETRLargeDeprecated` are left out. Raw YAML could still allow them later.

## Decisions to confirm

| Topic | Proposal |
|---|---|
| Dependency | **Core dependency**, like everything else (CLAUDE.md): `rfdetr[train]>=1.11.2,<1.12`. The upper bound is deliberate: the API moved to Lightning within the last six months, and the runner touches the training internals. Bump it after the smoke test passes on a new minor. Not included: `augment` (albumentations, only needed for custom `aug_config`), `loggers`, `onnx`, `plus` |
| Prediction | **Yes, as phase 3.** Without it, RF-DETR runs can't be used for labeling or Label QA, which is most of the point of training here |
| Licensed weights | Detection XL/2XL (PML) are **not offered**. Adding them later means `rfdetr[plus]` and a license note in the picker |

Decisions I made myself (tell me if you disagree):
- **The framework follows `model:`.** `model: rfdetr-medium` means RF-DETR; anything else means Ultralytics. There's no separate `framework:` key, and the payloads and the one-args-object rule stay as they are. Fine-tuning from an RF-DETR run is `model: <variant>` plus `pretrain_weights: <run>/checkpoint_best_total.pth`, using RF-DETR's own field name. The variant must be explicit because older checkpoints lack architecture metadata.
- **Args are RF-DETR's own names** (`lr`, `lr_encoder`, `batch_size`, `grad_accum_steps`, `resolution`, `early_stopping`, …), validated with RF-DETR's pydantic `TrainConfig` / `SegmentationTrainConfig`. I'm not translating Ultralytics names: the semantics differ (for example lr0 with SGD vs AdamW with encoder decay), and a fake common schema would mislead.
- **Pause and Finish early both use Lightning's `trainer.should_stop`**, set at epoch end. Lightning then saves `last.ckpt` for that epoch, because checkpoint callbacks run last. The difference is what follows:
  - **Finish early:** final val and test evals run, and the status is `completed`.
  - **Pause:** the status is `paused`, and the run is resumable.

  Unlike Ultralytics, a finished RF-DETR run's `last.ckpt` keeps its optimizer state, but the UI still treats `completed` as final, the same as for YOLO runs.
- **The data view is built by the runner** (phase `preparing-data`, with progress), not by the server under the dataset lock. The server writes only `workbench/splits.json` (image/label pairs per split, plus names) in `_launch`, which is as quick as today's `.txt` lists. The view is deleted when the runner exits and rebuilt on resume.
- **Losses keep RF-DETR's terms** (`train/ce_loss`, `bbox_loss`, `giou_loss`, `mask_ce_loss`, `mask_dice_loss`), not relabelled as box/cls/dfl. Metrics map onto the Ultralytics `metrics/*(B|M)` names, because they mean the same thing.
- **No Ultralytics augmentation cards for RF-DETR.** RF-DETR's default augmentations need nothing extra. A custom `aug_config` is accepted in raw YAML only, and only when albumentations is importable; otherwise the validator says to install `rfdetr[augment]`.

## On-disk layout (RF-DETR run)

```
<runs root>/<run-dir>/                     # RF-DETR output_dir = the run dir
  metrics.csv, training_config.json        (Lightning CSVLogger / RF-DETR)
  last.ckpt, checkpoint_epoch=N.ckpt       full checkpoints: resume
  checkpoint_best_total.pth                best weights (fit end); checkpoint_best_ema.pth / _regular.pth meanwhile
  workbench/
    run.json        + "framework": "rfdetr", "variant": "rfdetr-medium"   (absent framework = ultralytics: old runs unchanged)
    spec.json       {framework, args, resume, test_eval}
    splits.json     {names, splits: {train: [[image, label|null], …], val: …, test: …}}   written by the server at launch
    state.json, epochs.jsonl, val.json, test.json, control.json, process.json, train.log    (same contract as today)
    data/           the view, only while the runner is alive:
      data.yaml     {path: ., train: train/images, val: valid/images, test: test/images, names}
      train/images/<n>_<stem><ext>   hardlink → symlink (POSIX) → copy;  upright re-encode if EXIF-rotated
      train/labels/<n>_<stem>.txt    copied label (absent = background image)
      valid/…, test/…
```

- **Unique names** (`<n>_` prefix): a split may span several folders whose images share a stem, and RF-DETR pairs labels by stem.
- **Hardlinks cost no space**, and the run dir sits in the dataset dir (same volume). On a symlink or copy fallback, the runner logs how many files were copied.

## Backend

### `frameworks.py` (new): the framework layer
One small class per framework. `training.py` asks it instead of hard-coding Ultralytics:

| Method | Ultralytics (today's code, moved) | RF-DETR |
|---|---|---|
| `owns(model)` | the fallback | `model` matches `rfdetr-[seg-]<size>` |
| `available()` | `find_spec("ultralytics")` | `find_spec("rfdetr")`, so the server doesn't import it |
| `defaults()`, `help()` | `cfg/default.yaml` | `TrainConfig.model_fields` (+ `SegmentationTrainConfig`) + `resolution`/`pretrain_weights`, with a curated help dict where fields have no description |
| `managed_keys` | as today | `dataset_dir`, `output_dir`, `dataset_file`, `resume`, `class_names`, `run_test`, `devices`, `strategy`, `num_nodes`, `progress_bar`, `wandb`, `mlflow`, `clearml`, `project`, `run` |
| `validate(category, text)` | as today | pydantic validate → line-numbered errors (`_key_line`). Plus: unknown keys with difflib suggestions; `resolution` divisibility per variant; seg variant on a detection dataset (and vice versa); `batch_size: auto` without a CUDA device; >1 device; `aug_config` without albumentations; `pretrain_weights` must exist |
| `pretrained(category, weights_dir)` | YOLO family × size | the variant table, with a `cached` flag under `RF_HOME` |
| `fitness(row, category)` | as today | primary metric (box or mask mAP50-95). The max of regular and EMA when both are evaluated |
| `weights(run_dir)` | `weights/best.pt`, `weights/last.pt` | best: `checkpoint_best_total.pth` → `_ema` → `_regular`; last: `last_ema.pth` (weights only) |
| `resume_checkpoint(run_dir)` | `weights/last.pt` | `last.ckpt`, else the newest `checkpoint_epoch=*.ckpt` |
| `external_epochs(run_dir)` | `results.csv` | `metrics.csv`, rows grouped by `epoch` (step rows for lr are folded in) |

`rfdetr.config` is imported lazily, and only for validation. It pulls in torch and transformers (several seconds, once per server), the same pattern as `ultralytics.cfg.check_cfg` today. `options()` starts that import in a background thread so the first Validate is quick.

### `training.py`: where it changes
- **`validate_args` / `enqueue` / `copy_args`:** pick the framework from `model:` and delegate. `run.json` and `spec.json` get `framework` (+ `variant`).
- **`_data_plan`:** the "labels next to images inside `images/`" error and the `cache_files` list are Ultralytics-only. RF-DETR's view handles any layout, so colocated datasets can train with RF-DETR even where YOLO refuses. Class contiguity, a non-empty train split and a val split are still required.
- **`_launch`:** for RF-DETR, write `splits.json` instead of `data.yaml` + lists. The `test_eval` flag is unchanged.
- **`_summary` / `run_detail`:**
  - new fields: `framework`, `variant`, and `weights: {best, last}` (paths). `has_weights` is kept, derived from those paths;
  - `resumable` uses `framework.resume_checkpoint`;
  - for RF-DETR, args come from `spec.json` (there is no `args.yaml`).
- **`best_weights`:** keeps the real extension (`.pth`) in the download name.
- **`_started_from`:** compares against the framework's weights paths, not just `weights/`.
- **`options()`:**
  - adds `frameworks: [{id, label, available, error}]`;
  - gives defaults, help and the pretrained list per framework;
  - previous runs carry `framework`, so fine-tuning only offers same-framework weights.
- **Queue, lock, reattach, kill timer, delete:** no changes. `_process_alive` still matches `train_runner` on the cmdline, so the RF-DETR runner is entered through `train_runner` too.

### Runner
`train_runner.run()` reads `spec["framework"]`. For `rfdetr` it calls `rfdetr_runner.run(spec, reporter)` (new module, imported lazily). The file plumbing in `Reporter` is shared by both runners:
- `write`, `control`, `watch_kill`, the ETA from recent epoch times;
- appending an epoch row and tracking the best epoch.

The Ultralytics callbacks move into `UltralyticsCallbacks`; their behavior doesn't change.

`rfdetr_runner.run`:
1. **Build the view** from `splits.json` (`status: preparing`, `phase: preparing-data`, `done/total` in state.json). Read each image's EXIF header (`imaging.orientation`). Write upright copies for orientation ≠ 1 (JPEG q95 / PNG lossless); hardlink the rest.
2. **Build the model, config and trainer:**
   - the model is the variant class with `pretrain_weights` / `resolution` from args;
   - `TrainConfig` (or `SegmentationTrainConfig`) gets `dataset_dir=<view>`, `output_dir=<run dir>`, `class_names` = dataset names, and the device mapped to `accelerator` + `devices=[i]`;
   - then `module = RFDETRModelModule(...)`, `datamodule = RFDETRDataModule(...)`, `trainer = build_trainer(...)`, `trainer.callbacks.append(WorkbenchCallback(reporter))`.
3. **Fit:** `trainer.fit(module, datamodule, ckpt_path=<resume checkpoint or None>)`.
4. **Evaluate** (when completed or finished early): `evaluate(split="val", log_per_class_metrics=True)` on the best weights writes `val.json`, and `evaluate(split="test", …)` writes `test.json`. Class rows are `{Class, Instances, mAP50-95}`, with Instances counted from `splits.json`. RF-DETR has no per-class P/R. A failed test eval stays a warning.
5. **Clean up:** final status, as today; remove `workbench/data/`.

`WorkbenchCallback` (Lightning `Callback`):

| Hook | Does |
|---|---|
| `on_train_start` | `status: training`, `epochs`, `batches = trainer.num_training_batches` |
| `on_train_epoch_start` | resets the batch counter. Leaves the checkpoint window, raising `RunKilled` if a kill was deferred |
| `on_train_batch_end` (≤1/s) | `batch`, running losses from `trainer.progress_bar_metrics`, `gpu_mem`, ETA. A `finish` flag. **Kill check** (raise: a batch boundary, nothing being written) |
| `on_validation_batch_end` | kill check |
| `on_validation_end` | **enters the checkpoint window** (`critical`). `BestModelCallback` writes `.pth` files right after us, then `last.ckpt` at train-epoch end |
| `on_train_epoch_end` | reads `trainer.callback_metrics`, normalizes them, appends the epoch row (fitness, is_best, epoch_time, lr), updates best_epoch and ETA. On pause or finish, sets `trainer.should_stop = True` and records which |
| after `fit` returns | leaves the checkpoint window. The kill/pause/finish outcome decides the status |

Kill in the RF-DETR runner: the SIGTERM handler and the control.json watcher **only set a flag**. The callback raises at the next train or validation batch boundary outside the checkpoint window. This replaces the Ultralytics runner's raise-from-signal-handler, which was needed there because Ultralytics has no batch-level hook for it. The server's 30 s force-kill stays as the backstop.

Key normalization (RF-DETR → epoch row):

| RF-DETR | Row key |
|---|---|
| `val/mAP_50_95`, `val/mAP_50` | `metrics/mAP50-95(B)`, `metrics/mAP50(B)` |
| `val/precision`, `val/recall` | `metrics/precision(B)`, `metrics/recall(B)` |
| `val/segm_mAP_50_95`, `val/segm_mAP_50` | `metrics/mAP50-95(M)`, `metrics/mAP50(M)` |
| `val/mAP_75`, `val/mAR`, `val/F1`, `val/ema_mAP_50_95` | `extra/mAP75(B)`, `extra/mAR(B)`, `extra/F1(B)`, `extra/ema_mAP50-95(B)`: shown in the table, not in charts |
| `train/loss`, `train/loss_ce`, `train/loss_bbox`, `train/loss_giou`, `train/loss_mask_ce`, `train/loss_mask_dice` | `train/loss`, `train/ce_loss`, `train/bbox_loss`, `train/giou_loss`, `train/mask_ce_loss`, `train/mask_dice_loss` (`*_aux` terms dropped; they stay in `metrics.csv`) |
| `val/loss` (only if computed) | `val/loss` |
| `train/lr` | `lr/pg0` |

The runner also prints one line per epoch to `train.log`, since RF-DETR's progress bar is off by default and the log would otherwise be nearly silent.

### Predictor and Label QA (phase 3)
- **`RFDETRBackend` (`predictor.py`):**
  - **Load:** the variant class with `pretrain_weights=path`. The variant comes from the run's `run.json`, else from the checkpoint's `model_name` metadata via `RFDETR.from_checkpoint`.
  - **`names`:** from `model.class_names`. Fine-tuned models give a 0-based list. COCO-pretrained checkpoints use sparse COCO IDs, so `names` is a dict either way.
  - **`predict(path, conf)`:** `upright_rgb(path)` → `model.predict(image, threshold=conf)`. Then `Detections.with_nms(iou)`: DETR is NMS-free, but at Label QA's 0.05 floor it returns many overlapping queries. xyxy is normalized to `cx cy w h`. Masks become polygons through `drop_specks` + `mask_outlines`, moved out of `sam.py` into a shared `masks.py`.
- **Framework detection for a weights file:**
  1. the containing run's `run.json`;
  2. `.pth` means RF-DETR;
  3. for a `.pt`, look inside the zip's `data.pkl` for `ultralytics.` module references (a byte search, never unpickled). RF-DETR seg weights are `.pt` too.
- **`MODEL_EXTENSIONS`** gains `.pth`. `_trained_models()` lists RF-DETR runs through `framework.weights(run_dir)`.
- **Defaults:** RF-DETR scores are calibrated differently from YOLO's (`predict` defaults to 0.5). Loading an RF-DETR model without an explicit conf sets the predictor's conf to 0.5. Label QA keeps `QA_CONF`, but the NMS above keeps its findings sane. Check this on a real run before tuning further.

## API
No new endpoints. Changed payloads:
- `GET /training/options` gains `frameworks[]`, and `defaults` / `help` / `pretrained` / `managed` become per framework (`{ultralytics: …, rfdetr: …}`).
- Run summaries gain `framework`, `variant`, `weights{best,last}`.
- `POST /training/args/validate` and `/training/preflight` are unchanged; the framework comes from `model:`.

## Frontend
- **`trainform.js`:**
  - A **Framework** switch (Ultralytics YOLO | RF-DETR) at the top of Basics. It's shown for detection and segmentation when RF-DETR is available; otherwise it's disabled, with the install error.
  - Switching starts a fresh args object (name and notes kept). The last-used args are stored per framework.
  - `FIELDS` becomes per framework. RF-DETR groups:
    - **Basics:** model (variant picker with default resolution and downloaded flag, or a previous RF-DETR run → `pretrain_weights`), epochs, batch_size, grad_accum_steps, resolution (step 32/24/12 per variant), device, seed;
    - **Optimizer:** lr, lr_encoder, weight_decay, warmup_epochs, lr_scheduler (step/cosine), use_ema, drop_path;
    - **Stopping & checkpoints:** early_stopping, early_stopping_patience, early_stopping_min_delta, skip_best_epochs, eval_interval, checkpoint_interval, with a hint that periodic checkpoints are full Lightning checkpoints (hundreds of MB for Large);
    - **Advanced:** num_workers, amp_dtype, multi_scale, expanded_scales, clip_max_norm.
  - The Augmentation group is hidden for RF-DETR.
- **`training.js`:**
  - `LOSS_SLOTS` / `LOSS_ORDER` gain the RF-DETR terms;
  - `extra/*` columns go only in the table;
  - "best.pt" wording becomes "best weights";
  - **Use for labeling** uses `detail.weights[kind]` instead of building `…/weights/best.pt`;
  - the result tables already render whatever columns the class rows have;
  - the plots gallery hides when empty (RF-DETR writes no plots);
  - a framework badge in the list and the run header ("RF-DETR · Seg M").
- **`compare.js`:** runs from both frameworks can be compared on the shared `metrics/*` keys. The args diff compares each run against its own framework's defaults. A note when frameworks differ: the mAPs come from different evaluators (Ultralytics' vs RF-DETR's COCO-style), so close numbers are not a ranking.
- **`predictor.js`:** the IoU field's hint says it is duplicate suppression for RF-DETR.

## Spike first (before phase 1, on the GPU machine) — done on CPU, see *Implemented*
These are about half a day to settle, and they decide details above:
1. Custom Training API with a fresh `RFDETRNano` on a 20-image view: does it reproduce `train()`'s dataset-driven setup? That covers class-count alignment, the class-name check and `batch_size: auto`, done by private helpers inside `train()`. **If not**, fall back to calling `model.train(...)` with `build_trainer` wrapped (as `detr.py` sees it) to append our callback. That's one monkeypatch, isolated in `rfdetr_runner`.
2. Hook order: validation before `on_train_epoch_end`, and `ModelCheckpoint` subclasses (including `BestModelCallback`) moved to the end of the callbacks.
3. Kill: Lightning registers its own SIGTERM handler during `fit` on POSIX and stops at a batch boundary. Check how it and ours combine, whether it ends in `sys.exit(1)`, and that both map to `stopped`. On Windows, the watcher-flag path only.
4. That the hardlinked view passes RF-DETR's containment checks.
5. Whether model construction hits the Hugging Face hub (DINOv2) when offline.
6. Whether MPS training works on macOS. CUDA is the real target; CPU is enough for the smoke test.

## Phases
Each phase ships on its own, with tests green between phases.

1. **Training backend:**
   - the dependency;
   - `frameworks.py` (Ultralytics moved behind it, behavior identical);
   - RF-DETR validation/options;
   - `splits.json`;
   - `rfdetr_runner` with the view, callback, pause/finish/kill, resume, final val/test;
   - the `training.py` wiring.

   Runs work through raw YAML.
2. **Form and live view:**
   - the framework switch;
   - RF-DETR fields and model picker;
   - loss/metric display;
   - badges;
   - the compare note;
   - "use for labeling" via `weights`.
3. **Prediction:**
   - `RFDETRBackend` (`masks.py` shared with SAM);
   - framework detection;
   - `.pth` discovery;
   - conf defaults;
   - Label QA check on a real RF-DETR run.
4. **Later:**
   - discovery of external RF-DETR runs (`training_config.json` + `metrics.csv`);
   - a disk note for periodic checkpoints;
   - optional `rfdetr[augment]` with an `aug_config` editor;
   - export (ONNX);
   - an apples-to-apples "evaluate any run on a split" with one evaluator for both frameworks.

Rough size: phase 1 is the bulk (about 700–900 lines including tests); phase 2 about 300 lines of JS; phase 3 about 200 lines plus tests.

## Tests (temporary dirs only; no GPU or rfdetr needed except the opt-in smoke test)
- **Framework layer:**
  - `model:` decides the framework;
  - Ultralytics behavior is unchanged (the existing `test_training.py` passes untouched);
  - old runs without `framework` read as Ultralytics.
- **RF-DETR args:**
  - managed keys rejected with line numbers;
  - unknown keys get suggestions;
  - resolution divisibility per variant;
  - a seg variant on a detection dataset is rejected;
  - multi-device and `batch_size: auto` without CUDA are rejected;
  - copy strips defaults.

  The pydantic part is skipped when rfdetr isn't importable.
- **`splits.json` / view:**
  - unique names across multi-folder splits;
  - missing labels become background;
  - colocated layouts accepted;
  - EXIF-rotated images are re-encoded upright (a generated JPEG with orientation 6) and labels copied as-is;
  - hardlink → copy fallback (forced);
  - the view is removed at exit.
- **Runner unit:** `WorkbenchCallback` against a fake Lightning trainer. Covers:
  - key normalization;
  - `is_best` / fitness with and without EMA keys;
  - pause/finish set `should_stop`;
  - kill raises at batch boundaries and is deferred inside the checkpoint window;
  - ETA and throttling.
- **Results:**
  - `metrics.csv` fixtures with real 1.11 headers, folded per epoch, tolerating step rows and a partial last line;
  - `weights()` fallback order;
  - `resume_checkpoint()` picks `last.ckpt`, then the newest numbered checkpoint.
- **API:**
  - options per framework;
  - an RF-DETR run lifecycle with the fake launcher (queue, lock, pause → resume, delete);
  - the download name keeps `.pth`.
- **Predictor:**
  - a fake RF-DETR backend through `backend_factory`;
  - `.pt` framework sniffing on small fixture zips;
  - masks → polygons.
- **Smoke (`WORKBENCH_SLOW_TESTS=1`):** 1 epoch `rfdetr-nano` and `rfdetr-seg-nano` on CPU over a generated 4-image dataset. Checks `state.json`, `epochs.jsonl`, `val.json`, `test.json`, `checkpoint_best_total.pth`, and that the view is gone. Skipped if the weights can't be downloaded.

## Risks
- **API churn.** rfdetr went from 1.6 to 1.11 in six months, with the Lightning migration in between. The runner depends on the Custom Training API names (`RFDETRModelModule`, `RFDETRDataModule`, `build_trainer`) and on the logged metric keys. Mitigations: pin `<1.12`, the smoke test, and normalization in one table.
- **Disk.** Periodic full checkpoints at the default interval of 10 add up on long runs with Large models. Mitigation: shown in the form; run size is already in the run view.
- **Metric comparability.** See the Compare note. A fair cross-framework ranking needs the phase-4 single-evaluator option.
- **Dependency weight.** transformers 5, pytorch-lightning, torchmetrics, peft, roboflow and the COCO-eval packages add install time and size to every install, because everything is core here.

## Implemented

All phases except the *Later* list. rfdetr 1.11.2 is a core dependency (`rfdetr[train]>=1.11.2,<1.12`), detection XL/2XL are not offered, and predictions (phase 3) are in. Verified:
- **Unit and API tests:** `tests/test_rfdetr.py`.
- **Real CPU runs on macOS:** detection and segmentation smoke tests (`WORKBENCH_SLOW_TESTS=1`), plus pause → resume → finish early and kill on real subprocesses.
- **In the browser:** form, run view, compare, use for labeling, Label QA.

**Not yet verified:** a CUDA GPU (including `batch_size: auto`), Windows, and MPS.

What the spike settled, and how the build differs from the plan:
- **Training goes through `RFDETR.train()`, not the Custom Training API.**
  - `train()` does dataset-driven setup in private helpers: device mapping, resolution, `batch_size: auto`, class-count alignment, and `training_config.json`. Re-doing that would copy private code.
  - Instead, `rfdetr_runner.install_callback` wraps `rfdetr.training.build_trainer`, which `train()` and `evaluate()` import at call time. That appends `WorkbenchCallback` to every trainer they build.
  - Our callback ends up before Lightning's `ModelCheckpoint`s in each hook. So `trainer.should_stop` set in `on_train_epoch_end` still gets that epoch's `last.ckpt`.
- **No checkpoint window is needed for kill.**
  - The runner's SIGTERM handler and the control.json watcher only set a flag (`Reporter(raise_on_kill=False)`). The callback raises at train/val/test batch ends and at the next epoch start.
  - Lightning's own SIGTERM handler, composed with ours on POSIX, also stops only at batch boundaries. A real kill ends as `stopped` within a second and stays resumable from `last.ckpt`.
- **The server checks RF-DETR arguments against a static schema.**
  - `from rfdetr.config import TrainConfig` takes about 7 s and loads torch and transformers.
  - The schema is `frameworks._RF_FIELDS`, a copy of 1.11's TrainConfig. `test_schema_matches_the_installed_rfdetr` fails when an rfdetr upgrade changes it.
  - The runner still applies RF-DETR's pydantic validation when training starts.
- **`splits.json` holds the parsed annotations, not label paths.** The runner writes the labels itself, so lines the Validation tab flags as broken are left out, as the warning says, and the labels match what the editor shows.
- **Periodic full checkpoints are off by default.** The runner passes `checkpoint_interval = epochs + 1` unless the user sets one. Sizes measured on CPU:

  | File | Nano | Seg Nano |
  |---|---|---|
  | Pretrained weights | 352 MB | 128 MB |
  | `last.ckpt` | 461 MB | 508 MB |
  | `checkpoint_best_*.pth`, `last_ema.pth` | ~115–128 MB each | ~115–128 MB each |
- **The best epoch is marked by the runner.**
  - `is_best` means a strict improvement after `skip_best_epochs`; EMA's track starts above 0, as in `BestModelCallback`.
  - `training.best_epoch` and `compare.bestEpochRow` take the last marked epoch for RF-DETR. Ultralytics keeps "the last tie wins".
- **Any `rfdetr-…` model goes to RF-DETR's validator,** so a mistyped variant gets an RF-DETR error, not an Ultralytics one.
- **Phase-4 item done early: runs made with RF-DETR's own `train()` are listed.** They are recognized by `training_config.json` + `metrics.csv`; their epochs come from `metrics.csv`, folded per epoch, and their arguments from `training_config.json`.
- **`batch_size: auto` without a GPU** is an argument error for `device: cpu|mps`. When the device is auto, preflight rejects it if `nvidia-smi` finds no GPU.
- **`pi-heif` became a declared dependency.** Once Ultralytics is imported, it patches `Image.open` to load pi-heif on the first unreadable image, and pip-installs it if missing. It used to be in the venv outside the lockfile, and `uv add` removed it. That made an unreadable image crash the size probe whenever Ultralytics had been imported (`test_unreadable_image_issue_survives_reread` in the full suite).
- **Predictions:**
  - **Default threshold:** switching between YOLO and RF-DETR without an explicit `conf` sets the predictor's threshold to the backend's default (0.25 / 0.5).
  - **Discovery:** an RF-DETR run's other `.pth` files are not listed as "found" models next to its best weights.
  - **Label QA:** runs unchanged on RF-DETR weights.

