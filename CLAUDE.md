# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                   # set up environment (uv is the standard here); all features (embeddings, predict, tests) are core dependencies
uv run pytest                             # run all tests
uv run pytest tests/test_dataset.py -k name_of_test   # run a single test
uv run yolo-workbench /path/to/dataset.yaml --category detection|segmentation [--no-browser]
uv run yolo-workbench /path/to/dataset_dir --category classification   # <root>/<split>/<class>/ image folders
uv run yolo-workbench --projects /path/to/projects                        # projects mode: many datasets in one folder
```

The embeddings feature additionally requires a local MongoDB for FiftyOne (see README.md for the docker command and `~/.fiftyone/config.json`).

## Architecture

Local, single-user FastAPI app for annotating and validating YOLO datasets. Python backend in `src/yolo_workbench/`, vanilla-JS frontend served from `src/yolo_workbench/static/` with native ES modules and no Node.js/build step. `app.js` orchestrates the UI; shared state and API helpers live in `state.js`/`api.js`, virtualized galleries in `grid.js`, and the annotation editor, model-assisted labeling, and embedding renderer in `canvas.js`, `predictor.js`, and `embeddings.js`. `manage.js` is the Dataset tab (classes, import, splits); `projects.html`/`projects.js` is the projects list. Dataset API calls go through `api()`/`apiUrl()` with paths relative to the dataset base (`/api/v1` in single mode, `/api/v1/projects/<slug>` in projects mode); paths starting with `/api/` are absolute.

Two modes: **single-dataset** (`create_app(dataset)`, dataset API at `/api/v1`) and **projects** (`create_workspace_app(workspace)`, workspace API at `/api/v1/projects`, `/api/v1/trash`, and the same dataset router mounted at `/api/v1/projects/{slug}`). Both share `dataset_router()` in `web.py` (endpoints get a `DatasetSession` through a dependency) and `/api/v1/app` (mode) plus `/api/v1/jobs`.

- `cli.py` — Typer entrypoint: loads a `Dataset`, starts uvicorn on localhost, opens the browser.
- `dataset.py` — the core. `allow_empty=True` (projects) permits no images and no classes. Records can be added/removed at runtime (`_add_record`/`_remove_record`, `rescan()` diffs disk against the index). The undo stack holds transactions of mixed entries — `HistoryEntry` (label content), `FileEntry` (data.yaml), `MoveEntry` (classification class change), `RecordMoveEntry` (image + label to another split, to `.yolo-workbench/trash/`, or back), `DirEntry`/`DirMoveEntry` (class folders), `SchemaEntry` — replayed by `_run_transaction`, which checks every entry before touching anything and re-reads records at the end. New multi-file operations should build entries and call `_execute`. `delete_images`/`move_images` require `capabilities()["image_ops"]` (each split is one image directory). `Dataset` resolves the YOLO YAML (dirs, `.txt` file lists, colocated or parallel `images/`→`labels/` layouts), indexes every image with a stable sha1-based `record_id`, parses labels, and collects validation issues (malformed lines, unknown classes, out-of-range/zero-area/duplicate annotations, missing/orphan labels — some marked `fixable` and auto-repairable via `fix_issue`). All writes go through `_commit`: mtime-based conflict detection (`WriteConflict` → HTTP 409), one-time per-session backup under `<dataset>/.yolo-workbench/backups/<session-id>/`, atomic write (temp file + fsync + rename), and an undo/redo stack of transactions (each a list of before/after label-file contents, so bulk operations undo as one unit). Bulk operations (`fix_issues_bulk`, `bulk_edit_objects`) check all mtimes before writing any file — a conflict aborts the whole batch. Image dimensions are probed separately from label parsing: cached in `<dataset>/.yolo-workbench/index.json` (keyed by mtime_ns+size) and re-probed via a thread pool — in a background thread when `background_probe=True` (the CLI default), with progress in `Dataset.indexing` / the `indexing` field of `GET /api/v1/dataset`.
- `web.py` — `create_app(dataset)` builds the FastAPI app: JSON endpoints under `/api/v1` (images, annotations PUT, object crops, issues, history, embeddings) plus Pillow-rendered thumbnails/crops. `DatasetError` → 400, `WriteConflict` → 409.
- `media_cache.py` — disk cache for rendered thumbnails/crops under `~/.cache/yolo-workbench/media` (tests must pass a tmp dir via `create_app(dataset, media_cache=...)`). Keys hash image/label mtimes so edits self-invalidate; served with `Cache-Control: private, no-cache` + ETag (304 on revalidation); LRU-pruned by mtime past 512 MB. Bump `PALETTE_VERSION` when rendering or the palette changes.
- `predictor.py` — optional model-assisted labeling (ultralytics). `PredictorManager` on `app.state.predictor`: loads a .pt/.onnx model, auto-maps model class names to dataset classes case-insensitively (unmapped predictions are shown but not acceptable), and runs batch inference in a background thread. Predictions live in memory keyed by image id until accepted — acceptance merges them via `replace_annotations` so backup/undo/conflict machinery applies; rejection never touches disk. Tests inject a fake backend via `PredictorManager(dataset, backend_factory=...)` and `create_app(dataset, predictor=...)`. CLI can preload a model with `--model`.
- `embeddings.py` — optional FiftyOne/UMAP 3D patch-embedding visualization, computed in a background thread with status polled via `/api/v1/embeddings`; degrades to `status: "unavailable"` when fiftyone isn't installed.
- `models.py` — `Annotation` and `ImageRecord` dataclasses.
- `session.py` — `DatasetSession`: a `Dataset` plus its `PredictorManager` and `EmbeddingsManager`; `busy_reason()` blocks closing it while a job runs.
- `workspace.py` — projects mode. `Workspace(root)` lists, creates, updates, adopts, trashes (`<root>/.trash/`), restores, and purges projects (`<slug>/project.json` + always a `data.yaml` without `path:`, so Ultralytics resolves paths next to it). Exactly one project is open (`workspace.session(slug)` closes the previous one, or raises `WriteConflict` if it is busy). Card summaries and the cover (first image → `.yolo-workbench/cover.jpg`) are persisted in `project.json` so listing never loads a `Dataset`.
- `classes.py` — class schema operations (add, rename, remove as delete or merge, merge, reorder). Detection/segmentation compile to one transaction rewriting each label's class token (geometry text kept byte for byte) plus the YAML `names:` block (`write_names` edits only that block, keeping comments); classification compiles to folder/file moves. Each transaction ends in a `SchemaEntry`, so replay reloads names and schema listeners (predictor mapping, embeddings) get the old→new id mapping.
- `importer.py` — import from a local folder or a browser upload (`.yolo-workbench/staging/<id>/`, paths sanitized): `scan()` pairs images with labels, detects source splits and class names (`data.yaml`, `classes.txt`, `notes.json`), suggests a mapping by name; `run_import()` adds new classes, assigns splits (fixed / seeded ratios / keep source), skips exact duplicates, copies files, then `Dataset.rescan()`. Logged to `.yolo-workbench/imports.jsonl`; not on the undo stack.
- `jobs.py` — `JobManager` for background jobs with progress and cancel (import now; designed to fit subprocess training later). `JobManager(synchronous=True)` runs inline for tests.

The category (`detection` = 4 points cx/cy/w/h, `segmentation` = polygon, ≥6 even-count points, `classification` = no points) is fixed at launch and branches validation, rendering, and geometry throughout backend and frontend. All coordinates are normalized 0..1. Classification datasets are directories (`<root>/<split>/<class name>/images`, class ids = sorted class-dir names); there are no label files — the class edit is a file move committed via `_commit_moves` (`MoveEntry` history entries, same undo/redo stack, no backups since moves are non-destructive), and the predictor reports `unavailable`. Embeddings work for classification too: `compute_gt_viz` embeds whole images (`fo.Classification`, no `patches_field`) instead of object patches.

The class-color `PALETTE` is duplicated in `web.py` and `static/api.js` — keep them in sync.

The CLI enables two request guards (off by default in `create_app`/`create_workspace_app`, so tests need not send them): a Host allow-list (DNS rebinding) and a required `X-Workbench: 1` header on mutating `/api/` requests (cross-site forms/uploads). `api()` in `api.js` always sends the header.

## Tests

Tests build isolated temporary datasets (`make_dataset` in `tests/test_dataset.py`) and must never modify a real dataset. API tests use httpx `ASGITransport` against `create_app` — no live server.
