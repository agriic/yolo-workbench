# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                   # set up environment (uv is the standard here); all features (embeddings, predict, tests) are core dependencies
uv run pytest                             # run all tests
uv run pytest tests/test_dataset.py -k name_of_test   # run a single test
uv run yolo-workbench /path/to/dataset.yaml --category detection|segmentation [--no-browser]
```

The embeddings feature additionally requires a local MongoDB for FiftyOne (see README.md for the docker command and `~/.fiftyone/config.json`).

## Architecture

Local, single-user FastAPI app for annotating and validating YOLO datasets. Python backend in `src/yolo_workbench/`, vanilla-JS frontend served from `src/yolo_workbench/static/` with native ES modules and no Node.js/build step. `app.js` orchestrates the UI; shared state and API helpers live in `state.js`/`api.js`, virtualized galleries in `grid.js`, and the annotation editor, model-assisted labeling, and embedding renderer in `canvas.js`, `predictor.js`, and `embeddings.js`.

- `cli.py` — Typer entrypoint: loads a `Dataset`, starts uvicorn on localhost, opens the browser.
- `dataset.py` — the core. `Dataset` resolves the YOLO YAML (dirs, `.txt` file lists, colocated or parallel `images/`→`labels/` layouts), indexes every image with a stable sha1-based `record_id`, parses labels, and collects validation issues (malformed lines, unknown classes, out-of-range/zero-area/duplicate annotations, missing/orphan labels — some marked `fixable` and auto-repairable via `fix_issue`). All writes go through `_commit`: mtime-based conflict detection (`WriteConflict` → HTTP 409), one-time per-session backup under `<dataset>/.yolo-workbench/backups/<session-id>/`, atomic write (temp file + fsync + rename), and an undo/redo stack of transactions (each a list of before/after label-file contents, so bulk operations undo as one unit). Bulk operations (`fix_issues_bulk`, `bulk_edit_objects`) check all mtimes before writing any file — a conflict aborts the whole batch. Image dimensions are probed separately from label parsing: cached in `<dataset>/.yolo-workbench/index.json` (keyed by mtime_ns+size) and re-probed via a thread pool — in a background thread when `background_probe=True` (the CLI default), with progress in `Dataset.indexing` / the `indexing` field of `GET /api/v1/dataset`.
- `web.py` — `create_app(dataset)` builds the FastAPI app: JSON endpoints under `/api/v1` (images, annotations PUT, object crops, issues, history, embeddings) plus Pillow-rendered thumbnails/crops. `DatasetError` → 400, `WriteConflict` → 409.
- `media_cache.py` — disk cache for rendered thumbnails/crops under `~/.cache/yolo-workbench/media` (tests must pass a tmp dir via `create_app(dataset, media_cache=...)`). Keys hash image/label mtimes so edits self-invalidate; served with `Cache-Control: private, no-cache` + ETag (304 on revalidation); LRU-pruned by mtime past 512 MB. Bump `PALETTE_VERSION` when rendering or the palette changes.
- `predictor.py` — optional model-assisted labeling (ultralytics). `PredictorManager` on `app.state.predictor`: loads a .pt/.onnx model, auto-maps model class names to dataset classes case-insensitively (unmapped predictions are shown but not acceptable), and runs batch inference in a background thread. Predictions live in memory keyed by image id until accepted — acceptance merges them via `replace_annotations` so backup/undo/conflict machinery applies; rejection never touches disk. Tests inject a fake backend via `PredictorManager(dataset, backend_factory=...)` and `create_app(dataset, predictor=...)`. CLI can preload a model with `--model`.
- `embeddings.py` — optional FiftyOne/UMAP 3D patch-embedding visualization, computed in a background thread with status polled via `/api/v1/embeddings`; degrades to `status: "unavailable"` when fiftyone isn't installed.
- `models.py` — `Annotation` and `ImageRecord` dataclasses.

The category (`detection` = 4 points cx/cy/w/h, `segmentation` = polygon, ≥6 even-count points) is fixed at launch and branches validation, rendering, and geometry throughout backend and frontend. All coordinates are normalized 0..1.

The class-color `PALETTE` is duplicated in `web.py` and `static/api.js` — keep them in sync.

## Tests

Tests build isolated temporary datasets (`make_dataset` in `tests/test_dataset.py`) and must never modify a real dataset. API tests use httpx `ASGITransport` against `create_app` — no live server.
