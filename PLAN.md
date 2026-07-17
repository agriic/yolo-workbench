Maintainability plan

1. Split dataset.py (865 lines, one class doing five jobs). Dataset currently owns YAML/path resolution, image indexing, validation, statistics, persistence/undo, and bulk operations. Highest-value extractions, in order:
- statistics.py — statistics(), _build_statistics, _percentile, _distribution_summary, _iqr_fences, _statistics_outliers (~220 lines that only read records).
- validation.py — issue detection (_read_record's issue logic, _issue, _zero_area) and repair (fix_issue, fix_issues_bulk, _repair).
- store.py — _commit_many, _backup, _atomic_write, history, the undo/redo stacks. This is the most safety-critical code and deserves to be testable in isolation.

2. Give the dict payloads types. Issues, statistics rows, object-index items, and predictions are all string-keyed dicts built ad hoc (dataset.py:274, predictor.py:353). A typo in a key is a runtime bug today. Introduce TypedDict/dataclasses for Issue, Prediction, ObjectItem and return .as_dict() at the API boundary only. This also becomes de facto API documentation.

3. Add tooling — currently there is none. No ruff, no mypy, no CI. Concretely: ruff (lint+format), mypy on src/, and a GitHub Actions workflow running uv run pytest. The codebase is already type-annotated, so mypy adoption is cheap now and expensive later.

4. Fix the dependency layout in pyproject.toml. torch, torchvision, fiftyone, umap-learn, and ultralytics are hard dependencies of a Pillow+FastAPI annotation tool — that's a multi-GB install for someone who just wants to fix labels. The code already degrades gracefully (ultralytics_available(), fiftyone_available()), so move them to extras (yolo-workbench[predict,embeddings]). Also pytest and httpx are shipped as runtime dependencies; they belong in a [dependency-groups] dev group.

5. Tighten the locking story in Dataset. The mtime check in _commit_many protects the files, but in-memory state has races: fix_issues_bulk and bulk_edit_objects iterate _issue_index and call _prepare without holding _lock, while the background probe thread mutates the same indexes via _refresh_record_indexes. Also history() pops the undo stack outside the lock. Either take _lock at every public entry point, or document the invariant explicitly. While there: history() at dataset.py:748 finds records by linear scan over all images per entry — build a label_path → record map.

6. Kill the palette duplication. The comment discipline is good, but PALETTE in web.py:23 / api.js:15 plus PALETTE_VERSION living in a third file (media_cache.py) is a sync bug waiting to happen. Serve the palette in GET /api/v1/dataset metadata and have the frontend read it from there; derive PALETTE_VERSION from a hash of the palette.

7. Break up app.js's bind() (~220 lines) into per-feature functions (bindEditor, bindPredictor, bindEmbeddings, …). The configureCanvas/configurePredictor dependency-injection objects are a hand-rolled workaround for circular imports — a small shared event emitter (or just moving toast/statisticsChanged into their own module) would remove ~40 lines of wiring and make the module graph legible.

8. Test coverage gaps. ~3,200 lines of frontend JS have zero tests — the save-queue/revision logic in canvas.js:492 is the most intricate code in the app and only exercised manually. A Playwright smoke test (open editor, draw box, undo, verify label file) would catch most regressions. On the backend, media_cache pruning and the history() conflict path are untested.

Small correctness/consistency items found along the way:
- cli.py:24 says --model accepts .pt/.onnx, but MODEL_EXTENSIONS = {".pt"} in predictor.py:18; discovery and docs disagree.
- canvas.js:67 cache-busts the full-size image with Date.now(), defeating browser caching entirely on every editor open — use the image mtime (already available server-side) or an ETag like the thumbnails.
- statistics() computes under _lock, blocking all writes for the duration on large datasets — snapshot then compute outside the lock.
- The size index (index.json) is only saved when the whole probe finishes; killing the app mid-probe of a 100k-image dataset loses all progress. Save periodically.

Obvious missing features

Things a user of a "YOLO dataset workbench" will reach for and not find:

- Class management — no way to add, rename, or merge classes (i.e., edit YAML names and remap IDs across label files). For a dataset-repair tool this is the biggest gap; "merge class 3 into class 1" is a canonical cleanup task.
- Dataset watching / external-change pickup — images or labels added on disk after launch are invisible until restart; external label edits only surface as 409s. A rescan button would be cheap; a file watcher better.
- Export / subset creation — no way to export a cleaned copy, a filtered subset, or convert to/from COCO. Also no split management (move images between train/val, generate a split).
- Image-level actions — you can't delete a bad image from the dataset or mark an image as reviewed/done. A triage state ("needs review / approved") is standard in annotation tools.
- Grid filters for the workflows you already support — the gallery can't filter by "has issues" or "unlabeled only", even though the issues tab and the predictor (only_unlabeled) both know these sets.
- Cross-split leakage detection — duplicate images between train and val (content hash; you already hash paths). The embeddings feature makes near-duplicate detection a natural extension too.
- Prediction persistence — pending predictions are memory-only; a long-running assisted-labeling session dies with the server.
- Backup management — backups accumulate under .yolo-workbench/backups/<session>/ forever with no restore UI or retention policy.
- Other YOLO task types — OBB, pose/keypoints, and classification datasets are unsupported; the Category branching is already centralized enough that OBB would slot in.

- Other YOLO formats - separate images labels. Autodetect it.
