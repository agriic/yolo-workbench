1. Model-assisted labeling

Goal: load an Ultralytics/ONNX model, run inference on demand, show predictions as ghost annotations the user accepts/rejects/adjusts.

Dependencies: new optional extra predict = ["ultralytics>=8"] (pulls torch, which the embeddings extra already needs). Ultralytics handles both detection and segmentation and reads .pt and .onnx, so one integration covers both categories.

Backend — new predictor.py (mirrors the EmbeddingsManager pattern):

- PredictorManager(dataset) held on app.state. Lazy model load, status machine unavailable | idle | loading | ready | error, thread lock, background thread for batch runs.
- Predictions are never written to label files directly — they live in memory keyed by image_id until the user accepts them. This keeps the existing backup/undo/conflict machinery untouched: acceptance goes through the existing replace_annotations.
- Class mapping: model class names vs dataset names won't always match. On load, auto-map by case-insensitive name match, expose the mapping in the API, and let the UI remap or drop unmapped classes. Unmapped predictions are shown but not acceptable.

API (web.py):

┌──────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────┐
│                             Endpoint                             │                                           Purpose                                            │
├──────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
│ POST /api/v1/predictor/load {path, conf, iou}                    │ load model, return names + auto class-mapping                                                │
├──────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
│ GET /api/v1/predictor                                            │ status, model info, mapping                                                                  │
├──────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
│ PUT /api/v1/predictor/mapping                                    │ adjust class mapping                                                                         │
├──────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
│ POST /api/v1/predictor/run {image_ids?, split?, only_unlabeled?} │ start background batch; returns job status                                                   │
├──────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
│ GET /api/v1/images/{id}/predictions                              │ pending predictions for one image                                                            │
├──────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
│ POST /api/v1/images/{id}/predictions/accept {prediction_ids?}    │ merges chosen predictions into annotations via replace_annotations, clears them from pending │
└──────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────┘

Frontend (app.js / editor):

- Predictions rendered as dashed outlines with confidence badges, visually distinct from real annotations; a confidence slider filters client-side.
- Per-prediction: click → accept (becomes editable annotation) or x → reject. Toolbar: "Accept all ≥ threshold" for the image.
- Grid: badge on tiles that have pending predictions; filter "has predictions".

Model path input: simplest v1 is a text field for a filesystem path (the server is local, same trust model as the dataset path) plus an optional --model CLI flag.

Tests: fake predictor injected in place of Ultralytics (same trick as embeddings' availability check) — test mapping logic, accept-merge flow through replace_annotations, and that rejected predictions never touch disk.

Steps: ① predictor manager + load/run with fake backend → ② accept/reject API → ③ editor UI → ④ batch run + grid integration → ⑤ real Ultralytics smoke test. Roughly 4–6 sessions of work.

---
2. Performance on real datasets

Three independent sub-fixes, in order of pain:

a) Startup indexing.
- Split _read_record into label parsing (fast, keep eager) and image-size probing via Pillow (Image.open — the slow part).
- Add a persistent index at <dataset>/.yolo-workbench/index.json: {path: {mtime_ns, size, width, height}}. On startup, reuse dimensions when (mtime, size) match; probe only new/changed files, in a ThreadPoolExecutor (Pillow header reads release the GIL on I/O; 8–16 workers is fine).
- For files still unknown at bind time: don't block. Start the server immediately, finish probing in a backgress in GET /api/v1/dataset (indexing: {done, total}) so the UI can show a progress bar. Records without
dimensions yet just render thumbnails lazily anyway.

b) Thumbnail/crop disk cache.
- New media_cache.py: cache dir under platformdirs user-cache (or ~/.cache/yolo-workbench/), outside the data
- Key = sha1 of (image path, image mtime_ns, size, annotated-flag, label mtime_ns if annotated, palette versiakes annotated thumbnails self-invalidate after edits — no explicit invalidation hooks needed.
- render_thumbnail/render_crop become cache-through. Serve with Cache-Control: private, max-age=… plus an ETacing today's no-store.- Size cap: simple LRU-by-atime prune when the cache directory exceeds N MB, run opportunistically.
c) Query paths.                                                                                                                                                                                  - Precompute per-record class_ids: set[int] and a casefolded name (refresh in _read_record) so list_images fiator scans over annotations.
- Maintain a class_id → [(record, annotation)] index for /api/v1/objects instead of the current full scan; reby _commit). At 100k objects this matters; below that, the precomputed sets alone are probably enough — measure first, index only if needed.

Tests: index-cache hit/miss on mtime change, thumbnail cache invalidation after an annotation edit, server reetes.

Steps: ① thumbnail cache (biggest perceived win, self-contained) → ② index cache + parallel probe → ③ non-bloquery indexes if profiling justifies. Each step ships independently.

---
3. Bulk operations
Backend — multi-file history first. The undo stack holds single-file HistoryEntry items; a bulk fix must undoedo to hold list[HistoryEntry] transactions:

- New Dataset._commit_many(changes: list[(record, content, before)]): takes the lock once, checks all mtimes h on any conflict — no partial writes), then backs up + atomically writes each file, pushes one transaction,re-reads records. Existing _commit becomes a one-element wrapper, so history() and single edits keep working
- history() replays a transaction's entries in reverse order.

Bulk fix endpoint: POST /api/v1/issues/fix-bulk {kind, split?, issue_ids?} for the fixable kinds (missing_labduplicate). Implementation groups issues per record, applies the same per-kind repairs as fix_issue but builds
each record's final annotation list once (a record can have several issues), then commits via _commit_many. Rue_id, reason}]}.

Bulk object operations: POST /api/v1/objects/bulk {operations: [{image_id, annotation_id, action: "relabel"|"grouping-per-record strategy, one transaction. This is what the exploration tab and (later) embeddings
lasso-select call.

Frontend:

- Validation tab: per-kind "Fix all N" button (confirm dialog stating file count), grouped issue display by k
- Exploration tab: checkbox selection on crop cards + shift-click range select; action bar appears with "Rela the bulk endpoint, refreshes crops.
- Global toast after bulk ops: "Fixed 42 issues across 17 files — Undo", wired to the existing history endpoie transaction).

Tests: bulk fix touching multiple files undone by a single undo; conflict on one file aborts the whole batch -issue records repaired correctly in one pass.

Steps: ① transactional history refactor (small, pure backend) → ② fix-bulk + tests → ③ objects/bulk → ④ UI fosmallest of the three — the history refactor is the only delicate part.
