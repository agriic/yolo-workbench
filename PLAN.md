# Remaining improvements

## Easy wins (hours, not days)

1. Fix --model ONNX discovery mismatch — cli.py:24 documents .pt/.onnx, but MODEL_EXTENSIONS = {".pt"} in predictor.py:18, so ONNX models never show in the discovery dropdown. One-line fix.
2. Stop recomputing metadata on every poll — GET /api/v1/dataset recomputes split_counts and runs the full issues() aggregation on every call (web.py:111-114), and the indexing UI polls it every second. Cache and invalidate on write.
3. Cap the undo stack and fix session-id collisions — the undo stack is unbounded in memory (dataset.py:47), and session_id is second-resolution, so two launches in the same second share a backup dir (dataset.py:49).
4. Predictor batch resilience — one failing image aborts the whole batch and leaves job.state="error" with partial results (predictor.py:289-293). Skip-and-report-per-image is a small change with real UX payoff.
5. Small hygiene — dedupe the except KeyError → 404 boilerplate across ~9 endpoints, use JSONResponse instead of hand-built JSON strings (web.py:103-107), and factor the annotation-copy idiom repeated at dataset.py:367,400,423.

## Medium effort, high payoff

6. Per-class object index — GET /api/v1/objects iterates every image and annotation on every request (web.py:155); this was explicitly deferred in PLAN-2.md. A class_id → [(record, annotation)] index maintained on write would make the Exploration tab scale.
7. Reduce refetch storm after each save — every edit triggers loadImages, loadObjects, and loadIssues, each a full server scan (app.js:857-859). Return updated state from the PUT, or refresh only the affected image.
8. Move blocking work off the event loop — predict_image and Pillow thumbnail rendering run synchronously inside async def handlers (predictor.py:312, web.py:279), freezing the server during inference. Wrap in run_in_executor.
9. Server-side "has predictions" filter — the current filter is client-side over the fetched page, so counts and pagination are wrong when active (app.js:367-370).
10. Single source of truth for the palette — it's duplicated in web.py:20 and app.js:15 with a manual sync rule in CLAUDE.md. Serve it from an endpoint (or generate palette.js from Python) and delete the hazard.
11. Frontend tests — the entire app.js (all canvas editing, predictor UI, embeddings scatter) has zero tests. Even a Playwright smoke suite (load, rapid navigation, draw box, serialized save, undo) would protect the riskiest code.

## Larger investments

12. Split app.js into ES modules — one file with a single global state object holds annotation canvas, predictor, embeddings 3D renderer, and grid logic. Splitting into state / canvas / predictor / embeddings / api modules (still no build step needed with native ES modules) is the enabler for most future frontend work.
13. New feature: dataset statistics tab — class balance histograms, box size/aspect distributions, images-per-split, annotation density. Cheap to compute from data you already index, and very useful for dataset quality work.
14. New feature: split management — move images between train/val/test, or auto-split with stratification. Natural fit since all writes already go through the _commit backup/undo machinery.
15. New feature: export/import — export the cleaned dataset (or a filtered subset) as a new YOLO dataset, or convert to/from COCO. Turns the tool from a validator into a pipeline step.
16. Test the untested seams — cli.py, the embeddings compute_gt_viz path, undo/redo conflict handling, and the predictor batch-error path all have no coverage.

If I had to pick, the highest value-per-effort cluster is 6 + 7 + 2 (the scan-per-request performance issues, which compound on large datasets), followed by 10 (palette dedup) as a quick maintainability win.
