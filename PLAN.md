# Remaining improvements

## Easy wins (hours, not days)

1. Fix --model ONNX discovery mismatch — cli.py documents .pt/.onnx, but `MODEL_EXTENSIONS` only includes `.pt`, so ONNX models never show in the discovery dropdown.
2. Cap the undo stack and fix session-id collisions — the undo stack is unbounded in memory, and `session_id` is second-resolution, so two launches in the same second share a backup directory.
3. Small hygiene — dedupe the `KeyError` → 404 boilerplate across endpoints, use `JSONResponse` instead of hand-built JSON strings, and factor the repeated annotation-copy idiom.

## Medium effort, high payoff

4. Reduce the refetch storm after each save — every edit triggers `loadImages`, `loadObjects`, and `loadIssues`. Return enough updated state from mutations to patch the active image and affected views directly.
5. Single source of truth for the palette — it is duplicated in `web.py` and `app.js` with a manual synchronization rule. Serve it from the API or generate a shared module.
6. Frontend tests — canvas editing, predictor UI, and the embeddings scatter still lack browser coverage. Add a Playwright smoke suite for load, rapid navigation, drawing, serialized saves, conflicts, and undo.

## Larger investments

7. Split `app.js` into ES modules — one global state object holds the annotation canvas, predictor, embeddings renderer, and grid logic. Native state/canvas/predictor/embeddings/API modules would improve isolation without requiring a build step.
8. New feature: split management — move images between train/val/test, or auto-split with stratification through the existing transactional write path.
9. New feature: export/import — export the cleaned dataset or a filtered subset as YOLO, and convert to/from COCO.
10. Test the remaining seams — `cli.py`, the FiftyOne-backed `compute_gt_viz` path, and undo/redo conflict handling still lack coverage.

If I had to pick next, start with 1 + 2 + 3 for contained correctness and maintenance wins, then tackle 5 while splitting the frontend under 7.
