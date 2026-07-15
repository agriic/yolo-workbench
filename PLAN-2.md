# Remaining priorities

## Highest-priority code improvements

1. Remove blocking work from async endpoints. Model loading/inference and Pillow rendering happen synchronously inside async handlers. Use FastAPI threadpool helpers or synchronous route functions so slow work does not freeze other API activity.
2. Eliminate repeated full-dataset scans. Maintain class/issue indexes, cache aggregates with invalidation, refresh only affected records after mutations, and implement the predictions filter server-side so counts and pagination remain correct.
3. Make background jobs resilient. Report per-image failures instead of aborting a prediction batch, add cancellation/retry and completed/failed counts, and invalidate embedding results when annotation geometry changes.
4. Split and test the frontend. Move the global-state app into native modules for API/state, grid, editor, predictor, and embeddings; add Playwright coverage for rapid navigation, serialized saves, save conflicts, undo/redo, and prediction review.
5. Fix dependency and documentation boundaries. Define base, `predict`, `embeddings`, and `test` extras, then align the README and ONNX model-discovery behavior with those packages.
6. Protect network-exposed mode. Binding beyond loopback should require an explicit unsafe flag or token authentication because the API can edit labels and load model files.

## Best feature opportunities

- Repair workspace: raw-line editing for malformed labels, unknown-class remapping, orphan inspection/deletion, and previews before bulk fixes.
- Dataset statistics: class balance, annotations per image, box size/aspect distributions, split comparison, co-occurrence, and likely outliers.
- Prediction review queue: sort by confidence, unmapped class, or image; keyboard accept/reject; batch error reporting; and low-confidence filters.
- Split and export tooling: stratified train/validation/test management, filtered YOLO export, COCO conversion, and a dry-run change report.
- Persistent sessions: optionally retain pending predictions, review position, filters, and bounded undo history across restarts.
