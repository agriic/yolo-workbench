# Remaining priorities

## Highest-priority code improvements

1. Split and test the frontend. Move the global-state app into native modules for API/state, grid, editor, predictor, and embeddings; add Playwright coverage for rapid navigation, serialized saves, save conflicts, undo/redo, and prediction review.
2. Fix dependency and documentation boundaries. Define base, `predict`, `embeddings`, and `test` extras, then align the README and ONNX model-discovery behavior with those packages.
3. Protect network-exposed mode. Binding beyond loopback should require an explicit unsafe flag or token authentication because the API can edit labels and load model files.

## Best feature opportunities

- Repair workspace: raw-line editing for malformed labels, unknown-class remapping, orphan inspection/deletion, and previews before bulk fixes.
- Prediction review queue: sort by confidence, unmapped class, or image; keyboard accept/reject; and batch selection across images.
- Split and export tooling: stratified train/validation/test management, filtered YOLO export, COCO conversion, and a dry-run change report.
- Persistent sessions: optionally retain pending predictions, review position, filters, and bounded undo history across restarts.
