  ### Highest-priority code improvements

  1. Prevent stale frontend requests from replacing current editor state.
     src/yolo_workbench/static/app.js:396 assign responses to global state after asynchronous requests without checking whether the user has since navigated or made another edit. Rapid navigation or overlapping saves can display the wrong image, lose
     a newer edit, and produce inaccurate undo history.

     Add request generations/AbortController for reads, serialize saves per image, and include an annotation revision in PUT requests for optimistic concurrency.

  2. Make undo preserve file existence.
     History records only text, and restoration always writes a file in src/yolo_workbench/dataset.py:289. Consequently, fixing a missing label and then undoing leaves an empty label instead of restoring the original “file absent” state. I reproduced
     this in an isolated dataset.

     Store before_exists/after_exists and delete the label during undo when appropriate.

  3. Reject non-finite geometry.
     src/yolo_workbench/dataset.py:268 checks ranges but not math.isfinite(). Pydantic accepts NaN, which currently writes a nan coordinate and immediately turns the label into a malformed file that the UI refuses to edit. Validate finite
     coordinates, confidence and IoU ranges at the API boundary.

  4. Fix incomplete orphan detection.
     src/yolo_workbench/dataset.py:233 scans only direct label directories. A nested orphan label is not reported; I reproduced that too. Scan the known label roots recursively.

  5. Remove blocking work from async endpoints.
     Model loading/inference and Pillow rendering happen synchronously inside async handlers in src/yolo_workbench/web.py:134. A slow image or prediction can freeze all API activity. Use FastAPI’s threadpool helpers or synchronous route functions.

  6. Eliminate repeated full-dataset scans.
     Metadata, object exploration, issues, and post-save refreshes repeatedly traverse the dataset. Every save launches three broad reloads from src/yolo_workbench/static/app.js:837. Maintain class/issue indexes, cache aggregates with invalidation,
     return affected summaries from mutations, and implement “has predictions” filtering server-side so pagination remains correct.

  7. Make background jobs resilient.
     One bad image aborts an entire prediction batch in src/yolo_workbench/predictor.py:285. Add per-image failures, cancellation, retry, and completed/failed counts. Embedding results should also carry a dataset revision or be invalidated after
     geometry/class changes.

  8. Split and test the frontend.
     The 1,710-line global-state app.js contains grid loading, canvas editing, model management, and 3D embeddings. Native ES modules would suffice—no build tool is required. Start with api/state, grid, editor, predictor, and embeddings, then add
     Playwright tests for rapid navigation, draw/save, save failure, undo/redo, and prediction review.

  9. Fix dependency and documentation boundaries.
     pyproject.toml:11 installs FiftyOne, Torch, Ultralytics, HTTPX, and pytest for every user, while the README.md:9 references extras that do not exist. Define base, predict, embeddings, and test extras. Also resolve the documented ONNX support
     versus .pt-only discovery mismatch.

  10. Protect the network-exposed mode.
     Loopback is a sensible default, but src/yolo_workbench/cli.py:21 can expose unauthenticated write and model-loading endpoints. Binding beyond loopback should require an explicit unsafe flag or token authentication, especially because loading .pt
     models is a sensitive operation.

  ### Best feature opportunities

  My preferred order would be:

  - Repair workspace: raw-line editor for malformed labels, unknown-class remapping, orphan inspection/deletion, and a preview before bulk fixes. Currently malformed content forces users outside the application.
  - Dataset statistics: class balance, annotations per image, box size/aspect distributions, split comparison, co-occurrence, and likely outliers. This should come before further embeddings polish because it is cheaper and useful on every dataset.
  - Prediction review queue: sort by confidence, unmapped class, or image; keyboard accept/reject; batch error reporting; and filters for low-confidence/uncertain images.
  - Split and export tooling: stratified train/validation/test management, filtered YOLO export, COCO conversion, and a dry-run change report. This requires extending transactions beyond label files, so it is a larger project.
  - Persistent sessions: optionally retain pending predictions, review position, filters, and bounded undo history across restarts.

  The existing PLAN.md:1 already identifies many worthwhile performance and maintainability items. I would move the request-race, faithful undo, finite-value validation, and recursive orphan issues ahead of that backlog.
