# YOLO Dataset Workbench

## Goal

Build a local, single-user web application for creating, inspecting, correcting, and validating YOLO detection and segmentation annotations. The application is launched with a dataset YAML path and category, binds to localhost, and writes annotation changes directly to the dataset using atomic files, backups, and undo/redo.

## Technology

- Python 3.12, FastAPI, Uvicorn, PyYAML, Pillow, Typer, and Pydantic.
- Browser UI using semantic HTML, modular CSS, and native JavaScript canvas APIs. The UI ships inside the Python package and requires no Node.js runtime.
- Pytest and HTTPX for backend tests; Playwright is reserved for browser acceptance tests.
- `uv` for local environments, dependency locking, and packaging.

## Features

- Resolve YOLO YAML paths, class names, train/val/test splits, list files, colocated labels, and parallel `images`/`labels` layouts.
- Annotation tab with image grid, split/class/search filters, canvas box or polygon creation and editing, relabeling, deletion, zoom/pan, and history.
- Exploration tab with class-filtered object crops, split filters, direct relabel/delete actions, and source-image navigation.
- Validation dashboard for malformed labels, invalid class IDs and geometry, duplicate annotations, missing labels, orphan labels, and unreadable images.
- Atomic label writes, per-session original backups, modification conflict detection, and session undo/redo.
- Thumbnail and crop caches outside the dataset.

## Interfaces

```text
yolo-workbench DATASET_YAML --category detection|segmentation
  [--host HOST] [--port PORT] [--no-browser]
```

The local service exposes versioned JSON endpoints under `/api/v1` and streams images, thumbnails, and crops through dedicated media endpoints.

## Acceptance

- Both category modes can create, reshape, relabel, and delete annotations.
- Saved edits survive reload and affect only the intended label file.
- Undo/redo restores exact file content and original backups remain available.
- Exploration changes update annotation state and crop results immediately.
- Invalid data is reported without preventing valid images from being edited.
- The pebble detection dataset resolves all train images and labels without modifying them during tests.

## Boundaries

- One category applies to a launched dataset; mixed annotation formats are unsupported.
- Dataset YAML, image pixels, class definitions, model-assisted labeling, training, and collaborative editing are outside v1.
- The server is intended for trusted localhost use and has no authentication.
