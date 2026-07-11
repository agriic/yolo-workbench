# YOLO Dataset Workbench

Local browser-based tooling for YOLO detection and segmentation annotation, object exploration, relabeling, and dataset validation.

## Run

Python 3.10 or newer is required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
yolo-workbench /path/to/dataset.yaml --category detection
```

Open `http://127.0.0.1:8765` if the browser does not open automatically. Use `--category segmentation` for polygon datasets and `--no-browser` in headless environments.

The application writes edited YOLO labels atomically. Before a label is first changed in a session, its original content is copied below `<dataset>/.yolo-workbench/backups/<session-id>/`.

## Tests

```bash
pytest
```

Tests create isolated temporary datasets and never modify the dataset passed to the application.
