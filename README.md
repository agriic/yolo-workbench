# YOLO Dataset Workbench

Local browser-based tooling for YOLO detection and segmentation annotation, object exploration, relabeling, and dataset validation.

## Run

Python 3.10 or newer is required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test,embeddings]'
```
OR use UV
```bash
uv sync --extra embeddings --extra test
```

If you want to use embedings - 
Running for the first time, create MongoDB and its configuration
```bash
docker run -d --name fiftyone-mongo --restart unless-stopped -p 127.0.0.1:27017:27017 -v fiftyone-mongo-data:/data/db mongo:7 2>&1
```
$ cat ~/.fiftyone/config.json
```json
{
  "database_uri": "mongodb://localhost:27017"
}
```

```
yolo-workbench /path/to/dataset.yaml --category detection
```

Open `http://127.0.0.1:8765` if the browser does not open automatically. Use `--category segmentation` for polygon datasets and `--no-browser` in headless environments.

The application writes edited YOLO labels atomically. Before a label is first changed in a session, its original content is copied below `<dataset>/.yolo-workbench/backups/<session-id>/`.

## Tests

```bash
pytest
```

Tests create isolated temporary datasets and never modify the dataset passed to the application.

