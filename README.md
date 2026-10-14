# YOLO Dataset Workbench

Local browser-based tooling for YOLO detection and segmentation annotation, object exploration, relabeling, dataset statistics, and validation.

## Run

Python 3.10 or newer is required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test,embeddings]'
```
OR use UV
```bash
uv sync
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

### Projects mode

```
yolo-workbench --projects /path/to/projects
```

The projects page lists every project in the folder. Each project is one YOLO dataset that the tool creates, with `data.yaml`, `images/`, `labels/` (or `train/<class>/…` for classification), plus `models/` and `runs/` for trained models and training runs. From the UI you can:

- create, rename, and delete projects (deleted projects go to `<projects>/.trash/` and can be restored);
- adopt existing dataset folders that are dropped into the projects folder;
- add, rename, reorder, merge, and remove classes (all of these also work in single-dataset mode);
- import images by browser upload or from a local folder, with a choice of target split or a seeded random split, and class names mapped from the source's `data.yaml`/`classes.txt`;
- use several folders per split (`train: [images/train, images/extra]`): add a folder from the Splits panel, and pick the destination folder for each split when importing or moving images;
- select images in the Annotation grid with the checkboxes (Shift-click for a range) or by dragging a rectangle from the empty space between images (Shift/Ctrl adds to the selection, a click on empty space or Esc clears it), then move them between splits, or delete them into the dataset trash.

Only one project is open at a time. Class changes, image moves, and image deletes are undoable with Ctrl+Z. Imports are logged to `.yolo-workbench/imports.jsonl`.

### Training

The Training tab trains Ultralytics YOLO models on the open dataset (detection, segmentation or classification, following its category), in both modes.

- **New run:** a form for the common options (epochs, image size, batch, device, optimizer, augmentation, …), with a Raw YAML mode for any Ultralytics argument. The base model is an official pretrained one (YOLO26/YOLO11/YOLOv8, downloaded on first use to `~/.cache/yolo-workbench/weights/`) or a previous run's weights. **Duplicate** and **Fine-tune** prefill the form from an earlier run.
- **Live view:** current epoch and batch, ETA, the best epoch so far, loss and metric charts, and the per-epoch table the `yolo` CLI prints. **Pause** (resumable), **Finish early** (runs the final evaluation), or **Kill**.
- **Results:** final validation of `best.pt` and, when the dataset has a test split, a test-split evaluation, both per class, plus the Ultralytics plots and the log.
- **History and comparison:** every run in the dataset's `runs/` folder, including runs made with `yolo train`. Rename runs, add notes, and tick runs to compare their curves, results and the arguments that differ. **Use for labeling** loads a run's `best.pt` as the prediction model of the Annotation tab.
- **Queue:** runs are separate processes that keep going if the server stops; one runs at a time, the rest wait in a queue (shared by all projects in projects mode) that you can reorder. While a run trains, its dataset is read-only.

Runs are stored in `<project>/runs/<run>/`, or in single-dataset mode in `runs/` next to the dataset YAML. Delete a run by deleting its folder.

The application writes edited YOLO labels atomically. Before a label is first changed in a session, its original content is copied below `<dataset>/.yolo-workbench/backups/<session-id>/`.

## Tests

```bash
pytest
```

Tests create isolated temporary datasets and never modify the dataset passed to the application.
