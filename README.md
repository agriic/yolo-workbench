# YOLO Dataset Workbench

A local, browser-based workbench for YOLO datasets: annotate, explore, validate and repair labels, manage classes and splits, import images, and train Ultralytics models — for **detection**, **segmentation** and **classification** datasets.

It runs as a small web server on your machine (FastAPI + a plain-JavaScript frontend, no Node.js or build step) and edits your dataset files in place, safely: every write is atomic, conflict-checked, backed up once per session, and undoable.

Works on Linux and Windows (and should on macOS). Python 3.10 or newer.

## Install

[uv](https://docs.astral.sh/uv/) is the supported way to set up the environment. 

Windows:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Linux/OSX:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
(Restart the shell..)

Everything (model-assisted labeling, embeddings, training, tests) is a core dependency, so one command installs it all:

```bash
uv sync
```

The first sync is large because it includes PyTorch. On Linux the PyPI wheels already bundle CUDA. On Windows, `uv sync` takes torch and torchvision from the PyTorch CUDA 13.0 index (configured in `pyproject.toml`), because PyPI's Windows wheels are CPU-only. That needs a recent NVIDIA driver; without an NVIDIA GPU everything still runs on the CPU.

Without uv, a plain virtual environment also works (on Windows, install the CUDA build of torch from [pytorch.org](https://pytorch.org/get-started/locally/) first if you want GPU training):

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

## Run

**Single dataset** — open one existing dataset:

```bash
uv run yolo-workbench path/to/data.yaml --category detection
uv run yolo-workbench path/to/data.yaml --category segmentation
uv run yolo-workbench path/to/dataset_dir --category classification
```

**Projects** — a folder of datasets that the workbench creates and manages:

```bash
uv run yolo-workbench --projects path/to/projects
```

The browser opens `http://127.0.0.1:8765` automatically. Options:

| Option | |
| --- | --- |
| `--no-browser` | do not open a browser (headless machines) |
| `--port 8765` | port to listen on |
| `--host 127.0.0.1` | address to bind; `0.0.0.0` makes it reachable from your network (there is no authentication — only do this on a trusted network) |
| `--model best.pt` | preload a model for assisted labeling (single-dataset mode) |

Windows paths work as usual, e.g. `uv run yolo-workbench D:\data\cars\data.yaml --category detection`.

### Dataset formats

- **Detection / segmentation:** a YOLO `data.yaml` whose splits point at image folders, `.txt` image lists, or lists of folders (`train: [images/a, images/b]`). Labels are found the way Ultralytics finds them: `images/` → `labels/` in a parallel tree, or next to the images. Detection labels are `class cx cy w h`; segmentation labels are polygons. All coordinates are normalized.
- **Classification:** a directory `<root>/<split>/<class name>/*.jpg` (no label files; class ids follow the sorted class folder names).

Images: `.jpg .jpeg .png .webp .bmp .tif .tiff`.

## What you can do

- **Annotation** — a virtualized gallery of the whole dataset with filters (split, class, review status, validation issues, object count, search) and a canvas editor for boxes and polygons (keyboard shortcuts are listed in the editor). Select images with checkboxes (Shift-click for a range) or by dragging a rectangle across the gallery, then mark them reviewed, move them to another split or folder, or delete them into the dataset trash.
- **Review passes** — mark images **done** or **needs work** (D / W in the editor marks and moves to the next image; U jumps to the next unreviewed one). With the gallery's Review filter on *Unreviewed*, reviewed images drop out as you go. Statuses are kept in `.yolo-workbench/review.json`, follow images that move, and are not part of undo.
- **Model-assisted labeling** — load an Ultralytics `.pt`/`.onnx` model (or a run's `best.pt`), predict on selected images, and accept or reject predictions individually or above a confidence threshold. Model classes are mapped to dataset classes by name; predictions stay in memory until accepted.
- **Smart annotation** — in the editor, turn on **Smart annotation** (S) and click an object: [SAM 2.1](https://docs.ultralytics.com/models/sam-2/) outlines it. Right-click (or Shift+click) marks background, dragging gives a rough box, ⏎ adds the result in the draw class (a box for detection, a polygon for segmentation). Uses SAM 2.1 large on a GPU and base otherwise (or when large fails to load); the weights download on first use. Once you have used it, images are encoded as soon as they open so the first click is instant.
- **Exploration** — every object as a crop, filterable by class and split; select objects to relabel or delete them in bulk.
- **Statistics** — class balance, annotations per image, box sizes, aspect ratios, class co-occurrence, split comparison, and likely outliers to review.
- **Embeddings** — a 3D UMAP of object patches (whole images for classification), computed with FiftyOne; see [Embeddings](#embeddings) for the one-time setup.
- **Validation** — malformed or unreadable labels, unknown classes, out-of-range, zero-area and duplicate annotations, missing labels, unreadable images, and orphan label files. Most are fixable one by one or in bulk: malformed lines can be removed or repaired in a raw label-file editor (also in the editor's inspector), unknown class ids can be mapped to a class, added as a new class, or deleted, and orphan label files go to the trash. **Show images** filters the gallery to one kind. Every fix is undoable.
- **Label QA** — run a trained model (a training run's `best.pt` works well) over the dataset and list where it disagrees with the labels, most suspicious first: objects nobody labelled, wrong classes, loose boxes, and labels the model cannot find. Each finding shows the label and the model's box side by side and has a one-click fix (add the label, take the model's class or box, delete the label) or **Dismiss** when the label is right. Findings are recomputed from the current labels, so fixing something elsewhere clears it too; results survive a restart in `.yolo-workbench/labelqa.json`.
- **Dataset** — add, rename, reorder, merge and remove classes (labels are rewritten, `data.yaml` keeps its comments); manage splits and add folders to a split; see how much the trash and label backups take per session, and restore or delete earlier sessions' files; **import** images by browser upload or from a local folder, with labels, class mapping by name, a fixed or seeded random split, duplicate skipping, and an optional crop (labels are clipped to it).
- **Training** — see [Training](#training).

Edits, class changes, moves and deletes are undoable with Ctrl+Z / Ctrl+Shift+Z. When a file changed on disk since it was loaded, the save is refused instead of overwriting it; **Rescan** picks up changes made outside the app.

### Projects mode

The projects page lists every project in the folder. A project is a YOLO dataset the tool creates: `project.json`, `data.yaml`, `images/<split>/` and `labels/<split>/` (or `<split>/<class>/` for classification), plus `models/` and `runs/`. From the page you can create, rename and delete projects (deleted ones go to `<projects>/.trash/` and can be restored), and adopt existing dataset folders copied into the projects folder. One project is open at a time.

### Training

The Training tab trains Ultralytics YOLO models on the open dataset (detect, segment or classify, following its category), in both modes.

- **New run:** a form for the common options (epochs, image size, batch, device, optimizer, …) and a Raw YAML mode for any Ultralytics argument. The **Augmentation** group previews each augmentation at both ends of its range on your own images, using Ultralytics' own transforms. The base model is an official pretrained one (YOLO26 / YOLO11 / YOLOv8, downloaded on first use to `~/.cache/yolo-workbench/weights/`) or a previous run's weights. **Duplicate** and **Fine-tune** prefill the form from an earlier run.
- **Live view:** epoch and batch progress, ETA, best epoch so far, loss and metric charts, and the per-epoch table. **Pause** (resumable), **Finish early** (runs the final evaluation), or **Kill**.
- **Results:** final validation of `best.pt` and, when the dataset has a test split, a test-split evaluation, both per class, plus the Ultralytics plots and the log.
- **History and comparison:** every run in the dataset's `runs/` folder, including runs made with `yolo train`. Rename runs, add notes, and tick runs to compare their curves, results, and the arguments that differ. **Use for labeling** loads a run's `best.pt` as the Annotation tab's model; **Download best.pt** saves it as `<dataset>-<run>-<imgsz>-<model>-<YYYY.MM.DD>.pt` (the date `best.pt` was written; a fine-tune names the pretrained model it descends from).
- **Queue:** a run is a separate process that keeps going if the server stops (the server picks it up again on restart). One run trains at a time; the rest wait in a reorderable queue, shared by all projects in projects mode. While a run trains, its dataset is read-only.

Runs are stored in `<project>/runs/<run>/`, or in single-dataset mode in `runs/` next to the dataset YAML. Delete a run by deleting its folder.

### Embeddings

Embeddings need a MongoDB for FiftyOne. The simplest is a local Docker container (Docker Desktop on Windows), created once:

```bash
docker run -d --name fiftyone-mongo --restart unless-stopped -p 127.0.0.1:27017:27017 -v fiftyone-mongo-data:/data/db mongo:7
```

and pointed at from `~/.fiftyone/config.json` (on Windows `%USERPROFILE%\.fiftyone\config.json`):

```json
{
  "database_uri": "mongodb://localhost:27017"
}
```

Everything else works without it; the Embeddings tab then reports that it is unavailable.

## Where files go

| Path | Contents |
| --- | --- |
| `<dataset>/.yolo-workbench/backups/<session>/` | the original of every label file before its first change in a session |
| `<dataset>/.yolo-workbench/trash/<session>/` | images deleted from the gallery and orphan labels moved away (undoable; restore or delete earlier sessions in the Dataset tab) |
| `<dataset>/.yolo-workbench/index.json` | cached image sizes |
| `<dataset>/.yolo-workbench/review.json` | review status per image |
| `<dataset>/.yolo-workbench/labelqa.json` | the last Label QA analysis and its dismissed findings |
| `<dataset>/.yolo-workbench/imports.jsonl` | import log |
| `<dataset>/runs/` or `<project>/runs/` | training runs |
| `~/.cache/yolo-workbench/media/` | rendered thumbnails and crops (pruned past 512 MB; honors `XDG_CACHE_HOME`) |
| `~/.cache/yolo-workbench/weights/` | downloaded pretrained weights (training) and SAM 2.1 weights (smart annotation; a copy in Ultralytics' weights dir is used instead when present) |

Label and YAML files are written with LF line endings on every platform.

## Security

The server is meant for one user on one machine. It binds to `127.0.0.1`, rejects requests whose `Host` is not local (DNS rebinding), and requires an `X-Workbench` header on every change (cross-site forms). It has no login: binding to another address exposes your datasets to anyone who can reach it.

## Tests

```bash
uv run pytest
```

Tests build isolated temporary datasets and never modify a real one. `WORKBENCH_SLOW_TESTS=1 uv run pytest -k real_training` additionally trains a tiny real model (downloads `yolo11n.pt`).
