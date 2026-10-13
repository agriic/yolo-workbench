# Projects: plan

Add a **projects mode**. You pass a projects folder, and the tool lists, creates, edits and deletes datasets inside it. That includes managing classes and importing images. The current **single-dataset mode** (`yolo-workbench data.yaml --category …`) stays as it is.

## Decisions taken

| Topic | Decision |
|---|---|
| What a project is | **1 project = 1 dataset**, plus reserved `models/` and `runs/` folders for future training |
| Getting images in | **Browser upload** (files or folders) and **import from a local folder path**. Both copy into the project |
| Split on import | Choose a target split, or **auto** with seeded ratios (e.g. 80/20/0). Later, a "Move to split" action handles rebalancing |
| Labels on import | **Map by class name** when the source has `data.yaml` or `classes.txt`, and offer to add missing classes. Otherwise assume IDs match; unknown IDs show up in Validation |
| Removing a used class | A dialog asks whether to **delete its annotations or merge them into another class**. Remaining IDs are renumbered across all labels in **one undoable transaction** |
| Deleting a project | The folder is **moved to `<projects>/.trash/`**. It can be restored, and "Empty trash" is a separate, explicit step |

Decisions I made myself (tell me if you disagree):
- **The project folder name (slug) never changes after creation.** Renaming a project only changes its display name in `project.json`. Moving the folder would change every absolute path, and with them record IDs, the size index, media-cache keys and FiftyOne names.
- **Class management also works in single-dataset mode.** It only edits `data.yaml` and labels, which is useful for any dataset. Import, image delete and split moves need the standard layout, so they are only enabled for managed projects. The UI decides what to show from a `capabilities` field in the metadata.
- **The UI switches projects with a full page load** (`/?project=<slug>`), not an in-page reset. `app.js`, `canvas.js` and `predictor.js` keep a lot of state at module level, and a clean reload is far cheaper than making all of it resettable.

## On-disk layout

```
<projects>/
  .trash/<slug>-<timestamp>/          # deleted projects, restorable
  <slug>/
    project.json                      # {version, name, category, description, created, summary{images,classes,splits,updated}}
    data.yaml                         # Ultralytics-compatible, written by the tool (see below)
    images/{train,val,test}/          # detection / segmentation
    labels/{train,val,test}/
    train/<class>/ val/<class>/ ...   # classification instead of images/labels
    .yolo-workbench/                  # existing: backups/, index.json; new: trash/, staging/, imports.jsonl, cover.jpg
    models/                           # future: trained weights (predictor discovery scans this)
    runs/                             # future: training runs
```

- **`data.yaml` for managed projects leaves out `path:`.** Ultralytics resolves a relative `path` against its global `DATASETS_DIR`, not the YAML's folder. Without `path`, both Ultralytics and our `Dataset` resolve relative to the YAML. Splits are `train: images/train`, `val: images/val`, `test: images/test`. `names` is written as a contiguous `{0: a, 1: b}` dict.
- **Discovery:** a project is any direct subfolder that contains `project.json`. A subfolder with a `data.yaml` or split dirs but no `project.json` appears as "unmanaged" and gets an **Adopt** action, which asks for the category and writes `project.json`. This is cheap to add and makes dropping existing datasets into the folder painless.
- **Listing projects never loads a `Dataset`.** The cards read `project.json.summary` and `.yolo-workbench/cover.jpg`. An open session refreshes both, debounced after changes and imports.

## CLI

```bash
yolo-workbench /path/to/data.yaml --category detection     # unchanged
yolo-workbench --projects /path/to/projects                # new
```

- `DATASET` becomes optional. Exactly one of `DATASET` and `--projects` is required, and `--category` is required only with `DATASET`.
- `--projects` creates the folder if it is missing (after confirming on the TTY).
- `--model` in projects mode is rejected with a clear message. Models are chosen per project, and discovery also scans `<project>/models/`.

## Backend architecture

### New modules
- **`workspace.py`** contains `Workspace(root)`:
  - lists projects and unmanaged dirs, and creates, updates, trashes, restores and adopts projects;
  - empties the trash;
  - validates slugs (`[a-z0-9][a-z0-9-_]{0,63}`, and the resolved path must stay inside root);
  - reads and writes `project.json` atomically.
- **`session.py`** contains `DatasetSession`, which bundles `Dataset` + `PredictorManager` + `EmbeddingsManager`. `Workspace` keeps an LRU of open sessions:
  - opened lazily on the first request, default cap 2, configurable;
  - a session with a running job (predict batch, embeddings, import) is never evicted;
  - eviction unloads the model and drops pending predictions and the undo history;
  - the UI warns before you leave a project that has pending predictions.

  Single-dataset mode uses one fixed session.
- **`classes.py`** holds the schema operations (below). It keeps `dataset.py` from growing further, in line with PLAN.md item 1.
- **`importer.py`** scans a source, builds an import plan and commits it with progress.
- **`jobs.py`** contains `JobManager`, a workspace-level registry of background jobs: `{id, project, kind, state, progress{done,total}, message, log_path, started, finished}` with cancel. Import is the first user. It is designed so a future **subprocess** job (training) fits the same shape: log file plus exit code. Predictor and embeddings keep their own polling for now, and can migrate later.

### `web.py` split
- `dataset_router()` is an `APIRouter` with every current dataset endpoint. It resolves its `DatasetSession` through a dependency instead of a closure.
- `create_app(dataset, media_cache=None, predictor=None)` keeps its signature and mounts the router at `/api/v1`. **The existing tests run unchanged.**
- `create_workspace_app(workspace, media_cache=None)` mounts:
  - the workspace endpoints;
  - the same router at `/api/v1/projects/{slug}`, with a dependency that maps slug to session.
- `GET /api/v1/app` returns `{mode: "single" | "projects"}`, so the frontend knows its API base.
- The `MediaCache` is shared by all projects. Its keys already contain absolute paths and mtimes.

### `Dataset` changes (foundation)
1. **Allow empty datasets** (`allow_empty=True` for managed projects): no images and no classes must load. Statistics, the galleries and embeddings need empty-state handling.
2. **Add and remove records at runtime**: `_add_record(split, image_path)` and `_remove_record(record)`, which keep every index consistent. They are needed by import, image delete, undo of delete, and a general **`rescan()`**, which diffs disk against the index. That also covers PLAN.md's "external changes are invisible until restart".
3. **Mixed transactions in the undo stack.** Today a transaction is all `HistoryEntry` or all `MoveEntry`. Generalize to a list of entries replayed by kind:
   - `WriteEntry` (the current `HistoryEntry`: a file's content before and after; also used for `data.yaml`)
   - `MoveEntry` (src → dst, optionally with an image ID; also used for image→trash and split moves)
   - `DirEntry` (mkdir/rmdir, for classification class folders)
   - `SchemaEntry` (names before and after). After replay, `names` and the class IDs of affected records are reloaded, and `history()` returns `schema_changed: true`.

   All mtimes are checked before anything is written, as `_commit_many` does today. `data.yaml` gets the same one-time per-session backup as label files.
4. **Schema-change listeners** next to the existing change listeners:
   - `PredictorManager` remaps its class mapping and pending prediction class IDs (deleted classes become unmapped);
   - `EmbeddingsManager` invalidates its results;
   - the statistics cache is cleared.
5. **FiftyOne dataset name** becomes `yolo-workbench-{slug}-{session_id}`. With several sessions open, the timestamp alone can collide.

### Class management (`classes.py`)
One primitive, `remap_classes(dataset, mapping: dict[old_id, new_id | None], new_names: dict[int, str])`, implements every operation:

| Operation | Mapping |
|---|---|
| Add | identity mapping; new name appended at the next ID |
| Rename | identity mapping; name changed |
| Remove → delete | the class maps to `None` (its annotations are dropped); higher IDs shift down by 1 |
| Remove → merge into T | the class maps to T; higher IDs shift down |
| Merge A,B → T | the same, for each source class |
| Reorder | a permutation |

**Detection and segmentation:**
- Rewrite every affected label file and `data.yaml` in **one transaction**, as bulk ops do: check every mtime first, abort the whole batch on any conflict.
- Malformed label rows block the operation and are listed, since they can't be remapped safely. Unknown class IDs are kept untouched.
- `data.yaml` writing for **external** datasets (single mode): replace only the top-level `names:` block and keep its list or dict style. The rest of the file, including comments, stays as it is. If the block can't be located safely, fall back to a full `safe_dump`; the backup is already taken.

**Classification** (class IDs are the sorted folder names, so any add or rename can shift IDs):
- add = mkdir `<split>/<name>` in every split;
- rename = move the folder in every split;
- merge = move files into the target folder, then rmdir;
- delete = move the images to `.yolo-workbench/trash/<session>/…`, then rmdir.

All of these are `MoveEntry` and `DirEntry` items in one undoable transaction, followed by a reload of names and class IDs.

**Validation:** names must be non-empty, unique ignoring case, and contain no `/`. On classification, they must also be valid folder names.

### Import (`importer.py`)
Both input paths converge on one pipeline:

1. **Source:**
   - Local folder: `POST import/scan {path}`.
   - Browser upload: `POST import/upload` (multipart, sent in batches of about 50 files) into `.yolo-workbench/staging/<import_id>/`, keeping relative paths. The UI sends folders through `<input webkitdirectory>` or drag-drop `webkitGetAsEntry`. Then the staging dir is scanned like a local folder.
2. **Scan result:**
   - image count, and pairing with labels using the same rules as `Dataset._label_for` (an `images/`→`labels/` sibling, or colocated);
   - class names found in `data.yaml`, `classes.txt` or `notes.json`;
   - for classification, class names from the parent folder;
   - any `train/val/test` structure the source has;
   - count of duplicates (sha1) and filename collisions.
3. **Commit** (`POST import/commit`, runs as a job):
   - Split: one fixed split, or `auto` with ratios and a seed.
   - Class mapping: source name → project class ID, or "add new class" (applied first as one schema transaction), or "drop".
   - Labels from sources without names are copied as they are.
   - Duplicates are skipped by default. A colliding name gets a `_1` suffix, and the label is renamed with it.
   - Files are copied, never moved. Labels are rewritten with the mapped IDs.
   - Then `rescan()` runs and the summary and cover image are updated.
4. Each import appends a line to `.yolo-workbench/imports.jsonl` (source, counts, mapping, seed). This is provenance for later training.

Imports are **not** on the undo stack. To reverse one, select the imported images and delete them (the grid can filter by import batch later, if wanted). Staging dirs are deleted after commit or cancel, and stale ones are deleted on startup.

### Image management (managed projects)
- `POST images/delete {image_ids}`: moves the image and its label to the project trash. It is one undoable `MoveEntry` transaction and removes the records.
- `POST images/move {image_ids, split}`: moves the image and label to `images/<split>`, and `labels/<split>`, as one undoable transaction. Record IDs stay stable within the session.

### Hardening (needed now that the API can write anywhere in a folder)
- **Reject requests whose `Host` is not localhost or 127.0.0.1**, which stops DNS rebinding.
- **Require a custom header (`X-Workbench: 1`) on all mutating requests.** A multipart upload is a CORS "simple request", so without this any website could POST files to the local server. `api()` adds the header everywhere.
- Upload filenames are normalized: no absolute paths, no `..`, and the resolved path must be inside the staging dir.

## API sketch

Workspace (projects mode only):
```
GET    /api/v1/app                              {mode, version}
GET    /api/v1/projects                         cards + unmanaged dirs
POST   /api/v1/projects                         {name, category, classes[], description}
PATCH  /api/v1/projects/{slug}                  {name?, description?}
DELETE /api/v1/projects/{slug}                  → .trash (closes the session first; 409 if a job is running)
POST   /api/v1/projects/{slug}/adopt            {category} for unmanaged dirs
GET    /api/v1/trash  ·  POST /api/v1/trash/{entry}/restore  ·  DELETE /api/v1/trash/{entry}
GET    /api/v1/jobs?project=  ·  GET /api/v1/jobs/{id}  ·  POST /api/v1/jobs/{id}/cancel
GET    /api/v1/projects/{slug}/cover
```
Per dataset (prefix `/api/v1` in single mode, `/api/v1/projects/{slug}` in projects mode). All existing endpoints, plus:
```
GET  /classes                       names + annotation/image counts per class
POST /classes                       {name}                              add
PATCH /classes/{id}                 {name}                              rename
POST /classes/{id}/remove           {mode: "delete"|"merge", target_id?}
POST /classes/merge                 {source_ids[], target_id}
POST /classes/reorder               {order[]}
POST /rescan
POST /import/scan · /import/upload · /import/commit                    (managed only)
POST /images/delete · /images/move                                     (managed only)
```
`GET /dataset` metadata gains `project` (slug, name) and `capabilities` (`classes`, `import`, `image_ops`).

## Frontend

- **`api.js`:** `setApiBase()` and `apiUrl(path)`. All 34 hard-coded `/api/v1/...` strings change mechanically to `apiUrl("/...")`, including the `<img src>` templates in `app.js`, `canvas.js` and `embeddings.js`. `init()` reads `?project=` and `/api/v1/app` first. In projects mode with no project selected, it redirects to `/projects.html`.
- **`projects.html` + `projects.js`:** a new page with no shared state. It shows:
  - a grid of project cards (cover, name, category, image, class and split counts, last updated);
  - a **New project** dialog (name, category, initial classes, one per line);
  - per-card actions: open, rename or describe, delete (moves to trash);
  - unmanaged folders with **Adopt**;
  - a **Trash** drawer with restore and "delete permanently" (type the name to confirm).
- **Workbench header:** the dataset chip turns into a project switcher. It links back to Projects and lists the others; switching is a full navigation.
- **New "Dataset" tab** (capability-gated sections):
  - **Classes table:** color, ID, name with inline rename, annotation and image counts, and a remove button. Remove opens a dialog with **Delete annotations** or **Merge into [class]**, showing affected counts. Also an "Add class" field and drag-to-reorder.
  - **Import panel:** a drop zone plus a "From folder…" path field. It shows the scan summary, the split choice (fixed or auto with ratio sliders and seed), the class-mapping table with "add as new class", and progress from the job.
  - **Splits overview** (counts), with a **Rescan** button.
- **`renderMetaControls()`:** split the select- and chip-filling code out of `init()` so it can rerun after a schema change. That covers class ops and an undo or redo that returns `schema_changed`. Any open editor is closed first.
- **Image-grid selection** (Shift and Ctrl click, like the object grid), with a bulk bar for **Delete** and **Move to split** in managed projects.
- **Empty states:** a new project shows "Import images" calls to action in Annotation, Exploration, Statistics and Embeddings.

## Phases

Each phase can ship on its own, and the tests stay green between phases.

1. **Foundation** (no visible change):
   - `apiUrl`/`setApiBase` and the `X-Workbench` header;
   - the `dataset_router` and `DatasetSession` split;
   - the Host guard;
   - `allow_empty`, `_add_record`/`_remove_record`, `rescan()`;
   - mixed-entry transactions;
   - schema-change listeners;
   - `renderMetaControls()`.
2. **Class management** (useful in single mode right away): `classes.py`, minimal-edit `names:` writer, classification folder ops, Dataset tab with the classes table, and delete/merge dialog.
3. **Workspace & projects:**
   - CLI `--projects` and `workspace.py` with the session LRU;
   - the workspace API and `projects.html`;
   - create, rename, trash, restore and adopt;
   - the header switcher and empty states.
4. **Import:** `jobs.py`, `importer.py`, local-folder and upload flows, split assignment, name mapping, `imports.jsonl`, cover and summary refresh.
5. **Image management:** image-grid selection, delete to trash, move to split.
6. **Training** (future, not designed here). See below.

## Hooks left for training (not planned in detail)
- **Layout:** `models/` and `runs/` are reserved in every project. Predictor discovery scans `models/`, so a trained model is ready for assisted labeling right away and closes the loop.
- **Jobs:** `JobManager` jobs already have a project, progress, a log file, cancel, and a workspace scope, which a single-GPU training queue needs. Training should run as a **subprocess** (isolation, GPU memory, survives a UI reload), and the job shape allows that.
- **Ultralytics compatibility:** managed `data.yaml` has no `path` and contiguous names, which Ultralytics requires. Add a **"training readiness"** check to Validation later: `val` is non-empty, there are no unresolved errors, and every class has examples.
- **Reproducibility:** label edits during a training run would change its inputs. Later, add a snapshot step: hardlink the images and copy labels plus `data.yaml` into `runs/<run>/dataset/`. `imports.jsonl` and `project.json` give the provenance.

## Tests (per phase; temporary dirs only, never a real dataset)
- **Workspace:**
  - create, list, rename, trash, restore and adopt;
  - slug validation and path traversal;
  - deleting a project with a running job returns 409;
  - LRU eviction skips busy sessions.
- **Classes:**
  - delete and merge with renumbering, for both list-style and dict-style `names`;
  - undo and redo restore labels and `data.yaml` byte for byte;
  - a mid-batch conflict aborts the whole batch without writing anything;
  - comments elsewhere in `data.yaml` survive;
  - classification add, rename, merge and delete;
  - pending predictions are remapped.
- **Import:**
  - label pairing for both layouts;
  - mapping by name from `data.yaml` and `classes.txt`;
  - auto-split is deterministic for a given seed;
  - duplicate skip and collision rename;
  - upload filename sanitization;
  - `rescan()` picks up the new records.
- **API:**
  - single mode is unchanged (the existing suite);
  - the same endpoints work under `/api/v1/projects/{slug}`;
  - the Host and header guards reject cross-origin-style requests.

## Resolved during implementation

- **One open project at a time** (replaces the LRU idea). Opening another project closes the current one; if it is busy (prediction run, embeddings, import) the request gets a 409 and the projects page explains why.
- **Classification projects always get a `data.yaml`** (`train: train`, `val: val`, `test: test`, `names: [...]`), also when adopting a folder without one.
- **Cover image = first image** (sorted by split, then name), regenerated when that image changes.

## Implemented — deviations from the plan above

- Uploads are one raw `PUT /import/staging/{id}/files?path=…` per file (4 in parallel from the browser) instead of multipart, so no `python-multipart` dependency and every upload needs a CORS preflight anyway.
- Import, image delete and split moves are offered wherever every split is a single image directory (`capabilities.image_ops`), so they also work in single-dataset mode for standard layouts; file-list (`.txt`) datasets don't get them.
- The import dialog also offers **keep the source splits** when the source has `train/`, `val/` (or `valid/`), or `test/` folders.
- Class reorder is up/down buttons (each move is one undoable step) rather than drag and drop.
- A cancelled import keeps the files copied so far (they are indexed by the rescan).
- Added after the first pass: **several folders per split**. The Splits panel lists each split's folders with image counts and can add a folder (created with its `labels/` twin and appended to the split in `data.yaml`, undoable). The import review has a destination-folder picker per split (with “New folder…”), and the Move bar targets a split folder.
- Not done yet: filtering the grid by import batch; migrating predictor/embeddings polling onto `JobManager`.
