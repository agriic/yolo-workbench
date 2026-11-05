# Dataset snapshots and split rebalancing: plan

Two features that lean on each other:

- **Snapshots**: a named, frozen copy of a dataset or project (images, labels, classes, splits). It can be
  downloaded, trained on, compared with the current state, and restored.
- **Rebalance**: reassign images to train/val/test by rules, not only by ratio: class stratification, related
  images kept together, the test split kept fixed, rules for which images may go to val/test, and minimums per
  class. The user previews the result before anything moves, and it is applied as one undoable step.

A rebalance is the biggest structural change one click can make to a dataset, and it changes what every later run
trains and validates on. Snapshots make it safe: one is taken automatically before applying. They also keep runs
comparable: a run trained on a snapshot keeps that snapshot's splits.

## What it takes (short answer)

| Phase | What | Effort |
|---|---|---|
| S1 | Snapshots: take (a job), list, rename, delete, download; sizes in the storage view | 2–3 days |
| S2 | Train on a snapshot: the live dataset is not read-only during such a run | 1–2 days |
| S3 | Split table + compare a snapshot with the current state (shared with the rebalance preview) | 1–2 days |
| R1 | Rebalance: rules, engine, live preview, apply as one undo step (always after a snapshot); batch folders kept | 5–7 days |
| R1b | Train with a proposed split: a snapshot laid out by the plan, nothing in the dataset moves | 1 day |
| R2 | Moves on network drives: no filesystem call per image while planning and checking a big move | 1 day |
| R3 | Keep near-duplicates and capture bursts together (image hash scan job + cache) | 2 days |
| R4 | Import by the rebalance rules, a split-leakage check | 1–2 days |
| S4 | Restore: as a new project (half a day); in place (3–4 days) | |

S1 → S2 → S3 → R1 → R1b → R2 is the core (roughly 2.5–3 weeks). R3, R4 and S4 can come in any order after it.

## Findings that shape the design

- **The workbench never edits a file in place.** Label writes are temp file + rename (`_commit`). Image swaps
  (`replace_images`, resize), moves and deletes are renames (`Dataset._move_file` → `fsutil.replace`). A hardlink
  to an image therefore keeps that version, whatever happens to the dataset later. Labels are tiny and are copied
  anyway, in case a tool outside the workbench edits them in place.
- **The layout already exists.** `download.plan(dataset, "full")` turns any dataset into a self-contained YOLO
  layout with a rewritten `data.yaml`. That includes split folders outside the root (`external/`), `.txt` lists and
  labels outside the root. A snapshot is that plan written to a folder, with hardlinks instead of a zip.
- **A snapshot is itself a dataset.** `Dataset(<snapshot>/data.yaml)` opens it, so `_data_plan`,
  `_rfdetr_splits`, the folder subset and the download all work on it unchanged.
- **The training lock follows the active run's target.** `_refresh_lock` locks `target.yaml_path`, and
  `Dataset.write_guard` is a single callable that `TrainingManager.attach` sets. A run on a snapshot needs no lock.
  A second guard (taking a snapshot blocks image moves for a few minutes) needs guards to combine.
- **Moves are slow at scale on network drives.** `move_images` takes one target folder for every image and makes
  about 3 filesystem calls per image while planning (`exists` of the target image, the source label and the
  target label). `_check_entry` makes up to 5 more per image when the transaction runs (source/destination
  `exists` for the image and its label, plus a label `exists`). A 20,000-image rebalance on an SMB share at 2 ms per call is about 5 minutes under the
  dataset lock. That breaks the rule from `PLAN-network-drives.md`: no filesystem call per image in an interactive
  request.
- **Undo can hold it.** `_transaction_bytes` counts a `RecordMoveEntry` as about 256 bytes, so 100,000 moves are
  about 25 MB against the default `undo_max_bytes` of 512 MB. The newest transaction is never dropped.
- **Record ids are `sha1(split:path)`, made when a record is created.** A moved record keeps its id until the
  dataset is opened again. Snapshot manifests and diffs therefore key on dataset-relative paths, not ids.
- **Label digests are already in memory.** `_digests` holds `_digest(text)` (SHA-1 of the label text) for every
  label as last read or written. A manifest that stores the same digest can be compared with the live dataset
  without reading a file.
- **What can be reused:**
  - Split logic today is only the importer's `_assign_splits` (ratios, seed, largest remainder).
  - `statistics()` already builds per-split class object counts (`split_rows`).
  - `_filter_images` (review status, issue, reviewer, folder) is a ready-made selector language for the rules.
- **Classification works the same way.** A split move is a `RecordMoveEntry` from `<root>/<split>/<class>/` to the
  same class folder in another split. Stratification is single-label there.
- **Batches are separate split folders.** New data usually lands in a folder per date or revision. In this app
  that is one more folder of the split (`add_folder`; split folders cannot nest), named `images/train_2` by
  default or e.g. `images/train_2024-03` or `2024-03/images/train`. `move_images` puts every image into one target
  folder (the split's first by default), so a move would lose the batch. It does keep the path below the split
  folder. The new-run form's folder subset (`spec.json` `folders`) is built on these folders too.
  `importer.SPLIT_ALIASES` already maps `training`/`valid`/`validation`/`testing` to the split names.

## Snapshots

### On-disk layout

```
<dataset>/.yolo-workbench/snapshots/
  <id>/                     # YYYYMMDD-HHMMSS (+ -2, -3 … when taken in the same second)
    snapshot.json           # name, note, kind (manual | auto | run), created, by, content fingerprint, dataset
                            # revision, schema generation, category, class names, split table (S3), file counts,
                            # linked/copied bytes, release; for R1b the rebalance rules + plan digest
    manifest.jsonl          # one line per image: path in the dataset (as it was), path in the snapshot, split,
                            # folder label, label digest, image (mtime_ns, size), width/height when known, review
    data.yaml               # download.plan's: no path:, splits relative
    images/… labels/…       # download.plan's names; classification: <split>/<class>/
  .partial-<id>/            # while it is being taken; removed when it fails or is cancelled
```

The snapshot lives inside the dataset because hardlinks need the same filesystem. `.yolo-workbench/` is already
left out of indexing, orphan scans and project listing.

### Taking one (`snapshots.py`, job kind `snapshot`)

1. **Under the dataset lock** (seconds): build the `download.plan` entries, read the label texts with
   `_read_texts` (thread pool), and write them, the `data.yaml` text and the manifest into the partial folder.
   Then turn on a **structure guard** for the rest of the job: image moves, deletes, resizes, imports and class
   folder changes get a 409 ("a snapshot is being taken"). Label edits continue, since the labels are already
   copied.
2. **Images, outside the lock:**
   - Hardlink each one with `os.link`. Where that fails (`EXDEV`/`EPERM`/`ENOTSUP`: another volume, or an SMB share
     without hardlinks), copy it with `shutil.copy2`. Never symlink: a symlink follows the live file.
   - Before the first copy, check free space for the bytes still to copy (sizes come from `image_size`).
   - Work in a thread pool, because a link on a network drive is a round trip. Report progress per file.
   - Compare each linked file's (mtime, size) with the record's stamp. A mismatch means the image changed outside
     the workbench: note it in snapshot.json.
3. Rename the partial folder to `<id>`, write `snapshot.json` last, and turn the structure guard off. Cancelling
   removes the partial folder.

`Dataset.write_guard` becomes a list of named guards, each covering everything or structural changes only.
Training's lock is one guard and a snapshot in progress is another. `check_writable(structure=True)` is called by
moves, deletes, `replace_images`, imports and the classification class-folder operations.

The dialog says up front whether the snapshot will link or copy, for example: "Will copy 12 GB: this drive does
not support hardlinks". It finds out by test-linking one file.

### Sizes and retention

- **Sizes.** Each file's `st_nlink` gives `own_bytes` (only this snapshot holds the file) and `shared_bytes`
  (the dataset or another snapshot also holds it). They are computed when the snapshot is taken and again on
  demand ("Recount", a job), and shown in the storage section ("deleting frees about X"). Deleting images from
  the project does not free their space while a snapshot holds them; the trash and storage views say so.
- **Retention.** Snapshots are never purged by trash or backup retention. There are three kinds:
  - **manual** snapshots never expire;
  - **auto** snapshots (taken before a rebalance or a restore) expire after
    `[storage] auto_snapshot_retention_days` (default 30, desktop too; 0 keeps them) unless marked *keep*.
    Marking one *keep* makes it manual;
  - **run** snapshots (S2) stay as long as a run uses them and go when the last such run is deleted.

  Expired auto snapshots are purged when a dataset opens, when a snapshot is taken, and by the server's
  maintenance (`apply_retention`). The new setting goes into `config.py` and `deploy/config.example.toml` (a test
  checks).
- **Backups.** [docs/server.md](docs/server.md) needs a note: a copy of the projects root that does not keep
  hardlinks (`rsync` without `-H`, plain `cp`) stores every image once per snapshot.

### Training on a snapshot (S2)

- **The form.** The new-run form's Training data panel offers *Snapshot at launch* (the default), *Snapshot:
  <name>*, or *Current dataset*. The spec records it as `spec.json` `snapshot: "launch"` or `<id>`, or no key.
  - *Snapshot at launch* is the "frozen snapshot mode" from `PLAN-training.md`. The project stays writable while
    the run trains, and the run trains on exactly what the dataset held when it launched.
  - When the snapshot would be a full copy (no hardlinks there), the default falls back to *Current dataset*, and
    the form says why and what a copy would cost.
  - It reuses the newest snapshot whose content fingerprint matches the dataset, so ten runs launched on unchanged
    data share one snapshot. The fingerprint is the sha1 of each image's path, split and label digest, plus the
    class names and split folders. That snapshot becomes the run's: kind `run`, unless it is a manual one.
  - A resumed run trains on its snapshot again. Today a resume rebuilds the data from the live dataset, which may
    have changed since the run started.
- **Launching.**
  - `_launch` opens the snapshot as a read-only `Dataset` and plans from it. Read-only means a new
    `read_only=True` mode that writes nothing: no size index, no `.yolo-workbench/` inside the snapshot. It also
    sets no write guard on the live dataset.
  - `_refresh_lock` skips such runs.
  - `run.json` `dataset` records the snapshot id and name, and `compare.js` shows the name next to the fingerprint.
- **Deleting a snapshot that runs use.** It is refused while a queued, active or resumable run uses it, like
  `started_from` for models. After that it is allowed, with a warning that names the runs. Their
  `workbench/data.yaml` then points at files that are gone. `TrainingManager.delete_run` deletes a `run`
  snapshot that no other run uses.

### Split table and compare (S3, `splitstats.py`)

- **`split_table(rows, names)`.** Per split: images, background images, objects, and per class the images
  containing it and its objects. Optionally object-size buckets. The Statistics tab's `split_rows` moves onto it,
  and the rebalance preview uses it too.
- **`diff(manifest, dataset)`.**
  - Compares the snapshot with the current dataset, keyed by dataset-relative path: added, removed, moved to
    another split, and label changed (`_digests` against the manifest's digest).
  - Also compares the class list and the split folders.
  - Makes no filesystem call per image (checked with `tests/fscount.py`).
  - The UI shows counts and the split table side by side. Each list (added, moved, …) opens the gallery filtered
    to it.

### Restore (S4)

- **As a new project** (projects mode): hardlink the snapshot folder into `<root>/<new slug>/`, then
  `Workspace.adopt`. In single-dataset mode the snapshot can only be downloaded.
- **In place**, like an import: not on the undo stack, and an automatic snapshot of the current state ("Before
  restore") is the way back. Then the dataset is made to match the snapshot:
  - images that are missing are linked back from the snapshot;
  - extra images go to the trash;
  - images in the wrong split are moved;
  - label contents and `data.yaml` are rewritten.

  Then `rescan()`, clear the undo history, and record a `reset` change. It is refused while a run trains on the
  live dataset or any image job runs.

## Rebalance

### Rules (what the user sets)

A dialog: Dataset tab → Splits → *Rebalance…*. The rules are saved per dataset in
`.yolo-workbench/rebalance.json`, so re-running it after an import is one click.

1. **Targets.** A ratio per split, counted in images (default) or objects.
2. **What stays put** (pinned):
   - frozen splits: test is frozen by default when it has images, so test metrics stay comparable across runs;
   - frozen split folders;
   - images matching a gallery filter (the `_filter_images` query: review status, issue kind, reviewer, folder).
3. **Keep together.** Each rule gives every image a group key; images that share a key under any rule form one
   unit (union-find), and a unit never spans splits:
   - a subfolder below the split folder, at a chosen depth (`cam1/`, `video_017/`);
   - a filename pattern: a regex whose first group is the key. Presets include "everything before a trailing
     `_`/`-` + digits", so `clip12_000345.jpg` → `clip12`;
   - near-duplicates: a 64-bit dHash, within a Hamming distance (R3);
   - capture bursts: EXIF `DateTimeOriginal` within N seconds from the same camera model (R3).
4. **Balance** (soft goals, each weighted off / low / normal / high):
   - class presence: images containing each class, i.e. multi-label stratification (default normal);
   - class objects (default low);
   - background images, i.e. images with no objects (default normal);
   - batches: each batch spread over the splits in proportion, so val reflects every date or revision and not
     only the oldest data (default normal when a split has more than one folder). A batch is an image's split
     folder with the split name taken out of its label: `images/train_2024-03` and `images/val_2024-03` are both
     batch `images/*_2024-03`. *Keep together* by batch is the alternative, a time-based holdout;
   - object size buckets: small/medium/large by pixel area, COCO's 32²/96² (default off; needs probed image sizes);
   - spread a key: a folder or pattern key spread over the splits in proportion, so each camera is in each split.
     This is the opposite of keeping it together, and is meant for a coarser key, e.g. clips kept together but
     cameras spread.
5. **Hard limits:**
   - a minimum number of images per class per split, e.g. val ≥ 20 of each class. When it cannot be met, the
     preview says why (e.g. "class X has 7 eligible images");
   - which images may go to val/test, as a gallery filter, e.g. review = done and no validation errors.
6. **Stability:**
   - *Move as few images as possible* (default): start from the current splits; a move must improve the balance
     by more than its cost;
   - *Reassign everything not pinned*: start from scratch;
   - an optional maximum number of moves, and a seed.

### Engine (`rebalance.py`; standard library, pure functions)

**Input** is a snapshot of the records, taken under the dataset lock: path, split, folder, class id per object,
size when known, review status and issues. **Output** is a plan. The computation runs outside the lock.

1. **Units.** Union-find over the keep-together keys. Each unit has:
   - its members;
   - a sparse feature vector: images, background, per-class presence, per-class objects, size buckets, and
     spread-key indicators;
   - the splits it may go to: the intersection over its members;
   - a pinned split, if any member is pinned. Members pinned in different splits are a conflict: the unit stays
     as it is and the preview reports it.
2. **Targets.** Frozen splits leave the problem. The other splits share the remaining totals by their ratios:
   T[s][f] = rᵢ/Σr × (total_f − frozen_f). Pinned units count toward their split.
3. **Cost.** E = Σ_f w_f Σ_s (A[s][f] − T[s][f])² / max(T[s][f], 1), a chi-square-like error that weighs rare and
   common classes alike. On top of that:
   - a large penalty for each shortfall below a minimum;
   - λ per moved image in *as few moves* mode.
4. **Start.**
   - *As few moves*: each unit starts in its current split. A unit that spans splits (leakage) starts in the
     split holding most of its images, so its other images must move.
   - *Reassign*: greedy iterative stratification. Units are sorted by their rarest feature, then by size
     descending, and each goes to the allowed split with the largest relative deficit in that feature. Ties go to
     the split with the larger image deficit, then by the seed.
5. **Improve.** Passes over the units in seeded order:
   - take the best single move by ΔE (only two splits' rows change, so each candidate costs O(nnz));
   - then try seeded pair swaps of similar-size units between splits. They fix class balance without changing
     split sizes.

   Stop when a pass gains less than ε, or after 20 passes. The budget counts iterations, not time, so the result
   is deterministic for the same records, rules and seed.
6. **Report:**
   - the moves;
   - the split table before, after and at the target, per feature;
   - leakage before and after;
   - unmet minimums with the reason;
   - pinned conflicts and the number of images not allowed in val/test;
   - a `digest` of the plan.

At 100,000 images, 3 splits and about 20 features per unit, a pass is a few million additions in Python, about
1–2 s. The preview is a plain (threadpool) endpoint, one at a time per dataset.

### Applying

1. **Where a moved image goes: its batch is kept.** `twin_folder(folder, split)` looks among the target split's
   folders for the one whose label matches the source folder's label once the split name is swapped. A split name
   is a whole path component or a name part separated by `_`, `-` or `.`, and its aliases come from
   `SPLIT_ALIASES`. For example:
   - `images/train_2024-03` → `images/val_2024-03`;
   - `2024-03/images/train` → `2024-03/images/val`;
   - `images/train` → `images/val`.

   When the twin does not exist yet, the apply creates it (the folder, its `labels/` twin and the YAML entry, as
   `add_folder` does) inside the same transaction, so the same undo removes it. A source label without a split
   name (e.g. `batch7` listed under train) gets `batch7_val`. The path below the split folder is kept as today.
   The preview lists each source folder → target folder, marks the ones it will create, and lets the user change
   one. Choosing *the split's first folder* gives today's `move_images` behavior. Classification has one folder
   per split, so nothing changes there.
2. `POST /rebalance/apply` carries the rules, the preview's `digest`, and any folder overrides. The server
   computes the plan again; a different digest means the dataset changed since the preview, so it returns 409 and
   the dialog previews again.
3. A `rebalance` job does the work. It is in `IMAGE_JOBS`, so a queued run and a project switch wait for it.
   1. **Always a snapshot first**: an auto snapshot, "Before rebalance <date>", kept for 30 days. If it cannot be
      taken (no space for a copy), nothing moves and the job says why.
   2. `Dataset.relocate_images(moves)` with `identity.action("rebalance splits")`, as one transaction, so one
      undo puts every image back. It generalizes `move_images` to a target (split, folder) per image, plus the
      folders to create (`DirEntry`s, the YAML `FileEntry` and a layout-only `SchemaEntry`, ahead of the moves).
      `move_images` becomes a thin wrapper.
4. **R2: network drives.**
   - `relocate_images` checks targets against an `fsscan.Listing` of the target folders and knows which labels
     exist from `_mtimes`, so it makes no `exists()` call per image.
   - `_check_entry` gets a batched path for transactions with many `RecordMoveEntry`s, using a `Listing` of the
     source and destination folders. That also speeds up undoing a big delete today.
   - `_run_transaction` takes an optional progress callback for the job.
5. It is refused while a run trains on the live dataset (the training guard). Runs on snapshots do not block it.
   Afterwards the training fingerprint changes, and the run comparison already flags runs with different data.

### Train with a proposed split (R1b)

A virtual split: the rebalance dialog's *Train with this split* takes a snapshot laid out by the plan instead of
by the live splits. Each moved image goes where `twin_folder` would put it, and `data.yaml` lists those folders.
Nothing in the dataset moves.

- The snapshot is kind `run` (named "Proposed split <date>"), and it records the rules and the plan digest.
  The new-run form opens with it selected.
- Its compare (S3) shows the proposed moves against the live dataset. When the run turns out better, *Apply this
  split* applies the same rules, and the digest check makes sure it is the same plan. If the data changed since,
  the user previews again.
- This needs no new concept in training: it is a snapshot like any other. The snapshot writer takes an optional
  assignment ({image id: split}) and the shared `twin_folder`.

### Later (R4)

- **Import** gets a split mode *By the rebalance rules*: existing images stay pinned and the engine places the new
  ones (*as few moves* mode). `_assign_splits` stays for plain ratios.
- **Validation** gets a split-leakage check using the saved keep-together rules: groups and near-duplicates that
  span splits.

## API

Snapshots (dataset router, so they work in both modes):

- `GET /snapshots` · `POST /snapshots` {name, note} → job · `GET /snapshots/{id}` · `PATCH /snapshots/{id}` {name,
  note, keep} · `DELETE /snapshots/{id}`
- `GET /snapshots/{id}/download?kind=labels|full` (the zip stream, from the snapshot's files)
- `GET /snapshots/{id}/diff`
- `POST /snapshots/{id}/project` {name} (projects mode) · later `POST /snapshots/{id}/restore` → job
- `GET /snapshots/link-check`: whether a snapshot would link or copy, and the bytes it would copy

Rebalance:

- `GET /rebalance/rules` (saved) · `POST /rebalance/preview` {rules, folders} → tables, counts, warnings, the
  source → target folder table, digest; the moves stay in memory under the digest for the gallery ·
  `GET /images?plan=<digest>` (gallery filter: images the plan moves)
- `POST /rebalance/apply` {rules, digest, folders} → job (snapshot, then moves)
- `POST /rebalance/snapshot` {rules, digest, folders, name} → job (R1b: a snapshot laid out by the plan)
- R3: `POST /rebalance/scan` → job (image hash + EXIF time per image, cached in `.yolo-workbench/imagekeys.json`,
  keyed like `index.json` by mtime + size)

Training: `spec.json` `snapshot`, `run.json` `dataset.snapshot`.

## Frontend

- `snapshots.js`: the Dataset tab's Snapshots section. Take (name, note, link/copy notice), list (name, date, who,
  counts, own/shared size, runs using it), download, compare, new project from it, delete. `manage.js` is already
  large; it only mounts the section.
- `rebalance.js`: the rules dialog with a debounced live preview:
  - the split table: before → after against the target, with the deviation colored;
  - the number of moves, leakage, warnings;
  - where moved images go: source folder → target folder, new folders marked, each changeable;
  - *Show moved images* (the gallery `plan=` filter);
  - *Apply* (it says a snapshot is taken first) and *Train with this split* (R1b).
- `trainform.js`: the Training data panel's source picker (*Snapshot at launch* by default). `compare.js`: the
  snapshot name.

## Decisions

1. **Rules in R1:** targets, pins, keep-together by subfolder and filename pattern, the balance goals and hard
   limits above. Near-duplicates and bursts follow in R3.
2. **Batch folders (date / revision) are kept.** A moved image goes to its batch's folder in the target split,
   created when missing. Batches are spread over the splits in proportion by default. Losing them would hide where
   the data came from and break training on a folder subset; keeping them costs at most a few new folders, shown
   in the preview.
3. **Test is frozen by default** in a rebalance, so test metrics stay comparable across runs. Unfreezing it is
   one checkbox.
4. **Default stability:** *as few moves as possible*. It finishes in one step, so running it again moves nothing,
   and val stays mostly the same, so runs before and after can still be compared.
5. **A snapshot is always taken before a rebalance** (no checkbox). Auto snapshots expire after 30 days unless
   kept; manual snapshots never expire.
6. **Snapshots live inside each dataset** (`.yolo-workbench/snapshots/`), on the same filesystem, so hardlinks
   work.
7. **Training defaults to *Snapshot at launch*** on desktop and server wherever hardlinks work, reusing an
   identical snapshot. The project is no longer read-only during training, and resumes are reproducible.
8. **Virtual splits are in the core (R1b),** built as a snapshot laid out by the plan rather than as a training
   option.

## Tests (temporary dirs only)

- **Snapshots:**
  - images linked and labels copied;
  - a forced copy fallback (`os.link` raising `EXDEV`) and the space check;
  - cancel removes the partial folder;
  - image moves refused and label edits allowed while a snapshot is being taken;
  - a resize after the snapshot leaves the snapshot's image as it was;
  - its download equals a download made at that moment (outside-root folders, `.txt` lists, classification);
  - retention purges only expired auto snapshots that are not kept: never manual ones, never a `run` snapshot a
    run still uses;
  - delete is refused while a queued run uses it, and deleting the last run that uses a `run` snapshot deletes it.
- **Training on a snapshot:**
  - the live dataset stays writable;
  - Ultralytics' file lists and RF-DETR's `splits.json` name the snapshot's files;
  - `run.json` records the snapshot;
  - *snapshot at launch* takes one, and a second launch on unchanged data reuses it; a label edit in between makes
    a new one;
  - a resume trains on the run's snapshot after the live dataset changed;
  - the default falls back to *Current dataset* when linking fails.
- **Diff:** added, removed, moved and changed are found, with no filesystem call per image (`fscount`).
- **Engine** (pure, fast):
  - ratios are met to within one unit;
  - groups never span splits; pinned and frozen images never move;
  - images not allowed in val/test never go there;
  - minimums are met when possible and explained when not;
  - leakage is repaired;
  - previewing again after an apply plans zero moves;
  - the same seed gives the same plan, and rare classes reach val;
  - classification works;
  - batches are spread in proportion, or kept together when asked;
  - 100,000 synthetic images finish within a few seconds.
- **Twin folders:** `images/train_2024-03` ↔ `images/val_2024-03`, `2024-03/images/train` ↔ `.../val`,
  `valid`/`validation` aliases, a folder without a split name (`batch7` → `batch7_val`), and an override to the
  split's first folder.
- **Apply:**
  - one undo restores every image and removes the folders it created (and their YAML entries);
  - a stale digest gets a 409;
  - refused while a run trains on the live dataset;
  - the auto snapshot is always taken, and nothing moves when it fails;
  - other browsers get a `structure` change;
  - no filesystem call per image while planning and checking (`fscount`, network-drive simulation).
- **Train with a proposed split (R1b):** the snapshot's layout follows the plan, the dataset is untouched, and
  *Apply this split* is refused (409) once the data has changed.

## Risks

- **No hardlinks** (another volume, an SMB share) means snapshots are full copies. The dialog says so before
  taking one, and the space check refuses it when the disk would fill. Because the snapshot before a rebalance is
  mandatory, a large dataset on such a drive could not be rebalanced without the space for a copy. If that
  happens, add a labels-only auto snapshot: labels, `data.yaml` and the manifest, with images referenced rather
  than copied. That is enough to put the splits back, but not to train on.
- **Backups that do not keep hardlinks** multiply the space (docs note).
- **Over-constrained rules** (big groups, minimums, strict rules for val/test) may have no good solution. The
  preview must name the constraint, not just show a poor split. A 5,000-frame video cannot be split, so the
  achieved ratios are reported.
- **Undo after a rebalance** is refused once someone edits a moved image's label at its new place (existing
  `_check_entry` behavior). The automatic snapshot is the fallback.

## Not in scope

- Balancing classes by over- or undersampling or by loss weights (a training-time concern).
- Snapshots shared or deduplicated across projects.
- Snapshots of runs and models.

## Implemented

All phases (S1–S4, R1, R1b, R2–R4) are built, with tests in `tests/test_snapshots.py` and
`tests/test_rebalance.py`, and checked in the browser on a demo project (take, compare, gallery subset, rebalance
preview and apply with new batch folders, train with this split, restore, new project, import by the rules).
Where the build differs from the plan above:

- **Snapshot bookkeeping** lives in the snapshot's own `.yolo-workbench/` (`snapshot.json`, `manifest.jsonl`,
  `index.json` with the image sizes so opening it probes nothing, `source.yaml` for an in-place restore), which
  every walk already prunes. The manifest also records each label's dataset path.
- **A proposed split (R1b)** keeps every file where it is and lists the images per split in `<split>.txt`
  (data.yaml names the lists), instead of laying them out in twin folders. Ultralytics and the workbench read
  such lists already; training on a folder subset does not apply to it, and classification has no proposed
  split (its splits are folders: apply the split).
- **No stamp check per linked image**: it would be a call per image. The structure hold keeps the images in
  place while they are linked, and the workbench never edits an image in place.
- **Retention** of automatic snapshots runs when snapshots are listed or taken, and in the server's maintenance,
  not when a dataset opens.
- **The leakage check** is a note in the Splits panel (by the saved keep-together rules, with "show them") and
  part of every preview, not a Validation issue kind.
- **Import by the rules** imports into train first, then places the new images as a separate undoable move (the
  import itself is not on the undo stack).
- **An in-place restore** leaves the (empty) folders a later rebalance made on disk; they are no longer in
  data.yaml.
- **Not built**: the labels-only automatic snapshot for drives without hardlinks (see Risks). A full copy is made
  there, after a space check.
- **Performance**: a 5,000-image plan takes about 0.5 s (string path tests instead of pathlib; swap rounds only
  at a local optimum); a 100,000-image dataset should take around 10 s per preview.
