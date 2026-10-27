# Network drives: plan

Datasets and projects on a network drive (an SMB/CIFS share, a mapped Windows drive, NFS) should work as well as
on a local disk. No request should wait on one network round trip per image, and there should be no false or
missed write conflicts.

## Diagnosis

A file-metadata call (`stat`, `exists`, `open`, one step of `resolve()`) costs about 1–2 µs on a local disk. On a
network drive it costs 0.5–5 ms, because each one is a round trip to the file server; Windows' SMB client is at the
slow end. The workbench made calls like that **per image** in several places users wait on. Some of them held the
dataset lock meanwhile, so other requests stalled too.

Measured with a simulated network drive, 3,000 images. Every filesystem call under the dataset folder waited 1 ms.
A directory listing's per-entry `stat` was free, as it is on Windows and right after a listing on Linux's NFS and
SMB clients.

| What the user does | Before | Calls before | Cause | After |
|---|---|---|---|---|
| Label QA: change a filter, the confidence or the tolerance | 3.7 s | 3,000 `stat` | `findings()` stats every analyzed image to see whether it changed since the analysis, holding the dataset lock | 0.01–0.09 s, no calls |
| Label QA / Annotation tab: model list (`/predictor/models`) | 6.5 s | 6,000 `stat` | the weights search walks the dataset folder and calls `is_dir()` on every image and label | 0.014 s |
| Open the dataset or project | 53 s | 24,000 `lstat`, 21,000 `stat`, 3,000 `open` | per image: `exists` for the label layout, `stat` to check the size index, `exists` + `stat` around reading the label; per label: `resolve()` (an `lstat` per path component) to find orphan labels | 0.7 s |
| Delete / move one image, or undo it | 33 s | 30,000 | the orphan-label search (`resolve()` of every label) after every structural change | 0.04 s |
| Rescan | 36 s | 33,000 | as opening | 0.17 s |
| Rename a class | 13 s | 12,000 | every label re-read one at a time (`exists`, read, `exists`, `stat`) | 0.33 s |
| Reorder / merge classes (rewrites every label) | 20.6 s (undo 10.4 s) | 21,000 | reads, checks and writes one file at a time | 1.5 s (undo 1.0 s) |
| Download (labels) | 7 s | 3,000 `stat` + 3,000 `open` | `is_file()` per label while planning, then one file after another | 0.5 s |
| Save one image's labels | 27 calls | | `resolve()` for the backup path, `mkdir` of known folders, re-reading the file just written | 5–6 calls |
| Thumbnail already cached | 2 calls | | `stat` of the image and the label for the cache key | no calls |

With 1–3 ms per call and 5,000–10,000 images, that is the reported 7–10 s per Label QA filter change. On a local
disk the same requests take milliseconds, which is why it only showed on a network drive. Two contributors were not
about the drive: the listing held the dataset lock while it ran, and images analyzed under another id (opened
again after a move) were looked up with a scan of all images each.

Correctness problems on network filesystems:

- **Owner lock**: on a filesystem without lock support (some NAS, SMB or FUSE mounts), `flock`/`msvcrt.locking`
  fails. The error was reported as "already in use by another process", so the workbench refused to open the folder.
- **Write conflicts by modification time only**: SMB servers may report a file's modification time late (it can
  change after the write is acknowledged), and some shares keep only 1–2 s resolution. The first gives false
  "changed outside this application" errors. The second can miss a real outside edit made within the same second.

## Principles

1. Between rescans, the in-memory index is the truth about the files, as it already was for label contents. An
   interactive request makes no filesystem call per image. The file times the workbench needs (image and label
   modification times, image sizes) are kept in memory: read when the dataset is opened or rescanned, and updated
   by the workbench's own writes and moves.
2. Folders are read with directory listings instead of a call per file. One listing per folder gives every name,
   size and modification time: free on Windows, and Linux's NFS and SMB clients return them with the listing too.
3. No per-file `resolve()`. Folders are resolved once, and files found in them are already canonical.
4. Where many label files are read or written at once, that happens in parallel (a thread pool). Parsing stays in
   one thread under the lock.
5. Read-only endpoints do not touch the disk while holding the dataset lock.
6. Content decides write conflicts: the workbench remembers a digest of each label file it read or wrote.

## Steps

- [x] **Label QA** (`labelqa.py`): staleness compares the analysis with the image's known modification time, with
      no `stat` per image in listings, crops, apply or dismiss. Image keys are string prefixes, and an image
      analyzed under another id is found through one path index per listing.
- [x] **Model list** (`predictor.discover_models`): an `os.scandir` walk that does not enter the dataset's image
      and label folders, and asks nothing per file.
- [x] **Thumbnails, crops, Label QA crops** (`web.py`): cache keys come from the known image and label times
      (`Dataset.image_stamp`, `Dataset.label_mtime`), so a cached picture costs no round trip. The editor's
      `image_version` still stats the image (one call, next to downloading it).
- [x] **Scanner** (`fsscan.py`): `Listing`, one directory listing per folder (names, sizes, modification times),
      answering `exists`/`is_dir`/`stat` for files of listed folders, and walking folder trees without entering
      symlinked folders (as rglob). `read_text` reads a file and its time with one open.
- [x] **Opening and rescanning** (`dataset.py`): discovery, the label layout check, the size-index check and label
      existence come from the listing. Label files are read in parallel. Image records keep
      `image_mtime`/`image_size`. `.txt` lists resolve each folder once. Rescan now also notices a replaced image
      and probes its size again, which it did not do before.
- [x] **Orphan labels**: found from a listing and compared by normalized path, without `resolve()`.
- [x] **Size index**: probing takes the image's time and size from the open file, and saving the index needs no
      `stat`.
- [x] **Saving labels**: one `open` reads the current label and its time, and the backup path needs no
      `resolve()`. Folders are created only when a write finds them missing. The file just written is parsed from
      memory instead of read back.
- [x] **Class changes, bulk edits, undo/redo of many files**: label files are read, checked and written
      (backup + atomic write) in parallel when the paths are distinct. A new class change's check uses the content
      it was just built from.
- [x] **Write conflicts by content**: a new modification time with the same content is not a conflict (the known
      time is refreshed); the same time with different content is.
- [x] **Owner lock**: "locking is not supported here" (`ENOLCK`, `EOPNOTSUPP`, …) is logged as a warning and the
      folder opens. Only a lock held by another process refuses.
- [x] **Download**: label existence comes from memory, and files are opened (small ones read) ahead of the one
      being zipped.
- [x] **Import scan, storage summary, retention**: one listing per folder instead of `is_file()`/`stat` per file.
      The import reads label files in parallel and takes the dataset's image sizes from memory for duplicate
      detection.
- [x] **Docs**: README *Datasets on network drives*, `docs/server.md` *Folders*, CLAUDE.md.
- [x] **Tests**: `tests/fscount.py` counts (and can delay) filesystem calls under a folder.
      `tests/test_network_drive.py` checks that:
      - Label QA listings, cached thumbnails and the model list make no per-image calls;
      - opening and rescanning cost the same whatever the number of images, apart from one open per label;
      - conflicts are decided by content;
      - a replaced image is noticed by rescan;
      - parallel writes undo as one;
      - the download keeps its order;
      - the owner lock tolerates filesystems without locks.

## Behavior that changed

- An image replaced outside the app under the same name shows (thumbnails, Label QA "changed since the analysis")
  after **Rescan**, as edited labels already did. The editor's image URL still follows the file at once.
- Unreadable label files (bad UTF-8) are reported as an `unreadable_label` issue instead of failing to open the
  dataset.

## Not in scope

- Two workbench processes on different machines writing the same folder at once. The owner lock (which network
  filesystems with lock support honour across machines) still refuses the second one.
- Training reading images over the network: that is Ultralytics' / RF-DETR's data loader. **Cache images: RAM**
  (`cache: ram`) avoids re-reading every epoch when the dataset fits in memory. The docs say so.
- Copying files during an import: each file is still copied one after another (a background job with progress).
- Watching the network drive for outside changes: Rescan picks them up, as on a local disk.
- Not checked on a real share: the measurements above come from the simulation, not from an actual SMB or NFS
  mount.
