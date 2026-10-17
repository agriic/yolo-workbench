"""First imports of the heavy libraries (torch, cv2, torchvision, ultralytics, rfdetr, fiftyone), one thread at a time.

The server imports them lazily, often from request threads running side by side: the new-run form validates its
arguments (`ultralytics.cfg`) while its augmentation previews render (`ultralytics.data.augment`). Python's import
system cannot do that safely for packages whose modules import each other: two threads each holding one half of a
cycle trip its deadlock detection, which hands one of them a partially initialized module ("partially initialized
module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline'", "cannot import name 'set_module' from partially
initialized module 'torch.utils'"), and the half-imported package can stay broken until the server restarts.

Call `load()` with the modules a code path needs before its function-local imports; once they are imported it is a
dictionary lookup per name.
"""
from __future__ import annotations

import importlib
import sys
import threading

_lock = threading.RLock()  # reentrant: a guarded import may reach code that calls load() again


def _loaded(name: str) -> bool:
    module = sys.modules.get(name)
    return module is not None and not getattr(getattr(module, "__spec__", None), "_initializing", False)


def load(*names: str) -> None:
    """Import these modules (fully, before returning) unless they already are; another thread's import is waited for."""
    if all(_loaded(name) for name in names):
        return
    with _lock:
        for name in names:
            importlib.import_module(name)
