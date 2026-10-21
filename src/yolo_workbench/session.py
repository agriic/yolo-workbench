"""One opened dataset with the per-dataset services the API needs."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from . import identity
from .dataset import Dataset
from .embeddings import EmbeddingsManager
from .labelqa import LabelQAManager
from .predictor import UNLOAD_WAIT, PredictorManager

PRESENCE_TTL = 30.0  # seconds a browser counts as still having an image open after its last heartbeat


@dataclass
class DatasetSession:
    dataset: Dataset
    predictor: PredictorManager
    embeddings: EmbeddingsManager
    slug: str | None = None  # None in single-dataset mode
    project: dict = field(default_factory=dict)
    labelqa: LabelQAManager | None = None
    last_used: float = field(default_factory=time.monotonic)  # last API request (projects mode closes idle sessions)
    # who has which image open in the editor: browser tab -> {who, image_id, seen}; signed-in users only
    presence: dict = field(default_factory=dict)
    _presence_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def open(cls, dataset: Dataset, predictor: PredictorManager | None = None, slug: str | None = None, project: dict | None = None) -> "DatasetSession":
        predictor = predictor or PredictorManager(dataset)
        # a test may pass a stand-in predictor without the real manager's state; Label QA needs the real one
        labelqa = LabelQAManager(dataset, predictor) if isinstance(predictor, PredictorManager) else None
        return cls(dataset, predictor, EmbeddingsManager(dataset), slug, project or {}, labelqa)

    def unload_models(self, run: str) -> bool:
        """Free the GPU for training run `run`: stop Label QA's analysis and drop the prediction model."""
        if self.labelqa is not None:
            self.labelqa.stop(UNLOAD_WAIT)
        unload = getattr(self.predictor, "unload", None)  # a test's stand-in predictor may have none
        return bool(unload and unload(run))

    def touch(self) -> None:
        self.last_used = time.monotonic()

    def set_presence(self, image_id: str | None) -> None:
        """The requesting browser tab has `image_id` open in the editor (None: closed it)."""
        who, client = identity.who(), identity.current_client()
        if who is None or client is None:
            return
        with self._presence_lock:
            if image_id is None:
                self.presence.pop(client, None)
            else:
                self.presence[client] = {"who": who, "image_id": image_id, "seen": time.monotonic()}

    def others_present(self) -> dict[str, list[dict]]:
        """image id -> the other people who have it open (not the requester; each person once)."""
        me, my_name = identity.current_client(), identity.current_user_name()
        now = time.monotonic()
        found: dict[str, dict[str, dict]] = {}
        with self._presence_lock:
            for client, entry in list(self.presence.items()):
                if now - entry["seen"] > PRESENCE_TTL:
                    del self.presence[client]
                elif client != me and entry["who"]["name"] != my_name:
                    found.setdefault(entry["image_id"], {})[entry["who"]["name"]] = entry["who"]
        return {image_id: list(people.values()) for image_id, people in found.items()}

    def close(self, timeout: float = 5.0) -> None:
        """Stop this dataset's background work (prediction batch, Label QA, size probing) and drop its model.
        Embeddings cannot be interrupted; their thread is waited for up to `timeout` seconds."""
        if self.labelqa is not None:
            self.labelqa.stop(timeout)
        close = getattr(self.predictor, "close", None)
        if close is not None:
            close(timeout)
        close = getattr(self.embeddings, "close", None)
        if close is not None:
            close(timeout)
        self.dataset.close(timeout)

    def busy_reason(self) -> str | None:
        """Why this session cannot be closed right now, if anything is still running."""
        job = getattr(self.predictor, "job", None)
        if isinstance(job, dict) and job.get("state") == "running":
            return "a prediction run is in progress"
        if getattr(self.embeddings, "status", None) == "computing":
            return "embeddings are being computed"
        if self.labelqa is not None and self.labelqa.busy_reason():
            return "Label QA is analyzing images"
        return None
