"""One opened dataset with the per-dataset services the API needs."""

from __future__ import annotations

from dataclasses import dataclass, field

from .dataset import Dataset
from .embeddings import EmbeddingsManager
from .predictor import PredictorManager


@dataclass
class DatasetSession:
    dataset: Dataset
    predictor: PredictorManager
    embeddings: EmbeddingsManager
    slug: str | None = None  # None in single-dataset mode
    project: dict = field(default_factory=dict)

    @classmethod
    def open(cls, dataset: Dataset, predictor: PredictorManager | None = None, slug: str | None = None, project: dict | None = None) -> "DatasetSession":
        return cls(dataset, predictor or PredictorManager(dataset), EmbeddingsManager(dataset), slug, project or {})

    def busy_reason(self) -> str | None:
        """Why this session cannot be closed right now, if anything is still running."""
        job = getattr(self.predictor, "job", None)
        if isinstance(job, dict) and job.get("state") == "running":
            return "a prediction run is in progress"
        if getattr(self.embeddings, "status", None) == "computing":
            return "embeddings are being computed"
        return None
