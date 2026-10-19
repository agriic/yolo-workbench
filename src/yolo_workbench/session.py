"""One opened dataset with the per-dataset services the API needs."""

from __future__ import annotations

from dataclasses import dataclass, field

from .dataset import Dataset
from .embeddings import EmbeddingsManager
from .labelqa import LabelQAManager
from .predictor import UNLOAD_WAIT, PredictorManager


@dataclass
class DatasetSession:
    dataset: Dataset
    predictor: PredictorManager
    embeddings: EmbeddingsManager
    slug: str | None = None  # None in single-dataset mode
    project: dict = field(default_factory=dict)
    labelqa: LabelQAManager | None = None

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
