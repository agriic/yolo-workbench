from __future__ import annotations

import io
import json
import math
from pathlib import Path
from urllib.parse import quote
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from . import augment_preview, classes, importer, storage
from .dataset import Dataset, DatasetError, WriteConflict
from .imaging import upright_rgb
from .jobs import JobManager
from .media_cache import PALETTE_VERSION, MediaCache
from .models import Annotation, ImageRecord
from .predictor import PredictorManager
from .sam import SamManager
from .session import DatasetSession
from .training import TrainingManager, ultralytics_defaults
from .workspace import Workspace

# Keep in sync with PALETTE in static/api.js so server-rendered overlays match the editor.
PALETTE = ("#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#06b6d4", "#84cc16", "#a855f7", "#64748b")


def class_color(class_id: int) -> str:
    return PALETTE[class_id % len(PALETTE)]


class StrictPayload(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class AnnotationPayload(StrictPayload):
    id: str | None = None
    class_id: int
    points: list[float]


class AnnotationListPayload(StrictPayload):
    revision: int = Field(ge=1)
    annotations: list[AnnotationPayload]


class BulkFixPayload(BaseModel):
    kind: str
    split: str | None = None
    issue_ids: list[str] | None = None


class BulkObjectOperation(BaseModel):
    image_id: str
    annotation_id: str
    action: Literal["relabel", "delete"]
    class_id: int | None = None


class BulkObjectPayload(BaseModel):
    operations: list[BulkObjectOperation]


class PredictorLoadPayload(StrictPayload):
    path: str
    conf: float | None = Field(default=None, ge=0, le=1)
    iou: float | None = Field(default=None, ge=0, le=1)


class PredictorMappingPayload(BaseModel):
    mapping: dict[str, int | None]


class PredictorRunPayload(BaseModel):
    image_ids: list[str] | None = None
    split: str | None = None
    only_unlabeled: bool = False


class PredictionAcceptPayload(StrictPayload):
    prediction_ids: list[str] | None = None
    min_confidence: float | None = Field(default=None, ge=0, le=1)


class PredictionRejectPayload(BaseModel):
    prediction_ids: list[str] | None = None


class QARunPayload(BaseModel):
    split: str | None = None  # None / "all": every split


class SamPoint(StrictPayload):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    positive: bool = True


class SamSegmentPayload(StrictPayload):
    points: list[SamPoint] = Field(default_factory=list, max_length=64)
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)  # normalized x1, y1, x2, y2


class ClassAddPayload(BaseModel):
    names: list[str]


class ClassRenamePayload(BaseModel):
    name: str


class ClassRemovePayload(BaseModel):
    mode: Literal["delete", "merge"]
    target_id: int | None = None


class ClassMergePayload(BaseModel):
    source_ids: list[int]
    target_id: int


class ClassReorderPayload(BaseModel):
    order: list[int]


class UnknownClassPayload(BaseModel):
    action: Literal["map", "add", "delete"]
    target_id: int | None = None
    name: str | None = None


class LabelTextPayload(BaseModel):
    revision: int = Field(ge=1)
    text: str = Field(max_length=5_000_000)


class ImageIdsPayload(BaseModel):
    image_ids: list[str]


class ReviewPayload(BaseModel):
    image_ids: list[str]
    status: Literal["done", "needs_work"] | None = None  # None clears it


class ImageMovePayload(BaseModel):
    image_ids: list[str]
    split: Literal["train", "val", "test"]
    folder: str | None = None  # one of the split's folders; default: its first


class FolderAddPayload(BaseModel):
    split: Literal["train", "val", "test"]
    path: str


class ImportSourcePayload(BaseModel):
    path: str | None = None
    staging_id: str | None = None


class ImportCropPayload(BaseModel):
    mode: Literal["none", "center", "box"] = "none"  # center: width×height around the middle; box: the region at x, y
    unit: Literal["px", "percent"] = "px"
    x: float = Field(0, ge=0)
    y: float = Field(0, ge=0)
    width: float = Field(0, ge=0)
    height: float = Field(0, ge=0)


class ImportCommitPayload(ImportSourcePayload):
    split_mode: Literal["fixed", "auto", "source"] = "fixed"
    split: Literal["train", "val", "test"] = "train"
    ratios: dict[str, float] = Field(default_factory=lambda: {"train": 0.8, "val": 0.2, "test": 0.0})
    seed: int = 0
    mapping: dict[str, int | Literal["new"] | None] | None = None
    folders: dict[str, str] | None = None  # split -> target folder (as listed by GET /folders)
    skip_duplicates: bool = True
    crop: ImportCropPayload | None = None


class ProjectCreatePayload(BaseModel):
    name: str
    category: Literal["detection", "segmentation", "classification"]
    classes: list[str] = Field(default_factory=list)
    description: str = ""


class ProjectUpdatePayload(BaseModel):
    name: str | None = None
    description: str | None = None


class ProjectAdoptPayload(BaseModel):
    category: Literal["detection", "segmentation", "classification"]
    name: str | None = None


class TrainingArgsPayload(BaseModel):
    yaml: str = ""


class TrainingStartPayload(BaseModel):
    name: str = ""
    notes: str = ""
    yaml: str
    confirm: bool = False  # start despite preflight warnings


class TrainingRunUpdatePayload(BaseModel):
    name: str | None = None
    notes: str | None = None


class TrainingStopPayload(BaseModel):
    mode: Literal["pause", "finish", "kill"]


class TrainingReorderPayload(BaseModel):
    order: list[str]


MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
CSRF_HEADER = "x-workbench"


def dataset_router(get_session, cache: MediaCache, jobs: JobManager, training: TrainingManager, sam: SamManager) -> APIRouter:
    """Every per-dataset endpoint; mounted at /api/v1 (single mode) or /api/v1/projects/{slug}."""
    router = APIRouter()
    current = Depends(get_session)

    def label_mtime(record: ImageRecord) -> int:
        try:
            return record.label_path.stat().st_mtime_ns
        except OSError:
            return 0

    def image_mtime(record: ImageRecord) -> int:
        try:
            return record.path.stat().st_mtime_ns
        except OSError:
            return 0

    def require_image(session: DatasetSession, image_id: str) -> ImageRecord:
        try:
            return session.dataset.require_image(image_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    def found(call):
        try:
            return call()
        except KeyError as exc:
            raise HTTPException(404, str(exc.args[0]) if exc.args else "Not found") from exc

    @router.get("/dataset")
    def metadata(session: DatasetSession = current):
        meta = session.dataset.metadata()
        meta["training_lock"] = training.lock_info(session.dataset)
        if session.slug is not None:
            meta["project"] = session.project
        return meta

    @router.get("/statistics")
    def statistics(session: DatasetSession = current):
        return session.dataset.statistics()

    def image_filters(split: str = "all", class_id: int | None = None, search: str = "", has_predictions: bool = False, min_labels: int | None = Query(None, ge=0), max_labels: int | None = Query(None, ge=0), review: Literal["done", "needs_work", "unreviewed"] | None = None, issue: str | None = None, session: DatasetSession = current) -> dict:
        """The gallery filters, shared by the image list and "next unreviewed"."""
        prediction_ids = session.predictor.pending_image_ids() if has_predictions else None
        return {"split": split, "class_id": class_id, "search": search, "prediction_ids": prediction_ids, "min_labels": min_labels, "max_labels": max_labels, "review": review, "issue": issue or None}

    @router.get("/images")
    def images(filters: dict = Depends(image_filters), offset: int = 0, limit: int = Query(100, ge=1, le=500), session: DatasetSession = current):
        return session.dataset.list_images(offset=max(0, offset), limit=limit, **filters)

    @router.get("/images/next-unreviewed")
    def next_unreviewed(filters: dict = Depends(image_filters), start: int = Query(0, ge=0), session: DatasetSession = current):
        return session.dataset.next_unreviewed(start, **filters)

    @router.post("/images/review")
    def review_images(payload: ReviewPayload, session: DatasetSession = current):
        try:
            return session.dataset.set_review(payload.image_ids, payload.status)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.post("/images/delete")
    def delete_images(payload: ImageIdsPayload, session: DatasetSession = current):
        try:
            return session.dataset.delete_images(payload.image_ids)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.post("/images/move")
    def move_images(payload: ImageMovePayload, session: DatasetSession = current):
        try:
            return session.dataset.move_images(payload.image_ids, payload.split, payload.folder)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.get("/images/{image_id}")
    def image_detail(image_id: str, session: DatasetSession = current):
        try:
            detail = session.dataset.detail(image_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        detail["image_version"] = image_version(require_image(session, image_id))
        return detail

    @router.get("/images/{image_id}/file")
    def image_file(image_id: str, v: str | None = None, session: DatasetSession = current):
        record = require_image(session, image_id)
        # ?v= is the image_version from the image detail: a URL naming the current file may be kept,
        # anything else (no or an outdated version) is revalidated
        fresh = v is not None and v == image_version(record)
        return FileResponse(record.path, headers={"Cache-Control": "private, max-age=86400, immutable" if fresh else "no-cache"})

    @router.get("/images/{image_id}/thumbnail")
    def thumbnail(request: Request, image_id: str, size: int = Query(320, ge=64, le=1024), annotated: bool = False, session: DatasetSession = current):
        record = require_image(session, image_id)
        category = session.dataset.category
        key = cache.key("thumbnail", record.path, image_mtime(record), size, annotated, label_mtime(record) if annotated else 0, PALETTE_VERSION)
        if request.headers.get("if-none-match") == f'"{key}"':
            return Response(status_code=304)
        content = cache.get_or_render(key, lambda: render_thumbnail(record, size, category if annotated else None))
        return jpeg_response(content, etag=key)

    @router.get("/images/{image_id}/label-text")
    def label_text(image_id: str, session: DatasetSession = current):
        require_image(session, image_id)
        return session.dataset.label_text(image_id)

    @router.put("/images/{image_id}/label-text")
    def write_label_text(image_id: str, payload: LabelTextPayload, session: DatasetSession = current):
        require_image(session, image_id)
        return session.dataset.write_label_text(image_id, payload.text, payload.revision)

    @router.put("/images/{image_id}/annotations")
    def update_annotations(image_id: str, payload: AnnotationListPayload, session: DatasetSession = current):
        try:
            return session.dataset.replace_annotations(
                image_id,
                [annotation.model_dump() for annotation in payload.annotations],
                expected_revision=payload.revision,
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.get("/objects")
    def objects(class_id: int, split: str = "all", offset: int = 0, limit: int = Query(100, ge=1, le=500), session: DatasetSession = current):
        return session.dataset.list_objects(class_id, split, max(0, offset), limit)

    @router.get("/objects/{image_id}/{annotation_id:path}/crop")
    def object_crop(request: Request, image_id: str, annotation_id: str, padding: float = Query(0.15, ge=0, le=1), session: DatasetSession = current):
        record = require_image(session, image_id)
        annotation = next((item for item in record.annotations if item.id == annotation_id), None)
        if not annotation:
            raise HTTPException(404, "Annotation not found")
        category = session.dataset.category
        key = cache.key("crop", record.path, image_mtime(record), annotation.class_id, annotation.points, padding, PALETTE_VERSION)
        if request.headers.get("if-none-match") == f'"{key}"':
            return Response(status_code=304)
        content = cache.get_or_render(key, lambda: render_crop(record, annotation, category, padding))
        return jpeg_response(content, etag=key)

    @router.get("/issues")
    def issues(session: DatasetSession = current):
        return {"items": session.dataset.issues(), "unknown_classes": classes.unknown_classes(session.dataset)}

    @router.post("/issues/fix-bulk")
    def fix_issues_bulk(payload: BulkFixPayload, session: DatasetSession = current):
        return session.dataset.fix_issues_bulk(payload.kind, payload.split, payload.issue_ids)

    @router.post("/issues/{issue_id}/fix")
    def fix_issue(issue_id: str, session: DatasetSession = current):
        return session.dataset.fix_issue(issue_id)

    @router.post("/objects/bulk")
    def bulk_objects(payload: BulkObjectPayload, session: DatasetSession = current):
        return session.dataset.bulk_edit_objects([operation.model_dump() for operation in payload.operations])

    # ---------- classes ----------

    @router.get("/classes")
    def class_list(session: DatasetSession = current):
        return classes.class_summary(session.dataset)

    @router.post("/classes")
    def class_add(payload: ClassAddPayload, session: DatasetSession = current):
        return classes.add_classes(session.dataset, payload.names)

    @router.patch("/classes/{class_id}")
    def class_rename(class_id: int, payload: ClassRenamePayload, session: DatasetSession = current):
        return classes.rename_class(session.dataset, class_id, payload.name)

    @router.post("/classes/merge")
    def class_merge(payload: ClassMergePayload, session: DatasetSession = current):
        return classes.merge_classes(session.dataset, payload.source_ids, payload.target_id)

    @router.post("/classes/reorder")
    def class_reorder(payload: ClassReorderPayload, session: DatasetSession = current):
        return classes.reorder_classes(session.dataset, payload.order)

    @router.post("/classes/{class_id}/remove")
    def class_remove(class_id: int, payload: ClassRemovePayload, session: DatasetSession = current):
        return classes.remove_class(session.dataset, class_id, payload.mode, payload.target_id)

    @router.post("/classes/unknown/{class_id}")
    def class_resolve_unknown(class_id: int, payload: UnknownClassPayload, session: DatasetSession = current):
        return classes.resolve_unknown(session.dataset, class_id, payload.action, payload.target_id, payload.name)

    @router.get("/folders")
    def folder_list(session: DatasetSession = current):
        return session.dataset.folders()

    @router.post("/folders")
    def folder_add(payload: FolderAddPayload, session: DatasetSession = current):
        return session.dataset.add_folder(payload.split, payload.path)

    @router.post("/rescan")
    def rescan(session: DatasetSession = current):
        return session.dataset.rescan()

    # ---------- storage (trash and backups) ----------

    @router.get("/storage")
    def storage_summary(session: DatasetSession = current):
        return storage.summary(session.dataset)

    @router.post("/storage/trash/{session_id}/restore")
    def storage_restore(session_id: str, session: DatasetSession = current):
        return found(lambda: storage.restore_trash(session.dataset, session_id))

    @router.delete("/storage/{kind}/{session_id}")
    def storage_purge(kind: str, session_id: str, session: DatasetSession = current):
        return found(lambda: storage.purge(session.dataset, kind, session_id))

    # ---------- import ----------

    def require_import(session: DatasetSession) -> None:
        if not session.dataset.capabilities()["image_ops"]:
            raise DatasetError("This dataset's splits are not plain image folders, so images cannot be imported into it")

    @router.post("/import/staging")
    def staging_create(session: DatasetSession = current):
        require_import(session)
        return importer.create_staging(session.dataset)

    @router.put("/import/staging/{staging_id}/files")
    async def staging_upload(request: Request, staging_id: str, path: str, session: DatasetSession = current):
        content = await request.body()
        return importer.write_staged_file(session.dataset, staging_id, path, content)

    @router.delete("/import/staging/{staging_id}")
    def staging_discard(staging_id: str, session: DatasetSession = current):
        importer.discard_staging(session.dataset, staging_id)
        return {"discarded": staging_id}

    @router.post("/import/scan")
    def import_scan(payload: ImportSourcePayload, session: DatasetSession = current):
        require_import(session)
        return importer.scan(session.dataset, importer.resolve_source(session.dataset, payload.path, payload.staging_id))

    @router.get("/import/sample")
    def import_sample(file: str, path: str | None = None, staging_id: str | None = None, size: int = Query(640, ge=64, le=1280), session: DatasetSession = current):
        """One image of a scanned import (labels drawn) for the crop preview; X-Image-Size is its full upright size."""
        source = importer.resolve_source(session.dataset, path, staging_id)
        content, width, height = importer.sample_preview(source, file, session.dataset.category, size)
        response = jpeg_response(content)
        response.headers["X-Image-Size"] = f"{width}x{height}"
        return response

    @router.post("/import/commit")
    def import_commit(payload: ImportCommitPayload, session: DatasetSession = current):
        require_import(session)
        dataset = session.dataset
        dataset.check_writable()
        source = importer.resolve_source(dataset, payload.path, payload.staging_id)
        importer.crop_options(payload.crop.model_dump() if payload.crop else None)  # reject a bad crop before the job starts
        if jobs.running(session.slug):
            raise DatasetError("Another import is still running")
        options = payload.model_dump()

        def work(job):
            try:
                return importer.run_import(dataset, source, options, job)
            finally:
                if payload.staging_id:
                    importer.discard_staging(dataset, payload.staging_id)

        return jobs.start("import", session.slug, f"Import into {session.project.get('name', dataset.root.name)}", work)

    # ---------- model-assisted labeling ----------

    @router.get("/predictor")
    def predictor_state(session: DatasetSession = current):
        return session.predictor.payload()

    @router.get("/predictor/models")
    def predictor_models(session: DatasetSession = current):
        return session.predictor.discover_models()

    @router.post("/predictor/load")
    def predictor_load(payload: PredictorLoadPayload, session: DatasetSession = current):
        return session.predictor.load(payload.path, payload.conf, payload.iou)

    @router.put("/predictor/mapping")
    def predictor_mapping(payload: PredictorMappingPayload, session: DatasetSession = current):
        return session.predictor.set_mapping(payload.mapping)

    @router.post("/predictor/run")
    def predictor_run(payload: PredictorRunPayload, session: DatasetSession = current):
        return session.predictor.run(payload.image_ids, payload.split, payload.only_unlabeled)

    @router.post("/predictor/cancel")
    def predictor_cancel(session: DatasetSession = current):
        return session.predictor.cancel()

    @router.post("/predictor/retry")
    def predictor_retry(session: DatasetSession = current):
        return session.predictor.retry_failed()

    @router.post("/images/{image_id}/predictions/compute")
    def compute_predictions(image_id: str, session: DatasetSession = current):
        try:
            return session.predictor.predict_image(image_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.get("/images/{image_id}/predictions")
    def image_predictions(image_id: str, session: DatasetSession = current):
        try:
            return session.predictor.predictions_for(image_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.post("/images/{image_id}/predictions/accept")
    def accept_predictions(image_id: str, payload: PredictionAcceptPayload, session: DatasetSession = current):
        try:
            return session.predictor.accept(image_id, payload.prediction_ids, payload.min_confidence)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.post("/images/{image_id}/predictions/reject")
    def reject_predictions(image_id: str, payload: PredictionRejectPayload, session: DatasetSession = current):
        try:
            return session.predictor.reject(image_id, payload.prediction_ids)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    # ---------- label QA (model vs labels) ----------

    def labelqa(session: DatasetSession):
        if session.labelqa is None:
            raise DatasetError("Label QA is not available for this dataset")
        return session.labelqa

    @router.get("/qa")
    def qa_state(session: DatasetSession = current):
        return labelqa(session).payload()

    @router.post("/qa/run")
    def qa_run(payload: QARunPayload, session: DatasetSession = current):
        return labelqa(session).start(payload.split)

    @router.post("/qa/cancel")
    def qa_cancel(session: DatasetSession = current):
        return labelqa(session).cancel()

    @router.delete("/qa")
    def qa_clear(session: DatasetSession = current):
        return labelqa(session).clear()

    @router.get("/qa/findings")
    def qa_findings(kind: Literal["missing_label", "wrong_class", "box_mismatch", "unmatched_label"] | None = None, split: str = "all", min_conf: float = Query(0.5, ge=0, le=1), tolerance: float = Query(3, ge=0, le=500), offset: int = Query(0, ge=0), limit: int = Query(60, ge=1, le=500), session: DatasetSession = current):
        # tolerance: box edges this many pixels apart (or 5% of the object's size) count as the same box
        return labelqa(session).findings(kind, split, min_conf, offset, limit, tolerance)

    @router.get("/qa/images/{image_id}/findings/{key}/crop")
    def qa_crop(request: Request, image_id: str, key: str, session: DatasetSession = current):
        record = require_image(session, image_id)
        # the finding key encodes both shapes and their classes, so it names the picture
        cache_key = cache.key("qa", record.path, image_mtime(record), key, PALETTE_VERSION)
        if request.headers.get("if-none-match") == f'"{cache_key}"':
            return Response(status_code=304)
        content = cache.get_or_render(cache_key, lambda: labelqa(session).crop(image_id, key, class_color))
        return jpeg_response(content, etag=cache_key)

    @router.post("/qa/images/{image_id}/findings/{key}/apply")
    def qa_apply(image_id: str, key: str, session: DatasetSession = current):
        return found(lambda: labelqa(session).apply(image_id, key))

    @router.post("/qa/images/{image_id}/findings/{key}/dismiss")
    def qa_dismiss(image_id: str, key: str, session: DatasetSession = current):
        return found(lambda: labelqa(session).dismiss(image_id, key))

    # ---------- smart annotation (SAM) ----------

    @router.post("/images/{image_id}/sam/encode")
    def sam_encode(image_id: str, session: DatasetSession = current):
        return sam.encode(require_image(session, image_id).path)

    @router.post("/images/{image_id}/sam/segment")
    def sam_segment(image_id: str, payload: SamSegmentPayload, session: DatasetSession = current):
        record = require_image(session, image_id)
        points = [point.model_dump() for point in payload.points]
        return sam.segment(record.path, session.dataset.category, points, payload.box)

    @router.get("/embeddings")
    def embeddings_state(session: DatasetSession = current):
        return session.embeddings.payload()

    @router.post("/embeddings/compute")
    def embeddings_compute(session: DatasetSession = current):
        return session.embeddings.start()

    # ---------- training ----------

    @router.get("/training/options")
    def training_options(session: DatasetSession = current):
        return training.options(session.slug, session.dataset)

    @router.post("/training/args/validate")
    def training_validate(payload: TrainingArgsPayload, session: DatasetSession = current):
        return training.validate(session.slug, payload.yaml)

    @router.post("/training/preflight")
    def training_preflight(payload: TrainingArgsPayload, session: DatasetSession = current):
        return training.preflight(session.slug, session.dataset, training.validate(session.slug, payload.yaml)["args"])

    @router.get("/training/augment/original")
    def augment_original(sample: int = Query(0, ge=0), imgsz: int = Query(640, ge=32, le=4096), size: int = Query(256, ge=64, le=1024), labels: bool = True, session: DatasetSession = current):
        return augment_response(lambda: augment_preview.render_original(session.dataset, sample, imgsz, size, labels, class_color))

    @router.get("/training/augment/preview")
    def augment_preview_image(key: str, side: Literal["min", "max"], args: str = "{}", sample: int = Query(0, ge=0), size: int = Query(256, ge=64, le=1024), labels: bool = True, session: DatasetSession = current):
        values, imgsz = augment_args(args)
        return augment_response(lambda: augment_preview.render_augmentation(session.dataset, key, side, values, ultralytics_defaults()[0], sample, imgsz, size, labels, class_color))

    @router.get("/training/augment/sample")
    def augment_sample(args: str = "{}", sample: int = Query(0, ge=0), seed: int = Query(0, ge=0), size: int = Query(256, ge=64, le=1024), labels: bool = True, session: DatasetSession = current):
        values, imgsz = augment_args(args)
        return augment_response(lambda: augment_preview.render_pipeline(session.dataset, values, ultralytics_defaults()[0], sample, seed, imgsz, size, labels, class_color))

    @router.get("/training/runs")
    def training_runs(session: DatasetSession = current):
        return training.list_runs(session.slug)

    @router.post("/training/runs")
    def training_start(payload: TrainingStartPayload, session: DatasetSession = current):
        return training.enqueue(session.slug, session.dataset, payload.name, payload.notes, payload.yaml, payload.confirm)

    @router.get("/training/runs/{run_id}")
    def training_run(run_id: str, session: DatasetSession = current):
        return found(lambda: training.run_detail(session.slug, run_id))

    @router.patch("/training/runs/{run_id}")
    def training_run_update(run_id: str, payload: TrainingRunUpdatePayload, session: DatasetSession = current):
        return found(lambda: training.update_run(session.slug, run_id, payload.name, payload.notes))

    @router.delete("/training/runs/{run_id}")
    def training_run_delete(run_id: str, session: DatasetSession = current):
        return found(lambda: training.delete_run(session.slug, run_id))

    @router.get("/training/runs/{run_id}/epochs")
    def training_epochs(run_id: str, since: int = Query(0, ge=0), session: DatasetSession = current):
        return found(lambda: training.epochs(session.slug, run_id, since))

    @router.get("/training/runs/{run_id}/log")
    def training_log(run_id: str, offset: int | None = Query(None, ge=0), session: DatasetSession = current):
        return found(lambda: training.log(session.slug, run_id, offset))

    @router.get("/training/runs/{run_id}/args")
    def training_args(run_id: str, session: DatasetSession = current):
        return found(lambda: training.copy_args(session.slug, run_id))

    @router.get("/training/runs/{run_id}/files/{name:path}")
    def training_file(run_id: str, name: str, session: DatasetSession = current):
        path = found(lambda: training.run_file(session.slug, run_id, name))
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @router.get("/training/runs/{run_id}/weights/best")
    def training_best_weights(run_id: str, session: DatasetSession = current):
        path, filename = found(lambda: training.best_weights(session.slug, run_id))
        return FileResponse(path, media_type="application/octet-stream", filename=filename, headers={"Cache-Control": "no-cache"})

    @router.post("/training/runs/{run_id}/stop")
    def training_stop(run_id: str, payload: TrainingStopPayload, session: DatasetSession = current):
        return found(lambda: training.stop(session.slug, run_id, payload.mode))

    @router.post("/training/runs/{run_id}/resume")
    def training_resume(run_id: str, session: DatasetSession = current):
        return found(lambda: training.resume(session.slug, run_id))

    @router.post("/training/runs/{run_id}/cancel")
    def training_cancel(run_id: str, session: DatasetSession = current):
        return found(lambda: training.cancel(session.slug, run_id))

    @router.post("/history/{direction}")
    def history(direction: str, session: DatasetSession = current):
        if direction not in {"undo", "redo"}:
            raise HTTPException(400, "Direction must be undo or redo")
        return session.dataset.history(direction)

    return router


def _base_app(mode: str, jobs: JobManager, allowed_hosts: set[str] | None, require_header: bool) -> FastAPI:
    app = FastAPI(title="YOLO Dataset Workbench", version="0.1.0")
    app.state.jobs = jobs

    @app.exception_handler(DatasetError)
    async def dataset_error(_, exc: DatasetError):
        return Response(content=f'{{"detail": {json_string(str(exc))}}}', status_code=400, media_type="application/json")

    @app.exception_handler(WriteConflict)
    async def conflict_error(_, exc: WriteConflict):
        return Response(content=f'{{"detail": {json_string(str(exc))}}}', status_code=409, media_type="application/json")

    @app.exception_handler(RequestValidationError)
    async def validation_error(_, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": json_safe(jsonable_encoder(exc.errors()))})

    @app.get("/api/v1/app")
    def app_info():
        return {"mode": mode, "version": app.version}

    @app.get("/api/v1/jobs")
    def job_list(project: str | None = None):
        return {"items": jobs.list(project)}

    @app.get("/api/v1/jobs/{job_id}")
    def job_detail(job_id: str):
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        try:
            return jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # The API can write anywhere in the dataset (and the projects folder): refuse requests
        # that a web page in the same browser could forge (DNS rebinding, cross-site POSTs).
        if allowed_hosts is not None:
            host = (request.headers.get("host") or "").rsplit(":", 1)[0] if not (request.headers.get("host") or "").startswith("[") else (request.headers.get("host") or "").split("]")[0] + "]"
            if host not in allowed_hosts:
                return JSONResponse(status_code=403, content={"detail": "Host not allowed"})
        if require_header and request.method in MUTATING_METHODS and request.url.path.startswith("/api/") and request.headers.get(CSRF_HEADER) != "1":
            return JSONResponse(status_code=403, content={"detail": "Missing X-Workbench header"})
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            # the UI ships with the package; stale cached scripts break new endpoints
            response.headers["Cache-Control"] = "no-cache"
        return response

    return app


def _training_routes(app: FastAPI, training: TrainingManager) -> None:
    """The queue is shared by every dataset of the app (one GPU)."""
    app.state.training = training

    @app.get("/api/v1/training/queue")
    def training_queue():
        return training.queue()

    @app.post("/api/v1/training/queue/reorder")
    def training_reorder(payload: TrainingReorderPayload):
        return training.reorder(payload.order)


def _mount_static(app: FastAPI) -> None:
    static = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static, html=True), name="static")


def _sam_routes(app: FastAPI, sam: SamManager) -> None:
    """One SAM model for the whole app (in projects mode it outlives switching projects)."""
    app.state.sam = sam

    @app.get("/api/v1/sam")
    def sam_state():
        return sam.payload()

    @app.post("/api/v1/sam/start")
    def sam_start():
        return sam.start()


def create_app(dataset: Dataset, media_cache: MediaCache | None = None, predictor: PredictorManager | None = None, allowed_hosts: set[str] | None = None, require_header: bool = False, jobs: JobManager | None = None, training: TrainingManager | None = None, sam: SamManager | None = None) -> FastAPI:
    """Single-dataset mode: the dataset API lives directly under /api/v1."""
    jobs = jobs or JobManager()
    app = _base_app("single", jobs, allowed_hosts, require_header)
    training = training or TrainingManager.for_dataset(dataset, jobs=jobs)
    training.attach(dataset)
    _training_routes(app, training)
    sam = sam or SamManager()
    _sam_routes(app, sam)
    session = DatasetSession.open(dataset, predictor)
    app.state.dataset = dataset
    app.state.session = session
    app.state.predictor = session.predictor
    app.state.embeddings = session.embeddings
    cache = media_cache or MediaCache()
    app.state.media_cache = cache
    app.include_router(dataset_router(lambda: session, cache, jobs, training, sam), prefix="/api/v1")
    _mount_static(app)
    return app


def create_workspace_app(workspace: Workspace, media_cache: MediaCache | None = None, allowed_hosts: set[str] | None = None, require_header: bool = False, sam: SamManager | None = None) -> FastAPI:
    """Projects mode: workspace endpoints plus the dataset API under /api/v1/projects/{slug}."""
    app = _base_app("projects", workspace.jobs, allowed_hosts, require_header)
    app.state.workspace = workspace
    _training_routes(app, workspace.training)
    sam = sam or SamManager()
    _sam_routes(app, sam)
    cache = media_cache or MediaCache()
    app.state.media_cache = cache

    def project_session(slug: str) -> DatasetSession:
        try:
            return workspace.session(slug)
        except KeyError as exc:
            raise HTTPException(404, "Project not found") from exc

    def not_found(call):
        try:
            return call()
        except KeyError as exc:
            raise HTTPException(404, str(exc.args[0]) if exc.args else "Not found") from exc

    @app.get("/api/v1/projects")
    def project_list():
        return workspace.list_projects()

    @app.post("/api/v1/projects")
    def project_create(payload: ProjectCreatePayload):
        return workspace.create(payload.name, payload.category, payload.classes, payload.description)

    @app.patch("/api/v1/projects/{slug}")
    def project_update(slug: str, payload: ProjectUpdatePayload):
        return not_found(lambda: workspace.update(slug, payload.name, payload.description))

    @app.delete("/api/v1/projects/{slug}")
    def project_trash(slug: str):
        return not_found(lambda: workspace.trash(slug))

    @app.post("/api/v1/projects/{slug}/adopt")
    def project_adopt(slug: str, payload: ProjectAdoptPayload):
        return not_found(lambda: workspace.adopt(slug, payload.category, payload.name))

    @app.get("/api/v1/projects/{slug}/cover")
    def project_cover(slug: str):
        path = not_found(lambda: workspace.cover(slug))
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    @app.get("/api/v1/trash")
    def trash_list():
        return {"items": workspace.list_trash()}

    @app.post("/api/v1/trash/{entry}/restore")
    def trash_restore(entry: str):
        return not_found(lambda: workspace.restore(entry))

    @app.delete("/api/v1/trash/{entry}")
    def trash_purge(entry: str):
        return not_found(lambda: workspace.purge(entry))

    app.include_router(dataset_router(project_session, cache, workspace.jobs, workspace.training, sam), prefix="/api/v1/projects/{slug}")
    _mount_static(app)
    return app


def image_version(record: ImageRecord) -> str:
    """Changes whenever the image file does; the editor puts it in the image URL so the browser can cache it."""
    try:
        stat = record.path.stat()
    except OSError:
        return "0"
    return f"{stat.st_mtime_ns:x}-{stat.st_size:x}"


def json_string(value: str) -> str:
    import json

    return json.dumps(value)


def json_safe(value):
    """Replace non-finite floats so validation details remain valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def render_thumbnail(record: ImageRecord, size: int, category: str | None = None) -> bytes:
    with Image.open(record.path) as source:
        image = upright_rgb(source)
        image.thumbnail((size, size))
        if category:
            draw = ImageDraw.Draw(image)
            for annotation in record.annotations:
                draw_annotation(draw, annotation, category, image.width, image.height)
        output = io.BytesIO()
        image.save(output, "JPEG", quality=82)
        return output.getvalue()


def draw_annotation(draw: ImageDraw.ImageDraw, annotation: Annotation, category: str, width: int, height: int) -> None:
    if category == "classification":
        return  # the class covers the whole image; there is no geometry to draw
    color = class_color(annotation.class_id)
    if category == "detection":
        cx, cy, box_width, box_height = annotation.points
        xs = sorted(((cx - box_width / 2) * width, (cx + box_width / 2) * width))
        ys = sorted(((cy - box_height / 2) * height, (cy + box_height / 2) * height))
        draw.rectangle((xs[0], ys[0], xs[1], ys[1]), outline=color, width=2)
    else:
        points = [(x * width, y * height) for x, y in zip(annotation.points[::2], annotation.points[1::2])]
        draw.line(points + points[:1], fill=color, width=2)


def render_crop(record, annotation, category: str, padding: float) -> bytes:
    with Image.open(record.path) as source:
        image = upright_rgb(source)
        if category == "classification":
            image.thumbnail((420, 320))
            output = io.BytesIO()
            image.save(output, "JPEG", quality=88)
            return output.getvalue()
        if category == "detection":
            cx, cy, width, height = annotation.points
            left, top, right, bottom = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
        else:
            xs, ys = annotation.points[::2], annotation.points[1::2]
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        pad_x, pad_y = (right - left) * padding, (bottom - top) * padding
        box = (max(0, int((left - pad_x) * image.width)), max(0, int((top - pad_y) * image.height)), min(image.width, int((right + pad_x) * image.width)), min(image.height, int((bottom + pad_y) * image.height)))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            # zero-area or out-of-range annotations still need a visible crop so they can be reviewed
            cx = min(max(int((left + right) / 2 * image.width), 0), image.width)
            cy = min(max(int((top + bottom) / 2 * image.height), 0), image.height)
            box = (max(0, cx - 12), max(0, cy - 12), min(image.width, cx + 12), min(image.height, cy + 12))
        crop = image.crop(box)
        draw = ImageDraw.Draw(crop)
        color = class_color(annotation.class_id)
        if category == "detection":
            draw.rectangle(((left * image.width - box[0], top * image.height - box[1]), (right * image.width - box[0], bottom * image.height - box[1])), outline=color, width=3)
        else:
            polygon = [(x * image.width - box[0], y * image.height - box[1]) for x, y in zip(annotation.points[::2], annotation.points[1::2])]
            draw.line(polygon + polygon[:1], fill=color, width=3)
        crop.thumbnail((420, 320))
        output = io.BytesIO()
        crop.save(output, "JPEG", quality=88)
        return output.getvalue()


def augment_args(text: str) -> tuple[dict, int]:
    """Training arguments for an augmentation preview (a JSON object in the query) and the image size they train at."""
    try:
        values = json.loads(text)
    except ValueError as exc:
        raise HTTPException(400, "args must be a JSON object") from exc
    if not isinstance(values, dict) or any(not (value is None or isinstance(value, (str, int, float))) or isinstance(value, float) and not math.isfinite(value) for value in values.values()):
        raise HTTPException(400, "args must map argument names to numbers or strings")
    imgsz = values.get("imgsz", ultralytics_defaults()[0].get("imgsz", 640))
    if isinstance(imgsz, bool) or not isinstance(imgsz, (int, float)) or not 32 <= imgsz <= 4096:
        raise HTTPException(400, "imgsz must be a number between 32 and 4096")
    return values, int(imgsz)


def augment_response(render) -> Response:
    try:
        preview = render()
    except (AssertionError, TypeError, ValueError) as exc:  # Ultralytics/torchvision reject out-of-range arguments this way
        raise HTTPException(400, f"Cannot preview these values: {exc}") from exc
    # rendered from the current images and labels: kept briefly, never shared
    return Response(preview.jpeg, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300", "X-Preview-Note": quote(preview.note)})


def jpeg_response(content: bytes, etag: str | None = None) -> Response:
    if etag is None:
        return Response(content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    # no-cache (not no-store): the browser keeps the bytes but revalidates via ETag, so edits show immediately
    return Response(content, media_type="image/jpeg", headers={"Cache-Control": "private, no-cache", "ETag": f'"{etag}"'})
