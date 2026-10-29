from __future__ import annotations

import io
import json
import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote
from typing import Callable, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from . import augment_preview, classes, download, identity, importer, rfdetr_augment, storage
from .auth import Auth, AuthError, AuthMiddleware, current_user
from .dataset import Dataset, DatasetError, SchemaChanged, WriteConflict
from .fsutil import InsufficientSpace, ensure_free_space, replace
from .imaging import upright_rgb
from .importer import UploadTooLarge
from .jobs import JobManager
from .lifecycle import Lifecycle, LifecycleOptions
from .media_cache import PALETTE_VERSION, MediaCache
from .models import Annotation, ImageRecord
from .predictor import PredictorManager, free_gpu_memory, model_upload_target
from .release import release_info, release_name
from .runtime import RUNTIME
from .sam import SamManager
from .session import DatasetSession
from .training import TrainingManager, ultralytics_defaults
from .workspace import Workspace

log = logging.getLogger(__name__)

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
    schema_generation: int | None = None  # the class list the class ids were chosen from (409 when it changed)


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
    schema_generation: int | None = None


class PredictorLoadPayload(StrictPayload):
    path: str
    conf: float | None = Field(default=None, ge=0, le=1)
    iou: float | None = Field(default=None, ge=0, le=1)


class PredictorOptionsPayload(StrictPayload):
    max_points: int | None = Field(default=None, ge=3, le=10000)


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
    max_points: int | None = Field(default=None, ge=3, le=10000)  # segmentation: polygon vertex limit


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
    schema_generation: int | None = None


class PresencePayload(BaseModel):
    image_id: str | None = None  # None: the editor closed


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


# the split folders a run trains on, as the dataset's folder list names them; None = the whole dataset
TrainingFolders = dict[Literal["train", "val", "test"], list[str]] | None


class TrainingPreflightPayload(TrainingArgsPayload):
    folders: TrainingFolders = None


class TrainingStartPayload(BaseModel):
    name: str = ""
    notes: str = ""
    yaml: str
    confirm: bool = False  # start despite preflight warnings
    folders: TrainingFolders = None


class TrainingRunUpdatePayload(BaseModel):
    name: str | None = None
    notes: str | None = None


class TrainingStopPayload(BaseModel):
    mode: Literal["pause", "finish", "kill"]


class TrainingExportPayload(BaseModel):
    imgsz: int | list[int] | None = None  # None: the run's imgsz
    dynamic: bool = False
    simplify: bool = True
    nms: bool = False


class TrainingReorderPayload(BaseModel):
    order: list[str]


class TrainingHoldPayload(BaseModel):
    held: bool
    pause_active: bool = False  # also pause the active run at its next checkpoint; it resumes once released


class LoginPayload(BaseModel):
    name: str = Field(max_length=200)
    password: str = Field(max_length=1000)


class PasswordPayload(BaseModel):
    current: str = Field(max_length=1000)
    new: str = Field(max_length=1000)


MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
CSRF_HEADER = "x-workbench"
HEALTH_PATHS = {"/healthz", "/readyz"}


async def receive_file(request: Request, target: Path, limit: int, commit) -> dict | Response:
    """Stream a request body into a temporary file beside `target` (413 past `limit` bytes, nothing kept on an
    error or a dropped connection) and hand the complete file to `commit(partial, size)`."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise UploadTooLarge(f"{target.name} is larger than this server accepts ({limit / 2**20:,.0f} MB at most)")
    await run_in_threadpool(ensure_free_space, target.parent, int(declared) if declared and declared.isdigit() else 0, "the upload")
    partial = await run_in_threadpool(importer.partial_path, target)
    size = 0
    try:
        stream = await run_in_threadpool(open, partial, "wb")
        try:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise UploadTooLarge(f"{target.name} is larger than this server accepts ({limit / 2**20:,.0f} MB at most)")
                await run_in_threadpool(stream.write, chunk)
        finally:
            await run_in_threadpool(stream.close)
        return await run_in_threadpool(commit, partial, size)
    except ClientDisconnect:
        return JSONResponse(status_code=400, content={"detail": "The upload was interrupted"})
    finally:
        await run_in_threadpool(lambda: partial.unlink(missing_ok=True))


def dataset_router(get_session, cache: MediaCache, jobs: JobManager, training: TrainingManager, sam: SamManager, lifecycle: Callable[[], Lifecycle | None] = lambda: None, peek_session=None) -> APIRouter:
    """Every per-dataset endpoint; mounted at /api/v1 (single mode) or /api/v1/projects/{slug}.
    `peek_session` is the open session or None, without opening or touching it (the change feed's polls)."""
    router = APIRouter()
    current = Depends(get_session)
    peeked = Depends(peek_session or get_session)

    # cache keys from the times the dataset knows (its own writes, opening, rescan): a thumbnail served from the
    # cache asks the dataset's disk nothing, which on a network drive is a round trip per image
    def label_mtime(session: DatasetSession, record: ImageRecord) -> int:
        return session.dataset.label_mtime(record)

    def image_mtime(session: DatasetSession, record: ImageRecord) -> int:
        stamp = session.dataset.image_stamp(record)
        return stamp[0] if stamp else 0

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

    def image_filters(split: str = "all", class_id: int | None = None, search: str = "", has_predictions: bool = False, min_labels: int | None = Query(None, ge=0), max_labels: int | None = Query(None, ge=0), review: Literal["done", "needs_work", "unreviewed"] | None = None, issue: str | None = None, reviewer: str | None = None, folder: str | None = None, session: DatasetSession = current) -> dict:
        """The gallery filters, shared by the image list and "next unreviewed"."""
        prediction_ids = session.predictor.pending_image_ids() if has_predictions else None
        return {"split": split, "class_id": class_id, "search": search, "prediction_ids": prediction_ids, "min_labels": min_labels, "max_labels": max_labels, "review": review, "issue": issue or None, "reviewer": reviewer or None, "folder": folder or None}

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
        key = cache.key("thumbnail", record.path, image_mtime(session, record), size, annotated, label_mtime(session, record) if annotated else 0, PALETTE_VERSION)
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
        return session.dataset.write_label_text(image_id, payload.text, payload.revision, payload.schema_generation)

    @router.put("/images/{image_id}/annotations")
    def update_annotations(image_id: str, payload: AnnotationListPayload, session: DatasetSession = current):
        try:
            return session.dataset.replace_annotations(
                image_id,
                [annotation.model_dump() for annotation in payload.annotations],
                expected_revision=payload.revision,
                schema_generation=payload.schema_generation,
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    # ---------- several people in one project ----------

    @router.get("/changes")
    def changes(since: int = Query(0, ge=0), feed: str | None = None, session: DatasetSession | None = peeked):
        """What others changed since revision `since` of `feed`, and who has which image open (polled every few
        seconds). Polling neither opens a closed project nor keeps an idle one open: `closed` says it is closed."""
        if session is None:
            return {"closed": True, "feed": None, "revision": 0, "reset": False, "structure": False, "schema": False, "images": {}, "presence": {}}
        return {**session.dataset.changes(since, feed), "presence": session.others_present()}

    @router.post("/presence")
    def presence(payload: PresencePayload, session: DatasetSession = current):
        """The editor of this browser tab shows `image_id` (heartbeat), or closed (None)."""
        if payload.image_id is not None:
            require_image(session, payload.image_id)
        session.set_presence(payload.image_id)
        return {"presence": session.others_present()}

    @router.get("/activity")
    def activity(limit: int = Query(100, ge=1, le=1000), image_id: str | None = None, user: str | None = None, session: DatasetSession = current):
        return {"items": session.dataset.recent_activity(limit, image_id, user)}

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
        key = cache.key("crop", record.path, image_mtime(session, record), annotation.class_id, annotation.points, padding, PALETTE_VERSION)
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
        return session.dataset.bulk_edit_objects([operation.model_dump() for operation in payload.operations], payload.schema_generation)

    # ---------- classes ----------

    @router.get("/classes")
    def class_list(session: DatasetSession = current):
        return classes.class_summary(session.dataset)

    @router.post("/classes")
    def class_add(payload: ClassAddPayload, session: DatasetSession = current):
        with identity.action("add classes"):
            return classes.add_classes(session.dataset, payload.names)

    @router.patch("/classes/{class_id}")
    def class_rename(class_id: int, payload: ClassRenamePayload, session: DatasetSession = current):
        with identity.action("rename class"):
            return classes.rename_class(session.dataset, class_id, payload.name)

    @router.post("/classes/merge")
    def class_merge(payload: ClassMergePayload, session: DatasetSession = current):
        with identity.action("merge classes"):
            return classes.merge_classes(session.dataset, payload.source_ids, payload.target_id)

    @router.post("/classes/reorder")
    def class_reorder(payload: ClassReorderPayload, session: DatasetSession = current):
        with identity.action("reorder classes"):
            return classes.reorder_classes(session.dataset, payload.order)

    @router.post("/classes/{class_id}/remove")
    def class_remove(class_id: int, payload: ClassRemovePayload, session: DatasetSession = current):
        with identity.action("remove class"):
            return classes.remove_class(session.dataset, class_id, payload.mode, payload.target_id)

    @router.post("/classes/unknown/{class_id}")
    def class_resolve_unknown(class_id: int, payload: UnknownClassPayload, session: DatasetSession = current):
        with identity.action("resolve unknown class"):
            return classes.resolve_unknown(session.dataset, class_id, payload.action, payload.target_id, payload.name)

    @router.get("/folders")
    def folder_list(session: DatasetSession = current):
        return session.dataset.folders()

    @router.post("/folders")
    def folder_add(payload: FolderAddPayload, session: DatasetSession = current):
        with identity.action("add folder"):
            return session.dataset.add_folder(payload.split, payload.path)

    @router.post("/rescan")
    def rescan(session: DatasetSession = current):
        with identity.action("rescan"):
            return session.dataset.rescan()

    # ---------- storage (trash and backups) ----------

    @router.get("/download")
    def download_dataset(kind: download.Kind = "full", session: DatasetSession = current):
        entries = download.plan(session.dataset, kind)
        filename = download.archive_name(session.slug or session.dataset.root.name or "dataset", kind)
        log.info("download %s of %s: %d files", kind, session.slug or session.dataset.root, len(entries))
        return StreamingResponse(download.stream(entries), media_type="application/zip", headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
            "Cache-Control": "no-store",
        })

    @router.get("/storage")
    def storage_summary(session: DatasetSession = current):
        return storage.summary(session.dataset)

    @router.post("/storage/trash/{session_id}/restore")
    def storage_restore(session_id: str, session: DatasetSession = current):
        with identity.action("restore trash"):
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
        """One uploaded file, streamed to disk (the body is the file)."""
        root, target, relative = await run_in_threadpool(importer.staged_target, session.dataset, staging_id, path)
        limit = await run_in_threadpool(importer.upload_budget, root)
        return await receive_file(request, target, limit, lambda partial, size: importer.commit_staged_file(root, partial, target, size, relative))

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
        if jobs.running(session.slug, "import"):
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

    def predictor_payload(session: DatasetSession, payload: dict) -> dict:
        app_lifecycle = lifecycle()
        payload["gpu_warning"] = app_lifecycle.gpu_warning(payload.get("status") == "ready") if app_lifecycle else None
        return payload

    def with_model(session: DatasetSession, call):
        """A prediction that may load a parked model again (it sat idle): that load follows the GPU policy and
        counts toward max_loaded_models like a load by hand."""
        parked = getattr(session.predictor, "status", None) == "parked"
        app_lifecycle = lifecycle()
        if parked and app_lifecycle is not None:
            refused = app_lifecycle.refuse_gpu_load()
            if refused:
                raise DatasetError(refused)
        result = call()
        if parked and app_lifecycle is not None and getattr(session.predictor, "backend", None) is not None:
            app_lifecycle.enforce_model_limit(session)
        return result

    @router.get("/predictor")
    def predictor_state(session: DatasetSession = current):
        return predictor_payload(session, session.predictor.payload())

    @router.get("/predictor/models")
    def predictor_models(session: DatasetSession = current):
        return session.predictor.discover_models()

    @router.post("/predictor/load")
    def predictor_load(payload: PredictorLoadPayload, session: DatasetSession = current):
        app_lifecycle = lifecycle()
        refused = app_lifecycle.refuse_gpu_load() if app_lifecycle else None
        if refused:
            raise DatasetError(refused)
        result = session.predictor.load(payload.path, payload.conf, payload.iou)
        if app_lifecycle is not None:
            app_lifecycle.enforce_model_limit(session)
        return predictor_payload(session, result)

    @router.put("/models/upload")
    async def model_upload(request: Request, name: str, session: DatasetSession = current):
        """A model file (.pt, .pth, .onnx) streamed into the dataset's models/ folder; load it by the returned path."""
        target = await run_in_threadpool(model_upload_target, session.dataset, name)

        def commit(partial: Path, size: int) -> dict:
            replace(partial, target)
            log_model_change(session, f"upload model {target.name}")
            return {"path": target.as_posix(), "name": target.name, "size": size}

        return await receive_file(request, target, RUNTIME.max_model_upload_bytes, commit)

    def log_model_change(session: DatasetSession, action: str) -> None:
        who = identity.who()
        if who is not None and session.dataset.activity is not None:
            session.dataset.activity.append(who, action, [], kind="models")

    @router.get("/models")
    def folder_models(session: DatasetSession = current):
        return session.predictor.folder_models()

    @router.delete("/models/{name}")
    def delete_folder_model(name: str, session: DatasetSession = current):
        try:
            path = session.predictor.folder_model(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        run = training.started_from(path)
        if run:
            raise DatasetError(f"Training run '{run}' is queued to start from {name}; cancel it or wait until it finishes")
        result = session.predictor.delete_model(name)
        log_model_change(session, f"delete model {name}")
        return result

    @router.put("/predictor/options")
    def predictor_options(payload: PredictorOptionsPayload, session: DatasetSession = current):
        return session.predictor.set_options(payload.max_points)

    @router.put("/predictor/mapping")
    def predictor_mapping(payload: PredictorMappingPayload, session: DatasetSession = current):
        return session.predictor.set_mapping(payload.mapping)

    @router.post("/predictor/run")
    def predictor_run(payload: PredictorRunPayload, session: DatasetSession = current):
        return predictor_payload(session, with_model(session, lambda: session.predictor.run(payload.image_ids, payload.split, payload.only_unlabeled)))

    @router.post("/predictor/cancel")
    def predictor_cancel(session: DatasetSession = current):
        return session.predictor.cancel()

    @router.post("/predictor/retry")
    def predictor_retry(session: DatasetSession = current):
        return predictor_payload(session, with_model(session, session.predictor.retry_failed))

    @router.post("/predictor/accept-all")
    def predictor_accept_all(session: DatasetSession = current):
        with identity.action("accept all predictions"):
            result = session.predictor.accept_all()
        return {**result, "predictor": predictor_payload(session, session.predictor.payload())}

    @router.post("/predictor/reject-all")
    def predictor_reject_all(session: DatasetSession = current):
        result = session.predictor.reject_all()
        return {**result, "predictor": predictor_payload(session, session.predictor.payload())}

    @router.post("/images/{image_id}/predictions/compute")
    def compute_predictions(image_id: str, session: DatasetSession = current):
        try:
            return with_model(session, lambda: session.predictor.predict_image(image_id))
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
            with identity.action("accept predictions"):
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
        return with_model(session, lambda: labelqa(session).start(payload.split))

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
        cache_key = cache.key("qa", record.path, image_mtime(session, record), key, session.predictor.max_points, PALETTE_VERSION)
        if request.headers.get("if-none-match") == f'"{cache_key}"':
            return Response(status_code=304)
        content = cache.get_or_render(cache_key, lambda: labelqa(session).crop(image_id, key, class_color))
        return jpeg_response(content, etag=cache_key)

    @router.post("/qa/images/{image_id}/findings/{key}/apply")
    def qa_apply(image_id: str, key: str, session: DatasetSession = current):
        with identity.action("apply Label QA finding"):
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
        return sam.segment(record.path, session.dataset.category, points, payload.box, payload.max_points)

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
    def training_preflight(payload: TrainingPreflightPayload, session: DatasetSession = current):
        return training.preflight(session.slug, session.dataset, training.validate(session.slug, payload.yaml)["args"], payload.folders)

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

    @router.get("/training/augment/rfdetr/catalog")
    def augment_rfdetr_catalog(session: DatasetSession = current):
        """The Albumentations transforms RF-DETR's aug_config can name, and RF-DETR's presets."""
        base = {"presets": rfdetr_augment.PRESETS, "default": rfdetr_augment.DEFAULT_EQUIVALENT}
        if not rfdetr_augment.available():
            return {**base, "available": False, "reason": 'Albumentations is not installed (uv add "rfdetr[train,augment]")'}
        return {**base, "available": True, **rfdetr_augment.catalog()}

    @router.get("/training/augment/rfdetr")
    def augment_rfdetr(aug: str = "null", index: int | None = Query(None, ge=-1), sample: int = Query(0, ge=0), seed: int = Query(0, ge=0), resolution: int = Query(640, ge=32, le=4096), size: int = Query(256, ge=64, le=1024), labels: bool = True, session: DatasetSession = current):
        """A preview image: index -1 the image as RF-DETR's resize gives it, index N the aug_config's Nth transform
        (always applied), no index the whole pipeline; aug `null` is RF-DETR's default augmentation."""
        config = None if aug.strip() == "null" else rfdetr_augment.parse_config(aug)
        return augment_response(lambda: rfdetr_augment.render(session.dataset, config, index, sample, seed, resolution, size, labels, class_color))

    @router.get("/training/runs")
    def training_runs(session: DatasetSession = current):
        return training.list_runs(session.slug)

    @router.post("/training/runs")
    def training_start(payload: TrainingStartPayload, session: DatasetSession = current):
        return training.enqueue(session.slug, session.dataset, payload.name, payload.notes, payload.yaml, payload.confirm, payload.folders)

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

    @router.get("/training/runs/{run_id}/weights/onnx")
    def training_onnx_weights(run_id: str, session: DatasetSession = current):
        path, filename = found(lambda: training.best_weights(session.slug, run_id, onnx=True))
        return FileResponse(path, media_type="application/octet-stream", filename=filename, headers={"Cache-Control": "no-cache"})

    @router.post("/training/runs/{run_id}/export/onnx")
    def training_export_onnx(run_id: str, payload: TrainingExportPayload, session: DatasetSession = current):
        return found(lambda: training.export_onnx(session.slug, run_id, payload.imgsz, payload.dynamic, payload.simplify, payload.nms))

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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup and ordered shutdown of the background work (see lifecycle.py); uvicorn runs it, the tests' ASGITransport does not."""
    lifecycle = getattr(app.state, "lifecycle", None)
    if lifecycle is not None:
        await run_in_threadpool(lifecycle.startup)
    app.state.ready = True
    try:
        yield
    finally:
        app.state.ready = False
        app.state.stopping = True
        if lifecycle is not None:
            await run_in_threadpool(lifecycle.shutdown)


def _base_app(mode: str, jobs: JobManager, allowed_hosts: set[str] | None, require_header: bool, auth: Auth | None = None) -> FastAPI:
    app = FastAPI(title="YOLO Dataset Workbench", version="0.1.0", lifespan=_lifespan)
    app.state.jobs = jobs
    app.state.ready = False
    app.state.stopping = False
    app.state.owner_lock = None  # the OwnerLock the CLI/server holds, for /readyz
    app.state.auth = auth
    app.add_middleware(identity.RequestContextMiddleware)  # the browser tab (X-Workbench-Client), innermost
    if auth is not None:
        app.add_middleware(AuthMiddleware, auth=auth)  # inside the Host/header guard below (added later = outer)

    @app.exception_handler(DatasetError)
    async def dataset_error(_, exc: DatasetError):
        return Response(content=f'{{"detail": {json_string(str(exc))}}}', status_code=400, media_type="application/json")

    @app.exception_handler(AuthError)
    async def auth_error(_, exc: AuthError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(UploadTooLarge)
    async def upload_too_large(_, exc: UploadTooLarge):
        return JSONResponse(status_code=413, content={"detail": str(exc)})

    @app.exception_handler(InsufficientSpace)
    async def insufficient_space(_, exc: InsufficientSpace):
        return JSONResponse(status_code=507, content={"detail": str(exc)})

    @app.exception_handler(SchemaChanged)
    async def schema_changed(_, exc: SchemaChanged):
        mapping = None if exc.mapping is None else {str(old): new for old, new in exc.mapping.items()}
        return JSONResponse(status_code=409, content={"detail": str(exc), "schema_generation": exc.generation, "mapping": mapping})

    @app.exception_handler(WriteConflict)
    async def conflict_error(_, exc: WriteConflict):
        return Response(content=f'{{"detail": {json_string(str(exc))}}}', status_code=409, media_type="application/json")

    @app.exception_handler(RequestValidationError)
    async def validation_error(_, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": json_safe(jsonable_encoder(exc.errors()))})

    @app.get("/api/v1/app")
    def app_info():
        user = current_user()
        return {"mode": mode, "version": app.version, "release": release_info(), "server": RUNTIME.server_mode, "features": {"path_import": RUNTIME.allow_path_import, "model_upload": True}, "auth": auth is not None, "user": user.public() if user else None}

    # ---------- sign-in (server profile) ----------

    def require_auth() -> Auth:
        if auth is None:
            raise HTTPException(404, "Sign-in is not enabled on this server")
        return auth

    @app.get("/api/v1/auth/me")
    def auth_me():
        user = current_user()
        return {"auth": auth is not None, "user": user.public() if user else None}

    @app.post("/api/v1/auth/login")
    def auth_login(payload: LoginPayload, request: Request):
        try:
            user, token = require_auth().login(payload.name, payload.password)
        except AuthError:
            log.info("auth: failed sign-in as %r from %s", payload.name[:64], request.client.host if request.client else "?")
            raise
        log.info("auth: %s signed in from %s", user.name, request.client.host if request.client else "?")
        response = JSONResponse({"user": user.public()})
        response.headers["set-cookie"] = auth.session_cookie(token)
        response.headers["cache-control"] = "no-store"
        return response

    @app.post("/api/v1/auth/logout")
    def auth_logout(request: Request):
        response = JSONResponse({"signed_out": True})
        if auth is not None:
            auth.logout(auth.cookie_token(request.headers.get("cookie")))
            response.headers["set-cookie"] = auth.clear_cookie()
        return response

    @app.post("/api/v1/auth/password")
    def auth_password(payload: PasswordPayload):
        user = current_user()
        if user is None:
            raise HTTPException(401, "Sign in to continue")
        token = require_auth().change_password(user, payload.current, payload.new)
        log.info("auth: %s changed their password (other sessions ended)", user.name)
        response = JSONResponse({"changed": True})
        response.headers["set-cookie"] = auth.session_cookie(token)
        return response

    @app.get("/healthz")
    def healthz():
        """Liveness: the process answers."""
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        """Readiness: startup finished, not stopping, the data folder writable, the owner lock held. Image
        indexing, the GPU and MongoDB do not count: the app is usable without them. `release` tells an admin
        after an upgrade which release answers (/api/v1/app needs a sign-in and an allowed Host)."""
        problems = []
        if app.state.stopping:
            problems.append("stopping")
        elif not app.state.ready:
            problems.append("starting")
        lifecycle = getattr(app.state, "lifecycle", None)
        if lifecycle is not None and not os.access(lifecycle.data_root, os.W_OK):
            problems.append(f"{lifecycle.data_root} is not writable")
        lock = app.state.owner_lock
        if lock is not None and not lock.held:
            problems.append("the owner lock is not held")
        return JSONResponse(status_code=503 if problems else 200, content={"status": "unavailable" if problems else "ready", "problems": problems, "release": release_name()})

    @app.get("/api/v1/status")
    def server_status():
        """Warnings for every page (low disk space) and whether the server is stopping."""
        lifecycle = getattr(app.state, "lifecycle", None)
        status = lifecycle.status() if lifecycle is not None else {"disk": None, "warnings": [], "stopping": False}
        return {**status, "ready": app.state.ready}

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
        path = request.url.path
        if allowed_hosts is not None and path not in HEALTH_PATHS:
            raw = (request.headers.get("host") or "").lower()
            host = raw.split("]")[0] + "]" if raw.startswith("[") else raw.rsplit(":", 1)[0]
            if host not in allowed_hosts:
                return JSONResponse(status_code=403, content={"detail": "Host not allowed"})
        mutating = request.method in MUTATING_METHODS and path.startswith("/api/")
        if require_header and mutating and request.headers.get(CSRF_HEADER) != "1":
            return JSONResponse(status_code=403, content={"detail": "Missing X-Workbench header"})
        if mutating and app.state.stopping:
            return JSONResponse(status_code=503, content={"detail": "The server is shutting down; try again once it is back"})
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

    @app.post("/api/v1/training/queue/hold")
    def training_hold(payload: TrainingHoldPayload):
        """Hold the queue (maintenance: nothing new starts, also after a restart) or release it."""
        return training.set_hold(payload.held, payload.pause_active)


def _unload_before_training(training: TrainingManager, sam: SamManager, open_sessions: Callable[[], list[DatasetSession]], scope: str = "all") -> None:
    """A run about to start gets the GPU. scope "all": SAM and every open dataset's prediction model are
    unloaded first; "project" (the server): only the trained project's prediction model (its dataset is
    read-only while it trains); SAM and other projects' models stay until they sit idle."""

    def unload(run: str, project: str | None = None) -> None:
        freed = sam.unload(run) if scope == "all" else False
        for session in open_sessions():
            if scope == "all" or session.slug == project:
                freed = session.unload_models(run) or freed
        if freed:
            free_gpu_memory()

    training.before_launch = unload


def _mount_static(app: FastAPI) -> None:
    static = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static, html=True), name="static")


def _sam_routes(app: FastAPI, sam: SamManager) -> None:
    """One SAM model for the whole app (in projects mode it outlives switching projects)."""
    app.state.sam = sam

    def payload(state: dict) -> dict:
        lifecycle = getattr(app.state, "lifecycle", None)
        state["gpu_warning"] = lifecycle.gpu_warning(state.get("status") in ("ready", "loading")) if lifecycle else None
        return state

    @app.get("/api/v1/sam")
    def sam_state():
        return payload(sam.payload())

    @app.post("/api/v1/sam/start")
    def sam_start():
        lifecycle = getattr(app.state, "lifecycle", None)
        refused = lifecycle.refuse_gpu_load() if lifecycle else None
        if refused and sam.payload()["status"] in ("idle", "error"):
            raise DatasetError(refused)
        return payload(sam.start())


def create_app(dataset: Dataset, media_cache: MediaCache | None = None, predictor: PredictorManager | None = None, allowed_hosts: set[str] | None = None, require_header: bool = False, jobs: JobManager | None = None, training: TrainingManager | None = None, sam: SamManager | None = None, lifecycle_options: LifecycleOptions | None = None, auth: Auth | None = None) -> FastAPI:
    """Single-dataset mode: the dataset API lives directly under /api/v1."""
    jobs = jobs or JobManager()
    app = _base_app("single", jobs, allowed_hosts, require_header, auth)
    training = training or TrainingManager.for_dataset(dataset, jobs=jobs)
    training.attach(dataset)
    _training_routes(app, training)
    sam = sam or SamManager()
    _sam_routes(app, sam)
    session = DatasetSession.open(dataset, predictor)
    _unload_before_training(training, sam, lambda: [session], (lifecycle_options or LifecycleOptions()).unload_for_training)
    app.state.lifecycle = Lifecycle(training, jobs, sam, lambda: [session], dataset.root, lifecycle_options)
    app.state.dataset = dataset
    app.state.session = session
    app.state.predictor = session.predictor
    app.state.embeddings = session.embeddings
    cache = media_cache or MediaCache()
    app.state.media_cache = cache
    app.include_router(dataset_router(lambda: session, cache, jobs, training, sam, lambda: app.state.lifecycle), prefix="/api/v1")
    _mount_static(app)
    return app


def create_workspace_app(workspace: Workspace, media_cache: MediaCache | None = None, allowed_hosts: set[str] | None = None, require_header: bool = False, sam: SamManager | None = None, lifecycle_options: LifecycleOptions | None = None, auth: Auth | None = None) -> FastAPI:
    """Projects mode: workspace endpoints plus the dataset API under /api/v1/projects/{slug}."""
    app = _base_app("projects", workspace.jobs, allowed_hosts, require_header, auth)
    app.state.workspace = workspace
    _training_routes(app, workspace.training)
    sam = sam or SamManager()
    _sam_routes(app, sam)
    _unload_before_training(workspace.training, sam, workspace.open_sessions, (lifecycle_options or LifecycleOptions()).unload_for_training)
    app.state.lifecycle = Lifecycle(workspace.training, workspace.jobs, sam, workspace.open_sessions, workspace.root, lifecycle_options, workspace=workspace)
    cache = media_cache or MediaCache()
    app.state.media_cache = cache

    def project_session(slug: str) -> DatasetSession:
        try:
            return workspace.session(slug)
        except KeyError as exc:
            raise HTTPException(404, "Project not found") from exc

    def peek_session(slug: str) -> DatasetSession | None:
        return workspace.sessions.get(slug)

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
        return {"items": workspace.list_trash(), "retention_days": app.state.lifecycle.options.project_trash_retention_days}

    @app.post("/api/v1/trash/{entry}/restore")
    def trash_restore(entry: str):
        return not_found(lambda: workspace.restore(entry))

    @app.delete("/api/v1/trash/{entry}")
    def trash_purge(entry: str):
        return not_found(lambda: workspace.purge(entry))

    app.include_router(dataset_router(project_session, cache, workspace.jobs, workspace.training, sam, lambda: app.state.lifecycle, peek_session), prefix="/api/v1/projects/{slug}")
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
