from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import BaseModel

from .dataset import Dataset, DatasetError, WriteConflict
from .embeddings import EmbeddingsManager
from .models import Annotation, ImageRecord

# Keep in sync with PALETTE in static/app.js so server-rendered overlays match the editor.
PALETTE = ("#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#06b6d4", "#84cc16", "#a855f7", "#64748b")


def class_color(class_id: int) -> str:
    return PALETTE[class_id % len(PALETTE)]


class AnnotationPayload(BaseModel):
    id: str | None = None
    class_id: int
    points: list[float]


class AnnotationListPayload(BaseModel):
    annotations: list[AnnotationPayload]


def create_app(dataset: Dataset) -> FastAPI:
    app = FastAPI(title="YOLO Dataset Workbench", version="0.1.0")
    app.state.dataset = dataset
    embeddings = EmbeddingsManager(dataset)
    app.state.embeddings = embeddings

    @app.exception_handler(DatasetError)
    async def dataset_error(_, exc: DatasetError):
        return Response(content=f'{{"detail": {json_string(str(exc))}}}', status_code=400, media_type="application/json")

    @app.exception_handler(WriteConflict)
    async def conflict_error(_, exc: WriteConflict):
        return Response(content=f'{{"detail": {json_string(str(exc))}}}', status_code=409, media_type="application/json")

    @app.get("/api/v1/dataset")
    async def metadata():
        split_counts: dict[str, int] = {}
        for record in dataset.images.values():
            split_counts[record.split] = split_counts.get(record.split, 0) + 1
        return {"yaml": str(dataset.yaml_path), "root": str(dataset.root), "category": dataset.category, "names": dataset.names, "image_count": len(dataset.images), "split_counts": split_counts, "issue_count": len(dataset.issues()), "session_id": dataset.session_id}

    @app.get("/api/v1/images")
    async def images(split: str = "all", class_id: int | None = None, search: str = "", offset: int = 0, limit: int = Query(100, ge=1, le=500)):
        return dataset.list_images(split, class_id, search, max(0, offset), limit)

    @app.get("/api/v1/images/{image_id}")
    async def image_detail(image_id: str):
        try:
            return dataset.detail(image_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/v1/images/{image_id}/file")
    async def image_file(image_id: str):
        try:
            return FileResponse(dataset.require_image(image_id).path)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/v1/images/{image_id}/thumbnail")
    async def thumbnail(image_id: str, size: int = Query(320, ge=64, le=1024), annotated: bool = False):
        try:
            record = dataset.require_image(image_id)
            return jpeg_response(render_thumbnail(record, size, dataset.category if annotated else None))
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.put("/api/v1/images/{image_id}/annotations")
    async def update_annotations(image_id: str, payload: AnnotationListPayload):
        try:
            return dataset.replace_annotations(image_id, [annotation.model_dump() for annotation in payload.annotations])
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/v1/objects")
    async def objects(class_id: int, split: str = "all", offset: int = 0, limit: int = Query(100, ge=1, le=500)):
        items = []
        for record in sorted(dataset.images.values(), key=lambda item: (item.split, item.path.name.casefold())):
            if split != "all" and record.split != split:
                continue
            for annotation in record.annotations:
                if annotation.class_id == class_id:
                    items.append({"id": annotation.id, "image_id": record.id, "image_name": record.path.name, "split": record.split, "class_id": annotation.class_id})
        return {"total": len(items), "items": items[offset : offset + limit]}

    @app.get("/api/v1/objects/{image_id}/{annotation_id:path}/crop")
    async def object_crop(image_id: str, annotation_id: str, padding: float = Query(0.15, ge=0, le=1)):
        try:
            record = dataset.require_image(image_id)
            annotation = next((item for item in record.annotations if item.id == annotation_id), None)
            if not annotation:
                raise HTTPException(404, "Annotation not found")
            return jpeg_response(render_crop(record, annotation, dataset.category, padding))
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/v1/issues")
    async def issues():
        return {"items": dataset.issues()}

    @app.post("/api/v1/issues/{issue_id}/fix")
    async def fix_issue(issue_id: str):
        return dataset.fix_issue(issue_id)

    @app.get("/api/v1/embeddings")
    async def embeddings_state():
        return embeddings.payload()

    @app.post("/api/v1/embeddings/compute")
    async def embeddings_compute():
        return embeddings.start()

    @app.post("/api/v1/history/{direction}")
    async def history(direction: str):
        if direction not in {"undo", "redo"}:
            raise HTTPException(400, "Direction must be undo or redo")
        return dataset.history(direction)

    @app.middleware("http")
    async def no_static_cache(request, call_next):
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            # the UI ships with the package; stale cached scripts break new endpoints
            response.headers["Cache-Control"] = "no-cache"
        return response

    static = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static, html=True), name="static")
    return app


def json_string(value: str) -> str:
    import json

    return json.dumps(value)


def render_thumbnail(record: ImageRecord, size: int, category: str | None = None) -> bytes:
    with Image.open(record.path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size))
        if category:
            draw = ImageDraw.Draw(image)
            for annotation in record.annotations:
                draw_annotation(draw, annotation, category, image.width, image.height)
        output = io.BytesIO()
        image.save(output, "JPEG", quality=82)
        return output.getvalue()


def draw_annotation(draw: ImageDraw.ImageDraw, annotation: Annotation, category: str, width: int, height: int) -> None:
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
        image = source.convert("RGB")
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


def jpeg_response(content: bytes) -> Response:
    return Response(content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
