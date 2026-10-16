import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from test_dataset import make_dataset
from yolo_workbench import storage
from yolo_workbench.dataset import Dataset, DatasetError
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.web import create_app


def deleted_in_old_session(tmp_path):
    """A dataset whose only image was deleted in an earlier session (so undo cannot bring it back)."""
    dataset, label = make_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    dataset.session_id = "20250101-120000"
    dataset.replace_annotations(record.id, [{"class_id": 0, "points": [0.5, 0.5, 0.1, 0.1]}])  # backs the label up
    dataset.delete_images([record.id])
    reopened = Dataset(dataset.yaml_path, "detection", allow_empty=True)
    reopened.session_id = "20250102-090000"
    return reopened, record.path, label


def test_summary_lists_sessions_with_sizes(tmp_path):
    dataset, image, label = deleted_in_old_session(tmp_path)
    data = storage.summary(dataset)
    [trash] = data["trash"]
    assert trash["session"] == "20250101-120000" and trash["files"] == 2 and trash["images"] == 1
    assert trash["started"] == "2025-01-01T12:00:00" and not trash["current"]
    assert data["backups"][0]["files"] == 1
    assert data["total_bytes"] == trash["bytes"] + data["backups"][0]["bytes"]


def test_restore_puts_files_back_and_rescans(tmp_path):
    dataset, image, label = deleted_in_old_session(tmp_path)
    assert not dataset.images
    result = storage.restore_trash(dataset, "20250101-120000")
    assert result["restored"] == 2 and result["added"] == 1 and not result["skipped"]
    assert image.exists() and label.read_text().startswith("0 ")
    assert len(dataset.images) == 1
    assert not result["trash"]  # the emptied session folder is gone


def test_restore_keeps_files_whose_place_is_taken(tmp_path):
    dataset, image, label = deleted_in_old_session(tmp_path)
    Image.new("RGB", (10, 10)).save(image)  # a new file with the same name
    result = storage.restore_trash(dataset, "20250101-120000")
    assert result["skipped"] == ["train/sample.jpg"]
    assert result["trash"][0]["files"] == 1


def test_purge_deletes_old_sessions_only(tmp_path):
    dataset, _, _ = deleted_in_old_session(tmp_path)
    data = storage.purge(dataset, "trash", "20250101-120000")
    assert not data["trash"]
    data = storage.purge(dataset, "backups", "20250101-120000")
    assert not data["backups"]
    current = tmp_path / ".yolo-workbench" / "trash" / dataset.session_id
    current.mkdir(parents=True)
    with pytest.raises(DatasetError):
        storage.purge(dataset, "trash", dataset.session_id)
    with pytest.raises(DatasetError):
        storage.purge(dataset, "trash", "../../etc")
    with pytest.raises(KeyError):
        storage.purge(dataset, "trash", "20990101-000000")


def test_storage_api(tmp_path):
    dataset, image, _ = deleted_in_old_session(tmp_path)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media-cache"))

    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            assert len((await client.get("/api/v1/storage")).json()["trash"]) == 1
            assert (await client.post("/api/v1/storage/trash/20990101-000000/restore")).status_code == 404
            assert (await client.post("/api/v1/storage/trash/20250101-120000/restore")).json()["restored"] == 2
            assert (await client.delete("/api/v1/storage/backups/20250101-120000")).status_code == 200
            assert (await client.delete("/api/v1/storage/nope/20250101-120000")).status_code == 400

    asyncio.run(run())
