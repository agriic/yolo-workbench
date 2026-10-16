import io

import yaml
from PIL import Image

from yolo_workbench.dataset import Dataset
from yolo_workbench.imaging import ORIENTATION
from yolo_workbench.web import render_crop, render_thumbnail
from yolo_workbench.workspace import _write_cover


def make_rotated_dataset(tmp_path):
    """A 200×400 upright photo stored sideways (400×200 pixels + EXIF orientation 6), red block labelled."""
    upright = Image.new("RGB", (200, 400), "white")
    upright.paste((255, 0, 0), (120, 40, 180, 120))  # x 0.6–0.9, y 0.1–0.3 of the upright image
    raw = upright.transpose(Image.Transpose.ROTATE_90)  # orientation 6 turns it back with ROTATE_270
    exif = Image.Exif()
    exif[ORIENTATION] = 6
    (tmp_path / "train").mkdir()
    raw.save(tmp_path / "train" / "photo.jpg", exif=exif.tobytes(), quality=95)
    (tmp_path / "train" / "photo.txt").write_text("0 0.75 0.2 0.3 0.2\n")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"path": ".", "train": "train", "names": {0: "block"}}))
    return Dataset(tmp_path / "data.yaml", "detection")


def red(pixel) -> bool:
    r, g, b = pixel[:3]
    return r > 180 and g < 90 and b < 90


def test_probe_reports_the_upright_size(tmp_path):
    dataset = make_rotated_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    assert (record.width, record.height) == (200, 400)
    # and the size index keeps it
    reloaded = next(iter(Dataset(dataset.yaml_path, "detection").images.values()))
    assert (reloaded.width, reloaded.height) == (200, 400)


def test_thumbnail_and_crop_use_the_upright_image(tmp_path):
    dataset = make_rotated_dataset(tmp_path)
    record = next(iter(dataset.images.values()))

    thumbnail = Image.open(io.BytesIO(render_thumbnail(record, 200, "detection")))
    assert thumbnail.height > thumbnail.width
    assert red(thumbnail.getpixel((int(0.75 * thumbnail.width), int(0.2 * thumbnail.height))))

    crop = Image.open(io.BytesIO(render_crop(record, record.annotations[0], "detection", 0.0)))
    assert red(crop.getpixel((crop.width // 2, crop.height // 2)))


def test_cover_is_upright(tmp_path):
    dataset = make_rotated_dataset(tmp_path)
    record = next(iter(dataset.images.values()))
    cover = tmp_path / "cover.jpg"
    _write_cover(record.path, cover)
    with Image.open(cover) as image:
        assert image.height > image.width
