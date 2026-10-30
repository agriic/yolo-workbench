"""Images as the browser and Ultralytics see them: EXIF orientation applied.

Phone photos are often stored sideways with an EXIF Orientation tag. The editor
(browsers apply the tag), SAM, cropping and resizing, and Ultralytics all work on the
upright image, so normalized label coordinates refer to it; every server-side render
and size probe must use the same frame.
"""

from __future__ import annotations

import io

from PIL import Image, ImageOps

ORIENTATION = 0x0112  # EXIF tag
SWAPS_AXES = {5, 6, 7, 8}  # orientations that rotate by 90° (width and height trade places)


def orientation(image: Image.Image) -> int:
    try:
        return int(image.getexif().get(ORIENTATION, 1) or 1)
    except Exception:  # unreadable EXIF just means "as stored"
        return 1


def upright_size(image: Image.Image) -> tuple[int, int]:
    """(width, height) after EXIF orientation, without decoding the pixels."""
    width, height = image.size
    return (height, width) if orientation(image) in SWAPS_AXES else (width, height)


def upright_rgb(image: Image.Image) -> Image.Image:
    """The decoded image, EXIF orientation applied, as RGB."""
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:  # a broken EXIF block must not make the image unrenderable
        pass
    return image.convert("RGB")


def encode(image: Image.Image, source: Image.Image, suffix: str) -> bytes:
    """Encode an upright image made from `source` (cropped, resized) in the source's format, keeping its ICC
    profile and EXIF minus the orientation (the pixels are upright now)."""
    image_format = Image.registered_extensions().get(suffix.lower(), "PNG")
    params: dict = {}
    if image_format in ("JPEG", "WEBP"):
        params["quality"] = 95
    if image_format == "JPEG" and image.mode not in ("RGB", "L", "CMYK"):
        image = image.convert("RGB")
    if source.info.get("icc_profile"):
        params["icc_profile"] = source.info["icc_profile"]
    exif = source.getexif()
    if exif and image_format in ("JPEG", "WEBP", "PNG", "TIFF"):
        exif.pop(ORIENTATION, None)
        params["exif"] = exif.tobytes()
    output = io.BytesIO()
    image.save(output, image_format, **params)
    return output.getvalue()
