"""The pictures the encoder sees, and the arm mask. Pillow and numpy only, no model."""

import numpy as np
from PIL import Image

from . import check_path

TOP_W, TOP_H = 640, 360
## "top": the whole top view, resized to 512 x 288 (32 x 18 patches of 16 px).
TOP_SIZE = (512, 288)
## "fine": 8 overlapping 256 x 256 crops of the top view, each resized to 512 x 512 (2x zoom, 32 x 32 patches).
CROP = 256
CROP_X = (0, 128, 256, 384)
CROP_Y = (0, 104)
FINE_SIZE = (512, 512)
## Resize filter of the checkpoint's own image processor (resample 2 = bilinear).
RESAMPLE = Image.Resampling.BILINEAR

## Arm mask: a pixel is dark when its brightest channel is below DARK.
DARK = 50
## The mask keeps pixels that are dark in the first top view of every layout of the split,
## so labware (which moves between layouts) drops out and the arms at home stay.
## Then it is shrunk by ERODE pixels so labware next to an arm does not count.
ERODE = 3
## The flat colour painted over the arm mask in the arms-flat control.
FLAT = (128, 128, 128)


def crop_boxes() -> list[tuple[int, int, int, int]]:
    return [(x, y, x + CROP, y + CROP) for y in CROP_Y for x in CROP_X]


def first_top(row: dict) -> Image.Image:
    """The top camera view at the start of the layout: the top 360 rows of the first-frame composite."""
    im = Image.open(check_path(row["images"]["first_views"])).convert("RGB")
    top = im.crop((0, 0, TOP_W, TOP_H))
    top.load()
    return top


def last_top(row: dict) -> Image.Image:
    im = Image.open(check_path(row["images"]["last_top"])).convert("RGB")
    if im.size != (TOP_W, TOP_H):
        raise ValueError(f"last_top of {row['take_id']} is {im.size}, expected {(TOP_W, TOP_H)}")
    return im


def top_picture(top: Image.Image) -> Image.Image:
    return top.resize(TOP_SIZE, RESAMPLE)


def fine_pictures(top: Image.Image) -> list[Image.Image]:
    return [top.crop(b).resize(FINE_SIZE, RESAMPLE) for b in crop_boxes()]


def dark(top: Image.Image, threshold: int = DARK) -> np.ndarray:
    a = np.asarray(top, dtype=np.uint8)
    return a.max(axis=2) < threshold


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary erosion with a (2r+1) square: a pixel stays only if every pixel within r is set."""
    out = mask.copy()
    h, w = mask.shape
    pad = np.pad(mask, r, constant_values=False)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out &= pad[r + dy: r + dy + h, r + dx: r + dx + w]
    return out


def paint_flat(top: Image.Image, mask: np.ndarray, colour=FLAT) -> Image.Image:
    a = np.asarray(top, dtype=np.uint8).copy()
    a[mask] = colour
    return Image.fromarray(a)
