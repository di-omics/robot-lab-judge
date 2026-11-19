"""Decode take.mp4, crop the text banner away, split and rebuild the camera views.

A take frame is 640 x 566: a 26 px text banner on top (task sentence, model, clock, stage count),
then the 640 x 360 top camera view, then the left and right wrist views (320 x 180 each) side by side.
The judge must never see the banner rows, because they print the score.
"""

import av
import numpy as np
from PIL import Image

FRAME_W, FRAME_H = 640, 566
BANNER_H = 26
# boxes are (left, top, right, bottom) in take-frame pixels, as PIL crop takes them
BANNER_BOX = (0, 0, 640, BANNER_H)
VIEWS_BOX = (0, BANNER_H, 640, FRAME_H)  # top view over both wrists, 640 x 540
TOP_BOX = (0, BANNER_H, 640, BANNER_H + 360)  # top camera alone, 640 x 360
LEFT_BOX = (0, BANNER_H + 360, 320, FRAME_H)
RIGHT_BOX = (320, BANNER_H + 360, 640, FRAME_H)


def first_and_last(path) -> tuple[np.ndarray, np.ndarray, int]:
    """The first and the last decoded frame (RGB) and the frame count."""
    first = last = None
    count = 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            if first is None:
                first = rgb
            last = rgb
            count += 1
    if first is None:
        raise ValueError(f"no frames in {path}")
    return first, last, count


def check_size(rgb: np.ndarray) -> None:
    if rgb.shape[:2] != (FRAME_H, FRAME_W):
        raise ValueError(f"expected a {FRAME_W} x {FRAME_H} take frame, got {rgb.shape[1]} x {rgb.shape[0]}")


def views(rgb: np.ndarray) -> Image.Image:
    """The 3-view composite without the banner: 640 x 540."""
    check_size(rgb)
    return Image.fromarray(rgb).crop(VIEWS_BOX)


def top_view(rgb: np.ndarray) -> Image.Image:
    """The top camera view alone: 640 x 360."""
    check_size(rgb)
    return Image.fromarray(rgb).crop(TOP_BOX)


def uncropped(rgb: np.ndarray) -> Image.Image:
    """The full take frame with the banner (leak control only)."""
    check_size(rgb)
    return Image.fromarray(rgb)
