"""The judge must never see the banner rows. These tests fail if a judged image still holds them."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from labjudge import frames
from labjudge.__main__ import images_for

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "first_look" / "manifest.jsonl"
BANNER_RGB = (250, 250, 248)  # banner background in the take videos


def synthetic_frame() -> np.ndarray:
    """A take-shaped frame: a pure red banner, a green top view, blue and white wrists."""
    f = np.zeros((frames.FRAME_H, frames.FRAME_W, 3), np.uint8)
    f[: frames.BANNER_H] = (255, 0, 0)
    f[frames.BANNER_H : frames.BANNER_H + 360] = (0, 255, 0)
    f[frames.BANNER_H + 360 :, :320] = (0, 0, 255)
    f[frames.BANNER_H + 360 :, 320:] = (255, 255, 255)
    return f


def has_red(img: Image.Image) -> bool:
    a = np.asarray(img)
    return bool(((a[..., 0] == 255) & (a[..., 1] == 0) & (a[..., 2] == 0)).any())


def test_geometry_adds_up():
    assert frames.BANNER_H == 26
    assert frames.VIEWS_BOX == (0, 26, 640, 566)
    assert frames.TOP_BOX[3] - frames.TOP_BOX[1] == 360
    assert frames.LEFT_BOX[2] - frames.LEFT_BOX[0] == 320 and frames.LEFT_BOX[3] - frames.LEFT_BOX[1] == 180
    assert frames.RIGHT_BOX[0] == 320


def test_crops_drop_every_banner_row():
    f = synthetic_frame()
    views = frames.views(f)
    top = frames.top_view(f)
    assert views.size == (640, 540) and top.size == (640, 360)
    assert not has_red(views) and not has_red(top)
    a = np.asarray(views)
    assert (a[0] == (0, 255, 0)).all(), "first judged row must be the first camera row"
    assert (a[360:, :320] == (0, 0, 255)).all() and (a[360:, 320:] == (255, 255, 255)).all()
    # the control keeps the banner on purpose
    assert has_red(frames.uncropped(f))


def test_wrong_size_is_refused():
    with pytest.raises(ValueError):
        frames.views(np.zeros((540, 640, 3), np.uint8))


def _banner_like(rows: np.ndarray) -> bool:
    """True when the rows look like the banner: mostly the banner background colour."""
    close = (np.abs(rows.astype(int) - BANNER_RGB).sum(-1) <= 6).mean()
    return close > 0.6


@pytest.mark.skipif(not MANIFEST.exists(), reason="run extract first")
def test_saved_judged_images_have_no_banner():
    rows = [json.loads(x) for x in MANIFEST.read_text().splitlines() if x.strip()]
    assert rows
    for row in rows:
        for condition in ("after", "before_after"):
            for rel in images_for(row, condition):
                img = Image.open(ROOT / rel).convert("RGB")
                assert img.size == (640, 540), rel
                a = np.asarray(img)
                assert not _banner_like(a[: frames.BANNER_H]), f"banner rows in judged image {rel}"
        top = np.asarray(Image.open(ROOT / row["images"]["last_top"]).convert("RGB"))
        assert top.shape[:2] == (360, 640)
        assert not _banner_like(top[: frames.BANNER_H])
        # the control image must still carry the banner, or it is not a control
        full = np.asarray(Image.open(ROOT / row["images"]["last_uncropped"]).convert("RGB"))
        assert full.shape[:2] == (566, 640)
        assert _banner_like(full[: frames.BANNER_H])
        # the judged composite is exactly the frame below the banner
        assert (np.asarray(Image.open(ROOT / row["images"]["last_views"]).convert("RGB")) == full[26:]).all()
