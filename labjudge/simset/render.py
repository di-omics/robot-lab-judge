"""Pictures of a scene, laid out exactly like a take frame, then cropped by the judge's own crop.

The bench's own tile() makes the 640 x 566 take frame. The banner text is left empty, and frames.views() and
frames.top_view() cut the banner rows away, so no saved picture holds banner rows.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import frames
from ..sheet import BG, INK, SOFT, _font
from . import bench as B


def take_frame(images) -> np.ndarray:
    """The 640 x 566 take frame of one moment, with an empty banner."""
    return B.lb.video.tile(images, "", "")


def save_views(images, stem: Path, which: str) -> dict:
    """Save <stem>_<which>_views.png (640 x 540) and, for the last moment, <stem>_last_top.png (640 x 360)."""
    rgb = take_frame(images)
    out = {f"{which}_views": f"{stem}_{which}_views.png"}
    frames.views(rgb).save(out[f"{which}_views"])
    if which == "last":
        out["last_top"] = f"{stem}_last_top.png"
        frames.top_view(rgb).save(out["last_top"])
    return out


def sheet(rows: list[dict], out: Path, title: str, cols: int = 6) -> None:
    """A light contact sheet of top views, captioned with task, kind and truth."""
    tw, th, cap, pad = 224, 126, 34, 12
    title_h = 40
    n_rows = (len(rows) + cols - 1) // cols
    img = Image.new("RGB", (pad + cols * (tw + pad), title_h + pad + n_rows * (th + cap + pad)), BG)
    d = ImageDraw.Draw(img)
    small, big = _font(11), _font(14)
    d.text((pad, 12), title, fill=INK, font=big)
    for i, r in enumerate(rows):
        x = pad + (i % cols) * (tw + pad)
        y = title_h + pad + (i // cols) * (th + cap + pad)
        thumb = Image.open(r["images"]["last_top"]).convert("RGB").resize((tw, th), Image.LANCZOS)
        img.paste(thumb, (x, y))
        d.rectangle((x - 1, y - 1, x + tw, y + th), outline=SOFT, width=1)
        d.text((x, y + th + 4), f"{r['task']}  s{r['seed']}", fill=INK, font=small)
        d.text((x, y + th + 18), f"{r['kind']}  truth {r['truth']}", fill=INK, font=small)
    img.save(out, quality=88)
