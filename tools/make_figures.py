"""Draw the README figure: for 4 chores, a scene that passes above a scene that fails, zoomed on the labware.

Needs the dev scenes on disk (data/simset/dev) and one run's verdicts. Run from the repo root:

  uv run python tools/make_figures.py

The scenes are picked by a fixed rule, not by eye: for each chore, the first dev layout that has a clean scene
and a failing scene of the wanted kind. The number under each column is the balanced accuracy of the named run
on that chore, read from its verdicts file.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labjudge.score import metrics, read_verdicts  # noqa: E402
from labjudge.sheet import BG, INK, SOFT, _font  # noqa: E402

DATA = Path("data/simset/dev")
RUN = Path("results/simset_dev/verdicts_LiquidAI__LFM3-VL-2B-MLX-8bit_think_after.jsonl")  # best zero-shot model
TRAINED = Path("results/simset_dev/verdicts_frozen__fine-logreg_after.jsonl")  # the linear probe
OUT = Path("docs/img/pass_and_fail.jpg")
## chore, kind of the failing scene, short column title
COLUMNS = (
    ("sort_by_cap", "trap", "Sort 2 tubes by cap colour"),
    ("tubes_to_rack", "offset", "Put the tubes in the rack"),
    ("plate_to_tray", "offset", "Put the plate on the tray"),
    ("stack_plates", "offset", "Stack the two plates"),
)
CELL = (228, 152)
PAD = 12
LEFT = 64


def changed_box(a: Image.Image, b: Image.Image, pad: int = 16) -> tuple[int, int, int, int]:
    """Box around the pixels that differ between 2 pictures, grown to the cell's shape."""
    d = np.abs(np.asarray(a).astype(int) - np.asarray(b).astype(int)).sum(-1)
    ys, xs = np.nonzero(d > 40)
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    w, h = x1 - x0, y1 - y0
    want = CELL[0] / CELL[1]
    if w / h < want:
        grow = (h * want - w) / 2
        x0, x1 = x0 - grow, x1 + grow
    else:
        grow = (w / want - h) / 2
        y0, y1 = y0 - grow, y1 + grow
    dx = max(0, -x0) - max(0, x1 - a.width)
    dy = max(0, -y0) - max(0, y1 - a.height)
    return (int(x0 + dx), int(y0 + dy), int(x1 + dx), int(y1 + dy))


def pick(rows: list[dict], task: str, kind: str) -> tuple[dict, dict]:
    """The first layout of a chore with a clean scene and a failing scene of the wanted kind."""
    for seed in sorted({r["seed"] for r in rows if r["task"] == task}):
        here = [r for r in rows if r["task"] == task and r["seed"] == seed]
        clean = [r for r in here if r["kind"] == "clean" and r["truth"] == "pass"]
        fails = [r for r in here if r["kind"] == kind and r["truth"] == "fail"]
        if clean and fails:
            return clean[0], fails[0]
    raise SystemExit(f"no pair for {task} {kind}")


def main() -> None:
    rows = [json.loads(x) for x in (DATA / "manifest.jsonl").read_text().splitlines() if x.strip()]
    header, verdicts = read_verdicts(RUN)
    _, trained = read_verdicts(TRAINED)
    small, big = _font(13), _font(15)
    top = 34
    width = LEFT + len(COLUMNS) * (CELL[0] + PAD)
    height = top + 2 * (CELL[1] + PAD) + 46
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    for label, y in (("passes", top), ("fails", top + CELL[1] + PAD)):
        d.text((PAD, y + CELL[1] // 2 - 8), label, fill=INK, font=big)
    for i, (task, kind, title) in enumerate(COLUMNS):
        clean, fail = pick(rows, task, kind)
        a = Image.open(clean["images"]["last_top"]).convert("RGB")
        b = Image.open(fail["images"]["last_top"]).convert("RGB")
        box = changed_box(a, b)
        x = LEFT + i * (CELL[0] + PAD)
        d.text((x, 10), title, fill=INK, font=big)
        for pic, y in ((a, top), (b, top + CELL[1] + PAD)):
            img.paste(pic.crop(box).resize(CELL, Image.LANCZOS), (x, y))
            d.rectangle((x - 1, y - 1, x + CELL[0], y + CELL[1]), outline=SOFT, width=1)
        m = metrics([v for v in verdicts if v["task"] == task])
        t = metrics([v for v in trained if v["task"] == task])
        y = top + 2 * (CELL[1] + PAD)
        d.text((x, y), f"zero-shot model: {m['balanced_accuracy']:.2f}", fill=INK, font=small)
        d.text((x, y + 18), f"linear probe: {t['balanced_accuracy']:.2f}", fill=INK, font=small)
        print(f"{task}: pass {clean['take_id']}, fail {fail['take_id']} (error {fail['error']}), box {box}, "
              f"zero-shot {m['balanced_accuracy']:.2f}, trained {t['balanced_accuracy']:.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, quality=90)
    print(f"wrote {OUT} {img.size}, judge {header.get('model_id')} thinking {header.get('thinking')}")


if __name__ == "__main__":
    main()
