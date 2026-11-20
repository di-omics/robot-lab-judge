"""How fragile is the linear probe? The saved heads on dev pictures that are changed a little, with no refit.

  HF_HUB_OFFLINE=1 uv run python tools/probe_perturbed.py

The changes were fixed before the run:

- none: the pictures as they are. Must give the saved verdicts again.
- jpeg: saved as JPEG quality 75 and read back.
- shift: moved 3 px right and 2 px down, edge pixels repeated.
- all: shift, then brightness times 0.95, then JPEG quality 75.

`top-logreg` gets every change. `fine-logreg` is slower, so it gets `all`, plus `none` on the first 40 scenes
as a check of this script. Writes results/frozen/perturbed.md. Needs the dev pictures and the model on disk.
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labjudge.frozen.encoder import Tower  # noqa: E402
from labjudge.frozen.extract import mean_max  # noqa: E402
from labjudge.frozen.pictures import fine_pictures, last_top, top_picture  # noqa: E402
from labjudge.score import metrics, prop, read_verdicts  # noqa: E402

DEV = ROOT / "results" / "simset_dev"
OUT = ROOT / "results" / "frozen" / "perturbed.md"


def jpeg(im: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=75)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def shift(im: Image.Image, dx: int = 3, dy: int = 2) -> Image.Image:
    a = np.asarray(im)
    moved = np.pad(a, ((dy, 0), (dx, 0), (0, 0)), mode="edge")[: a.shape[0], : a.shape[1]]
    return Image.fromarray(moved)


def dim(im: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(im).enhance(0.95)


CHANGES = {"none": lambda im: im, "jpeg": jpeg, "shift": shift, "all": lambda im: jpeg(dim(shift(im)))}
FEATURES = {"top-logreg": lambda tower, im: mean_max(tower.encode([top_picture(im)])[0]),
            "fine-logreg": lambda tower, im: mean_max(tower.encode(fine_pictures(im))[0])}
PLAN = (("top-logreg", "none", 0), ("top-logreg", "jpeg", 0), ("top-logreg", "shift", 0), ("top-logreg", "all", 0),
        ("fine-logreg", "none", 40), ("fine-logreg", "all", 0))


def verdicts(tower, row_name: str, change: str, rows: list[dict]) -> list[dict]:
    heads = np.load(ROOT / "results" / "frozen" / f"heads_{row_name}.npz")
    out = []
    for r in rows:
        x = FEATURES[row_name](tower, CHANGES[change](last_top(r))).astype(np.float64)
        t = r["task"]
        z = (x - heads[f"{t}__mean"]) / heads[f"{t}__std"]
        p = 1.0 / (1.0 + np.exp(-(z @ heads[f"{t}__w"] + float(heads[f"{t}__b"]))))
        out.append({"take_id": r["take_id"], "task": t, "truth": r["truth"], "verdict": "PASS" if p >= 0.5 else "FAIL"})
    return out


def main() -> None:
    rows = [json.loads(x) for x in (ROOT / "data/simset/dev/manifest.jsonl").read_text().splitlines() if x.strip()]
    tower = Tower()
    lines = ["# The linear probe on changed pictures", "",
             "The saved heads, the dev pictures changed a little, no refit. `none` checks this script against the",
             "saved verdicts. Made by `tools/probe_perturbed.py`, which lists the changes.", "",
             "| Row | Change | Scenes | Same verdict as saved | Says PASS on true passes | False passes "
             "| Balanced accuracy |", "|---|---|---|---|---|---|---|"]
    for row_name, change, limit in PLAN:
        use = rows[:limit] if limit else rows
        got = verdicts(tower, row_name, change, use)
        _, saved = read_verdicts(DEV / f"verdicts_frozen__{row_name}_after.jsonl")
        saved = {r["take_id"]: r["verdict"] for r in saved}
        same = sum(g["verdict"] == saved[g["take_id"]] for g in got)
        m = metrics(got)
        line = (f"| `{row_name}` | {change} | {len(got)} | {same} of {len(got)} | {prop(m['pass_PASS'], m['n_pass'])} "
                f"| {prop(m['fail_PASS'], m['n_fail'])} | {m['balanced_accuracy']:.2f} |")
        print(line, flush=True)
        lines.append(line)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
