"""Leak check: do the arm pixels differ between the first and last top view of a layout?

Both arms should rest at home in every scene. If arm pixels changed in a way that goes with
kind or truth, a fitted head could read the answer off the arms instead of the labware.
"""

import json
from collections import defaultdict

import numpy as np

from . import FEATURES, OUT, layout_of, read_rows
from .pictures import DARK, ERODE, dark, erode, first_top, last_top

MASK_FILE = FEATURES / "arm_mask.npz"


def build_mask() -> dict:
    """The arm mask from the TRAIN first top views only: dark in every train layout, then shrunk."""
    first = {}
    for r in read_rows("train"):
        first.setdefault(layout_of(r), r)
    darks = [dark(first_top(r), DARK) for r in first.values()]
    common = np.logical_and.reduce(darks)
    mask = erode(common, ERODE)
    counts = [int(d.sum()) for d in darks]
    return {"raw": common, "mask": mask, "n_layouts": len(darks),
            "dark_min": min(counts), "dark_max": max(counts)}


def load_mask() -> np.ndarray:
    if not MASK_FILE.exists():
        m = build_mask()
        MASK_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(MASK_FILE, raw=m["raw"], mask=m["mask"])
    return np.load(MASK_FILE)["mask"]


def scene_diffs(split: str, mask: np.ndarray) -> list[dict]:
    """Per scene: pixels inside the mask that differ at all, that differ by more than 16, and the largest difference."""
    rows = read_rows(split)
    firsts = {}
    out = []
    for r in rows:
        key = layout_of(r)
        if key not in firsts:
            firsts[key] = np.asarray(first_top(r), dtype=np.int16)
        a = firsts[key]
        b = np.asarray(last_top(r), dtype=np.int16)
        d = np.abs(a - b).max(axis=2)
        inside = d[mask]
        out.append({"take_id": r["take_id"], "task": r["task"], "kind": r["kind"], "truth": r["truth"],
                    "n_diff": int((inside > 0).sum()), "n_diff16": int((inside > 16).sum()),
                    "max_diff": int(inside.max()), "outside_n_diff": int((d[~mask] > 0).sum())})
    return out


def table(per_scene: list[dict]) -> list[str]:
    groups = defaultdict(list)
    for s in per_scene:
        groups[(s["kind"], s["truth"])].append(s)
    lines = ["| kind | truth | n | scenes with any differing arm pixel | scenes with an arm pixel off by more than 16 "
             "| largest arm pixel difference (0-255) | most differing arm pixels in 1 scene |",
             "|---|---|---|---|---|---|---|"]
    for (kind, truth), ss in sorted(groups.items(), key=lambda x: (x[0][1] != "pass", x[0][0])):
        lines.append(f"| {kind} | {truth} | {len(ss)} | {sum(s['n_diff'] > 0 for s in ss)} of {len(ss)} "
                     f"| {sum(s['n_diff16'] > 0 for s in ss)} of {len(ss)} | {max(s['max_diff'] for s in ss)} "
                     f"| {max(s['n_diff'] for s in ss)} |")
    return lines


def run() -> dict:
    m = build_mask()
    MASK_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MASK_FILE, raw=m["raw"], mask=m["mask"])
    mask = m["mask"]
    result = {"dark_threshold": DARK, "erode_px": ERODE, "n_train_layouts": m["n_layouts"],
              "dark_px_per_layout_min": m["dark_min"], "dark_px_per_layout_max": m["dark_max"],
              "mask_px_before_erode": int(m["raw"].sum()), "mask_px": int(mask.sum()), "splits": {}}
    md = ["## Leak check: arm pixels, last top view against the first top view of the same layout", "",
          f"Arm mask: a pixel is dark when its brightest channel is below {DARK} (of 255). The mask keeps the pixels "
          f"that are dark in the first top view of all {m['n_layouts']} train layouts "
          f"({int(m['raw'].sum())} pixels; a single layout has {m['dark_min']} to {m['dark_max']} dark pixels), "
          f"then shrinks it by {ERODE} pixels ({int(mask.sum())} pixels left). The same mask is used for dev.", ""]
    for split in ("train", "dev"):
        per = scene_diffs(split, mask)
        result["splits"][split] = per
        md += [f"### {split}", "", *table(per), ""]
        any_diff = sum(s["n_diff"] > 0 for s in per)
        md += [f"{any_diff} of {len(per)} {split} scenes have any differing pixel inside the arm mask. "
               f"Outside the mask, the median scene has {int(np.median([s['outside_n_diff'] for s in per]))} "
               f"differing pixels (the labware moved).", ""]
        for s in per:
            if s["n_diff"]:
                md.append(f"- `{s['take_id']}` ({s['kind']}, {s['truth']}): {s['n_diff']} arm pixels differ, "
                          f"largest difference {s['max_diff']}.")
        md.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leakcheck.md").write_text("\n".join(md) + "\n")
    (OUT / "leakcheck.json").write_text(json.dumps(
        {k: v for k, v in result.items() if k != "splits"} |
        {"summary": {s: {"n": len(v), "any_diff": sum(x["n_diff"] > 0 for x in v),
                         "max_diff": max(x["max_diff"] for x in v)} for s, v in result["splits"].items()},
         "flagged": [x for v in result["splits"].values() for x in v if x["n_diff"]]},
        indent=1) + "\n")
    print("\n".join(md))
    return result
