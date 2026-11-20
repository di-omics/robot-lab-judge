"""results/frozen/report.md: the frozen-features judge on dev, with the leak check and the control."""

from pathlib import Path

import numpy as np

from ..score import metrics, prop
from . import FEATURES, OUT
from .head import L2_GRID, N_FOLDS, auroc, balanced_accuracy

PLATE_CHORES = ("plate_to_tray", "plate_to_hotel_top", "plate_to_hotel_bottom", "stack_plates", "unstack_plate",
                "stack_tip_boxes")


def _size_mb(folder: Path) -> float:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e6


def write_report(fitted, control, control_per, dev, rows_dv, task_dv, y_dv, meta_dv, meta_tr, counts) -> str:
    from .fit import ROWS

    chores = sorted(set(task_dv.tolist()))
    out = ["# A linear probe on frozen features, on the scene set", "",
           "Question: is the pass or fail answer in the top-view pixels at all, for a probe fitted on the train split?",
           "", "A frozen vision tower turns pictures into numbers. A small logistic regression per chore, trained on "
           "train only, turns the numbers into PASS or FAIL. Every choice was made on train. Dev was predicted at the end.",
           "", "## Features", "",
           f"- Vision tower: the SigLIP 2 encoder inside `{meta_dv['model_id']}` (snapshot `{meta_dv['snapshot']}`), "
           f"{meta_dv['layers']} layers, hidden size {meta_dv['hidden']}, weights {meta_dv['dtype']}, "
           f"mlx {meta_dv['mlx']}, mlx-vlm {meta_dv['mlx_vlm']}.",
           f"- Tokens: {meta_dv['tokens']}. Pooled in float32.",
           f"- Patch grid, \"top\" picture (640 x 360 resized to 512 x 288): {meta_dv['grid_top'][0]} rows x "
           f"{meta_dv['grid_top'][1]} columns = {meta_dv['grid_top'][0] * meta_dv['grid_top'][1]} patches.",
           f"- Patch grid, each of the 8 \"fine\" crops (256 x 256 resized to 512 x 512): {meta_dv['grid_fine'][0]} x "
           f"{meta_dv['grid_fine'][1]} = {meta_dv['grid_fine'][0] * meta_dv['grid_fine'][1]} patches, "
           f"{8 * meta_dv['grid_fine'][0] * meta_dv['grid_fine'][1]} per frame.",
           "- Resize filter: bilinear, as the checkpoint's own image processor. Our patches match that processor's "
           "to 1.2e-07 on both picture sizes (`smoke.log`).",
           f"- Feature cache: `data/features/`, {_size_mb(FEATURES):.0f} MB in all (pooled numbers only, no per-patch "
           "features are kept).",
           f"- Train: {meta_tr['n']} scenes in {meta_tr['n_layouts']} layouts. Dev: {meta_dv['n']} scenes in "
           f"{meta_dv['n_layouts']} layouts.",
           "", "## Head", "",
           "- 1 logistic regression per chore (10 per row), class-balanced loss (weight n / (2 n_class)), "
           "L2 penalty on the weights (bias not penalized), loss = sum of weighted log losses + L2 / 2 * |w|^2.",
           "- Inputs standardized with the mean and standard deviation of the rows the head is fitted on.",
           f"- L2 picked from {list(L2_GRID)} by {N_FOLDS}-fold cross-validation on train, folds grouped by layout "
           "seed (the same folds for every chore and row), scaler fitted inside each fold. Score: mean over chores "
           "of the balanced accuracy of the pooled out-of-fold verdicts. A tie goes to the larger L2. Then refit on "
           "all of train.",
           "- Verdict PASS when p >= 0.5. The head sees only the pixel features and the chore name.",
           "", "## Results on dev", "",
           "Balanced accuracy is pooled over all dev scenes, as on the board. Ranges are exact 95% (Clopper-Pearson).",
           "",
           "| row | condition | L2 | train CV balanced accuracy | dev pass recall | dev fail recall | dev false-pass rate "
           "| dev balanced accuracy | mean of per-chore dev balanced accuracy | mean s per scene | dev predictions made |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    per_row = {}
    for name, condition, key, what, images in ROWS:
        p = dev[name]["p"]
        verdict = np.where(p >= 0.5, "PASS", "FAIL")
        rows = [{"truth": r["truth"], "verdict": v} for r, v in zip(rows_dv, verdict)]
        m = metrics(rows)
        cv = fitted[name]["cv"][fitted[name]["l2"]]
        per = {}
        for t in chores:
            sel = task_dv == t
            mt = metrics([x for x, s in zip(rows, sel) if s])
            per[t] = {"m": mt, "ba": balanced_accuracy(y_dv[sel], p[sel] >= 0.5), "auc": auroc(y_dv[sel], p[sel]),
                      "cv": cv["per_chore"][t]}
        per_row[name] = per
        out.append(f"| `{name}` | {condition} | {fitted[name]['l2']} | {cv['mean']:.3f} | "
                   f"{prop(m['pass_PASS'], m['n_pass'])} | {prop(m['fail_FAIL'], m['n_fail'])} | "
                   f"{prop(m['fail_PASS'], m['n_fail'])} | {m['balanced_accuracy']:.3f} | "
                   f"{np.mean([v['ba'] for v in per.values()]):.3f} | {dev[name]['seconds']:.3f} | {counts[name]} |")
    out += ["", "Dev confusion counts:", "",
            "| row | true pass judged PASS | true pass judged FAIL | true fail judged FAIL | true fail judged PASS |",
            "|---|---|---|---|---|"]
    for name, *_ in ROWS:
        p = dev[name]["p"]
        tp = int(((p >= 0.5) & (y_dv == 1)).sum())
        fn = int(((p < 0.5) & (y_dv == 1)).sum())
        tn = int(((p < 0.5) & (y_dv == 0)).sum())
        fp = int(((p >= 0.5) & (y_dv == 0)).sum())
        out.append(f"| `{name}` | {tp} | {fn} | {tn} | {fp} |")
    out += ["", "For reference, the best zero-shot judge on the board scored 0.65 on dev "
            "(`results/simset_dev/board.md`).", ""]

    out += ["## Train cross-validation by L2", "",
            "Mean over chores of the out-of-fold balanced accuracy.", "",
            "| row | " + " | ".join(f"L2 {x}" for x in L2_GRID) + " | chosen |",
            "|---|" + "---|" * (len(L2_GRID) + 1)]
    for name, *_ in ROWS:
        cv = fitted[name]["cv"]
        out.append(f"| `{name}` | " + " | ".join(f"{cv[x]['mean']:.3f}" for x in L2_GRID)
                   + f" | {fitted[name]['l2']} |")

    for name, condition, key, what, images in ROWS:
        per = per_row[name]
        out += ["", f"## Per chore: `{name}` ({condition})", "", f"Feature: {what}.", "",
                "| chore | n pass | n fail | true pass judged PASS | true fail judged FAIL | dev balanced accuracy "
                "| dev AUROC | train CV balanced accuracy |",
                "|---|---|---|---|---|---|---|---|"]
        for t in chores:
            v = per[t]
            mark = " (plate)" if t in PLATE_CHORES else ""
            out.append(f"| {t}{mark} | {v['m']['n_pass']} | {v['m']['n_fail']} | "
                       f"{prop(v['m']['pass_PASS'], v['m']['n_pass'])} | {prop(v['m']['fail_FAIL'], v['m']['n_fail'])} "
                       f"| {v['ba']:.3f} | {v['auc']:.3f} | {v['cv']:.3f} |")
        plate = [per[t] for t in chores if t in PLATE_CHORES]
        rest = [per[t] for t in chores if t not in PLATE_CHORES]
        out.append(f"| mean of the {len(plate)} plate chores | | | | | {np.mean([v['ba'] for v in plate]):.3f} "
                   f"| {np.mean([v['auc'] for v in plate]):.3f} | {np.mean([v['cv'] for v in plate]):.3f} |")
        out.append(f"| mean of the {len(rest)} other chores | | | | | {np.mean([v['ba'] for v in rest]):.3f} "
                   f"| {np.mean([v['auc'] for v in rest]):.3f} | {np.mean([v['cv'] for v in rest]):.3f} |")

    out += ["", "## Control: row B with shuffled train labels", "",
            "Row B's features with the train labels shuffled within each chore (seeded), then the same "
            "cross-validation. It must land near 0.50. It is not a verdict file.", "",
            "| | " + " | ".join(f"L2 {x}" for x in L2_GRID) + " |", "|---|" + "---|" * len(L2_GRID),
            "| mean over chores | " + " | ".join(f"{control[str(x)]:.3f}" for x in L2_GRID) + " |"]
    lo = min(min(v.values()) for v in control_per.values())
    hi = max(max(v.values()) for v in control_per.values())
    out += ["", f"Single chores range from {lo:.3f} to {hi:.3f} over all L2 values (about 180 scenes each, so "
            "single chores swing more).", ""]

    leak = OUT / "leakcheck.md"
    if leak.exists():
        out += ["", leak.read_text()]
        import json

        js = json.loads((OUT / "leakcheck.json").read_text())
        flagged = js.get("flagged", [])
        if flagged and all(f["task"] == "unstack_plate" and f["kind"] == "trap" for f in flagged):
            out += [f"All {len(flagged)} scenes with a differing arm pixel are `unstack_plate` traps where the "
                    "knocked-away plate or its shadow ends up on the edge of the left arm's black base plate (each "
                    "one was looked at). The arms themselves are pixel-identical between the first and last top view in "
                    "every train and dev scene, so nothing about the arms goes with kind or truth, and the "
                    "arms-flat control row B2 was not added.", ""]
        elif flagged:
            out += ["WARNING: arm pixels differ in scenes that are not unstack_plate traps. Read the list above.", ""]

    out += ["## Seconds per scene", "",
            "Measured during dev extraction on 1 machine, wall time with the GPU possibly shared. Encoding includes "
            "reading the PNG, cropping and resizing. Row C counts the first frame's 8 crops in full for every scene "
            "(a judge on its own would encode it every time).", "",
            "| row | mean s per scene (encode + head) | median encode s | head s per scene |", "|---|---|---|---|"]
    for name, *_ in ROWS:
        out.append(f"| `{name}` | {dev[name]['seconds']:.3f} | {dev[name]['enc_median']:.3f} | "
                   f"{dev[name]['head_s']:.6f} |")
    out += ["", "## Files", "",
            "- Verdicts: `results/simset_dev/verdicts_frozen__<row>_<condition>.jsonl`.",
            "- Fitted heads: `results/frozen/heads_<row>.npz`. Choices and folds: `results/frozen/fit_log.json`.",
            "- Every dev prediction: `results/frozen/dev_predictions.jsonl`.",
            "- Leak check: `results/frozen/leakcheck.md`. Smoke test: `results/frozen/smoke.log`.", ""]
    text = "\n".join(out) + "\n"
    (OUT / "report.md").write_text(text)
    return text
