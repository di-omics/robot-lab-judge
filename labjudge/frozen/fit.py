"""Cross-validate on train, refit on all of train, predict dev once per row, write verdicts and the report.

The head reads only the cached pixel features and the chore name. No other manifest field reaches it.
"""

import datetime
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from . import OUT, TOWER, VERDICTS, read_rows
from .extract import load
from .head import (L2_GRID, N_FOLDS, Scaler, auroc, balanced_accuracy, cv_oof, fit_logreg, grouped_folds,
                   predict_proba)

FOLD_SEED = 0
SHUFFLE_SEED = 1
## name, condition, cached feature, what it is, which images it uses
ROWS = (
    ("top-logreg", "after", "top",
     "top picture of the last frame (512 x 288), mean and max over patches, 1536 numbers", ("last_top",)),
    ("fine-logreg", "after", "fine",
     "8 fine crops of the last frame (512 x 512 each), mean and max over all patches, 1536 numbers", ("last_top",)),
    ("fine-change-logreg", "before_after", "change",
     "8 fine crops of the first and last frame, the 32 patches with the largest change, "
     "[mean f_last, mean f_first, mean d], 2304 numbers", ("first_views", "last_top")),
)
DEV_LOG = OUT / "dev_predictions.jsonl"


def head_inputs(split: str) -> tuple[list[dict], dict, np.ndarray, np.ndarray, np.ndarray]:
    """Rows, cached features, task, seed and label (1 = pass). Only these leave this function for the head."""
    rows = read_rows(split)
    feats = load(split)
    ids = [r["take_id"] for r in rows]
    if list(feats["take_id"]) != ids:
        raise SystemExit(f"cached {split} features do not match the {split} manifest; run extract again")
    task = np.array([r["task"] for r in rows])
    seed = np.array([int(r["seed"]) for r in rows])
    y = np.array([1 if r["truth"] == "pass" else 0 for r in rows])
    return rows, feats, task, seed, y


def cross_validate(x, task, y, folds) -> dict:
    """Per L2: per-chore balanced accuracy of the pooled out-of-fold verdicts, and their mean."""
    out = {}
    for l2 in L2_GRID:
        per = {}
        oof = np.full(len(y), np.nan)
        for t in sorted(set(task.tolist())):
            m = task == t
            p = cv_oof(x[m], y[m], folds[m], l2)
            oof[m] = p
            per[t] = balanced_accuracy(y[m], p >= 0.5)
        out[l2] = {"per_chore": per, "mean": float(np.mean(list(per.values()))), "oof": oof}
    return out


def pick(cv: dict) -> float:
    """Best mean balanced accuracy; a tie goes to the larger L2."""
    best = max(v["mean"] for v in cv.values())
    return max(l2 for l2, v in cv.items() if v["mean"] == best)


def refit(x, task, y, l2) -> dict:
    heads = {}
    for t in sorted(set(task.tolist())):
        m = task == t
        sc = Scaler(x[m])
        w, b = fit_logreg(sc(x[m]), y[m], l2)
        heads[t] = {"mean": sc.mean, "std": sc.std, "w": w, "b": b}
    return heads


def predict(heads, x, task) -> np.ndarray:
    p = np.full(len(task), np.nan)
    for t, h in heads.items():
        m = task == t
        p[m] = predict_proba((np.asarray(x[m], np.float64) - h["mean"]) / h["std"], h["w"], h["b"])
    if np.isnan(p).any():
        raise SystemExit("a dev chore has no fitted head")
    return p


def save_heads(path: Path, heads: dict, l2: float) -> None:
    arrays = {"l2": np.array(l2)}
    for t, h in heads.items():
        for k, v in h.items():
            arrays[f"{t}__{k}"] = np.asarray(v)
    np.savez_compressed(path, **arrays)


def write_verdicts(folder: Path, name: str, header: dict, rows: list[dict], p: np.ndarray, images, seconds: float):
    """verdicts_frozen__<row>_<condition>.jsonl: a header line, then 1 line per scene. Written to .partial first."""
    condition = header["condition"]
    path = Path(folder) / f"verdicts_frozen__{name}_{condition}.jsonl"
    partial = path.with_name(path.name + ".partial")
    with partial.open("w") as fh:
        fh.write(json.dumps(header) + "\n")
        for r, pi in zip(rows, p):
            fh.write(json.dumps({
                "take_id": r["take_id"], "run": r["run"], "task": r["task"], "seed": r["seed"],
                "truth": r["truth"], "condition": condition, "images": [r["images"][k] for k in images],
                "verdict": "PASS" if pi >= 0.5 else "FAIL", "reply": f"p={pi:.4f}",
                "seconds": round(seconds, 4), "p": round(float(pi), 6),
            }) + "\n")
    partial.replace(path)
    return path


def dev_count(row: str) -> int:
    if not DEV_LOG.exists():
        return 0
    return sum(json.loads(x)["row"] == row for x in DEV_LOG.read_text().splitlines() if x.strip())


def k_of_n(k: int, n: int) -> str:
    from ..score import prop

    return prop(k, n)


def run(control_only: bool = False) -> None:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    rows_tr, f_tr, task_tr, seed_tr, y_tr = head_inputs("train")
    folds = grouped_folds(seed_tr, N_FOLDS, FOLD_SEED)
    for f in range(N_FOLDS):  # a seed never sits in 2 folds
        assert not set(seed_tr[folds == f]) & set(seed_tr[folds != f])
    log = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "folds": {
        str(f): sorted(set(seed_tr[folds == f].tolist())) for f in range(N_FOLDS)}, "rows": {}}

    # 1. every choice, from train only
    fitted = {}
    for name, condition, key, what, images in ROWS:
        x = f_tr[key].astype(np.float64)
        s = time.perf_counter()
        cv = cross_validate(x, task_tr, y_tr, folds)
        l2 = pick(cv)
        heads = refit(x, task_tr, y_tr, l2)
        save_heads(OUT / f"heads_{name}.npz", heads, l2)
        fitted[name] = {"cv": cv, "l2": l2, "heads": heads}
        log["rows"][name] = {"l2": l2, "cv_mean": {str(k): v["mean"] for k, v in cv.items()},
                             "cv_per_chore": cv[l2]["per_chore"], "seconds": round(time.perf_counter() - s, 1)}
        print(f"{name}: train CV mean balanced accuracy by L2 "
              f"{ {k: round(v['mean'], 3) for k, v in cv.items()} }, chose {l2}", flush=True)

    # 2. control: row B with labels shuffled within each chore
    rng = np.random.default_rng(SHUFFLE_SEED)
    y_sh = y_tr.copy()
    for t in sorted(set(task_tr.tolist())):
        idx = np.flatnonzero(task_tr == t)
        y_sh[idx] = y_tr[rng.permutation(idx)]
    cv_sh = cross_validate(f_tr["fine"].astype(np.float64), task_tr, y_sh, folds)
    control = {str(k): v["mean"] for k, v in cv_sh.items()}
    control_per = {str(k): v["per_chore"] for k, v in cv_sh.items()}
    log["shuffled_control"] = {"mean": control, "per_chore": control_per}
    print(f"shuffled-label control (row B): train CV mean balanced accuracy by L2 {control}", flush=True)
    (OUT / "fit_log.json").write_text(json.dumps(log, indent=1) + "\n")
    if control_only:
        print("control only: dev not predicted")
        return

    # 3. dev, once per row, after every choice above is fixed
    rows_dv, f_dv, task_dv, seed_dv, y_dv = head_inputs("dev")
    meta = f_dv["meta"]
    dev = {}
    for name, condition, key, what, images in ROWS:
        heads = fitted[name]["heads"]
        s = time.perf_counter()
        p = predict(heads, f_dv[key], task_dv)
        head_s = (time.perf_counter() - s) / len(p)
        with DEV_LOG.open("a") as fh:
            fh.write(json.dumps({"row": name, "date": datetime.datetime.now().isoformat(timespec="seconds"),
                                 "n": len(p)}) + "\n")
        if key == "top":
            enc = f_dv["sec_top"]
        elif key == "fine":
            enc = f_dv["sec_fine"]
        else:
            enc = f_dv["sec_fine"] + f_dv["sec_first"] + f_dv["sec_change"]
        seconds = float(np.mean(enc) + head_s)
        dev[name] = {"p": p, "seconds": seconds, "enc_median": float(np.median(enc)), "head_s": head_s}
        cv = fitted[name]["cv"][fitted[name]["l2"]]
        header = {
            "kind": "header", "model_id": f"frozen/{TOWER}/{name}", "thinking": "-", "condition": condition,
            "data": "data/simset/dev", "date": datetime.date.today().isoformat(),
            "trained_on": "data/simset/train", "l2": fitted[name]["l2"], "cv_balanced_accuracy": round(cv["mean"], 4),
            "cv": f"{N_FOLDS}-fold grouped by layout seed, fold seed {FOLD_SEED}", "feature": what,
            "tokens": meta["tokens"], "vision_model_id": meta["model_id"], "snapshot": meta["snapshot"],
            "mlx": version("mlx"), "mlx_vlm": version("mlx-vlm"), "threshold": 0.5,
            "head": "1 logistic regression per chore, class-balanced, L2, standardized with train statistics",
        }
        path = write_verdicts(VERDICTS, name, header, rows_dv, p, images, seconds)
        print(f"wrote {path}", flush=True)

    from .report import write_report

    write_report(fitted, control, control_per, dev, rows_dv, task_dv, y_dv, meta, f_tr["meta"],
                 {k: dev_count(k) for k, *_ in ROWS})
    print(f"done in {time.perf_counter() - t0:.0f} s")
