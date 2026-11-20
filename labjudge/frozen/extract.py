"""Encode every scene of train or dev once and cache the features the 3 rows need.

Per scene (all float32):
- `top`: mean and max over the patches of the "top" picture of the last frame (1536 numbers), row A.
- `fine`: mean and max over all patches of the 8 "fine" crops of the last frame (1536 numbers), row B.
- `change`: per patch d = f_last - f_first over the 8 "fine" crops of both frames; the 32 patches with the
  largest |d| (L2 norm over the 768 numbers); then [mean f_last, mean f_first, mean d] over them (2304), row C.
The first frame is shared by every scene of a layout, so it is encoded once per layout.
Per-patch features are never written to disk, only these pooled numbers.
"""

import datetime
import json
import time
from importlib.metadata import version

import numpy as np

from . import FEATURES, MODEL_ID, check_path, check_split, layout_of, read_rows
from .pictures import fine_pictures, first_top, last_top, top_picture

TOP_K = 32  # fixed in the design, never tuned


def mean_max(f: np.ndarray) -> np.ndarray:
    """f: (..., hidden). Mean and max over every patch of every picture."""
    flat = f.reshape(-1, f.shape[-1])
    return np.concatenate([flat.mean(axis=0), flat.max(axis=0)])


def change_feature(f_last: np.ndarray, f_first: np.ndarray, k: int = TOP_K) -> tuple[np.ndarray, np.ndarray]:
    """[mean f_last, mean f_first, mean d] over the k patches with the largest |d|, and their indices.

    Ties (for example an untouched scene where d is 0 everywhere) keep patch order, so the result is fixed.
    """
    a = f_last.reshape(-1, f_last.shape[-1])
    b = f_first.reshape(-1, f_first.shape[-1])
    d = a - b
    norm = np.sqrt((d.astype(np.float64) ** 2).sum(axis=1))
    idx = np.argsort(-norm, kind="stable")[:k]
    return np.concatenate([a[idx].mean(axis=0), b[idx].mean(axis=0), d[idx].mean(axis=0)]), idx


def cache_path(split: str):
    return check_path(FEATURES / f"{check_split(split)}.npz")


def extract(split: str, limit: int = 0) -> None:
    from .encoder import Tower

    rows = read_rows(split)
    t0 = time.perf_counter()
    tower = Tower()
    print(f"loaded {MODEL_ID} vision tower ({tower.n_layers} layers, hidden {tower.hidden}, {tower.dtype}) "
          f"in {time.perf_counter() - t0:.1f} s", flush=True)
    by_layout: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        by_layout.setdefault(layout_of(r), []).append(i)
    n = len(rows)
    top = np.zeros((n, 2 * tower.hidden), np.float32)
    fine = np.zeros((n, 2 * tower.hidden), np.float32)
    change = np.zeros((n, 3 * tower.hidden), np.float32)
    sec_top = np.zeros(n)
    sec_fine = np.zeros(n)
    sec_first = np.zeros(n)
    sec_change = np.zeros(n)
    grids = {}
    done = 0
    for li, (key, idxs) in enumerate(by_layout.items(), 1):
        s = time.perf_counter()
        f_first, g = tower.encode(fine_pictures(first_top(rows[idxs[0]])))
        t_first = time.perf_counter() - s
        grids["fine"] = g
        for i in idxs:
            r = rows[i]
            s = time.perf_counter()
            f_top, g = tower.encode([top_picture(last_top(r))])
            top[i] = mean_max(f_top)
            sec_top[i] = time.perf_counter() - s
            grids["top"] = g
            s = time.perf_counter()
            f_last, g = tower.encode(fine_pictures(last_top(r)))
            fine[i] = mean_max(f_last)
            sec_fine[i] = time.perf_counter() - s
            s = time.perf_counter()
            change[i], _ = change_feature(f_last, f_first)
            sec_change[i] = time.perf_counter() - s
            sec_first[i] = t_first
            done += 1
        print(f"{split} layout {li}/{len(by_layout)} {key[0]} s{key[1]}: {len(idxs)} scenes, "
              f"{done}/{n} done, {time.perf_counter() - t0:.0f} s", flush=True)
    meta = {
        "model_id": MODEL_ID, "snapshot": tower.snapshot, "mlx": version("mlx"), "mlx_vlm": version("mlx-vlm"),
        "tokens": "vision tower last hidden state after its post layer norm, before the multimodal projector",
        "layers": tower.n_layers, "hidden": tower.hidden, "dtype": tower.dtype,
        "grid_top": list(grids["top"]), "grid_fine": list(grids["fine"]), "top_k": TOP_K,
        "split": split, "n": n, "n_layouts": len(by_layout), "date": datetime.date.today().isoformat(),
    }
    out = cache_path(split)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.npz")
    np.savez(tmp, take_id=np.array([r["take_id"] for r in rows]), top=top, fine=fine, change=change,
             sec_top=sec_top, sec_fine=sec_fine, sec_first=sec_first, sec_change=sec_change,
             meta=np.array(json.dumps(meta)))
    tmp.replace(out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - t0:.0f} s; "
          f"grids top {grids['top']} fine {grids['fine']}", flush=True)


def load(split: str) -> dict:
    """The cached features of a split, rows in manifest order."""
    z = np.load(cache_path(split))
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def smoke(n: int = 20) -> None:
    """On the first n train scenes: same picture twice gives the same features, different scenes differ, timing."""
    from .encoder import Tower, to_patches

    rows = read_rows("train")[:n]
    tower = Tower()
    # 1. our preprocessing equals the checkpoint's own image processor on a picture that needs no resize
    from mlx_vlm.utils import load_processor

    proc = load_processor(tower.path)
    pic = fine_pictures(last_top(rows[0]))[0]
    ref = proc.image_processor([pic], do_image_splitting=False)
    mine, grid = to_patches(pic)
    rv = np.asarray(ref["pixel_values"])[0][: mine.shape[0]]
    print(f"processor parity on a 512 x 512 crop: grid ours {grid}, processor {tuple(np.asarray(ref['spatial_shapes'])[0])}, "
          f"max abs patch difference {np.abs(rv - mine).max():.2e}")
    tp = top_picture(last_top(rows[0]))
    ref = proc.image_processor([tp], do_image_splitting=False)
    mine, grid = to_patches(tp)
    rv = np.asarray(ref["pixel_values"])[0][: mine.shape[0]]
    print(f"processor parity on the 512 x 288 top picture: grid ours {grid}, processor "
          f"{tuple(np.asarray(ref['spatial_shapes'])[0])}, max abs patch difference {np.abs(rv - mine).max():.2e}")
    # 2. repeatability and timing
    feats, times = [], []
    worst = 0.0
    for r in rows:
        s = time.perf_counter()
        a, ga = tower.encode([top_picture(last_top(r))])
        b, gb = tower.encode(fine_pictures(last_top(r)))
        times.append(time.perf_counter() - s)
        a2, _ = tower.encode([top_picture(last_top(r))])
        b2, _ = tower.encode(fine_pictures(last_top(r)))
        worst = max(worst, float(np.abs(a - a2).max()), float(np.abs(b - b2).max()))
        feats.append((r, mean_max(a), mean_max(b)))
    print(f"grids: top {ga}, fine {gb} x 8 crops; hidden {a.shape[-1]}")
    print(f"same picture twice: largest absolute difference {worst} over {len(rows)} scenes")
    print(f"seconds per scene for top + 8 fine crops of the last frame: median {np.median(times):.3f}, "
          f"max {np.max(times):.3f}")

    # 3. different scenes give different features
    def cos(u, v):
        return float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))

    for i in range(1, len(feats)):
        r0, r1 = feats[0][0], feats[i][0]
        print(f"  {r0['take_id']} vs {r1['take_id']}: top |diff| max {np.abs(feats[0][1] - feats[i][1]).max():.3f}, "
              f"fine |diff| max {np.abs(feats[0][2] - feats[i][2]).max():.3f}, fine cosine {cos(feats[0][2], feats[i][2]):.4f}")
