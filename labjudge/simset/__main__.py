"""CLI for the near-miss scene set. Runs under the Python that has the simulator (except seal).

  python -m labjudge.simset calibrate              (bench from --bench or env LAB_HANDLING_BENCH)
  python -m labjudge.simset generate --split dev
  python -m labjudge.simset check
  python -m labjudge.simset seal
  python -m labjudge.simset sheet

Data goes to data/simset/<split>/; the calibration to results/simset/.
"""

import argparse
import datetime
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import spec

RESULTS = Path("results/simset")
CALIBRATION = RESULTS / "calibration.json"
DIRECTIONS = 3  # per calibration layout
TILTS = 5  # per calibration layout
SWEEP = 16  # random offsets per calibration layout, drawn the way generate draws them
BISECT = 5
DEFAULT_SCENES = 6
RETRIES = 4
OFFSET_RETRIES = 12  # a blocked offset costs no physics, so offsets get more tries


def _load(args):
    from . import bench

    bench.load(args.bench)
    return bench


# calibrate


def fit_flip(pairs):
    """The one threshold that best splits (r mm, passed) pairs into pass below and fail above.
    @return (threshold mm, pairs it gets wrong). Ties take the geometric middle of the best gap."""
    if not pairs:
        return 60.0, 0
    rs = sorted({r for r, _ in pairs})
    cuts = [0.5] + [math.sqrt(a * b) for a, b in zip(rs, rs[1:])] + [60.0]
    scored = [(sum((r < c) != p for r, p in pairs), c) for c in cuts]
    best = min(w for w, _ in scored)
    tied = [c for w, c in scored if w == best]
    return math.sqrt(tied[0] * tied[-1]), best


def flip_distance(layout, plan_at, reasons):
    """Bisect the planar offset (log scale, 0.5 to 60 mm) where the label turns from pass to fail.
    @return (flip mm, note, evaluations, discards). A discarded scene counts as not a pass."""
    evals, discards = 0, 0

    def passes(r):
        nonlocal evals, discards
        evals += 1
        result, _ = layout.run_plan(plan_at(r))
        if result["discard"]:
            discards += 1
            reasons[result["discard"]] += 1
            return False
        return result["truth"] == "pass"

    lo, hi = 0.5, 60.0
    if not passes(lo):
        return lo, "fails at 0.5 mm", evals, discards
    if passes(hi):
        return hi, "passes at 60 mm", evals, discards
    for _ in range(BISECT):
        mid = math.sqrt(lo * hi)
        if passes(mid):
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi), "", evals, discards


def cmd_calibrate(args):
    bench = _load(args)
    from . import scenes as S

    t0 = time.perf_counter()
    included, excluded = bench.included_tasks()
    tasks = args.tasks or included
    out = {"date": datetime.date.today().isoformat(), "seeds": list(spec.CALIBRATION_SEEDS),
           "versions": bench.versions(), "excluded": excluded, "included": included, "tasks": {}}
    for task in tasks:
        flips, notes, tilt_pass, tilt_n, evals, discards = [], [], 0, 0, 0, 0
        reasons = Counter()
        sweep = []
        for seed in spec.CALIBRATION_SEEDS:
            L = S.Layout(task, seed, render=False)
            try:
                if not L.oracle_pass:
                    notes.append(f"s{seed}: oracle did not pass")
                    continue
                for d in range(DIRECTIONS):
                    rng = S.rng_for(seed, task, 1000 + d)
                    base = S.plan_offset(L, rng, r_mm=1.0, yaw=False)
                    angle = math.atan2(base["dy_mm"], base["dx_mm"])

                    def plan_at(r, base=base, angle=angle):
                        return {**base, "r_mm": r, "dx_mm": r * math.cos(angle), "dy_mm": r * math.sin(angle)}

                    f, note, e, dc = flip_distance(L, plan_at, reasons)
                    flips.append(round(f, 2))
                    evals += e
                    discards += dc
                    if note:
                        notes.append(f"s{seed} {base['prop']} {math.degrees(angle):.0f} deg: {note}")
                for k in range(SWEEP):
                    for attempt in range(RETRIES):
                        plan = S.plan_offset(L, S.rng_for(seed, task, 3000 + k, attempt))
                        result, _ = L.run_plan(plan)
                        evals += 1
                        if result["discard"]:
                            discards += 1
                            reasons[result["discard"]] += 1
                            continue
                        sweep.append((plan["r_mm"], result["truth"] == "pass"))
                        break
                for k in range(TILTS):
                    result, _ = L.run_plan(S.plan_tilt(L, S.rng_for(seed, task, 2000 + k)))
                    evals += 1
                    if result["discard"]:
                        discards += 1
                        reasons[result["discard"]] += 1
                        continue
                    tilt_n += 1
                    tilt_pass += result["truth"] == "pass"
            finally:
                L.close()
        med = statistics.median(flips) if flips else None
        fit, wrong = fit_flip(sweep)
        out["tasks"][task] = {"flips_mm": flips, "median_flip_mm": round(med, 2) if med else None,
                              "fit_flip_mm": round(fit, 2), "fit_wrong": wrong, "sweep_n": len(sweep),
                              "sweep_pass": sum(p for _, p in sweep),
                              "tilt_pass": tilt_pass, "tilt_n": tilt_n, "evals": evals, "discards": discards,
                              "discard_reasons": dict(reasons), "notes": notes}
        print(f"{task}: median flip {med} mm from {flips}; fit flip {fit:.2f} mm ({wrong} of {len(sweep)} off); "
              f"tilt pass {tilt_pass} of {tilt_n}; "
              f"{evals} scenes, {discards} discarded", flush=True)
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.tasks and CALIBRATION.exists():  # a partial run adds to the saved one
        saved = json.loads(CALIBRATION.read_text())
        saved["tasks"].update(out["tasks"])
        out = {**saved, "tasks": saved["tasks"]}
    CALIBRATION.write_text(json.dumps(out, indent=2) + "\n")
    (RESULTS / "calibration.md").write_text(calibration_md(out))
    print(f"wrote {CALIBRATION} and {RESULTS / 'calibration.md'} in {out['wall_s']} s")


def calibration_md(cal: dict) -> str:
    lines = [
        "# Scene set calibration: where the label flips",
        "",
        f"Date {cal['date']}. Bench code `{cal['versions']['code']}`, mujoco {cal['versions']['mujoco']}.",
        f"Layout seeds {', '.join(str(s) for s in cal['seeds'])} (train range). Per layout, {DIRECTIONS} random "
        f"directions, each with one random placed prop and no yaw turn. The offset is bisected on a log scale "
        f"from 0.5 to 60 mm ({BISECT} steps after the 2 ends). A discarded scene counts as not a pass.",
        "The flip distance is the planar offset where the frozen rule's label on the settled state turns "
        "from pass to fail.",
        "Discard reasons: `blocked` means the moved prop still overlapped something after being raised 80 mm "
        "(for example a plate pushed into a side wall of the hotel); `not at rest` means a prop still moved after "
        "3 rest windows. Both count as not a pass here, so flips in directions that block early are short. "
        "In the generated splits a discarded scene is drawn again (up to 12 tries for an offset, 4 for others) and counted in info.json. When the sweep saw no fail at all, `generate` aims at the bisection median instead.",
        "",
        f"Then, per layout, {SWEEP} random offsets drawn the way `generate` draws them (log-uniform 0.5 to 60 mm, "
        "30% with a yaw turn, a discarded scene drawn again), and one threshold fitted to them: the fit flip. "
        "`generate` aims its offsets around the fit flip, because the bisection counts blocked scenes as fails.",
        "",
        "| task | median flip mm | flips mm (each direction) | fit flip mm (sweep scenes it gets wrong) "
        "| tilt scenes that pass | discarded scenes | notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for task, t in cal["tasks"].items():
        flips = ", ".join(f"{f:g}" for f in t["flips_mm"])
        notes = "; ".join(t["notes"]) or "-"
        why = ", ".join(f"{k} {v}" for k, v in sorted(t.get("discard_reasons", {}).items()))
        lines.append(f"| {task} | {t['median_flip_mm']} | {flips} | {t.get('fit_flip_mm')} "
                     f"({t.get('fit_wrong')} of {t.get('sweep_n')}) | {t['tilt_pass']} of {t['tilt_n']} "
                     f"| {t['discards']} of {t['evals']}{' (' + why + ')' if why else ''} | {notes} |")
    lines += ["", "## Tasks left out", "",
              "A still frame cannot show the way there, so tasks whose pass also needs an earlier stage are left out.", ""]
    for task, why in cal["excluded"].items():
        lines.append(f"- `{task}`: {why}.")
    lines.append("")
    return "\n".join(lines)


# generate


def aims(cal: dict, n_scenes: int) -> dict:
    """Per task: the offset range that should bring the pass share near one half."""
    from . import scenes as S

    out = {}
    n_tilt, n_off = S.kind_means(n_scenes)
    for task, t in cal["tasks"].items():
        flip = t.get("fit_flip_mm") or t["median_flip_mm"]
        if flip >= 59.9 and t["median_flip_mm"]:  # no fail seen in the sweep: fall back to the bisection
            flip = min(flip, t["median_flip_mm"])
        if not flip or not n_off:
            continue
        tilt = t["tilt_pass"] / t["tilt_n"] if t["tilt_n"] else 0.0
        q = (n_scenes / 2 - 1 - tilt * n_tilt) / n_off  # clean passes, trap or partial fails
        q = min(0.85, max(0.2, q))
        lo, hi = S.aim_range(flip, q)
        out[task] = {"flip_mm": flip, "q": round(q, 3), "range_mm": [round(lo, 3), round(hi, 3)]}
    return out


def cmd_generate(args):
    bench = _load(args)
    from . import render
    from . import scenes as S

    split = args.split
    cal_path = Path(args.calibration)
    if not cal_path.exists():
        raise SystemExit(f"no {cal_path}; run calibrate first")
    cal = json.loads(cal_path.read_text())
    included = cal["included"]
    tasks = args.tasks or spec.tasks_for(split, included)
    seeds = list(spec.SPLITS[split])[: args.layouts or None]
    aim = aims(cal, args.scenes)
    folder = Path(args.out) / split
    frames_dir = folder / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    partial = folder / "manifest.jsonl.partial"
    partial.write_text("")
    t0 = time.perf_counter()
    rows, discards, retries, layout_s, scene_s = [], Counter(), Counter(), [], []
    skipped = []

    for task in tasks:
        task_aim = aim.get(task, {}).get("range_mm")
        for ordinal, seed in enumerate(seeds):
            t_lay = time.perf_counter()
            L = S.Layout(task, seed)
            try:
                if not L.oracle_pass or L.oracle_problems:
                    skipped.append(f"{task} s{seed}: oracle {L.oracle_problems or 'did not pass'}")
                    continue
                stem = frames_dir / f"{task}-s{seed}"
                first = render.save_views(L.first_images, stem, "first")
                layout_s.append(time.perf_counter() - t_lay)
                for index, (kind, trap) in enumerate(S.schedule(args.scenes, ordinal, L)):
                    tries = 1 if kind == "trap" else OFFSET_RETRIES if kind == "offset" else RETRIES
                    for attempt in range(tries):
                        t_sc = time.perf_counter()
                        plan = S.make_plan(L, kind, trap, S.rng_for(seed, task, index, attempt), task_aim)
                        result, b = L.run_plan(plan)
                        scene_s.append(time.perf_counter() - t_sc)
                        if result["discard"]:
                            discards[(task, plan["kind"], result["discard"])] += 1
                            continue
                        sid = spec.scene_id(split, task, seed, index, plan["kind"])
                        images = {**first, **render.save_views(L.env.images(), frames_dir / sid, "last")}
                        rows.append(scene_row(split, L, sid, index, attempt, plan, result, images))
                        with partial.open("a") as f:
                            f.write(json.dumps(rows[-1]) + "\n")
                        if attempt:
                            retries[(task, plan["kind"])] += 1
                        break
            finally:
                L.close()
            done = len(rows)
            print(f"{split} {task} s{seed}: {done} scenes so far, {time.perf_counter() - t0:.0f} s", flush=True)
    (folder / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    partial.unlink()
    wall = time.perf_counter() - t0
    info = split_info(split, bench, cal, aim, args, rows, discards, retries, skipped, layout_s, scene_s, wall, tasks)
    (folder / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("n_scenes", "pass_share", "discards_total", "wall_s",
                                           "seconds_per_scene")}, indent=1))


def scene_row(split, L, sid, index, attempt, plan, result, images):
    task = L.task
    commanded = {
        "dx_mm": plan.get("dx_mm"), "dy_mm": plan.get("dy_mm"), "r_mm": plan.get("r_mm"),
        "yaw_deg": plan.get("yaw_deg"), "tilt_deg": plan.get("tilt_deg"), "tilt_axis_deg": plan.get("tilt_axis_deg"),
        "jitter_mm": plan.get("jitter"),
    }
    return {
        "take_id": sid, "run": split, "task": task.name, "title": task.title, "prompt": task.prompt,
        "seed": L.seed, "policy": "scene", "model": None, "truth": result["truth"], "frames": 1,
        "images": images,
        "kind": plan["kind"], "prop": result["prop"], "trap": plan.get("trap"), "commanded": commanded,
        "error": result["error"], "errors": result["errors"], "raised_mm": result["raised_mm"],
        "rest": result["rest"], "passed_while_settling": result["passed_while_settling"],
        "rule": task.rule, "seen_task": task.name not in spec.HELD_OUT,
        "scene_index": index, "attempt": attempt, "plan": plan,
    }


def split_info(split, bench, cal, aim, args, rows, discards, retries, skipped, layout_s, scene_s, wall, tasks):
    from . import scenes as S

    per = defaultdict(Counter)
    for r in rows:
        per[r["task"]][r["kind"]] += 1
        per[r["task"]]["pass"] += r["truth"] == "pass"
        per[r["task"]]["n"] += 1
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "split": split, "seeds": [min(spec.SPLITS[split]), max(spec.SPLITS[split])],
        "versions": bench.versions(), "python": sys.version.split()[0],
        "generator": {
            "scenes_per_layout": args.scenes, "layouts": args.layouts or len(spec.SPLITS[split]),
            "settle_s": S.SETTLE_S, "rest_s": S.REST_S, "oracle_extra_s": S.ORACLE_EXTRA_S, "lift_m": S.LIFT,
            "max_raise_m": S.MAX_RAISE, "rest_speed_m_s": S.REST_SPEED, "rest_move_m": S.REST_MOVE,
            "rest_turn_deg": S.REST_TURN, "extra_rest_windows": S.EXTRA_WINDOWS, "tilt_deg": S.TILT_DEG, "yaw_share": S.YAW_SHARE,
            "yaw_max_deg": S.YAW_MAX_DEG, "span": S.SPAN, "retries": RETRIES, "offset_retries": OFFSET_RETRIES, "aim": aim,
            "calibration_date": cal["date"],
        },
        "tasks": tasks, "held_out": list(spec.HELD_OUT), "excluded": cal["excluded"],
        "n_scenes": len(rows),
        "counts": {t: dict(c) for t, c in sorted(per.items())},
        "pass_share": {t: round(c["pass"] / c["n"], 3) for t, c in sorted(per.items())},
        "discards": [{"task": t, "kind": k, "reason": why, "n": n} for (t, k, why), n in sorted(discards.items())],
        "discards_total": sum(discards.values()),
        "retried": [{"task": t, "kind": k, "n": n} for (t, k), n in sorted(retries.items())],
        "skipped_layouts": skipped,
        "wall_s": round(wall, 1),
        "seconds_per_layout_setup": round(statistics.mean(layout_s), 2) if layout_s else None,
        "seconds_per_scene": round(statistics.mean(scene_s), 2) if scene_s else None,
    }


# check


def banner_like(rows_rgb) -> bool:
    import numpy as np

    close = (np.abs(rows_rgb.astype(int) - (250, 250, 248)).sum(-1) <= 6).mean()
    return close > 0.6


def cmd_check(args):
    import random

    import numpy as np
    from PIL import Image

    from ..__main__ import images_for, read_manifest

    root = Path(args.out)
    hard = []
    seeds = {}
    all_rows = {}
    print("== manifests")
    for split in spec.SPLITS:
        folder = root / split
        if not (folder / "manifest.jsonl").exists():
            print(f"{split}: no manifest")
            continue
        rows = read_manifest(folder)  # the judge's own reader
        all_rows[split] = rows
        seeds[split] = {r["seed"] for r in rows}
        bad = [(r["take_id"], p) for r in rows for p in spec.row_problems(r)]
        missing = [p for r in rows for c in ("after", "before_after") for p in images_for(r, c) if not Path(p).exists()]
        missing += [r["images"]["last_top"] for r in rows if not Path(r["images"]["last_top"]).exists()]
        ids = Counter(r["take_id"] for r in rows)
        dup = [k for k, n in ids.items() if n > 1]
        print(f"{split}: {len(rows)} rows, {len(bad)} row problems, {len(missing)} missing images, {len(dup)} duplicate ids")
        hard += [f"{split} {sid}: {p}" for sid, p in bad[:5]] + [f"{split}: missing {p}" for p in missing[:5]]
        hard += [f"{split}: duplicate id {d}" for d in dup[:5]]
        clean_fail = [r["take_id"] for r in rows if r["kind"] == "clean" and r["truth"] != "pass"]
        trap_pass = [r["take_id"] for r in rows if r["kind"] == "trap" and r["truth"] != "fail"]
        print(f"{split}: clean scenes {sum(r['kind'] == 'clean' for r in rows)}, not pass {len(clean_fail)}; "
              f"trap scenes {sum(r['kind'] == 'trap' for r in rows)}, not fail {len(trap_pass)}")
        hard += [f"clean scene is not pass: {s}" for s in clean_fail] + [f"trap scene is not fail: {s}" for s in trap_pass]
    print("== seed ranges")
    problems = spec.splits_disjoint()
    names = list(seeds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if seeds[a] & seeds[b]:
                problems.append(f"{a} and {b} manifests share seeds")
    for split, s in seeds.items():
        if s - set(spec.SPLITS[split]):
            problems.append(f"{split} manifest has seeds outside its range")
    print("disjoint, no bench seeds" if not problems else "; ".join(problems))
    hard += problems
    print("== banner rows")
    n_img, banner = 0, []
    for split, rows in all_rows.items():
        paths = sorted({p for r in rows for p in r["images"].values()})
        for p in paths:
            a = np.asarray(Image.open(p).convert("RGB"))
            n_img += 1
            want = (360, 640) if p.endswith("_top.png") else (540, 640)
            if a.shape[:2] != want or banner_like(a[:26]):
                banner.append(p)
    print(f"{n_img} images checked, {len(banner)} with banner rows or the wrong size")
    hard += [f"banner or size: {p}" for p in banner[:5]]
    print("== pass share per task (aim 35% to 65%)")
    outside = []
    for split, rows in all_rows.items():
        per = defaultdict(list)
        for r in rows:
            per[r["task"]].append(r["truth"] == "pass")
        for task, v in sorted(per.items()):
            share = sum(v) / len(v)
            if not spec.PASS_SHARE[0] <= share <= spec.PASS_SHARE[1]:
                outside.append(f"{split} {task}: {sum(v)} of {len(v)} ({share:.0%})")
    print("all inside" if not outside else "outside: " + "; ".join(outside))
    print(f"== regenerate {args.sample} scenes in fresh environments")
    bench = _load(args)
    from . import scenes as S

    pool = [r for rows in all_rows.values() for r in rows]
    random.Random(7).shuffle(pool)
    same = 0
    for r in pool[: args.sample]:
        L = S.Layout(r["task"], r["seed"], render=False)  # a new environment, built from nothing
        try:
            plan = {k: v for k, v in r["plan"].items() if k != "trap_problems"}
            result, _ = L.run_plan(plan)
        finally:
            L.close()
        ok = result["truth"] == r["truth"] and not result["discard"]
        d = abs(result["error"]["planar_mm"] - r["error"]["planar_mm"]) if result.get("error") else None
        same += ok
        if r["run"] == "test":  # a sealed scene is compared, but its label and error are never printed
            print(f"{'same' if ok else 'DIFFERENT'} {r['take_id']}: label hidden (sealed), error diff {d} mm")
            if not ok:
                hard.append(f"regenerated label differs: {r['take_id']}")
            continue
        print(f"{'same' if ok else 'DIFFERENT'} {r['take_id']}: {r['truth']} -> {result['truth']}"
              f" (planar error now {result.get('error', {}).get('planar_mm')} mm, was {r['error']['planar_mm']} mm,"
              f" diff {d} mm)")
        if not ok:
            hard.append(f"regenerated label differs: {r['take_id']}")
    print(f"{same} of {min(args.sample, len(pool))} regenerated scenes give the same label")
    sealed = root / "test" / spec.SEALED
    if sealed.exists():
        match = spec.seal_matches(root / "test")
        print(f"== seal: SEALED.md {'matches' if match else 'DOES NOT MATCH'} the test files")
        if not match:
            hard.append("test split changed after sealing")
    print("== result")
    print("all checks pass" if not hard else "PROBLEMS:\n  " + "\n  ".join(hard))
    if hard:
        raise SystemExit(1)
    _ = bench


# seal and sheet


def cmd_seal(args):
    folder = Path(args.out) / "test"
    if (folder / spec.SEALED).exists() and not args.again:
        raise SystemExit(f"{folder / spec.SEALED} exists; pass --again to rewrite it")
    hashes = spec.seal_hashes(folder)
    text = spec.seal_text(hashes, datetime.date.today().isoformat())
    (folder / spec.SEALED).write_text(text)
    print(text)


def cmd_sheet(args):
    from . import render

    folder = Path(args.out) / args.split
    if args.split == "test":
        raise SystemExit("no contact sheets of the sealed test split")
    rows = sorted(spec.read_rows(folder), key=lambda r: r["take_id"])
    step = max(1, len(rows) // args.n)
    picked = rows[::step][: args.n]
    out = folder / "sheet.jpg"
    render.sheet(picked, out, f"Scene set, {args.split}: {len(picked)} of {len(rows)} scenes, top views. "
                              "Caption: task, seed, kind, truth.")
    print(f"wrote {out} ({len(picked)} scenes)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="labjudge.simset")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--bench", help="the lab handling bench folder (else env LAB_HANDLING_BENCH)")
        q.add_argument("--out", default=str(spec.DATA))

    c = sub.add_parser("calibrate", help="find the offset where each task's label flips")
    common(c)
    c.add_argument("--tasks", nargs="+")
    g = sub.add_parser("generate", help="make one split's scenes")
    common(g)
    g.add_argument("--split", required=True, choices=list(spec.SPLITS))
    g.add_argument("--scenes", type=int, default=DEFAULT_SCENES, help="scenes per layout")
    g.add_argument("--layouts", type=int, default=0, help="only the first N layout seeds (smoke test)")
    g.add_argument("--tasks", nargs="+")
    g.add_argument("--calibration", default=str(CALIBRATION))
    k = sub.add_parser("check", help="verify labels, seeds, banners, pass shares and regeneration")
    common(k)
    k.add_argument("--sample", type=int, default=20)
    s = sub.add_parser("seal", help="write data/simset/test/SEALED.md")
    s.add_argument("--out", default=str(spec.DATA))
    s.add_argument("--again", action="store_true")
    h = sub.add_parser("sheet", help="a contact sheet of about 60 scenes of one split (never test)")
    h.add_argument("--out", default=str(spec.DATA))
    h.add_argument("--split", default="dev")
    h.add_argument("--n", type=int, default=60)
    args = p.parse_args(argv)
    {"calibrate": cmd_calibrate, "generate": cmd_generate, "check": cmd_check, "seal": cmd_seal,
     "sheet": cmd_sheet}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
