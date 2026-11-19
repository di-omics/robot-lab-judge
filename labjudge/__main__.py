"""CLI: extract frames, judge them, score one run, list every run.

  python -m labjudge extract --bench PATH      (or set LAB_HANDLING_BENCH)
  python -m labjudge judge --model mlx-community/Qwen3.5-4B-4bit
  python -m labjudge table --model mlx-community/Qwen3.5-4B-4bit
  python -m labjudge board

A run is one model with thinking off or on. Add --thinking to judge and table for a thinking run.
"""

import argparse
import datetime
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

DATA = Path("data/first_look")
RESULTS = Path("results/first_look")
DEFAULT_MODEL = "mlx-community/Qwen3.5-4B-4bit"
CONDITIONS = ("after", "before_after", "leak_control")


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def run_name(model_id: str, thinking: bool) -> str:
    """File stem of one run: the model, plus a mark when thinking was on."""
    return slug(model_id) + ("_think" if thinking else "")


def read_manifest(data: Path) -> list[dict]:
    path = data / "manifest.jsonl"
    if not path.exists():
        raise SystemExit(f"no {path}; run extract first")
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def cmd_extract(args) -> None:
    from . import frames, takes

    root = takes.bench_root(args.bench)
    usable, excluded = takes.load(root)
    data = Path(args.data)
    (data / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, t in enumerate(usable, 1):
        first, last, count = frames.first_and_last(t.video)
        stem = data / "frames" / t.take_id
        paths = {
            "first_views": f"{stem}_first_views.png",
            "last_views": f"{stem}_last_views.png",
            "last_top": f"{stem}_last_top.png",
            "last_uncropped": f"{stem}_last_uncropped.png",
        }
        frames.views(first).save(paths["first_views"])
        frames.views(last).save(paths["last_views"])
        frames.top_view(last).save(paths["last_top"])
        frames.uncropped(last).save(paths["last_uncropped"])
        rows.append({
            "take_id": t.take_id, "run": t.run, "task": t.task, "title": t.title, "prompt": t.prompt,
            "seed": t.seed, "policy": t.policy, "model": t.model, "truth": t.truth, "frames": count,
            "images": paths,
        })
        print(f"[{i}/{len(usable)}] {t.run}/{t.take_id} {t.truth} ({count} frames)", flush=True)
    (data / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    n_pass = sum(r["truth"] == "pass" for r in rows)
    print(f"usable {len(rows)} (pass {n_pass}, fail {len(rows) - n_pass}), excluded {len(excluded)}")
    for name, why in excluded:
        print(f"  excluded {name}: {why}")


def images_for(row: dict, condition: str) -> list[str]:
    im = row["images"]
    if condition == "after":
        return [im["last_views"]]
    if condition == "before_after":
        return [im["first_views"], im["last_views"]]
    if condition == "leak_control":
        return [im["last_uncropped"]]
    raise ValueError(condition)


def cmd_judge(args) -> None:
    from .judge_mlx import Judge, snapshot_of
    from .prompts import PROMPT_VERSION
    from .sealed import is_sealed, log_look

    rows = read_manifest(Path(args.data))
    if args.limit:
        rows = rows[: args.limit]
    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    judge = Judge(args.model, max_tokens=args.max_tokens, thinking=args.thinking)
    print(f"loaded {args.model} in {time.perf_counter() - t0:.1f} s", flush=True)
    header = {
        "kind": "header", "model_id": args.model, "snapshot": snapshot_of(args.model),
        "mlx_vlm": version("mlx-vlm"), "mlx": version("mlx"), "python": platform.python_version(),
        "prompt_version": PROMPT_VERSION, "thinking": "on" if args.thinking else "off",
        "temperature": 0.0, "max_tokens": args.max_tokens,
        "date": datetime.date.today().isoformat(), "data": str(args.data),
    }
    sealed = is_sealed(args.data)
    for condition in args.conditions:
        if sealed:  # a look at a sealed split is logged before it starts
            log_look(args.sealed_log, args.data, args.model, args.thinking, condition, len(rows))
        path = out_dir / f"verdicts_{run_name(args.model, args.thinking)}_{condition}.jsonl"
        partial = path.with_name(path.name + ".partial")  # renamed when every row is in: a stopped run is never scored
        with partial.open("w") as f:
            f.write(json.dumps({**header, "condition": condition}) + "\n")
            for i, row in enumerate(rows, 1):
                answer = judge.ask(row["prompt"], images_for(row, condition))
                f.write(json.dumps({
                    "take_id": row["take_id"], "run": row["run"], "task": row["task"], "seed": row["seed"],
                    "truth": row["truth"], "condition": condition, "images": images_for(row, condition),
                    "verdict": answer["verdict"], "reply": answer["reply"], "seconds": answer["seconds"],
                }) + "\n")
                f.flush()
                print(f"{condition} [{i}/{len(rows)}] {row['task']} s{row['seed']} truth={row['truth']} "
                      f"verdict={answer['verdict']} {answer['seconds']} s reply={answer['reply']!r}", flush=True)
        partial.replace(path)
        print(f"wrote {path}")


def cmd_table(args) -> None:
    from .score import read_verdicts, table
    from .sheet import contact_sheet

    out_dir = Path(args.results)
    name = run_name(args.model, args.thinking)
    text = table(out_dir, name)
    (out_dir / f"table_{name}.md").write_text(text)
    print(text)
    after = out_dir / f"verdicts_{name}_after.jsonl"
    if after.exists():
        _, verdicts = read_verdicts(after)
        manifest = {r["take_id"]: r for r in read_manifest(Path(args.data))}
        sheet = out_dir / f"sheet_{name}.jpg"
        contact_sheet(verdicts, manifest, sheet)
        print(f"wrote {sheet}")


def cmd_board(args) -> None:
    from .score import board

    out_dir = Path(args.results)
    text = board(out_dir)
    (out_dir / "board.md").write_text(text)
    print(text)


def cmd_breakdown(args) -> None:
    from .score import breakdown

    out_dir = Path(args.results)
    text = breakdown(out_dir, read_manifest(Path(args.data)))
    (out_dir / "breakdown.md").write_text(text)
    print(text)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="labjudge")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="decode takes, crop the banner, save frames and manifest")
    e.add_argument("--bench", help="folder of the lab handling simulation takes (else env LAB_HANDLING_BENCH)")
    e.add_argument("--data", default=str(DATA))
    j = sub.add_parser("judge", help="ask the model PASS or FAIL for every take and condition")
    j.add_argument("--model", default=DEFAULT_MODEL)
    j.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    j.add_argument("--max-tokens", type=int, default=16)
    j.add_argument("--thinking", action="store_true", help="turn the chat template's thinking switch on")
    j.add_argument("--limit", type=int, default=0, help="only the first N takes (smoke test)")
    j.add_argument("--data", default=str(DATA))
    j.add_argument("--results", default=str(RESULTS))
    j.add_argument("--sealed-log", default="results/sealed_log.jsonl",
                   help="where a look at a sealed split (its folder holds SEALED.md) is logged")
    t = sub.add_parser("table", help="score one run's saved verdicts into a table and a contact sheet")
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--thinking", action="store_true", help="score the run made with thinking on")
    t.add_argument("--data", default=str(DATA))
    t.add_argument("--results", default=str(RESULTS))
    b = sub.add_parser("board", help="every saved run in one table, best balanced accuracy first")
    b.add_argument("--results", default=str(RESULTS))
    k = sub.add_parser("breakdown", help="scene set only: judged PASS by kind of end state, distance and task")
    k.add_argument("--data", required=True, help="a scene set split, for example data/simset/dev")
    k.add_argument("--results", required=True)
    args = p.parse_args(argv)
    {"extract": cmd_extract, "judge": cmd_judge, "table": cmd_table, "board": cmd_board,
     "breakdown": cmd_breakdown}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
