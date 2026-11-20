"""Fill the result tables in README.md from the saved verdict files, so the numbers cannot drift.

  python tools/update_readme.py           rewrite the tables
  python tools/update_readme.py --check   exit 1 if README.md does not match the saved verdicts

A table sits between 2 marker lines: <!-- BEGIN name --> and <!-- END name -->.
"""

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labjudge.score import metrics, prop, read_verdicts, trivial  # noqa: E402

DEV = ROOT / "results" / "simset_dev"
BEST = "LiquidAI__LFM3-VL-2B-MLX-8bit_think"  # the run the by-chore table shows
FRAMES = {"after": "last", "before_after": "first + last"}


def judge_name(header: dict) -> str:
    think = ", thinking on" if header.get("thinking") == "on" else ""
    return f"`{header.get('model_id')}`{think}"


def board() -> str:
    """2 tables: the zero-shot judges, then the probes fitted on the train split."""
    runs = [read_verdicts(p) for p in sorted(DEV.glob("verdicts_*.jsonl"))]
    lines = {"zero-shot": [], "probe": []}
    for header, rows in runs:
        m = metrics(rows)
        seconds = [r["seconds"] for r in rows if r.get("seconds") is not None]
        kind = "probe" if header.get("trained_on") else "zero-shot"
        lines[kind].append((-m["balanced_accuracy"], judge_name(header), (
            f"| {judge_name(header)} | {FRAMES.get(header.get('condition'), header.get('condition'))} "
            f"| {prop(m['pass_PASS'], m['n_pass'])} | {prop(m['fail_PASS'], m['n_fail'])} "
            f"| {m['balanced_accuracy']:.2f} | {_seconds(statistics.median(seconds))} |")))
    any_rows = runs[0][1]
    columns = "Frames | Says PASS on true passes | False passes | Balanced accuracy | Seconds per scene |"
    out = [f"| Zero-shot judge | {columns}", "|---|---|---|---|---|---|"]
    out += [line for _, _, line in sorted(lines["zero-shot"])]
    for name, verdict in (("always PASS", "PASS"), ("always FAIL", "FAIL")):
        m = metrics(trivial(any_rows, verdict))
        out.append(f"| {name} (reference) | - | {prop(m['pass_PASS'], m['n_pass'])} "
                   f"| {prop(m['fail_PASS'], m['n_fail'])} | {m['balanced_accuracy']:.2f} | - |")
    if lines["probe"]:
        out += ["", "Linear probes, fitted on the train scenes of the same simulator. A test of the pictures, not "
                "zero-shot judges:", "", f"| Probe | {columns}", "|---|---|---|---|---|---|"]
        out += [line for _, _, line in sorted(lines["probe"])]
    return "\n".join(out)


TRAINED = "frozen__fine-logreg"  # the linear probe the by-chore table shows


def _seconds(x: float) -> str:
    return f"{x:.1f}" if x >= 1 else f"{x:.2f}"


def by_chore() -> str:
    _, last = read_verdicts(DEV / f"verdicts_{BEST}_after.jsonl")
    _, both = read_verdicts(DEV / f"verdicts_{BEST}_before_after.jsonl")
    _, head = read_verdicts(DEV / f"verdicts_{TRAINED}_after.jsonl")
    out = ["| Chore | True passes | True fails | Zero-shot, last frame | Zero-shot, first + last frame "
           "| Linear probe, last frame |", "|---|---|---|---|---|---|"]

    def score(rows: list[dict], task: str) -> float:
        return metrics([r for r in rows if r["task"] == task])["balanced_accuracy"]

    def best_first(task: str) -> tuple[float, str]:  # ties go by name, so the order is the same on every run
        return (-score(last, task), task)

    for task in sorted({r["task"] for r in last}, key=best_first):
        a = metrics([r for r in last if r["task"] == task])
        out.append(f"| `{task}` | {a['n_pass']} | {a['n_fail']} | {score(last, task):.2f} | {score(both, task):.2f} "
                   f"| {score(head, task):.2f} |")
    return "\n".join(out)


APOS = "['" + chr(8217) + "]"  # a plain or a curly apostrophe, spelled so that this file stays ASCII

## What the best judge's reasoning says on clean scenes. Patterns are matched on the lower-cased thinking text.
KEYWORDS = (
    ("stack_plates", "a single plate", r"only one|single plate|one plate|a single|just one"),
    ("plate_to_tray", "no tray, or the plate is not on a tray",
     r"no (black )?tray|not on (any|a|the) (black )?tray|(don" + APOS + r"t|do not|cannot|can" + APOS
     + r"t) see a (black )?tray"),
    ("unstack_plate", "", ""),
    ("plate_to_hotel_top", "", ""),
    ("plate_to_hotel_bottom", "", ""),
    ("stack_tip_boxes", "", ""),
)


def keywords() -> str:
    import json
    import re

    manifest = ROOT / "data" / "simset" / "dev" / "manifest.jsonl"
    clean = {json.loads(x)["take_id"] for x in manifest.read_text().splitlines()
             if x.strip() and json.loads(x)["kind"] == "clean"}
    _, rows = read_verdicts(DEV / f"verdicts_{BEST}_after.jsonl")
    out = ["| Chore | Clean scenes | Judged FAIL | Its reasoning speaks of | In |", "|---|---|---|---|---|"]
    for task, words, pattern in KEYWORDS:
        mine = [r for r in rows if r["task"] == task and r["take_id"] in clean]
        failed = sum(r["verdict"] == "FAIL" for r in mine)
        hits = sum(bool(re.search(pattern, r["reply"].split("</think>")[0].lower())) for r in mine) if pattern else None
        out.append(f"| `{task}` | {len(mine)} | {failed} of {len(mine)} | {words or '-'} "
                   f"| {f'{hits} of {len(mine)}' if pattern else '-'} |")
    return "\n".join(out)


def keywords_file() -> str:
    lines = ["# What the best judge says on clean plate scenes", "",
             "Run: `LiquidAI/LFM3-VL-2B-MLX-8bit`, thinking on, last frame, dev split. A clean scene is one where",
             "every prop is where the chore's oracle left it, so it passes. The patterns are matched on the",
             "lower-cased thinking text of each reply. Made by `tools/update_readme.py`.", "", keywords(), "",
             "Patterns:", ""]
    lines += [f"- `{task}`: `{pattern.encode('ascii', 'backslashreplace').decode()}`" for task, _, pattern in KEYWORDS
              if pattern]
    return "\n".join(lines) + "\n"


def fragile() -> str:
    """The table of results/frozen/perturbed.md, as written by tools/probe_perturbed.py (it needs the model)."""
    text = (ROOT / "results" / "frozen" / "perturbed.md").read_text()
    return "\n".join(x for x in text.splitlines() if x.startswith("|"))


TABLES = {"BOARD": board, "BY CHORE": by_chore, "KEYWORDS": keywords, "FRAGILE": fragile}


def fill(text: str) -> str:
    for name, make in TABLES.items():
        begin, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
        if begin not in text or end not in text:
            raise SystemExit(f"README.md has no {begin} ... {end} block")
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{make()}\n{end}{tail}"
    return text


def main() -> None:
    readme = ROOT / "README.md"
    notes = DEV / "reasoning_keywords.md"
    old = readme.read_text()
    new = fill(old)
    if "--check" in sys.argv:
        if new != old or not notes.exists() or notes.read_text() != keywords_file():
            raise SystemExit("README.md tables do not match the saved verdicts; run tools/update_readme.py")
        print("README.md tables match the saved verdicts")
        return
    readme.write_text(new)
    notes.write_text(keywords_file())
    print("README.md tables rewritten" if new != old else "README.md tables already up to date")


if __name__ == "__main__":
    main()
