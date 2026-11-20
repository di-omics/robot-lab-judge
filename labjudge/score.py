"""Score verdicts against simulator truth, with exact (Clopper-Pearson) 95% ranges."""

import json
import statistics
from math import comb
from pathlib import Path


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _bisect(f, lo=0.0, hi=1.0, iters=100) -> float:
    """Root of an increasing function f on [lo, hi]."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def exact_range(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson range for k successes in n trials, from the binomial distribution."""
    if n <= 0:
        return (0.0, 1.0)
    a = (1 - level) / 2
    # lower: p where P(X >= k | p) = a; P(X >= k) rises with p
    lower = 0.0 if k == 0 else _bisect(lambda p: (1 - _binom_cdf(k - 1, n, p)) - a)
    # upper: p where P(X <= k | p) = a; P(X <= k) falls with p
    upper = 1.0 if k == n else _bisect(lambda p: a - _binom_cdf(k, n, p))
    return (lower, upper)


def prop(k: int, n: int) -> str:
    """'k of n (lo-hi%)'."""
    if n == 0:
        return "0 of 0"
    lo, hi = exact_range(k, n)
    return f"{k} of {n} ({lo * 100:.0f}-{hi * 100:.0f}%)"


def confusion(rows: list[dict]) -> dict:
    """Counts from rows with 'truth' (pass/fail) and 'verdict' (PASS/FAIL/unparsed). Unparsed counts as wrong."""
    c = {"pass_PASS": 0, "pass_FAIL": 0, "fail_FAIL": 0, "fail_PASS": 0, "pass_unparsed": 0, "fail_unparsed": 0}
    for r in rows:
        v = r["verdict"] if r["verdict"] in ("PASS", "FAIL") else "unparsed"
        c[f"{r['truth']}_{v}"] += 1
    c["n_pass"] = c["pass_PASS"] + c["pass_FAIL"] + c["pass_unparsed"]
    c["n_fail"] = c["fail_FAIL"] + c["fail_PASS"] + c["fail_unparsed"]
    c["unparsed"] = c["pass_unparsed"] + c["fail_unparsed"]
    return c


def metrics(rows: list[dict]) -> dict:
    c = confusion(rows)
    pass_recall = c["pass_PASS"] / c["n_pass"] if c["n_pass"] else float("nan")
    fail_recall = c["fail_FAIL"] / c["n_fail"] if c["n_fail"] else float("nan")
    return {
        **c,
        "pass_recall": pass_recall,
        "fail_recall": fail_recall,
        "balanced_accuracy": (pass_recall + fail_recall) / 2,
        "false_pass_rate": c["fail_PASS"] / c["n_fail"] if c["n_fail"] else float("nan"),
    }


def trivial(rows: list[dict], verdict: str) -> list[dict]:
    return [{**r, "verdict": verdict} for r in rows]


def read_verdicts(path: Path) -> tuple[dict, list[dict]]:
    """A verdicts file: first line is the header (kind=header), the rest are one row per take."""
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    header = lines[0] if lines and lines[0].get("kind") == "header" else {}
    rows = [x for x in lines if x.get("kind") != "header"]
    return header, rows


HEAD = (
    "| condition | n | true pass judged PASS | true pass judged FAIL | true fail judged FAIL | true fail judged PASS "
    "| unparsed | pass recall | fail recall | balanced accuracy | false-pass rate | median s per take |"
)
RULE = "|" + "---|" * 12


def _row(name: str, rows: list[dict], seconds: list[float] | None) -> str:
    m = metrics(rows)
    med = f"{statistics.median(seconds):.1f}" if seconds else "-"
    return (
        f"| {name} | {len(rows)} | {m['pass_PASS']} | {m['pass_FAIL']} | {m['fail_FAIL']} | {m['fail_PASS']} "
        f"| {m['unparsed']} | {prop(m['pass_PASS'], m['n_pass'])} | {prop(m['fail_FAIL'], m['n_fail'])} "
        f"| {m['balanced_accuracy']:.2f} | {prop(m['fail_PASS'], m['n_fail'])} | {med} |"
    )


def table(results_dir: Path, model_slug: str) -> str:
    """table.md text from the saved verdicts files of one model."""
    files = {c: results_dir / f"verdicts_{model_slug}_{c}.jsonl" for c in ("after", "before_after", "leak_control")}
    loaded = {c: read_verdicts(p) for c, p in files.items() if p.exists()}
    if not loaded:
        raise SystemExit(f"no verdicts files for {model_slug} in {results_dir}")
    header = next(iter(loaded.values()))[0]
    out = ["# First look: zero-shot vision judge", ""]
    out += [
        f"Model: `{header.get('model_id')}` (snapshot `{header.get('snapshot')}`), mlx-vlm {header.get('mlx_vlm')}, "
        f"prompt {header.get('prompt_version')}, thinking {header.get('thinking')}, "
        f"temperature {header.get('temperature')}, run date {header.get('date')}.",
        "",
        "Truth is the simulator's frozen pass rule. Unparsed replies count as wrong answers. "
        "Ranges are exact 95% (Clopper-Pearson).",
        "",
        "## Results",
        "",
        HEAD,
        RULE,
    ]
    for c in ("after", "before_after"):
        if c in loaded:
            h, rows = loaded[c]
            out.append(_row(c, rows, [r["seconds"] for r in rows if r.get("seconds") is not None]))
    any_rows = next(iter(loaded.values()))[1]
    out.append(_row("always PASS (reference)", trivial(any_rows, "PASS"), None))
    out.append(_row("always FAIL (reference)", trivial(any_rows, "FAIL"), None))
    if "after" in loaded:
        rows = loaded["after"][1]
        out += ["", "## Per task, condition `after`", "",
                "| task | n | true pass | judged PASS on true pass | true fail | judged FAIL on true fail | unparsed |",
                "|---|---|---|---|---|---|---|"]
        for task in sorted({r["task"] for r in rows}):
            m = metrics([r for r in rows if r["task"] == task])
            out.append(f"| {task} | {m['n_pass'] + m['n_fail']} | {m['n_pass']} | {m['pass_PASS']} | {m['n_fail']} "
                       f"| {m['fail_FAIL']} | {m['unparsed']} |")
    if "leak_control" in loaded:
        rows = loaded["leak_control"][1]
        out += ["", "## Control (not a result): `leak_control`", "",
                "The judge saw the uncropped last frame, with the banner that prints the stage count. "
                "This only tests whether the model reads on-screen text. "
                "On 3 takes from a different recorder the banner shows only the task and the clock.",
                "", HEAD, RULE,
                _row("leak_control", rows, [r["seconds"] for r in rows if r.get("seconds") is not None])]
    out += _notes(metrics(any_rows)["n_pass"])
    return "\n".join(out)


def _notes(n_pass: int) -> list[str]:
    return ["", "## Notes", "",
            f"- n is small: {n_pass} true passes. One take more or less moves pass recall by "
            f"{100 / max(n_pass, 1):.0f} points.",
            "- These takes were already seen during development of the simulation. "
            "This is a first look and a pipeline check, not a sealed test.",
            "- Simulation only. Nothing here says how the judge does on real camera frames.",
            ""]


BOARD_HEAD = (
    "| model | thinking | condition | n | pass recall | fail recall | false-pass rate "
    "| balanced accuracy | unparsed | median s per take |"
)
BOARD_RULE = "|" + "---|" * 10


def _board_row(model: str, thinking: str, condition: str, rows: list[dict], timed: bool = True) -> tuple[float, str]:
    """(balanced accuracy, table line) for one run and condition."""
    m = metrics(rows)
    seconds = [r["seconds"] for r in rows if r.get("seconds") is not None] if timed else []
    med = f"{statistics.median(seconds):.1f}" if seconds else "-"
    line = (
        f"| {model} | {thinking} | {condition} | {len(rows)} | {prop(m['pass_PASS'], m['n_pass'])} "
        f"| {prop(m['fail_FAIL'], m['n_fail'])} | {prop(m['fail_PASS'], m['n_fail'])} "
        f"| {m['balanced_accuracy']:.2f} | {m['unparsed']} | {med} |"
    )
    return m["balanced_accuracy"], line


def board(results_dir: Path) -> str:
    """board.md text: every saved run in one table, best balanced accuracy first, controls apart."""
    runs = [read_verdicts(p) for p in sorted(results_dir.glob("verdicts_*.jsonl"))]
    runs = [(h, rows) for h, rows in runs if h and rows]
    if not runs:
        raise SystemExit(f"no verdicts files in {results_dir}")

    def kind_of(h: dict) -> str:
        """A run is a control, a probe fitted on the train split, or a zero-shot judge."""
        if h.get("condition") == "leak_control":
            return "control"
        return "fitted" if h.get("trained_on") else "zero-shot"

    def lines(kind: str) -> list[str]:
        picked = [
            _board_row(f"`{h.get('model_id')}`", str(h.get("thinking")), str(h.get("condition")), rows)
            for h, rows in runs
            if kind_of(h) == kind
        ]
        return [line for _, line in sorted(picked, key=lambda x: -x[0])]

    any_rows = runs[0][1]
    data = Path(_data_of(runs[0][0]))
    scenes = "simset" in data.parts
    out = [
        f"# Board: vision judges on {'the scene set, ' + data.name + ' split' if scenes else 'the first-look takes'}",
        "",
        "Every saved run, best balanced accuracy first. Truth is the simulator's frozen pass rule. "
        "Unparsed replies count as wrong answers. Ranges are exact 95% (Clopper-Pearson).",
        "",
        "## Results",
        "",
        BOARD_HEAD,
        BOARD_RULE,
        *lines("zero-shot"),
        _board_row("always PASS (reference)", "-", "-", trivial(any_rows, "PASS"), timed=False)[1],
        _board_row("always FAIL (reference)", "-", "-", trivial(any_rows, "FAIL"), timed=False)[1],
    ]
    fitted = lines("fitted")
    if fitted:
        out += ["", "## Probes fitted on the train split (not zero-shot)", "",
                "A probe is a small readout fitted on scenes from the same simulator, camera and lighting. "
                "It tests whether the pictures hold the answer. It is not a judge for new settings.",
                "", BOARD_HEAD, BOARD_RULE, *fitted]
    controls = lines("control")
    if controls:
        out += ["", "## Control (not a result): the banner left in", "",
                "The judge saw the uncropped last frame, with the banner that prints the stage count. "
                "This only tests whether a model reads on-screen text.",
                "", BOARD_HEAD, BOARD_RULE, *controls]
    out += _scene_notes(data.name) if scenes else _notes(metrics(any_rows)["n_pass"])
    return "\n".join(out)


def _data_of(header: dict) -> str:
    """The data folder a run judged. Runs saved before the header named it are first-look runs."""
    return str(header.get("data") or "data/first_look")


def _scene_notes(split: str) -> list[str]:
    use = ("It is sealed. Every scoring run on it is in `results/sealed_log.jsonl`." if split == "test" else
           "It is for choosing prompts and thresholds, so it is not a final number. "
           "Final numbers come from the sealed test split only.")
    return ["", "## Notes", "",
            "- These are simulator scenes, not robot attempts. Both arms rest at home in every scene. "
            "A real attempt ends with the arms anywhere.",
            "- Truth is the frozen pass rule on the settled end state. Some rules are loose "
            "(a bottle anywhere on the platform passes), so a true pass can look untidy.",
            f"- This is the {split} split. {use}",
            "- Simulation only. Nothing here says how the judge does on real camera frames.",
            ""]


## Kinds of end state in reading order: the true passes first, then the true fails.
KIND_ROWS = (("clean", "pass"), ("offset", "pass"), ("tilt", "pass"), ("offset", "fail"), ("tilt", "fail"),
             ("partial", "fail"), ("untouched", "fail"), ("trap", "fail"))
## How far the moved prop ended from where the oracle left it, mm, in 3 dimensions.
ERROR_BINS = ((0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, float("inf")))


def _run_label(header: dict) -> str:
    think = ", thinking" if header.get("thinking") == "on" else ""
    return f"{str(header.get('model_id')).split('/')[-1]}{think}, {header.get('condition')}"


def _said_pass(rows: list[dict]) -> str:
    return f"{sum(r['verdict'] == 'PASS' for r in rows)} of {len(rows)}" if rows else "-"


def _far_mm(scene: dict) -> float:
    e = scene["error"]
    return (e["planar_mm"] ** 2 + e["height_mm"] ** 2) ** 0.5


def breakdown(results_dir: Path, manifest: list[dict]) -> str:
    """Scene-set tables: how often each run said PASS, by kind of end state, by distance and by task."""
    scene = {r["take_id"]: r for r in manifest}
    runs = [read_verdicts(p) for p in sorted(results_dir.glob("verdicts_*.jsonl"))]
    runs = [(h, [r for r in rows if r["take_id"] in scene]) for h, rows in runs if h and rows]
    if not runs:
        raise SystemExit(f"no verdicts files in {results_dir}")
    labels = [_run_label(h) for h, _ in runs]

    def grid(first_cols: list[str], groups: list[tuple[list[str], object]], cell=_said_pass) -> list[str]:
        """One line per group: its name cells, its n, then one cell per run over the scenes the group keeps."""
        lines = ["| " + " | ".join(first_cols + ["n"] + labels) + " |", "|" + "---|" * (len(first_cols) + 1 + len(labels))]
        for names, keep in groups:
            n = sum(1 for s in scene.values() if keep(s))
            if n:
                cells = [cell([r for r in rows if keep(scene[r["take_id"]])]) for _, rows in runs]
                lines.append("| " + " | ".join(names + [str(n)] + cells) + " |")
        return lines

    out = ["# Scene set: where each judge says PASS", "",
           "Each cell is the number of scenes the run judged PASS, of the scenes it saw. "
           "A good judge says PASS on every true pass and on no true fail. An unparsed reply is not a PASS.",
           "", "## By kind of end state", ""]
    out += grid(["kind", "truth"], [([k, t], lambda s, k=k, t=t: s["kind"] == k and s["truth"] == t)
                                    for k, t in KIND_ROWS])
    out += ["", "## Offset scenes with no turn, by how far the moved prop ended from its place", "",
            "Distance in 3 dimensions from where the task's oracle left the prop, after settling. "
            "Offset scenes that also turned the prop are left out here, so distance is the only change. "
            "Chores differ in how far a prop can move and still pass, and a judge can call a whole chore PASS or "
            "FAIL, so read this table next to the by-task table.", ""]

    def shifted(s: dict) -> bool:
        return s["kind"] == "offset" and not (s.get("commanded") or {}).get("yaw_deg")

    bins = [([f"{lo} mm and over" if hi == float("inf") else f"{lo} to {hi} mm", t],
             lambda s, lo=lo, hi=hi, t=t: shifted(s) and s["truth"] == t and lo <= _far_mm(s) < hi)
            for t in ("pass", "fail") for lo, hi in ERROR_BINS]
    out += grid(["distance", "truth"], bins)
    out += ["", "## Balanced accuracy by task", ""]
    tasks = sorted({s["task"] for s in scene.values()})
    out += grid(["task"], [([t], lambda s, t=t: s["task"] == t) for t in tasks],
                cell=lambda rows: f"{metrics(rows)['balanced_accuracy']:.2f}" if rows else "-")
    return "\n".join(out) + "\n"
