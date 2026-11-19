"""The board lists every saved run, best balanced accuracy first, with the control apart."""

import json

from labjudge.score import board


def write(path, model, thinking, condition, pairs):
    header = {"kind": "header", "model_id": model, "thinking": thinking, "condition": condition}
    rows = [{"take_id": str(i), "truth": t, "verdict": v, "seconds": 1.0} for i, (t, v) in enumerate(pairs)]
    path.write_text("\n".join(json.dumps(x) for x in [header, *rows]) + "\n")


def test_board_orders_runs_and_keeps_the_control_apart(tmp_path):
    good = [("pass", "PASS"), ("fail", "FAIL"), ("fail", "FAIL"), ("pass", "PASS")]
    yes_man = [("pass", "PASS"), ("fail", "PASS"), ("fail", "PASS"), ("pass", "PASS")]
    write(tmp_path / "verdicts_a_after.jsonl", "org/yes-man", "off", "after", yes_man)
    write(tmp_path / "verdicts_b_think_after.jsonl", "org/good", "on", "after", good)
    write(tmp_path / "verdicts_b_think_leak_control.jsonl", "org/good", "on", "leak_control", good)
    text = board(tmp_path)
    results, control = text.split("## Control")
    results = results.split("## Results")[1]
    assert results.index("org/good") < results.index("org/yes-man")
    assert "leak_control" not in results
    assert "leak_control" in control
    assert "| 1.00 |" in results and "| 0.50 |" in results


def test_unparsed_replies_count_as_wrong(tmp_path):
    garbled = [("pass", "unparsed"), ("fail", "unparsed"), ("fail", "FAIL"), ("pass", "PASS")]
    write(tmp_path / "verdicts_c_after.jsonl", "org/garbled", "off", "after", garbled)
    line = next(x for x in board(tmp_path).splitlines() if "org/garbled" in x)
    assert "| 0.50 | 2 |" in line
