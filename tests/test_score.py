import math

import pytest

from labjudge.judge_mlx import parse
from labjudge.score import exact_range, metrics, prop, trivial


@pytest.mark.parametrize(
    "k,n,lo,hi",
    [(0, 5, 0, 52), (5, 5, 48, 100), (10, 20, 27, 73), (20, 20, 83, 100)],
)
def test_exact_range_known_values(k, n, lo, hi):
    a, b = exact_range(k, n)
    assert round(a * 100) == lo and round(b * 100) == hi


def test_exact_range_precise():
    a, b = exact_range(0, 5)
    assert math.isclose(b, 1 - 0.025 ** (1 / 5), abs_tol=1e-9)
    a, b = exact_range(5, 5)
    assert math.isclose(a, 0.025 ** (1 / 5), abs_tol=1e-9)


def test_prop_text():
    assert prop(10, 20) == "10 of 20 (27-73%)"


def rows(spec):
    return [{"truth": t, "verdict": v, "task": "x"} for t, v in spec]


def test_metrics_and_unparsed_counts_wrong():
    r = rows([("pass", "PASS"), ("pass", "unparsed"), ("fail", "FAIL"), ("fail", "PASS"), ("fail", "unparsed")])
    m = metrics(r)
    assert (m["pass_PASS"], m["pass_FAIL"], m["fail_FAIL"], m["fail_PASS"], m["unparsed"]) == (1, 0, 1, 1, 2)
    assert m["pass_recall"] == 0.5 and math.isclose(m["fail_recall"], 1 / 3)
    assert math.isclose(m["false_pass_rate"], 1 / 3)


def test_trivial_judges_are_half():
    r = rows([("pass", "PASS")] * 10 + [("fail", "FAIL")] * 49)
    assert metrics(trivial(r, "PASS"))["balanced_accuracy"] == 0.5
    assert metrics(trivial(r, "FAIL"))["balanced_accuracy"] == 0.5


@pytest.mark.parametrize(
    "reply,verdict",
    [("PASS", "PASS"), ("fail.", "FAIL"), ("<think>maybe PASS</think>\n\nFAIL", "FAIL"),
     ("I think it passed", "unparsed"), ("", "unparsed"), ("FAIL, not PASS", "FAIL")],
)
def test_parse(reply, verdict):
    assert parse(reply) == verdict

