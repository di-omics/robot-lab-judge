"""The frozen-features judge: the head, the folds, the scaler, the split guard and the verdict file shape.

No test needs the model or the cached features.
"""

import json

import numpy as np
import pytest

from labjudge.frozen import check_path, check_split
from labjudge.frozen import head as H
from labjudge.frozen.__main__ import main
from labjudge.frozen.extract import change_feature, mean_max
from labjudge.frozen.fit import pick, write_verdicts
from labjudge.score import board, breakdown


def test_logistic_regression_separates_an_easy_set():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-2, 1, (60, 5)), rng.normal(2, 1, (40, 5))])
    y = np.r_[np.zeros(60), np.ones(40)].astype(int)
    sc = H.Scaler(x)
    w, b = H.fit_logreg(sc(x), y, l2=1.0)
    p = H.predict_proba(sc(x), w, b)
    assert H.balanced_accuracy(y, p >= 0.5) > 0.97
    assert H.auroc(y, p) > 0.99


def test_logistic_regression_with_more_columns_than_rows_matches_the_full_solve():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 50))
    y = (x[:, 0] + 0.3 * rng.normal(size=20) > 0).astype(int)
    w, b = H.fit_logreg(x, y, l2=3.0)  # solved in the span of the rows
    s = H.balanced_weights(y)
    w2, b2 = H._newton(x, y.astype(float), s, 3.0)  # solved in all 50 columns
    assert np.allclose(w, w2, atol=1e-5) and abs(b - b2) < 1e-5


def test_balanced_weights_give_each_class_half():
    y = np.array([1, 0, 0, 0])
    s = H.balanced_weights(y)
    assert s[y == 1].sum() == pytest.approx(s[y == 0].sum())


def test_grouped_folds_never_share_a_seed_and_are_fixed():
    seeds = np.repeat(np.arange(10000, 10030), 6)
    folds = H.grouped_folds(seeds, 5, seed=0)
    assert set(folds.tolist()) == set(range(5))
    for f in range(5):
        assert not set(seeds[folds == f]) & set(seeds[folds != f])
    assert np.array_equal(folds, H.grouped_folds(seeds, 5, seed=0))


def test_the_scaler_sees_the_train_folds_only(monkeypatch):
    seen = []

    class Spy(H.Scaler):
        def __init__(self, x):
            seen.append(np.asarray(x).copy())
            super().__init__(x)

    monkeypatch.setattr(H, "Scaler", Spy)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 3))
    x[:, 0] = np.arange(30)  # a row id in column 0
    y = np.tile([0, 1], 15)
    folds = H.grouped_folds(np.repeat(np.arange(10), 3), 5, seed=0)
    H.cv_oof(x, y, folds, 1.0)
    assert len(seen) == 5
    for f, fitted in zip(sorted(set(folds.tolist())), seen):
        held_out = set(np.flatnonzero(folds == f).tolist())
        assert not held_out & set(fitted[:, 0].astype(int).tolist())
        assert len(fitted) == int((folds != f).sum())


def test_a_tie_in_cross_validation_goes_to_the_larger_l2():
    cv = {0.01: {"mean": 0.7}, 0.1: {"mean": 0.8}, 1.0: {"mean": 0.8}, 10.0: {"mean": 0.6}}
    assert pick(cv) == 1.0


def test_auroc_counts_ties_as_half():
    assert H.auroc([0, 1], [0.5, 0.5]) == 0.5
    assert H.auroc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0


def test_pooled_features_have_the_designed_sizes():
    rng = np.random.default_rng(3)
    last = rng.normal(size=(8, 1024, 768)).astype(np.float32)
    first = last.copy()
    first[2, 5] += 10  # 1 changed patch
    assert mean_max(last).shape == (1536,)
    feat, idx = change_feature(last, first)
    assert feat.shape == (2304,) and len(idx) == 32
    assert idx[0] == 2 * 1024 + 5


@pytest.mark.parametrize("argv", [
    ["extract", "--split", "test"],
    ["extract", "--split", "holdout"],
    ["extract", "--split", "dev", "--data", "data/simset/test"],
    ["extract", "--split", "train", "--data", "data/simset/test/frames"],
])
def test_extract_refuses_the_test_split(argv):
    with pytest.raises(SystemExit):
        main(argv)


def test_the_guards_refuse_test_paths():
    with pytest.raises(SystemExit):
        check_split("test")
    for p in ("data/simset/test", "data/simset/test/manifest.jsonl", "./data/simset/test/frames/x.png"):
        with pytest.raises(SystemExit):
            check_path(p)
    assert str(check_path("data/simset/dev/manifest.jsonl")) == "data/simset/dev/manifest.jsonl"


def test_a_verdict_file_is_read_by_board_and_breakdown(tmp_path):
    manifest = [
        {"take_id": f"dev-plate_to_tray-s20000-0{i}-{k}", "run": "dev", "task": "plate_to_tray", "seed": 20000,
         "truth": t, "kind": k, "images": {"first_views": "a.png", "last_views": "b.png", "last_top": "c.png"},
         "error": {"planar_mm": 3.0, "height_mm": 0.0}, "commanded": {}}
        for i, (k, t) in enumerate([("clean", "pass"), ("offset", "pass"), ("offset", "fail"), ("trap", "fail")])
    ]
    header = {"kind": "header", "model_id": "frozen/siglip2-lfm2.5-vl-450m/fine-logreg", "thinking": "-",
              "condition": "after", "data": "data/simset/dev", "date": "2026-01-01"}
    p = np.array([0.9, 0.3, 0.2, 0.6])
    path = write_verdicts(tmp_path, "fine-logreg", header, manifest, p, ("last_top",), 0.5)
    assert path.name == "verdicts_frozen__fine-logreg_after.jsonl"
    assert not list(tmp_path.glob("*.partial"))
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert lines[0]["kind"] == "header" and [x["verdict"] for x in lines[1:]] == ["PASS", "FAIL", "FAIL", "PASS"]
    text = board(tmp_path)
    row = next(x for x in text.splitlines() if "frozen/siglip2-lfm2.5-vl-450m/fine-logreg" in x)
    assert "| - | after | 4 | 1 of 2" in row and "| 0.50 |" in row
    text = breakdown(tmp_path, manifest)
    assert "fine-logreg, after" in text
    assert "| plate_to_tray | 4 | 0.50 |" in text
