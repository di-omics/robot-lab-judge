"""Scene set parts that need no simulator: manifest keys, split seeds, the seal hash and the sealed log."""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from labjudge import __main__ as cli
from labjudge.simset import spec

ROOT = Path(__file__).resolve().parents[1]


def good_row(**change) -> dict:
    row = {
        "take_id": "dev-plate_to_tray-s20000-00-clean", "run": "dev", "task": "plate_to_tray",
        "title": "Load a plate onto the reader tray", "prompt": "Put the plate on the black tray.",
        "seed": 20000, "truth": "pass",
        "images": {"first_views": "a.png", "last_views": "b.png", "last_top": "c.png"},
        "kind": "clean", "prop": "plate", "trap": None, "commanded": {}, "error": {}, "rule": "r",
        "seen_task": True, "plan": {"kind": "clean"},
    }
    row.update(change)
    return row


def test_a_good_row_has_no_problems():
    assert spec.row_problems(good_row()) == []


@pytest.mark.parametrize("change,word", [
    ({"truth": "PASS"}, "truth"),
    ({"kind": "wobble"}, "kind"),
    ({"seed": 10000}, "not in split"),
    ({"seen_task": False}, "seen_task"),
    ({"task": "rack_into_box", "seen_task": False}, "held-out"),
    ({"images": {"first_views": "a.png", "last_views": "b.png"}}, "last_top"),
])
def test_bad_rows_are_caught(change, word):
    assert any(word in p for p in spec.row_problems(good_row(**change)))


def test_missing_judge_keys_are_caught():
    for key in spec.JUDGE_KEYS:
        row = good_row()
        del row[key]
        assert f"missing {key}" in spec.row_problems(row)


def test_split_seeds_are_disjoint_and_avoid_the_bench_seeds():
    assert spec.splits_disjoint() == []
    assert spec.SPLITS["train"] == range(10000, 10030)
    assert spec.SPLITS["dev"] == range(20000, 20008)
    assert spec.SPLITS["test"] == range(30000, 30020)
    assert all(spec.split_of_seed(s) is None for s in spec.BENCH_SEEDS)
    assert set(spec.CALIBRATION_SEEDS) <= set(spec.SPLITS["train"])


def test_held_out_tasks_only_in_test():
    included = ["plate_to_tray", "plate_to_hotel_middle", "small_tubes_to_rack", "rack_into_box"]
    assert spec.tasks_for("train", included) == ["plate_to_tray"]
    assert spec.tasks_for("dev", included) == ["plate_to_tray"]
    assert spec.tasks_for("test", included) == included


def fake_split(tmp_path: Path, n: int = 3) -> Path:
    folder = tmp_path / "data" / "simset" / "test"
    (folder / "frames").mkdir(parents=True)
    rows = []
    for i in range(n):
        images = {}
        for k in spec.IMAGE_KEYS:
            rel = f"data/simset/test/frames/{i}_{k}.png"
            Image.fromarray(np.full((4, 4, 3), i * 10 + len(k), np.uint8)).save(tmp_path / rel)
            images[k] = rel
        rows.append(good_row(take_id=f"test-x-s30000-{i:02d}-clean", run="test", seed=30000, images=images))
    (folder / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return folder


def test_seal_hash_names_the_files_and_changes_when_an_image_changes(tmp_path):
    folder = fake_split(tmp_path)
    h1 = spec.seal_hashes(folder, tmp_path)
    assert h1 == spec.seal_hashes(folder, tmp_path)
    assert h1["n_rows"] == 3 and h1["n_images"] == 9
    text = spec.seal_text(h1, "2026-01-01")
    assert h1["manifest_sha256"] in text and h1["images_sha256"] in text and "train and dev only" in text
    (folder / spec.SEALED).write_text(text)
    assert spec.seal_matches(folder, tmp_path)
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tmp_path / "data/simset/test/frames/1_last_views.png")
    h2 = spec.seal_hashes(folder, tmp_path)
    assert h2["images_sha256"] != h1["images_sha256"] and h2["manifest_sha256"] == h1["manifest_sha256"]
    assert not spec.seal_matches(folder, tmp_path)


class StubJudge:
    """Stands in for the model: no weights, always FAIL."""

    def __init__(self, model_id, max_tokens=16, thinking=False):
        self.model_id = model_id

    def ask(self, prompt, images):
        return {"verdict": "FAIL", "reply": "FAIL", "seconds": 0.0}


@pytest.fixture
def stub_model(monkeypatch):
    fake = types.ModuleType("labjudge.judge_mlx")
    fake.Judge = StubJudge
    fake.snapshot_of = lambda model_id: "stub"
    monkeypatch.setitem(sys.modules, "labjudge.judge_mlx", fake)
    monkeypatch.setattr(cli, "version", lambda name: "0")


def run_judge(tmp_path, data, thinking=False):
    argv = ["judge", "--model", "org/stub", "--data", str(data), "--results", str(tmp_path / "results" / "x"),
            "--conditions", "after", "before_after"]
    cli.main(argv + (["--thinking"] if thinking else []))


def test_judge_logs_every_look_at_a_sealed_split(tmp_path, monkeypatch, stub_model):
    monkeypatch.chdir(tmp_path)
    folder = fake_split(tmp_path)
    log = tmp_path / "results" / "sealed_log.jsonl"
    run_judge(tmp_path, folder)
    assert not log.exists(), "an unsealed folder must not be logged"
    (folder / spec.SEALED).write_text("sealed")
    run_judge(tmp_path, folder)
    run_judge(tmp_path, folder, thinking=True)
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == 4  # 2 runs x 2 conditions
    assert [x["condition"] for x in lines] == ["after", "before_after", "after", "before_after"]
    assert [x["thinking"] for x in lines] == ["off", "off", "on", "on"]
    assert all(x["model_id"] == "org/stub" and x["n"] == 3 and x["date"] for x in lines)


def _banner_like(rows: np.ndarray) -> bool:
    return (np.abs(rows.astype(int) - (250, 250, 248)).sum(-1) <= 6).mean() > 0.6


@pytest.mark.parametrize("split,limit", [("dev", None), ("train", 120)])
def test_saved_scene_images_have_no_banner(split, limit):
    """Train and dev only: the sealed test images are checked by `simset check`, never opened here."""
    folder = ROOT / "data" / "simset" / split
    if not (folder / "manifest.jsonl").exists():
        pytest.skip(f"no {split} scenes yet")
    rows = spec.read_rows(folder)
    assert rows and all(spec.row_problems(r) == [] for r in rows)
    paths = sorted({p for r in rows for p in r["images"].values()})
    if not (ROOT / paths[0]).exists():  # a fresh checkout has the manifests but not the pictures
        pytest.skip(f"the {split} pictures are not on disk")
    for rel in paths[:limit]:
        a = np.asarray(Image.open(ROOT / rel).convert("RGB"))
        assert a.shape[:2] == ((360, 640) if rel.endswith("_top.png") else (540, 640)), rel
        assert not _banner_like(a[:26]), rel
