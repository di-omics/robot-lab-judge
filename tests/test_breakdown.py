"""The scene set breakdown counts judged PASS per kind, distance and task; the board names the scene split."""

import json

from labjudge.score import board, breakdown


def scene(i, kind, truth, planar=0.0, height=0.0, task="plate_to_tray"):
    return {"take_id": f"s{i}", "task": task, "kind": kind, "truth": truth,
            "error": {"planar_mm": planar, "height_mm": height}}


def write(path, verdicts, manifest, data="data/simset/dev", thinking="off"):
    header = {"kind": "header", "model_id": "org/m", "thinking": thinking, "condition": "after", "data": data}
    truth = {s["take_id"]: s["truth"] for s in manifest}
    rows = [{"take_id": k, "truth": truth[k], "verdict": v, "seconds": 1.0} for k, v in verdicts.items()]
    path.write_text("\n".join(json.dumps(x) for x in [header, *rows]) + "\n")


MANIFEST = [
    scene(0, "clean", "pass"),
    scene(1, "offset", "pass", planar=3.0),
    scene(2, "offset", "fail", planar=3.0, height=4.0),  # 5 mm away in 3 dimensions
    scene(3, "offset", "fail", planar=30.0),
    scene(4, "trap", "fail", task="stack_plates"),
    scene(5, "clean", "pass", task="stack_plates"),
    {**scene(6, "offset", "fail", planar=30.0, task="stack_plates"), "commanded": {"yaw_deg": 12.0}},  # turned too
]
VERDICTS = {"s0": "PASS", "s1": "PASS", "s2": "PASS", "s3": "FAIL", "s4": "unparsed", "s5": "PASS", "s6": "FAIL"}


def line(text, *starts):
    return next(x for x in text.splitlines() if x.startswith("| " + " | ".join(starts) + " |"))


def test_breakdown_counts_judged_pass_per_kind_distance_and_task(tmp_path):
    write(tmp_path / "verdicts_m_after.jsonl", VERDICTS, MANIFEST)
    text = breakdown(tmp_path, MANIFEST)
    assert "m, after" in text
    assert line(text, "clean", "pass").endswith("| 2 | 2 of 2 |")
    assert line(text, "offset", "fail").endswith("| 3 | 1 of 3 |")
    assert line(text, "trap", "fail").endswith("| 1 | 0 of 1 |")  # unparsed is not a PASS
    assert line(text, "2 to 5 mm", "pass").endswith("| 1 | 1 of 1 |")
    assert line(text, "5 to 10 mm", "fail").endswith("| 1 | 1 of 1 |")  # height counts in the distance
    assert line(text, "20 to 40 mm", "fail").endswith("| 1 | 0 of 1 |")  # the turned scene is left out
    assert "| tilt |" not in text  # a kind with no scenes has no line
    assert line(text, "plate_to_tray").endswith("| 4 | 0.75 |")
    assert line(text, "stack_plates").endswith("| 3 | 0.75 |")


def test_board_names_the_scene_split_and_keeps_the_first_look_title(tmp_path):
    write(tmp_path / "verdicts_m_after.jsonl", VERDICTS, MANIFEST)
    text = board(tmp_path)
    assert text.startswith("# Board: vision judges on the scene set, dev split")
    assert "sealed test split" in text and "already seen" not in text
    old = tmp_path / "old"
    old.mkdir()
    write(old / "verdicts_m_after.jsonl", VERDICTS, MANIFEST, data=None)
    assert board(old).startswith("# Board: vision judges on the first-look takes")
