"""A linear probe on frozen features: a frozen vision tower, then 1 small logistic head per chore.

Train on the scene set train split, report on dev. The sealed test split is refused everywhere.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download weights

MODEL_ID = "LiquidAI/LFM2.5-VL-450M-MLX-bf16"
TOWER = "siglip2-lfm2.5-vl-450m"
DATA = Path("data/simset")
FEATURES = Path("data/features")
OUT = Path("results/frozen")
VERDICTS = Path("results/simset_dev")
ALLOWED = ("train", "dev")


def check_split(split: str) -> str:
    """The split name, if it is train or dev. Anything else stops the program."""
    if split not in ALLOWED:
        raise SystemExit(f"split {split!r} refused: only train and dev are allowed")
    return split


def check_path(path) -> Path:
    """A data path, if it is not under the sealed test split. Checked by name, the folder is never opened."""
    p = Path(path)
    parts = [x for x in p.as_posix().split("/") if x not in ("", ".")]
    for i in range(len(parts) - 1):
        if parts[i] == "simset" and parts[i + 1] not in ALLOWED:
            raise SystemExit(f"path {p} refused: only data/simset/train and data/simset/dev are allowed")
    if "test" in parts:
        raise SystemExit(f"path {p} refused: it names a test folder")
    return p


def split_dir(split: str) -> Path:
    return check_path(DATA / check_split(split))


def read_rows(split: str) -> list[dict]:
    """Manifest rows of train or dev, in file order."""
    path = check_path(split_dir(split) / "manifest.jsonl")
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    for r in rows:
        for p in r["images"].values():
            check_path(p)
    return rows


def layout_of(row: dict) -> tuple[str, int]:
    """A layout is one task and one seed. All its scenes share 1 first frame."""
    return (row["task"], int(row["seed"]))
