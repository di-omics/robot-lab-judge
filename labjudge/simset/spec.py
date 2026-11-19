"""Fixed settings of the near-miss scene set, and checks that need no simulator.

This module imports nothing from the simulator, so the judge's own environment can test it.
"""

import hashlib
import json
from pathlib import Path

DATA = Path("data/simset")

## Layout seeds per split. The ranges never overlap, and none touches the bench's own seeds 0-4 or 100-104.
SPLITS = {
    "train": range(10000, 10030),
    "dev": range(20000, 20008),
    "test": range(30000, 30020),
}
BENCH_SEEDS = set(range(0, 5)) | set(range(100, 105))

## Tasks that appear only in the test split, to measure unseen chores.
HELD_OUT = ("plate_to_hotel_middle", "small_tubes_to_rack", "rack_into_box")

## Calibration uses the first layouts of the train range only.
CALIBRATION_SEEDS = (10000, 10001, 10002)

KINDS = ("clean", "offset", "tilt", "partial", "untouched", "trap")

## Keys the existing judge reads from every manifest row.
JUDGE_KEYS = ("take_id", "run", "task", "title", "prompt", "seed", "truth", "images")
IMAGE_KEYS = ("first_views", "last_views", "last_top")
## Keys this scene set adds.
EXTRA_KEYS = ("kind", "prop", "trap", "commanded", "error", "rule", "seen_task", "plan")

## Pass share per task that the generator aims for.
PASS_SHARE = (0.35, 0.65)

SEALED = "SEALED.md"
SEAL_RULE = (
    "Prompts, thresholds and training use train and dev only. "
    "Every scoring run on test is logged in results/sealed_log.jsonl."
)


def split_of_seed(seed: int) -> str | None:
    for name, seeds in SPLITS.items():
        if seed in seeds:
            return name
    return None


def splits_disjoint() -> list[str]:
    """Problems with the seed ranges; empty means the splits share no seed and avoid the bench's seeds."""
    problems = []
    names = list(SPLITS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(SPLITS[a]) & set(SPLITS[b])
            if shared:
                problems.append(f"{a} and {b} share {len(shared)} seeds")
        if set(SPLITS[a]) & BENCH_SEEDS:
            problems.append(f"{a} uses a bench seed")
    return problems


def tasks_for(split: str, included: list[str]) -> list[str]:
    """The tasks a split holds: test has every included task, train and dev leave the held-out ones out."""
    if split == "test":
        return list(included)
    return [t for t in included if t not in HELD_OUT]


def scene_id(split: str, task: str, seed: int, index: int, kind: str) -> str:
    return f"{split}-{task}-s{seed}-{index:02d}-{kind}"


def row_problems(row: dict) -> list[str]:
    """What is missing or wrong in one manifest row."""
    problems = [f"missing {k}" for k in JUDGE_KEYS + EXTRA_KEYS if k not in row]
    if problems:
        return problems
    if row["truth"] not in ("pass", "fail"):
        problems.append(f"truth is {row['truth']!r}")
    if row["kind"] not in KINDS:
        problems.append(f"kind is {row['kind']!r}")
    for k in IMAGE_KEYS:
        if not isinstance(row["images"].get(k), str):
            problems.append(f"no image path {k}")
    if split_of_seed(row["seed"]) != row["run"]:
        problems.append(f"seed {row['seed']} is not in split {row['run']}")
    if row["seen_task"] != (row["task"] not in HELD_OUT):
        problems.append("seen_task does not match the held-out list")
    if row["run"] != "test" and row["task"] in HELD_OUT:
        problems.append(f"held-out task {row['task']} in {row['run']}")
    return problems


def read_rows(folder: Path) -> list[dict]:
    path = Path(folder) / "manifest.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal_hashes(folder: Path, root: Path = Path(".")) -> dict:
    """The manifest hash, and one hash over the sorted list of every image's hash."""
    folder, root = Path(folder), Path(root)
    rows = read_rows(folder)
    paths = sorted({p for r in rows for p in r["images"].values()})
    image_hashes = sorted(sha256_file(root / p) for p in paths)
    joined = "\n".join(image_hashes).encode()
    return {
        "manifest_sha256": sha256_file(folder / "manifest.jsonl"),
        "images_sha256": hashlib.sha256(joined).hexdigest(),
        "n_rows": len(rows),
        "n_images": len(paths),
    }


def seal_text(hashes: dict, date: str) -> str:
    return "\n".join([
        "# Sealed: near-miss scene set, test split",
        "",
        f"Sealed on {date}.",
        "",
        f"- manifest.jsonl sha256: `{hashes['manifest_sha256']}`",
        f"- sha256 of the sorted list of image sha256 values: `{hashes['images_sha256']}`",
        f"- rows: {hashes['n_rows']}, images: {hashes['n_images']}",
        "",
        "## Rule",
        "",
        SEAL_RULE,
        "Do not look at, tune on, or make contact sheets of these images.",
        "",
    ])


def seal_matches(folder: Path, root: Path = Path(".")) -> bool:
    """True when SEALED.md still names the hashes the folder has now."""
    text = (Path(folder) / SEALED).read_text()
    h = seal_hashes(folder, root)
    return h["manifest_sha256"] in text and h["images_sha256"] in text
