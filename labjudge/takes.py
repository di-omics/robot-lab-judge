"""Find take records in the lab handling simulation takes and keep the usable ones."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

ENV_VAR = "LAB_HANDLING_BENCH"


@dataclass
class Take:
    take_id: str  # the folder name of the take
    run: str  # the run folder name ("takes" for the top-level takes folder)
    task: str
    title: str | None
    prompt: str
    seed: int
    policy: str | None
    model: str | None
    truth: str  # "pass" or "fail"
    video: Path


def bench_root(arg: str | None) -> Path:
    """The bench folder from the CLI argument, else from the env var."""
    value = arg or os.environ.get(ENV_VAR)
    if not value:
        raise SystemExit(f"give --bench PATH or set {ENV_VAR}")
    root = Path(value).expanduser()
    if not root.is_dir():
        raise SystemExit(f"not a folder: {root}")
    return root


def summary_paths(root: Path) -> list[Path]:
    return sorted(root.glob("runs/*/*/summary.json")) + sorted(root.glob("takes/*/summary.json"))


def load(root: Path) -> tuple[list[Take], list[tuple[str, str]]]:
    """Usable takes (error is null or absent, take.mp4 present) and the excluded ones with a reason."""
    usable, excluded = [], []
    for path in summary_paths(root):
        folder = path.parent
        record = json.loads(path.read_text())
        name = f"{folder.parent.name}/{folder.name}"
        if record.get("error") is not None:
            excluded.append((name, "error"))
            continue
        video = folder / "take.mp4"
        if not video.is_file():
            excluded.append((name, "no take.mp4"))
            continue
        if not isinstance(record.get("success"), bool):
            excluded.append((name, "success is not a bool"))
            continue
        usable.append(
            Take(
                take_id=folder.name,
                run=folder.parent.name,
                task=record["task"],
                title=record.get("title"),
                prompt=record["prompt"],
                seed=record["seed"],
                # a few early records name the driver "actor" instead of "model" and carry no policy
                policy=record.get("policy"),
                model=record.get("model") or record.get("actor"),
                truth="pass" if record["success"] else "fail",
                video=video,
            )
        )
    return usable, excluded
