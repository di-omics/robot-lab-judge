"""Every look at a sealed split is logged: one line per model and condition, before the judging starts."""

import datetime
import json
from pathlib import Path

SEALED = "SEALED.md"
LOG = Path("results/sealed_log.jsonl")


def is_sealed(data) -> bool:
    return (Path(data) / SEALED).is_file()


def log_look(log, data, model_id: str, thinking: bool, condition: str, n: int) -> dict:
    """Append one line to the sealed log. @return the line as a dict."""
    line = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "data": str(data), "model_id": model_id, "thinking": "on" if thinking else "off",
        "condition": condition, "n": n,
    }
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps(line) + "\n")
    return line
