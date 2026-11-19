"""Load a local MLX vision-language model once and ask it one PASS or FAIL question per take."""

import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download weights

from .prompts import question  # noqa: E402

VERDICT = re.compile(r"\b(PASS|FAIL)\b", re.IGNORECASE)
THINK_END = "</think>"


def parse(reply: str, thinking: bool = False) -> str:
    """The first PASS or FAIL in the visible answer, else 'unparsed'.

    The visible answer is the text after the closing think tag. With thinking on, a reply that
    never closes its thinking has no answer: words inside the reasoning are not a verdict.
    """
    if THINK_END in reply:
        visible = reply.split(THINK_END)[-1]
    elif thinking:
        return "unparsed"
    else:
        visible = reply
    match = VERDICT.search(visible)
    return match.group(1).upper() if match else "unparsed"


def local_snapshot(model_id: str) -> Path:
    """The snapshot folder of a model in the local Hugging Face cache, found without any network call.

    A local folder path is used as is. Some cached models have no refs/main entry, so the
    hub's own offline lookup fails; this reads the snapshots folder directly.
    """
    path = Path(model_id).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub.constants import HF_HUB_CACHE

    snaps = Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--")) / "snapshots"
    found = sorted(p for p in snaps.glob("*") if (p / "config.json").exists())
    if len(found) != 1:
        raise SystemExit(f"expected 1 cached snapshot of {model_id} under {snaps}, found {len(found)}")
    return found[0]


def snapshot_of(model_id: str) -> str:
    return local_snapshot(model_id).name


class Judge:
    def __init__(self, model_id: str, max_tokens: int = 16, thinking: bool = False):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.thinking = thinking
        path = local_snapshot(model_id)
        self.model, self.processor = load(str(path))
        self.config = load_config(path)

    def ask(self, prompt: str, images: list[str]) -> dict:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        text = question(prompt, len(images))
        formatted = apply_chat_template(
            self.processor, self.config, text, num_images=len(images), enable_thinking=self.thinking
        )
        start = time.perf_counter()
        result = generate(
            self.model,
            self.processor,
            formatted,
            image=images,
            max_tokens=self.max_tokens,
            temperature=0.0,
            enable_thinking=self.thinking,
            verbose=False,
        )
        seconds = time.perf_counter() - start
        reply = result.text if hasattr(result, "text") else str(result)
        verdict = parse(reply, self.thinking)
        return {"reply": reply, "verdict": verdict, "seconds": round(seconds, 2), "question": text}
