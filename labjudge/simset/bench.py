"""Load the simulator bench (read-only) and the physics helpers for one layout.

The bench is imported from its folder and never written: no bytecode is saved, and nothing in it is called that
writes files. One process builds the stock arm scene once and reuses the same XML text for every layout
(checked byte for byte against a fresh build), because building it is the slowest part of a new environment.
"""

import os
import sys
from pathlib import Path

ENV_VAR = "LAB_HANDLING_BENCH"

lb = None  # the loaded bench modules, filled by load()


class _Modules:
    pass


def load(path: str | None = None):
    """Import the bench package from its folder. Safe to call twice."""
    global lb
    if lb is not None:
        return lb
    value = path or os.environ.get(ENV_VAR)
    if not value:
        raise SystemExit(f"give --bench PATH or set {ENV_VAR}")
    root = Path(value).expanduser().resolve()
    if not (root / "labbench" / "tasks.py").is_file():
        raise SystemExit(f"no labbench/tasks.py under {root}")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    import mujoco
    import numpy as np

    from labbench import __version__, oracle, score, sim, take, tasks, video

    stock = sim._STOCK_BUILD
    cache = {}

    def cached_stock():
        key = (sim.yam_env.TOP_CAMERA_LOWER_M, sim.yam_env.TOP_CAMERA_TILT_UP_DEG)
        if key not in cache:
            cache[key] = stock()
            if stock() != cache[key]:
                raise RuntimeError("the stock scene XML is not the same twice; cannot reuse it")
        return cache[key]

    sim._STOCK_BUILD = cached_stock
    m = _Modules()
    m.root, m.mujoco, m.np = root, mujoco, np
    m.version, m.oracle, m.score, m.sim, m.take, m.tasks, m.video = __version__, oracle, score, sim, take, tasks, video
    lb = m
    return m


def versions() -> dict:
    from importlib import metadata

    return {"labbench": lb.version, "code": lb.take.code_id(), "dreamlabs": metadata.version("dreamlabs"),
            "mujoco": lb.mujoco.__version__}


def included_tasks() -> tuple[list[str], dict]:
    """Tasks whose pass rule reads only the final state, and the left-out ones with the reason."""
    keep, out = [], {}
    for name, task in lb.tasks.TASKS.items():
        probe = _probe_scoring(task)
        if len(probe) > 2 and probe[2]:
            out[name] = f"success also needs earlier stages: {', '.join(probe[2])}"
        else:
            keep.append(name)
    return keep, out


def _probe_scoring(task):
    """The scoring tuple of a task, from a seeded bench (seed 10000, a train layout)."""
    bench = lb.take.set_bench(task.name, 10000)
    try:
        z0 = {n: float(bench.world.pos(n)[2]) for n in task.movable}
        return task.scoring(bench.specs, z0)
    finally:
        bench.env.close()
