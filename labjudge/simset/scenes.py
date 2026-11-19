"""Scenes: one task, one layout seed, one end state of the labware, arms at home, settled and at rest.

A Layout builds the seeded bench once, keeps the settled start state, and plays the task's oracle to find the
final pose of every movable prop. Each scene then starts again from that saved start state (a fresh copy of the
physics state and a fresh score tracker), moves the labware into one end state, lets physics settle, and reads
the label from the bench's own success test on the settled state.
"""

import math
import zlib
from types import SimpleNamespace

import numpy as np

from . import bench as B

SETTLE_S = 2.0  # physics runs this long after the labware is placed
REST_S = 1.0  # then this long more, while rest and a steady label are checked
ORACLE_EXTRA_S = 1.0  # extra settle after the oracle's last step
LIFT = 0.002  # moved props are dropped from this high, as the bench's own traps do
MAX_RAISE = 0.08  # a moved prop that still overlaps something after this much raising is blocked
OVERLAP_TOL = 0.0015
REST_SPEED = 0.01  # m/s, the largest speed of any prop allowed in the rest window
REST_MOVE = 0.0015  # m, the largest drift of any prop over the rest window
REST_TURN = 2.0  # deg
EXTRA_WINDOWS = 2  # more rest windows for a prop that is still rolling
TILT_DEG = (4.0, 35.0)
YAW_SHARE = 0.3  # share of offset scenes that also get a yaw turn
YAW_MAX_DEG = 20.0
R_RANGE_MM = (0.5, 60.0)
SPAN = 8.0  # aimed offsets spread over this factor around the flip distance
## Props that ride along when their carrier moves: the tubes stand in the rack.
RIDERS = {"rack_into_box": {"rack": ("tube_a", "tube_b")}}


def _quat_to_mat(q):
    R = np.empty(9)
    B.lb.mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _mat_to_quat(R):
    return B.lb.oracle.mat_to_quat(R)


def _yaw_deg(R):
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_axis(axis, a):
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def rng_for(seed, task, index, attempt=0, salt=0):
    return np.random.default_rng([seed, zlib.crc32(task.encode()), index, attempt, salt])


class Layout:
    """One seeded bench of one task: its start state, its first views and the oracle's final poses."""

    def __init__(self, task_name, seed, render=True):
        lb = B.lb
        self.task = lb.tasks.TASKS[task_name]
        self.name, self.seed = task_name, seed
        self.bench = lb.take.set_bench(task_name, seed)
        self.env = self.bench.env
        self.movable = list(self.task.movable)
        m = lb.mujoco
        self._spec = m.mjtState.mjSTATE_INTEGRATION
        self._state0 = np.empty(m.mj_stateSize(self.env.model, self._spec))
        m.mj_getState(self.env.model, self.env.data, self._state0, self._spec)
        self.start = self.poses()
        self.first_images = self.env.images() if render else None
        b = self.fresh()
        self.oracle_problems = lb.oracle.play(b, self.task.oracle(b.specs))
        self._run(b, ORACLE_EXTRA_S)
        self.oracle_pass = bool(b.tracker.success_now)
        self.final = self.poses()
        self.placed = [n for n in self.movable if self._moved(n)]
        traps = self.task.traps(self.bench.specs) if self.task.traps else {}
        ## Traps whose end state needs no hand: the last step lets go and settles.
        self.traps = {k: v for k, v in traps.items() if v and v[-1].settle_s > 0}
        self.held_traps = sorted(set(traps) - set(self.traps))

    def close(self):
        self.env.close()

    def _moved(self, n):
        (p0, q0), (p1, q1) = self.start[n], self.final[n]
        turn = _angle(_quat_to_mat(q0), _quat_to_mat(q1))
        return np.linalg.norm(p1 - p0) > 0.005 or turn > 3.0

    def poses(self):
        out = {}
        for n in self.movable:
            q = self.env.data.joint(f"{n}_free").qpos
            out[n] = (q[:3].copy(), q[3:7].copy())
        return out

    def fresh(self):
        """The start state again, with a new world view and a new tracker primed on the start layout."""
        lb, m = B.lb, B.lb.mujoco
        m.mj_setState(self.env.model, self.env.data, self._state0, self._spec)
        m.mj_forward(self.env.model, self.env.data)
        world = lb.score.World(self.env)
        z0 = {n: float(world.pos(n)[2]) for n in self.movable}
        stages, success, *rest = self.task.scoring(self.bench.specs, z0)
        tracker = lb.score.Tracker(world, stages, success, self.movable, requires=rest[0] if rest else ())
        tracker.prime()
        return SimpleNamespace(task=self.task, env=self.env, world=world, tracker=tracker, specs=self.bench.specs,
                               layout=self.bench.layout)

    def _run(self, b, seconds, watch=None):
        home = B.lb.sim.HOME
        for _ in range(round(seconds / B.lb.sim.STEP_S)):
            b.env.step(home, home)
            b.tracker.update()
            if watch is not None:
                watch(b)

    # Placing labware.

    def _set(self, n, pos, quat):
        B.lb.oracle.place(self.env, n, pos, quat)

    def _group_ids(self, names):
        m = self.env.model
        return {m.body(n).id for n in names}

    def overlap(self, names):
        """Deepest overlap between the named props and anything else, metres."""
        m, d = self.env.model, self.env.data
        ids = self._group_ids(names)
        deepest = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            a = m.body_rootid[m.geom_bodyid[c.geom1]] in ids
            b = m.body_rootid[m.geom_bodyid[c.geom2]] in ids
            if a != b:
                deepest = max(deepest, -float(c.dist))
        return deepest

    def raise_clear(self, names):
        """Raise the named props in 1 mm steps until they overlap nothing. @return mm raised, or None if blocked."""
        raised = 0.0
        while self.overlap(names) > OVERLAP_TOL:
            if raised >= MAX_RAISE:
                return None
            raised += 0.001
            for n in names:
                q = self.env.data.joint(f"{n}_free").qpos
                self._set(n, q[:3] + np.array([0.0, 0.0, 0.001]), q[3:7].copy())
        return round(raised * 1000, 1)

    def move_group(self, carrier, G, shift):
        """Turn the carrier and its riders by G about the carrier's final origin, then shift them."""
        names = [carrier, *RIDERS.get(self.name, {}).get(carrier, ())]
        c = self.final[carrier][0]
        for n in names:
            p, q = self.final[n]
            R = G @ _quat_to_mat(q)
            self._set(n, c + G @ (p - c) + shift, _mat_to_quat(R))
        return names

    def apply(self, plan):
        """A fresh start state with the labware moved as the plan says. @return (bench view, mm raised or None)."""
        b = self.fresh()
        kind = plan["kind"]
        raised = 0.0
        if kind == "untouched":
            return b, raised
        if kind == "trap":
            problems = B.lb.oracle.play(b, self.traps[plan["trap"]])
            plan["trap_problems"] = problems
            return b, raised
        for n in self.movable:
            self._set(n, *self.final[n])
        if kind == "clean":
            for n, (dx, dy) in plan["jitter"].items():
                p, q = self.final[n]
                self._set(n, p + np.array([dx, dy, 0.0]) / 1000, q)
            for n in plan["jitter"]:
                r = self.raise_clear([n])
                if r is None:
                    return b, None
                raised = max(raised, r)
            return b, raised
        if kind == "partial":
            n = plan["prop"]
            self._set(n, *self.start[n])
            return b, self.raise_clear([n])
        if kind == "offset":
            G = _rot_z(math.radians(plan["yaw_deg"]))
            shift = np.array([plan["dx_mm"] / 1000, plan["dy_mm"] / 1000, LIFT])
        elif kind == "tilt":
            a = math.radians(plan["tilt_axis_deg"])
            G = _rot_axis((math.cos(a), math.sin(a), 0.0), math.radians(plan["tilt_deg"]))
            shift = np.array([0.0, 0.0, LIFT])
        else:
            raise ValueError(kind)
        names = self.move_group(plan["prop"], G, shift)
        return b, self.raise_clear(names)

    # Settling and reading the result.

    def settle(self, b):
        """Let physics run, then check rest and a steady label over the rest window. @return a result dict."""
        lb = B.lb
        table_z = lb.sim.TABLE_Z
        worst = {"speed": 0.0}

        def blew(bb):
            q = bb.env.data.qpos
            if not np.all(np.isfinite(q)):
                return True
            return any(np.abs(bb.world.pos(n)).max() > 3.0 or bb.world.speed(n) > 5.0 for n in self.movable)

        bad = {"blew": False}

        def watch_settle(bb):
            if not bad["blew"] and blew(bb):
                bad["blew"] = True

        self._run(b, SETTLE_S, watch_settle)
        for window in range(1 + EXTRA_WINDOWS):  # a prop still rolling gets up to 2 more rest windows
            start = {n: (b.world.pos(n), b.world.rot(n)) for n in self.movable}
            labels = []
            worst["speed"] = 0.0

            def watch_rest(bb):
                watch_settle(bb)
                worst["speed"] = max(worst["speed"], max(bb.world.speed(n) for n in self.movable))
                labels.append(bool(bb.tracker.success_now))

            self._run(b, REST_S, watch_rest)
            drift = max(float(np.linalg.norm(b.world.pos(n) - start[n][0])) for n in self.movable)
            turn = max(_angle(b.world.rot(n), start[n][1]) for n in self.movable)
            if bad["blew"] or not (worst["speed"] > REST_SPEED or drift > REST_MOVE or turn > REST_TURN):
                break
        low = min(float(b.world.pos(n)[2]) for n in self.movable)
        discard = None
        if bad["blew"]:
            discard = "physics blew up"
        elif low < table_z - 0.03:
            discard = "a prop left the table"
        elif worst["speed"] > REST_SPEED or drift > REST_MOVE or turn > REST_TURN:
            discard = "not at rest"
        elif len(set(labels)) > 1:
            discard = "label not steady"
        return {
            "truth": "pass" if b.tracker.success_now else "fail",
            "passed_while_settling": b.tracker.success_at is not None,
            "discard": discard,
            "rest": {"max_speed_m_s": round(worst["speed"], 5), "drift_mm": round(drift * 1000, 3),
                     "turn_deg": round(turn, 3), "windows": window + 1},
        }

    def error(self, b, n):
        """Final error of one prop against its oracle final pose."""
        p, R = b.world.pos(n), b.world.rot(n)
        pf, qf = self.final[n]
        Rf = _quat_to_mat(qf)
        return {
            "planar_mm": round(float(np.linalg.norm(p[:2] - pf[:2])) * 1000, 2),
            "height_mm": round(float(p[2] - pf[2]) * 1000, 2),
            "yaw_deg": round(_wrap(_yaw_deg(R) - _yaw_deg(Rf)), 2),
            "tilt_deg": round(_angle_up(R, Rf), 2),
        }

    def run_plan(self, plan):
        """Apply, settle and measure one plan. @return (result dict, bench view)."""
        b, raised = self.apply(plan)
        if raised is None:
            return {"discard": "blocked", "truth": None}, b
        result = self.settle(b)
        result["raised_mm"] = raised
        focus = plan.get("prop")
        if focus is None:
            errs = {n: self.error(b, n) for n in (self.placed or self.movable)}
            focus = max(errs, key=lambda n: errs[n]["planar_mm"])
        result["prop"] = focus
        result["error"] = self.error(b, focus)
        result["errors"] = {n: self.error(b, n) for n in self.movable}
        return result, b


def _angle(Ra, Rb):
    """Rotation angle between two rotation matrices, degrees."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, c)))))


def _angle_up(Ra, Rb):
    c = float(Ra[:, 2] @ Rb[:, 2])
    return float(math.degrees(math.acos(max(-1.0, min(1.0, c)))))


# Plans.


def plan_offset(layout, rng, r_mm=None, aim=None, yaw=True):
    """One prop moved in the table plane by r mm in a random direction, sometimes turned too."""
    prop = str(rng.choice(layout.placed))
    if r_mm is None:
        lo, hi = aim or R_RANGE_MM
        r_mm = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    angle = float(rng.uniform(0, 2 * math.pi))
    yaw_deg = float(rng.uniform(-YAW_MAX_DEG, YAW_MAX_DEG)) if yaw and rng.random() < YAW_SHARE else 0.0
    return {"kind": "offset", "prop": prop, "r_mm": round(r_mm, 3), "dx_mm": round(r_mm * math.cos(angle), 3),
            "dy_mm": round(r_mm * math.sin(angle), 3), "yaw_deg": round(yaw_deg, 2), "tilt_deg": 0.0}


def plan_tilt(layout, rng):
    prop = str(rng.choice(layout.placed))
    return {"kind": "tilt", "prop": prop, "dx_mm": 0.0, "dy_mm": 0.0, "yaw_deg": 0.0,
            "tilt_deg": round(float(rng.uniform(*TILT_DEG)), 2),
            "tilt_axis_deg": round(float(rng.uniform(0, 360)), 1)}


def plan_clean(layout, rng):
    jitter = {}
    for n in layout.placed:
        r, a = float(rng.uniform(0, 0.5)), float(rng.uniform(0, 2 * math.pi))
        jitter[n] = [round(r * math.cos(a), 3), round(r * math.sin(a), 3)]
    return {"kind": "clean", "prop": None, "jitter": jitter}


def plan_partial(layout, rng):
    if len(layout.placed) < 2:
        return {"kind": "untouched", "prop": None}
    return {"kind": "partial", "prop": str(rng.choice(layout.placed))}


def plan_trap(layout, name):
    """One of the task's own traps; the prop is the one its last step moves."""
    return {"kind": "trap", "prop": layout.traps[name][-1].prop, "trap": name}


def schedule(n_scenes, ordinal, layout):
    """Kinds for one layout: clean, then a trap or the partial scene in turn, then a tilt and offsets.
    With fewer than 5 scenes per layout the tilt comes on every second layout only."""
    names = sorted(layout.traps)
    second = ("trap", names[(ordinal // 2) % len(names)]) if names and ordinal % 2 == 0 else ("partial", None)
    rest = max(0, n_scenes - 2)
    tilt = 1 if rest >= 3 or (rest >= 1 and ordinal % 2 == 1) else 0
    kinds = [("clean", None), second] + [("tilt", None)] * tilt + [("offset", None)] * (rest - tilt)
    return kinds[:n_scenes]


def kind_means(n_scenes):
    """Mean tilt and offset scenes per layout under schedule()."""
    def count(ordinal):
        rest = max(0, n_scenes - 2)
        tilt = 1 if rest >= 3 or (rest >= 1 and ordinal % 2 == 1) else 0
        return tilt, rest - tilt

    (t0, o0), (t1, o1) = count(0), count(1)
    return (t0 + t1) / 2, (o0 + o1) / 2


def make_plan(layout, kind, trap, rng, aim=None):
    if kind == "clean":
        return plan_clean(layout, rng)
    if kind == "trap":
        return plan_trap(layout, trap)
    if kind == "partial":
        return plan_partial(layout, rng)
    if kind == "tilt":
        return plan_tilt(layout, rng)
    if kind == "offset":
        return plan_offset(layout, rng, aim=aim)
    raise ValueError(kind)


def aim_range(flip_mm, q):
    """Offsets log-uniform over a factor SPAN around the flip distance, so a share q falls below it."""
    lo_cap, hi_cap = R_RANGE_MM
    flip_mm = min(max(flip_mm, lo_cap * 1.01), hi_cap / 1.01)
    lo = flip_mm / SPAN ** q
    hi = flip_mm * SPAN ** (1 - q)
    if hi > hi_cap:  # keep the share below the flip at q when the top end is cut at 60 mm
        hi = hi_cap
        lo = math.exp((math.log(flip_mm) - q * math.log(hi)) / (1 - q))
    if lo < lo_cap:
        lo = lo_cap
        hi = min(hi_cap, math.exp((math.log(flip_mm) - (1 - q) * math.log(lo)) / q))
    return lo, hi
