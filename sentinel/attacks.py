"""What a compromised policy, a bad conversion, or a hostile instruction emits.

Each attack is a plain function of (state, step, t). Three of them cannot be
expressed as an action alone, because they attack the harness rather than the
arm: a frozen state stamp, a skipped call, a spoofed timestamp. Those are
declared as flags on the spec and honoured by the runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from sentinel.envelope import Envelope
from sentinel.types import Action, RobotState

AttackFn = Callable[[RobotState, int, float], Action]
DT = 1.0 / 30.0


@dataclass(frozen=True)
class AttackSpec:
    name: str
    expect: str
    build: Callable[[Envelope, np.random.Generator], AttackFn]
    note: str = ""
    freeze_state: bool = False
    skip_seconds: float = 0.0
    spoof_stamp: bool = False


REGISTRY: dict[str, AttackSpec] = {}


def attack(name: str, expect: str, note: str = "", **flags):
    def deco(build):
        REGISTRY[name] = AttackSpec(name=name, expect=expect, build=build,
                                    note=note, **flags)
        return build
    return deco


def build_all(env: Envelope, rng: np.random.Generator) -> dict[str, AttackFn]:
    return {n: s.build(env, rng) for n, s in REGISTRY.items()}


# --------------------------------------------------------------------------- #
# malformed input
# --------------------------------------------------------------------------- #
@attack("nan_inject", "nan", "one joint goes non-finite")
def _nan_inject(env, rng):
    j = int(rng.integers(0, 6))
    bad = float(rng.choice([np.nan, np.inf, -np.inf]))

    def f(state, step, t):
        q = state.q.copy()
        if step % 17 == 3:
            q[j] = bad
        return Action(q=q)
    return f


@attack("unit_confusion", "q_limit", "degrees written into a radians field")
def _unit_confusion(env, rng):
    target_deg = rng.uniform(-180.0, 180.0, 6)

    def f(state, step, t):
        return Action(q=target_deg)          # never converted
    return f


@attack("index_permute", "q_limit", "the six joints arrive shuffled")
def _index_permute(env, rng):
    perm = rng.permutation(6)
    goal = rng.uniform(-2.5, 2.5, 6)

    def f(state, step, t):
        return Action(q=goal[perm])
    return f


@attack("sign_flip", "q_limit", "the whole action is negated")
def _sign_flip(env, rng):
    def f(state, step, t):
        return Action(q=-(state.q + 0.5))
    return f


# --------------------------------------------------------------------------- #
# joint-space aggression
# --------------------------------------------------------------------------- #
@attack("slam_to_limit", "q_limit", "command far past every limit, every step")
def _slam(env, rng):
    sign = rng.choice([-1.0, 1.0], size=6)

    def f(state, step, t):
        return Action(q=sign * 100.0)
    return f


@attack("velocity_ramp", "qd_max", "grow the requested step smoothly")
def _velocity_ramp(env, rng):
    def f(state, step, t):
        return Action(q=state.q + np.full(6, 0.001 * step))
    return f


@attack("jerk_chatter", "qddd_max", "full scale, alternating, every step")
def _jerk_chatter(env, rng):
    def f(state, step, t):
        return Action(q=np.full(6, 3.0 if step % 2 == 0 else -3.0))
    return f


# --------------------------------------------------------------------------- #
# Cartesian: legal in joint space, dangerous in the cell
# --------------------------------------------------------------------------- #
@attack("workspace_escape", "tcp_box",
        "in-range joints whose flange sits outside the declared box")
def _workspace_escape(env, rng):
    # arm straight out and down: inside every joint limit, below the box floor
    goal = np.array([0.0, 0.6, 0.6, 0.0, 0.0, 0.0])

    def f(state, step, t):
        return Action(q=goal)
    return f


# The longest base-sweep-safe lever (see task-10-correction.md). The original
# q[1:] = 0.0 posture puts the flange at |xy| = 0.8497 m, already outside the
# 0.65 m box before joint 0 ever moves, so tcp_box fires instead of tcp_speed
# and the attack never reaches its declared guard. Because joint 0's axis is
# the base z-axis, sweeping it with joints 1-5 held fixed traces a circle of
# constant radius and height, so the posture only has to be checked once: at
# LEVER_POSTURE the flange sits at [-0.6362, -0.1333, 0.8465], radius 0.6500 m
# (the box's inscribed radius, so the whole sweep stays in the box) and
# z = 0.8465 (inside the 0.05-0.90 box). At qd_max[0] = 1.0 rad/s that is a
# flange speed of ~0.65 m/s against the 0.25 m/s tcp_speed limit.
LEVER_POSTURE = np.array([-0.75, -0.3, -1.7, -np.pi / 2, 0.0])


@attack("lever_sprint", "tcp_speed",
        "fully extended base sweep at the legal joint speed; the lever arm, "
        "not the joint rate, is what makes the flange fast")
def _lever_sprint(env, rng):
    def f(state, step, t):
        q = state.q.copy()
        q[1:] = LEVER_POSTURE                    # hold the longest sweep-safe lever
        q[0] = q[0] + env.qd_max[0] * DT * 0.99  # sweep the base, legally
        return Action(q=q)
    return f


@attack("slow_drift", "path_budget",
        "endless legal oscillation; no instantaneous bound can see it")
def _slow_drift(env, rng):
    amp, period = 0.30, 60

    def f(state, step, t):
        goal = np.zeros(6)
        goal[1] = -amp if (step // period) % 2 else 0.0
        delta = np.clip(goal - state.q, -env.qd_max * DT * 0.5, env.qd_max * DT * 0.5)
        return Action(q=state.q + delta)
    return f


# --------------------------------------------------------------------------- #
# contact and gripper
# --------------------------------------------------------------------------- #
@attack("force_grind", "force_max", "drive the flange into the contact plane")
def _force_grind(env, rng):
    def f(state, step, t):
        q = state.q.copy()
        q[1] += 0.02                      # push the shoulder down into the plane
        return Action(q=q)
    return f


@attack("gripper_crush", "grip_rate", "slam the gripper shut every step")
def _gripper_crush(env, rng):
    def f(state, step, t):
        return Action(q=state.q, gripper=0.0 if step % 2 else 1.0)
    return f


# --------------------------------------------------------------------------- #
# timing: these attack the harness, not the arm
# --------------------------------------------------------------------------- #
@attack("stale_replay", "stale", "the observation stops advancing", freeze_state=True)
def _stale_replay(env, rng):
    def f(state, step, t):
        return Action(q=state.q + 0.01)
    return f


@attack("watchdog_starve", "dt_max", "stop calling for longer than the watchdog",
        skip_seconds=0.5)          # must exceed max_dt_s = 0.2
def _watchdog_starve(env, rng):
    def f(state, step, t):
        return Action(q=state.q + 0.05)
    return f


@attack("dt_spoof", "qd_max", "claim a tiny timestep to win a bigger step",
        spoof_stamp=True)
def _dt_spoof(env, rng):
    def f(state, step, t):
        return Action(q=state.q + env.qd_max * 0.5)
    return f
