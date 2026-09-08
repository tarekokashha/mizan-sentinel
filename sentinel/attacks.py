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


# The longest base-sweep-safe lever with real box margin (see
# task-10-fix-1.md, finding 2). Maximising the lever to the box's exact
# inscribed radius (the original [-0.75, -0.3, -1.7, -pi/2, 0.0]) landed
# 0.04 mm outside the box with zero margin by construction, forcing a
# borrowed plant_margin_m tolerance onto an unrelated containment test.
# Searching instead for the longest lever that keeps 5 mm of clearance gives
# a posture that is inside exactly, verified over a 289-point sweep of
# joint 0 across the full +-pi range:
#   minimum sustained lever        = 0.645023 m
#   worst box excursion, true box  = 0.000e+00, inside with no tolerance at all
#   also clears a 5 mm inset box   = 0.000e+00
#   flange z constant at 0.393318, well inside the 0.05 to 0.90 box
#   flange at base angle 0         = [-0.631099, -0.1333, 0.393318]
# At the declared qd_max of 1.0 rad/s that is 0.645023 m/s against a 0.25 m/s
# limit, a ratio of 2.58, and the test's own meaningfulness threshold of
# lever * qd_max > 0.5 still holds comfortably.
LEVER_POSTURE = np.array([-0.56, 0.32, 0.35, -1.5708, 0.0])


@attack("lever_sprint", "tcp_speed",
        "fully extended base sweep at the legal joint speed; the lever arm, "
        "not the joint rate, is what makes the flange fast")
def _lever_sprint(env, rng):
    def f(state, step, t):
        q = state.q.copy()
        # Ramp into the lever posture at a legal rate rather than snapping to
        # it (see task-10-fix-1.md, finding 1). Snapping q[1:] = LEVER_POSTURE
        # unconditionally requests joint steps tens of times over qd_max from
        # an arbitrary start (Task 11's episode start state is nowhere near
        # the posture), which trips the velocity guard before tcp_speed ever
        # can and falsifies this attack's own premise: that every joint-space
        # limit is respected exactly and only the Cartesian guard can see the
        # hazard.
        cap = env.qd_max[1:] * DT * 0.5
        q[1:] = q[1:] + np.clip(LEVER_POSTURE - q[1:], -cap, cap)
        q[0] = q[0] + env.qd_max[0] * DT * 0.99   # sweep the base, legally
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
