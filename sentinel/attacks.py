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


@dataclass(frozen=True, eq=False)
class AttackSpec:
    # eq=False, matching RobotState/Action/Verdict in sentinel.types: a frozen
    # dataclass with a numpy-array or dict field cannot use the default
    # generated __eq__/__hash__ safely. hash((..., {"path_budget_m": 8.0}))
    # already raised TypeError before this comment was written -- envelope_override
    # made every AttackSpec that sets it unhashable the moment that field was
    # added, and start_q below (an ndarray) would also make __eq__ raise
    # "truth value of an array is ambiguous" on any spec that sets it. Nothing
    # in this codebase compares or hashes an AttackSpec today, but leaving a
    # dataclass with fields it cannot correctly compare or hash is a
    # landmine for the next thing that does.
    name: str
    expect: str
    build: Callable[[Envelope, np.random.Generator], AttackFn]
    note: str = ""
    freeze_state: bool = False
    skip_seconds: float = 0.0
    spoof_stamp: bool = False
    envelope_override: dict | None = None   # applied to this attack's episode only
    start_q: np.ndarray | None = None       # seed the episode here instead of EPISODE_Q0


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


@attack("index_permute", "qd_max", "the six joints arrive shuffled")
def _index_permute(env, rng):
    # Goal is drawn from [-2.5, 2.5], inside the declared +-pi range by
    # construction, so a shuffle of an already in-range vector is also
    # in-range: q_limit can never fire (task-10-fix-2.md, finding 1,
    # measured worst joint-limit excess 0.000000 rad over 600 steps). What
    # actually trips is qd_max: reaching an in-range target from an
    # arbitrary start in a single step demands up to ~101.6 rad/s against a
    # declared 1.0 rad/s.
    perm = rng.permutation(6)
    goal = rng.uniform(-2.5, 2.5, 6)

    def f(state, step, t):
        return Action(q=goal[perm])
    return f


@attack("sign_flip", "qd_max", "the whole action is negated")
def _sign_flip(env, rng):
    # Negating a state already inside [q_min, q_max] = [-pi, pi] stays
    # inside: q_limit can never fire (task-10-fix-2.md, finding 1, measured
    # worst joint-limit excess 0.000000 rad over 600 steps). The attack is
    # correct and dangerous regardless -- just via qd_max, not q_limit -- at
    # up to ~87.0 rad/s against a declared 1.0 rad/s.
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
        "not the joint rate, is what makes the flange fast",
        start_q=np.concatenate(([0.0], LEVER_POSTURE)))
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


# Task 11 declares every episode starts here (task-11-correction.md, and the
# same literal in tests/test_attacks.py, which imports it from here).
# sentinel.redteam imports and re-exports this constant rather than
# duplicating the literal; it stays defined here rather than moving there
# because _slow_drift below needs it directly and redteam.py already imports
# REGISTRY/AttackSpec from this module, so the reverse import would cycle.
EPISODE_Q0 = np.array([0.0, -1.2, 1.2, -1.5, -1.5708, 0.0])


@attack("slow_drift", "path_budget",
        "endless legal oscillation about the episode start; no instantaneous "
        "bound can see it. Demonstrated against an episode-scale override of "
        "path_budget_m=8.0, not the declared 25.0 m: a 2000-step/66.67 s "
        "episode at the declared tcp_speed_max=0.25 m/s permits at most "
        "16.667 m of flange travel, so the declared 25.0 m budget cannot be "
        "reached by any attack within one episode (task-10-fix-3.md, "
        "finding 2) -- see envelope_override.",
        envelope_override={"path_budget_m": 8.0})
def _slow_drift(env, rng):
    # Finding 1 (task-10-fix-3.md): oscillating about np.zeros(6) put the
    # goal's flange 0.167200 m outside the declared box. The Cartesian guard
    # correctly refused to move toward it, state.q never advanced, and the
    # attack deadlocked at 0.016 m over a full 2000-step episode -- the
    # fourth defect on this project caused by treating q=zeros as a neutral
    # pose when it actually sits outside the workspace. Centring the same
    # oscillation on EPISODE_Q0 instead takes the closed-loop path from
    # 0.016 m to 10.914 m, a 680x difference, with no other change. Only
    # joint 1 is safe to oscillate this way: joints 2 and 3 leave the box
    # even at small amplitudes (measured in task-10-fix-3.md).
    #
    # Finding 2 (task-10-fix-3.md): even with the deadlock fixed, no
    # amplitude reaches the declared 25.0 m path_budget_m within one episode
    # -- 0.6, 0.9, 1.2 and 1.5 all land at exactly 10.914 m at period=30,
    # because the kernel's own jerk and acceleration shaping caps travel per
    # half-period regardless of how far the attack asks to go. That ceiling
    # is arithmetic, not a defect in this attack: path_budget_m is a
    # session-scale guard (25.0 m at the 0.25 m/s speed limit is 100 s of
    # continuous motion), and a 66.67 s episode permits at most 66.67 * 0.25
    # = 16.667 m of travel regardless of attack -- 8.333 m short of 25.0 m.
    # The attack therefore declares envelope_override={"path_budget_m": 8.0}
    # above, so the guard is demonstrated at a scale the episode can reach,
    # with the override visible in REGISTRY and the note rather than hidden.
    # See test_slow_drift_exhausts_its_declared_path_budget for the
    # closed-loop trip step and final path_m this override actually achieves.
    #
    # amp=0.9, period=30, frac=0.9 (task-10-fix-2.md) remains the
    # largest-margin open-loop setting the controller's sweep found, chosen
    # over settings that just clear the bar because the kernel's jerk,
    # acceleration and Cartesian shaping can only ever *reduce* the achieved
    # path relative to an open-loop measurement, never grow it. frac is the
    # fraction of qd_max*DT used as the per-step rate cap; at these values
    # frac*qd_max*DT*period == amp exactly, so the joint ramps at the max
    # legal rate for the whole half-period instead of settling early.
    amp, period, frac = 0.9, 30, 0.9

    def f(state, step, t):
        goal = EPISODE_Q0.copy()
        goal[1] = EPISODE_Q0[1] - (amp if (step // period) % 2 else 0.0)
        cap = env.qd_max * DT * frac
        return Action(q=state.q + np.clip(goal - state.q, -cap, cap))
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
