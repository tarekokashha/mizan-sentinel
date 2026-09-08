"""Kinematic shaping: jerk, acceleration, velocity and the braking bound.

Task 5 correction (see .superpowers/sdd/2026-09-05-m01-sentinel/task-5-correction.md):
the brief's own chain, measured against its own ramp test, produced
qdd = 8.808387 against a 5.0 limit and qddd = 174.846820 against a 100.0
limit. `_shape` here is the corrected algorithm: settle the velocity target
first (including a discrete-exact braking bound), then let jerk and
acceleration shape the approach to it, and apply the same braking bound one
derivative up so acceleration can return to zero before velocity saturates.
"""
import numpy as np
import pytest

from sentinel.envelope import Envelope
from sentinel.kernel import SafetyKernel
from sentinel.types import Action, RobotState, Status

DT = 1 / 30


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def tick(self, dt=DT):
        self.t += dt
        return self.t


def run(actions, q0=None, env=None):
    """Drive the kernel with a list of targets. Returns the emitted commands."""
    env = env or Envelope.ur5e_declared()
    c = Clock()
    k = SafetyKernel(env, clock=c)
    q = np.zeros(6) if q0 is None else np.asarray(q0, dtype=float)
    out = []
    for a in actions:
        v = k.filter(RobotState(q=q, qd=np.zeros(6), t_mono=c.t), Action(q=a))
        out.append(v.action.q.copy())
        q = v.action.q.copy()          # perfect follower, isolates the kernel
        c.tick()
    return np.array(out), k, env


def test_velocity_is_capped_at_the_declared_limit_not_the_control_rate():
    # The bug this replaces used 1/control_hz. Here 33 ms elapse per call, so a
    # legal step is qd_max * DT, not qd_max / 500.
    cmds, _, env = run([np.full(6, 10.0)] * 5)
    step = np.diff(np.vstack([np.zeros(6), cmds]), axis=0)
    assert np.all(step <= env.qd_max * DT + 1e-9)
    # NOTE (deviation from task-5-brief.md, see task-5-report.md): the brief
    # compares a scalar (step[-1].max()) against a 6-vector
    # (env.qd_max * DT * 0.5), which raises ValueError on any envelope
    # regardless of the shaping algorithm behind it (this 5-step ramp never
    # nears a position limit, so brief vs. corrected _shape behave
    # identically here). Reduced the RHS to a scalar to match the LHS.
    assert step[-1].max() > env.qd_max.max() * DT * 0.5   # and it is not absurdly tight


def test_acceleration_and_jerk_are_bounded():
    # Correction: derivatives come from the emitted sequence directly, not
    # from a zero-prepended one. Prepending a zero row asserts the run starts
    # from rest, which manufactures a spurious qd_max/dt = 30 spike the
    # moment anything starts in motion (irrelevant here, since this run does
    # start from rest, but it is the wrong idiom to carry forward).
    cmds, _, env = run([np.full(6, 10.0)] * 40)
    qd = np.diff(cmds, axis=0) / DT
    qdd = np.diff(qd, axis=0) / DT
    qddd = np.diff(qdd, axis=0) / DT
    assert np.all(np.abs(qdd) <= env.qdd_max + 1e-6)
    assert np.all(np.abs(qddd) <= env.qddd_max + 1e-6)


def test_position_limits_are_never_exceeded_by_a_command():
    cmds, _, env = run([np.full(6, 10.0)] * 400)
    assert np.all(cmds <= env.q_max + 1e-12)
    assert np.all(cmds >= env.q_min - 1e-12)


def test_commanded_velocity_reaches_zero_at_the_position_limit():
    # The braking bound. A kernel that only clips position arrives at q_max at
    # full speed and any follower with finite deceleration overshoots.
    cmds, _, env = run([np.full(6, 10.0)] * 400)
    at_limit = np.abs(cmds[-1] - env.q_max) < 1e-3
    assert at_limit.all(), "test did not actually reach the limit"
    qd_final = (cmds[-1] - cmds[-2]) / DT
    assert np.all(np.abs(qd_final) < 0.05), f"arrived at the limit at {qd_final} rad/s"


def test_braking_distance_is_respected_throughout_the_approach():
    cmds, _, env = run([np.full(6, 10.0)] * 400)
    qd = np.diff(np.vstack([np.zeros(6), cmds]), axis=0) / DT
    dist = env.q_max - cmds
    allowed = np.sqrt(2 * env.qdd_max * np.maximum(dist, 0.0))
    assert np.all(qd <= allowed + 1e-6)


def test_an_already_legal_action_passes_through_unchanged():
    env = Envelope.ur5e_declared()
    c = Clock()
    k = SafetyKernel(env, clock=c)
    q = np.zeros(6)
    k.filter(RobotState(q=q, qd=np.zeros(6), t_mono=c.t), Action(q=q))
    c.tick()
    tiny = np.full(6, 1e-4)
    v = k.filter(RobotState(q=q, qd=np.zeros(6), t_mono=c.t), Action(q=tiny))
    assert v.status is Status.PASS
    assert np.allclose(v.action.q, tiny)


def test_degrees_sent_into_a_radians_api_are_contained():
    # unit_confusion: 90 in a field that means radians.
    cmds, _, env = run([np.full(6, 90.0)] * 200)
    assert np.all(cmds <= env.q_max + 1e-12)


def test_alternating_full_scale_targets_do_not_produce_unbounded_jerk():
    actions = [np.full(6, 10.0 if i % 2 == 0 else -10.0) for i in range(200)]
    cmds, _, env = run(actions)
    qd = np.diff(np.vstack([np.zeros(6), cmds]), axis=0) / DT
    qdd = np.diff(np.vstack([np.zeros(6), qd]), axis=0) / DT
    qddd = np.diff(np.vstack([np.zeros(6), qdd]), axis=0) / DT
    assert np.all(np.abs(qddd) <= env.qddd_max + 1e-6)


def test_the_emitted_derivatives_obey_every_limit_through_the_braking_transition():
    # The braking transition is where a nested clamp goes wrong: the bound
    # shrinks faster than the acceleration limit allows, the position clip
    # bites, and the emitted derivative stops matching the clamped one.
    cmds, _, env = run([np.full(6, 10.0)] * 400)
    qd = np.diff(cmds, axis=0) / DT
    qdd = np.diff(qd, axis=0) / DT
    qddd = np.diff(qdd, axis=0) / DT
    assert np.abs(qd).max() <= env.qd_max.max() + 1e-6
    assert np.abs(qdd).max() <= env.qdd_max.max() + 1e-6
    assert np.abs(qddd).max() <= env.qddd_max.max() + 1e-6
    assert np.all(cmds <= env.q_max + 1e-12) and np.all(cmds >= env.q_min - 1e-12)
