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

from sentinel.envelope import DATASHEET_TCP_SPEED_M_S, Box, Envelope
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


# Task 6 correction / task-6-7-resume.md: these tests are unit tests for the
# joint-space shaping chain (jerk, acceleration, velocity, the braking
# bound), and they ramp every joint toward +-pi to exercise the position
# limit and the braking bound. q = +-pi on every joint is inside the
# declared tcp_box, but the path there is not, so once Task 6 added the
# Cartesian guard it stopped the ramp long before any joint limit was
# approached and these tests could no longer do their job -- five of them
# failed, all from the same cause (q = zeros starts 0.1672 m outside the
# declared box; see task-6-7-resume.md for the controller's measurement).
#
# Deviation from task-6-7-resume.md (see task-6-7-report.md for the full
# investigation): the note's prescribed fix -- give `run()` an unbounded
# tcp_box and keep going through `k.filter(...)` -- is not sufficient.
# _cartesian also enforces tcp_speed_max, and q = zeros is a near-maximal
# lever (tcp radius 0.8497 m, close to the full 0.85 m reach). Measured with
# only the box widened: the Cartesian guard still bound on every one of a
# 5-step ramp's steps at the default 0.25 m/s speed limit. Widening
# tcp_speed_max to the UR5e datasheet ceiling (1.0 m/s, the most this
# Envelope will accept -- see envelope.py's own validation) is not enough
# either: with all six joints ramping toward the same full-scale target
# simultaneously from a near-maximal lever, measured combined TCP speed
# reaches ~1.19 m/s during the early acceleration phase, still over even
# that ceiling, for roughly the first 45-50 of a 400-step run. A bisected
# step is a position discontinuity a finite-difference reconstruction of
# jerk cannot tell apart from a real violation, so any test asserting a
# bound over the *whole* trajectory (not just its tail, and not just "did
# this rule fire somewhere") breaks.
#
# _shape has no dependency on tcp_box or tcp_speed_max at all -- it only
# reads the q/qd/qdd/qddd limits and brake_headroom -- so `run()` now calls
# it directly and never invokes `_cartesian`. That is what "isolate the
# joint-space shaping chain" means literally, and it removes any doubt about
# whether a given assertion happens to be insensitive to the interference
# rather than genuinely unaffected by it. The Cartesian guard has its own
# tests in test_kernel_cartesian.py.
def _unbounded_box_env():
    return Envelope.ur5e_declared().replace(
        tcp_box=Box(lo=[-10.0, -10.0, -10.0], hi=[10.0, 10.0, 10.0]),
        tcp_speed_max=DATASHEET_TCP_SPEED_M_S)


def run(actions, q0=None, env=None):
    """Drive the shaping chain alone with a list of targets, calling _shape
    directly so _cartesian is never in the loop. Returns the emitted
    commands."""
    env = env or _unbounded_box_env()
    k = SafetyKernel(env)
    q = np.zeros(6) if q0 is None else np.asarray(q0, dtype=float)
    out = []
    for a in actions:
        q, _ = k._shape(q, np.asarray(a, dtype=float), DT)
        out.append(q.copy())
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
    # Fix round 1 (task-5-fix-1.md): the bound is a statement about the room
    # the controller HAD when it sized the step, not the room left once the
    # step landed. Comparing against the latter is unsatisfiable: on the step
    # that first touches the limit the remaining room is zero, so the
    # permitted velocity is zero, while the step still has to cover the last
    # of the distance. Controller-measured against the real shipped kernel:
    # room-after gives 36/2394 violations (unsatisfiable by construction),
    # room-before gives 0/2394.
    cmds, _, env = run([np.full(6, 10.0)] * 400)
    qd = np.diff(cmds, axis=0) / DT
    room_before = env.q_max - cmds[:-1]
    allowed = np.sqrt(2 * env.qdd_max * np.maximum(room_before, 0.0))
    assert np.all(qd <= allowed + 1e-6)


def test_an_already_legal_action_passes_through_unchanged():
    env = _unbounded_box_env()
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
    # Every-step full-reversal targets are the most adversarial input to
    # _shape in this file (verified during development: even at the
    # datasheet-max 1.0 m/s tcp_speed_max, going through _cartesian bound 15
    # of 200 steps and produced an apparent max|qddd| = 245.9 against the
    # 100.0 limit purely from the resulting position discontinuities, even
    # though _shape's own bookkeeping never exceeds it -- run() now calls
    # _shape directly for exactly this reason; see its docstring).
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


def test_brake_accel_violation_fires_during_the_braking_transition():
    # Fix 2 (task-5-fix-1.md): the acceleration-level braking clamp (qdd_want
    # narrowed to qdd_target via qdd_hi/qdd_lo, the same shape as the
    # position-level clamp one level down) used to shape the command
    # silently: no Violation of its own, unlike its position-level
    # counterpart which reports "brake". This project's claim rests on every
    # clamp being auditable from the journal, so a guard that shapes a
    # command without saying so cannot be audited after the fact. Confirm
    # "brake_accel" now fires somewhere in the same 400-step approach that
    # exercises the position-level "brake" clamp -- it engages early, while
    # velocity is still ramping toward qd_max, not near the wall.
    #
    # Calls _shape directly (see run()'s docstring for why): going through
    # filter()/_cartesian for a 400-step full-scale ramp from q = zeros lets
    # the Cartesian speed guard bind during the early acceleration phase
    # (measured ~1.19 m/s against even the 1.0 m/s datasheet ceiling), which
    # would mix "tcp_speed" violations into rules_seen and, via the
    # resynchronisation after every bisected step, drive this test's
    # trajectory away from the one _shape alone would have produced.
    env = _unbounded_box_env()
    k = SafetyKernel(env)
    q = np.zeros(6)
    rules_seen = set()
    for _ in range(400):
        q, viols = k._shape(q, np.full(6, 10.0), DT)
        rules_seen.update(v.rule for v in viols)
    assert "brake_accel" in rules_seen
    assert "brake" in rules_seen  # the position-level clamp still fires too


# Task 9 correction section 3 / progress.md Ruling R31: _brake_bound's room/dt
# term keeps the final np.clip(q_ref + qd*dt, q_min, q_max) from ever biting
# across this whole suite and all fifteen catalogued attacks -- but it is not
# a universal guarantee, and tests/test_properties.py's hypothesis strategies
# are exactly the kind of input that finds the gap: a start close to a joint
# limit, carrying real velocity, driven by a fresh unconstrained target every
# step. These two tests are the wiring check for
# telemetry["position_clip_engaged"] (added to _shape/filter for exactly this
# task): False during ordinary operation, True on the one step that actually
# clips.
def test_position_clip_engaged_is_false_during_an_ordinary_ramp():
    cmds, k, env = run([np.full(6, 10.0)] * 5)
    assert k._position_clip_engaged is False


def test_position_clip_engaged_is_true_exactly_on_the_step_that_bites():
    # Hand-searched, deterministic instance of the "stoppable start +
    # random-walk target" family that tests/test_properties.py drives at
    # scale (seed 199), verified against the real kernel to trip the clip on
    # its 14th _shape call and nowhere earlier. Joint 5 lands exactly on
    # q_max, having consumed its last ~5.6 mrad of room in the clip -- the
    # same "about 5 mm" order of magnitude task-9-correction.md measured.
    env = Envelope.ur5e_declared()
    rng = np.random.default_rng(199)
    q = rng.uniform(-2.8, 2.8, 6)
    room = np.minimum(env.q_max - q, q - env.q_min)
    cap = np.minimum(env.qd_max, np.sqrt(2.0 * env.qdd_max * np.maximum(room, 0.0)))
    qd0 = rng.uniform(-1.0, 1.0, 6) * cap

    k = SafetyKernel(env)
    k._qd_cmd = qd0.copy()          # seed the derivative state directly;
    for i in range(14):             # _shape has no other way to accept qd0
        target = rng.uniform(-50, 50, 6)
        q, _ = k._shape(q, target, DT)
        if i < 13:
            assert k._position_clip_engaged is False, f"clip engaged early, at step {i}"
    assert k._position_clip_engaged is True, "expected this scenario to trip the clip by call 14"
    assert q[5] == pytest.approx(env.q_max[5], abs=1e-9)
