"""Property-based tests for the safety kernel's headline invariants.

These drive the fully assembled kernel (`SafetyKernel.filter`, which chains
`_shape`, `_cartesian` and the gripper guard behind all ten guards) with
`hypothesis`-generated adversarial input. Every other test in this project
exercises one guard or one component in isolation; this file is the first
that exercises the composition, because a controller measurement taken
against a component alone has already twice turned out not to survive
composition with a guard that did not exist yet when the measurement was
made.

`.superpowers/sdd/2026-09-05-m01-sentinel/task-9-correction.md` overrides the
brief in three places (the correction is authoritative where they differ):

1. Invariant (A) only holds from a *stoppable* initial state -- see
   `stoppable_states()`. A joint already closer to its limit than its own
   velocity can brake within (at qdd_max) is committed to an overshoot before
   the kernel is ever called; no acceleration-limited controller can rescue
   that state, so the property does not assert over it.
   `test_an_unstoppable_start_is_reported_and_not_silently_accepted` documents
   the excluded region explicitly, in code, rather than only in prose.

2. Derivatives are taken from the emitted sequence directly
   (`np.diff(cmds, axis=0)`), never from a zero-prepended one. Prepending
   asserts the run starts from rest, which the stoppable-start strategy
   deliberately does not guarantee, and manufactures a spurious first-sample
   spike of qd_max/dt = 30 that looks exactly like a real acceleration
   violation and is not one.

3. The position clip inside `_shape`'s final
   `np.clip(q_ref + qd*dt, q_min, q_max)` DOES bite under adversarial input --
   measured in the correction at 361 times across 200 random-walk trials,
   worst magnitude about 5 mm. Position safety (`q_min <= q <= q_max`) is
   unconditional either way and is asserted as such, alone, in
   `test_invariant_A_position_is_never_violated`. What narrows on a clipped
   step is the *derivative* guarantee: the realised qd is smaller than the qd
   the shaping chain targeted, so the realised qdd on that transition can
   exceed qdd_max. `test_invariant_A_derivatives_hold_except_where_the_position_clip_engages`
   states this precondition explicitly -- it excludes only the transitions the
   kernel itself flags via `telemetry["position_clip_engaged"]`, and reports
   the observed rate with `note()` rather than silently discarding it.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, note, settings
from hypothesis import strategies as st

from sentinel.envelope import DATASHEET_TCP_SPEED_M_S, Box, Envelope
from sentinel.kernel import SafetyKernel
from sentinel.kinematics import tcp_position
from sentinel.sim import SimPlant
from sentinel.types import Action, RobotState, Status

DT = 1 / 30
ENV = Envelope.ur5e_declared()

# deliberately vicious: huge magnitudes, NaN, inf, zeros, subnormals
wild = st.one_of(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.just(float("nan")), st.just(float("inf")), st.just(float("-inf")),
    st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False),
)
action_vec = st.lists(wild, min_size=6, max_size=6).map(np.array)
sequences = st.lists(action_vec, min_size=1, max_size=120)


@st.composite
def stoppable_states(draw):
    """Initial (q, qd) pairs the kernel can actually be held responsible for.

    A joint closer to its limit than its own velocity can brake within (at
    qdd_max) is already committed to an overshoot before the kernel is ever
    called. That is a state outside the envelope's premise, not a kernel
    failure -- see task-9-correction.md section 1 and
    test_an_unstoppable_start_is_reported_and_not_silently_accepted below for
    the excluded region, documented explicitly rather than only in prose.
    """
    q = np.array(draw(st.lists(
        st.floats(min_value=-2.8, max_value=2.8, allow_nan=False, allow_infinity=False),
        min_size=6, max_size=6)))
    room = np.minimum(ENV.q_max - q, q - ENV.q_min)
    cap = np.minimum(ENV.qd_max, np.sqrt(2.0 * ENV.qdd_max * np.maximum(room, 0.0)))
    frac = np.array(draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=6, max_size=6)))
    return q, frac * cap


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def tick(self, dt=DT):
        self.t += dt


def _drive(seq, start):
    """Run one kernel + plant pair over `seq`, seeded at (q0, qd0) = start.

    The anchor call that establishes the kernel's own reference happens here,
    before the loop, and is not yielded: every property below reads the
    kernel's SHAPED response to a real target, not the fixed `first_call`
    HOLD that `filter()` always returns for the very first observation it
    ever sees (see its "first call establishes the reference" branch). The
    plant is seeded with qd0 directly, since `SimPlant.__init__` has no
    constructor hook for an initial velocity and otherwise always starts at
    rest -- task-9-correction.md section 1 is explicit that the run must be
    driven from (q0, qd0), not from rest, since a from-rest start is a
    strictly easier case than the one a deployed kernel actually faces.
    """
    q0, qd0 = start
    c = Clock()
    k = SafetyKernel(ENV, clock=c)
    plant = SimPlant(q0=q0, qdd_max=ENV.qdd_max)
    plant.qd = np.array(qd0, dtype=float).copy()
    k.filter(plant.state(t_mono=c.t), Action(q=q0))
    c.tick()
    for a in seq:
        v = k.filter(plant.state(t_mono=c.t), Action(q=a))
        plant.step(v.action.q, DT)
        c.tick()
        yield v, plant


# --------------------------------------------------------------------------- #
# invariant (A): the kernel's own emitted command
# --------------------------------------------------------------------------- #
@settings(max_examples=250, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seq=sequences, start=stoppable_states())
def test_invariant_A_position_is_never_violated(seq, start):
    # Unconditional. This is the claim the programme makes: whatever the
    # kernel emits, on every single call, satisfies q_min <= q <= q_max and is
    # finite. Nothing here is excused by status, by clipping, or by anything
    # else -- see the module docstring, point 3.
    for v, _plant in _drive(seq, start):
        q = v.action.q
        assert np.all(np.isfinite(q)), "a non-finite command escaped the kernel"
        assert np.all(q <= ENV.q_max + 1e-9)
        assert np.all(q >= ENV.q_min - 1e-9)
        if v.action.gripper is not None:
            assert ENV.grip_min - 1e-9 <= v.action.gripper <= ENV.grip_max + 1e-9


@settings(max_examples=150, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seq=sequences, start=stoppable_states())
def test_invariant_A_derivatives_hold_except_where_the_position_clip_engages(seq, start):
    """The derivative guarantee is conditional, and the condition is stated,
    not hidden -- see the module docstring, point 3.

    On the one step where the final position clip inside `_shape` engages,
    the realised velocity is smaller than the one the shaping chain targeted,
    so the realised acceleration on the transition INTO that step can exceed
    qdd_max. Position stays safe regardless (see
    test_invariant_A_position_is_never_violated above) -- this is a
    documented narrowing of the derivative guarantee, not a defect. Velocity
    itself is never at risk from a clip: the position-level clip can only
    move the command closer to the previous one, never further, so it can
    only shrink |qd|, never grow it past qd_max. Only the qdd window whose
    LAST endpoint is the clipped command is excluded, both here and by
    construction, since the kernel resynchronises its own internal
    `_qd_cmd`/`_qdd_cmd` to the realised (clipped) value at the end of every
    `_shape` call -- the very next transition is shaped fresh relative to
    that real value and is bound by qdd_max again.

    Status.STOP is skipped exactly as the brief's own text does, because a
    STOP snaps the command to the arm's measured position rather than
    continuing the shaped trajectory -- a deliberate discontinuity, not a
    shaping failure, and permanent once latched, so only the single step
    that first trips it needs excluding. Status.HOLD is deliberately NOT
    excluded here, unlike STOP: a census of ~90000 filter() calls across
    3000 independent stoppable-start random-walk trials (see the Task 9
    report) found zero HOLD verdicts anywhere in this scenario space --
    `_drive` never configures contact (so force/torque/budget guards cannot
    fire), the clock always advances by exactly DT (so dt_max/staleness
    cannot fire), and the plant's own servo authority (1.5x qdd_max) tracks
    the kernel's already-smooth commands well inside tracking_tol_rad. So
    there is no HOLD-induced jump to hide here; if hypothesis ever finds one,
    excluding only STOP will fail loudly rather than mask it, which is the
    point of leaving it out.
    """
    steps = list(_drive(seq, start))
    clipped_steps = sum(bool(v.telemetry.get("position_clip_engaged", False))
                         for v, _plant in steps)
    total_steps = len(steps)

    # Prepend the TRUE anchor position (start[0] = q0), not a zero row: q0 is
    # the real q_ref the kernel's first real _shape call used (the anchor
    # call sets self._q_cmd = q0 and touches self._qd_cmd/self._qdd_cmd not
    # at all, so they are genuinely 0 going into that first call regardless
    # of qd0 -- see _drive's docstring). This makes the first entry of `qd`
    # below exactly the first call's own qd_real, the same quantity the
    # kernel itself computed and bounded, so the SECOND transition's qdd has
    # a real velocity to difference against instead of an assumed one. This
    # is the fix for exactly the mistake task-9-correction.md section 2 warns
    # about, just reachable a different way: an *implicit* zero-velocity
    # prepend, hidden inside a running "prev_qd" variable that never gets
    # its first real update, rather than an explicit np.vstack. hypothesis
    # found it (seq=[zeros, zeros], start=(q0=[0,0,-2,0,0,0], qd0=zeros)):
    # the manufactured qdd was 8.33 against the 5.0 limit on the very
    # transition into the second call, computed against a false qd=0
    # baseline instead of the real ~0.11 rad/s the first call actually
    # produced. See the Task 9 report for the full trace.
    q0, _qd0 = start
    cmds = [np.asarray(q0, dtype=float)] + [v.action.q for v, _plant in steps]
    statuses = [None] + [v.status for v, _plant in steps]          # None: the
    clip_flags = [False] + [bool(v.telemetry.get("position_clip_engaged", False))
                             for v, _plant in steps]                # anchor is
    # neither a STOP nor a clip event -- it is just the starting reference.

    prev = None
    prev_qd = None
    for i in range(len(cmds)):
        q = cmds[i]
        if prev is not None and statuses[i] is not Status.STOP:
            qd = (q - prev) / DT
            assert np.all(np.abs(qd) <= ENV.qd_max + 1e-6)
            if prev_qd is not None:
                qdd = (qd - prev_qd) / DT
                if not clip_flags[i]:
                    assert np.all(np.abs(qdd) <= ENV.qdd_max + 1e-6)
            prev_qd = qd
        else:
            prev_qd = None    # a STOP breaks the chain; do not compare across it
        prev = q

    note(f"position clip engaged on {clipped_steps} of {total_steps} steps "
         f"({clipped_steps / total_steps:.4%})")


# --------------------------------------------------------------------------- #
# the excluded region, documented in code rather than only in prose
# --------------------------------------------------------------------------- #
def test_an_unstoppable_start_is_reported_and_not_silently_accepted():
    # Outside the invariant's premise: 0.99 rad/s with 0.05 rad of room needs
    # 9.8 rad/s^2 against a 5.0 limit. The kernel cannot prevent the overshoot,
    # but it must brake at its limit and say so rather than pretend.
    c = Clock()
    k = SafetyKernel(ENV, clock=c)
    q0 = np.full(6, np.pi - 0.05)
    st0 = RobotState(q=q0, qd=np.full(6, 0.99), t_mono=c.t)
    k.filter(st0, Action(q=q0))
    c.tick()
    v = k.filter(RobotState(q=q0, qd=np.full(6, 0.99), t_mono=c.t), Action(q=np.full(6, 10.0)))
    assert v.action.q.max() <= ENV.q_max.max() + 1e-9      # the command still obeys
    assert v.status is not Status.PASS                      # and it is not silent


# --------------------------------------------------------------------------- #
# invariant (B): the physical plant, not just the kernel's commanded number
# --------------------------------------------------------------------------- #
# FINDING, reported DONE_WITH_CONCERNS (see task-9-report.md): as specified
# (stoppable_states() for `start`, per task-9-correction.md section 1's
# explicit instruction to use it "for the initial state in all three
# property tests"), this property is false for a correct kernel, for a
# reason structurally analogous to -- but mechanistically distinct from --
# the joint-limit stoppability gap section 1 already closed for invariant A.
#
# stoppable_states() bounds q0/qd0 relative to JOINT-space room (distance to
# q_max/q_min). It has no relationship at all to the CARTESIAN tcp_box this
# property makes a claim about. Two distinct counterexamples, both minimised
# and captured as permanent regression tests in test_kernel_cartesian.py
# (test_a_plant_that_starts_outside_the_box_is_already_past_the_margin and
# test_a_joint_stoppable_velocity_can_still_carry_the_plant_past_the_cartesian_margin):
#
#   1. Trivial: hypothesis shrank straight to q0 = zeros, qd0 = zeros,
#      seq = [<a single NaN-laced action>]. tcp_position(zeros) sits 0.1672 m
#      outside the declared box (progress.md Ruling R34) -- the plant is
#      *born* past the margin, before the kernel is ever called, regardless
#      of what it then does.
#
#   2. Deeper: restricting the search to starts already inside the box
#      (`assume(ENV.tcp_box.excursion(tcp_position(q0)) <= ENV.plant_margin_m)`)
#      still finds a failure. A joint-stoppable qd0 (satisfying
#      stoppable_states()'s own qd**2 <= 2*qdd_max*room formula exactly) can
#      carry the TCP past the Cartesian margin via the Jacobian's lever-arm
#      effect at a configuration far from the base axis, purely from the
#      plant's own physical inertia settling toward a kernel command that is
#      itself correct (status PASS, commanded q == q0 exactly -- see the
#      regression test for the full numbers: 16 mm of excursion from a
#      qd_max-legal, joint-stoppable start).
#
# Ruling: property stated too strongly, not a kernel defect -- neither
# counterexample involves the kernel commanding anything unsafe; both are
# about what stoppable_states() fails to constrain. Fixing this properly
# needs either (a) a Cartesian-aware precondition analogous to
# stoppable_states() (nontrivial: it would need the Jacobian-scaled velocity
# margin against tcp_box, not a closed-form per-joint formula like the
# existing one), or (b) documenting it as a second stated narrowing of
# invariant B in LIMITATIONS.md, next to stoppability. Per this task's brief
# ("if a bound genuinely cannot hold, that is a specification change: report
# DONE_WITH_CONCERNS... and let the controller rule on it"), that choice is
# not made here. The property itself is left exactly as specified --
# unmodified strategy, unmodified assertions -- and marked xfail(strict=True)
# so it stays honest: a silent pass here would mean the finding was fixed out
# from under this ruling without anyone noticing.
@pytest.mark.xfail(
    strict=True,
    reason="stoppable_states() has no Cartesian precondition; see the "
           "comment above and task-9-report.md for the two counterexamples "
           "and the DONE_WITH_CONCERNS ruling awaiting controller review.")
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seq=sequences, start=stoppable_states())
def test_invariant_B_the_plant_stays_within_the_declared_margin(seq, start):
    for _v, plant in _drive(seq, start):
        assert np.all(plant.q <= ENV.q_max + ENV.plant_margin_rad)
        assert np.all(plant.q >= ENV.q_min - ENV.plant_margin_rad)
        assert ENV.tcp_box.excursion(tcp_position(plant.q)) <= ENV.plant_margin_m


# --------------------------------------------------------------------------- #
# metamorphic properties (brief, unchanged by the correction)
# --------------------------------------------------------------------------- #
def _unbounded_box_env():
    # Matches tests/test_kernel_shaping.py's own helper of the same name and
    # for the same reason (see its module docstring / progress.md Ruling
    # R34): q = zeros sits 0.1672 m outside the declared tcp_box, so the
    # Cartesian guard refuses to move it at all once the box is the real,
    # declared one. These two metamorphic properties are claims about the
    # JOINT-SPACE shaping chain specifically (identity on an already-legal
    # step; monotonicity under looser derivative limits) and have nothing to
    # do with the Cartesian guard, so they isolate it exactly the way the
    # existing joint-space shaping tests already do.
    return Envelope.ur5e_declared().replace(
        tcp_box=Box(lo=[-10.0, -10.0, -10.0], hi=[10.0, 10.0, 10.0]),
        tcp_speed_max=DATASHEET_TCP_SPEED_M_S)


def test_metamorphic_a_legal_action_is_the_identity():
    env = _unbounded_box_env()
    c = Clock()
    k = SafetyKernel(env, clock=c)
    q = np.zeros(6)
    k.filter(RobotState(q=q, qd=np.zeros(6), t_mono=c.t), Action(q=q))
    c.tick()
    for _ in range(50):
        target = q + 1e-4
        v = k.filter(RobotState(q=q, qd=np.zeros(6), t_mono=c.t), Action(q=target))
        assert v.status is Status.PASS
        assert np.allclose(v.action.q, target, atol=1e-12)
        q = v.action.q.copy()
        c.tick()


def test_metamorphic_a_looser_envelope_never_clamps_more():
    tight = _unbounded_box_env()
    loose = tight.replace(qd_max=tight.qd_max * 2, qdd_max=tight.qdd_max * 2,
                          qddd_max=tight.qddd_max * 2)
    rng = np.random.default_rng(0)
    seq = [rng.uniform(-2.0, 2.0, 6) for _ in range(80)]
    steps = {}
    for name, env in (("tight", tight), ("loose", loose)):
        c = Clock()
        k = SafetyKernel(env, clock=c)
        q = np.zeros(6)
        total = 0.0
        for a in seq:
            v = k.filter(RobotState(q=q, qd=np.zeros(6), t_mono=c.t), Action(q=a))
            total += float(np.abs(v.action.q - q).sum())
            q = v.action.q.copy()
            c.tick()
        steps[name] = total
    assert steps["loose"] >= steps["tight"] - 1e-9
