import numpy as np
import pytest

from sentinel.envelope import Envelope
from sentinel.kernel import SafetyKernel
from sentinel.kinematics import tcp_position
from sentinel.types import Action, RobotState

DT = 1 / 30


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def tick(self, dt=DT):
        self.t += dt


def run(targets, q0, env=None, steps=None):
    env = env or Envelope.ur5e_declared()
    c = Clock()
    k = SafetyKernel(env, clock=c)
    q = np.asarray(q0, dtype=float)
    out = []
    seq = targets if steps is None else [targets] * steps
    for a in seq:
        v = k.filter(RobotState(q=q, qd=np.zeros(6), t_mono=c.t), Action(q=a))
        q = v.action.q.copy()
        out.append(q)
        c.tick()
    return np.array(out), env


def test_the_tcp_never_leaves_the_declared_box():
    # An arm folded up inside the box, told to extend straight out past it.
    q0 = np.array([0.0, -1.6, 1.6, -1.6, -1.57, 0.0])
    cmds, env = run(np.array([0.0, 0.0, 0.0, 0.0, -1.57, 0.0]), q0, steps=400)
    for q in cmds:
        assert env.tcp_box.excursion(tcp_position(q)) == 0.0


def test_tcp_speed_stays_within_the_declared_limit():
    q0 = np.array([0.0, -1.6, 1.6, -1.6, -1.57, 0.0])
    cmds, env = run(np.array([3.0, -1.6, 1.6, -1.6, -1.57, 0.0]), q0, steps=200)
    p = np.array([tcp_position(q) for q in cmds])
    speed = np.linalg.norm(np.diff(p, axis=0), axis=1) / DT
    assert speed.max() <= env.tcp_speed_max + 1e-6


# Task 6 correction (see .superpowers/sdd/2026-09-05-m01-sentinel/task-6-correction.md):
# the brief's original start state for this test was q0 = zeros, called "arm
# straight out". Measured: tcp([0,0,0,0,0,0]) = [-0.8172, -0.2329, 0.0628],
# x = -0.8172 against a box lower bound of -0.65 -- the flange starts OUTSIDE
# the declared box, so the test failed at step 0 and proved nothing about the
# speed guard. A pose maximising the lever at a single point is also wrong.
# because the test sweeps joint 0, tracing a circle of that radius about the
# base axis, and the circle must fit inside the box for the WHOLE sweep, not
# just the starting pose. Use the final measured posture (also used by
# Task 10) so the two stay in step:
LEVER_Q0 = np.array([0.0, -0.56, 0.32, 0.35, -1.5708, 0.0])
# Verified over a 289-point sweep of joint 0 across the full +-pi range, and
# independently re-derived by a reviewer against this repo's own kinematics:
#   minimum sustained lever = 0.6450233005863028 m
#   maximum                 = 0.6450233005863032 m, constant to 11 figures,
#                             which confirms the circle-of-constant-radius premise
#   worst box excursion     = exactly 0.0, so containment asserts == 0.0 with no
#                             borrowed tolerance
#   flange z constant at 0.3933176, well inside the 0.05 to 0.90 box
#   flange at base angle 0  = [-0.63109925, -0.13329963, 0.3933176]
# At the declared qd_max of 1.0 rad/s that is 0.645023 m/s of flange speed
# against a 0.25 m/s limit, a ratio of 2.58, while every joint-space limit is
# respected exactly. Only the Cartesian speed guard can see it.


def test_lever_sprint_is_caught_by_the_cartesian_guard_alone():
    # Fully extended, sweeping the base at the legal joint speed. Every
    # joint-space limit is respected; TCP speed would be far over the limit.
    env = Envelope.ur5e_declared()
    q0 = LEVER_Q0
    assert env.tcp_box.contains(tcp_position(q0)), "the test must start inside the box"
    lever = float(np.linalg.norm(tcp_position(q0)[:2]))
    assert lever * float(env.qd_max[0]) > env.tcp_speed_max * 2, (
        "this configuration is not a lever long enough to be a test")
    cmds, _ = run(np.array([3.0, -0.56, 0.32, 0.35, -1.5708, 0.0]), q0, steps=200)
    p = np.array([tcp_position(q) for q in cmds])
    speed = np.linalg.norm(np.diff(p, axis=0), axis=1) / DT
    assert speed.max() <= env.tcp_speed_max + 1e-6
    # and the joint-space limits were never the binding constraint
    qd = np.abs(np.diff(cmds, axis=0)) / DT
    assert qd.max() < float(env.qd_max[0]) * 0.999


def test_a_step_that_is_already_inside_is_not_shortened():
    env = Envelope.ur5e_declared()
    q0 = np.array([0.0, -1.6, 1.6, -1.6, -1.57, 0.0])
    assert env.tcp_box.contains(tcp_position(q0))
    c = Clock()
    k = SafetyKernel(env, clock=c)
    k.filter(RobotState(q=q0, qd=np.zeros(6), t_mono=c.t), Action(q=q0))
    c.tick()
    target = q0 + 1e-4
    v = k.filter(RobotState(q=q0, qd=np.zeros(6), t_mono=c.t), Action(q=target))
    assert np.allclose(v.action.q, target, atol=1e-9)


# Latent gap closed per task-6-7-resume.md: `_cartesian` seeded lo = 0.0 as
# known-good and never tested it. When q_ref is already outside the box that
# assumption is false. q_ref here is LEVER_Q0 with joint 1 opened by 0.10 rad
# -- measured tcp x = -0.65098969, excursion 0.0009896940424202194 m (~0.99mm
# outside); q_cmd is LEVER_Q0 itself, 19mm inside. Measured against the
# unfixed guard (which never special-cased q_ref and just bisected anyway):
# it found lo = 0.1240115761756897 and moved the arm to the box surface, and
# its violation list didn't even mention "tcp_box" -- only "tcp_speed"
# (measured ~2.0151 m/s against p_bad = tcp(q_cmd), which is inside the box)
# -- because it judges q_cmd, not the fact q_ref already failed. Motion
# cannot fix an arm that is already outside, so the fixed guard must refuse
# to search at all and hold at q_ref exactly.
def test_a_kernel_that_starts_outside_the_box_holds_rather_than_moving():
    env = Envelope.ur5e_declared()
    c = Clock()
    k = SafetyKernel(env, clock=c)
    q_ref = LEVER_Q0.copy()
    q_ref[1] += 0.10
    assert not env.tcp_box.contains(tcp_position(q_ref)), "must start outside the box"
    q_cmd = LEVER_Q0.copy()
    assert env.tcp_box.contains(tcp_position(q_cmd)), "target must be inside the box"

    q_out, viols = k._cartesian(q_ref, q_cmd, DT)

    assert np.array_equal(q_out, q_ref), "must hold at q_ref, not search for a rescue"
    assert len(viols) == 1
    assert viols[0].rule == "tcp_box"
    assert viols[0].limit == 0.0
    assert viols[0].measured == pytest.approx(env.tcp_box.excursion(tcp_position(q_ref)))
