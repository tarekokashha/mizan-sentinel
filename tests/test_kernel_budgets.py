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


def boot(env=None, q0=None):
    env = env or Envelope.ur5e_declared()
    c = Clock()
    k = SafetyKernel(env, clock=c)
    q0 = np.zeros(6) if q0 is None else np.asarray(q0, dtype=float)
    k.filter(RobotState(q=q0, qd=np.zeros(6), t_mono=c.t), Action(q=q0))
    c.tick()
    return k, c, env


def test_excess_force_latches_a_stop():
    k, c, env = boot()
    w = np.zeros(6)
    w[0] = env.force_max * 1.5
    v = k.filter(RobotState(q=np.zeros(6), qd=np.zeros(6), t_mono=c.t, wrench=w),
                 Action(q=np.zeros(6)))
    assert v.status is Status.STOP
    assert any(x.rule == "force_max" for x in v.violations)
    assert k.tripped


def test_excess_torque_latches_a_stop():
    k, c, env = boot()
    w = np.zeros(6)
    w[5] = env.torque_max * 2
    v = k.filter(RobotState(q=np.zeros(6), qd=np.zeros(6), t_mono=c.t, wrench=w),
                 Action(q=np.zeros(6)))
    assert v.status is Status.STOP
    assert any(x.rule == "torque_max" for x in v.violations)


def test_time_in_contact_accumulates_and_eventually_stops():
    env = Envelope.ur5e_declared().replace(contact_time_budget_s=0.2)
    k, c, env = boot(env)
    w = np.zeros(6)
    w[0] = env.contact_force_threshold_n * 2      # in contact, under force_max
    last = None
    for _ in range(20):
        last = k.filter(RobotState(q=np.zeros(6), qd=np.zeros(6), t_mono=c.t, wrench=w),
                        Action(q=np.zeros(6)))
        if last.status is Status.STOP:
            break
        c.tick()
    assert last.status is Status.STOP
    assert any(x.rule == "contact_budget" for x in last.violations)


def test_light_touching_does_not_count_as_contact():
    k, c, env = boot()
    w = np.zeros(6)
    w[0] = env.contact_force_threshold_n * 0.5
    for _ in range(50):
        k.filter(RobotState(q=np.zeros(6), qd=np.zeros(6), t_mono=c.t, wrench=w),
                 Action(q=np.zeros(6)))
        c.tick()
    assert k.contact_s == 0.0


def test_endless_legal_oscillation_exhausts_the_path_budget():
    # slow_drift. Every per-step limit is respected forever; only the
    # cumulative path budget can see this.
    #
    # Deviation from task-7-brief.md (see task-6-7-report.md): the brief's
    # a/b pair is zeros and zeros + [0,-0.3,0,0,0,0]. zeros sits 0.1672 m
    # outside the declared box -- the same defect task-6-correction.md found
    # in Task 5's shaping tests, just not caught here before dispatch. With
    # the Cartesian guard from Task 6 active, a kernel booted at zeros holds
    # q_ref exactly forever (verified: path_m stays 0.0 through all 4000
    # iterations, every verdict carrying a tcp_box violation of 0.1672 m),
    # so the test fails not because the budget guard is missing but because
    # the arm can never move a millimetre. Replaced with A/B: A is the same
    # pose test_kernel_cartesian.py already confirms is inside the box, and
    # B = A + [0,-0.3,0,0,0,0] mirrors the brief's original delta. Measured
    # over a 61-point sweep of the whole segment between them: worst box
    # excursion is exactly 0.0, so only the path budget can stop this.
    env = Envelope.ur5e_declared().replace(path_budget_m=0.5)
    A = np.array([0.0, -1.6, 1.6, -1.6, -1.57, 0.0])
    B = A + np.array([0.0, -0.3, 0.0, 0.0, 0.0, 0.0])
    k, c, env = boot(env, q0=A)
    a, b = A, B
    status = None
    for i in range(4000):
        target = a if (i // 30) % 2 == 0 else b
        v = k.filter(RobotState(q=k._q_cmd.copy(), qd=np.zeros(6), t_mono=c.t),
                     Action(q=target))
        status = v.status
        if status is Status.STOP:
            break
        c.tick()
    assert status is Status.STOP
    assert k.path_m >= env.path_budget_m


def test_gripper_position_and_rate_are_clamped():
    k, c, env = boot()
    # NOTE (deviation from task-7-brief.md, see task-6-7-report.md): the brief
    # passes RobotState(..., grip=0.0), but the dataclass field declared in
    # sentinel/types.py is `gripper`, not `grip` -- RobotState is a frozen
    # dataclass, so an unknown kwarg raises TypeError regardless of the kernel
    # implementation behind it. Using the correct field name here.
    v = k.filter(RobotState(q=np.zeros(6), qd=np.zeros(6), t_mono=c.t, gripper=0.0),
                 Action(q=np.zeros(6), gripper=5.0))
    assert env.grip_min <= v.action.gripper <= env.grip_max
    assert v.action.gripper <= 0.0 + env.grip_rate_max * DT + 1e-9
    assert any(x.rule in ("grip_range", "grip_rate") for x in v.violations)


def test_rearm_resets_both_budgets():
    env = Envelope.ur5e_declared().replace(contact_time_budget_s=0.2)
    k, c, env = boot(env)
    w = np.zeros(6)
    w[0] = env.contact_force_threshold_n * 2
    for _ in range(20):
        if k.filter(RobotState(q=np.zeros(6), qd=np.zeros(6), t_mono=c.t, wrench=w),
                    Action(q=np.zeros(6))).status is Status.STOP:
            break
        c.tick()
    k.rearm("cell cleared")
    assert k.contact_s == 0.0 and k.path_m == 0.0 and not k.tripped
