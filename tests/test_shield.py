"""Shield: the drop-in wrapper that puts the kernel in front of any
LeRobot-style follower without editing that follower.

Covers the five tests from task-12-brief.md plus two from task-12-note.md:

- A driver that never supplies `timestamp_monotonic` must eventually fail
  closed. The missing-key default of 0.0 reads as a frozen stamp, which is
  correct fail-closed behaviour, not an oversight (see the note for why).
- `send_action` clears `_last_state` after forwarding, so the "observation
  required" guard must hold on every cycle, not just the very first call
  the Shield ever sees. A one-shot version of that test would still pass
  even if the reset were removed from `send_action`.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel.envelope import Envelope
from sentinel.kernel import SafetyKernel
from sentinel.shield import Shield
from sentinel.sim import SimPlant
from sentinel.types import Status


class FakeRobot:
    """Minimal stand-in with the LeRobot follower shape."""

    def __init__(self):
        self.env = Envelope.ur5e_declared()
        self.plant = SimPlant(q0=np.zeros(6), qdd_max=self.env.qdd_max)
        self.t = 0.0
        self.sent = []
        self._connected = False

    observation_features = {"joint_position": (6,)}
    action_features = {"joint_position": (6,)}

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    @property
    def is_connected(self):
        return self._connected

    def get_observation(self):
        s = self.plant.state(t_mono=self.t)
        return {"joint_position": s.q, "joint_velocity": s.qd,
                "tcp_force_torque": s.wrench, "gripper_position": np.array([0.0]),
                "timestamp_monotonic": self.t}

    def send_action(self, action):
        self.sent.append(np.asarray(action["joint_position"], dtype=float).copy())
        self.plant.step(action["joint_position"], 1 / 30)
        self.t += 1 / 30
        return action


class Clock:
    def __init__(self, robot):
        self.robot = robot

    def __call__(self):
        return self.robot.t


def test_the_shield_forwards_a_legal_action_unchanged():
    r = FakeRobot()
    s = Shield(r, SafetyKernel(r.env, clock=Clock(r)))
    s.get_observation()
    s.send_action({"joint_position": np.zeros(6)})
    s.get_observation()
    out = s.send_action({"joint_position": np.full(6, 1e-4)})
    assert np.allclose(out["joint_position"], np.full(6, 1e-4), atol=1e-9)


def test_the_shield_stops_a_non_finite_action_before_the_robot_sees_it():
    r = FakeRobot()
    s = Shield(r, SafetyKernel(r.env, clock=Clock(r)))
    s.get_observation()
    s.send_action({"joint_position": np.zeros(6)})
    s.get_observation()
    s.send_action({"joint_position": np.full(6, np.nan)})
    assert all(np.all(np.isfinite(a)) for a in r.sent)
    assert s.last_verdict.status is Status.STOP


def test_the_shield_never_lets_a_slam_reach_the_robot_out_of_range():
    r = FakeRobot()
    s = Shield(r, SafetyKernel(r.env, clock=Clock(r)))
    for _ in range(300):
        s.get_observation()
        s.send_action({"joint_position": np.full(6, 100.0)})
    sent = np.array(r.sent)
    assert np.all(sent <= r.env.q_max + 1e-9)
    assert np.all(sent >= r.env.q_min - 1e-9)


def test_the_shield_passes_lifecycle_calls_through():
    r = FakeRobot()
    s = Shield(r, SafetyKernel(r.env, clock=Clock(r)))
    s.connect()
    assert s.is_connected and r.is_connected
    s.disconnect()
    assert not r.is_connected
    assert s.observation_features == r.observation_features


def test_send_action_requires_an_observation_first():
    r = FakeRobot()
    s = Shield(r, SafetyKernel(r.env, clock=Clock(r)))
    with pytest.raises(RuntimeError, match="observation"):
        s.send_action({"joint_position": np.zeros(6)})


def test_send_action_requires_a_fresh_observation_every_cycle_not_just_the_first():
    # send_action clears _last_state after forwarding (decision #3 in the
    # task brief), so a *second* cycle without an intervening
    # get_observation must raise too, not only the very first call the
    # Shield ever sees. A kernel judging an action against a stale state is
    # exactly the failure this project exists to prevent, and a test that
    # only tries the first-ever call would keep passing even if the reset
    # at the end of send_action were deleted.
    r = FakeRobot()
    s = Shield(r, SafetyKernel(r.env, clock=Clock(r)))
    s.get_observation()
    s.send_action({"joint_position": np.zeros(6)})
    n_sent = len(r.sent)
    with pytest.raises(RuntimeError, match="observation"):
        s.send_action({"joint_position": np.zeros(6)})
    assert len(r.sent) == n_sent, "the unobserved second call must not reach the robot"


def test_a_robot_that_supplies_no_timestamp_eventually_fails_closed():
    # A driver that cannot say when it sampled is a driver whose freshness
    # cannot be established. The missing-key default of 0.0 reads as a
    # frozen stamp, so the watchdog holds and then latches, which is the
    # correct answer rather than an oversight.
    r = FakeRobot()
    r.get_observation = lambda: {"joint_position": r.plant.q,
                                 "joint_velocity": r.plant.qd}   # no timestamp
    s = Shield(r, SafetyKernel(r.env, clock=Clock(r)))
    seen = []
    for _ in range(6):
        s.get_observation()
        s.send_action({"joint_position": np.zeros(6)})
        seen.append(s.last_verdict.status)
        r.t += 1 / 3          # advance well past watchdog_s = 0.2
    assert Status.STOP in seen
