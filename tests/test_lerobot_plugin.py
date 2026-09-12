"""The LeRobot contract, from the side that ships it.

M-03 KEYSTONE composed the Shield with a real driver and found that the
Shield was not a LeRobot plugin: it implemented eight of the ten members
`lerobot.robots.Robot` declares, missing `configure` and `is_calibrated`,
which are exactly the two the vendor driver skeleton was missing. This
programme had diagnosed that defect one layer down and shipped it one layer
up, and could not see it, because every Shield test here wraps a fake built
to the dict contract and a dict-contract fake performs no abstract-API
check.

Two kinds of test live here, and the split is the point.

The first runs everywhere, including in this repository's own numpy-only
environment where `lerobot` is not installed. It checks the member names
against a list recorded from a measured run, so the gap cannot reopen in
the environment where the package is actually developed.

The second runs only where `lerobot` is present and checks against the real
base class, because a recorded list is a snapshot and the installed library
is the authority. If the two ever disagree, the second fails and the list
is wrong, which is the correct direction for that error to point.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel.envelope import Envelope
from sentinel.kernel import SafetyKernel
from sentinel.shield import Shield
from sentinel.sim import SimPlant

# Measured from lerobot 0.4.4 by reading Robot.__abstractmethods__, not
# copied from documentation. tests/test_lerobot_plugin.py's second half
# re-derives it from the installed library wherever that library exists.
LEROBOT_ROBOT_MEMBERS = (
    "action_features",
    "calibrate",
    "configure",
    "connect",
    "disconnect",
    "get_observation",
    "is_calibrated",
    "is_connected",
    "observation_features",
    "send_action",
)

Q0 = np.array([0.0, -1.2, 1.2, -1.5, -1.5708, 0.0])

try:
    import lerobot.robots as _lerobot_robots
except Exception:  # pragma: no cover - depends on the environment
    _lerobot_robots = None

requires_lerobot = pytest.mark.skipif(
    _lerobot_robots is None,
    reason="lerobot is not installed; this package depends on numpy alone by design")


class _MinimalDriver:
    """The shape Shield promises to accept: get_observation and send_action,
    and no lifecycle hooks at all. Shield must complete its own LeRobot
    surface without requiring them, which is why the hooks soft delegate.
    """

    observation_features = {"joint_position": (6,)}
    action_features = {"joint_position": (6,)}

    def __init__(self) -> None:
        self.env = Envelope.ur5e_declared()
        self.plant = SimPlant(q0=Q0, qdd_max=self.env.qdd_max)
        self.t = 0.0
        self.sent: list[np.ndarray] = []

    def connect(self):
        return None

    def disconnect(self):
        return None

    @property
    def is_connected(self):
        return True

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


def _shield():
    driver = _MinimalDriver()
    return driver, Shield(driver, SafetyKernel(Envelope.ur5e_declared()))


# ---- runs everywhere, including without lerobot ------------------------------ #

@pytest.mark.parametrize("member", LEROBOT_ROBOT_MEMBERS)
def test_shield_exposes_every_member_lerobot_declares(member):
    """The regression test for the defect M-03 found.

    Parametrised so a failure names the missing member rather than reporting
    that a set comparison failed.
    """
    _, shield = _shield()
    assert hasattr(shield, member), (
        f"Shield is missing {member!r}, which lerobot.robots.Robot declares "
        f"abstract. This is the defect M-03 KEYSTONE found by composition.")


def test_the_lifecycle_hooks_work_against_a_driver_that_has_none():
    """Shield's stated contract is that the wrapped object needs only
    get_observation and send_action. So the hooks must not hard delegate:
    `calibrate` used to, and would have raised AttributeError against the
    minimal fake this repository's own tests wrap, had anything called it.
    """
    driver, shield = _shield()
    for name in ("calibrate", "configure", "is_calibrated"):
        assert not hasattr(driver, name), f"the fixture is wrong: driver has {name}"

    assert shield.calibrate() is None
    assert shield.configure() is None
    # lerobot documents this as "always True if not applicable".
    assert shield.is_calibrated is True


def test_a_driver_that_does_have_the_hooks_is_delegated_to():
    """Soft delegation must still delegate, or the fallback would silently
    replace a real driver's behaviour.
    """
    driver, shield = _shield()
    calls = []
    driver.configure = lambda: calls.append("configure")
    driver.calibrate = lambda: calls.append("calibrate")
    driver.is_calibrated = False

    shield.configure()
    shield.calibrate()

    assert calls == ["configure", "calibrate"]
    assert shield.is_calibrated is False, "the fallback overrode the driver"


# ---- runs only where lerobot is installed ------------------------------------ #

@requires_lerobot
def test_the_recorded_member_list_still_matches_the_installed_library():
    """A recorded list is a snapshot. The installed library is the authority,
    so where it exists it wins, and a disagreement fails here rather than
    letting the snapshot quietly go stale.
    """
    declared = set(_lerobot_robots.Robot.__abstractmethods__)
    assert declared == set(LEROBOT_ROBOT_MEMBERS), (
        f"lerobot's abstract surface has changed: "
        f"added {sorted(declared - set(LEROBOT_ROBOT_MEMBERS))}, "
        f"removed {sorted(set(LEROBOT_ROBOT_MEMBERS) - declared)}")


@requires_lerobot
def test_shield_satisfies_the_installed_abstract_surface():
    _, shield = _shield()
    missing = sorted(m for m in _lerobot_robots.Robot.__abstractmethods__
                     if not hasattr(shield, m))
    assert missing == [], f"Shield is missing {missing}"


@requires_lerobot
def test_shield_robot_is_a_real_plugin_and_instantiates(tmp_path):
    from sentinel.lerobot_plugin import ShieldRobot

    assert issubclass(ShieldRobot, _lerobot_robots.Robot)
    assert ShieldRobot.__abstractmethods__ == frozenset(), (
        f"unimplemented: {sorted(ShieldRobot.__abstractmethods__)}")

    _, shield = _shield()
    cfg = _lerobot_robots.RobotConfig(id="test", calibration_dir=tmp_path)
    robot = ShieldRobot(shield, cfg)
    assert isinstance(robot, _lerobot_robots.Robot)


@requires_lerobot
def test_the_adapter_adds_no_safety_logic(tmp_path):
    """The guarantee must stay a statement about the kernel.

    If the adapter clamped or filtered anything, "no action reaches the
    robot that the kernel did not authorise" would become a claim about
    sentinel/lerobot_plugin.py instead, and that is not where anyone should
    have to look for it.
    """
    from sentinel.lerobot_plugin import ShieldRobot

    driver, shield = _shield()
    cfg = _lerobot_robots.RobotConfig(id="test", calibration_dir=tmp_path)
    robot = ShieldRobot(shield, cfg)

    robot.connect()
    robot.get_observation()
    robot.send_action({"joint_position": np.full(6, 1e9)})

    assert driver.sent, "nothing reached the driver at all"
    assert np.array_equal(driver.sent[-1], robot.last_verdict.action.q), (
        "the value the driver received is not the value the kernel emitted, "
        "so the adapter modified it")
    assert np.all(np.isfinite(driver.sent[-1]))
