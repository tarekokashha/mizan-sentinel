"""Optional: make a Shield something LeRobot will accept as a Robot.

Import this module only if you have `lerobot` installed. Nothing else in
this package imports it, and `import sentinel` does not pull it in, because
the safety kernel deliberately depends on numpy and nothing else. A safety
kernel that cannot be installed without a large ML stack is a safety kernel
people vendor a copy of instead, which is worse.

WHY THIS EXISTS. `Shield` is a plain wrapper. It now implements every member
`lerobot.robots.Robot` declares, so duck-typed use works, but it is not a
subclass and `isinstance(shield, Robot)` is False. That was found by M-03
KEYSTONE when the Shield was first composed with a real driver, and it was
the same defect the vendor driver skeleton had: the right method names and
no abstract API. See LIMITATIONS.md.

For callers that only invoke methods, `Shield` alone is enough. For callers
that check the type, or that go through LeRobot's own registry and factory
machinery, wrap it here:

    from sentinel.lerobot_plugin import ShieldRobot
    robot = ShieldRobot(Shield(driver, kernel), config)

THIS ADAPTER ADDS NO SAFETY LOGIC. It forwards every call to the Shield
unmodified. If it clamped, filtered or reordered anything, the guarantee
"no action reaches the robot that the kernel did not authorise" would
become a statement about this file rather than about the kernel, and this
file is not where anyone should have to look for it.
"""
from __future__ import annotations

from typing import Any

from lerobot.robots import Robot, RobotConfig

from sentinel.shield import Shield


class ShieldRobot(Robot):
    """A `lerobot.robots.Robot` whose every call goes through a `Shield`."""

    name = "sentinel_shield"

    def __init__(self, shield: Shield, config: RobotConfig) -> None:
        super().__init__(config)
        self.shield = shield

    # ---- feature contracts ------------------------------------------------ #
    @property
    def observation_features(self) -> dict:
        return self.shield.observation_features

    @property
    def action_features(self) -> dict:
        return self.shield.action_features

    # ---- lifecycle -------------------------------------------------------- #
    @property
    def is_connected(self) -> bool:
        return self.shield.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.shield.connect()

    def disconnect(self) -> None:
        self.shield.disconnect()

    @property
    def is_calibrated(self) -> bool:
        return self.shield.is_calibrated

    def calibrate(self) -> None:
        self.shield.calibrate()

    def configure(self) -> None:
        self.shield.configure()

    # ---- guarded I/O ------------------------------------------------------ #
    def get_observation(self) -> dict:
        return self.shield.get_observation()

    def send_action(self, action: dict) -> dict:
        # No clamp, no filter, no reordering. See the module docstring.
        return self.shield.send_action(action)

    @property
    def last_verdict(self) -> Any:
        """Read-only forward of the Shield's last verdict, so a caller can
        see what the kernel decided without reaching past this adapter.
        """
        return self.shield.last_verdict
