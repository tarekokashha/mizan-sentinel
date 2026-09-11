"""Put the kernel in front of a robot without editing the robot.

The wrapped object needs only `get_observation` and `send_action` in the
LeRobot dict shape, which is what `lerobot_ur.UR5eFollower` already provides.
Nothing in that file changes, which is the point: safety is not the driver's
responsibility and should not live in the driver's diff.
"""
from __future__ import annotations

import numpy as np

from sentinel.journal import Journal
from sentinel.kernel import SafetyKernel
from sentinel.types import Action, RobotState, Status


class Shield:
    def __init__(self, robot, kernel: SafetyKernel, journal: Journal | None = None):
        self.robot = robot
        self.kernel = kernel
        self.journal = journal
        self.last_verdict = None
        self._last_state: RobotState | None = None
        self._seq = 0

    # ---- lifecycle passthrough ------------------------------------------- #
    def connect(self):
        return self.robot.connect()

    def disconnect(self):
        return self.robot.disconnect()

    def calibrate(self):
        return self.robot.calibrate()

    @property
    def is_connected(self):
        return self.robot.is_connected

    @property
    def observation_features(self):
        return self.robot.observation_features

    @property
    def action_features(self):
        return self.robot.action_features

    # ---- guarded I/O ------------------------------------------------------ #
    def get_observation(self) -> dict:
        obs = self.robot.get_observation()
        self._last_state = RobotState(
            q=obs["joint_position"],
            qd=obs.get("joint_velocity", np.zeros(6)),
            t_mono=float(obs.get("timestamp_monotonic", 0.0)),
            wrench=obs.get("tcp_force_torque", np.zeros(6)),
            gripper=float(np.asarray(obs.get("gripper_position", [0.0])).ravel()[0]),
        )
        return obs

    def send_action(self, action: dict) -> dict:
        if self._last_state is None:
            raise RuntimeError(
                "call get_observation before send_action; the kernel cannot judge "
                "an action without a state to judge it against")
        g = action.get("gripper_position")
        g = None if g is None else float(np.asarray(g).ravel()[0])
        verdict = self.kernel.filter(self._last_state, Action(q=action["joint_position"], gripper=g))
        self.last_verdict = verdict

        if self.journal is not None:
            self.journal.append({
                "seq": self._seq, "status": verdict.status.value,
                "violations": [str(v) for v in verdict.violations],
                "q_cmd": None if verdict.action.q is None else [float(x) for x in verdict.action.q],
                "dt": verdict.dt,
            })
        self._seq += 1
        self._last_state = None          # force a fresh observation each cycle

        # CRITICAL 1 / final-fix-1.md: a None joint command means the kernel
        # refused to fabricate a pose (no trusted state existed to hold).
        # Nothing reaches the robot in that case -- not zeros, not the raw
        # request, nothing. self.robot.send_action is not called at all, and
        # the return value deliberately does not echo the caller's request
        # back as if it had been honoured: an empty dict is the honest answer
        # to "what did the robot receive."
        if verdict.action.q is None:
            return {}

        out = dict(action)
        out["joint_position"] = verdict.action.q
        if verdict.action.gripper is not None:
            out["gripper_position"] = np.array([verdict.action.gripper])
        sent = self.robot.send_action(out)
        return sent
