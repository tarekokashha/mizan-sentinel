"""A simulated arm, good enough to make the guards falsifiable.

The plant is a critically damped position servo with saturated joint
acceleration, integrated semi-implicitly at a finer step than the control
period. The saturation is the whole point: an infeasible command produces
overshoot, so a kernel that clips position but ignores braking distance can
actually be caught failing. A position-replay plant would make the most
important guard in the kernel untestable.
"""
from __future__ import annotations

import numpy as np

from sentinel.kinematics import tcp_position
from sentinel.types import N_JOINTS, RobotState


class PlaneContact:
    """A rigid horizontal plane the flange can push into."""

    def __init__(self, z: float, stiffness: float = 5000.0):
        self.z = float(z)
        self.stiffness = float(stiffness)

    def wrench(self, p_tcp) -> np.ndarray:
        w = np.zeros(6)
        pen = self.z - float(np.asarray(p_tcp)[2])
        if pen > 0.0:
            w[2] = self.stiffness * pen
        return w


class SimPlant:
    def __init__(self, q0, qdd_max, kp: float = 400.0, zeta: float = 1.0,
                 accel_authority: float = 1.5, substeps: int = 10,
                 contact: PlaneContact | None = None, seed: int = 0):
        self.q = np.asarray(q0, dtype=float).reshape(N_JOINTS).copy()
        self.qd = np.zeros(N_JOINTS)
        self.kp = float(kp)
        self.kd = 2.0 * zeta * np.sqrt(self.kp)
        self.a_max = np.asarray(qdd_max, dtype=float).reshape(N_JOINTS) * accel_authority
        self.substeps = int(substeps)
        self.contact = contact
        self.rng = np.random.default_rng(seed)

    def step(self, q_cmd, dt: float) -> None:
        q_cmd = np.asarray(q_cmd, dtype=float).reshape(N_JOINTS)
        h = dt / self.substeps
        for _ in range(self.substeps):
            a = np.clip(self.kp * (q_cmd - self.q) - self.kd * self.qd,
                        -self.a_max, self.a_max)
            self.qd = self.qd + a * h          # semi-implicit Euler: stable at 30 Hz
            self.q = self.q + self.qd * h

    def state(self, t_mono: float, gripper: float = 0.0) -> RobotState:
        w = self.contact.wrench(tcp_position(self.q)) if self.contact else np.zeros(6)
        return RobotState(q=self.q.copy(), qd=self.qd.copy(), t_mono=float(t_mono),
                          wrench=w, gripper=float(gripper))
