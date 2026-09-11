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
                 contact: PlaneContact | None = None, seed: int = 0,
                 q0_jitter_rad: float = 0.0, sensor_noise_std: float = 0.0):
        # q0_jitter_rad and sensor_noise_std default to 0.0, which reproduces
        # the plant's previous exact behaviour for every direct construction
        # in this codebase (unit tests that pin exact numeric outcomes --
        # e.g. test_kernel_cartesian.py's momentum-carries-past-the-margin
        # check, test_redteam.py's force_grind trip step -- all rely on it).
        # sentinel.redteam is the one caller that opts in, with both
        # parameters nonzero: CRITICAL 2 / final-fix-1.md found that
        # SimPlant.seed was stored and never read, every episode started from
        # the same fixed q0, and 12 of 15 attacks produced bit-identical
        # episodes across all 200 repetitions as a result. A confidence
        # sequence over 200 identical repeats carries the evidential content
        # of n = 1. Reading self.rng here is what makes the seed the runner
        # already threads through per episode actually do something.
        self.q = np.asarray(q0, dtype=float).reshape(N_JOINTS).copy()
        self.qd = np.zeros(N_JOINTS)
        self.kp = float(kp)
        self.kd = 2.0 * zeta * np.sqrt(self.kp)
        self.a_max = np.asarray(qdd_max, dtype=float).reshape(N_JOINTS) * accel_authority
        self.substeps = int(substeps)
        self.contact = contact
        self.rng = np.random.default_rng(seed)
        self._sensor_noise_std = float(sensor_noise_std)
        if q0_jitter_rad:
            # Per-episode start-state jitter, drawn once, before any control
            # loop sees this plant. Deliberately small -- "a few
            # milliradians" -- so it stays inside every declared joint and
            # Cartesian margin measured for the episode starts this
            # catalogue uses (lever_sprint's start keeps the least margin,
            # about 19 mm from the box wall; the caller re-verifies the
            # actual jittered start against the envelope rather than trusting
            # this comment -- see run_episode).
            self.q = self.q + self.rng.normal(0.0, float(q0_jitter_rad), N_JOINTS)

    def step(self, q_cmd, dt: float) -> None:
        q_cmd = np.asarray(q_cmd, dtype=float).reshape(N_JOINTS)
        h = dt / self.substeps
        for _ in range(self.substeps):
            a = np.clip(self.kp * (q_cmd - self.q) - self.kd * self.qd,
                        -self.a_max, self.a_max)
            self.qd = self.qd + a * h          # semi-implicit Euler: stable at 30 Hz
            self.q = self.q + self.qd * h

    def state(self, t_mono: float, gripper: float = 0.0) -> RobotState:
        # Sensor noise is added only to what is *reported*: the wrench and
        # the ground truth used by step()'s dynamics both read self.q/self.qd
        # directly, unperturbed, the same way a real encoder's noise never
        # feeds back into the arm's actual physical position.
        if self._sensor_noise_std:
            q_report = self.q + self.rng.normal(0.0, self._sensor_noise_std, N_JOINTS)
            qd_report = self.qd + self.rng.normal(0.0, self._sensor_noise_std, N_JOINTS)
        else:
            q_report, qd_report = self.q.copy(), self.qd.copy()
        w = self.contact.wrench(tcp_position(self.q)) if self.contact else np.zeros(6)
        return RobotState(q=q_report, qd=qd_report, t_mono=float(t_mono),
                          wrench=w, gripper=float(gripper))
