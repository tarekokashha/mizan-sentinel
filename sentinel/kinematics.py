"""UR5e forward kinematics from the manufacturer's DH table.

Source: Universal Robots, "DH Parameters for calculations of kinematics and
dynamics", verified 2026-09-05. The numbers below are transcribed from that
page and must not be edited from memory.
"""
from __future__ import annotations

import numpy as np

from sentinel.types import N_JOINTS

DH_A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0], dtype=float)
DH_D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996], dtype=float)
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=float)


def _link(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ], dtype=float)


def _chain(q) -> list[np.ndarray]:
    """Cumulative transforms base -> joint i, for i in 0..6."""
    q = np.asarray(q, dtype=float).reshape(N_JOINTS)
    out = [np.eye(4)]
    T = np.eye(4)
    for i in range(N_JOINTS):
        T = T @ _link(DH_A[i], DH_D[i], DH_ALPHA[i], q[i])
        out.append(T)
    return out


def fk(q) -> np.ndarray:
    """Base to flange transform.

    >>> fk(np.zeros(6)).shape
    (4, 4)
    """
    return _chain(q)[-1]


def tcp_position(q) -> np.ndarray:
    """Flange origin in the base frame, metres."""
    return fk(q)[:3, 3].copy()


def wrist_center(q) -> np.ndarray:
    """Origin of frame 3, which is the elbow-end of the two long links."""
    return _chain(q)[3][:3, 3].copy()


def jacobian(q) -> np.ndarray:
    """Geometric Jacobian at the flange, rows [vx vy vz wx wy wz]."""
    Ts = _chain(q)
    p_e = Ts[-1][:3, 3]
    J = np.zeros((6, N_JOINTS))
    for i in range(N_JOINTS):
        z = Ts[i][:3, 2]
        p = Ts[i][:3, 3]
        J[:3, i] = np.cross(z, p_e - p)
        J[3:, i] = z
    return J
