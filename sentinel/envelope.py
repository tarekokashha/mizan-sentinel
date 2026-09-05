"""The declared safety envelope: what the arm is permitted to do.

Nothing here is measured. Every value is either copied from the Universal
Robots datasheet or declared outright by the owner. The kit's operating
contract forbids this code writing a real-robot number, so `provenance`
records where each value came from and loading a "measured" value without a
named human and a date is an error.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, replace as _dc_replace

import numpy as np

from sentinel.types import N_JOINTS

# UR5e manufacturer maxima. Verified 2026-09-05 against Universal Robots'
# published technical specifications and DH parameter pages. These bound the
# envelope; they are not the envelope.
DATASHEET_Q_RANGE_RAD = float(np.deg2rad(360.0))
DATASHEET_QD_MAX_RAD_S = float(np.deg2rad(180.0))
DATASHEET_TCP_SPEED_M_S = 1.0
DATASHEET_REACH_M = 0.850
DATASHEET_PAYLOAD_KG = 5.0

PROVENANCE_VALUES = ("datasheet", "declared", "measured")

_ARRAY_FIELDS = ("q_min", "q_max", "qd_max", "qdd_max", "qddd_max")


@dataclass(frozen=True, eq=False)
class Box:
    """An axis-aligned region of the base frame, in metres."""

    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "lo", np.asarray(self.lo, dtype=float).reshape(3))
        object.__setattr__(self, "hi", np.asarray(self.hi, dtype=float).reshape(3))
        if not np.all(self.hi > self.lo):
            raise ValueError("Box hi must exceed lo on every axis")

    def contains(self, p) -> bool:
        p = np.asarray(p, dtype=float).reshape(3)
        return bool(np.all(p >= self.lo) and np.all(p <= self.hi))

    def excursion(self, p) -> float:
        """Largest distance outside the box on any axis. 0.0 when inside."""
        p = np.asarray(p, dtype=float).reshape(3)
        return float(np.max(np.maximum(np.maximum(self.lo - p, p - self.hi), 0.0)))


@dataclass(frozen=True, eq=False)
class Envelope:
    """A declaration of permitted motion, hashable so a run can cite it."""

    name: str
    robot: str
    q_min: np.ndarray
    q_max: np.ndarray
    qd_max: np.ndarray
    qdd_max: np.ndarray
    qddd_max: np.ndarray
    tcp_box: Box
    tcp_speed_max: float
    force_max: float
    torque_max: float
    grip_min: float
    grip_max: float
    grip_rate_max: float
    watchdog_s: float
    min_dt_s: float
    max_dt_s: float
    stale_escalate_n: int
    tracking_tol_rad: float
    path_budget_m: float
    contact_time_budget_s: float
    contact_force_threshold_n: float
    plant_margin_rad: float
    plant_margin_m: float
    bisect_iters: int
    provenance: dict = field(default_factory=dict)
    measured_by: str | None = None
    measured_on: str | None = None

    def __post_init__(self) -> None:
        for name in _ARRAY_FIELDS:
            object.__setattr__(
                self, name, np.asarray(getattr(self, name), dtype=float).reshape(N_JOINTS)
            )
        if not np.all(self.q_max > self.q_min):
            raise ValueError("q_max must exceed q_min on every joint")
        for name in ("qd_max", "qdd_max", "qddd_max"):
            if not np.all(getattr(self, name) > 0):
                raise ValueError(f"{name} must be positive on every joint")
        if np.any(self.q_max > DATASHEET_Q_RANGE_RAD) or np.any(self.q_min < -DATASHEET_Q_RANGE_RAD):
            raise ValueError("joint range exceeds the UR5e datasheet")
        if np.any(self.qd_max > DATASHEET_QD_MAX_RAD_S):
            raise ValueError("joint speed exceeds the UR5e datasheet")
        if self.tcp_speed_max > DATASHEET_TCP_SPEED_M_S:
            raise ValueError("TCP speed exceeds the UR5e datasheet")
        for name in ("tcp_speed_max", "force_max", "torque_max", "grip_rate_max",
                     "watchdog_s", "min_dt_s", "max_dt_s", "tracking_tol_rad",
                     "path_budget_m", "contact_time_budget_s",
                     "contact_force_threshold_n", "plant_margin_rad", "plant_margin_m"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if self.min_dt_s >= self.max_dt_s:
            raise ValueError("min_dt_s must be below max_dt_s")
        if self.grip_max <= self.grip_min:
            raise ValueError("grip_max must exceed grip_min")
        if self.stale_escalate_n < 1 or self.bisect_iters < 1:
            raise ValueError("stale_escalate_n and bisect_iters must be at least 1")
        bad = {k: v for k, v in self.provenance.items() if v not in PROVENANCE_VALUES}
        if bad:
            raise ValueError(f"unknown provenance values: {bad}")
        if "measured" in self.provenance.values() and not (self.measured_by and self.measured_on):
            raise ValueError(
                "a provenance of 'measured' requires measured_by and measured_on; "
                "only a human who ran the arm may write a measured number"
            )

    # ---- serialisation ---------------------------------------------------- #
    def to_dict(self) -> dict:
        d: dict = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                d[f.name] = [float(x) for x in v]
            elif isinstance(v, Box):
                d[f.name] = {"lo": [float(x) for x in v.lo], "hi": [float(x) for x in v.hi]}
            elif isinstance(v, dict):
                d[f.name] = dict(v)
            else:
                d[f.name] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Envelope":
        d = dict(d)
        d["tcp_box"] = Box(lo=d["tcp_box"]["lo"], hi=d["tcp_box"]["hi"])
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "Envelope":
        return cls.from_dict(json.loads(s))

    def sha256(self) -> str:
        """Content hash of the declaration. Goes in every journal line."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def replace(self, **kw) -> "Envelope":
        return _dc_replace(self, **kw)

    # ---- factory ---------------------------------------------------------- #
    @classmethod
    def ur5e_declared(cls) -> "Envelope":
        """The conservative default. Every value sits well inside the datasheet."""
        six = np.ones(N_JOINTS)
        return cls(
            name="ur5e-declared-v1",
            robot="ur5e",
            q_min=-np.pi * six,
            q_max=np.pi * six,
            qd_max=1.0 * six,
            qdd_max=5.0 * six,
            qddd_max=100.0 * six,
            tcp_box=Box(lo=[-0.65, -0.65, 0.05], hi=[0.65, 0.65, 0.90]),
            tcp_speed_max=0.25,
            force_max=40.0,
            torque_max=8.0,
            grip_min=0.0,
            grip_max=1.0,
            grip_rate_max=2.0,
            watchdog_s=0.2,
            min_dt_s=1e-4,
            max_dt_s=0.2,
            stale_escalate_n=3,
            tracking_tol_rad=0.25,
            path_budget_m=25.0,
            contact_time_budget_s=5.0,
            contact_force_threshold_n=5.0,
            plant_margin_rad=0.02,
            plant_margin_m=0.005,
            bisect_iters=24,
            provenance={
                "q_min": "declared", "q_max": "declared", "qd_max": "declared",
                "qdd_max": "declared", "qddd_max": "declared", "tcp_box": "declared",
                "tcp_speed_max": "declared", "force_max": "declared",
                "torque_max": "declared", "grip_rate_max": "declared",
                "watchdog_s": "declared", "max_dt_s": "declared",
                "tracking_tol_rad": "declared", "path_budget_m": "declared",
                "contact_time_budget_s": "declared",
            },
        )
