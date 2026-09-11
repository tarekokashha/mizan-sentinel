"""Shared vocabulary for the safety kernel. No logic lives here."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

N_JOINTS = 6


class Status(str, Enum):
    """Outcome of one kernel call, ordered by severity."""

    PASS = "PASS"
    CLAMPED = "CLAMPED"
    HOLD = "HOLD"
    STOP = "STOP"


_SEVERITY = {Status.PASS: 0, Status.CLAMPED: 1, Status.HOLD: 2, Status.STOP: 3}


def worst(a: Status, b: Status) -> Status:
    """The more severe of two statuses.

    >>> worst(Status.PASS, Status.STOP) is Status.STOP
    True
    """
    return a if _SEVERITY[a] >= _SEVERITY[b] else b


@dataclass(frozen=True)
class Violation:
    """One broken rule, with the number that broke it."""

    rule: str
    measured: float
    limit: float
    index: int | None = None

    def __str__(self) -> str:
        where = f" j{self.index}" if self.index is not None else ""
        return f"{self.rule}{where}: {self.measured:.6g} vs limit {self.limit:.6g}"


def _vec(x, n: int = N_JOINTS) -> np.ndarray:
    return np.array(x, dtype=float).reshape(n)


@dataclass(frozen=True, eq=False)
class RobotState:
    """One observation. `t_mono` comes from the source and is not trusted."""

    q: np.ndarray
    qd: np.ndarray
    t_mono: float
    wrench: np.ndarray = field(default_factory=lambda: np.zeros(6))
    gripper: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", _vec(self.q))
        object.__setattr__(self, "qd", _vec(self.qd))
        object.__setattr__(self, "wrench", _vec(self.wrench))


@dataclass(frozen=True, eq=False)
class Action:
    """One commanded joint target, plus an optional gripper position.

    `q` is `None` only when the kernel has no trusted state to hold and
    refuses to fabricate one -- see SafetyKernel.filter's guard for a
    non-finite observation on the very first call. `None` means command no
    motion at all; it is not a stand-in for "hold the last position", and a
    caller forwarding an Action to a robot must check for it before sending
    anything.
    """

    q: np.ndarray | None
    gripper: float | None = None

    def __post_init__(self) -> None:
        if self.q is not None:
            object.__setattr__(self, "q", np.array(self.q, dtype=float).reshape(N_JOINTS))


@dataclass(frozen=True, eq=False)
class Verdict:
    """What the kernel decided, and everything needed to audit the decision."""

    status: Status
    action: Action
    violations: tuple[Violation, ...] = ()
    dt: float = 0.0
    telemetry: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in (Status.PASS, Status.CLAMPED)
