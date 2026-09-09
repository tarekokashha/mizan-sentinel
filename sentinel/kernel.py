"""The safety kernel.

`filter` is a pure decision over the kernel's own state. It performs no I/O,
takes its clock by injection, and fails closed: anything it cannot understand
becomes a hold or a latching stop, never a pass.

Guard order is fixed and each guard may short-circuit. The order matters: a
non-finite action must be rejected before any arithmetic touches it, and a
stale state must be caught before its numbers are believed.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from sentinel.envelope import Envelope
from sentinel.kinematics import tcp_position
from sentinel.types import Action, RobotState, Status, Verdict, Violation


class SafetyKernel:
    def __init__(
        self,
        envelope: Envelope,
        # time.perf_counter, not time.monotonic. On Windows, monotonic is
        # GetTickCount64 with 15.625 ms resolution, which is half a control
        # period at 30 Hz, and the shaping chain divides by dt three times.
        # perf_counter is QueryPerformanceCounter at 100 ns, and it is also
        # the clock the UR driver stamps its observations with.
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.env = envelope
        self._clock = clock
        self._t_prev: float | None = None
        self._q_cmd: np.ndarray | None = None
        self._qd_cmd = np.zeros(6)
        self._qdd_cmd = np.zeros(6)
        self._grip_cmd: float | None = None
        self._tcp_prev: np.ndarray | None = None
        self._stale_n = 0
        self._tripped = False
        self._trip_reason = ""
        self.path_m = 0.0
        self.contact_s = 0.0
        # Staleness bookkeeping. The driver's own stamp is compared only to
        # its own previous value, never to this kernel's clock - see the
        # comment on guard 2 in filter() for why.
        self._t_src_prev: float | None = None
        self._t_src_fresh_at: float = 0.0

    # ---- public state ----------------------------------------------------- #
    @property
    def tripped(self) -> bool:
        return self._tripped

    def rearm(self, reason: str) -> None:
        """Clear a latched stop. Only a caller who knows why may do this."""
        if not reason:
            raise ValueError("rearm requires a reason")
        self._tripped = False
        self._trip_reason = ""
        self._stale_n = 0
        self.path_m = 0.0
        self.contact_s = 0.0
        self._qd_cmd = np.zeros(6)
        self._qdd_cmd = np.zeros(6)
        self._t_prev = None
        self._t_src_prev = None
        self._t_src_fresh_at = 0.0

    # ---- helpers ---------------------------------------------------------- #
    def _hold(self, q_hold, violations, dt, extra=None) -> Verdict:
        self._qd_cmd = np.zeros(6)
        self._qdd_cmd = np.zeros(6)
        self._q_cmd = np.asarray(q_hold, dtype=float).copy()
        return Verdict(Status.HOLD, Action(q=self._q_cmd.copy(), gripper=self._grip_cmd),
                       tuple(violations), dt, extra or {})

    def _stop(self, q_hold, violations, dt, reason: str) -> Verdict:
        self._tripped = True
        self._trip_reason = reason
        self._qd_cmd = np.zeros(6)
        self._qdd_cmd = np.zeros(6)
        self._q_cmd = np.asarray(q_hold, dtype=float).copy()
        return Verdict(Status.STOP, Action(q=self._q_cmd.copy(), gripper=self._grip_cmd),
                       tuple(violations), dt, {"reason": reason})

    @staticmethod
    def _brake_bound(room, accel, dt):
        """Largest velocity that can be brought to rest within `room`.

        Two terms, and both matter. The first is the dt-corrected stopping
        bound: it reduces to sqrt(2*a*room) as dt goes to zero and is strictly
        tighter for a finite step, because a discrete controller sheds velocity
        in dt-sized bites rather than continuously. The second, room/dt, keeps
        the caller from overshooting `room` in the very next step; without it
        the final position clip bites, the emitted velocity stops matching the
        clamped one, and the emitted acceleration exceeds its limit.

        Used at two levels: velocity against position headroom, and
        acceleration against velocity headroom.
        """
        room = np.maximum(np.asarray(room, dtype=float), 0.0)
        adt = accel * dt
        stop = -0.5 * adt + np.sqrt(np.maximum((0.5 * adt) ** 2 + 2.0 * accel * room, 0.0))
        return np.minimum(stop, room / dt)

    def _shape(self, q_ref, q_des, dt) -> tuple[np.ndarray, list[Violation]]:
        """Shape a request into a command whose own derivatives obey the envelope.

        Order matters and is not the obvious one. Settle the velocity target
        first, including the braking bound, then let jerk and acceleration
        shape the approach to that target. Clamping the derivatives first and
        recomputing velocity from a clipped position, which is the intuitive
        order, means the number that was clamped is not the number that gets
        emitted.
        """
        env = self.env
        k = env.brake_headroom
        v: list[Violation] = []

        # --- position level: cap velocity so the joint can still stop -------- #
        qd_raw = (q_des - q_ref) / dt
        qd_want = np.clip(qd_raw, -env.qd_max, env.qd_max)
        if np.any(np.abs(qd_raw) > env.qd_max + 1e-12):
            i = int(np.argmax(np.abs(qd_raw) - env.qd_max))
            v.append(Violation("qd_max", float(abs(qd_raw[i])), float(env.qd_max[i]), i))

        brake_hi = self._brake_bound(env.q_max - q_ref, env.qdd_max * k, dt)
        brake_lo = self._brake_bound(q_ref - env.q_min, env.qdd_max * k, dt)
        qd_target = np.clip(qd_want, -brake_lo, brake_hi)
        if np.any(np.abs(qd_target - qd_want) > 1e-12):
            i = int(np.argmax(np.abs(qd_target - qd_want)))
            v.append(Violation("brake", float(qd_want[i]),
                               float(brake_hi[i] if qd_want[i] > 0 else -brake_lo[i]), i))

        # --- velocity level: cap acceleration so it can return to zero ------- #
        qdd_want = (qd_target - self._qd_cmd) / dt
        qdd_hi = self._brake_bound(env.qd_max - self._qd_cmd, env.qddd_max * k, dt)
        qdd_lo = self._brake_bound(self._qd_cmd + env.qd_max, env.qddd_max * k, dt)
        qdd_target = np.clip(qdd_want, -qdd_lo, qdd_hi)
        if np.any(np.abs(qdd_target - qdd_want) > 1e-12):
            i = int(np.argmax(np.abs(qdd_target - qdd_want)))
            v.append(Violation("brake_accel", float(qdd_want[i]),
                               float(qdd_hi[i] if qdd_want[i] > 0 else -qdd_lo[i]), i))
        qdd_target = np.clip(qdd_target, -env.qdd_max, env.qdd_max)
        if np.any(np.abs(qdd_want) > env.qdd_max + 1e-12):
            i = int(np.argmax(np.abs(qdd_want) - env.qdd_max))
            v.append(Violation("qdd_max", float(abs(qdd_want[i])), float(env.qdd_max[i]), i))

        # --- jerk level ------------------------------------------------------ #
        qddd_want = (qdd_target - self._qdd_cmd) / dt
        qddd = np.clip(qddd_want, -env.qddd_max, env.qddd_max)
        if np.any(np.abs(qddd_want) > env.qddd_max + 1e-12):
            i = int(np.argmax(np.abs(qddd_want) - env.qddd_max))
            v.append(Violation("qddd_max", float(abs(qddd_want[i])), float(env.qddd_max[i]), i))

        qdd = np.clip(self._qdd_cmd + qddd * dt, -env.qdd_max, env.qdd_max)
        qd = np.clip(self._qd_cmd + qdd * dt, -env.qd_max, env.qd_max)
        q_cmd = np.clip(q_ref + qd * dt, env.q_min, env.q_max)

        q_des_clipped = np.clip(q_des, env.q_min, env.q_max)
        if np.any(np.abs(q_des - q_des_clipped) > 1e-12):
            i = int(np.argmax(np.abs(q_des - q_des_clipped)))
            lim = env.q_max[i] if q_des[i] > q_des_clipped[i] else env.q_min[i]
            v.append(Violation("q_limit", float(q_des[i]), float(lim), i))

        # keep the stored derivatives equal to what was actually emitted
        qd_real = (q_cmd - q_ref) / dt
        self._qdd_cmd = (qd_real - self._qd_cmd) / dt
        self._qd_cmd = qd_real
        return q_cmd, v

    def _cartesian(self, q_ref, q_cmd, dt) -> tuple[np.ndarray, list[Violation]]:
        """Shorten the step until the flange is inside the box and slow enough.

        Bisection returns the largest fraction it actually tested, never an
        interpolated one, so the emitted command is always known-good rather
        than believed-good. The one exception is q_ref itself: if the arm is
        already outside the box, no fraction is tested at all -- see below.
        """
        env = self.env
        p_ref = tcp_position(q_ref)

        if env.tcp_box.excursion(p_ref) > 0.0:
            # Already outside. Motion cannot fix this and a shortened step
            # would be a guess, so hold and report. This is the one case
            # where the returned fraction is not a tested one, and it is
            # zero (task-6-7-resume.md: the latent gap where lo = 0.0 was
            # seeded as known-good and never actually tested).
            return q_ref.copy(), [Violation("tcp_box", env.tcp_box.excursion(p_ref), 0.0)]

        def ok(q) -> bool:
            p = tcp_position(q)
            if env.tcp_box.excursion(p) > 0.0:
                return False
            return float(np.linalg.norm(p - p_ref)) / dt <= env.tcp_speed_max

        if ok(q_cmd):
            return q_cmd, []

        lo, hi = 0.0, 1.0                      # lo is known good, hi is bad
        for _ in range(env.bisect_iters):
            mid = 0.5 * (lo + hi)
            if ok(q_ref + mid * (q_cmd - q_ref)):
                lo = mid
            else:
                hi = mid
        q_out = q_ref + lo * (q_cmd - q_ref)
        p_bad = tcp_position(q_cmd)
        v = []
        exc = env.tcp_box.excursion(p_bad)
        if exc > 0.0:
            v.append(Violation("tcp_box", exc, 0.0))
        sp = float(np.linalg.norm(p_bad - p_ref)) / dt
        if sp > env.tcp_speed_max:
            v.append(Violation("tcp_speed", sp, env.tcp_speed_max))
        return q_out, v

    def _gripper(self, g, dt) -> tuple[float | None, list[Violation]]:
        """Clamp a gripper request to range and rate. Runs last, and always,
        regardless of what the arm guards decided."""
        env = self.env
        if g is None:
            return self._grip_cmd, []
        v = []
        g = float(g)
        clamped = float(np.clip(g, env.grip_min, env.grip_max))
        if abs(clamped - g) > 1e-12:
            v.append(Violation("grip_range", g, env.grip_max if g > env.grip_max else env.grip_min))
        ref = self._grip_cmd if self._grip_cmd is not None else clamped
        step = env.grip_rate_max * dt
        rated = float(np.clip(clamped, ref - step, ref + step))
        if abs(rated - clamped) > 1e-12:
            v.append(Violation("grip_rate", abs(clamped - ref) / dt, env.grip_rate_max))
        self._grip_cmd = rated
        return rated, v

    # ---- the decision ----------------------------------------------------- #
    def filter(self, state: RobotState, action: Action) -> Verdict:
        env = self.env
        now = float(self._clock())
        dt_raw = 0.0 if self._t_prev is None else now - self._t_prev
        self._t_prev = now

        q_now = np.asarray(state.q, dtype=float)
        q_safe = q_now if np.all(np.isfinite(q_now)) else (
            self._q_cmd if self._q_cmd is not None else np.zeros(6))

        # 0. already latched
        if self._tripped:
            return Verdict(Status.STOP, Action(q=self._q_cmd.copy(), gripper=self._grip_cmd),
                           (Violation("latched", 1.0, 0.0),), dt_raw,
                           {"reason": self._trip_reason})

        # 1. validate
        if not np.all(np.isfinite(q_now)) or not np.all(np.isfinite(state.qd)) \
                or not np.all(np.isfinite(state.wrench)):
            return self._stop(q_safe, [Violation("nan", float("nan"), 0.0)], dt_raw,
                              "non-finite state")
        if not np.all(np.isfinite(action.q)) or (
                action.gripper is not None and not np.isfinite(action.gripper)):
            return self._stop(q_now, [Violation("nan", float("nan"), 0.0)], dt_raw,
                              "non-finite action")

        # first call establishes the reference and commands no motion
        if self._q_cmd is None:
            self._q_cmd = q_now.copy()
            self._grip_cmd = float(state.gripper)
            self._tcp_prev = tcp_position(q_now)
            self._t_src_fresh_at = now
            self._t_src_prev = float(state.t_mono)
            return Verdict(Status.HOLD, Action(q=q_now.copy(), gripper=self._grip_cmd),
                           (), 0.0, {"first_call": True})

        # 2. staleness: is the driver's observation actually advancing?
        #
        # Deliberately never compares state.t_mono against this kernel's clock.
        # Those are two different clocks owned by two different processes and
        # they need not share an epoch; on Windows the default pair differ by
        # seconds. What is meaningful is whether the driver's own stamp moves,
        # and how long this kernel has been waiting for it to move.
        t_src = float(state.t_mono)
        if self._t_src_prev is not None and t_src < self._t_src_prev:
            # a stamp that goes backwards is a driver fault, not mere staleness
            self._stale_n += 1
            v = [Violation("stale", t_src - self._t_src_prev, 0.0)]
            if self._stale_n >= env.stale_escalate_n:
                return self._stop(q_now, v, dt_raw, "observation timestamp went backwards")
            return self._hold(q_now, v, dt_raw)

        if self._t_src_prev is not None and t_src == self._t_src_prev:
            stale_for = now - self._t_src_fresh_at
            if stale_for > env.watchdog_s:
                self._stale_n += 1
                v = [Violation("stale", stale_for, env.watchdog_s)]
                if self._stale_n >= env.stale_escalate_n:
                    return self._stop(q_now, v, dt_raw, "observation stopped advancing")
                return self._hold(q_now, v, dt_raw)
        else:
            self._t_src_fresh_at = now
            self._stale_n = 0
        self._t_src_prev = t_src

        # 3. timestep, measured by this kernel and not by the caller
        if dt_raw > env.max_dt_s:
            return self._hold(q_now, [Violation("dt_max", dt_raw, env.max_dt_s)], dt_raw)
        dt = float(np.clip(dt_raw, env.min_dt_s, env.max_dt_s))

        # 4. tracking: is the arm actually where we last told it to be
        gap = float(np.max(np.abs(self._q_cmd - q_now)))
        if gap > env.tracking_tol_rad:
            self._stale_n += 1
            v = [Violation("tracking", gap, env.tracking_tol_rad)]
            if self._stale_n >= env.stale_escalate_n:
                return self._stop(q_now, v, dt_raw, "arm is not tracking the command")
            return self._hold(q_now, v, dt_raw)

        # 5. contact: excess force or torque stops outright, never shortens
        f = float(np.linalg.norm(state.wrench[:3]))
        t_norm = float(np.linalg.norm(state.wrench[3:]))
        if f > env.force_max:
            return self._stop(q_now, [Violation("force_max", f, env.force_max)],
                              dt_raw, "force envelope exceeded")
        if t_norm > env.torque_max:
            return self._stop(q_now, [Violation("torque_max", t_norm, env.torque_max)],
                              dt_raw, "torque envelope exceeded")

        # 6. cumulative budgets, reset only by rearm -- a budget that resets
        # itself is not a budget. Checked before shaping for the same reason
        # as guard 5: both are grounds to stop outright, not to shorten a step.
        if f > env.contact_force_threshold_n:
            self.contact_s += dt
        self.path_m += float(np.linalg.norm(tcp_position(q_now) - self._tcp_prev)) \
            if self._tcp_prev is not None else 0.0
        self._tcp_prev = tcp_position(q_now)
        if self.contact_s > env.contact_time_budget_s:
            return self._stop(q_now, [Violation("contact_budget", self.contact_s,
                                                env.contact_time_budget_s)],
                              dt_raw, "contact time budget exhausted")
        if self.path_m > env.path_budget_m:
            return self._stop(q_now, [Violation("path_budget", self.path_m,
                                                env.path_budget_m)],
                              dt_raw, "path budget exhausted")

        # 7. kinematic shaping
        q_ref = self._q_cmd.copy()
        q_cmd, viols = self._shape(q_ref, np.asarray(action.q, dtype=float), dt)

        # 8. cartesian box and TCP speed, by conservative bisection
        q_cmd, cviols = self._cartesian(q_ref, q_cmd, dt)
        viols = list(viols) + list(cviols)
        if cviols:
            # the step was shortened after shaping; resynchronise the
            # derivative state or the next call inherits a velocity the arm
            # was never given
            qd_real = (q_cmd - q_ref) / dt
            self._qdd_cmd = (qd_real - self._qd_cmd) / dt
            self._qd_cmd = qd_real
        self._q_cmd = q_cmd

        # 9. gripper position and rate clamp -- always runs, independent of
        # whatever the arm guards above decided
        g_out, gviols = self._gripper(action.gripper, dt)
        viols = viols + gviols

        status = Status.CLAMPED if viols else Status.PASS
        return Verdict(status, Action(q=q_cmd.copy(), gripper=g_out),
                       tuple(viols), dt,
                       {"qd_cmd": self._qd_cmd.copy(), "tcp": tcp_position(q_cmd),
                        "path_m": self.path_m, "contact_s": self.contact_s})
