"""Lifecycle guards: validate, staleness, timestep, tracking, latch, rearm.

No motion shaping yet (that is Task 5 onward) - a passing action is emitted
unchanged. Every test injects its own fake `Clock`, so the kernel's clock and
the driver's timestamp are both under the test's control.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel.envelope import Envelope
from sentinel.kernel import SafetyKernel
from sentinel.types import Action, RobotState, Status


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def tick(self, dt):
        self.t += dt
        return self.t


def kernel(**kw):
    env = Envelope.ur5e_declared().replace(**kw) if kw else Envelope.ur5e_declared()
    c = Clock()
    return SafetyKernel(env, clock=c), c, env


def state(q=None, qd=None, t=0.0, wrench=None, grip=0.0):
    return RobotState(
        q=np.zeros(6) if q is None else q,
        qd=np.zeros(6) if qd is None else qd,
        t_mono=t,
        wrench=np.zeros(6) if wrench is None else wrench,
        gripper=grip,
    )


def test_a_small_legal_step_passes_through():
    k, c, _ = kernel()
    k.filter(state(t=c.t), Action(q=np.zeros(6)))
    c.tick(1 / 30)
    v = k.filter(state(t=c.t), Action(q=np.full(6, 0.005)))
    assert v.status in (Status.PASS, Status.CLAMPED)
    assert np.all(np.isfinite(v.action.q))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_action_latches_stop_and_never_passes_through(bad):
    k, c, _ = kernel()
    a = np.zeros(6)
    a[3] = bad
    v = k.filter(state(t=c.t), Action(q=a))
    assert v.status is Status.STOP
    assert np.all(np.isfinite(v.action.q))
    assert any(x.rule == "nan" for x in v.violations)
    assert k.tripped
    c.tick(1 / 30)
    assert k.filter(state(t=c.t), Action(q=np.zeros(6))).status is Status.STOP


def test_rearm_clears_the_latch_and_resets_the_budgets():
    k, c, _ = kernel()
    k.filter(state(t=c.t), Action(q=np.full(6, np.nan)))
    assert k.tripped
    k.rearm("operator cleared the cell")
    assert not k.tripped and k.path_m == 0.0 and k.contact_s == 0.0


def test_rearm_without_a_reason_raises():
    # Decision #4: recovery from a safety stop is a deliberate human act, not
    # a default. A rejected rearm must not clear the latch it failed to lift.
    k, c, _ = kernel()
    k.filter(state(t=c.t), Action(q=np.full(6, np.nan)))
    assert k.tripped
    with pytest.raises(ValueError):
        k.rearm("")
    assert k.tripped


def test_a_stale_state_holds_then_escalates_to_stop():
    # NOTE (deviation from task-4-brief.md, see task-4-report.md): the brief
    # ticks by `env.watchdog_s * 0.5` here. That can never make the very
    # first affected call exceed the watchdog threshold, under either the
    # brief's original age formula or the corrected one - 0.5x a value can
    # never be strictly greater than that value - so the escalation this
    # test names never happens within `stale_escalate_n` iterations. Ticking
    # past the watchdog on every iteration (1.5x) is what the test's name
    # and assertions actually require.
    k, c, env = kernel()
    k.filter(state(t=c.t), Action(q=np.zeros(6)))
    seen = []
    for _ in range(env.stale_escalate_n):
        c.tick(env.watchdog_s * 1.5)          # kernel clock advances past the watchdog
        seen.append(k.filter(state(t=0.0), Action(q=np.full(6, 0.01))).status)
    assert seen[:-1] == [Status.HOLD] * (env.stale_escalate_n - 1)
    assert seen[-1] is Status.STOP


def test_one_fresh_sample_resets_the_stale_counter():
    # NOTE (deviation, see task-4-report.md): rewritten from the brief on two
    # counts.
    #
    # 1. Its tick amounts (0.05s against a 0.2s watchdog) can never make a
    #    frozen sample exceed the watchdog, for the same arithmetic reason as
    #    the test above, so its HOLD assertions would see PASS instead.
    #
    # 2. Structurally, two trips separated by one fresh call can never tell a
    #    working reset apart from a broken one when stale_escalate_n=3: 1
    #    trip, fresh, 1 more trip reaches stale_n=2 either way (2 without a
    #    reset, 0-then-1 with one) and both are just HOLD. To actually prove
    #    the reset fires, drive the counter to exactly one below escalation,
    #    let one fresh sample intervene, then trip once more: with a working
    #    reset that lands on stale_n=1 (HOLD); without one it would land on
    #    stale_n=3 and return STOP instead.
    k, c, env = kernel()
    k.filter(state(t=c.t), Action(q=np.zeros(6)))
    for _ in range(env.stale_escalate_n - 1):
        c.tick(env.watchdog_s * 1.5)
        assert k.filter(state(t=0.0), Action(q=np.zeros(6))).status is Status.HOLD
    c.tick(0.05)
    t_fresh = c.t
    k.filter(state(t=t_fresh), Action(q=np.zeros(6)))      # fresh: resets the counter
    c.tick(env.watchdog_s * 1.5)
    v = k.filter(state(t=t_fresh), Action(q=np.zeros(6)))  # frozen again, one trip only
    assert v.status is Status.HOLD


def test_a_driver_clock_with_a_different_epoch_is_not_treated_as_stale():
    # The kernel's clock and the driver's clock are different clocks owned by
    # different processes. On Windows the default pair differ by seconds. A
    # kernel that compares them latches STOP against a healthy driver.
    k, c, _ = kernel()
    offset = 5_000_000.0                      # driver epoch, wildly different
    k.filter(state(t=offset), Action(q=np.zeros(6)))
    for i in range(1, 10):
        c.tick(1 / 30)
        v = k.filter(state(t=offset + i / 30), Action(q=np.full(6, 0.001 * i)))
        assert v.status is not Status.STOP, f"healthy driver rejected at step {i}"
    assert not k.tripped


def test_a_frozen_driver_stamp_is_still_caught_regardless_of_epoch():
    k, c, env = kernel()
    offset = 5_000_000.0
    k.filter(state(t=offset), Action(q=np.zeros(6)))
    seen = []
    for _ in range(env.stale_escalate_n + 2):
        c.tick(env.watchdog_s)
        seen.append(k.filter(state(t=offset), Action(q=np.full(6, 0.01))).status)
    assert Status.HOLD in seen
    assert seen[-1] is Status.STOP


def test_an_over_long_timestep_holds():
    k, c, env = kernel()
    k.filter(state(t=c.t), Action(q=np.zeros(6)))
    c.tick(env.max_dt_s * 2)
    v = k.filter(state(t=c.t), Action(q=np.full(6, 0.01)))
    assert v.status is Status.HOLD
    assert any(x.rule == "dt_max" for x in v.violations)


def test_the_kernel_ignores_the_caller_supplied_timestamp_for_dt():
    # dt_spoof: the state claims a tiny timestep, the kernel measures its own.
    k, c, _ = kernel()
    k.filter(state(t=c.t), Action(q=np.zeros(6)))
    c.tick(0.05)
    v = k.filter(state(t=c.t), Action(q=np.full(6, 0.01)))
    assert v.dt == pytest.approx(0.05, rel=1e-9)


def test_divergence_between_command_and_measurement_holds_and_resyncs():
    k, c, env = kernel()
    k.filter(state(t=c.t), Action(q=np.zeros(6)))
    c.tick(1 / 30)
    far = np.full(6, env.tracking_tol_rad * 3)
    v = k.filter(state(q=far, t=c.t), Action(q=far))
    assert v.status is Status.HOLD
    assert any(x.rule == "tracking" for x in v.violations)
    assert np.allclose(v.action.q, far)      # resynchronised to the measurement


def test_the_first_call_establishes_a_reference_without_commanding_motion():
    k, c, _ = kernel()
    q0 = np.full(6, 0.3)
    v = k.filter(state(q=q0, t=c.t), Action(q=np.full(6, 2.0)))
    assert np.allclose(v.action.q, q0)
