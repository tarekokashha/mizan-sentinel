# Task 12 note: the brief is correct, and here is why its timestamp handling is

No corrections. One thing to understand rather than change, because it looks
wrong and is not.

## The Shield must pass the driver's own timestamp through, unmodified

The brief has:

```python
            t_mono=float(obs.get("timestamp_monotonic", 0.0)),
```

That looks like it trusts an untrusted number. It does not, and it is correct.

Task 4's staleness guard was rewritten (see `task-4-correction.md`) after the
controller measured that the original compared the kernel's clock against the
driver's. On Windows `time.monotonic()` is `GetTickCount64()` and
`time.perf_counter()` is `QueryPerformanceCounter()`, and on this machine they
read 5.558460 seconds apart. The UR driver stamps with `perf_counter`. Comparing
them made the guard fire on every call and latch a permanent STOP against a
perfectly healthy arm.

The rewritten guard never compares the two clocks. It compares the driver's
**successive stamps to each other**, and measures how long they have failed to
advance using the kernel's own clock. That is epoch independent, which is exactly
why the Shield should hand the driver's raw stamp straight through: the kernel
only ever asks "is this number bigger than the last one you gave me".

Two consequences worth knowing:

- **Do not** convert, offset, or re-stamp the observation's timestamp with the
  kernel's clock. Doing so would destroy the only signal the watchdog has, namely
  whether the driver's own clock is advancing, and would make a frozen driver
  look perfectly healthy.
- The `0.0` default when `timestamp_monotonic` is missing is correct fail-closed
  behaviour, not laziness. A driver that never supplies a stamp produces a frozen
  0.0 on every call, which the guard reads as an observation that has stopped
  advancing, and it holds and then stops. That is the right answer for a driver
  that cannot say when it sampled.

Add a test for that second point, since it is behaviour the brief relies on and
does not cover:

```python
def test_a_robot_that_supplies_no_timestamp_eventually_fails_closed():
    # A driver that cannot say when it sampled is a driver whose freshness
    # cannot be established. The missing-key default of 0.0 reads as a frozen
    # stamp, so the watchdog holds and then latches, which is the correct
    # answer rather than an oversight.
    r = FakeRobot()
    r.get_observation = lambda: {"joint_position": r.plant.q,
                                 "joint_velocity": r.plant.qd}   # no timestamp
    s = Shield(r, SafetyKernel(r.env, clock=Clock(r)))
    seen = []
    for _ in range(6):
        s.get_observation()
        s.send_action({"joint_position": np.zeros(6)})
        seen.append(s.last_verdict.status)
        r.t += 1 / 3          # advance well past watchdog_s = 0.2
    assert Status.STOP in seen
```

Adjust the fake to match whatever shape your `FakeRobot` ends up with; the point
is the missing key, not the exact plumbing.

## One real check the brief does cover but is worth doing carefully

`send_action` sets `self._last_state = None` after forwarding, so every control
cycle must call `get_observation` first. The brief's
`test_send_action_requires_an_observation_first` covers the first call. Make sure
it also holds on the *second* cycle, meaning two `send_action` calls in a row
without an intervening observation must raise. A kernel judging an action against
a stale state is precisely the failure mode this project exists to prevent, and
the one-shot version of that test would pass even if the reset were removed.
