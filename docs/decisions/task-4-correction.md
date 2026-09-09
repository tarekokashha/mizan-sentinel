# Task 4 correction: the staleness guard compares two different clocks, and the
# default clock is too coarse to run at 30 Hz

**This overrides the clock handling and the staleness guard in the Task 4
brief.** Everything else stands.

## The measurements

Run on this machine, Windows 11, Python 3.11.15:

```
time.monotonic()     = 200567.484000
time.perf_counter()  = 200573.042460
offset (mono - perf) = -5.558460 s

monotonic     info: implementation='GetTickCount64()',        resolution=0.015625
perf_counter  info: implementation='QueryPerformanceCounter()', resolution=1e-07
```

## Bug 1: the staleness guard compares clocks that do not share an epoch

The brief has:

```python
        age = now - float(state.t_mono)
        if age > env.watchdog_s or age < 0.0:
```

`now` comes from the kernel's clock, defaulting to `time.monotonic`. `state.t_mono`
comes from the driver. The driver this project exists to wrap,
`lerobot_ur/robot_ur5e.py`, stamps observations with `time.perf_counter()`:

```python
        t = time.perf_counter()
        obs = { ..., "timestamp_monotonic": t }
```

On Windows those are two different clocks with two different origins, measured
5.558 seconds apart above. So `age` is about `-5.56`, the `age < 0.0` branch
fires on every call, and after `stale_escalate_n` = 3 calls the kernel latches a
permanent `STOP`.

The kernel would refuse to operate against the exact driver it was written for,
and every test in the plan passes anyway, because every test injects one fake
clock and uses it for both sides. This is a bug that only exists in production
and is invisible to the entire suite. That is the worst kind.

Failing closed is the right direction and is small comfort: a safety layer that
always stops is not a safety layer.

## Bug 2: `time.monotonic()` cannot measure a 30 Hz control period on Windows

Its resolution is **15.625 ms**, which is 47 percent of a 33.3 ms control period.
`dt` would quantise to roughly 0, 15.6, or 31.2 ms with nothing in between. The
shaping chain divides by `dt` three times over, so a `dt` that is wrong by half
makes acceleration and jerk wrong by four and eight times respectively. Every
guarantee in Task 5 rests on `dt` being accurate.

`time.perf_counter()` has 100 ns resolution, five orders of magnitude better, and
is what the driver already uses.

## The fix, two parts

### Part 1: default to `perf_counter`

```python
    def __init__(self, envelope: Envelope, clock: Callable[[], float] = time.perf_counter):
```

Add a comment saying why, because the obvious-looking `time.monotonic` is wrong
here and someone will try to "fix" it back:

```python
        # time.perf_counter, not time.monotonic. On Windows, monotonic is
        # GetTickCount64 with 15.625 ms resolution, which is half a control
        # period at 30 Hz, and the shaping chain divides by dt three times.
        # perf_counter is QueryPerformanceCounter at 100 ns, and it is also the
        # clock the UR driver stamps its observations with.
```

### Part 2: make the staleness guard epoch independent

Do not compare the driver's stamp to the kernel's clock at all. Compare the
driver's successive stamps **to each other**, and measure how long they have
failed to advance using the kernel's own clock. That is exactly what a watchdog
should measure and it does not care what epoch the driver uses.

Replace the staleness block with:

```python
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
```

Initialise in `__init__`:

```python
        self._t_src_prev: float | None = None
        self._t_src_fresh_at: float = 0.0
```

and set `self._t_src_fresh_at = now`, `self._t_src_prev = float(state.t_mono)` in
the first-call branch. `rearm()` resets `_t_src_prev` to `None`.

## Tests to change

The brief's `test_a_stale_state_holds_then_escalates_to_stop` and
`test_one_fresh_sample_resets_the_stale_counter` both pass a frozen
`t_mono=0.0`. Under the new guard a frozen stamp is still detected, so keep both,
but the second one must advance the stamp to show freshness rather than relying
on the shared epoch. Adjust so the "fresh" call passes a *larger* `t_mono` than
the previous one, whatever its absolute value.

Add these two, which are the ones that would have caught the bug:

```python
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
```

## One more thing, and it is small

The brief imports `worst` from `sentinel.types` into `kernel.py` and never uses
it. Drop it from the import list. If a later guard needs to combine statuses it
can be re-added then.
