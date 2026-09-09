# Task 5 correction: the plan's shaping chain is wrong. Use this instead.

**This file overrides the `_shape` code in the Task 5 brief.** The brief's
version was measured against its own test and fails it. Everything else in the
Task 5 brief still stands: the same tests, the same file, the same call site.

## What the controller measured

The brief's chain clamps jerk, then acceleration, then velocity, then applies a
continuous braking bound `sqrt(2*a*room)`, then clips position. Driven with the
brief's own `test_acceleration_and_jerk_are_bounded` ramp, the derivatives of
the emitted command sequence come out at:

| quantity | brief's chain | declared limit | verdict |
|---|---|---|---|
| max abs qd | 1.000000 | 1.0 | ok |
| max abs qdd | 8.808387 | 5.0 | **fails** |
| max abs qddd | 174.846820 | 100.0 | **fails** |

Three separate causes, all in the braking transition:

1. **The continuous bound shrinks faster than the acceleration limit allows.**
   Near the wall, `sqrt(2*a*room)` drops by more than `a*dt` in one step, so the
   step has to shed more velocity than `qdd_max` permits.
2. **The final position clip breaks the derivative contract.** When
   `clip(q_ref + qd*dt, q_min, q_max)` bites, the emitted velocity is no longer
   the clamped one, so the emitted acceleration is not the clamped one either.
3. **Velocity saturation forces an acceleration step change.** When `qd` reaches
   `qd_max`, acceleration must fall from 5.0 to 0 in a single step. That is a
   jerk of `qdd_max/dt` = 150, over the 100 limit, and no amount of clamping
   downstream can undo it.

## The fix, in three parts

1. **Settle the velocity target first, then shape the approach to it.** The old
   order clamped derivatives and then recomputed the velocity from a clipped
   position, so the number that was clamped was not the number that was emitted.
2. **Use a discrete-exact braking bound**, and cap it by `room/dt` so the final
   position clip never bites.
3. **Apply the same braking bound one derivative up**: acceleration must be able
   to return to zero before velocity reaches `qd_max`, for exactly the same
   reason velocity must be able to return to zero before position reaches
   `q_max`. This is the part the brief missed entirely.

A headroom factor is needed on both bounds because each bound assumes full
authority at the next derivative level is available instantly, and the limit one
level up means it is not. Measured sweep of the headroom factor, worst case over
both directions and six seeded start states:

| kv | ka | max abs qdd | max abs qddd | verdict |
|---|---|---|---|---|
| 1.0 | 1.0 | 10.297802 | 272.8101 | violates both |
| 0.8 | 1.0 | 5.000000 | 110.0899 | violates jerk |
| **0.8** | **0.8** | **5.000000** | **100.0000** | **exact on every limit** |
| 0.6 | 0.6 | 5.000000 | 82.3063 | passes, more conservative than needed |

`0.8` on both is the largest value that satisfies every limit exactly, so it is
the least conservative correct choice. Add `brake_headroom: float = 0.8` to the
`Envelope` with provenance `"declared"`.

## Verified behaviour at kv = ka = 0.8

- max abs qd = 1.000000 against a 1.0 limit
- max abs qdd = 5.000000 against a 5.0 limit
- max abs qddd = 100.0000 against a 100.0 limit
- commands stay inside the position limits, and the final position clip never bites
- arrival velocity at `q_max` is 0.000000000, so the braking test passes
- reaches `q_max` from rest in 101 steps, so the brief's 400-step tests have margin
- holds under alternating full scale, degrees-into-radians, and uniformly random
  targets in [-50, 50]: all three give exactly qd 1.0, qdd 5.0, qddd 100.0

## The code

```python
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
        qdd_target = np.clip(np.clip(qdd_want, -qdd_lo, qdd_hi), -env.qdd_max, env.qdd_max)
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
```

## Two changes to the Task 5 tests

**1. `test_acceleration_and_jerk_are_bounded` must not prepend zeros.** The
brief writes:

```python
    qd = np.diff(np.vstack([np.zeros(6), cmds]), axis=0) / DT
    qdd = np.diff(np.vstack([np.zeros(6), qd]), axis=0) / DT
```

Prepending a zero row asserts the run starts from rest. That is true for these
particular tests, so it is harmless here, but it manufactures a spurious spike
of `qd_max/dt` = 30 the moment anything starts in motion, and the property tests
in Task 9 do start in motion. Take the derivatives of the emitted sequence
directly instead, in every test that measures them:

```python
    qd = np.diff(cmds, axis=0) / DT
    qdd = np.diff(qd, axis=0) / DT
    qddd = np.diff(qdd, axis=0) / DT
```

**2. Add this test**, which is the one that would have caught the original bug:

```python
def test_the_emitted_derivatives_obey_every_limit_through_the_braking_transition():
    # The braking transition is where a nested clamp goes wrong: the bound
    # shrinks faster than the acceleration limit allows, the position clip
    # bites, and the emitted derivative stops matching the clamped one.
    cmds, _, env = run([np.full(6, 10.0)] * 400)
    qd = np.diff(cmds, axis=0) / DT
    qdd = np.diff(qd, axis=0) / DT
    qddd = np.diff(qdd, axis=0) / DT
    assert np.abs(qd).max() <= env.qd_max.max() + 1e-6
    assert np.abs(qdd).max() <= env.qdd_max.max() + 1e-6
    assert np.abs(qddd).max() <= env.qddd_max.max() + 1e-6
    assert np.all(cmds <= env.q_max + 1e-12) and np.all(cmds >= env.q_min - 1e-12)
```

## One consequence for the spec, which the controller will record

Invariant (A) is conditioned on a **stoppable** initial state, meaning
`qd^2 <= 2 * qdd_max * (distance to the nearer joint limit)` on every joint. A
state that starts closer to a limit than its own velocity can brake within is
outside the envelope's premise, and no acceleration-limited controller can
rescue it: it is already committed to an overshoot before the kernel sees it.
The kernel still does the best available thing, braking at `qdd_max` and
reporting the violation, but it cannot promise (A) from there. Task 9's
property-test strategies must therefore generate stoppable initial states, and
this goes in `LIMITATIONS.md`.
