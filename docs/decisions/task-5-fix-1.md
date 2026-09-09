# Task 5 fix round 1: the failing test is wrong, the kernel is right

The implementer reported `test_braking_distance_is_respected_throughout_the_approach`
failing, declined to weaken `_shape`, `_brake_bound` or `brake_headroom` to force
a pass, and argued the test is unsatisfiable. **That judgement was correct and
the controller has verified it against the shipped kernel.**

## The evidence

Driving the real `SafetyKernel` for 400 steps toward `q = 10.0`:

| formulation | violations |
|---|---|
| room **after** the step, as the brief wrote it | 36 of 2394 |
| room **before** the step, the correct form | **0 of 2394** |

First failure, step 96, joint 0:

```
qd                            = 0.579308728
room after the step  = 0.032294680  -> allowed 0.568284083
room before the step = 0.051604971  -> allowed 0.718365999
distance travelled this step  = 0.019310291
```

The velocity that *traversed* the step is being compared against the room left
*once it landed*. Any controller sizes velocity from the room it had before
moving.

It is not merely wrong, it is unsatisfiable. On the step that first touches the
limit, room after is exactly 0, so the allowed velocity is exactly 0, while the
step still has to cover the remaining gap. Only an arm that teleports the last
increment at zero speed could pass.

Separately confirmed: arrival velocity is 0.000000000, the final command lands on
`q_max` exactly, and the limit is first reached at step 101, matching the
controller's independent pre-implementation measurement of 101 steps.

## Fix 1, Important: correct the test

Compare against the room available before each step, and take the derivative from
the emitted sequence without a zero-prepend.

```python
def test_braking_distance_is_respected_throughout_the_approach():
    # The bound is a statement about the room the controller HAD when it sized
    # the step, not the room left once the step landed. Comparing against the
    # latter is unsatisfiable: on the step that first touches the limit the
    # remaining room is zero, so the permitted velocity is zero, while the step
    # still has to cover the last of the distance.
    cmds, _, env = run([np.full(6, 10.0)] * 400)
    qd = np.diff(cmds, axis=0) / DT
    room_before = env.q_max - cmds[:-1]
    allowed = np.sqrt(2 * env.qdd_max * np.maximum(room_before, 0.0))
    assert np.all(qd <= allowed + 1e-6)
```

## Fix 2, Minor but it is about auditability: the acceleration-level brake is silent

The implementer flagged that the acceleration-level braking clamp emits no
`Violation`, while its position-level counterpart emits `"brake"`. That asymmetry
is real and worth closing: this project's whole claim rests on every clamp being
visible in the journal, and a guard that shapes a command without saying so is a
guard whose effect cannot be audited after the fact.

Add a distinct rule id so the two levels are separable in telemetry:

```python
        qdd_target = np.clip(qdd_want, -qdd_lo, qdd_hi)
        if np.any(np.abs(qdd_target - qdd_want) > 1e-12):
            i = int(np.argmax(np.abs(qdd_target - qdd_want)))
            v.append(Violation("brake_accel", float(qdd_want[i]),
                               float(qdd_hi[i] if qdd_want[i] > 0 else -qdd_lo[i]), i))
        qdd_target = np.clip(qdd_target, -env.qdd_max, env.qdd_max)
```

`"brake"` stays the position-level bound capping velocity. `"brake_accel"` is the
velocity-level bound capping acceleration. Add `"brake_accel"` to the rule-id list
in the module docstring if that list is present, and add a test asserting it fires
during the braking transition of the 400-step ramp.

## Fix 3: confirm the unrelated test bug fix was right

The implementer reported repairing `test_velocity_is_capped_at_the_declared_limit_not_the_control_rate`,
which compared a scalar against a 6-vector and raised `ValueError` regardless of
the algorithm. Confirm the repaired assertion still tests what its name claims:
that the per-step change is bounded by `qd_max * DT` and is not absurdly tighter,
which is the entire point, since the defect this task exists to fix was a clamp
roughly 16 times too tight.

## Do not change

`_shape`, `_brake_bound`, and `brake_headroom = 0.8` are correct as shipped and
verified. The three measured maxima are exactly at their limits. Leave them
alone.
