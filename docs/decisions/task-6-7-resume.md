# Tasks 6 and 7: resume notes. Keep the uncommitted work.

A previous implementer was killed by an account session limit at "Task 6, Step 4:
run tests to verify GREEN". Its work is **uncommitted but present and correct**:

    M  sentinel/kernel.py          (+53 lines: the `_cartesian` guard and its call site)
    ?? tests/test_kernel_cartesian.py   (107 lines)

**Do not discard it.** The controller reviewed the diff: `_cartesian` matches
`task-6-correction.md` exactly, captures `q_ref` before `_shape` overwrites
`self._q_cmd`, bisects for the largest tested fraction, and resynchronises the
derivative state afterwards. It is the right code.

## The five failing tests are NOT its fault

Running the suite with that work in place gives 81 passed, 5 failed:

    tests/test_kernel_shaping.py::test_velocity_is_capped_at_the_declared_limit_not_the_control_rate
    tests/test_kernel_shaping.py::test_commanded_velocity_reaches_zero_at_the_position_limit
    tests/test_kernel_shaping.py::test_an_already_legal_action_passes_through_unchanged
    tests/test_kernel_shaping.py::test_brake_accel_violation_fires_during_the_braking_transition
    tests/test_shield.py::test_the_shield_forwards_a_legal_action_unchanged

All five share one cause, measured by the controller:

```
tcp(zeros)  = [-0.8172, -0.2329, 0.0628]
|xy|        = 0.849740
box         = lo [-0.65, -0.65, 0.05]  hi [0.65, 0.65, 0.90]
excursion   = 0.167200 m        inside = False
```

**Those tests start at `q = zeros`, whose flange is 0.167 m outside the declared
workspace box.** They passed only because no Cartesian guard existed to notice.
Now one does, it correctly refuses to move an arm that is already outside the
box, so "a legal action passes through unchanged" fails exactly as it should.

This is a defect in the plan, not in the guard. The controller should have caught
it in the pre-flight scan: Task 5's tests use `zeros`, Task 6 adds a box guard,
and `zeros` is outside the box.

## The resolution, and why it is not "move the tests inside the box"

The shaping tests ramp every joint toward `+-pi` to exercise the joint position
limit and the braking bound. `q = +-pi` on all joints happens to be *inside* the
box, but the path there is not, so with the guard active the ramp is stopped long
before any joint limit is approached and those tests cannot do their job at all.

They are unit tests for the joint-space shaping chain. They should isolate it.
Give them an effectively unbounded box so the Cartesian guard never binds:

```python
def run(actions, q0=None, env=None):
    # The shaping chain is joint-space. Isolate it from the Cartesian guard with
    # an unbounded box, so these tests exercise the jerk, acceleration, velocity
    # and braking clamps without the box guard stopping the ramp first. The box
    # guard has its own tests in test_kernel_cartesian.py.
    if env is None:
        env = Envelope.ur5e_declared().replace(
            tcp_box=Box(lo=[-10.0, -10.0, -10.0], hi=[10.0, 10.0, 10.0]))
    ...
```

Verified constructible: `Envelope.replace` accepts it, the hash changes, and
excursion is 0.0 at both `q = zeros` and `q = pi`.

For `tests/test_shield.py::test_the_shield_forwards_a_legal_action_unchanged`,
the right fix is different: the Shield test is an integration test and should use
a realistic start. Seed the `FakeRobot`'s plant at a pose verified inside the
box:

```python
SHIELD_Q0 = np.array([0.0, -1.2, 1.2, -1.5, -1.5708, 0.0])
# tcp = [-0.638607, -0.1333, 0.452214], excursion 0.0, inside the declared box.
```

Only that one Shield test needs it; the others assert refusals and are unaffected
either way, but seeding the plant once for the whole file is cleaner than
special-casing.

## One latent issue in the guard, worth closing while you are here

`_cartesian` seeds `lo = 0.0` as known-good and never tests it. When `q_ref` is
already outside the box that assumption is false: the guard returns `q_ref` and
reports a `tcp_box` violation rather than refusing outright.

Holding position when you are already outside is arguably the correct action,
since moving cannot fix it and stopping is the fail-safe. But the docstring
claims the guard "returns the largest fraction it actually tested", and 0.0 is
never tested, so the code and its documentation disagree.

Resolve it explicitly rather than leaving it implicit:

```python
        if not ok(q_ref):
            # Already outside. Motion cannot fix this and a shortened step would
            # be a guess, so hold and report. This is the one case where the
            # returned fraction is not a tested one, and it is zero.
            return q_ref.copy(), [Violation("tcp_box",
                                            env.tcp_box.excursion(p_ref), 0.0)]
```

and add a test that a kernel started outside the box holds rather than moving.

## What remains

Task 6: finish the tests, apply the two fixes above, get the suite green, commit.
Task 7 has not been started at all. Both briefs and both notes still apply
unchanged.
