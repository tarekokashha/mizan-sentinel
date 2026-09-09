# Task 8 note: the brief is correct as written. Do not retune the plant.

No correction needed. The controller measured all three claims the Task 8 tests
assert, before dispatch, and every one holds at the brief's default settings
(`kp=400.0`, `zeta=1.0`, `accel_authority=1.5`, `substeps=10`):

| claim | measured | test threshold | verdict |
|---|---|---|---|
| converges to a held command | q 0.40000000, qd 0.00000000 | within 1e-3 | holds |
| feasible ramp, no overshoot | peak 0.50000000, overshoot 0.0 | 0.5 + 0.02 | holds |
| infeasible step overshoots | peak 1.36535138, first at step 16 | above 1.02 | holds |

## Why this is worth telling you

The third claim is the one the whole design rests on. If the plant tracked an
infeasible command perfectly, `test_an_infeasible_step_overshoots` would fail,
and worse, the braking bound in Task 5 would have no test that could ever fail,
so the most important guard in the kernel would be unfalsifiable. The brief says
as much in a comment. It is now measured rather than hoped for.

## The settings are load bearing. A sweep shows how narrow the window is.

At `zeta = 1.0`, lowering the proportional gain destroys the overshoot entirely
and would silently break claim 3:

| zeta | kp | authority | peak | overshoots? |
|---|---|---|---|---|
| 1.0 | 400 | 1.5 | 1.36535138 | yes |
| 1.0 | 100 | 1.5 | 1.00000909 | **no** |
| 1.0 | 25 | 1.5 | 0.99999987 | **no** |

And in the other direction, dropping the damping ratio breaks claim 2 instead:

| zeta | feasible-ramp peak | within the 0.52 margin? |
|---|---|---|
| 1.0 | 0.50000000 | yes |
| 0.7 | 0.50922129 | yes |
| 0.5 | 0.52784011 | **no** |

So `kp = 400.0` with `zeta = 1.0` sits in the only region where claims 2 and 3
both hold. Use the brief's values exactly. If a test fails and you are tempted to
adjust `kp`, `zeta`, `accel_authority` or `substeps` to make it pass, stop and
report instead: those four numbers are jointly constrained by two tests pulling
in opposite directions, and moving one to satisfy one test will break the other.

## One thing to keep

The semi-implicit integration order in the brief (update `qd` from the
acceleration, then update `q` from the *new* `qd`) is deliberate. Explicit Euler
at 30 Hz with `kp = 400` is marginally unstable. Keep the order, and keep
`substeps = 10`.
