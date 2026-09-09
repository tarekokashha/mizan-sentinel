# Task 10 fix round 2: three attacks cannot trip the guard they declare

Found by sweeping every attack's requested actions against every limit, which
nobody had done. Two attacks declare a guard they provably never reach, and one
declares a guard it cannot reach within the pre-registered episode length.

All three are defects in the controller's own attack catalogue. The
implementation transcribed it faithfully.

## Finding 1: `index_permute` and `sign_flip` declare `q_limit` and never reach it

Driven for 600 steps from the episode start, both were measured:

| attack | worst joint-limit excess | worst requested abs(qd) |
|---|---|---|
| `index_permute` | **0.000000 rad** | 101.60 rad/s |
| `sign_flip` | **0.000000 rad** | 87.00 rad/s |

Against `qd_max = 1.0`. Neither ever requests a configuration outside
`[q_min, q_max]`, so `q_limit` can never fire for either. Both massively violate
the velocity limit instead.

`index_permute` shuffles a goal drawn from `[-2.5, 2.5]`, which is inside the
declared `+-pi` range by construction. `sign_flip` negates a state that is already
inside the limits, and the negation of an in-range value is in range. Both were
declared against `q_limit` on the assumption that scrambling or negating an
action produces an out-of-range target. It does not; it produces an in-range
target reached at an impossible speed.

**Fix:** change the declared guard on both to `"qd_max"`. The attacks themselves
are correct and stay exactly as they are; only the `expect` string is wrong.

```python
@attack("index_permute", "qd_max", "the six joints arrive shuffled")
@attack("sign_flip", "qd_max", "the whole action is negated")
```

Update their `note` text if it implies a position violation.

## Finding 2: `slow_drift` cannot exhaust the declared budget in an episode

Measured over a full 2000-step episode from the episode start:

```
path accumulated              =  8.467 m
declared path_budget_m        = 25.0 m
shortfall factor              =  2.95x
```

It cannot trip `path_budget`. The Task 7 test that "verified" this guard used a
**reduced** budget of 0.5 m, so it proved the guard works without proving this
attack can reach it. That is the same mistake as the oscillation test's
out-of-box poses: a measurement taken against a different configuration than the
one that ships.

**Fix:** widen and slow the oscillation so it accumulates enough path within the
pre-registered episode, while remaining legal at every step. Do NOT change
`path_budget_m`; it is a declared envelope value and the episode length is
pre-registered.

Measured sweep, path over 2000 steps, staying inside the box throughout:

| amp | period | frac | path (m) | worst abs(qd) | in box | reaches 25 m |
|---|---|---|---|---|---|---|
| 0.30 | 30 | 0.90 | 13.884 | 0.900 | yes | no |
| 0.60 | 30 | 0.90 | 27.768 | 0.900 | yes | yes |
| 0.90 | 30 | 0.90 | **41.442** | 0.900 | yes | yes |
| 1.20 | 60 | 0.90 | 27.348 | 0.900 | yes | yes |

**Use `amp = 0.9`, `period = 30`, `frac = 0.9`**, giving 41.442 m against a 25.0 m
budget, a 66 percent margin, worst requested velocity 0.900 rad/s against a 1.0
limit, and zero box excursion throughout.

### Why the largest margin and not the tightest fit

This measurement is **open loop**. It clips only to joint limits and does not run
the kernel, so it does not include the jerk, acceleration and Cartesian shaping
that will reduce the achieved motion. The real accumulated path will be **lower
than these numbers**, by an amount this probe cannot predict.

That is exactly the error that produced this finding in the first place, and the
one before it: a controller measurement taken against a component behaves
differently once composed with guards that did not exist when it was taken. So
pick the setting with the most headroom rather than the one that just clears the
bar, and **verify the real figure against the assembled kernel** before you
believe it.

### Verify it properly

Add a test that runs the real thing, not the open-loop approximation:

```python
def test_slow_drift_actually_exhausts_the_declared_path_budget():
    # This attack's declared guard is path_budget, and the Task 7 test proved
    # only that the guard works, using a reduced budget. This proves the attack
    # can reach the DECLARED budget within the pre-registered episode length,
    # through the real kernel with every guard active.
    env = Envelope.ur5e_declared()
    clock = _Clock()
    kernel = SafetyKernel(env, clock=clock)
    plant = SimPlant(q0=EPISODE_Q0, qdd_max=env.qdd_max)
    fn = build_all(env, np.random.default_rng(0))["slow_drift"]
    tripped_at = None
    for i in range(2000):
        state = plant.state(t_mono=clock.t)
        v = kernel.filter(state, fn(state, i, clock.t))
        if v.status is Status.STOP and any(x.rule == "path_budget" for x in v.violations):
            tripped_at = i
            break
        plant.step(v.action.q, DT)
        clock.tick(DT)
    assert tripped_at is not None, (
        f"slow_drift never exhausted the {env.path_budget_m} m budget in 2000 steps; "
        f"reached {kernel.path_m:.3f} m")
```

Report the step it trips at and the final `path_m`. If it does not trip even at
`amp = 0.9`, report that rather than inflating the amplitude further without
saying so: it would mean the shaping reduces the path far more than expected and
that is worth knowing on the record.

## Not in scope

Eight attacks stray outside the workspace box on the way to their real target and
will therefore report `tcp_box` alongside their declared guard. That is harmless:
`expected_guard_fired` checks the declared guard is **among** those that fired,
not that it is the only one. Do not try to make them Cartesian-clean. The one
exception is `lever_sprint`, already handled by the `start_q` seeding in
`task-11-correction.md`.
