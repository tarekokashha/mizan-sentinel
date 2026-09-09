# Task 11 correction: three fixes to the red-team runner

**This overrides three specific things in the Task 11 brief.** Everything else in
that brief stands.

## 1. `watchdog_starve` cannot trip the guard it declares

The brief gives `watchdog_starve` `skip_steps=1` and has the runner tick
`DT * (1 + spec.skip_steps)` on marked steps. That is `2 * (1/30)` = 0.0667 s,
which is well below `max_dt_s` = 0.2 s. The `dt_max` guard can never fire, so
`test_every_attack_actually_trips_the_guard_it_declared` fails for this attack.

An attack that cannot reach its own guard is worse than no attack: it reports a
clean run and proves nothing.

Fix. In `sentinel/attacks.py`, replace the `skip_steps: int = 0` field on
`AttackSpec` with:

```python
    skip_seconds: float = 0.0   # extra wall time to burn on marked steps
```

and declare the attack as:

```python
@attack("watchdog_starve", "dt_max", "stop calling for longer than the watchdog",
        skip_seconds=0.5)          # must exceed max_dt_s = 0.2
def _watchdog_starve(env, rng):
    ...
```

In `sentinel/redteam.py`, the tick becomes:

```python
        extra = spec.skip_seconds if (spec.skip_seconds and step % 20 == 19) else 0.0
        clock.tick(DT + extra)
```

Assert in the attacks test that the declared value actually exceeds the limit it
targets, so this class of mistake cannot recur silently:

```python
def test_watchdog_starve_actually_exceeds_the_watchdog():
    env = Envelope.ur5e_declared()
    assert REGISTRY["watchdog_starve"].skip_seconds > env.max_dt_s, (
        "the attack cannot reach the guard it declares")
```

## 2. The episode start state must be verified, not assumed

`run_episode` picks `q0 = [0.0, -1.2, 1.2, -1.5, -1.57, 0.0]` and a
`PlaneContact(z=0.10)` without ever checking either against the envelope. If the
start is outside the box, every attack reports an escape at step 0 and the whole
run is meaningless.

The controller measured both, so these values are confirmed good and should be
kept:

- `tcp(q0)` = `[-0.6386, -0.1333, 0.4522]`, inside the declared box
  (lo `[-0.65, -0.65, 0.05]`, hi `[0.65, 0.65, 0.90]`), with 0.011 m of margin
  on x.
- `force_grind`'s `q[1] += 0.02` sweep drives the flange down to z = -0.479, so a
  contact plane at z = 0.10 is comfortably reachable, and 0.10 sits just above
  the box floor of 0.05.

Keep the values, but make the runner check rather than trust. At the top of
`run_episode`, before step 0:

```python
    q0 = np.array([0.0, -1.2, 1.2, -1.5, -1.57, 0.0])
    # An episode that starts outside the envelope reports an escape on step 0
    # for every attack and proves nothing. Verified: tcp(q0) is
    # [-0.6386, -0.1333, 0.4522], inside the declared box.
    if np.any(q0 > env.q_max) or np.any(q0 < env.q_min):
        raise ValueError(f"episode start {q0} is outside the joint limits")
    if env.tcp_box.excursion(tcp_position(q0)) > 0.0:
        raise ValueError(f"episode start puts the flange outside the box: {tcp_position(q0)}")
```

and add:

```python
def test_the_episode_start_state_is_inside_the_envelope():
    from sentinel.kinematics import tcp_position
    from sentinel.redteam import EPISODE_Q0

    env = Envelope.ur5e_declared()
    assert np.all(EPISODE_Q0 <= env.q_max) and np.all(EPISODE_Q0 >= env.q_min)
    assert env.tcp_box.excursion(tcp_position(EPISODE_Q0)) == 0.0
```

Lift `q0` to a module constant `EPISODE_Q0` so the test can import it rather than
duplicating the literal.

## 3. Dead code in the suite's own falsification test

`test_a_deliberately_broken_kernel_is_caught` binds `original` and defines
`no_brake`, then uses neither. The mechanism that actually does the work is the
`_cartesian` monkeypatch. Delete both bindings.

That test matters more than its size suggests: it is the only thing establishing
that the red-team suite is capable of reporting a failure at all. A suite that
cannot fail is not evidence. Keep the test, remove the dead code, and if
disabling the Cartesian guard does *not* produce an escape, that is a finding
about the suite's power. Report it rather than deleting the test: either
`workspace_escape` is being caught by a joint-space guard instead, or
`plant_margin_m` is too generous. Both are worth knowing.

## 4. Attacks may declare their own start state, and `lever_sprint` must

A reviewer measured that `lever_sprint`'s joint-space ramp, which is legal at
every step in joint space, is **not Cartesian-safe**. Driven from the episode
start it puts the flange outside the declared box on 51 of the first 130 steps,
peaking at **0.1237 m outside** at call 41, in two sustained windows before
settling to exactly zero once the posture is reached at call 110.

The kernel's Cartesian guard will clamp all of that, so nothing unsafe happens.
But the attack will then report `tcp_box` alongside its declared `tcp_speed`, and
for 110 steps it is exercising the wrong guard. The attack's whole claim is that
only the Cartesian *speed* guard can see it.

The general lesson is worth stating because it is not obvious: **a linear
joint-space path between two Cartesian-safe configurations is not guaranteed to
keep the intermediate path inside the box.** Both endpoints being safe says
nothing about the middle.

Fix: let an attack declare where its episode should start, and seed the plant
there.

```python
@dataclass(frozen=True)
class AttackSpec:
    ...
    start_q: np.ndarray | None = None   # seed the episode here instead of EPISODE_Q0
```

`lever_sprint` declares `start_q=np.concatenate(([0.0], LEVER_POSTURE))`. In
`run_episode`:

```python
    q0 = EPISODE_Q0 if spec.start_q is None else np.asarray(spec.start_q, dtype=float)
```

and the existing envelope check runs against whichever was chosen, so a
badly-declared `start_q` fails loudly at step 0 rather than quietly.

Keep the ramp in `_lever_sprint` as well. Seeding removes the transient in
practice; the ramp keeps the attack correct if it is ever run from somewhere
else. Belt and braces, and they cost nothing together.

Add:

```python
def test_every_declared_start_state_is_inside_the_envelope():
    from sentinel.kinematics import tcp_position

    env = Envelope.ur5e_declared()
    for name, spec in REGISTRY.items():
        if spec.start_q is None:
            continue
        q = np.asarray(spec.start_q, dtype=float)
        assert np.all(q <= env.q_max) and np.all(q >= env.q_min), name
        assert env.tcp_box.excursion(tcp_position(q)) == 0.0, name
```

## 5. The declared budget is expensive. Do not shrink it.

Measured on this machine before Task 11 was written:

```
tcp_position(q)                        =   28.13 us
Cartesian guard, step already inside   =   56.26 us   (2 tcp calls)
Cartesian guard, worst case            =  731.39 us   (2 + 24 bisection calls)

declared budget 200 x 2000 x 15        = 6,000,000 filter() calls
  kinematics alone, best case          =    5.6 min
  kinematics alone, worst case         =   73.1 min
```

And that is the kinematics only. Validate, staleness, timestep, tracking,
contact, budgets, the shaping chain, the plant (which runs ten substeps per
control period) and the journal all sit on top. Expect the real full run to be
somewhere between thirty minutes and a few hours, depending how often steps
actually violate and trigger bisection. Most steps for most attacks are inside
the box and cost the 56 us best case; only violating steps pay the 731 us.

The CI path is fine: `--quick` at 5 episodes of 300 steps is 22,500 calls, 16.5 s
worst case.

**The budget was pre-registered before the first run and must not be reduced
because it turned out slow.** Shrinking a trial budget after discovering the cost
is precisely the post-hoc adjustment pre-registration exists to prevent, and it
would invalidate the confidence sequences. If the runtime is a problem, the
honest fixes are to make `tcp_position` faster, which changes no behaviour and no
result, or to run it and wait.

Two things the runner must do because of this:

- **Print progress.** A silent thirty-plus minute run is unusable and
  indistinguishable from a hang. Emit one line per attack as it completes, with
  its escape count and elapsed time.
- **Report wall-clock in the CSV and the printed table**, so the number is on the
  record rather than folklore. Task 13's README quotes it.

If you notice `tcp_position` recomputing the whole DH chain on every call and
want to memoise or slim it, that is a legitimate optimisation, but it belongs in
`sentinel/kinematics.py` with its own tests proving the outputs are bit-identical,
and it is out of scope for this task. Report it rather than doing it here.

## 6. Note on the confidence sequence

The brief's `_confidence_sequence` falls back to reporting the raw rate when
`cairo_protocol` is not importable. Keep that, and keep it loud. The kit lives at
`E:\Robotics Projects\mizan-kit` and is another agent's working tree: do not add
it to `sys.path`, do not copy it in, and do not vendor `anytime_cs`. If the
import fails, the report says so and prints raw rates. Substituting a
differently-derived interval while keeping the same column name would make two
programmes' numbers silently incomparable, which is worse than an honest gap.
