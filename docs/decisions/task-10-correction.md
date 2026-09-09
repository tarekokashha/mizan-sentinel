# Task 10 correction: two attacks cannot reach the guards they declare

**This overrides two attack definitions and one field on `AttackSpec` in the
Task 10 brief.** Everything else stands.

An attack that cannot reach its own guard is worse than no attack. It reports a
clean run and proves nothing, and the suite's
`test_every_attack_actually_trips_the_guard_it_declared` is what catches that.
Both of these were found by measurement before dispatch.

## 1. `lever_sprint` leaves the workspace box before the speed guard engages

The brief has:

```python
def f(state, step, t):
    q = state.q.copy()
    q[1:] = 0.0                                  # hold the arm straight out
    q[0] = q[0] + env.qd_max[0] * DT * 0.99      # sweep the base, legally
    return Action(q=q)
```

`q[1:] = 0.0` drives toward the zero configuration, whose flange sits at
`[-0.8172, -0.2329, 0.0628]`, `|xy|` = 0.8497, **outside** the declared box
(`lo [-0.65, -0.65, 0.05]`, `hi [0.65, 0.65, 0.90]`). So `tcp_box` fires, not
`tcp_speed`, and the attack's declared guard never trips.

A first replacement was found and rejected, and the reason is the interesting
part. Searching for the longest lever at a single in-box pose gives
`q[1:] = [-0.3, 0.5, 0.5, -pi/2, 0]` at 0.6635 m. But this attack *sweeps joint
0*, and a base sweep traces a circle of that radius about the base axis. A circle
of radius 0.6635 does not fit inside a square of half-width 0.65: at base angle
zero the flange sits at x = 0.6635, outside. The box guard would fire first
again.

The lever has to survive the **whole sweep**, not one pose. The box's inscribed
circle has radius 0.6500 m, and a search for the longest sustained lever reaches
exactly it:

```python
# Verified over a 145-point sweep of joint 0 across the full +-pi range:
#   minimum lever anywhere in the sweep = 0.6500 m, the box's inscribed radius
#   flange z constant at 0.8465, inside the 0.05 to 0.90 box
#   flange at the start pose = [-0.6362, -0.1333, 0.8465]
# At the declared qd_max of 1.0 rad/s that is 0.6500 m/s against a 0.25 m/s
# limit, a ratio of 2.60, with every joint-space limit respected exactly.
LEVER_POSTURE = np.array([-0.75, -0.3, -1.7, -np.pi / 2, 0.0])


@attack("lever_sprint", "tcp_speed",
        "fully extended base sweep at the legal joint speed; the lever arm, "
        "not the joint rate, is what makes the flange fast")
def _lever_sprint(env, rng):
    def f(state, step, t):
        q = state.q.copy()
        q[1:] = LEVER_POSTURE                    # hold the longest sweep-safe lever
        q[0] = q[0] + env.qd_max[0] * DT * 0.99  # sweep the base, legally
        return Action(q=q)
    return f
```

## 2. `watchdog_starve` cannot exceed the watchdog

The brief gives it `skip_steps=1`, and the Task 11 runner ticks
`DT * (1 + skip_steps)` = `2/30` = 0.0667 s on marked steps. That is well below
`max_dt_s` = 0.2, so the `dt_max` guard can never fire.

Replace the `skip_steps: int = 0` field on `AttackSpec` with:

```python
    skip_seconds: float = 0.0   # extra wall time to burn on marked steps
```

and declare:

```python
@attack("watchdog_starve", "dt_max", "stop calling for longer than the watchdog",
        skip_seconds=0.5)          # must exceed max_dt_s = 0.2
def _watchdog_starve(env, rng):
    def f(state, step, t):
        return Action(q=state.q + 0.05)
    return f
```

Task 11's runner consumes `skip_seconds`; that side is covered in
`task-11-correction.md`.

Add this test, so the class of mistake cannot recur silently:

```python
def test_watchdog_starve_actually_exceeds_the_watchdog():
    env = Envelope.ur5e_declared()
    assert REGISTRY["watchdog_starve"].skip_seconds > env.max_dt_s, (
        "the attack cannot reach the guard it declares")
```

## 3. Strengthen the two attacks whose whole point is that they look legal

`test_slow_drift_never_requests_an_illegal_per_step_change` and
`test_lever_sprint_requests_only_legal_joint_speeds` already exist in the brief
and both should stay. Add one assertion to the lever test, because it is the one
that would have caught defect 1 above:

```python
def test_lever_sprint_stays_inside_the_workspace_box():
    # The attack's whole claim is that it is legal everywhere the box guard can
    # see, and dangerous only in flange speed. If it leaves the box, tcp_box
    # fires instead of tcp_speed and the attack proves nothing.
    from sentinel.kinematics import tcp_position

    env = Envelope.ur5e_declared()
    fn = build_all(env, np.random.default_rng(0))["lever_sprint"]
    q = np.concatenate(([0.0], LEVER_POSTURE))
    for i in range(400):
        a = fn(RobotState(q=q, qd=np.zeros(6), t_mono=0.0), i, i / 30)
        assert env.tcp_box.excursion(tcp_position(a.q)) == 0.0, (
            f"lever_sprint left the box at step {i}: {tcp_position(a.q)}")
        q = a.q
```

## 4. A note on `force_grind`, which is fine but worth understanding

Its `q[1] += 0.02` sweep drives the flange down to z = -0.479 from the episode
start pose, so a `PlaneContact(z=0.10)` is comfortably reachable and the plane
sits just above the box floor of 0.05. Measured, no change needed. Do not
"improve" the step size.
