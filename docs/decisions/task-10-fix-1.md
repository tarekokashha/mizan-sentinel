# Task 10 fix round 1: `lever_sprint` cannot survive contact with Task 11

Task 8 and Task 10 were both approved. This fix addresses one Important finding
the reviewer raised at the Task 10 / Task 11 boundary, plus one Minor it raised
about a tolerance. Both trace back to defects in the controller's own correction,
not to the implementation, which transcribed that correction faithfully.

## Finding 1, Important: the attack snaps instead of ramping

`_lever_sprint` sets `q[1:] = LEVER_POSTURE` unconditionally on every call.
Task 11 declares every episode starts at `q0 = [0.0, -1.2, 1.2, -1.5, -1.57, 0.0]`,
which is nowhere near the posture. The reviewer computed the first call's request
from that start: joint deltas of `[0.45, -1.5, -0.2]` rad on joints 1 to 3,
against a legal per-step bound of about 0.0333 rad. That is 13x, 45x and 6x over.

Two things go wrong. The kernel clamps it, so `qd_max` fires long before
`tcp_speed` ever can, which is precisely the "attack cannot reach the guard it
declares" failure this correction series exists to eliminate, relocated from
`tcp_box` to a speed guard. And more importantly it makes the attack's own claim
false: `lever_sprint`'s entire premise is that every joint-space limit is
respected exactly and only the Cartesian guard can see the hazard. An opening
request 45 times over a limit is not that attack.

### Fix: ramp into the posture at a legal rate

Match the clip-toward-goal pattern `slow_drift` already uses, so the attack is
robust to any starting state rather than depending on the runner seeding it:

```python
@attack("lever_sprint", "tcp_speed",
        "fully extended base sweep at the legal joint speed; the lever arm, "
        "not the joint rate, is what makes the flange fast")
def _lever_sprint(env, rng):
    def f(state, step, t):
        q = state.q.copy()
        # Ramp into the lever posture at a legal rate rather than snapping to it.
        # Snapping requests joint steps tens of times over qd_max from an
        # arbitrary start, which trips the velocity guard and falsifies this
        # attack's own premise, that every joint-space limit is respected exactly.
        cap = env.qd_max[1:] * DT * 0.5
        q[1:] = q[1:] + np.clip(LEVER_POSTURE - q[1:], -cap, cap)
        q[0] = q[0] + env.qd_max[0] * DT * 0.99   # sweep the base, legally
        return Action(q=q)
    return f
```

Ramping in from Task 11's `q0` takes 111 steps at half the legal cap, which is
5.5 percent of a 2000-step episode. The base keeps sweeping throughout, so no
time is wasted.

### Test that would have caught it

The two existing `lever_sprint` tests both hard-code the arm as already sitting
at the posture, so neither can see a transition violation. Add one that starts
where Task 11 actually starts:

```python
EPISODE_Q0 = np.array([0.0, -1.2, 1.2, -1.5, -1.5708, 0.0])


def test_lever_sprint_is_legal_from_the_episode_start_state():
    # The existing lever tests start already at the posture, so they cannot see
    # a transition-into-posture violation. Task 11 starts every episode at
    # EPISODE_Q0, which is nowhere near it.
    env = Envelope.ur5e_declared()
    fn = build_all(env, np.random.default_rng(0))["lever_sprint"]
    q = EPISODE_Q0.copy()
    cap = env.qd_max * (1 / 30)
    for i in range(600):
        a = fn(RobotState(q=q, qd=np.zeros(6), t_mono=0.0), i, i / 30)
        assert np.all(np.abs(a.q - q) <= cap + 1e-9), (
            f"step {i} requested {np.abs(a.q - q)} against a cap of {cap}")
        q = a.q
```

## Finding 2, Minor: replace the posture so no tolerance is needed

The controller chose `LEVER_POSTURE` by maximising the lever, which landed it at
radius 0.6500415 against the box's inscribed radius of exactly 0.6500. That is
0.04 mm outside, with zero margin by construction, and it is why the implementer
had to widen a containment test by borrowing `plant_margin_m`, a field
provenanced for the plant's tracking tolerance and unrelated to this.

Maximising to the boundary was the error. A search for the longest lever that
keeps 5 mm of clearance gives a posture that is inside exactly:

```python
# Verified over a 289-point sweep of joint 0 across the full +-pi range:
#   minimum sustained lever        = 0.645023 m
#   worst box excursion, true box  = 0.000e+00, inside with no tolerance at all
#   also clears a 5 mm inset box   = 0.000e+00
#   flange z constant at 0.393318, well inside the 0.05 to 0.90 box
#   flange at base angle 0         = [-0.631099, -0.1333, 0.393318]
# At the declared qd_max of 1.0 rad/s that is 0.645023 m/s against a 0.25 m/s
# limit, a ratio of 2.58, and the test's own meaningfulness threshold of
# lever * qd_max > 0.5 still holds comfortably.
LEVER_POSTURE = np.array([-0.56, 0.32, 0.35, -1.5708, 0.0])
```

With this posture, `test_lever_sprint_stays_inside_the_workspace_box` can assert
exact containment again:

```python
        assert env.tcp_box.excursion(tcp_position(a.q)) == 0.0
```

Drop the `plant_margin_m` tolerance. It was borrowed to absorb an artifact that
no longer exists, and coupling an attack test to an unrelated envelope field
means a future retune of that field breaks this test for no reason.

Update the two existing lever tests to use the new posture. They should still
pass unchanged otherwise.

## Not in scope, deliberately deferred to the final review

- `test_attacks_are_deterministic_under_a_seed` compares two registries at a
  single sample point rather than probing per-call mutable state. The reviewer
  manually audited all fifteen closures and confirmed none retain state, so the
  property holds; only the test is indirect.
- `SimPlant.seed` is inert plumbing, never read by `.step()`. Inherited verbatim
  from the Task 8 brief.

Leave both alone.
