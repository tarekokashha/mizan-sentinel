# Task 6 correction: one snippet to ignore, one test start state to replace

**This overrides two things in the Task 6 brief.** The rest stands.

## 1. Ignore the first `filter` snippet. Write only the restructured tail.

The brief's Step 3 shows this first:

```python
        q_cmd, cviols = self._cartesian(self._q_cmd_prev_ref, q_cmd, dt)
        viols += cviols
```

`self._q_cmd_prev_ref` is a field that exists nowhere and is never created. That
snippet cannot run. The brief then says "restructure the tail of `filter` as"
and gives the correct version. Write only the second one.

The reason the restructure is needed at all is worth understanding rather than
copying: `_shape` overwrites `self._q_cmd` as a side effect, so by the time the
Cartesian guard runs, the reference it needs is gone. Capture it first:

```python
        q_ref = self._q_cmd.copy()
        q_cmd, viols = self._shape(q_ref, np.asarray(action.q, dtype=float), dt)
        q_cmd, cviols = self._cartesian(q_ref, q_cmd, dt)
        viols = list(viols) + list(cviols)
```

Keep the resynchronisation block that follows it in the brief. It is load
bearing: when the Cartesian guard shortens a step after `_shape` has already
recorded a velocity, the stored derivative describes motion the arm was never
given, and the next call inherits it. Task 9's property tests are specifically
looking for that.

## 2. `lever_sprint`'s start state is outside the box

The brief's `test_lever_sprint_is_caught_by_the_cartesian_guard_alone` starts at
`q0 = zeros` and calls it "arm straight out". The controller measured it:

```
tcp([0,0,0,0,0,0]) = [-0.8172, -0.2329, 0.0628]     |xy| = 0.8497
declared box:        lo [-0.65, -0.65, 0.05]  hi [0.65, 0.65, 0.90]
```

x is `-0.8172` against a lower bound of `-0.65`, so the flange starts **outside**
the box. The test fails at step 0 and proves nothing about the speed guard.

**A first replacement was found and then rejected, and the reason matters.** A
search for the longest lever at a single in-box pose gave
`q[1:] = [-0.3, 0.5, 0.5, -pi/2, 0]`, lever 0.6635 m, tcp
`[-0.65, -0.1333, 0.1981]`. That pose is inside the box, but the test *sweeps
joint 0*, and a base sweep traces a circle of that radius about the base axis. A
circle of radius 0.6635 does not fit inside a square of half-width 0.65: at a
base angle of zero the flange sits at x = 0.6635, outside. The box guard would
fire before the speed guard ever engaged, and the test would pass while proving
the wrong thing.

The lever must therefore survive the **whole sweep**, not one pose. The box's
inscribed circle has radius 0.6500 m, and a search for the longest sustained
lever finds a posture that reaches exactly it:

That posture was itself later superseded. Maximising the lever put it at radius
0.6500415 against an inscribed radius of exactly 0.6500, so 0.04 mm **outside**,
with zero margin by construction. Use the final measured posture, which Task 10
now also uses, so the two stay in step:

```python
LEVER_Q0 = np.array([0.0, -0.56, 0.32, 0.35, -1.5708, 0.0])
# Verified over a 289-point sweep of joint 0 across the full +-pi range, and
# independently re-derived by a reviewer against this repo's own kinematics:
#   minimum sustained lever = 0.6450233005863028 m
#   maximum                 = 0.6450233005863032 m, constant to 11 figures,
#                             which confirms the circle-of-constant-radius premise
#   worst box excursion     = exactly 0.0, so containment asserts == 0.0 with no
#                             borrowed tolerance
#   flange z constant at 0.3933176, well inside the 0.05 to 0.90 box
#   flange at base angle 0  = [-0.63109925, -0.13329963, 0.3933176]
# At the declared qd_max of 1.0 rad/s that is 0.645023 m/s of flange speed
# against a 0.25 m/s limit, a ratio of 2.58, while every joint-space limit is
# respected exactly. Only the Cartesian speed guard can see it.
```

The test keeps both of its existing assertions. Add the start-state check, which
is what would have caught the original:

```python
def test_lever_sprint_is_caught_by_the_cartesian_guard_alone():
    env = Envelope.ur5e_declared()
    q0 = LEVER_Q0
    assert env.tcp_box.contains(tcp_position(q0)), "the test must start inside the box"
    lever = float(np.linalg.norm(tcp_position(q0)[:2]))
    assert lever * float(env.qd_max[0]) > env.tcp_speed_max * 2, (
        "this configuration is not a lever long enough to be a test")
    cmds, _ = run(np.array([3.0, -0.56, 0.32, 0.35, -1.5708, 0.0]), q0, steps=200)
    p = np.array([tcp_position(q) for q in cmds])
    speed = np.linalg.norm(np.diff(p, axis=0), axis=1) / DT
    assert speed.max() <= env.tcp_speed_max + 1e-6
    # and the joint-space limits were never the binding constraint
    qd = np.abs(np.diff(cmds, axis=0)) / DT
    assert qd.max() < float(env.qd_max[0]) * 0.999
```

Use `LEVER_Q0` for the base-sweep target too, varying only joint 0, so the sweep
holds the lever extended rather than folding it.

The other tests in this task also start at `q0 = [0.0, -1.6, 1.6, -1.6, -1.57, 0.0]`.
That one is fine and should be left alone; only the `lever_sprint` start was
wrong.

## 3. A note for LIMITATIONS

The 0.645023 m lever is the longest a full base sweep can sustain inside the
declared box with real margin, not the 0.85 m the arm can actually reach. The
guard is therefore demonstrated against a hazard about 24 percent smaller than
the machine's worst case. That understates it, and Task 13 must say so rather than letting the reader
assume the test covers the maximum.
