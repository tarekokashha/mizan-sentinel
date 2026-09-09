# Task 2 correction: one test in the brief fails on correct code

**This overrides exactly one test in the Task 2 brief.** Everything else in that
brief stands unchanged.

## The defect

The brief contains:

```python
def test_tcp_never_leaves_the_datasheet_reach_sphere():
    from sentinel.envelope import DATASHEET_REACH_M

    rng = np.random.default_rng(1)
    q = rng.uniform(-np.pi, np.pi, (500, 6))
    r = np.linalg.norm([tcp_position(x) - np.array([0.0, 0.0, DH_D[0]]) for x in q], axis=1)
    assert r.max() <= DATASHEET_REACH_M + abs(DH_D[5]) + 1e-9
```

The controller measured the real maximum over 4000 random configurations:
**0.962623**. The assertion's bound is `0.850 + 0.0996` = **0.9496**. The test
fails on a correct implementation.

This is not a DH transcription error. The DH table is confirmed correct by two
independent exact invariants, both of which are already tests in the brief and
both of which pass:

- `wrist_center(0)[2]` = 0.1625 exactly, matching `d1`
- max planar wrist reach over a 181x181 sweep of q2 and q3 = 0.817200, matching
  `abs(a2) + abs(a3)` = 0.817200 to six figures

The discrepancy is a category error in the assertion. Universal Robots publishes
850 mm as a *working radius to the flange in the arm's principal plane*. The
quantity being measured is the 3D distance from the shoulder origin, which also
picks up the `d5` and `d6` wrist offsets standing perpendicular to that plane.
The two are not comparable and the datasheet number is not an upper bound on the
measured one.

## The replacement

Use the sound triangle-inequality bound, which cannot fail for a correct DH
chain, and additionally pin the measured maximum as a regression value.

```python
def test_tcp_stays_within_the_geometric_reach_bound():
    # The sound bound: no chain of links can put the flange further from the
    # shoulder than the sum of the link lengths that separate them. This can
    # never fail for a correct DH table, so a failure here means the table is
    # wrong, not the tolerance.
    #
    # Deliberately NOT compared against the 850 mm datasheet reach: that is a
    # working radius to the flange in the arm's principal plane, while this is
    # a 3D shoulder-to-flange distance that also picks up the perpendicular d5
    # and d6 wrist offsets. Measured maximum is 0.9626, above 850 mm, and
    # correctly so.
    bound = abs(DH_A[1]) + abs(DH_A[2]) + DH_D[3] + DH_D[4] + DH_D[5]
    shoulder = np.array([0.0, 0.0, DH_D[0]])
    rng = np.random.default_rng(1)
    q = rng.uniform(-np.pi, np.pi, (500, 6))
    r = np.array([np.linalg.norm(tcp_position(x) - shoulder) for x in q])
    assert r.max() <= bound + 1e-9


def test_measured_maximum_reach_is_pinned():
    # Regression pin. The controller measured 0.962623 over 4000 seeded
    # configurations before this file existed. A change here means the
    # kinematics changed.
    shoulder = np.array([0.0, 0.0, DH_D[0]])
    rng = np.random.default_rng(1)
    q = rng.uniform(-np.pi, np.pi, (4000, 6))
    r = np.array([np.linalg.norm(tcp_position(x) - shoulder) for x in q])
    assert r.max() == pytest.approx(0.962623, abs=1e-5)
```

Keep `DATASHEET_REACH_M` in `envelope.py`. It is still the right constant for
documentation and for the declared workspace box. It is simply not the bound for
this particular measurement.
