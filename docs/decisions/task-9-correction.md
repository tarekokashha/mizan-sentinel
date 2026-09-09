# Task 9 correction: the invariant needs a precondition, and the derivative
# measurement in the brief is wrong

**This overrides two things in the Task 9 brief.** The rest stands.

## 1. Invariant (A) only holds from a stoppable start

The brief's strategy generates `q0` uniformly in `[-1.5, 1.5]` with the plant
starting at rest, which happens to be safe. But the property as stated in the
spec says "for any initial state inside the envelope", and that is too strong.

A joint at `q = pi - 0.05` moving at `0.99 rad/s` needs
`qd^2 / (2 * room)` = `0.98 / 0.1` = `9.8 rad/s^2` to stop before the limit,
against a declared `qdd_max` of `5.0`. It is committed to the overshoot before
the kernel ever sees it. No acceleration-limited controller can rescue that
state, and asserting otherwise would make the property test fail against a
correct kernel.

So the precondition is **stoppability**, per joint:

```
qd^2 <= 2 * qdd_max * (distance to the nearer joint limit)
```

Any strategy that seeds a non-zero initial velocity must filter on it. A
`hypothesis` composite strategy is the clean way:

```python
@st.composite
def stoppable_states(draw):
    """Initial states the kernel can actually be held responsible for.

    A joint closer to its limit than its own velocity can brake within is
    already committed to an overshoot. That is a state outside the envelope's
    premise, not a kernel failure, so the property does not assert over it.
    """
    q = np.array(draw(st.lists(
        st.floats(min_value=-2.8, max_value=2.8, allow_nan=False, allow_infinity=False),
        min_size=6, max_size=6)))
    room = np.minimum(ENV.q_max - q, q - ENV.q_min)
    cap = np.minimum(ENV.qd_max, np.sqrt(2.0 * ENV.qdd_max * np.maximum(room, 0.0)))
    frac = np.array(draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=6, max_size=6)))
    return q, frac * cap
```

Use it for the initial state in all three property tests, and drive the plant
from `(q0, qd0)` rather than from rest. Starting every run at rest tests a
strictly easier problem than the one the kernel faces in service, so this makes
the property stronger, not weaker, despite adding a precondition.

Add one explicit test for the excluded region, so the boundary is documented in
code rather than only in prose:

```python
def test_an_unstoppable_start_is_reported_and_not_silently_accepted():
    # Outside the invariant's premise: 0.99 rad/s with 0.05 rad of room needs
    # 9.8 rad/s^2 against a 5.0 limit. The kernel cannot prevent the overshoot,
    # but it must brake at its limit and say so rather than pretend.
    c = Clock()
    k = SafetyKernel(ENV, clock=c)
    q0 = np.full(6, np.pi - 0.05)
    st0 = RobotState(q=q0, qd=np.full(6, 0.99), t_mono=c.t)
    k.filter(st0, Action(q=q0))
    c.tick()
    v = k.filter(RobotState(q=q0, qd=np.full(6, 0.99), t_mono=c.t), Action(q=np.full(6, 10.0)))
    assert v.action.q.max() <= ENV.q_max.max() + 1e-9      # the command still obeys
    assert v.status is not Status.PASS                      # and it is not silent
```

The spec and `LIMITATIONS.md` both gain this clause. The controller has already
recorded it as a ruling; Task 13 must carry it into the documents.

## 2. Do not prepend zeros when measuring derivatives

Several tests in the plan take derivatives like this:

```python
    qd = np.diff(np.vstack([np.zeros(6), cmds]), axis=0) / DT
    qdd = np.diff(np.vstack([np.zeros(6), qd]), axis=0) / DT
```

Prepending a zero row asserts that the run starts from rest. Once the strategies
above seed a non-zero initial velocity that is false, and the prepend
manufactures a spurious first sample of `qd_max / dt` = 30 rad/s^2, which looks
exactly like an acceleration violation and is not one. The controller hit this
while probing and lost time to it.

Take the derivatives of the emitted sequence directly, everywhere:

```python
    qd = np.diff(cmds, axis=0) / DT
    qdd = np.diff(qd, axis=0) / DT
    qddd = np.diff(qdd, axis=0) / DT
```

## 3. The position clip DOES bite under adversarial targets, and you will hit it

This is the single most important thing on this page, because your strategies
generate exactly the input that triggers it and nobody else's do.

`_brake_bound`'s `room/dt` term exists so that the final
`np.clip(q_ref + qd*dt, q_min, q_max)` never bites, which is what keeps the
emitted derivatives equal to the clamped ones. A reviewer verified that holds
with **zero exceptions** across the whole test suite and across all fifteen real
attacks driven for 2000 steps each.

It does **not** hold universally. Fed a fresh uniformly random target in
`[-50, 50]` on *every single step*, which is unlike anything in the suite or the
catalogue but is precisely what `hypothesis` will produce, the clip bites: **361
times across 200 random-walk trials, worst single-step magnitude about 5 mm.**

When it bites, `qd_real = (q_cmd - q_ref)/dt` is smaller than the `qd` the chain
computed, so the emitted acceleration on that one step is larger in magnitude
than `qdd_max`. That is a genuine, narrow gap, not a test artifact.

**Do not weaken the property to hide it, and do not change the kernel to chase
it.** Position safety is absolute either way: the clip is what enforces it, and
`q_min <= q_cmd <= q_max` holds unconditionally on every step, which is the claim
that actually matters for a safety kernel. What degrades is the derivative
guarantee, on the steps where the clip engages.

Structure the properties to say exactly that:

```python
@settings(max_examples=250, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seq=sequences, start=stoppable_states())
def test_invariant_A_position_is_never_violated(seq, start):
    # Unconditional. This is the claim the programme makes.
    for v, _plant, _cmds in _drive(seq, start):
        q = v.action.q
        assert np.all(np.isfinite(q))
        assert np.all(q <= ENV.q_max + 1e-9)
        assert np.all(q >= ENV.q_min - 1e-9)


@settings(max_examples=150, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seq=sequences, start=stoppable_states())
def test_invariant_A_derivatives_hold_except_where_the_position_clip_engages(seq, start):
    # The derivative guarantee is conditional and the condition is stated, not
    # hidden. On a step where the final position clip engages, the emitted
    # velocity is smaller than the one the chain computed, so the emitted
    # acceleration on that step exceeds qdd_max. Position stays safe regardless,
    # which is why this is a documented narrowing rather than a defect.
    clipped_steps = 0
    total_steps = 0
    ...
    for each step:
        total_steps += 1
        if <the clip engaged on this step>:
            clipped_steps += 1
            continue
        assert np.all(np.abs(qd) <= ENV.qd_max + 1e-6)
        assert np.all(np.abs(qdd) <= ENV.qdd_max + 1e-6)
    # and report the rate rather than silently skipping
    note(f"position clip engaged on {clipped_steps} of {total_steps} steps")
```

You will need the kernel to tell you whether the clip engaged. Add it to the
verdict telemetry rather than recomputing it in the test, because recomputing it
in the test is how you end up asserting against your own guess instead of the
kernel's behaviour:

```python
        telemetry["position_clip_engaged"] = bool(np.any(np.abs(q_pre_clip - q_cmd) > 1e-12))
```

Measure and record the observed clip rate under your final strategies. It goes in
`LIMITATIONS.md`, next to the stoppability precondition, as a second stated
narrowing of invariant (A). The controller has already ruled that `_brake_bound`'s
docstring must stop claiming the guarantee unconditionally.

## 4. Expect failures here, and treat each as a finding

This is the task whose job is to find holes. Two specific things to watch:

- The brief already notes that the velocity property skips `Status.STOP` samples.
  Check that skip is honest. A `STOP` legitimately produces a discontinuous
  command because it is a hold, but if the skip is hiding a jump that happens on
  a `PASS` or `CLAMPED` step, that is a real finding.
- The Cartesian resynchronisation in Task 6 exists because the first draft
  produced an inherited velocity the arm was never given. If that block was
  omitted or written wrongly, this task is what catches it.

Do not widen a tolerance, relax a strategy, or add `assume()` to skip a failing
case. If a bound genuinely cannot hold, that is a specification change: report it
as `DONE_WITH_CONCERNS` with the minimal reproducing sequence and let the
controller rule on it.
