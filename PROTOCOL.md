# PROTOCOL.md  (pre-registration; commit before the first trial)

Programme: M-01 SENTINEL
Owner: Tarek
Date committed: 2026-09-09
Git hash of the harness at commit: <fill after commit>

**A note on sequencing, since it matters for what this document is worth.**
This file is committed after the red-team run already completed, as the last
task of the implementation plan. That does not make it a pre-registration
written to fit the results. The claim in section 1 and the budget in section 2
were fixed in the design specification,
`docs/superpowers/specs/2026-09-05-m01-sentinel-design.md`, committed
2026-09-05, before a single line of kernel code existed and long before the
run. Section 10 records the one amendment made to that specification, and it
was made before trial 1, not after. This file formalises that same claim and
that same budget into the kit's template; it does not change either of them,
and the run in the outcome section below was executed against the budget as
declared, not adjusted afterward.

## 1. Claim under test

> No action sequence from the attack catalogue, adversarial or malformed,
> drives the simulated UR5e outside its declared envelope when every command
> passes through the kernel.

Falsifiable: a single escape refutes it. "Escape" is defined precisely in
`sentinel/redteam.py` as any of: a non-finite commanded joint value; a
commanded joint outside `[q_min, q_max]`; a simulated plant joint outside
those limits inflated by `plant_margin_rad`; or a simulated plant TCP outside
`tcp_box` inflated by `plant_margin_m`. Nothing in that check rounds or
excuses a violation.

## 2. Primary metric and decision rule

- **Primary metric.** Escape indicator, binary per episode, as defined above.
  Reported per attack and pooled.
- **Statistical procedure.** Anytime-valid confidence sequences,
  `cairo_protocol.stats.anytime_cs`, alpha = 0.05, on the escape indicator.
  `cairo_protocol` lives in the `mizan-kit` repository and is used by import,
  not copied, so there is one implementation of the statistics across
  programmes (design spec section 8). It is deliberately not vendored into
  this repository and is not on this repository's own path.
- **A consequence of that, stated plainly.** Because `cairo_protocol` is not
  importable when `sentinel/redteam.py` runs on its own, the runner's own
  `cs_lo`/`cs_hi` columns in `results/redteam.csv` are **raw escape rates,
  not confidence sequences**. The runner detects this at import time and
  prints a loud warning rather than silently substituting a different
  interval under the same column name (design spec section 8: "the runner
  degrades to reporting raw rates and says so"). The anytime-valid confidence
  sequences this protocol calls for were computed **post-hoc**, after the
  run, directly from the recorded per-episode outcomes, using the actual
  `cairo_protocol.stats.anytime_cs` implementation. See the outcome section
  at the end of this document, and do not read `cs_lo`/`cs_hi` in the CSV as
  anytime-valid intervals.
- **Stopping rule.** None. This is a fixed-budget red-team run, not a
  sequential experiment with an early-stopping rule. The declared budget
  below is exhausted regardless of the escape rate observed along the way.
- **Trial budget.** 200 episodes per attack, 2000 control steps per episode,
  15 attacks: 6,000,000 kernel `filter()` calls, at a declared control period
  of 1/30 s. Declared in the design specification, section 8, before the
  first trial, and not adjusted after.
- **Minimum effect of interest.** None, in the two-arm sense the kit's
  template is written for. This is a falsification test of one system
  against a hard invariant, not a comparison between two policies where a
  minimum detectable difference would apply. The budget was sized to bound
  how small an escape rate the confidence sequence could still fail to rule
  out at 0 escapes, not to detect a difference between arms.

## 3. Tasks (attack catalogue)

There is no task suite of manipulation episodes here; the analogue is the
fifteen-attack catalogue from the design specification, section 7. Each
attack is a generator of shape `(state, t) -> action`, run for the full
declared budget against the single declared envelope `ur5e-declared-v1`.

| id | what it does | guard expected to catch it | rule id (from `results/redteam.csv`) |
|---|---|---|---|
| `nan_inject` | NaN or infinity in one joint | validate (fail closed) | `nan` |
| `slam_to_limit` | command far past `q_max` every step | position, braking | `q_limit` |
| `velocity_ramp` | grow the per-step delta smoothly | velocity | `qd_max` |
| `jerk_chatter` | alternate plus and minus full scale each step | jerk, acceleration | `qddd_max` |
| `unit_confusion` | send degrees into a radians API | position | `q_limit` |
| `index_permute` | shuffle the six joint indices | position, velocity | `qd_max` |
| `sign_flip` | negate the whole action | position | `qd_max` |
| `stale_replay` | replay one old action forever | watchdog | `stale` |
| `watchdog_starve` | stop calling for longer than `watchdog_s` | timestep | `dt_max` |
| `workspace_escape` | in-range joints, TCP outside the box | Cartesian | `tcp_box` |
| `force_grind` | drive into a simulated contact | force, contact budget | `force_max` |
| `gripper_crush` | slam the gripper shut at maximum rate | gripper rate | `grip_rate` |
| `lever_sprint` | full extension at legal joint speed | TCP speed | `tcp_speed` |
| `dt_spoof` | lie in the supplied timestamps | kernel's own clock | `qd_max` |
| `slow_drift` | endless legal oscillation | path budget | `path_budget` |

`slow_drift` and `lever_sprint` were recorded before the first run as the two
expected to find real holes in a first kernel, because neither violates any
instantaneous joint-space bound (design spec section 7). Every attack's rule
id above is the value the runner checked for and recorded as
`expected_guard_fired`; the outcome section confirms all fifteen fired it.

## 4. Perturbation strata

Not applicable in the sense the kit's template means it: there is no real
world here, no lighting, no distractor objects, no instruction paraphrase.
The nearest analogue is the attack catalogue in section 3 itself, which is
the deliberate source of variation this protocol tests against. Within that,
the dimensions that do vary are:

- **Episode start state.** `EPISODE_Q0`, verified inside the declared
  envelope before step 0 of every episode (`sentinel/redteam.py`), except
  where an attack declares its own start (`lever_sprint`, whose start is
  Cartesian-safe by construction; see `docs/decisions/task-6-correction.md`).
- **Seed.** The full run uses the runner's default seed (0). Runs are seeded
  and replay exactly (design spec section 8); nothing here is stochastic
  across repetitions of the same seed.
- **Envelope override.** `slow_drift` runs against a declared per-episode
  override, not the production envelope. See section 10.

Trials are not interleaved or randomised across the fifteen attacks; each
attack runs its full 200-episode budget in sequence, which is a property of
`run_all` in `sentinel/redteam.py`, not a scientific claim about
independence.

## 5. Arms and baselines

One arm: the kernel (`SafetyKernel`) under the declared envelope
`ur5e-declared-v1`, wrapping the simulated plant (`sentinel/sim.py`). There
is no baseline arm run for comparison. The unprotected `send_action` method
described in the design specification's section 1 (the six defects this
programme exists to fix) is the implicit baseline this kernel replaces, but
it is not executed here: running the attack catalogue against it would mean
executing an adversarial instruction with no enforcement in place, which the
kit's own operating contract forbids regardless of whether hardware is
involved.

## 6. Hardware disclosure

No hardware was used. Every number in this protocol comes from the simulated
plant in `sentinel/sim.py`. No real-robot number appears anywhere in this
repository, and the envelope values are declared rather than measured. This
section stays open until a human runs the arm.

## 7. Compute disclosure

No GPU. The kernel, the simulated plant, and the runner are pure Python plus
numpy, and the full run was executed on the development machine's CPU alone.
No cost was incurred beyond that machine's time. Wall-clock time is on the
record: `results/redteam.csv`'s `wall_clock_s` column gives it per attack,
summing to **4976.2 s** across the full 6,000,000-call run. No peak-memory
assertion log exists for this run; nothing here approached a scale where one
seemed necessary.

## 8. What counts as an intervention

No human touches a trial while it runs; the runner is autonomous, seeded, and
deterministic. Two distinct failure modes are logged, and both fail the
whole run in `sentinel/redteam.py`'s exit code:

- **An escape**, as defined in section 1: the invariant this protocol tests
  is false.
- **A silent attack**: `expected_guard_fired` is `False`, meaning the guard
  the attack was written to trip never fired, which is a finding about the
  red-team suite's own power (design spec self-review, task-13-brief.md) and
  not about the kernel.

Neither occurred in this run. See the outcome section.

## 9. Video and logging

No video; there is no camera and nothing physical to record. In its place,
every episode's summary is appended to a tamper-evident, hash-chained JSONL
log (`sentinel/journal.py`), written to `results/redteam.jsonl` for this run,
with `results/redteam.csv` as the tabular summary. Every line and every CSV
row carries the SHA-256 of the envelope that produced it
(`envelope_sha256`), so a result can never be silently attributed to the
wrong declaration. The chain for this run was checked with
`sentinel.journal.verify` and verifies with no break.

## 10. Deviations

The specification was corrected in commit `8e8ed4d`
("docs: amend the pre-registration before trial 1"), **before any trial
ran**, in two places:

1. **Invariant (A) gained a stoppability precondition.** The unconditional
   form was false, not merely ambitious: a joint closer to its limit than
   its own velocity can brake within, at `qdd_max`, is committed to an
   overshoot before the kernel is ever called, and no acceleration-limited
   controller can rescue that state. The precondition is, per joint,
   `qd^2 <= 2 * qdd_max * d` where `d` is the distance to the nearer
   position limit.
2. **The braking bound was replaced after measurement.** The textbook
   continuous bound, applied last in a nested clamp, was measured against
   the design's own ramp test and produced **8.808387 rad/s^2** against a
   declared 5.0 limit and **174.846820 rad/s^3** against a declared 100.0
   limit (`docs/decisions/task-5-correction.md`). The corrected chain
   settles the velocity target first, including a discrete-exact braking
   bound capped by `room/dt`, and applies the same bound one derivative up.
   Measured at the resulting `brake_headroom = 0.8`: max `|qd|` = 1.000000
   against 1.0, max `|qdd|` = 5.000000 against 5.0, max `|qddd|` = 100.0000
   against 100.0, each exactly at its limit.

Separately, `slow_drift` runs against a declared episode-scale override of
`path_budget_m = 8.0`, not the production envelope's `path_budget_m = 25.0`.
The reason is arithmetic, not a weakening of the attack: at
`tcp_speed_max = 0.25` m/s, one 2000-step episode at the declared 1/30 s
control period lasts 66.667 s and permits at most 0.25 * 66.667 =
**16.667 m** of travel, so the production budget of 25.0 m cannot be reached
inside a single episode by any attack, `slow_drift` included. The override is
never silent: it is printed in the runner's table and written to the
`envelope_override` column of `results/redteam.csv`. What this demonstrates
and what it does not demonstrate are both recorded in `LIMITATIONS.md`.

---

## Outcome (addendum, written after the run in section 2's declared budget)

The full declared run completed: 200 episodes x 2000 steps x 15 attacks,
6,000,000 kernel calls, 4976.2 s wall clock, against envelope
`ur5e-declared-v1`, SHA-256 `0d45a619219f` (truncated; the full digest is in
every row of `results/redteam.csv`). Every attack: **0 of 200 escapes**.
Worst joint excursion 0.0000, worst TCP excursion 0.0000, across every
attack. All fifteen fired the guard they declared
(`expected_guard_fired = True` in every row). The journal's hash chain
verifies.

Confidence sequences, computed post-hoc with `cairo_protocol.stats.anytime_cs`
at alpha = 0.05 from the recorded outcomes, independently reproduced while
writing this document:

```
per attack,  0/200    anytime-valid 95% CS   [0.000001, 0.040501]
pooled,      0/3000   anytime-valid 95% CS   [0.000001, 0.003501]
Wilson fixed-n 95%,   0/3000                 [0.000000, 0.001279]
```

So the escape rate is below 4.05 percent per attack and below 0.35 percent
pooled, valid at every stopping time rather than only at n = 200. As stated
in section 2: `results/redteam.csv`'s own `cs_lo`/`cs_hi` columns are raw
rates, not these confidence sequences, because `cairo_protocol` was not on
the path when the runner itself produced that file. The numbers above are a
separate, later computation from the same recorded per-episode outcomes.

This result is a claim about the simulated plant in `sentinel/sim.py`, run
against the fixed start states the fifteen named attacks use. It is not, on
its own, a demonstration that invariant (B) holds in general: the
property-based test suite (`tests/test_properties.py`) finds a Cartesian
counterexample to (B) using start states this red-team catalogue does not
generate. `LIMITATIONS.md` carries that finding in full; this protocol
records the red-team outcome honestly rather than reading more into a zero
than the zero supports.
