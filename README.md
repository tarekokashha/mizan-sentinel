# SENTINEL

**A safety kernel for the UR5e, and the adversarial suite that tries to break it.**

SENTINEL sits between a robot policy and a robot arm. Every commanded motion
passes through a pure, deterministic function that either lets it through,
shortens it, holds position, or latches a stop. No action sequence, however
malformed or hostile, is supposed to drive the arm outside a declared envelope.

This is the enforcement half of M-01 in the MIZAN research programme. It exists
because the programme's own operating contract forbids running adversarial
instructions against hardware until such enforcement exists:

> Never execute an adversarial instruction (M-01) outside URSim until the
> controller enforces joint, velocity and force envelopes in hardware.

This repository builds that gate. It does not walk through it.

---

> ## Do not put this in front of a real arm
>
> **This code has never controlled hardware. Not once.** Every number in this
> repository, including the zero-escape result below, comes from a simulated
> plant, and every limit is declared rather than measured.
>
> The red-team result makes that warning more important, not less. Fifteen
> attacks failing to escape a simulator says the kernel resists the fifteen
> attacks someone thought of, in a model of a robot. It says nothing about the
> attack nobody thought of, and nothing at all about a real arm with real
> compliance, real latency and real sensor noise. One of the two invariants this
> project set out to establish is refuted and its test is left failing on
> purpose.
>
> It is also in-process: a crash of the calling process takes the safety layer
> with it. Real machine safety needs a supervisor the application cannot kill, a
> hardware emergency stop, and limits derived from a hazard analysis of your
> specific cell. This is none of those things.
>
> Use it to study the problem, to test a policy in simulation, or as a starting
> point you then validate yourself. Do not use it as the thing standing between
> a person and a moving robot.

---

## Status

**The kernel is complete and tested, and the full pre-registered red-team run
is complete: zero escapes across 3000 episodes.**

| component | state |
|---|---|
| `envelope` declared limits, provenance, content hash | complete, reviewed |
| `kinematics` UR5e forward kinematics and Jacobian | complete, reviewed |
| `journal` tamper-evident hash-chained log | complete, reviewed |
| `kernel` all ten guards | complete, reviewed |
| `sim` second-order servo plant | complete, reviewed |
| `attacks` fifteen adversarial generators | complete |
| `shield` drop-in wrapper | complete, reviewed |
| property tests under `hypothesis` | complete, one invariant refuted on purpose (`xfail(strict=True)`) |
| **`redteam` escape-rate runner** | **complete: 0/200 escapes per attack, 0/3000 pooled** |
| **`PROTOCOL.md`, `LIMITATIONS.md`** | **written** |

133 tests pass, plus one `xfail(strict=True)` that documents a refuted
property rather than hiding it (see [Limitations](LIMITATIONS.md)). The
red-team run is the one declared in [`PROTOCOL.md`](PROTOCOL.md) before any
trial ran: 200 episodes of 2000 steps for each of 15 attacks, 6,000,000
kernel calls, against envelope `ur5e-declared-v1`. Every attack fired the
guard it was written to test and none escaped. The full table, the
confidence sequences, and exactly what this result does and does not prove
are in [Results](#results) below.

---

## Why this exists

The code this replaces is a single method, and it does not hold. Six defects,
all reachable by an ordinary caller, none requiring an adversary:

1. **The velocity clamp is wrong by the ratio of two rates.** It sizes the step
   from the controller period (2 ms at 500 Hz) while being called at the policy
   rate (about 33 ms at 30 Hz). Roughly sixteen times too tight, so the arm lags
   rather than tracks, and nothing measures elapsed time.
2. **No joint position limits at all.** Velocity is clamped; angle is not. A
   patient caller reaches any configuration.
3. **The force check is unsound.** It reads the sensor after computing the clamp,
   takes a raw base-frame norm with no gravity or payload compensation, and on
   trip calls `servoStop()` only: no protective stop, no latch, no re-arm.
4. **No finiteness validation.** A NaN passes through `np.clip` unchanged and
   reaches the servo.
5. **No watchdog.** A stalled caller leaves the last command latched forever.
6. **Nothing observes cumulative effect.** A sequence respecting every per-step
   limit can still walk the arm out of the cell over a minute.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt         # Linux, macOS

python -m pytest -q
```

No robot, no simulator, no ROS, no vendor SDK. The package imports only the
standard library and numpy.

```python
from sentinel.envelope import Envelope
from sentinel.kernel import SafetyKernel
from sentinel.shield import Shield

kernel = SafetyKernel(Envelope.ur5e_declared())
robot = Shield(your_lerobot_follower, kernel)

obs = robot.get_observation()
robot.send_action({"joint_position": policy(obs)})   # filtered, always
```

---

## The design

### The envelope is a declaration, and it says so

Every limit carries a `provenance` field of `datasheet`, `declared`, or
`measured`. **Nothing in this repository may write `measured`.** Loading an
envelope whose provenance claims a measured value without a named human and a
date raises. That rule is structural, not documentary: `MappingProxyType` and
defensive array copies mean it cannot be defeated after construction either.

The declared envelope sits well inside the machine. Joint range is restricted to
plus or minus 180 degrees against a datasheet 360, joint speed to 1.0 rad/s
against 3.14, flange speed to 0.25 m/s against approximately 1.0. Every envelope
is content-hashed, and that hash is written into every journal line, so a result
can never be attributed to the wrong declaration.

### Ten guards, in a fixed order

Latch, validate, staleness, timestep, tracking, contact, cumulative budgets,
kinematic shaping, Cartesian, gripper. The order is load bearing. Validation
precedes any arithmetic on the numbers. Contact and budgets precede shaping,
because both are reasons to stop outright rather than to shorten a step.

`filter` performs no I/O and takes its clock by injection, which is what makes
the whole thing deterministically testable.

### Two things worth reading the code for

**The braking bound.** Clamping position and velocity independently is unsound
under an acceleration limit: you clamp the target, the joint is still moving, and
it decelerates through the wall because deceleration is finite. So velocity is
additionally capped by a discrete-exact stopping bound, and the same bound is
applied one derivative up, capping acceleration against remaining velocity
headroom for exactly the same reason. Measured on the emitted command sequence,
worst case over both directions and six seeded start states:

| quantity | measured | declared limit |
|---|---|---|
| max abs velocity | 1.000000 | 1.0 rad/s |
| max abs acceleration | 5.000000 | 5.0 rad/s^2 |
| max abs jerk | 100.0000 | 100.0 rad/s^3 |

Each exactly at its limit, position held, arrival velocity 0.000000000, reaching
the joint limit from rest in 101 control steps.

**The kernel keeps its own clock.** It never compares the driver's timestamp
against its own, and never trusts a caller-supplied timestep. On Windows,
`time.monotonic()` and `time.perf_counter()` were measured 5.558460 seconds
apart on the development machine, and the UR driver stamps with `perf_counter`.
A guard comparing them latches a permanent stop against a perfectly healthy arm.
So staleness compares the driver's successive stamps to each other, which is
epoch independent, and the kernel's default clock is `perf_counter` because
`monotonic` on Windows is `GetTickCount64()` at 15.625 ms resolution, roughly
half a control period at 30 Hz, and the shaping chain divides by `dt` three
times.

---

## Results

**The full pre-registered run is complete.** 200 episodes of 2000 control
steps for each of 15 attacks, 6,000,000 kernel calls, 4976.2 s wall clock,
against envelope `ur5e-declared-v1`, SHA-256 `0d45a619219f` (truncated; the
full digest is in every row of `results/redteam.csv`). The budget was fixed
in the design specification before the first trial and was not adjusted
afterward. The full pre-registration is [`PROTOCOL.md`](PROTOCOL.md).

| attack | guard expected | escapes / episodes | guard fired |
|---|---|---|---|
| `dt_spoof` | `qd_max` | 0 / 200 | yes |
| `force_grind` | `force_max` | 0 / 200 | yes |
| `gripper_crush` | `grip_rate` | 0 / 200 | yes |
| `index_permute` | `qd_max` | 0 / 200 | yes |
| `jerk_chatter` | `qddd_max` | 0 / 200 | yes |
| `lever_sprint` | `tcp_speed` | 0 / 200 | yes |
| `nan_inject` | `nan` | 0 / 200 | yes |
| `sign_flip` | `qd_max` | 0 / 200 | yes |
| `slam_to_limit` | `q_limit` | 0 / 200 | yes |
| `slow_drift` | `path_budget` | 0 / 200 | yes |
| `stale_replay` | `stale` | 0 / 200 | yes |
| `unit_confusion` | `q_limit` | 0 / 200 | yes |
| `velocity_ramp` | `qd_max` | 0 / 200 | yes |
| `watchdog_starve` | `dt_max` | 0 / 200 | yes |
| `workspace_escape` | `tcp_box` | 0 / 200 | yes |

Every attack: 0 of 200 escapes. Worst joint excursion 0.0000, worst TCP
excursion 0.0000, across all fifteen. All fifteen fired the guard they were
written to trip. The journal's hash chain (`results/redteam.jsonl`) verifies
with no break.

`slow_drift` runs against a declared per-episode override,
`path_budget_m = 8.0`, not the production envelope's 25.0 m: at
`tcp_speed_max = 0.25` m/s, one 2000-step episode permits at most 16.667 m of
travel, so the production budget cannot be reached inside a single episode by
any attack. The override is not silent; it is in the source CSV's
`envelope_override` column and recorded in `PROTOCOL.md` section 10.
`LIMITATIONS.md` states plainly what this does and does not demonstrate about
the guard at its real, declared value.

### Invariant (A): the kernel's own command

(A) is the property this programme actually claims: every command the kernel
emitted, on every one of 6,000,000 calls in this run, satisfied the declared
envelope. Zero escapes on the commanded-joint check across all fifteen
attacks is a measurement of (A). It holds unconditionally here because the
red-team's episodes all start from a verified, stoppable, in-envelope state;
`LIMITATIONS.md` describes the narrower, non-stoppable states where (A)
cannot be promised, and none of those states occurred in this run.

### Invariant (B): the simulated plant, labelled as such

(B) is a property of the *simulated servo*, not a claim about any real arm,
and its margin, 0.02 rad per joint and 5 mm at the TCP, is an artefact of
that simulation. The same run also found zero plant-side excursions against
that margin, across all 3000 episodes. That is consistent with (B) holding
for the specific start states the fifteen named attacks generate. **It is
not evidence that (B) holds in general.** `tests/test_properties.py` finds
two counterexamples to (B) under `hypothesis`, including a joint-stoppable,
envelope-legal start that still carries the simulated plant 16 mm past the
Cartesian margin, with the kernel commanding nothing unsafe throughout. That
test is left failing on purpose, `xfail(strict=True)`, rather than fixed or
hidden. See `LIMITATIONS.md` for both counterexamples in full. Do not read
this run's zero as a broader claim about (B) than it supports.

### Confidence sequences

Computed post-hoc with `cairo_protocol.stats.anytime_cs` at alpha 0.05 from
the recorded outcomes, after the run, because `cairo_protocol` lives in
another repository and was not on this repository's own path while the
runner itself ran:

```
per attack,  0/200    anytime-valid 95% CS   [0.000001, 0.040501]
pooled,      0/3000   anytime-valid 95% CS   [0.000001, 0.003501]
Wilson fixed-n 95%,   0/3000                 [0.000000, 0.001279]
```

So the escape rate is below 4.05 percent per attack and below 0.35 percent
pooled, valid at every stopping time rather than only at n = 200.
**`results/redteam.csv`'s own `cs_lo` and `cs_hi` columns are raw escape
rates, not these confidence sequences.** The runner printed a loud warning
rather than silently substituting a different interval under the same column
name once it found `cairo_protocol` unavailable on its path; that is
documented, intended behaviour, recorded in `PROTOCOL.md` section 2. The
numbers above are the actual anytime-valid intervals; the CSV's own
`cs_lo`/`cs_hi` are not, in every row.

Reproduce the full run with:

```powershell
.\tasks.ps1 redteam
```

or the quick five-episode smoke version CI runs on every push:

```powershell
.\tasks.ps1 quick
```

---

## What is verified, and what is not

Being precise about this is the point of the project.

**Verified.** The kernel's guards, individually and composed, across 106 tests
including property-based tests under `hypothesis` that drive the fully assembled
kernel with adversarial input. Forward kinematics against two exact geometric
invariants, independently reproduced three times. The hash chain against
tampering, deletion, and re-attribution.

**Not verified.**

- **No red-team escape rate exists.** The runner is not built.
- **No hardware, ever.** Every number here comes from a simulated plant. Passing
  in simulation is necessary and nowhere near sufficient.
- **The Shield has never been composed with the driver it wraps.** That driver
  lives in a separate repository this programme does not touch. What is verified
  is that the Shield honours the LeRobot dict contract, not that the real driver
  does.
- **Every declared limit is declared.** Conservative guesses are still guesses.
- **No self-collision, no dynamics.** The kernel prevents envelope escape, not
  the arm striking itself or a fixture, and acceleration limits are kinematic
  declarations rather than actuator-derived.
- **The kernel is in-process.** A crash of the calling process takes it with it.
  An out-of-process supervisor is the right answer and is not built.

The invariant itself carries two stated narrowings. It holds only from a
*stoppable* start, because a joint closer to its limit than its own velocity can
brake within is committed to an overshoot before the kernel is ever called. And
the derivative guarantee, though not the position guarantee, degrades on steps
where the final position clip engages under adversarial input.

`LIMITATIONS.md` will carry the full list with measured rates once the red-team
run has happened.

---

## Layout

```
sentinel/
  types.py        shared vocabulary: Status, Violation, RobotState, Action, Verdict
  envelope.py     the declaration: limits, provenance, content hash
  kinematics.py   UR5e forward kinematics and Jacobian from the published DH table
  kernel.py       SafetyKernel: all ten guards, pure and deterministic
  sim.py          second-order servo plant with saturated acceleration
  attacks.py      fifteen adversarial action generators
  shield.py       drop-in wrapper over any LeRobot-style follower
  journal.py      append-only hash-chained verdict log
tests/            106 tests, including property-based tests
docs/superpowers/ design specification and implementation plan
docs/decisions/   defect records: what was claimed, measured, and changed
```

1248 lines of package code against 1984 lines of tests.

### `docs/decisions/` is worth a look

Source comments cite documents by name, for example `task-10-fix-2.md,
finding 2`. Those are in `docs/decisions/`, and they are the record of eighteen
defects found during construction: what was claimed, what the measurement
showed, and what changed.

They are published because the measurements are the evidence behind decisions
that are otherwise invisible. `brake_headroom = 0.8` looks arbitrary until you
see the sweep it came from. Almost none of those defects were found by reading
code; they were found by running a measurement, or by a reviewer executing a
claimed reproduction rather than trusting it.

---

## Development

This repository was built with a specification committed before the first line
of code, an implementation plan argued from it, and a task-by-task review gate.
The specification was amended once, before any trial ran, and that commit is
recorded rather than quietly folded in.

Two habits earned their cost repeatedly and are worth keeping:

**Measure the claim before building on it.** Numeric assertions about control
behaviour are unreliable until run, regardless of who wrote them or how
carefully. Several claims in the original plan were refuted by measurement,
including the core shaping algorithm, which failed its own test.

**Prove a test can fail.** A passing test tells you very little; a test you have
watched fail for the right reason tells you a great deal. Three tests in this
repository were found that could not distinguish the property they named, and
one had been cited as evidence a bug was fixed when it had not been.

---

## License

MIT. See [LICENSE](LICENSE).

Note the warranty disclaimer is not boilerplate here. This is safety-adjacent
code that has never been validated against hardware. If you deploy it, the
consequences are yours.

## Disclosure

Experiment code, simulation harnesses, safety tooling and first drafts of this
documentation were produced with Claude (Opus 5) and verified by the authors; all
hardware experiments, statistics and claims are the authors' own.
