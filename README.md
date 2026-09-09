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
> repository comes from a simulated plant, every limit is declared rather than
> measured, and the red-team suite that would give any of it weight has not been
> run.
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

**The kernel is complete and tested. The red-team results do not exist yet.**

| component | state |
|---|---|
| `envelope` declared limits, provenance, content hash | complete, reviewed |
| `kinematics` UR5e forward kinematics and Jacobian | complete, reviewed |
| `journal` tamper-evident hash-chained log | complete, reviewed |
| `kernel` all ten guards | complete, reviewed |
| `sim` second-order servo plant | complete, reviewed |
| `attacks` fifteen adversarial generators | complete, one fix outstanding |
| `shield` drop-in wrapper | complete, reviewed |
| property tests under `hypothesis` | complete |
| **`redteam` escape-rate runner** | **not built** |
| **`PROTOCOL.md`, `LIMITATIONS.md`** | **not written** |

106 tests pass. **No red-team escape rate has been measured, because the runner
does not exist.** Any number you want about how well this resists attack is not
in this repository yet, and this README will not invent one.

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
