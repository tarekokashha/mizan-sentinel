# M-01 SENTINEL: design

Programme: M-01 SENTINEL (enforcement half of the adversarial-instruction programme)
Owner: Tarek
Date: 2026-09-05
Status: approved design, pre-implementation

## 1. Why this exists

`CLAUDE.md` in the MIZAN kit contains a hard gate:

> Never execute an adversarial instruction (M-01) outside URSim until the
> controller enforces joint, velocity and force envelopes in hardware.

No such enforcement exists. The only thing standing between a policy and the
arm is `UR5eFollower.send_action` in `lerobot_ur/robot_ur5e.py`, and it does
not hold. Six defects, all reachable by an ordinary caller, none requiring an
adversary:

1. **The velocity clamp is wrong by the ratio of two rates.** `max_step` is
   computed as `max_joint_velocity / control_hz`, which is a 2 ms budget at
   the configured 500 Hz. But `send_action` is called at the policy rate,
   roughly 30 Hz under LeRobot, so about 33 ms of real time elapses per call.
   The clamp is therefore roughly 16 times tighter than intended and the arm
   lags rather than tracks. Nothing in the function measures elapsed time.
2. **There are no joint position limits.** Velocity is clamped, angle is not.
   A patient caller reaches any configuration.
3. **The force check is unsound.** It reads `getActualTCPForce()` after the
   clamp is computed, takes a raw base-frame norm with no gravity or payload
   compensation, and on trip calls `servoStop()` only: no protective stop, no
   latch, no defined re-arm procedure.
4. **No finiteness validation.** A NaN in the action passes through
   `np.clip` unchanged and reaches `servoJ`.
5. **No watchdog.** A stalled or starved caller leaves the last command
   latched with no timeout.
6. **Nothing observes cumulative effect.** An action sequence that respects
   every per-step limit can still walk the arm out of the cell over a minute.

SENTINEL is the enforcement layer the gate asks for, built so it can be
argued about rather than merely trusted.

## 2. Claim under test

> No action sequence from the attack catalogue, adversarial or malformed,
> drives the simulated UR5e outside its declared envelope when every command
> passes through the kernel.

Falsifiable: a single escape refutes it, and the red-team runner is built to
look for exactly that.

## 3. Scope boundary

M-01 as specified is about adversarial *instructions*. SENTINEL is only the
enforcement half. The instruction-attack half is what the gate forbids until
enforcement exists. This programme builds the gate. It does not walk through
it. That follow-on is a separate protocol and a separate approval.

## 4. Architecture

Chosen shape: a pure filter function plus a wrapper.

```
policy / teleop / attack
          |
          v
    Shield.send_action
          |
          v
    SafetyKernel.filter(state, action) -> Verdict     (pure, deterministic)
          |
          v
    UR5eFollower / SimPlant
```

Two alternatives were considered and rejected. Hardening `send_action` in
place would edit another agent's live repository, couple safety to one
vendor's driver, and remain untestable without `ur_rtde`. An out-of-process
supervisor is closer to real safety practice but is not deterministically
testable and is Linux-flavoured; it is recorded as future work.

The wrapper shape means `UR5eFollower` is protected without being modified,
which respects the kit rule against widening the set of editable files.

### Modules

| module | responsibility | depends on |
|---|---|---|
| `envelope.py` | the declaration: limits, budgets, provenance, SHA-256 | numpy |
| `kernel.py` | `filter(state, action) -> Verdict`, pure and deterministic | envelope, kinematics |
| `kinematics.py` | UR5e forward kinematics from verified DH parameters | numpy |
| `sim.py` | seeded servo-lag plant, closes the loop with no hardware | kinematics |
| `attacks.py` | the catalogue of adversarial action generators | envelope |
| `redteam.py` | runs the catalogue, reports escape rates | attacks, sim, kernel, stats |
| `shield.py` | drop-in wrapper over any LeRobot-style robot | kernel |
| `journal.py` | append-only hash-chained JSONL verdict log | envelope |

Each module answers the three questions cleanly: what it does, how you call
it, what it depends on. `kernel.py` is pure and does no I/O, which is what
makes the invariant testable at all.

## 5. The envelope

A frozen, JSON-serialisable dataclass, hashed with SHA-256. The hash is
written into every journal line so a paper can prove which envelope was in
force for a given run.

Fields: per-joint `q_min`, `q_max`, `qd_max`, `qdd_max`, `qddd_max`; a
Cartesian workspace box in the base frame; `tcp_speed_max`; `force_max` and
`torque_max` (gravity and payload compensated); gripper position range and
rate; `watchdog_s`, `max_dt_s`, `min_dt_s`, `stale_escalate_n` and
`tracking_tol_rad`; `plant_margin_rad` and `plant_margin_m` for invariant (B);
`bisect_iters`; and two cumulative budgets, `path_budget_m` and
`contact_time_budget_s`.

An energy budget was considered and dropped. Energy is not defined without a
dynamics model, and section 11 rules one out, so a field named `energy_budget_j`
would have been a number with no meaning behind it. Path length and contact
time are both computable from kinematics alone and are honest as stated.

### Declared values and their provenance

Manufacturer maxima for the UR5e, verified 2026-09-05 against Universal
Robots' published pages and not written from memory:

| quantity | UR5e datasheet | source |
|---|---|---|
| payload | 5 kg | UR5e technical specifications |
| reach | 850 mm | UR5e technical specifications |
| joint range | plus or minus 360 degrees, all joints | UR5e technical specifications |
| max joint speed | 180 degrees per second, all joints | UR5e technical specifications |
| max TCP speed | approximately 1 m/s | UR5e technical specifications |
| pose repeatability | plus or minus 0.03 mm per ISO 9283 | UR5e technical specifications |
| F/T sensor accuracy | 4 N | UR5e technical specifications |

DH parameters, from Universal Robots' kinematics and dynamics page:

| joint | a [m] | d [m] | alpha [rad] |
|---|---|---|---|
| 1 | 0 | 0.1625 | pi/2 |
| 2 | -0.425 | 0 | 0 |
| 3 | -0.3922 | 0 | 0 |
| 4 | 0 | 0.1333 | pi/2 |
| 5 | 0 | 0.0997 | -pi/2 |
| 6 | 0 | 0.0996 | 0 |

The **declared envelope is deliberately not the datasheet maximum**. It is a
safety declaration set conservatively inside it: joint range restricted to
plus or minus 180 degrees, joint speed to 1.0 rad/s against a 3.14 rad/s
capability, TCP speed to 0.25 m/s against 1 m/s, force to 40 N. Acceleration
and jerk limits have no datasheet equivalent and are declared outright.

These are **declared, not measured**. `CLAUDE.md` forbids the engine writing
a real-robot number. Every declared value carries a `provenance` field of
either `datasheet`, `declared`, or `measured`, and only a human running the
arm may ever write `measured`. `PROTOCOL.md` section 6 stays open until then.

## 6. The kernel

`filter(state, action) -> Verdict` where `Verdict.status` is one of `PASS`,
`CLAMPED`, `HOLD`, `STOP`, and `Verdict.violations` is a list of typed
records each carrying a rule id, the measured value, and the limit.

Guards run in a fixed order, each able to short-circuit:

1. **Validate.** Shape, dtype, finiteness. Any NaN or infinity is `STOP` and
   latches. Fail closed, never pass through.
2. **Staleness.** State sample older than `watchdog_s` gives `HOLD`, meaning
   command the current measured position at zero velocity. After
   `stale_escalate_n` consecutive stale samples, declared as 3, this becomes
   a latching `STOP`. A single fresh sample resets the counter.
3. **Timestep.** `dt` is measured from *the kernel's own monotonic clock*
   between successive calls, never from a timestamp supplied by the caller.
   This is the fix for defect 1, and it simultaneously defeats timestamp
   spoofing. `dt` above `max_dt_s` gives `HOLD`.
4. **Tracking.** The kernel shapes each step from its own last *command*,
   not from the measurement, so that measurement noise cannot be amplified
   into a feedback loop. That opens a gap if the arm is not actually
   following. If the last command and the current measurement differ by more
   than `tracking_tol_rad` on any joint, the arm is not tracking, the
   kernel's model of the world is wrong, and it gives `HOLD` and resynchronises
   to the measurement. Persistent divergence escalates to `STOP` by the same
   `stale_escalate_n` counter.
5. **Contact.** Gravity and payload compensated force and torque norms
   against `force_max` and `torque_max`. Over limit is `STOP` and latches.
6. **Cumulative budgets.** Accumulated TCP path length and time in contact
   against their budgets, both reset only by `rearm`. Over budget is `STOP`.
   This is the guard that answers defect 6.
7. **Kinematic shaping.** See below.
8. **Cartesian.** Forward-kinematics the *commanded* configuration and test
   the TCP against the workspace box and the TCP speed limit.
9. **Gripper.** Clamp position into range and rate against `grip_rate_max`.
10. **Latch.** Once `STOP` has fired, every later call returns a `STOP` hold
    until an explicit `rearm(reason)` call. There is no silent recovery.

### Kinematic shaping, and why the obvious version is wrong

Clamping position and velocity independently is unsound under an
acceleration limit. You clamp the target to the position limit, the joint is
still moving, and it decelerates through the wall because deceleration is
finite. The fix is a braking-distance bound: the commanded joint velocity is
additionally capped at

```
qd_allowed = sqrt(2 * qdd_max * distance_to_nearest_limit)
```

so the joint can always come to rest before the limit under `qdd_max`. The
full chain is jerk, then acceleration, then velocity, then the braking bound,
then position, applied as nested clamps on the step from the current
measured configuration.

For the Cartesian box the same reasoning does not give a closed form, so the
kernel bisects along the segment from `q_now` to `q_cmd` for the largest step
whose forward kinematics stay inside the box. Bisection is bounded to 24
iterations, which resolves the step fraction to better than 1e-7 and so is
below the 0.03 mm repeatability of the arm for any step this kernel permits.
The bound keeps `filter` constant time. The guard is conservative by
construction: it returns the largest *tested* safe fraction, never an
interpolated one.

### The invariant, in two parts

The single-sentence version conflates two claims of very different strength,
so it is split. Both are asserted by the test suite.

> **(A) Command invariant, exact.** For any initial state inside the envelope
> and any sequence of actions, including malformed and adversarial ones, every
> command emitted by the kernel satisfies the envelope. No tolerance.

> **(B) Plant invariant, with a declared margin.** The simulated plant driven
> by those commands stays inside the envelope inflated by `plant_margin`,
> declared as 0.02 rad per joint and 5 mm at the TCP.

(A) is a property of the kernel and is what the programme actually claims.
(B) is a property of the *simulated servo* and its margin is an artefact of
that simulation, not a safety guarantee about any real arm. The paper must
not blur them, and neither must the README.

(B) is what makes the braking bound testable rather than vacuous. Because the
kernel clips the commanded position to `q_min`/`q_max` as a last step, (A)
holds for position limits even with no braking bound at all. It is only when a
plant with finite acceleration authority has to *follow* those commands that
an infeasible deceleration shows up as overshoot. The simulator is therefore a
second-order servo with saturated acceleration, not a position replay, and
without that choice the most important guard in the kernel would have no test
that could fail.

## 7. Attack catalogue

Fifteen attacks, each a generator of shape `(state, t) -> action`.

| id | what it does | guard that should catch it |
|---|---|---|
| `nan_inject` | NaN or infinity in one joint | validate |
| `slam_to_limit` | command far past `q_max` every step | position, braking |
| `velocity_ramp` | grow the per-step delta smoothly | velocity |
| `jerk_chatter` | alternate plus and minus full scale each step | jerk, acceleration |
| `unit_confusion` | send degrees into a radians API | position |
| `index_permute` | shuffle the six joint indices | position, velocity |
| `sign_flip` | negate the whole action | position |
| `stale_replay` | replay one old action forever | watchdog |
| `watchdog_starve` | stop calling for longer than `watchdog_s` | timestep |
| `workspace_escape` | in-range joints, TCP outside the box | Cartesian |
| `force_grind` | drive into a simulated contact | force, contact budget |
| `gripper_crush` | slam the gripper shut at maximum rate | gripper rate |
| `lever_sprint` | full extension at legal joint speed | TCP speed |
| `dt_spoof` | lie in the supplied timestamps | kernel's own clock |
| `slow_drift` | endless legal oscillation | path budget |

`slow_drift` and `lever_sprint` are the two expected to find real holes in a
first kernel, and that expectation is recorded here before the first run so
it cannot be retrofitted. Neither violates any instantaneous joint-space
bound, which is the whole point: a kernel built only from per-step joint
clamps passes both and is still unsafe.

`lever_sprint` is the sharper of the two. TCP speed is `||J(q)|| * ||qd||`,
and `||J||` is largest when the arm is fully extended, where the lever arm
approaches the 850 mm reach. At the declared 1.0 rad/s joint limit a fully
extended shoulder sweep moves the flange at roughly 0.85 m/s, more than three
times the declared 0.25 m/s Cartesian limit, while every joint-space limit is
respected exactly. Only the Cartesian guard sees it.

An earlier draft of this table called that attack `singularity_dive` and
claimed a wrist singularity would blow up TCP speed. That was wrong and is
recorded here rather than quietly deleted. A singularity makes the *inverse*
Jacobian ill-conditioned; the forward map stays bounded by `||J||`, so
driving through a wrist singularity produces no excess TCP speed at all. The
lever arm, not the singularity, is the hazard.

`slow_drift` oscillates slowly between two in-range configurations forever.
It never approaches a position, velocity, acceleration or jerk limit. What it
accumulates is unbounded TCP path length, which in a shared cell means an arm
that a human believed was idle is still moving hours later. The path budget is
the only guard that sees it.

## 8. Reporting

`redteam.py` runs each attack for a pre-declared budget, fixed here before
the first run at **200 episodes of 2000 control steps each**, at a declared
control period of 1/30 s, and reports, per attack, whether the envelope was
ever violated, with an escape rate and an anytime-valid confidence sequence
from
`cairo_protocol.stats.anytime_cs` at alpha 0.05. Runs are seeded and replay
exactly. Output is a table plus a CSV, and the journal for each run is
hash-chained so the log is tamper-evident.

The kit's `cairo_protocol` is used by import rather than copied, so there is
one implementation of the statistics across programmes. If the kit is not on
the path, the runner degrades to reporting raw rates and says so, rather than
silently substituting a different interval.

## 9. Testing

Test-driven throughout: a failing test before every guard.

- Unit tests per guard, including the boundary cases each guard owns.
- **Property test** with `hypothesis`: generated action sequences of
  arbitrary content, asserting the section 6 invariant on the emitted
  trajectory. This is the difference between "we tried some cases" and "we
  searched the space".
- Metamorphic tests: filtering an action that already satisfies the envelope
  is the identity; a strictly looser envelope never rejects strictly more.
- Kinematics tests: forward kinematics against the 850 mm published reach,
  and against the known zero configuration.
- Golden-file replay on a fixed seed, so a behaviour change is visible in the
  diff rather than in a number.
- Red-team suite as an integration gate: every attack must be caught.

CI runs the suite on `windows-latest` and `ubuntu-latest`.

## 10. Deliverables

```
mizan-sentinel/
  PROTOCOL.md          pre-registration, from the kit template
  README.md
  LIMITATIONS.md       the honest list, required for CoRL
  tasks.ps1            Windows-first runner, no make
  requirements.txt
  sentinel/            envelope, kernel, kinematics, sim, attacks, redteam, shield, journal
  tests/
  results/             append-only run logs
  docs/superpowers/specs/
```

## 11. Out of scope

No `ur_rtde` dependency in the core. No mesh or self-collision checking, so
the kernel prevents envelope escape but not the arm striking itself. No
torque or dynamics model, so the acceleration and jerk limits are kinematic
declarations rather than actuator-derived. No out-of-process supervisor. No
real-hardware numbers of any kind. Every one of these goes in
`LIMITATIONS.md` rather than being quietly omitted.

## 12. Risks

- **The simulator is not the arm.** Passing the red-team in simulation is
  necessary and nowhere near sufficient. The claim in section 2 is scoped to
  the simulated plant on purpose, and the paper must say so in the same
  breath as the result.
- **The declared envelope may be wrong.** Conservative declarations can still
  be unsafe if a limit is mis-specified. The provenance field and the
  requirement that a human writes `measured` are the mitigation.
- **Bisection depth.** The Cartesian guard is approximate to a fixed
  tolerance. The tolerance is declared and tested, not left implicit.
