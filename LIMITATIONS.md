# LIMITATIONS

This is the honest counterweight to a headline of zero escapes. Every entry
below is real and measured, sourced against a specific commit, test, or
number in `results/redteam.csv`. None is softened and none is omitted because
it complicates the story. If you read only one document from this
repository besides the code, read this one, not `README.md`.

## Scope of evidence

**The simulator is not the arm.** Every number in this repository, including
the zero-escapes result, comes from the simulated plant in `sentinel/sim.py`.
Passing in simulation is necessary and nowhere near sufficient. `PROTOCOL.md`
section 6 says this in the required, unsoftened form.

**Fifteen named attacks are evidence, not proof.** The catalogue in
`sentinel/attacks.py` was written by the same process that wrote the kernel
it tests. An adversary who reads the kernel, rather than working from the
same design document the catalogue's author had, will find attacks this
catalogue does not contain. Absence of an escape across all fifteen is
evidence the kernel resists the attacks someone thinking about its own
design would generate. It is not evidence it resists every attack.

**The Shield has never been composed with the driver it exists to wrap.**
`lerobot_ur/robot_ur5e.py` lives in the `mizan-kit` repository, which this
programme deliberately does not touch, so it is not present here. Every test
of `sentinel/shield.py` runs against a fake robot that implements the
LeRobot dict contract. What is verified is that the Shield honours that
contract. What is not verified is that the real driver honours it too, or
that the composition works end to end. The first person to put these
together should expect to find something, and nothing in this repository
should be read as "drop-in" without that qualification attached.

**The confidence sequences were computed post-hoc.** `PROTOCOL.md` section 2
calls for anytime-valid confidence sequences from `cairo_protocol.stats.anytime_cs`
at alpha 0.05. `cairo_protocol` lives in another repository and is
deliberately not vendored here, so it was not importable when
`sentinel/redteam.py` produced `results/redteam.csv`. The runner detected
that and printed a warning rather than substituting something else under the
same column name: `results/redteam.csv`'s own `cs_lo` and `cs_hi` columns
are **raw escape rates**, not confidence sequences, in every row. The actual
anytime-valid intervals quoted in `PROTOCOL.md` and `README.md` were computed
separately, afterward, directly from the recorded per-episode outcomes, using
the real `cairo_protocol.stats.anytime_cs`. Do not read the CSV's `cs_lo` and
`cs_hi` as anytime-valid intervals; they are not.

## Both invariants are narrowed

**(A) holds only from a stoppable start.** The kernel's command invariant is
conditioned on, per joint, `qd^2 <= 2 * qdd_max * d`, where `d` is the
distance to the nearer position limit. A joint closer to its limit than its
own velocity can brake within, at the declared `qdd_max`, is committed to an
overshoot before the kernel is ever called: no acceleration-limited
controller can rescue that state, and the unconditional claim would be false
rather than merely ambitious. A joint at `pi - 0.05` moving at 0.99 rad/s
needs 9.8 rad/s^2 to stop against a declared 5.0, and cannot
(`docs/decisions/task-9-correction.md`,
`tests/test_properties.py::test_an_unstoppable_start_is_reported_and_not_silently_accepted`).
The kernel still does the best available thing from such a state: it brakes
at `qdd_max` and reports the violation. It does not promise (A) from there.

**(A)'s derivative guarantee, not its position guarantee, degrades when the
final position clip engages under adversarial input.** Position safety is
unconditional: `q_min <= q_cmd <= q_max` holds on every single call, with no
exception, and `tests/test_properties.py::test_invariant_A_position_is_never_violated`
asserts exactly that with no precondition beyond a finite command. What can
fail on a clipped step is the derivative: the realised velocity is smaller
than the one the shaping chain targeted, so the realised acceleration on
that one transition can exceed `qdd_max`. This is real and measured, not
theoretical: driven with a fresh uniformly random target in `[-50, 50]` on
every single step, unlike anything in the attack catalogue but exactly what
`hypothesis` generates, the clip engaged **361 times across 200 random-walk
trials, worst single-step magnitude about 5 milliradians**
(`docs/decisions/task-9-correction.md`, section 3, reproduced in the module
docstring of `tests/test_properties.py`). `test_invariant_A_derivatives_hold_except_where_the_position_clip_engages`
reports its own observed clip rate at run time via `hypothesis.note()`, tied
to that run's random search and not a fixed constant; the 361-of-200 figure
above is the specific measurement recorded in the decision that established
this finding, and is the number to cite.

**(B) is refuted, and its test is left failing on purpose,
`xfail(strict=True)`.** `test_invariant_B_the_plant_stays_within_the_declared_margin`
in `tests/test_properties.py` is marked `pytest.mark.xfail(strict=True)`
rather than fixed, weakened, or deleted, so that a silent pass would fail the
suite loudly rather than let the finding be lost. Two counterexamples, both
captured as permanent regression tests in `tests/test_kernel_cartesian.py`:

1. A plant that starts outside the declared Cartesian box is already past
   the margin before the kernel is ever called: `tcp_position(zeros)` sits
   0.1672 m outside `tcp_box`, so nothing the kernel does from there can
   satisfy a property about staying inside it.
2. Restricting the search to starts already inside the box, a
   **joint-stoppable** velocity, one that satisfies the same
   `qd^2 <= 2 * qdd_max * d` formula (A)'s own precondition uses, can still
   carry the plant **16 mm past the Cartesian margin** from a
   `qd_max`-legal, joint-stoppable start, purely from the plant's own
   physical inertia settling toward a kernel command that is itself correct:
   status `PASS`, commanded `q` exactly equal to `q0`. The kernel commands
   nothing unsafe in this counterexample. The plant still exceeds the
   declared 5 mm `plant_margin_m`.

The cause is structural, not a bug to patch: `stoppable_states()`'s
precondition constrains joint-space room only, and **joint-space
stoppability does not imply Cartesian stoppability**. A lever-arm effect
through the Jacobian at a configuration far from the base axis can carry a
joint-legal velocity well past a Cartesian margin that per-joint reasoning
never sees. A correct precondition would need a Jacobian-scaled velocity
margin against `tcp_box`, not a closed-form per-joint formula like the one
(A) uses; none exists in this repository. (B)'s margin was always a property
of the simulated servo, declared as 0.02 rad per joint and 5 mm at the TCP,
and this finding is why the design specification and the README are careful
to state it is not a claim about any real arm even where it holds, and
plainly false as a general property even in simulation where it does not.

## What the kernel does not model

- **No self-collision or mesh collision.** The kernel prevents envelope
  escape, not the arm striking itself or a fixture in its cell.
- **No dynamics or torque model.** Acceleration and jerk limits
  (`qdd_max`, `qddd_max`) are kinematic declarations, not derived from any
  actuator, and the plant's own `accel_authority` in `sentinel/sim.py` is a
  chosen number, not a measured one.
- **The contact guard assumes the wrench arrives already gravity and payload
  compensated.** Producing that compensation is the driving software's job,
  not this kernel's, and it is not implemented here. A caller that wires a
  raw force/torque sensor reading straight into the kernel would silently
  defeat the contact guard: gravity and the payload's own weight would be
  read as applied force.
- **The kernel is in-process.** A crash of the calling process takes the
  safety layer with it. An out-of-process supervisor, one the calling
  application cannot kill, is the right answer for real deployment and is
  not built here.

## Declared values, and what they cost

**Every limit is declared, not measured.** A conservative-looking
declaration is still a declaration: `provenance` records `datasheet` or
`declared` for every field in `sentinel/envelope.py`, and nothing in this
repository is permitted to write `measured`. A mis-specified limit is still
a mis-specified limit no matter how conservative it looks next to the
datasheet.

**The braking bounds cost speed for exactness.** Each of the two nested
bounds (velocity against position headroom, acceleration against velocity
headroom) plans to use only `brake_headroom = 0.8` of the authority
available at the level above, so the arm decelerates earlier and more gently
than a perfect controller would. Reaching a joint limit from rest takes 101
control steps rather than the theoretical minimum
(`docs/decisions/task-5-correction.md`). That is the price of the
acceleration and jerk limits holding exactly on the emitted command
sequence, and it is the right trade, but it is a real cost and this is where
it is stated.

**The Cartesian guard is conservative to its own bisection tolerance.**
`bisect_iters = 24` resolves the safe step fraction to better than `1e-7`,
which is below the arm's declared 0.03 mm pose repeatability for any step
this kernel permits, but the guard is conservative by construction: it
returns the largest fraction it actually tested as safe, never an
interpolated one. A legal step near the box boundary can therefore be
shortened slightly more than the true boundary would require.

**`path_budget_m = 25.0` is a red-team default, not a production value.** At
the declared `tcp_speed_max = 0.25` m/s, 25.0 m is 100 seconds of continuous
motion, so an ordinary pick-and-place cell moving for a few seconds per cycle
would latch after roughly a dozen cycles if this default were deployed
unchanged. Both cumulative budgets, path length and contact time, reset only
on an explicit `rearm` call, by design: silent recovery from a latched budget
would defeat the point of having one.

## What the red-team suite does not demonstrate

**The path budget cannot be exercised at its declared production value in
one episode.** At `tcp_speed_max = 0.25` m/s and the declared 1/30 s control
period, one 2000-step episode lasts 66.667 s and permits at most
0.25 * 66.667 = **16.667 m** of travel, against a **25.0 m** production
budget. No attack, `slow_drift` included, can trip `path_budget` at its
production value inside one episode; the arithmetic forbids it regardless of
how the attack is written. `slow_drift` instead runs against a declared,
non-silent episode-scale override of `path_budget_m = 8.0`
(`results/redteam.csv`, `envelope_override` column) and trips it at
**step 1474** (`tests/test_redteam.py`). What this demonstrates is that the
path-budget guard fires correctly when its budget is reachable. What it does
not demonstrate is that guard firing at the value the envelope actually
declares for production use.

**`lever_sprint` exercises a shorter lever than the machine's full reach.**
The attack's start posture, `LEVER_Q0`, sustains a lever of
**0.6450233005863028 to 0.6450233005863032 m** (constant to eleven figures)
across a full base sweep inside the declared box, verified over a 289-point
sweep of joint 0 across the full range
(`docs/decisions/task-6-correction.md`). That is the longest lever a full
base sweep can sustain with real margin inside the declared workspace box,
not the arm's datasheet 0.85 m reach. At the declared `qd_max` of 1.0 rad/s
that produces 0.645023 m/s of flange speed against the 0.25 m/s TCP limit, a
ratio of 2.58, entirely caught by the Cartesian speed guard alone with every
joint-space limit respected exactly. The guard is demonstrated against a
hazard about 24 percent smaller than the longest lever the real machine's
0.85 m reach could in principle present.

## Housekeeping honesty

**`scipy` is declared in `requirements.txt` and imported by no module under
`sentinel/`.** `grep -rn "scipy" sentinel/ tests/` returns nothing. That is
not neglect: `cairo_protocol.stats`, which `sentinel/redteam.py` imports for
`anytime_cs`, uses `scipy.special.betaln` in its mixture-martingale
computation when scipy is importable, and falls back to an equivalent
`math.lgamma` combination when it is not (`cairo_protocol/stats.py`,
`_log_mixture_martingale`). Dropping scipy from this repository's own
declared dependencies would silently push that computation onto the fallback
path whenever `cairo_protocol` itself is importable, changing the confidence
sequences this programme reports with no error raised anywhere. It stays
declared for that reason, not by oversight. In this run, scipy was in fact
importable (version 1.17.1 in the development environment) and
`cairo_protocol.stats._betaln` resolved to the real function, so the
confidence sequences quoted in `PROTOCOL.md` and `README.md` used
`scipy.special.betaln`, not the `math.lgamma` fallback. `sentinel/` itself
still imports only the standard library and numpy; scipy is exclusively
`cairo_protocol`'s dependency, reached transitively through `redteam.py`'s
import of `anytime_cs`.

**`sentinel/redteam.py` applies a hardcoded tolerance the plant-side checks
do not.** The commanded-joint escape check compares the kernel's own emitted
`q_cmd` against `[q_min, q_max]` with a bare `1e-9` tolerance
(`sentinel/redteam.py`, the `j_exc > 1e-9` check). The two plant-side checks
a few lines below it, joint position and TCP position, use the envelope's
own declared margins, `plant_margin_rad` and `plant_margin_m`, not an
independent hardcoded number. The `1e-9` figure is floating-point slack
around an exact invariant, not a second declared margin, and it is
considerably tighter than either `plant_margin_rad` (0.02 rad) or
`plant_margin_m` (0.005 m). It is recorded here because a reader comparing
the three checks side by side in the source should not have to work out for
themselves that only two of the three are keyed to a declared, documented
number.
