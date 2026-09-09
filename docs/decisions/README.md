# Decision records

Source comments and tests in this repository cite documents by name, for example
`task-10-fix-2.md, finding 2`. Those documents live here.

Each one records a defect found during construction: what was claimed, what was
measured, what the measurement showed, and what changed as a result. They are
kept because the measurements are the evidence behind decisions that are
otherwise invisible in the final code. A constant like `brake_headroom = 0.8` or
a posture like `LEVER_POSTURE` looks arbitrary until you can see the sweep it
came from.

## What these are, and what they are not

These are **corrections**. Every one exists because something in the original
plan was wrong. They are not the design; the design is in
`docs/superpowers/specs/`. Read them as a defect log with the reasoning attached.

Eighteen defects were found this way. The distribution is the interesting part:
almost none were found by reading code. They were found by running a
measurement, or by a reviewer executing a claimed repro rather than trusting it.

## The ones worth reading even if you skip the rest

| document | what it found |
|---|---|
| `task-5-correction.md` | The core shaping algorithm failed its own test. Emitted acceleration 8.81 against a 5.0 limit and jerk 174.8 against 100. Contains the corrected chain and the headroom sweep behind `brake_headroom = 0.8`. |
| `task-5-fix-1.md` | A braking assertion that no controller could satisfy: it compared the velocity that traversed a step against the room remaining after the step landed. 36 violations of 2394 in that form, 0 of 2394 in the correct one. |
| `task-4-correction.md` | The staleness guard compared two different clocks, measured 5.558460 s apart on Windows, and would have latched a permanent stop against a healthy arm. Also why the default clock is `perf_counter`, not `monotonic`. |
| `task-10-fix-1.md` | An attack that could not reach the guard it declared, and the geometric reason: a circle of radius 0.6635 does not fit inside a square of half-width 0.65. |
| `task-6-7-resume.md` | Five tests that passed only because a guard did not exist yet. The flange at `q = zeros` sits 0.167200 m outside the declared workspace box. |
| `task-9-correction.md` | The position clip does bite under adversarial input, 361 times across 200 trials, and why the derivative guarantee is narrowed rather than the kernel changed. |

## The recurring lesson

Three separate defects came from the same root cause, and it is worth naming
because it is not obvious:

**Measuring a component does not validate the composition.** A probe written
before a guard exists silently encodes that guard's absence as an assumption. A
path-budget measurement taken without the Cartesian guard in the loop said the
budget tripped at step 67; with the guard present the arm never moved and the
budget accumulated exactly 0.0 m. The measurement was correct about what it
measured and wrong about the system that shipped.

The property tests in `tests/test_properties.py` exist to close that gap: they
are the only thing here that drives the fully assembled kernel rather than one
piece of it.

## A note on provenance

These documents were written during construction and some reference an internal
task-numbering scheme and a process ledger that are not published. The
measurements, the reasoning and the conclusions are all self-contained; only the
scaffolding around them is missing.
