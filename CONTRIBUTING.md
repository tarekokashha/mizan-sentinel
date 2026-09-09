# Contributing to SENTINEL

This is safety-critical code. A bug here does not produce a wrong answer, it
produces a moving arm. The conventions below are not style preferences; each one
exists because its absence cost this project real time.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt         # Linux, macOS
python -m pytest -q
```

The package imports only the standard library and numpy. If you find yourself
adding a runtime dependency, stop and make the case first.

## The four rules that matter

### 1. Never write a real-robot number

Every envelope value carries a `provenance` of `datasheet`, `declared`, or
`measured`. Only a human who ran the arm may write `measured`, and the code
refuses to load such a value without a named person and a date. Do not weaken
that check, and do not fill in a plausible-looking number because a field is
empty. An empty field is honest; a guessed one is not.

## 2. Measure the claim before you build on it

Numeric assertions about control behaviour are unreliable until run, no matter
who wrote them. In this project the core shaping algorithm failed its own test,
a braking assertion turned out unsatisfiable by any controller, and a path-budget
measurement was found to have been taken against a configuration that never
shipped.

If a plan, a comment, or a review says a number, run it before you depend on it.

**And be careful what you measured.** A probe that exercises one component
encodes the absence of every guard that did not exist when it was written. Three
separate defects here came from exactly that: measurements that were correct
about the part and wrong about the whole. Verify against the assembled system.

## 3. Prove your test can fail

Before claiming a test covers something, break the thing it covers and watch it
fail for the right reason. Then revert.

Three tests in this repository were found that could not distinguish the property
they named, and one of them had been cited as evidence a bug was fixed when it
had not been. A passing test is weak evidence. A test you have watched fail is
strong evidence.

## 4. Escalate rather than loosen

If a bound will not hold, that is a specification finding and it gets reported.
It is not an invitation to widen a tolerance, relax a strategy, or add a skip
until the suite goes green. Every time someone on this project escalated instead
of papering over, the planning document turned out to be the thing that was
wrong.

## Conventions

- **No em dashes in prose**, including docstrings and comments. Plain sentences.
- `from __future__ import annotations` at the top of every module in `sentinel/`.
- Joint arrays are float64, shape `(6,)`. Positions in rad, velocities rad/s,
  forces N, torques Nm, lengths m, time s.
- `kernel.py` performs no I/O and takes its clock by injection. Keep it that way;
  it is what makes the kernel deterministically testable.
- Stage commits by explicit path. `git add -A` has swept another agent's
  in-flight work into the wrong commit on this project before.

## Guard order is load bearing

`SafetyKernel.filter` runs its ten guards in a fixed sequence: latch, validate,
staleness, timestep, tracking, contact, cumulative budgets, kinematic shaping,
Cartesian, gripper. Validation precedes any arithmetic on the numbers. Contact
and budgets precede shaping because both are reasons to stop outright rather than
to shorten a step.

If you add a guard, place it deliberately and say why in the commit message.

## Tests

`tests/` mirrors module names. Three files deserve a note:

- `test_kernel_shaping.py` calls `_shape` directly rather than going through
  `filter`. That is deliberate. These are unit tests for a joint-space chain, and
  routing them through `filter` means the Cartesian guard stops the ramp before
  any joint limit is approached, so the tests silently stop testing anything.
- `test_kernel_cartesian.py` covers the composition through `filter`.
- `test_properties.py` drives the fully assembled kernel with `hypothesis`. It is
  the only thing here that tests the whole rather than the parts, which makes it
  the most valuable file in the suite and the slowest.

## Pull requests

Say what you measured, not what you expect. If you changed a number, give the
before and after. If you added a test, say what you saw when you broke the code
it covers.
