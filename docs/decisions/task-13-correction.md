# Task 13 additions: things decided during the build that must reach the docs

**This adds to the Task 13 brief.** Nothing in it is removed.

Several decisions were made while building that change what the documents have
to say. A decision recorded only in a controller ledger is a decision made in
secret, so each of these has to land in a committed file.

## 1. `.gitignore` gains `.superpowers/`

The controller's scratch directory (ledger, task briefs, implementer reports,
review packages, correction files) is currently excluded only through
`.git/info/exclude`, which is local to this machine and not shared. Add
`.superpowers/` to the committed `.gitignore` so a fresh clone inherits it.
Without that, the next contributor running this workflow commits the whole
scratch tree.

## 2. `LIMITATIONS.md` gains four entries beyond those already listed

**Invariant (A) holds only from a stoppable start.** On every joint,
`qd^2 <= 2 * qdd_max * d`, where `d` is the distance to the nearer position
limit. A joint at `pi - 0.05` moving at 0.99 rad/s needs 9.8 rad/s^2 to stop
against a declared 5.0. It is committed to an overshoot before the kernel is
called, and no acceleration-limited controller can rescue it. The kernel brakes
at its limit and reports the violation, but it does not promise (A) from there.
Say this plainly. It is a real narrowing of the headline claim.

**The braking bounds are deliberately conservative.** Each plans to use only
`brake_headroom` = 0.8 of the authority at the level above, so the arm decelerates
earlier and more gently than a perfect controller would. Reaching a joint limit
from rest takes 101 control steps rather than the theoretical minimum. That is
the price of the acceleration and jerk limits holding exactly, and it is the
right trade, but it is a cost and it should be stated.

**The `lever_sprint` test exercises a 0.6635 m lever, not the arm's full
0.85 m reach.** The longest lever that fits inside the declared workspace box is
0.6635 m, giving 2.65 times the declared TCP speed limit at full joint speed. The
real machine can do worse. The Cartesian guard is therefore demonstrated against
a hazard roughly 22 percent smaller than the machine's worst case.

**The Shield has never been composed with the driver it exists to wrap.**
`lerobot_ur/robot_ur5e.py` lives in the `mizan-kit` repository, which this
programme deliberately does not touch, so it is not present here. Every Shield
test runs against a fake robot implementing the LeRobot dict shape. What is
verified is that the Shield honours that contract; what is not verified is that
the real driver honours it too, or that the composition works end to end. The
first person to put these together should expect to find something, and the
README must not say "drop-in" without saying that.

**The red-team suite proves the kernel resists fifteen named attacks, not that
it resists all attacks.** The catalogue was written by the same process that
wrote the kernel. An adversary who reads the kernel will find attacks the
catalogue does not contain. Absence of an escape in this suite is evidence, not
proof, and the paper must not let the table imply otherwise.

## 3. `README.md` must not blur invariants (A) and (B)

(A) is a property of the kernel and is what the programme claims. (B) is a
property of the *simulated servo* and its margin is an artefact of that
simulation, not a statement about any real arm. Report them separately, with (B)
labelled as simulation-derived in the same sentence that gives its number. If the
README has room for only one, it is (A).

## 4. `PROTOCOL.md` section 6, hardware disclosure, is verbatim

Use exactly the text the Task 13 brief specifies. Do not soften it, do not add a
number to it, and do not fill it in from a datasheet. The datasheet values that
appear in `envelope.py` are bounds on the declaration, not measurements of this
arm, and section 6 is specifically about this arm.

## 5. The design spec has already been amended once

`docs/superpowers/specs/2026-09-05-m01-sentinel-design.md` was amended in commit
`8e8ed4d`, before any trial ran, to add the stoppability precondition and the
corrected braking bound. `PROTOCOL.md` section 10, deviations, should note that
the specification was corrected before the first run rather than after it, and
give that commit hash. A pre-registration that quietly changes is worthless; one
that records when and why it changed is not.

## 6. `scipy` is declared and imported nowhere. Say why, or drop it.

Verified: `grep -rn "scipy" sentinel/ tests/` returns **nothing**. Not one module
in this package imports it, yet `requirements.txt` declares `scipy>=1.11`.

That is not necessarily wrong, but it is currently unexplained, and an
unexplained dependency is a lie about what the project needs. The reason it is
there is indirect: `cairo_protocol.stats`, which lives in the `mizan-kit`
repository and which `redteam.py` imports for `anytime_cs`, uses
`scipy.special.betaln` when it is available and falls back to `math.lgamma` when
it is not. Dropping scipy here would silently push that computation onto the
fallback path whenever `cairo_protocol` *is* importable, changing the confidence
sequences this programme reports without anyone noticing.

Do one of these, not neither:

- **Keep it and annotate it**, which is the recommendation:

  ```
  numpy>=1.26
  # scipy is not imported by sentinel/ at all. It is declared because
  # cairo_protocol.stats, which redteam.py imports for anytime_cs, uses
  # scipy.special.betaln when present and silently falls back to math.lgamma
  # when absent. Removing scipy would change the confidence sequences this
  # programme reports without any error.
  scipy>=1.11
  ```

- **Or drop it** and state in `LIMITATIONS.md` that the confidence sequences use
  the `math.lgamma` fallback path.

The verification step in Step 5 of the brief should also confirm that a clean
checkout with only `requirements.txt` installed can actually run `pytest` and
`python -m sentinel.redteam --quick`, which is what would have caught this.

## 7. Portability, already verified

The controller checked and there is nothing to fix here, but confirm it stays
true at the end: `sentinel/` and `tests/` contain no hardcoded interpreter paths,
no `.venv` references, no `sys.path` manipulation, and no imports outside stdlib
and numpy. The suite is genuinely portable, so the CI matrix on `windows-latest`
and `ubuntu-latest` will work from a fresh clone. Do not introduce any of those
while writing the runner scripts.

## 8. Numbers come from runs, not from this file

Everything above is prose. Every figure in `README.md` is copied from
`results/redteam.csv` produced by the actual run in Step 1 of the brief. If a
number in this correction file disagrees with the run, the run wins and the
disagreement is itself worth a line in `LIMITATIONS.md`.
