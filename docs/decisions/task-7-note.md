# Task 7 note: the brief is correct as written, plus one thing for LIMITATIONS

No correction needed. Both budget tests were measured before dispatch and both
pass with a wide margin.

| test | measured | loop bound in the test | verdict |
|---|---|---|---|
| path budget exhausted by legal oscillation | trips at step **67** | 4000 | holds, 1.7 percent of the loop |
| contact-time budget exhausted | needs **6** steps | 20 | holds |

Supporting numbers, in case a test ever starts failing and someone needs to know
what changed: 30 shaped steps toward `q[1] = -0.3` move the flange **0.4328 m**
and reach `q[1] = -0.2886` of the -0.3 requested. The full 4000-step oscillation
accumulates **62.0108 m** of flange path.

## The one thing that is not a defect but must reach LIMITATIONS

The declared default is `path_budget_m = 25.0`. At the declared TCP speed limit
of 0.25 m/s that is **100 seconds**, about 1.7 minutes, of continuous full-speed
motion before the kernel latches a STOP that only an explicit `rearm` clears.

That is the right order of magnitude for its purpose, which is catching an arm
that keeps moving when nobody asked it to. It is far too small for production:
an ordinary pick-and-place cycle moves a metre or two of flange path, so this
budget would latch after roughly a dozen cycles and demand a human rearm.

This is not a bug and the value should not be changed. It is a *declared* value
sized for red-team episodes, and the whole point of the provenance system is that
a real deployment sets its own. But shipping it without saying so would invite
someone to drop this kernel in front of a working cell and be baffled when it
stops after two minutes.

`LIMITATIONS.md` must say, in plain words: the cumulative budgets are sized for
the red-team suite, not for production; a real deployment sets `path_budget_m`
and `contact_time_budget_s` from its own task cycle and its own hazard analysis;
and both reset only on an explicit `rearm`, by design, because a budget that
resets itself is not a budget.

## Interaction with Task 11 worth knowing

Red-team episodes are 2000 steps at 1/30 s, so about 66.7 s. A legal oscillation
accumulates roughly 31 m in that time, above the 25 m budget, so attacks that
move a lot will trip the path budget incidentally, not only `slow_drift`.

This is harmless because `run_episode` calls `rearm` after every STOP and
continues, which resets both budgets. But it does mean `rules_fired` for several
attacks will contain `path_budget` alongside their declared guard. That is
correct behaviour and not a finding: `expected_guard_fired` checks that the
declared guard is *among* those that fired, not that it is the only one.
