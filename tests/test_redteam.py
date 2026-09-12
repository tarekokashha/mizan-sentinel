import numpy as np
import pytest

from sentinel.attacks import EPISODE_Q0, REGISTRY, build_all
from sentinel.envelope import Envelope
from sentinel.journal import verify
from sentinel.kernel import SafetyKernel
from sentinel.kinematics import tcp_position
from sentinel.redteam import run_all, run_attack, run_episode
from sentinel.sim import PlaneContact, SimPlant
from sentinel.types import Status


def test_no_attack_escapes_the_envelope():
    env = Envelope.ur5e_declared()
    reports = run_all(env, episodes=3, steps=300, seed=0)
    failed = [r.attack for r in reports if r.escape_rate > 0.0]
    assert failed == [], f"attacks escaped the envelope: {failed}"


# Previously known-silent (task-11-report.md, tracked under xfail(strict=True)
# below): force_grind's q[1] += 0.02 shoulder sweep drove the flange's X
# coordinate past box.lo[0] = -0.65 at q1 ~= -1.155 (tcp.x reaches -0.651)
# while tcp.z was still ~0.43 -- far short of the z=0.10 contact plane -- and
# the Cartesian guard's bisection permanently held q1 at that X boundary, so
# force_max never fired. Fixed in task-11-fix-1.md, "Finding 2": sweeping
# joint 2 instead of joint 1 reaches z = 0.0625, well under the plane, and
# trips force_max against the unmodified declared envelope (no override
# needed). See test_force_grind_trips_force_max_against_the_declared_plane
# below for the closed-loop step and force this now produces. No attack in
# this catalogue is known-silent any longer, so this parametrization is a
# fully positive assertion with no exceptions.
@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_attack_actually_trips_the_guard_it_declared(name):
    # A suite where nothing fires is a suite that proves nothing.
    #
    # steps=2000 (the full per-episode budget), not the brief's steps=300:
    # measured directly against the real kernel and plant, slow_drift's
    # path_budget guard -- a session-scale guard by design (task-10-fix-3.md)
    # -- trips at step 1474 (49.1 s) against its declared envelope_override,
    # not within 300 steps (10 s). That is a gap in this test's own budget,
    # not a defect in the attack: tests/test_attacks.py already proves
    # slow_drift reaches the override over a full 2000-step episode. Widening
    # the budget here gives every guard, fast or session-scale, a fair chance
    # to fire in the same test.
    env = Envelope.ur5e_declared()
    r = run_attack(name, env, episodes=2, steps=2000, seed=1)
    assert r.expected_guard_fired, (
        f"{name} expected {REGISTRY[name].expect} and it never fired; "
        "the attack is not exercising the guard it claims to")


def test_force_grind_trips_force_max_against_the_declared_plane():
    # task-11-fix-1.md, "Finding 2": force_grind's original joint-1 sweep
    # never reached the contact plane closed loop -- it deadlocked against
    # the workspace box's X-wall at tcp.z ~= 0.43, far short of z = 0.10, so
    # force_max never fired (see the note above
    # test_every_attack_actually_trips_the_guard_it_declared). Sweeping
    # joint 2 instead was measured to reach z = 0.0625, well under the
    # plane, and to trip force_max at step 180 with 40.21 N against the
    # 40.0 N limit -- against the declared PlaneContact(z=0.10) and the
    # unmodified declared envelope, no override needed. This proves that
    # measurement through the real, unmodified SafetyKernel and SimPlant,
    # not by inspecting the raw action: the same class of open-loop mistake
    # this project has now made four times (task-11-fix-1.md).
    env = Envelope.ur5e_declared()
    assert REGISTRY["force_grind"].envelope_override is None, (
        "force_grind should trip its declared guard against the unmodified "
        "declared envelope -- an override here would be a weaker result")

    class Clock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            return self.t

        def tick(self, dt=1.0 / 30.0):
            self.t += dt

    clock = Clock()
    kernel = SafetyKernel(env, clock=clock)
    contact = PlaneContact(z=0.10, stiffness=8000.0)
    plant = SimPlant(q0=EPISODE_Q0, qdd_max=env.qdd_max, contact=contact, seed=0)
    fn = build_all(env, np.random.default_rng(0))["force_grind"]

    tripped_at, tripped_force = None, None
    for i in range(2000):
        state = plant.state(t_mono=clock.t)
        v = kernel.filter(state, fn(state, i, clock.t))
        if v.status is Status.STOP and any(x.rule == "force_max" for x in v.violations):
            tripped_at = i
            tripped_force = [x.measured for x in v.violations if x.rule == "force_max"][0]
            break
        plant.step(v.action.q, 1.0 / 30.0)
        clock.tick()

    assert tripped_at is not None, (
        "force_grind never tripped force_max in 2000 steps against the "
        "declared contact plane")
    assert tripped_force > env.force_max, (
        f"force_max fired but the measured force {tripped_force} did not "
        f"exceed the declared limit {env.force_max}")
    # Not a session-scale guard like slow_drift's path_budget: this should
    # fire early, well inside even a --quick run's 300-step budget. Measured
    # at step 180 with 40.21 N; a generous margin around that guards against
    # incidental drift without pinning brittle exact values.
    assert tripped_at < 300, (
        f"force_grind tripped force_max at step {tripped_at}, far later "
        "than the measured step 180 -- investigate before trusting the fix")


def test_the_escape_rate_carries_an_anytime_valid_interval(monkeypatch):
    # I7 / final-fix-1.md: the original version of this test asserted
    # cs_lo <= escape_rate <= cs_hi while _ANYTIME_CS is None in this
    # environment (cairo_protocol lives in another repository and is
    # deliberately not on this path). _confidence_sequence always returns
    # (p, p) on that fallback path, so the inequality held by construction
    # -- it could not fail no matter what the code around it did, the same
    # thing test_confidence_sequence_degrades_loudly_when_cairo_protocol_is_unavailable
    # already proves directly. Fake a non-degenerate anytime-valid interval
    # so this test actually exercises the wiring that carries cs_lo/cs_hi
    # from _confidence_sequence through to the report, and can fail if that
    # wiring breaks -- e.g. if run_attack silently ignored _ANYTIME_CS and
    # always used the raw-rate fallback, cs_lo and cs_hi below would both
    # equal escape_rate exactly, not diverge from it by 0.1 in each
    # direction. Deliberately not clamped to [0, 1]: this is a fake standing
    # in for an unknown real interval shape, not a realistic one, and an
    # unclamped interval is what makes the assertion exact rather than an
    # inequality that could hold by coincidence near the boundary.
    import sentinel.redteam as rt

    def fake_cs(outcomes, alpha=0.05):
        p = float(np.mean(outcomes)) if len(outcomes) else 0.0
        return p - 0.1, p + 0.1

    monkeypatch.setattr(rt, "_ANYTIME_CS", fake_cs)
    env = Envelope.ur5e_declared()
    r = run_attack("slam_to_limit", env, episodes=5, steps=200, seed=0)
    assert r.cs_lo == pytest.approx(r.escape_rate - 0.1)
    assert r.cs_hi == pytest.approx(r.escape_rate + 0.1)


def test_runs_are_reproducible_under_a_seed():
    env = Envelope.ur5e_declared()
    a = run_attack("jerk_chatter", env, episodes=3, steps=200, seed=42)
    b = run_attack("jerk_chatter", env, episodes=3, steps=200, seed=42)
    assert a.escape_rate == b.escape_rate
    assert a.rules_fired == b.rules_fired


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_episode_traces_genuinely_vary_across_repetitions(name):
    # CRITICAL 2 / final-fix-1.md: measured by the reviewer, 12 of 15 attacks
    # produced bit-identical episodes across all 200 repetitions before this
    # fix -- 11 attack builders never touched the rng they were handed,
    # SimPlant.seed was stored and never read, and every episode started
    # from the same fixed q0. A confidence sequence over 200 identical
    # deterministic repeats carries the evidential content of n = 1. This is
    # the test the suite silently lacked: trace-hash episodes 0, 1 and 7 (the
    # same per-episode seed formula run_attack uses, seed * 100_003 + e) and
    # assert they actually differ, for every attack in the catalogue, not
    # just the three the reviewer found already varying.
    env = Envelope.ur5e_declared()
    spec = REGISTRY[name]
    hashes = [run_episode(spec, env, seed=0 * 100_003 + e, steps=300).trace_sha256
             for e in (0, 1, 7)]
    assert len(set(hashes)) == 3, (
        f"{name}: episodes 0, 1 and 7 are not all distinct ({hashes}) -- the "
        "plant's start-state jitter and sensor noise (both driven by "
        "SimPlant's per-episode seed) should make every attack in this "
        "catalogue vary episode to episode; if one genuinely cannot, that is "
        "a finding to report, not a reason to weaken this assertion")


def test_repeating_the_same_episode_seed_reproduces_the_same_trace():
    # The other half of CRITICAL 2's contract: genuine variation across
    # different seeds must not come at the cost of reproducibility under the
    # same seed -- test_runs_are_reproducible_under_a_seed already covers
    # run_attack's aggregate escape_rate/rules_fired; this covers the
    # underlying per-episode trace directly.
    env = Envelope.ur5e_declared()
    spec = REGISTRY["slow_drift"]
    a = run_episode(spec, env, seed=12345, steps=300)
    b = run_episode(spec, env, seed=12345, steps=300)
    assert a.trace_sha256 == b.trace_sha256


def test_a_deliberately_broken_kernel_is_caught(monkeypatch):
    # The suite must be able to fail. Disable the Cartesian guard entirely
    # (the mechanism that actually does the work here) and the plant should
    # escape the declared box.
    import sentinel.kernel as kern

    env = Envelope.ur5e_declared()
    monkeypatch.setattr(kern.SafetyKernel, "_cartesian",
                        lambda self, q_ref, q_cmd, dt: (q_cmd, []))
    r = run_attack("workspace_escape", env, episodes=2, steps=400, seed=0)
    assert r.escape_rate > 0.0, "disabling the Cartesian guard did not cause an escape"


def test_the_journal_from_a_run_verifies(tmp_path):
    env = Envelope.ur5e_declared()
    p = tmp_path / "redteam.jsonl"
    run_all(env, episodes=1, steps=100, seed=0, journal_path=p)
    assert verify(p)[0] is True


def test_the_episode_start_state_is_inside_the_envelope():
    # task-11-correction.md #2: an episode that starts outside the envelope
    # reports an escape on step 0 for every attack and proves nothing. Verify
    # rather than trust.
    env = Envelope.ur5e_declared()
    assert np.all(EPISODE_Q0 <= env.q_max) and np.all(EPISODE_Q0 >= env.q_min)
    assert env.tcp_box.excursion(tcp_position(EPISODE_Q0)) == 0.0


def test_a_badly_declared_start_q_fails_loudly_at_step_zero():
    # task-11-correction.md #4 (start_q): a badly declared start_q must fail
    # the same envelope check as EPISODE_Q0, not be silently accepted.
    from sentinel.attacks import AttackSpec

    env = Envelope.ur5e_declared()
    bad_spec = AttackSpec(
        name="_bogus_out_of_box_start", expect="tcp_box",
        build=lambda env, rng: (lambda state, step, t: None),
        start_q=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # tcp is outside the box
    )
    with pytest.raises(ValueError):
        run_episode(bad_spec, env, seed=0, steps=1)


def test_confidence_sequence_degrades_loudly_when_cairo_protocol_is_unavailable():
    # cairo_protocol lives in another repository (E:\Robotics Projects\mizan-kit)
    # and is deliberately not on this path (task-11-correction.md #6). Confirm
    # the degradation is real, not aspirational: the fallback must be exactly
    # the raw rate, never a differently-derived interval kept under the same
    # column name.
    import sentinel.redteam as rt

    assert rt._ANYTIME_CS is None, (
        "cairo_protocol imported successfully in this environment; if that is "
        "genuinely now true, update this test rather than deleting it -- the "
        "anytime-valid path should be exercised for real instead of asserting "
        "the raw-rate fallback")
    lo, hi = rt._confidence_sequence([1, 0, 1, 1])
    assert lo == hi == 0.75


def test_the_envelope_override_is_visible_in_the_table_and_the_csv(tmp_path, capsys):
    # An attack demonstrated against a modified envelope is a weaker result
    # than one demonstrated against the declared envelope, and the report has
    # to say which it is. A silent override would be worse than the original
    # defect (task-10-fix-3.md).
    from sentinel.redteam import main

    out_csv = tmp_path / "redteam.csv"
    rc = main(["--attack", "slow_drift", "--quick", "--out", str(out_csv)])
    assert rc in (0, 1)
    printed = capsys.readouterr().out
    assert "path_budget_m=8.0" in printed, "the override never appears in the printed table"
    csv_text = out_csv.read_text(encoding="utf-8")
    assert "path_budget_m" in csv_text and "8.0" in csv_text, (
        "the override never appears in the CSV")


def test_the_escape_epsilon_is_named_and_cannot_be_quietly_widened():
    """The escape criterion is the most important definition in this
    programme, and its slack used to be a bare 1e-9 inline in the
    expression that applies it.

    A magic constant buried in a safety decision is exactly the kind of
    value that gets nudged upward one day to make a run look clean. This
    pins the direction: ESCAPE_EPS_RAD may be tightened, never loosened,
    and loosening it has to fail a test with a name that says what happened
    rather than slipping through as a one-character diff.

    The bound is arithmetic, not a judgement. One ulp near the declared
    joint limit of pi is about 4.4e-16, so 1e-9 is already about a million
    times the floating point noise it exists to absorb.
    """
    from sentinel.redteam import ESCAPE_EPS_RAD

    assert ESCAPE_EPS_RAD <= 1e-9, (
        f"ESCAPE_EPS_RAD was widened to {ESCAPE_EPS_RAD:g}. An escape smaller "
        f"than the threshold is still an escape: report it and investigate, "
        f"do not raise the bar until it disappears.")
    assert ESCAPE_EPS_RAD > 0.0, "a zero epsilon makes exact-limit commands flap on float noise"

    ulp_at_limit = float(np.spacing(np.pi))
    assert ESCAPE_EPS_RAD > ulp_at_limit * 1000, (
        f"ESCAPE_EPS_RAD {ESCAPE_EPS_RAD:g} is not comfortably above one ulp "
        f"at the joint limit ({ulp_at_limit:g}), so it no longer absorbs the "
        f"arithmetic noise it exists for")


def test_an_escape_just_above_the_epsilon_is_still_reported(monkeypatch):
    """The epsilon must not become a silent amnesty band.

    Follows the same shape as test_a_deliberately_broken_kernel_is_caught:
    break the kernel deliberately, in a way that lands a command just past
    the declared joint limit by ten times ESCAPE_EPS_RAD, and require the
    runner to call it an escape. Without this, widening the constant would
    be caught only by the bound above, and a bound is easier to argue with
    than a demonstration.
    """
    import sentinel.kernel as kern
    from sentinel.redteam import ESCAPE_EPS_RAD

    env = Envelope.ur5e_declared()
    real_filter = kern.SafetyKernel.filter
    over = ESCAPE_EPS_RAD * 10.0

    # Joint 5 on purpose. Nudging joint 0 instead makes this test pass for
    # the wrong reason: it swings the flange out of the Cartesian box, so the
    # plant TCP check reports the escape and the commanded-joint check under
    # test never has to work. Verified by mutation: with joint 0, disabling
    # the commanded-joint check entirely left this test green. Joint 5 is the
    # tool roll and moving it does not translate the flange origin, so the
    # joint check is the only thing that can catch this.
    def nudge_past_the_limit(self, state, action):
        verdict = real_filter(self, state, action)
        if verdict.action.q is not None:
            verdict.action.q[5] = env.q_max[5] + over
        return verdict

    monkeypatch.setattr(kern.SafetyKernel, "filter", nudge_past_the_limit)
    r = run_attack("slam_to_limit", env, episodes=1, steps=20, seed=0)
    assert r.escape_rate > 0.0, (
        f"a command {over:g} rad beyond q_max, ten times the epsilon, was not "
        f"reported as an escape")
