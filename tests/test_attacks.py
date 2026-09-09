import numpy as np

from sentinel.attacks import EPISODE_Q0, LEVER_POSTURE, REGISTRY, build_all
from sentinel.envelope import Envelope
from sentinel.kernel import SafetyKernel
from sentinel.kinematics import tcp_position
from sentinel.sim import SimPlant
from sentinel.types import Action, RobotState, Status

EXPECTED = {
    "nan_inject", "slam_to_limit", "velocity_ramp", "jerk_chatter",
    "unit_confusion", "index_permute", "sign_flip", "stale_replay",
    "watchdog_starve", "workspace_escape", "force_grind", "gripper_crush",
    "lever_sprint", "dt_spoof", "slow_drift",
}

DT = 1 / 30


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def tick(self, dt=DT):
        self.t += dt


def test_the_catalogue_is_complete_and_named_as_the_spec_says():
    assert set(REGISTRY) == EXPECTED


def test_every_attack_declares_the_guard_it_expects_to_trip():
    for name, spec in REGISTRY.items():
        assert spec.expect is not None, f"{name} declares no expected guard"


def test_every_attack_produces_a_well_formed_action():
    env = Envelope.ur5e_declared()
    rng = np.random.default_rng(0)
    s = RobotState(q=np.zeros(6), qd=np.zeros(6), t_mono=0.0)
    for name, fn in build_all(env, rng).items():
        a = fn(s, 0, 0.0)
        assert isinstance(a, Action)
        assert a.q.shape == (6,), name


def test_attacks_are_deterministic_under_a_seed():
    env = Envelope.ur5e_declared()
    s = RobotState(q=np.zeros(6), qd=np.zeros(6), t_mono=0.0)
    a = build_all(env, np.random.default_rng(3))
    b = build_all(env, np.random.default_rng(3))
    for name in REGISTRY:
        assert np.array_equal(a[name](s, 5, 0.1).q, b[name](s, 5, 0.1).q), name


def test_slow_drift_never_requests_an_illegal_per_step_change():
    # The whole point of this attack: it is legal at every instant.
    env = Envelope.ur5e_declared()
    fn = build_all(env, np.random.default_rng(0))["slow_drift"]
    q = np.zeros(6)
    for i in range(600):
        a = fn(RobotState(q=q, qd=np.zeros(6), t_mono=0.0), i, i / 30)
        assert np.all(np.abs(a.q - q) <= env.qd_max * (1 / 30) + 1e-9)
        assert np.all(a.q <= env.q_max) and np.all(a.q >= env.q_min)
        q = a.q


def test_lever_sprint_requests_only_legal_joint_speeds():
    # Starts already at LEVER_POSTURE: this covers the steady-state sweep,
    # where the ramp's delta is zero and only joint 0 moves, at a legal rate.
    # test_lever_sprint_is_legal_from_the_episode_start_state below covers the
    # other half -- ramping into the posture at a legal rate from Task 11's
    # actual episode start, which is nowhere near it (task-10-fix-1.md).
    env = Envelope.ur5e_declared()
    fn = build_all(env, np.random.default_rng(0))["lever_sprint"]
    q = np.concatenate(([0.0], LEVER_POSTURE))
    for i in range(300):
        a = fn(RobotState(q=q, qd=np.zeros(6), t_mono=0.0), i, i / 30)
        assert np.all(np.abs(a.q - q) <= env.qd_max * (1 / 30) + 1e-9)
        q = a.q


def test_lever_sprint_stays_inside_the_workspace_box():
    # The attack's whole claim is that it is legal everywhere the box guard
    # can see, and dangerous only in flange speed. If it leaves the box,
    # tcp_box fires instead of tcp_speed and the attack proves nothing.
    #
    # LEVER_POSTURE (task-10-fix-1.md) was chosen by searching for the
    # longest lever that keeps 5 mm of clearance from the box, rather than
    # maximising to the boundary -- the earlier posture landed 0.04 mm
    # outside with zero margin by construction, which is why containment can
    # be asserted exactly here with no tolerance.
    env = Envelope.ur5e_declared()
    fn = build_all(env, np.random.default_rng(0))["lever_sprint"]
    q = np.concatenate(([0.0], LEVER_POSTURE))
    for i in range(400):
        a = fn(RobotState(q=q, qd=np.zeros(6), t_mono=0.0), i, i / 30)
        assert env.tcp_box.excursion(tcp_position(a.q)) == 0.0, (
            f"lever_sprint left the box at step {i}: {tcp_position(a.q)}")
        q = a.q


def test_lever_sprint_is_legal_from_the_episode_start_state():
    # The two lever tests above both start already sitting at LEVER_POSTURE,
    # so neither can see a transition-into-posture violation. Task 11
    # declares every episode starts at EPISODE_Q0, which is nowhere near it:
    # this is the test that would have caught the attack snapping q[1:] to
    # the posture in one illegal step (task-10-fix-1.md, finding 1).
    env = Envelope.ur5e_declared()
    fn = build_all(env, np.random.default_rng(0))["lever_sprint"]
    q = EPISODE_Q0.copy()
    cap = env.qd_max * (1 / 30)
    for i in range(600):
        a = fn(RobotState(q=q, qd=np.zeros(6), t_mono=0.0), i, i / 30)
        assert np.all(np.abs(a.q - q) <= cap + 1e-9), (
            f"step {i} requested {np.abs(a.q - q)} against a cap of {cap}")
        q = a.q


def test_watchdog_starve_actually_exceeds_the_watchdog():
    env = Envelope.ur5e_declared()
    assert REGISTRY["watchdog_starve"].skip_seconds > env.max_dt_s, (
        "the attack cannot reach the guard it declares")


def test_index_permute_and_sign_flip_never_violate_q_limit_but_do_trip_qd_max():
    # task-10-fix-2.md, finding 1. Both attacks' requested targets are
    # always inside [q_min, q_max] by construction -- index_permute
    # permutes an in-range draw from [-2.5, 2.5], sign_flip negates a state
    # that is already in [-pi, pi] -- so q_limit can never fire for either.
    # That was the original, wrong declared guard (both said q_limit). What
    # they actually and reliably trip is qd_max: reaching that in-range
    # target in one step from the episode start demands up to ~101.6 rad/s
    # (index_permute) or ~87.0 rad/s (sign_flip) against a declared 1.0
    # rad/s, measured over 600 steps driven from the episode start.
    #
    # Run through the real kernel (every guard active) rather than just
    # inspecting the raw action, so this is the same kind of proof
    # Task 11's test_every_attack_actually_trips_the_guard_it_declared will
    # demand: not "the action looks dangerous" but "the kernel actually
    # calls it out, on the declared rule, and never on q_limit".
    env = Envelope.ur5e_declared()
    for name in ("index_permute", "sign_flip"):
        assert REGISTRY[name].expect == "qd_max", (
            f"{name} should declare qd_max, not {REGISTRY[name].expect!r}")

        clock = Clock()
        kernel = SafetyKernel(env, clock=clock)
        fn = build_all(env, np.random.default_rng(0))[name]

        q_limit_ever_fired = False
        qd_max_ever_fired = False
        q = EPISODE_Q0.copy()
        for i in range(600):
            state = RobotState(q=q, qd=np.zeros(6), t_mono=clock.t)
            v = kernel.filter(state, fn(state, i, clock.t))
            rules = {viol.rule for viol in v.violations}
            q_limit_ever_fired = q_limit_ever_fired or "q_limit" in rules
            qd_max_ever_fired = qd_max_ever_fired or "qd_max" in rules
            q = v.action.q.copy()
            clock.tick()

        assert qd_max_ever_fired, f"{name} never tripped qd_max in 600 steps"
        assert not q_limit_ever_fired, (
            f"{name} tripped q_limit, contradicting task-10-fix-2.md finding 1"
        )


def test_declared_overrides_are_visible_and_justified():
    # An attack demonstrated against a modified envelope is a weaker result.
    # Overrides are allowed, but never silent.
    for name, spec in REGISTRY.items():
        if spec.envelope_override is None:
            continue
        assert spec.note, f"{name} overrides the envelope without explaining why"
        Envelope.ur5e_declared().replace(**spec.envelope_override)   # must construct


def test_slow_drift_exhausts_its_declared_path_budget():
    # The declared 25.0 m path_budget_m is unreachable in a 2000-step episode
    # by any attack whatsoever: 2000 steps * 1/30 s = 66.67 s, and at the
    # declared tcp_speed_max = 0.25 m/s that caps total flange travel at
    # 66.67 * 0.25 = 16.667 m, 8.333 m short of 25.0 m (task-10-fix-3.md,
    # finding 2). tests/test_kernel_budgets.py proved only that the guard
    # itself works, using a reduced 0.5 m budget; it never proved this attack
    # -- or any attack -- could reach the DECLARED value within an episode,
    # because nothing can.
    #
    # slow_drift therefore declares an episode-scale envelope_override
    # instead, and this test proves it reaches THAT, through the real
    # assembled kernel and plant with every guard active -- jerk,
    # acceleration, velocity, braking, and Cartesian box/speed shaping
    # included, none of which the controller's own open-loop sweep (which
    # only clips to joint limits) could see. Both prior attempts at this
    # number were open loop and both were wrong: the first ignored the
    # Cartesian guard entirely and deadlocked at 0.016 m with the goal
    # centred on np.zeros(6) (task-10-fix-3.md, finding 1); this closed-loop
    # check is what would have caught that.
    spec = REGISTRY["slow_drift"]
    assert spec.envelope_override == {"path_budget_m": 8.0}
    env = Envelope.ur5e_declared().replace(**spec.envelope_override)
    clock = Clock()
    kernel = SafetyKernel(env, clock=clock)
    plant = SimPlant(q0=EPISODE_Q0, qdd_max=env.qdd_max)
    fn = build_all(env, np.random.default_rng(0))["slow_drift"]
    tripped_at = None
    for i in range(2000):
        state = plant.state(t_mono=clock.t)
        v = kernel.filter(state, fn(state, i, clock.t))
        if v.status is Status.STOP and any(x.rule == "path_budget" for x in v.violations):
            tripped_at = i
            break
        plant.step(v.action.q, DT)
        clock.tick(DT)
    assert tripped_at is not None, (
        f"slow_drift never exhausted the {env.path_budget_m} m override budget "
        f"in 2000 steps; reached {kernel.path_m:.3f} m")


def test_every_declared_start_state_is_inside_the_envelope():
    # task-11-correction.md #4: an attack may declare where its episode
    # should start (lever_sprint does, because its joint-space ramp is not
    # Cartesian-safe from EPISODE_Q0). A badly declared start_q must fail
    # loudly rather than quietly, and this is the same check the red-team
    # runner applies at step 0.
    env = Envelope.ur5e_declared()
    for name, spec in REGISTRY.items():
        if spec.start_q is None:
            continue
        q = np.asarray(spec.start_q, dtype=float)
        assert np.all(q <= env.q_max) and np.all(q >= env.q_min), name
        assert env.tcp_box.excursion(tcp_position(q)) == 0.0, name


def test_lever_sprint_declares_its_own_start_state():
    # The general lesson (task-11-correction.md #4): a linear joint-space path
    # between two Cartesian-safe configurations is not guaranteed to keep the
    # intermediate path inside the box. Both endpoints being safe says
    # nothing about the middle. lever_sprint's ramp from EPISODE_Q0 is legal
    # in joint space but strays outside the box for 51 of its first 130
    # steps, peaking 0.1237 m outside at call 41 -- so it must declare its
    # own Cartesian-safe start rather than rely on the shared episode start.
    spec = REGISTRY["lever_sprint"]
    assert spec.start_q is not None, "lever_sprint must declare start_q"
    assert np.array_equal(spec.start_q, np.concatenate(([0.0], LEVER_POSTURE)))
