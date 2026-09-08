import numpy as np

from sentinel.attacks import LEVER_POSTURE, REGISTRY, build_all
from sentinel.envelope import Envelope
from sentinel.kinematics import tcp_position
from sentinel.types import Action, RobotState

EXPECTED = {
    "nan_inject", "slam_to_limit", "velocity_ramp", "jerk_chatter",
    "unit_confusion", "index_permute", "sign_flip", "stale_replay",
    "watchdog_starve", "workspace_escape", "force_grind", "gripper_crush",
    "lever_sprint", "dt_spoof", "slow_drift",
}


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


EPISODE_Q0 = np.array([0.0, -1.2, 1.2, -1.5, -1.5708, 0.0])


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
