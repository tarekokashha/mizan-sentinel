import numpy as np
import pytest

from sentinel.envelope import Envelope
from sentinel.kinematics import tcp_position
from sentinel.sim import PlaneContact, SimPlant

DT = 1 / 30


def test_the_plant_converges_to_a_held_command():
    p = SimPlant(q0=np.zeros(6), qdd_max=Envelope.ur5e_declared().qdd_max)
    target = np.full(6, 0.4)
    for _ in range(200):
        p.step(target, DT)
    assert np.allclose(p.q, target, atol=1e-3)
    assert np.allclose(p.qd, 0.0, atol=1e-3)


def test_a_feasible_ramp_is_followed_without_overshoot():
    env = Envelope.ur5e_declared()
    p = SimPlant(q0=np.zeros(6), qdd_max=env.qdd_max)
    q = np.zeros(6)
    peak = 0.0
    for _ in range(300):
        q = np.minimum(q + env.qd_max * DT, 0.5)     # legal ramp to 0.5 then hold
        p.step(q, DT)
        peak = max(peak, float(p.q.max()))
    assert peak <= 0.5 + env.plant_margin_rad


def test_an_infeasible_step_overshoots():
    # This is why the plant is second order. A position-replay plant could
    # never fail, and the braking bound would have no test.
    env = Envelope.ur5e_declared()
    p = SimPlant(q0=np.zeros(6), qdd_max=env.qdd_max)
    for _ in range(60):
        p.step(np.full(6, 1.0), DT)            # a full step, no shaping at all
        if p.q.max() > 1.0 + env.plant_margin_rad:
            return
    pytest.fail("the plant tracked an infeasible command perfectly; it is not a servo")


def test_the_plant_is_deterministic():
    a = SimPlant(q0=np.zeros(6), qdd_max=Envelope.ur5e_declared().qdd_max, seed=7)
    b = SimPlant(q0=np.zeros(6), qdd_max=Envelope.ur5e_declared().qdd_max, seed=7)
    for i in range(50):
        cmd = np.full(6, 0.01 * i)
        a.step(cmd, DT)
        b.step(cmd, DT)
    assert np.array_equal(a.q, b.q)


def test_plane_contact_pushes_back_only_below_the_plane():
    c = PlaneContact(z=0.20, stiffness=5000.0)
    assert np.allclose(c.wrench(np.array([0.3, 0.0, 0.30])), 0.0)
    w = c.wrench(np.array([0.3, 0.0, 0.19]))
    assert w[2] == pytest.approx(5000.0 * 0.01)


def test_state_reports_the_contact_wrench():
    c = PlaneContact(z=2.0, stiffness=100.0)     # always in contact
    p = SimPlant(q0=np.zeros(6), qdd_max=Envelope.ur5e_declared().qdd_max, contact=c)
    p.step(np.zeros(6), DT)
    s = p.state(t_mono=0.0)
    assert np.linalg.norm(s.wrench[:3]) > 0.0
    assert np.allclose(s.q, p.q)
