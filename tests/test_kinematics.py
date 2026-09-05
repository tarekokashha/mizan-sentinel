import numpy as np
import pytest

from sentinel.kinematics import DH_A, DH_D, fk, jacobian, tcp_position, wrist_center


def test_base_height_at_zero_configuration_is_d1():
    # At q = 0 the shoulder sits d1 above the base and the arm lies along +x/-z
    # depending on sign conventions, but the wrist centre height is fixed by
    # the a and d table alone.
    wc = wrist_center(np.zeros(6))
    assert wc[2] == pytest.approx(DH_D[0], abs=1e-12)


def test_maximum_planar_reach_of_the_wrist_centre_is_a2_plus_a3():
    # Sweep the shoulder and elbow; the furthest the wrist centre can get from
    # the base axis is |a2| + |a3|. This validates the two a values together.
    best = 0.0
    for q2 in np.linspace(-np.pi, np.pi, 181):
        for q3 in np.linspace(-np.pi, np.pi, 181):
            q = np.array([0.0, q2, q3, 0.0, 0.0, 0.0])
            best = max(best, float(np.linalg.norm(wrist_center(q)[:2])))
    assert best == pytest.approx(abs(DH_A[1]) + abs(DH_A[2]), rel=1e-3)


def test_forward_kinematics_is_a_rigid_transform_everywhere():
    # A mistyped alpha or a transposed DH matrix breaks orthonormality. This
    # catches it across the whole configuration space, which inspection cannot.
    rng = np.random.default_rng(0)
    for _ in range(200):
        T = fk(rng.uniform(-np.pi, np.pi, 6))
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
        assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0])


def test_tcp_stays_within_the_geometric_reach_bound():
    # The sound bound: no chain of links can put the flange further from the
    # shoulder than the sum of the link lengths that separate them. This can
    # never fail for a correct DH table, so a failure here means the table is
    # wrong, not the tolerance.
    #
    # Deliberately NOT compared against the 850 mm datasheet reach: that is a
    # working radius to the flange in the arm's principal plane, while this is
    # a 3D shoulder-to-flange distance that also picks up the perpendicular d5
    # and d6 wrist offsets. Measured maximum is 0.9626, above 850 mm, and
    # correctly so.
    bound = abs(DH_A[1]) + abs(DH_A[2]) + DH_D[3] + DH_D[4] + DH_D[5]
    shoulder = np.array([0.0, 0.0, DH_D[0]])
    rng = np.random.default_rng(1)
    q = rng.uniform(-np.pi, np.pi, (500, 6))
    r = np.array([np.linalg.norm(tcp_position(x) - shoulder) for x in q])
    assert r.max() <= bound + 1e-9


def test_measured_maximum_reach_is_pinned():
    # Regression pin. The controller measured 0.962623 over 4000 seeded
    # configurations before this file existed. A change here means the
    # kinematics changed.
    shoulder = np.array([0.0, 0.0, DH_D[0]])
    rng = np.random.default_rng(1)
    q = rng.uniform(-np.pi, np.pi, (4000, 6))
    r = np.array([np.linalg.norm(tcp_position(x) - shoulder) for x in q])
    assert r.max() == pytest.approx(0.962623, abs=1e-5)


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(2)
    q = rng.uniform(-2.0, 2.0, 6)
    J = jacobian(q)
    eps = 1e-7
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        num = (tcp_position(q + dq) - tcp_position(q - dq)) / (2 * eps)
        assert np.allclose(J[:3, i], num, atol=1e-6)


def test_jacobian_angular_rows_match_finite_differences():
    # The linear rows are covered; the angular rows are not, and an axis or
    # sign error there would pass every existing test. For a small rotation,
    # R(q+eps) @ R(q-eps).T is approximately I + skew(omega * 2*eps), so the
    # skew part recovers omega directly.
    rng = np.random.default_rng(4)
    q = rng.uniform(-2.0, 2.0, 6)
    J = jacobian(q)
    eps = 1e-6
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        dR = fk(q + dq)[:3, :3] @ fk(q - dq)[:3, :3].T
        omega = np.array([dR[2, 1] - dR[1, 2],
                          dR[0, 2] - dR[2, 0],
                          dR[1, 0] - dR[0, 1]]) / (2.0 * 2.0 * eps)
        assert np.allclose(J[3:, i], omega, atol=1e-5)


def test_fk_is_deterministic():
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    assert np.array_equal(fk(q), fk(q))
