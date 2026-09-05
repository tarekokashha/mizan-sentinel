import numpy as np
import pytest

from sentinel.envelope import Box, Envelope
from sentinel.types import Status, Violation, worst


def test_status_ordering_picks_the_more_severe():
    assert worst(Status.PASS, Status.CLAMPED) is Status.CLAMPED
    assert worst(Status.STOP, Status.HOLD) is Status.STOP
    assert worst(Status.HOLD, Status.HOLD) is Status.HOLD


def test_declared_envelope_is_strictly_inside_the_datasheet():
    env = Envelope.ur5e_declared()
    assert np.all(env.q_max <= np.deg2rad(360.0))
    assert np.all(env.q_min >= -np.deg2rad(360.0))
    assert np.all(env.qd_max < np.deg2rad(180.0))
    assert env.tcp_speed_max < 1.0
    assert env.provenance["qd_max"] == "declared"


def test_sha256_is_stable_across_instances_and_sensitive_to_change():
    a = Envelope.ur5e_declared()
    assert a.sha256() == Envelope.ur5e_declared().sha256()
    assert a.replace(tcp_speed_max=a.tcp_speed_max + 1e-9).sha256() != a.sha256()


def test_json_roundtrip_preserves_hash_and_arrays():
    a = Envelope.ur5e_declared()
    b = Envelope.from_json(a.to_json())
    assert b.sha256() == a.sha256()
    assert np.array_equal(b.q_max, a.q_max)


def test_measured_provenance_is_refused_without_a_named_human():
    d = Envelope.ur5e_declared().to_dict()
    d["provenance"]["force_max"] = "measured"
    with pytest.raises(ValueError, match="measured"):
        Envelope.from_dict(d)


def test_measured_provenance_is_accepted_with_a_named_human():
    d = Envelope.ur5e_declared().to_dict()
    d["provenance"]["force_max"] = "measured"
    d["measured_by"] = "Tarek"
    d["measured_on"] = "2026-09-05"
    assert Envelope.from_dict(d).measured_by == "Tarek"


def test_envelope_rejects_inverted_joint_limits():
    with pytest.raises(ValueError):
        Envelope.ur5e_declared().replace(q_min=np.full(6, 10.0))


def test_box_contains_and_excursion():
    b = Box(lo=[-1.0, -1.0, 0.0], hi=[1.0, 1.0, 2.0])
    assert b.contains([0.0, 0.0, 1.0])
    assert not b.contains([0.0, 0.0, -0.1])
    assert b.excursion([0.0, 0.0, 1.0]) == 0.0
    assert b.excursion([0.0, 0.0, -0.25]) == pytest.approx(0.25)


def test_violation_is_readable():
    v = Violation(rule="qd_max", measured=2.0, limit=1.0, index=3)
    assert "qd_max" in str(v) and "j3" in str(v)


def test_envelope_rejects_joint_speed_above_datasheet():
    with pytest.raises(ValueError, match="joint speed exceeds the UR5e datasheet"):
        Envelope.ur5e_declared().replace(qd_max=np.full(6, 10.0))


def test_envelope_rejects_joint_range_above_datasheet():
    with pytest.raises(ValueError, match="joint range exceeds the UR5e datasheet"):
        Envelope.ur5e_declared().replace(q_max=np.full(6, 10.0))


def test_envelope_rejects_tcp_speed_above_datasheet():
    with pytest.raises(ValueError, match="TCP speed exceeds the UR5e datasheet"):
        Envelope.ur5e_declared().replace(tcp_speed_max=1.5)


def test_envelope_brake_headroom_validation():
    with pytest.raises(ValueError, match="brake_headroom must be in"):
        Envelope.ur5e_declared().replace(brake_headroom=0.0)
    with pytest.raises(ValueError, match="brake_headroom must be in"):
        Envelope.ur5e_declared().replace(brake_headroom=1.5)
    # Valid values should not raise
    Envelope.ur5e_declared().replace(brake_headroom=0.5)
    Envelope.ur5e_declared().replace(brake_headroom=1.0)


def test_mutating_input_provenance_dict_does_not_change_envelope():
    prov = {"q_min": "declared", "q_max": "declared"}
    env = Envelope.ur5e_declared().replace(provenance=prov)
    prov["q_min"] = "measured"  # Mutate the input dict
    assert env.provenance["q_min"] == "declared"  # Envelope should not be affected


def test_mutating_input_arrays_does_not_change_stored_arrays():
    arr = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    env = Envelope.ur5e_declared().replace(q_min=arr)
    arr[0] = 999.0  # Mutate the input
    assert env.q_min[0] == -1.0
    assert not np.shares_memory(env.q_min, arr)


def test_box_input_arrays_do_not_alias():
    lo_arr = np.array([0.0, 0.0, 0.0])
    hi_arr = np.array([1.0, 1.0, 1.0])
    b = Box(lo=lo_arr, hi=hi_arr)
    lo_arr[0] = 999.0
    hi_arr[0] = 999.0
    assert b.lo[0] != 999.0
    assert b.hi[0] != 999.0
    assert not np.shares_memory(b.lo, lo_arr)
    assert not np.shares_memory(b.hi, hi_arr)


def test_json_roundtrip_includes_brake_headroom():
    a = Envelope.ur5e_declared()
    b = Envelope.from_json(a.to_json())
    assert b.brake_headroom == a.brake_headroom
    assert b.brake_headroom == 0.8


def test_provenance_coverage_is_complete():
    env = Envelope.ur5e_declared()
    # Every field except measured_by, measured_on, provenance itself should have a provenance entry
    expected_fields = {
        "q_min", "q_max", "qd_max", "qdd_max", "qddd_max", "tcp_box",
        "tcp_speed_max", "force_max", "torque_max", "grip_min", "grip_max",
        "grip_rate_max", "watchdog_s", "min_dt_s", "max_dt_s",
        "stale_escalate_n", "tracking_tol_rad", "path_budget_m",
        "contact_time_budget_s", "contact_force_threshold_n",
        "plant_margin_rad", "plant_margin_m", "bisect_iters", "brake_headroom"
    }
    assert set(env.provenance.keys()) == expected_fields
