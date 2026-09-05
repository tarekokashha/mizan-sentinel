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
