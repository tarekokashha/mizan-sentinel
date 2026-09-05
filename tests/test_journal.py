import json

from sentinel.envelope import Envelope
from sentinel.journal import Journal, verify


def _write(path, n=5):
    env = Envelope.ur5e_declared()
    with Journal(path, env) as j:
        for i in range(n):
            j.append({"step": i, "status": "PASS"})
    return env


def test_a_fresh_journal_verifies(tmp_path):
    p = tmp_path / "run.jsonl"
    _write(p)
    assert verify(p) == (True, -1)


def test_every_line_carries_the_envelope_hash(tmp_path):
    p = tmp_path / "run.jsonl"
    env = _write(p)
    for line in p.read_text().splitlines():
        assert json.loads(line)["envelope_sha"] == env.sha256()


def test_tampering_with_a_payload_is_detected(tmp_path):
    p = tmp_path / "run.jsonl"
    _write(p)
    lines = p.read_text().splitlines()
    rec = json.loads(lines[2])
    rec["payload"]["status"] = "STOP"
    lines[2] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")
    ok, bad = verify(p)
    assert ok is False and bad == 2


def test_deleting_a_line_is_detected(tmp_path):
    p = tmp_path / "run.jsonl"
    _write(p)
    lines = p.read_text().splitlines()
    del lines[1]
    p.write_text("\n".join(lines) + "\n")
    assert verify(p)[0] is False


def test_rewriting_the_envelope_hash_is_detected(tmp_path):
    # The envelope hash is the whole point of the chain: it is what ties a
    # report to the declaration that was in force. Leaving it outside the
    # digest means a run can be silently re-attributed to any envelope.
    p = tmp_path / "run.jsonl"
    _write(p)
    lines = p.read_text().splitlines()
    rec = json.loads(lines[2])
    rec["envelope_sha"] = "0" * 64
    lines[2] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")
    ok, bad = verify(p)
    assert ok is False and bad == 2
