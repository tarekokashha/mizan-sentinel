import json

import pytest

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


def test_deleting_the_tail_is_detected(tmp_path):
    # final-fix-1.md (I4): writing 5 records and deleting the last 2 used to
    # verify clean, (True, -1) -- an internally consistent shorter chain is
    # exactly what a truncated tail looks like, and dropping trailing
    # records is the natural way to hide an escape from a hash-chained log.
    # Journal.close() now writes a terminal record that verify() requires as
    # the chain's last line, so a truncated tail is no longer silently valid.
    p = tmp_path / "run.jsonl"
    _write(p, n=5)
    lines = p.read_text().splitlines()
    assert len(lines) == 6, "5 data records plus the terminal marker"
    p.write_text("\n".join(lines[:-2]) + "\n")   # drop the last 2 lines
    ok, bad = verify(p)
    assert ok is False


def test_a_journal_missing_its_terminal_record_does_not_verify(tmp_path):
    # The terminal record is what commits to the chain's length. A file that
    # is otherwise a perfectly well-formed, internally consistent chain but
    # simply never got its closing record (Journal.close() never ran, or the
    # one line it wrote was the only thing removed) must not verify.
    p = tmp_path / "run.jsonl"
    _write(p, n=3)
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:-1]) + "\n")   # drop only the terminal record
    ok, bad = verify(p)
    assert ok is False


def test_reopening_an_existing_journal_file_refuses_to_append(tmp_path):
    # final-fix-1.md (I4): Journal.__init__ used to open "a" and silently
    # restart the chain (prev=GENESIS, seq=0) over whatever was already
    # there, so running the documented workflow twice against the same path
    # produced a journal that fails verification -- two chains concatenated
    # in one file. Refuse loudly instead.
    p = tmp_path / "run.jsonl"
    _write(p, n=2)
    with pytest.raises(FileExistsError):
        Journal(p, Envelope.ur5e_declared())


def test_a_structurally_malformed_line_fails_verification_without_crashing(tmp_path):
    # journal.verify() is the tamper-evidence entry point; a crafted or
    # corrupted line is exactly the kind of tampering it exists to catch,
    # not an exception for the caller to handle.
    p = tmp_path / "run.jsonl"
    _write(p, n=2)
    lines = p.read_text().splitlines()
    lines[1] = "{not valid json"
    p.write_text("\n".join(lines) + "\n")
    ok, bad = verify(p)
    assert ok is False and bad == 1


def test_a_line_missing_a_required_field_fails_verification_without_crashing(tmp_path):
    p = tmp_path / "run.jsonl"
    _write(p, n=2)
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1])
    del rec["hash"]
    lines[1] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")
    ok, bad = verify(p)
    assert ok is False and bad == 1
