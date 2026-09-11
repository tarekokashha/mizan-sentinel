"""Append-only, hash-chained verdict log.

Each record commits to the one before it, so a log cannot be edited after the
fact without the edit showing. Runs cite the envelope hash on every line, so
a report can never be attributed to the wrong declaration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sentinel.envelope import Envelope

GENESIS = "0" * 64


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)


def _digest(prev: str, envelope_sha: str, payload) -> str:
    return hashlib.sha256((prev + envelope_sha + _canon(payload)).encode("utf-8")).hexdigest()


class Journal:
    """Write records; each returns the new head hash."""

    def __init__(self, path, envelope: Envelope):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # final-fix-1.md (I4): opening in "a" mode used to silently restart
        # the chain (self._prev = GENESIS, self._seq = 0) over whatever a
        # prior run had already written to this path, so running the
        # documented workflow twice against the same path produced a file
        # that fails the verification the README cites -- two chains, both
        # starting at seq 0, concatenated in one file. Refuse rather than
        # guess at how to resume a chain this instance did not write.
        if self.path.exists() and self.path.stat().st_size > 0:
            raise FileExistsError(
                f"{self.path} already contains journal records; appending here "
                "would restart the hash chain from genesis over an existing "
                "tail, and the result fails verification instead of extending "
                "it. Use a fresh path for each run.")
        self.envelope_sha = envelope.sha256()
        self._prev = GENESIS
        self._seq = 0
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")

    def append(self, payload: dict, envelope_sha: str | None = None) -> str:
        """Append one record. `envelope_sha` overrides the journal's own hash
        for this record only -- final-fix-1.md (I1): a run whose per-episode
        envelope differs from the base declaration (an override) must cite
        the hash of the envelope that actually produced the row, not the
        declaration's hash, or the row is misattributed."""
        if self._fh.closed:
            raise RuntimeError("cannot append to a journal that has been closed")
        sha = self.envelope_sha if envelope_sha is None else envelope_sha
        h = _digest(self._prev, sha, payload)
        rec = {
            "seq": self._seq,
            "prev": self._prev,
            "envelope_sha": sha,
            "payload": payload,
            "hash": h,
        }
        self._fh.write(_canon(rec) + "\n")
        self._prev, self._seq = h, self._seq + 1
        return h

    def close(self) -> None:
        if self._fh.closed:
            return
        # final-fix-1.md (I4): commit to the chain's length. Writing 5
        # records and deleting the last 2 used to verify clean -- (True, -1)
        # -- because nothing in the format said how many records there
        # should be, and dropping trailing records is the natural way to
        # hide an escape from a hash-chained log. A terminal record, checked
        # by verify() to be the last one in the chain, makes a truncated
        # tail internally inconsistent instead of silently valid.
        self.append({"_final": True, "total": self._seq})
        self._fh.flush()
        self._fh.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _is_terminal(payload) -> bool:
    return isinstance(payload, dict) and payload.get("_final") is True


def verify(path) -> tuple[bool, int]:
    """Check the chain. Returns (ok, index of the first bad line or -1).

    A verified journal must end with the terminal record Journal.close()
    writes. Without that requirement, deleting trailing records leaves a
    chain that is internally consistent but silently shorter -- exactly how
    an escape would be hidden from a hash-chained log -- and would still
    verify clean.
    """
    prev, seq = GENESIS, 0
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    last_payload = None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if rec["seq"] != seq or rec["prev"] != prev:
                return False, i
            if rec["hash"] != _digest(prev, rec["envelope_sha"], rec["payload"]):
                return False, i
        except Exception:
            # This is the tamper-evidence entry point; a crafted or
            # structurally malformed line is exactly the kind of tampering
            # it exists to catch, not an exception for the caller to handle.
            return False, i
        prev, seq = rec["hash"], seq + 1
        last_payload = rec["payload"]
    if not _is_terminal(last_payload):
        return False, len(lines)
    return True, -1
