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


def _digest(prev: str, payload) -> str:
    return hashlib.sha256((prev + _canon(payload)).encode("utf-8")).hexdigest()


class Journal:
    """Write records; each returns the new head hash."""

    def __init__(self, path, envelope: Envelope):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.envelope_sha = envelope.sha256()
        self._prev = GENESIS
        self._seq = 0
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")

    def append(self, payload: dict) -> str:
        h = _digest(self._prev, payload)
        rec = {
            "seq": self._seq,
            "prev": self._prev,
            "envelope_sha": self.envelope_sha,
            "payload": payload,
            "hash": h,
        }
        self._fh.write(_canon(rec) + "\n")
        self._prev, self._seq = h, self._seq + 1
        return h

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def verify(path) -> tuple[bool, int]:
    """Check the chain. Returns (ok, index of the first bad line or -1)."""
    prev, seq = GENESIS, 0
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["seq"] != seq or rec["prev"] != prev:
            return False, i
        if rec["hash"] != _digest(prev, rec["payload"]):
            return False, i
        prev, seq = rec["hash"], seq + 1
    return True, -1
