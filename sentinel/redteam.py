"""Run the attack catalogue against the kernel and report what escaped.

An escape is the falsification of the claim in the spec. Nothing here rounds,
smooths, or excuses one: it is any of a non-finite command, a commanded joint
outside [q_min, q_max], a plant joint outside those limits inflated by
plant_margin_rad, or a plant TCP outside the box inflated by plant_margin_m.

This module supersedes the Task 11 brief in five places, all measured defects
found while implementing it (task-11-correction.md):

1. watchdog_starve is ticked by DT + spec.skip_seconds on marked steps, not
   DT * (1 + spec.skip_steps) -- the brief's skip_steps=1 could never exceed
   max_dt_s.
2. The episode start state (EPISODE_Q0) is verified against the envelope
   before step 0, not assumed. A bad start would otherwise report an escape
   for every attack and mean nothing.
3. An attack may declare its own start_q (lever_sprint does: its joint-space
   ramp from EPISODE_Q0 is legal at every step but not Cartesian-safe). The
   same verification runs against whichever start was chosen.
4. An attack may declare an envelope_override applied to its own episode only
   (slow_drift does: path_budget_m=25.0 cannot be reached within one 2000
   step / 66.67 s episode at tcp_speed_max=0.25 m/s, so it is demonstrated
   against a declared override instead). The override is never silent: it is
   printed in the table and written to the CSV.
5. The declared budget (200 episodes x 2000 steps x 15 attacks) is expensive
   and is not shrunk because of that. Progress prints per attack, and
   wall-clock is on the record in both the table and the CSV.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sentinel.attacks import REGISTRY, AttackSpec, EPISODE_Q0
from sentinel.envelope import Envelope
from sentinel.journal import Journal
from sentinel.kernel import SafetyKernel
from sentinel.kinematics import tcp_position
from sentinel.sim import PlaneContact, SimPlant
from sentinel.types import Status

__all__ = [
    "EPISODE_Q0", "EpisodeResult", "AttackReport", "run_episode",
    "run_attack", "run_all", "main",
]

DT = 1.0 / 30.0
EPISODES = 200          # declared in the spec before the first run
STEPS = 2000            # 200 * 2000 * 15 = 6,000,000 filter() calls; see module docstring

# CRITICAL 2 / final-fix-1.md: the plant's per-episode start-state jitter and
# sensor noise, small enough to stay well inside every declared joint and
# Cartesian margin -- lever_sprint's declared start keeps the least Cartesian
# margin of any attack in the catalogue, about 19 mm from the box wall, and
# these magnitudes move the flange by tenths of a millimetre, not that. Both
# are re-verified against the envelope for the actual jittered start below,
# not just asserted here. This is what makes SimPlant's per-episode seed
# (already threaded through as seed * 100_003 + e in run_attack) produce a
# genuinely different episode instead of a bit-identical replay of the same
# nominal start.
Q0_JITTER_RAD = 0.002          # ~0.11 deg per joint, 1 std
SENSOR_NOISE_STD_RAD = 2e-4    # far under tracking_tol_rad (0.25 rad)

# cairo_protocol is another repository's code (E:\Robotics Projects\mizan-kit)
# and is deliberately not on this path: not vendored, not added to sys.path.
# Resolved once at import time so the fallback is a fact checked at load time,
# not a guess re-made on every call.
try:
    from cairo_protocol.stats import anytime_cs as _ANYTIME_CS
except Exception:
    _ANYTIME_CS = None


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float = DT) -> None:
        self.t += dt


@dataclass
class EpisodeResult:
    attack: str
    escaped: bool
    first_escape_step: int
    worst_joint_excursion: float
    worst_tcp_excursion: float
    rules_fired: set = field(default_factory=set)
    # CRITICAL 2 / final-fix-1.md: a content hash of the whole episode's
    # commanded-q and fired-violation trace, so a test can prove two episodes
    # actually differ rather than trusting that they should. See
    # test_episode_traces_vary_across_repetitions_under_the_same_seed.
    trace_sha256: str = ""


@dataclass
class AttackReport:
    attack: str
    expect: str
    episodes: int
    escapes: int
    escape_rate: float
    cs_lo: float
    cs_hi: float
    rules_fired: tuple
    expected_guard_fired: bool
    worst_joint_excursion: float
    worst_tcp_excursion: float
    envelope_override: dict | None = None
    wall_clock_s: float = 0.0
    # I1 / final-fix-1.md: the envelope actually in force for this attack's
    # episodes, which differs from the base declaration exactly when
    # envelope_override is set (slow_drift). The CSV and journal used to cite
    # env.sha256() (the base declaration) on every row, misattributing the
    # one row where it mattered.
    envelope_sha256: str = ""


def _confidence_sequence(outcomes, alpha: float = 0.05) -> tuple[float, float]:
    """Anytime-valid interval from the kit if it is importable, else report raw.

    Degrading loudly is deliberate. Substituting a different interval and
    calling it the same thing would make two programmes' numbers incomparable.
    """
    if _ANYTIME_CS is None:
        p = float(np.mean(outcomes)) if len(outcomes) else 0.0
        return p, p
    return _ANYTIME_CS(outcomes, alpha)


def _fmt_override(override: dict | None) -> str:
    if not override:
        return "-"
    return ",".join(f"{k}={v}" for k, v in sorted(override.items()))


def run_episode(spec: AttackSpec, env: Envelope, seed: int, steps: int) -> EpisodeResult:
    # envelope_override (task-10-fix-3.md / task-11-correction.md #4) applies
    # to this attack's episode only. Everything below -- the start-state
    # check, the kernel, the escape determination -- runs against ep_env, the
    # envelope actually in force for this episode, not the base declaration.
    ep_env = env if spec.envelope_override is None else env.replace(**spec.envelope_override)

    # start_q (task-11-correction.md #3): an attack may declare where its
    # episode should start (lever_sprint does). A badly declared start_q must
    # fail the same check EPISODE_Q0 itself is held to, loudly, at step 0 --
    # not quietly, and not by reporting a false escape for every attack.
    q0 = EPISODE_Q0 if spec.start_q is None else np.asarray(spec.start_q, dtype=float)
    if np.any(q0 > ep_env.q_max) or np.any(q0 < ep_env.q_min):
        raise ValueError(f"{spec.name}: episode start {q0} is outside the joint limits")
    p0 = tcp_position(q0)
    if ep_env.tcp_box.excursion(p0) > 0.0:
        raise ValueError(f"{spec.name}: episode start puts the flange outside the box: {p0}")

    rng = np.random.default_rng(seed)
    fn = spec.build(ep_env, rng)
    clock = _Clock()
    kernel = SafetyKernel(ep_env, clock=clock)
    contact = PlaneContact(z=0.10, stiffness=8000.0)
    plant = SimPlant(q0=q0, qdd_max=ep_env.qdd_max, contact=contact, seed=seed,
                     q0_jitter_rad=Q0_JITTER_RAD, sensor_noise_std=SENSOR_NOISE_STD_RAD)

    # CRITICAL 2 / final-fix-1.md: SimPlant jitters its own start from this
    # seed, so the q0 checked above is the nominal start, not the one the
    # episode actually runs from. Re-verify the real, jittered start against
    # the same envelope this attack is held to -- not just trust the jitter
    # magnitude declared as a constant above -- so a badly-declared start_q
    # or an over-large jitter fails loudly at step 0, the same contract
    # EPISODE_Q0 and start_q are already held to.
    if np.any(plant.q > ep_env.q_max) or np.any(plant.q < ep_env.q_min):
        raise ValueError(f"{spec.name}: jittered episode start {plant.q} is outside "
                         "the joint limits")
    p0_jittered = tcp_position(plant.q)
    if ep_env.tcp_box.excursion(p0_jittered) > 0.0:
        raise ValueError(f"{spec.name}: jittered episode start puts the flange "
                         f"outside the box: {p0_jittered}")

    escaped, first, worst_j, worst_p = False, -1, 0.0, 0.0
    rules: set = set()
    frozen_stamp = 0.0
    trace = hashlib.sha256()

    for step in range(steps):
        stamp = frozen_stamp if spec.freeze_state else clock.t
        if spec.spoof_stamp:
            stamp = clock.t          # honest-looking, while the action lies
        state = plant.state(t_mono=stamp)
        action = fn(state, step, clock.t)
        verdict = kernel.filter(state, action)
        rules.update(v.rule for v in verdict.violations)

        # An escape is any of: a non-finite command; a commanded joint
        # outside [q_min, q_max]; a plant joint outside those limits inflated
        # by plant_margin_rad; a plant TCP outside the box inflated by
        # plant_margin_m. Nothing here rounds, smooths, or excuses one.
        q_cmd = verdict.action.q
        if q_cmd is None:
            # CRITICAL 1 / final-fix-1.md: the kernel refuses to fabricate a
            # pose when no trusted state exists. No attack in this catalogue
            # corrupts an episode's very first observation, so this is not
            # currently reachable from run_all -- but "no motion commanded"
            # is unambiguously not an escape, and must not crash the loop
            # that judges one.
            q_cmd = plant.q.copy()
        trace.update(np.asarray(q_cmd, dtype=float).tobytes())
        trace.update(",".join(sorted(v.rule for v in verdict.violations)).encode("utf-8"))
        if not np.all(np.isfinite(q_cmd)):
            escaped, first = True, step if first < 0 else first
        j_exc = float(max(np.max(q_cmd - ep_env.q_max), np.max(ep_env.q_min - q_cmd), 0.0))
        worst_j = max(worst_j, j_exc)
        if j_exc > 1e-9:
            escaped, first = True, step if first < 0 else first

        plant.step(q_cmd, DT)
        pj = float(max(np.max(plant.q - ep_env.q_max), np.max(ep_env.q_min - plant.q), 0.0))
        worst_j = max(worst_j, pj)
        if pj > ep_env.plant_margin_rad:
            escaped, first = True, step if first < 0 else first
        p_exc = ep_env.tcp_box.excursion(tcp_position(plant.q))
        worst_p = max(worst_p, p_exc)
        if p_exc > ep_env.plant_margin_m:
            escaped, first = True, step if first < 0 else first

        # watchdog_starve (task-11-correction.md #1): tick DT + skip_seconds
        # on marked steps, not DT * (1 + skip_steps). skip_steps=1 could only
        # ever produce 2 * (1/30) = 0.0667 s, well below max_dt_s = 0.2 s, so
        # the dt_max guard it declares could never fire.
        extra = spec.skip_seconds if (spec.skip_seconds and step % 20 == 19) else 0.0
        clock.tick(DT + extra)
        if verdict.status is Status.STOP:
            kernel.rearm("red-team episode continues after a trip")

    return EpisodeResult(spec.name, escaped, first, worst_j, worst_p, rules,
                         trace.hexdigest())


def run_attack(name: str, env: Envelope, episodes: int = EPISODES,
               steps: int = STEPS, seed: int = 0) -> AttackReport:
    spec = REGISTRY[name]
    # I1 / final-fix-1.md: the envelope actually in force for every episode
    # of this attack -- same rule run_episode uses -- so the report can cite
    # ep_env.sha256() rather than the base declaration's hash.
    ep_env = env if spec.envelope_override is None else env.replace(**spec.envelope_override)
    t0 = time.perf_counter()
    outcomes, rules = [], set()
    wj = wp = 0.0
    for e in range(episodes):
        r = run_episode(spec, env, seed=seed * 100_003 + e, steps=steps)
        outcomes.append(1 if r.escaped else 0)
        rules |= r.rules_fired
        wj, wp = max(wj, r.worst_joint_excursion), max(wp, r.worst_tcp_excursion)
    elapsed = time.perf_counter() - t0
    lo, hi = _confidence_sequence(outcomes)
    return AttackReport(
        attack=name, expect=spec.expect, episodes=episodes, escapes=int(sum(outcomes)),
        escape_rate=float(np.mean(outcomes)), cs_lo=lo, cs_hi=hi,
        rules_fired=tuple(sorted(rules)), expected_guard_fired=spec.expect in rules,
        worst_joint_excursion=wj, worst_tcp_excursion=wp,
        envelope_override=spec.envelope_override, wall_clock_s=elapsed,
        envelope_sha256=ep_env.sha256(),
    )


def run_all(env: Envelope, episodes: int = EPISODES, steps: int = STEPS,
            seed: int = 0, journal_path=None) -> list[AttackReport]:
    names = sorted(REGISTRY)
    reports: list[AttackReport] = []
    t_all0 = time.perf_counter()
    # A silent thirty-plus-minute run is indistinguishable from a hang
    # (task-11-correction.md #5): one progress line per attack as it
    # completes, with its escape count and elapsed time.
    for i, name in enumerate(names, start=1):
        r = run_attack(name, env, episodes, steps, seed)
        reports.append(r)
        override = f" override={_fmt_override(r.envelope_override)}" if r.envelope_override else ""
        print(f"[{i}/{len(names)}] {name}: {r.escapes}/{r.episodes} escapes "
              f"in {r.wall_clock_s:.2f}s{override}", flush=True)
    print(f"red-team run complete: {len(names)} attacks in "
          f"{time.perf_counter() - t_all0:.1f}s total", flush=True)

    if journal_path is not None:
        with Journal(journal_path, env) as j:
            for r in reports:
                # I1 / final-fix-1.md: cite the envelope that actually
                # produced this row (ep_env, carried on the report as
                # envelope_sha256), not the base declaration's hash. They
                # differ for exactly one row today -- slow_drift, which
                # declares envelope_override -- and PROTOCOL.md's claim that
                # every row carries the hash of the envelope that produced it
                # was false for that row before this fix.
                j.append({
                    "attack": r.attack, "expect": r.expect, "episodes": r.episodes,
                    "escapes": r.escapes, "escape_rate": r.escape_rate,
                    "cs_lo": r.cs_lo, "cs_hi": r.cs_hi,
                    "rules_fired": list(r.rules_fired),
                    "expected_guard_fired": r.expected_guard_fired,
                    "worst_joint_excursion": r.worst_joint_excursion,
                    "worst_tcp_excursion": r.worst_tcp_excursion,
                    "envelope_override": r.envelope_override,
                    "wall_clock_s": r.wall_clock_s,
                }, envelope_sha=r.envelope_sha256)
    return reports


def _print(reports: list[AttackReport], env: Envelope) -> None:
    print(f"envelope {env.name} sha256={env.sha256()[:12]}")
    if _ANYTIME_CS is None:
        print("WARNING: cairo_protocol is not importable on this path (it lives in "
              "another repository and is deliberately not vendored or added to "
              "sys.path -- task-11-correction.md #6). cs_lo/cs_hi below are RAW "
              "escape rates (cs_lo == escape_rate == cs_hi), NOT anytime-valid "
              "confidence sequences.")
    head = (f"{'attack':18} {'expect':12} {'escapes':>10} {'rate':>6} "
            f"{'cs_lo':>6} {'cs_hi':>6} {'fired':>6} {'worst_q':>9} {'worst_p':>9} "
            f"{'override':18} {'time_s':>8}")
    print(head)
    print("-" * len(head))
    for r in reports:
        esc = f"{r.escapes}/{r.episodes}"
        print(f"{r.attack:18} {r.expect:12} {esc:>10} {r.escape_rate:6.3f} "
              f"{r.cs_lo:6.3f} {r.cs_hi:6.3f} {str(r.expected_guard_fired):>6} "
              f"{r.worst_joint_excursion:9.4f} {r.worst_tcp_excursion:9.4f} "
              f"{_fmt_override(r.envelope_override):18} {r.wall_clock_s:8.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the SENTINEL red-team suite.")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="5 episodes of 300 steps")
    ap.add_argument("--attack", default=None, help="run one attack by name")
    ap.add_argument("--out", default=None, help="write a CSV here")
    ap.add_argument("--journal", default=None, help="write a hash-chained log here")
    args = ap.parse_args(argv)
    if args.quick:
        args.episodes, args.steps = 5, 300

    env = Envelope.ur5e_declared()
    n_attacks = 1 if args.attack else len(REGISTRY)
    print(f"running {n_attacks} attack(s) x {args.episodes} episodes x {args.steps} steps "
          f"= {n_attacks * args.episodes * args.steps:,} filter() calls")

    if args.attack:
        reports = [run_attack(args.attack, env, args.episodes, args.steps, args.seed)]
    else:
        reports = run_all(env, args.episodes, args.steps, args.seed, args.journal)
    _print(reports, env)

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["attack", "expect", "episodes", "escapes", "escape_rate",
                        "cs_lo", "cs_hi", "expected_guard_fired",
                        "worst_joint_excursion", "worst_tcp_excursion",
                        "rules_fired", "envelope_override", "wall_clock_s",
                        "envelope_sha256"])
            for r in reports:
                # I1 / final-fix-1.md: cite the envelope that actually
                # produced this row, not the base declaration -- see the
                # matching comment in run_all's journal write above.
                w.writerow([r.attack, r.expect, r.episodes, r.escapes, r.escape_rate,
                            r.cs_lo, r.cs_hi, r.expected_guard_fired,
                            r.worst_joint_excursion, r.worst_tcp_excursion,
                            "|".join(r.rules_fired),
                            json.dumps(r.envelope_override, sort_keys=True) if r.envelope_override else "",
                            r.wall_clock_s, r.envelope_sha256])
        print(f"wrote {p}")

    escaped = [r.attack for r in reports if r.escape_rate > 0.0]
    silent = [r.attack for r in reports if not r.expected_guard_fired]
    if escaped:
        print(f"FAIL: escaped the envelope: {', '.join(escaped)}")
    if silent:
        print(f"FAIL: expected guard never fired: {', '.join(silent)}")
    print("PASS" if not escaped and not silent else "FAIL")
    return 1 if (escaped or silent) else 0


if __name__ == "__main__":
    sys.exit(main())
