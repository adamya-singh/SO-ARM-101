"""Would the episode guards have stopped a recorded physical episode? Logs only; nothing touches the arm.

Feeds each row of an episode's steps.csv (measured joints, the command that was sent) and its stored boundary
observations through ``so_arm101_v2.physical.guards.EpisodeGuards`` exactly as the runner would, and reports the
first trip. Use it to check a change to the limits against every episode on record: the successes must pass
and physical/episode_20_20261004_pi05 must be stopped before it presses on the table.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so_arm101_v2.contracts.bench import scene_bench_config  # noqa: E402
from so_arm101_v2.contracts.coordinates import JOINT_NAMES  # noqa: E402
from so_arm101_v2.physical.guards import EpisodeGuards  # noqa: E402

DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


def replay(episode_dir: Path, guards: EpisodeGuards) -> dict:
    from PIL import Image
    rows = list(csv.DictReader(open(episode_dir / "steps.csv")))
    guards.reset()
    for row in rows:
        step = int(row["step"])
        measured = np.array([float(row[f"measured_act_{j}"]) for j in JOINT_NAMES], np.float32)
        decision = SimpleNamespace(sent_physical=np.array([float(row[f"sent_physical_{j}"]) for j in JOINT_NAMES], np.float32),
                                   held=bool(row["hold_reason"]))
        observation = None
        png = episode_dir / "boundaries" / f"step_{step:03d}.obs.png"
        if row["boundary"] == "1" and png.is_file():
            observation = SimpleNamespace(image=np.asarray(Image.open(png).convert("RGB")))
        reason = guards.before_send(step, measured, decision, observation)
        if reason is not None:
            break
    return dict(episode=episode_dir.name, steps=len(rows), **guards.record())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode-dir", type=Path, action="append", required=True)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--no-survey-checks", action="store_true", help="episodes of a policy without the survey move (before 2026-09-10)")
    p.add_argument("--no-survey-gate", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)
    bench = scene_bench_config(args.model)
    guards = EpisodeGuards(args.model, bench, survey_gate=not args.no_survey_gate)
    if args.no_survey_checks:
        guards.survey_step = None
    results = []
    for directory in args.episode_dir:
        result = replay(directory, guards)
        results.append(result)
        trip = result["trip"]
        print(f"{result['episode']}: " + ("no trip" if trip is None else f"TRIP at step {trip['step']}: {trip['reason']} {trip.get('detail', '')}")
              + f" (min commanded tip height {result['min_commanded_tip_height_m']} m over {result['steps']} steps)")
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
