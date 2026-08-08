"""Generate a randomized cube-pose suite and write it content-addressed.

Usage:
    PYTHONNOUSERSITE=1 python tools/generate_random_suite.py \
        --generator-seed 7 --count 25 --repeats 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--x-range", type=float, nargs=2, default=(-0.06, 0.06))
    parser.add_argument("--y-range", type=float, nargs=2, default=(0.24, 0.31))
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("artifacts/so_arm101_v2/suites"),
    )
    parser.add_argument(
        "--screen-model", type=Path, default=None,
        help="MuJoCo scene: admit only poses the privileged teacher executes to "
        "success with zero safety events (full-episode screen)",
    )
    args = parser.parse_args(argv)

    from so_arm101_v2.data._serialization import content_sha256, write_immutable_json
    from so_arm101_v2.simulation.suites import generate_randomized_suite, suite_payload

    suite = generate_randomized_suite(
        args.generator_seed, args.count,
        x_range=tuple(args.x_range), y_range=tuple(args.y_range),
        repeats=args.repeats,
        screen_with_model=str(args.screen_model) if args.screen_model else None,
    )
    payload = suite_payload(suite)
    payload["generator"] = {
        "generator_seed": args.generator_seed,
        "count": args.count,
        "repeats": args.repeats,
        "x_range": list(args.x_range),
        "y_range": list(args.y_range),
        "exclude_napkin": True,
        "screened": "full_episode_teacher_v1" if args.screen_model else None,
    }
    payload["content_sha256"] = content_sha256(payload)
    digest = content_sha256(payload)[:16]
    destination = args.output_root / digest / "suite.json"
    write_immutable_json(destination, payload)
    print(destination)
    print(f"suite_id={suite.suite_id} scenarios={len(suite.scenarios)} repeats={suite.repeats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
