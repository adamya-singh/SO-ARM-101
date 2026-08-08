"""Exploratory vision-v0 driver: train one seed on a frames capture, then
run the 15-rollout Stage A evaluation (optionally black-image ablated).

Exploratory lane (notes/vision-rung-notebook.md): no gate, no promotion
claim — artifacts stay content-addressed and the numerics regime is
fingerprinted, but this driver just trains, evaluates, and prints.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--mujoco-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--black-image", action="store_true",
                        help="also evaluate the trained policy with zeroed frames")
    parser.add_argument("--clamp-gripper", action="store_true", default=True,
                        help="(vision policies do not clamp; reserved)")
    args = parser.parse_args(argv)

    from so_arm101_v2.data._serialization import content_sha256
    from so_arm101_v2.learning.vision import VisionChunkedConfig, train_vision_chunked
    from so_arm101_v2.simulation.policy_specs import PolicySpec
    from so_arm101_v2.simulation.rollout import evaluate_closed_loop
    from so_arm101_v2.simulation.suites import load_simulation_suite
    from so_arm101_v2.simulation.vision_policy import VisionChunkedPolicy
    from dataclasses import replace
    import hashlib

    config = VisionChunkedConfig(
        seed=args.seed, hidden_width=args.hidden_width,
        max_steps=args.max_steps, batch_size=args.batch_size,
    )
    result = train_vision_chunked(args.capture_manifest, args.output_dir, config=config)
    print(f"TRAINED {result.directory} mse={result.normalized_mse:.3e}")

    probe = VisionChunkedPolicy(result.checkpoint, black_image=args.black_image)
    checkpoint_sha = hashlib.sha256(result.checkpoint.read_bytes()).hexdigest()
    suite = load_simulation_suite("fixed_pick_place_v3")
    label = f"vision_v0_seed{args.seed}" + (".black" if args.black_image else "")
    cell_suite = replace(suite, suite_id=f"{suite.suite_id}.{label}")
    evaluation_identity = content_sha256({
        "exploration": "vision_v0",
        "checkpoint_sha256": checkpoint_sha,
        "black_image": bool(args.black_image),
        "suite_id": cell_suite.suite_id,
    })
    options = (("black_image", True),) if args.black_image else ()
    evaluation = evaluate_closed_loop(
        args.mujoco_model,
        cell_suite,
        {probe.policy_id: PolicySpec(
            kind="vision_chunked",
            checkpoint=str(result.checkpoint.resolve()),
            options=options,
        )},
        args.output_dir / "vision_explorations" / evaluation_identity[:16],
        environment_proven=True,
        record_video=True,
        provenance={
            "exploration": "vision_v0",
            "checkpoint_sha256": checkpoint_sha,
            "offline_report_content_sha256": probe.report_content_sha256,
            "black_image": bool(args.black_image),
        },
    )
    report = json.loads(evaluation.report_json.read_text(encoding="utf-8"))
    rollouts = report["rollouts"]
    successes = sum(
        1 for item in rollouts
        if item["success"] and not item["invalidated"]
    )
    frames = sum(
        int(item[name]) for item in rollouts
        for name in ("clipping_frames", "limiting_frames", "nonfinite_frames", "unsafe_contact_frames")
    )
    categories = Counter(item["failure_category"] for item in rollouts)
    print(f"EVAL {evaluation.report_json}")
    print(f"SUMMARY seed={args.seed} black={args.black_image} "
          f"successes={successes}/15 safety_frames={frames} categories={dict(categories)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
