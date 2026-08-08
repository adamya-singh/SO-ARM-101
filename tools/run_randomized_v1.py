"""Randomized-data engine v1: train on a randomized capture, evaluate on a
held-out randomized suite — the project's first generalization measurement.

Exploratory lane. Trains BOTH policy families on the same randomized capture:
- the state policy (privileged features, frozen recipe + cosine, w512/90k);
- the vision policy (pixels + proprioception + clock, minibatched);
  --skip-state trains/evaluates only vision (compute probes).
Both are evaluated WITH the validated gripper clamp on a held-out generated
suite (different generator seed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def _summarize(evaluation_report: Path, label: str) -> dict:
    report = json.loads(evaluation_report.read_text(encoding="utf-8"))
    rollouts = report["rollouts"]
    successes = sum(1 for item in rollouts if item["success"] and not item["invalidated"])
    frames = sum(
        int(item[name]) for item in rollouts
        for name in ("clipping_frames", "limiting_frames", "nonfinite_frames", "unsafe_contact_frames")
    )
    categories = Counter(item["failure_category"] for item in rollouts)
    print(f"SUMMARY {label} successes={successes}/{len(rollouts)} "
          f"safety_frames={frames} categories={dict(categories)}")
    return {
        "successes": successes,
        "rollouts": len(rollouts),
        "safety_frames": frames,
        "categories": dict(categories),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--heldout-suite-path", type=Path, required=True)
    parser.add_argument("--mujoco-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--vision-max-steps", type=int, default=20_000)
    parser.add_argument(
        "--skip-state", action="store_true",
        help="train/evaluate only the vision policy (compute probes)",
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None,
        help="record loss curves + eval summaries to this wandb project "
        "(observer-only; digests unaffected)",
    )
    args = parser.parse_args(argv)

    from so_arm101_v2.data._serialization import content_sha256
    from so_arm101_v2.learning.chunked import ChunkedCloneConfig, train_chunked_clone
    from so_arm101_v2.learning.vision import VisionChunkedConfig, train_vision_chunked
    from so_arm101_v2.simulation.chunked import ChunkedClonePolicy
    from so_arm101_v2.simulation.policy_specs import PolicySpec
    from so_arm101_v2.simulation.rollout import evaluate_closed_loop
    from so_arm101_v2.simulation.scaling import FROZEN_RECIPE
    from so_arm101_v2.simulation.suites import load_suite_from_path
    from so_arm101_v2.simulation.vision_policy import VisionChunkedPolicy

    heldout = load_suite_from_path(args.heldout_suite_path)
    print(f"held-out suite: {heldout.suite_id} "
          f"({len(heldout.scenarios)} scenarios x {heldout.repeats})")

    # Optional experiment tracking: strictly observational (loss curve + eval
    # summaries); never touches identities or digests.
    wandb_run = None
    if args.wandb_project is not None:
        import os

        import wandb

        capture_body = json.loads(args.capture_manifest.read_text(encoding="utf-8"))
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=os.environ.get("WANDB_NAME"),
            config={
                "seed": args.seed,
                "vision_max_steps": args.vision_max_steps,
                "skip_state": bool(args.skip_state),
                "capture_manifest": str(args.capture_manifest),
                "collection_digest": capture_body.get("collection_digest"),
                "capture_rows": capture_body.get("arrays", {}).get("rows"),
                "heldout_suite_id": heldout.suite_id,
            },
        )
        print(f"WANDB_URL {wandb_run.url}", flush=True)

    # State policy on randomized data (established recipe + clamp at eval).
    state_result = None
    if not args.skip_state:
        state_result = train_chunked_clone(
            args.capture_manifest, args.output_dir,
            config=ChunkedCloneConfig(**{
                **FROZEN_RECIPE,
                "hidden_width": 512,
                "max_steps": 90_000,
                "lr_schedule": "cosine_floor_v1",
                "seed": args.seed,
            }),
        )
        print(f"STATE TRAINED {state_result.directory} mse={state_result.normalized_mse:.3e}")

    vision_result = train_vision_chunked(
        args.capture_manifest, args.output_dir,
        config=VisionChunkedConfig(seed=args.seed, max_steps=args.vision_max_steps),
        on_loss=(
            (lambda step, mse: wandb_run.log({"train/batch_normalized_mse": mse}, step=step))
            if wandb_run is not None else None
        ),
    )
    print(f"VISION TRAINED {vision_result.directory} mse={vision_result.normalized_mse:.3e}")
    if wandb_run is not None:
        wandb_run.summary["vision/normalized_mse"] = vision_result.normalized_mse
        wandb_run.summary["vision/model_dir"] = str(vision_result.directory)

    evaluations = []
    if state_result is not None:
        evaluations.append((
            "state+clamp",
            ChunkedClonePolicy(state_result.checkpoint, clamp_channels=(5,)),
            PolicySpec(
                kind="chunked_clone",
                checkpoint=str(state_result.checkpoint.resolve()),
                options=(("clamp_channels", (5,)),),
            ),
        ))
    evaluations.append((
        "vision+clamp",
        VisionChunkedPolicy(vision_result.checkpoint, clamp_channels=(5,)),
        PolicySpec(
            kind="vision_chunked",
            checkpoint=str(vision_result.checkpoint.resolve()),
            options=(("clamp_channels", (5,)),),
        ),
    ))
    for label, probe, spec in evaluations:
        checkpoint_sha = hashlib.sha256(Path(spec.checkpoint).read_bytes()).hexdigest()
        evaluation_identity = content_sha256({
            "exploration": "randomized_v1",
            "label": label,
            "checkpoint_sha256": checkpoint_sha,
            "suite_content_sha256": json.loads(
                args.heldout_suite_path.read_text(encoding="utf-8")
            )["content_sha256"],
        })
        evaluation = evaluate_closed_loop(
            args.mujoco_model,
            heldout,
            {probe.policy_id: spec},
            args.output_dir / "randomized_explorations" / evaluation_identity[:16],
            environment_proven=True,
            record_video=True,
            provenance={
                "exploration": "randomized_v1",
                "label": label,
                "checkpoint_sha256": checkpoint_sha,
            },
        )
        metrics = _summarize(evaluation.report_json, label)
        if wandb_run is not None:
            slug = label.replace("+", "_")
            wandb_run.summary[f"eval/{slug}/successes"] = metrics["successes"]
            wandb_run.summary[f"eval/{slug}/rollouts"] = metrics["rollouts"]
            wandb_run.summary[f"eval/{slug}/safety_frames"] = metrics["safety_frames"]
            wandb_run.summary[f"eval/{slug}/report"] = str(evaluation.report_json)
            for category, count in metrics["categories"].items():
                wandb_run.summary[f"eval/{slug}/category/{category}"] = count
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
