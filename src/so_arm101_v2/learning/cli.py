"""CLI for trivial baselines and the tiny memorization gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from so_arm101_v2.data import build_split_manifests, inventory_physical_dataset

from .baselines import baseline_resource_name, evaluate_baselines
from .decision import build_act_decision_report
from .full_dataset import (
    SmallModelConfig,
    build_lead3_training_cache,
    evaluate_image_ablation,
    load_small_model_comparison,
    train_small_models,
)
from .numerics import resolve_numerics
from .tiny_model import TinyModelConfig, run_tiny_overfit
from .oracle_distillation import (
    OracleDistillationConfig,
    analyze_oracle_residuals,
    distill_oracle_policy,
)


def _numerics_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="training numerics device (auto = pinned GPU regime when CUDA is live)",
    )
    parser.add_argument("--no-compile", action="store_true",
                        help="disable torch.compile in the numerics regime")
    parser.add_argument(
        "--legacy-numerics", action="store_true",
        help="exact legacy CPU-eager training lane; reproduces pre-v2 digests byte-for-byte",
    )


def _resolve_cli_numerics(args: argparse.Namespace):
    if args.legacy_numerics:
        return None
    return resolve_numerics(args.device, compile=not args.no_compile)


def _require_pinned_torch() -> None:
    """Guard parity with the simulation CLI's mujoco pin: the learn lane's
    evidence is pinned to the torch series recorded in pyproject."""
    try:
        import torch
    except ImportError:
        return
    if not torch.__version__.startswith("2.7."):
        raise RuntimeError(
            f"torch {torch.__version__} does not match the pinned 2.7.x series; "
            "run inside the lerobot env with PYTHONNOUSERSITE=1"
        )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("imitation-learning/datasets/so101_pickplace_v1"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="so-arm101-v2-learn")
    commands = parser.add_subparsers(dest="command", required=True)
    baselines = commands.add_parser("baselines")
    _common(baselines)
    baselines.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/so_arm101_v2/baselines"),
    )
    memorize = commands.add_parser("memorize")
    _common(memorize)
    memorize.add_argument("--baseline-report", type=Path, required=True)
    memorize.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/so_arm101_v2/tiny_model"),
    )
    memorize.add_argument("--sample-count", type=int, default=16)
    memorize.add_argument("--max-steps", type=int, default=5000)
    train = commands.add_parser("train-small")
    _common(train)
    train.add_argument("--cache-dir", type=Path, default=Path("artifacts/so_arm101_v2/cache"))
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/small_models"))
    train.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    train.add_argument("--max-epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    compare = commands.add_parser("compare-small")
    _common(compare)
    compare.add_argument("--cache-dir", type=Path, default=Path("artifacts/so_arm101_v2/cache"))
    compare.add_argument("--comparison-report", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/small_models"))
    decide = commands.add_parser("decide-act")
    _common(decide)
    decide.add_argument("--cache-dir", type=Path, default=Path("artifacts/so_arm101_v2/cache"))
    decide.add_argument("--comparison-report", type=Path, required=True)
    decide.add_argument("--image-ablation-report", type=Path, required=True)
    decide.add_argument("--preflight-report", type=Path, required=True)
    decide.add_argument("--simulation-report", type=Path, required=True)
    decide.add_argument("--output", type=Path, default=Path("artifacts/so_arm101_v2/act_gate.json"))
    oracle = commands.add_parser("distill-oracle")
    oracle.add_argument("--manifest", type=Path, required=True)
    oracle.add_argument(
        "--model-kind",
        choices=(
            "phase_state", "feedback_state", "phase_dynamics",
            "phase_dynamics_contact", "phase_dynamics_contact_history2",
        ),
        required=True,
    )
    oracle.add_argument("--seed", type=int, default=101)
    oracle.add_argument("--max-steps", type=int, default=10_000)
    oracle.add_argument("--hidden-width", type=int, choices=(128, 256), default=128)
    oracle.add_argument(
        "--training-rows",
        choices=(
            "full", "memorization32", "memorization64", "memorization128",
            "memorization256",
        ),
        default="full",
    )
    oracle.add_argument(
        "--lr-schedule", choices=("fixed", "decay_10k_20k"), default="fixed"
    )
    oracle.add_argument("--prefix-parity-report", type=Path)
    oracle.add_argument("--recovery-manifest", type=Path)
    oracle.add_argument("--observability-manifest", type=Path)
    oracle.add_argument("--correction-manifest", type=Path)
    oracle.add_argument("--recovery-loss-weight", type=float, default=1.0)
    oracle.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/so_arm101_v2/oracle_distillation"),
    )
    _numerics_flags(oracle)
    analysis = commands.add_parser("analyze-oracle")
    analysis.add_argument("--manifest", type=Path, required=True)
    analysis.add_argument("--checkpoint", type=Path, required=True)
    analysis.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/so_arm101_v2/oracle_distillation"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _require_pinned_torch()
        if args.command == "distill-oracle":
            result = distill_oracle_policy(
                args.manifest, args.output_dir, kind=args.model_kind,
                config=OracleDistillationConfig(
                    seed=args.seed,
                    max_steps=args.max_steps,
                    hidden_width=args.hidden_width,
                    training_rows=args.training_rows,
                    lr_schedule=args.lr_schedule,
                    recovery_loss_weight=args.recovery_loss_weight,
                ),
                prefix_parity_report=args.prefix_parity_report,
                recovery_manifest_path=args.recovery_manifest,
                observability_manifest_path=args.observability_manifest,
                correction_manifest_path=args.correction_manifest,
                numerics=_resolve_cli_numerics(args),
            )
            print(result.report_json)
            print(
                f"passed={str(result.passed).lower()} steps={result.steps} "
                f"closed_loop_eligible={str(result.closed_loop_eligible).lower()} "
                f"normalized_mse={result.normalized_mse:.10g} "
                f"max_act_error={result.max_act_error:.10g}"
            )
            return 0 if result.passed else 2
        if args.command == "analyze-oracle":
            result = analyze_oracle_residuals(
                args.manifest, args.checkpoint, args.output_dir
            )
            print(result.report_json)
            return 0
        inventory = inventory_physical_dataset(args.dataset_root)
        manifests = build_split_manifests(inventory)
        if args.command == "baselines":
            result = evaluate_baselines(
                inventory,
                manifests,
                output_dir=args.output_dir,
            )
            print(args.output_dir / baseline_resource_name(result.dataset_digest))
            return 0
        if args.command == "memorize":
            result = run_tiny_overfit(
                args.dataset_root, inventory, args.baseline_report, args.output_dir,
                config=TinyModelConfig(sample_count=args.sample_count, max_steps=args.max_steps),
            )
            print(result.report_json)
            print(
                f"passed={str(result.passed).lower()} steps={result.steps} "
                f"normalized_mse={result.normalized_mse:.10g} "
                f"max_act_error={result.max_act_error:.10g}"
            )
            return 0 if result.passed else 2
        cache = build_lead3_training_cache(
            args.dataset_root, inventory, manifests, args.cache_dir
        )
        if args.command == "train-small":
            comparison = train_small_models(
                cache, args.output_dir,
                config=SmallModelConfig(
                    seeds=tuple(args.seeds), max_epochs=args.max_epochs,
                    batch_size=args.batch_size, device=args.device,
                ),
            )
            print(comparison.report_json)
            return 0
        comparison = load_small_model_comparison(args.comparison_report)
        if comparison.dataset_digest != inventory.dataset_digest:
            raise ValueError("comparison report dataset digest mismatch")
        if args.command == "compare-small":
            report = evaluate_image_ablation(cache, comparison, args.output_dir)
            print(Path(args.output_dir) / f"image_ablation.lead3.v1.{inventory.dataset_digest[:12]}.json")
            print(f"status={report['status']}")
            return 0
        report = build_act_decision_report(
            cache, comparison, args.image_ablation_report, args.preflight_report,
            args.simulation_report, args.output,
        )
        print(args.output)
        print(f"status={report['status']} act_implemented=false")
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
