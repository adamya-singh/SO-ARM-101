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
from .tiny_model import TinyModelConfig, run_tiny_overfit


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
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
