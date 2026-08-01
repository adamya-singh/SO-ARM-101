"""Command-line interface for reproducible physical-data artifacts."""

from __future__ import annotations

import argparse
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

from ._serialization import canonical_json_bytes, write_immutable_json
from .inventory import inventory_physical_dataset, inventory_resource_name
from .splits import build_split_manifests, split_resource_name
from .targets import audit_future_target_candidates


def target_audit_resource_name(dataset_id: str, dataset_digest: str) -> str:
    return f"{dataset_id}.future_targets.v1.{dataset_digest[:12]}.json"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("imitation-learning/datasets/so101_pickplace_v1"),
        help="path to the physical dataset",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="so-arm101-v2-data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "make-splits", "audit-targets"):
        child = subparsers.add_parser(name)
        _add_common(child)
        child.add_argument("--output-dir", type=Path)
    check = subparsers.add_parser("check")
    _add_common(check)
    visualize = subparsers.add_parser("visualize-sample")
    _add_common(visualize)
    visualize.add_argument("--episode", type=int, default=45)
    visualize.add_argument("--frame", type=int, default=168)
    visualize.add_argument("--lead", type=int, choices=(1, 3, 5), default=3)
    visualize.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/so_arm101_v2/sample_visualizer"),
    )
    visualize.add_argument("--viewer", action="store_true")
    visualize.add_argument(
        "--mujoco-model",
        type=Path,
        default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"),
    )
    visualize.add_argument("--playback-fps", type=float, default=5.0)
    return parser


def _artifacts(dataset_root: Path) -> tuple[Any, dict[str, Any], dict[str, object]]:
    inventory = inventory_physical_dataset(dataset_root)
    manifests = build_split_manifests(inventory)
    audit = audit_future_target_candidates(inventory, manifests)
    return inventory, manifests, audit


def _write_audit(output_dir: Path, inventory: Any, audit: dict[str, object]) -> Path:
    path = output_dir / target_audit_resource_name(
        inventory.dataset_id, inventory.dataset_digest
    )
    write_immutable_json(path, audit)
    return path


def _check_packaged(name: str, value: object) -> None:
    expected = canonical_json_bytes(value, pretty=True)
    resource = files("so_arm101_v2.data.resources").joinpath(name)
    if not resource.is_file():
        raise FileNotFoundError(f"packaged canonical resource is missing: {name}")
    if resource.read_bytes() != expected:
        raise ValueError(f"packaged canonical resource differs from regeneration: {name}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inventory":
            inventory = inventory_physical_dataset(
                args.dataset_root, output_dir=args.output_dir
            )
            print(inventory_resource_name(inventory))
            return 0

        if args.command == "visualize-sample":
            from so_arm101_v2.visualization import (
                build_sample_report,
                launch_mujoco_sample_viewer,
            )

            from .samples import SampleReference, load_future_state_samples

            inventory = inventory_physical_dataset(args.dataset_root)
            sample = load_future_state_samples(
                args.dataset_root,
                [SampleReference(args.episode, args.frame, args.lead)],
                inventory=inventory,
            )[0]
            report = build_sample_report(sample, args.output_dir)
            print(report.html)
            print(report.json)
            if args.viewer:
                launch_mujoco_sample_viewer(
                    sample, args.mujoco_model, playback_fps=args.playback_fps
                )
            return 0

        inventory, manifests, audit = _artifacts(args.dataset_root)
        if args.command == "make-splits":
            if args.output_dir is not None:
                build_split_manifests(inventory, output_dir=args.output_dir)
            for manifest in manifests.values():
                print(split_resource_name(manifest))
            return 0
        if args.command == "audit-targets":
            if args.output_dir is not None:
                print(_write_audit(args.output_dir, inventory, audit))
            else:
                print(target_audit_resource_name(
                    inventory.dataset_id, inventory.dataset_digest
                ))
            return 0

        _check_packaged(inventory_resource_name(inventory), inventory.to_dict())
        for manifest in manifests.values():
            _check_packaged(split_resource_name(manifest), manifest.to_dict())
        _check_packaged(
            target_audit_resource_name(inventory.dataset_id, inventory.dataset_digest),
            audit,
        )
        print(
            f"canonical data artifacts match dataset {inventory.dataset_digest}"
        )
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "target_audit_resource_name"]
