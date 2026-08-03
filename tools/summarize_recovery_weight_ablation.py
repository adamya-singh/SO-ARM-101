"""Build the immutable summary for the approved recovery-row weight ablation."""

from __future__ import annotations

import json
from pathlib import Path

from so_arm101_v2.data._serialization import content_sha256, write_immutable_json


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/so_arm101_v2/oracle_distillation"


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("content_sha256")
    body = dict(payload)
    body.pop("content_sha256", None)
    if stated != content_sha256(body):
        raise ValueError(f"content hash mismatch: {path}")
    return payload


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    control_scans = {
        "0.00": ARTIFACT_ROOT / "static_command_scans/0255f8e093a6649f/report.json",
        "1.00": ARTIFACT_ROOT / "static_command_scans/f95b3aba27c4a12d/report.json",
    }
    candidates = {
        "0.10": {
            "offline": ARTIFACT_ROOT / "models/phase_state/f26b6c646a76823e/report.json",
            "scan": ARTIFACT_ROOT / "static_command_scans/460243a473cde400/report.json",
            "nominal": ARTIFACT_ROOT / "clone_evaluations/80d1706cce5d9675/policies/fixed_pick_place_v3.nominal/evaluation.json",
        },
        "0.25": {
            "offline": ARTIFACT_ROOT / "models/phase_state/5a402430f069faa7/report.json",
        },
        "0.50": {
            "offline": ARTIFACT_ROOT / "models/phase_state/f27127f973a1b048/report.json",
            "scan": ARTIFACT_ROOT / "static_command_scans/859c763c50c7b754/report.json",
            "nominal": ARTIFACT_ROOT / "clone_evaluations/45084642c47d1200/policies/fixed_pick_place_v3.nominal/evaluation.json",
        },
    }
    controls = {}
    for weight, path in control_scans.items():
        report = load_hashed(path)
        controls[weight] = {
            "report": relative(path),
            "report_content_sha256": report["content_sha256"],
            "passed_strict_bounds": report["passed"],
            "samples_with_any_violation": report["counts"]["samples_with_any_violation"],
            "nonfinite_samples": report["counts"]["nonfinite_samples"],
            "maximum_absolute_normalized_delta": report["maximum_absolute_normalized_delta"],
        }
    if controls["0.00"]["passed_strict_bounds"]:
        raise RuntimeError("summary decision rule assumes the observed weight-0 scan failure")

    results = []
    for weight, paths in candidates.items():
        offline = load_hashed(paths["offline"])
        entry = {
            "recovery_loss_weight": float(weight),
            "offline_report": relative(paths["offline"]),
            "offline_report_content_sha256": offline["content_sha256"],
            "offline_passed": offline["passed"],
            "steps": offline["steps"],
            "nominal_metrics": offline["nominal_metrics"],
            "recovery_metrics": offline["recovery_metrics"],
            "weighted_objective": offline["weighted_objective"],
            "static_scan": None,
            "nominal_mujoco": None,
            "all_five_starts": None,
            "eight_anchor_handoffs": None,
        }
        if not offline["passed"]:
            entry["stopped_at"] = "nominal_only_offline_gate"
            results.append(entry)
            continue
        scan = load_hashed(paths["scan"])
        entry["static_scan"] = {
            "report": relative(paths["scan"]),
            "report_content_sha256": scan["content_sha256"],
            "strict_bounds_passed": scan["passed"],
            "samples": scan["total_samples"],
            "counts": scan["counts"],
            "blocking": False,
            "reason": "weight-0 control fails the same strict scan; retain as telemetry only",
        }
        nominal = load_hashed(paths["nominal"])
        rollouts = nominal["rollouts"]
        passed = bool(
            nominal["deterministic"] and len(rollouts) == 3
            and all(
                row["success"] and not row["invalidated"]
                and row["clipping_frames"] == 0 and row["limiting_frames"] == 0
                and row["nonfinite_frames"] == 0 and row["unsafe_contact_frames"] == 0
                for row in rollouts
            )
        )
        entry["nominal_mujoco"] = {
            "report": relative(paths["nominal"]),
            "report_content_sha256": nominal["content_sha256"],
            "passed": passed,
            "successes": sum(int(row["success"]) for row in rollouts),
            "rollouts": len(rollouts),
            "deterministic": nominal["deterministic"],
            "per_repeat_safety": [
                {
                    key: row[key] for key in (
                        "invalidated", "clipping_frames", "limiting_frames",
                        "nonfinite_frames", "unsafe_contact_frames",
                    )
                } for row in rollouts
            ],
        }
        entry["stopped_at"] = "nominal_mujoco_3_of_3" if not passed else None
        if passed:
            raise RuntimeError("summary inputs omit required later gates for a nominal-passing candidate")
        results.append(entry)

    identity = {
        "schema_version": 1,
        "experiment": "recovery_row_loss_weight_ablation_v1",
        "candidate_weights_in_order": [0.10, 0.25, 0.50],
        "control_scan_content_sha256": {
            weight: value["report_content_sha256"] for weight, value in controls.items()
        },
        "candidate_offline_content_sha256": [row["offline_report_content_sha256"] for row in results],
        "candidate_scan_content_sha256": [
            None if row["static_scan"] is None else row["static_scan"]["report_content_sha256"]
            for row in results
        ],
        "candidate_nominal_mujoco_content_sha256": [
            None if row["nominal_mujoco"] is None else row["nominal_mujoco"]["report_content_sha256"]
            for row in results
        ],
    }
    digest = content_sha256(identity)
    report = {
        **identity,
        "experiment_digest": digest,
        "objective": "(sum nominal element losses + w * sum recovery element losses) / (450 + 8w)",
        "selection_gate": "nominal-only offline metrics; never combined weighted MSE",
        "static_scan_calibration": {
            "controls": controls,
            "decision": "nonblocking_telemetry",
            "reason": "the known weight-0 control fails strict interpolated command bounds",
            "physical_safety_gate": "nominal MuJoCo deterministic 3/3 with zero safety events",
        },
        "candidates": results,
        "accepted_weight": None,
        "passed": False,
        "decision": "reject_all_three_weights",
        "meaning": "lower sparse-anchor influence did not yield a checkpoint that preserves nominal autonomous safety; 0.25 also missed the nominal offline fit gate",
        "unrun_by_gate": ["five_fixed_starts_15_of_15", "eight_exact_anchor_handoffs_8_of_8"],
        "claim": "controlled_sparse_recovery_weight_ablation_only_not_a_general_recovery_learning_result",
    }
    report["content_sha256"] = content_sha256(report)
    path = ARTIFACT_ROOT / "recovery_weight_ablations" / digest[:16] / "report.json"
    write_immutable_json(path, report)
    print(path)


if __name__ == "__main__":
    main()
