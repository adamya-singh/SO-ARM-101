"""Compare the 3.9.0 re-run of research-log entries 03-07 against the quarantined 3.11.0 reports.

Reads ``<root>/ledger.json`` written by ``tools/rerun_entries_03_07.py``, loads each step's new report
and the quarantined report it re-runs, extracts the headline numbers each research-log entry quotes
(successes out of N, safety frames per try, gate verdicts, fit results), prints a markdown table and
writes it to ``<root>/comparison.md``.  Anything missing or unreadable shows as ``n/a``.

    python tools/compare_quarantine.py [--root artifacts/so_arm101_v2/rerun_mujoco_3_9_0/entries_03_07]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "artifacts/so_arm101_v2/rerun_mujoco_3_9_0/entries_03_07"
QUARANTINE = REPO / "artifacts/so_arm101_v2/quarantine_mujoco_3_11_0/oracle_distillation"
ORIGINAL = REPO / "artifacts/so_arm101_v2/oracle_distillation"
NA = "n/a"

# step id -> (tree, relative path) of the original report it re-runs.  "orig" = the non-quarantined
# tree: torch-only training reports (valid, but some trained on 3.11 capture data) and the
# pre-quarantine 3.9 baseline evaluation.
ORIGINAL_REPORTS: dict[str, tuple[str, str]] = {
    "00.prefix_reference": ("orig", "models/phase_state/6fc677db0c3ae755/report.json"),
    "00.baseline_eval": ("orig", "clone_evaluations/943cf536710e3d84/policies/fixed_pick_place_v3.nominal/evaluation.json"),
    "03.capture_recovery": ("q", "recovery/0570ec8c0d37002f/manifest.json"),
    "03.distill": ("orig", "models/phase_state/c3ca76dc2c0fa42d/report.json"),
    "03.evaluate_clone": ("q", "clone_evaluations/e67ba4433d3aa98c/policies/fixed_pick_place_v3/evaluation.json"),
    "03.recovery_starts": ("q", "recovery_evaluations/4e5e4f1252d582ae/report.json"),
    "03.recovery_starts_baseline": ("q", "recovery_evaluations/b0e6e56a018c1d7f/report.json"),
    "04.scan_w0.00": ("q", "static_command_scans/0255f8e093a6649f/report.json"),
    "04.scan_w1.00": ("q", "static_command_scans/f95b3aba27c4a12d/report.json"),
    "04.distill_w0.10": ("orig", "models/phase_state/f26b6c646a76823e/report.json"),
    "04.scan_w0.10": ("q", "static_command_scans/460243a473cde400/report.json"),
    "04.eval_w0.10": ("q", "clone_evaluations/80d1706cce5d9675/policies/fixed_pick_place_v3.nominal/evaluation.json"),
    "04.distill_w0.25": ("orig", "models/phase_state/5a402430f069faa7/report.json"),
    "04.distill_w0.50": ("orig", "models/phase_state/f27127f973a1b048/report.json"),
    "04.scan_w0.50": ("q", "static_command_scans/859c763c50c7b754/report.json"),
    "04.eval_w0.50": ("q", "clone_evaluations/45084642c47d1200/policies/fixed_pick_place_v3.nominal/evaluation.json"),
    "05.capture_observability": ("q", "observability/ec2d88b418e15186/manifest.json"),
    "05.observability_gate": ("q", "observability_gates/3cdb2c3d2c467a5b/report.json"),
    "06.capture_corrections": ("q", "corrections/7feed472071fd725/manifest.json"),
    "06.correction_gate": ("q", "correction_gates/faf13e95b8caa2e8/report.json"),
    "06.correction_probe": ("q", "correction_probes/63e2717b530a2cf0/report.json"),
    "07.chunked_gate": ("q", "chunked_gates/591f0686e94d27fd/report.json"),
}


def _load(path: Path | str | None) -> dict | None:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _get(d: Any, *keys: Any, default: Any = None) -> Any:
    for key in keys:
        try:
            d = d[key]
        except Exception:
            return default
    return d


def _fmt(value: Any) -> str:
    if value is None:
        return NA
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _safety(rollout: Mapping[str, Any], name: str) -> Any:
    # clone evaluations store flat counts, gate summaries nest them under safety_counts
    if name in rollout:
        return rollout[name]
    return _get(rollout, "safety_counts", name)


def _height(rollout: Mapping[str, Any]) -> Any:
    return rollout.get("maximum_height_gain_m", rollout.get("maximum_cube_height_gain_m"))


def rollout_metrics(rollouts: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(rollouts, list) or not rollouts:
        return {}
    nominal = [r for r in rollouts if r.get("scenario_id") == "nominal"] or rollouts

    def per_try(name: str) -> Any:
        values = [_safety(r, name) for r in nominal]
        values = [v for v in values if isinstance(v, (int, float))]
        return round(sum(values) / len(values), 2) if values else None

    heights = [_height(r) for r in nominal if isinstance(_height(r), (int, float))]
    milestones = set()
    for r in nominal:
        milestones.update(r.get("milestones") or [])
    out = {
        "successes": f"{sum(1 for r in rollouts if r.get('success'))}/{len(rollouts)}",
        "safety_invalidated": f"{sum(1 for r in rollouts if r.get('invalidated'))}/{len(rollouts)}",
        "nominal_clipping_frames_per_try": per_try("clipping_frames"),
        "nominal_limiting_frames_per_try": per_try("limiting_frames"),
        "nominal_unsafe_contact_frames_per_try": per_try("unsafe_contact_frames"),
        "nominal_max_lift_mm": round(max(heights) * 1000, 1) if heights else None,
        "nominal_strict_grasp": ("strict_grasp_acquired" in milestones) if milestones else None,
        "failure_categories": ",".join(sorted({str(r.get("failure_category")) for r in rollouts})),
    }
    return {prefix + k: v for k, v in out.items()}


def m_evaluation(d: dict) -> dict[str, Any]:
    return rollout_metrics(d.get("rollouts"))


def m_distill(d: dict) -> dict[str, Any]:
    return {
        "fit_passed": d.get("passed"),
        "steps": d.get("steps"),
        "normalized_mse": d.get("normalized_mse"),
        "max_act_error": d.get("max_act_error"),
    }


def m_recovery_manifest(d: dict) -> dict[str, Any]:
    return {"rows": _get(d, "arrays", "rows"), "collection_digest": (d.get("collection_digest") or "")[:16] or None}


def m_recovery_starts(d: dict) -> dict[str, Any]:
    results = d.get("results") or []
    return {
        "anchors_finished": f"{sum(1 for r in results if r.get('success'))}/{len(results)}" if results else None,
        "passed": d.get("passed"),
    }


def m_scan(d: dict) -> dict[str, Any]:
    return {
        "scan_passed": d.get("passed"),
        "samples_with_any_violation": _get(d, "counts", "samples_with_any_violation"),
        "max_abs_normalized_delta": d.get("maximum_absolute_normalized_delta"),
    }


def m_observability_manifest(d: dict) -> dict[str, Any]:
    return {"collection_digest": (d.get("collection_digest") or "")[:16] or None}


def m_observability_gate(d: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"status": d.get("status"), "next_action": d.get("next_action")}
    order = d.get("candidate_order") or []
    for index, cand in enumerate(d.get("candidates") or []):
        name = order[index] if index < len(order) else f"candidate{index}"
        out[f"{name}.fit_passed"] = _get(cand, "offline", "passed")
        out[f"{name}.fit_steps"] = _get(cand, "offline", "steps")
        out.update(rollout_metrics(_get(cand, "evaluation", "rollouts"), prefix=f"{name}."))
    return out


def m_corrections(d: dict) -> dict[str, Any]:
    episodes = d.get("episodes") or []
    return {
        "rows": _get(d, "arrays", "rows"),
        "sites_accepted": f"{sum(1 for e in episodes if e.get('accepted'))}/{len(episodes)}" if episodes else None,
    }


def m_correction_gate(d: dict) -> dict[str, Any]:
    cand = d.get("candidate") or {}
    return {
        "status": d.get("status"),
        "candidate_fit_passed": _get(cand, "offline", "passed"),
        "candidate_normalized_mse": _get(cand, "offline", "normalized_mse"),
        "candidate_max_act_error": _get(cand, "offline", "maximum_act_error"),
        "nominal_evaluations_started": d.get("nominal_evaluations_started"),
    }


def m_correction_probe(d: dict) -> dict[str, Any]:
    out = {
        "status": d.get("status"),
        "safety_regressed": _get(d, "resolution", "safety_regressed"),
        "new_milestones": ",".join(_get(d, "resolution", "new_milestones", default=[]) or []) or None,
    }
    out.update(rollout_metrics(d.get("probe_rollouts"), prefix="probe."))
    return out


def m_chunked_gate(d: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"status": d.get("status")}
    for cand in d.get("candidates") or []:
        name = f"H{cand.get('chunk_horizon')}"
        out[f"{name}.offline_normalized_mse"] = _get(cand, "offline_telemetry", "normalized_mse")
        out.update(rollout_metrics(_get(cand, "evaluation", "rollouts"), prefix=f"{name}."))
    return out


EXTRACTORS: dict[str, Callable[[dict], dict[str, Any]]] = {
    "00.prefix_reference": m_distill,
    "00.baseline_eval": m_evaluation,
    "03.capture_recovery": m_recovery_manifest,
    "03.distill": m_distill,
    "03.evaluate_clone": m_evaluation,
    "03.recovery_starts": m_recovery_starts,
    "03.recovery_starts_baseline": m_recovery_starts,
    "04.scan_w0.00": m_scan,
    "04.scan_w1.00": m_scan,
    "05.capture_observability": m_observability_manifest,
    "05.observability_gate": m_observability_gate,
    "06.capture_corrections": m_corrections,
    "06.correction_gate": m_correction_gate,
    "06.correction_probe": m_correction_probe,
    "07.chunked_gate": m_chunked_gate,
}
for _w in ("0.10", "0.25", "0.50"):
    EXTRACTORS[f"04.distill_w{_w}"] = m_distill
    EXTRACTORS[f"04.scan_w{_w}"] = m_scan
    EXTRACTORS[f"04.eval_w{_w}"] = m_evaluation
# canonical (driver) order, used when the ledger lacks a step
STEP_ORDER = [s for s in EXTRACTORS if not s.startswith(("05.", "06.", "07."))] + [
    s for s in EXTRACTORS if s.startswith(("05.", "06.", "07."))]


def original_path(step_id: str, quarantine: Path = QUARANTINE, original: Path = ORIGINAL) -> Path | None:
    if step_id not in ORIGINAL_REPORTS:
        return None
    tree, rel = ORIGINAL_REPORTS[step_id]
    return (quarantine if tree == "q" else original) / rel


def _metrics(step_id: str, doc: dict | None) -> dict[str, Any]:
    if doc is None:
        return {}
    try:
        return EXTRACTORS[step_id](doc)
    except Exception:
        return {}


def rows_for(ledger: Mapping[str, Any], quarantine: Path = QUARANTINE,
             original: Path = ORIGINAL) -> list[tuple[str, str, str, str, str]]:
    """(entry, step, metric, original value, new value) for every known step, ledger order first."""
    steps = list(ledger.get("steps", {}).keys())
    steps += [s for s in STEP_ORDER if s not in steps]
    rows = []
    for step_id in steps:
        if step_id not in EXTRACTORS:
            continue
        record = ledger.get("steps", {}).get(step_id, {})
        old = _metrics(step_id, _load(original_path(step_id, quarantine, original)))
        new = _metrics(step_id, _load(_get(record, "outputs", "report")))
        if record.get("status") == "skipped_by_rule":
            new = {k: "skipped (rule)" for k in old} or {"run": "skipped (rule)"}
        names = list(old) + [k for k in new if k not in old]
        if not names:
            names = ["report"]
        for name in names:
            rows.append((step_id.split(".")[0], step_id, name, _fmt(old.get(name)), _fmt(new.get(name))))
    return rows


def headline_numbers(entry: str, ledger: Mapping[str, Any], quarantine: Path = QUARANTINE) -> dict[str, float]:
    """Numeric headline metrics of one entry's new reports, for W&B (01_outcome/entryNN_*)."""
    out: dict[str, float] = {}
    for step_id, record in ledger.get("steps", {}).items():
        if not step_id.startswith(entry + ".") or step_id not in EXTRACTORS:
            continue
        for name, value in _metrics(step_id, _load(_get(record, "outputs", "report"))).items():
            key = f"{step_id.split('.', 1)[1]}_{name}".replace(".", "_").replace("/", "_")
            if isinstance(value, bool):
                out[key] = float(value)
            elif isinstance(value, (int, float)):
                out[key] = float(value)
            elif isinstance(value, str) and "/" in value and value.replace("/", "").isdigit():
                num, den = value.split("/")
                out[key] = float(num)
                out[key + "_of"] = float(den)
    return out


def render(rows: list[tuple[str, str, str, str, str]], root: Path) -> str:
    lines = [
        "# Entries 03-07: MuJoCo 3.11.0 (quarantined) vs 3.9.0 (re-run)",
        "",
        f"Ledger: `{root / 'ledger.json'}`. The \"3.11 value\" column is the quarantined report; for "
        "`00.*` steps and the training reports (`*distill*`) it is the original non-quarantined artifact "
        "(torch-only training, or the pre-quarantine 3.9 baseline evaluation). Training under the default "
        "numerics regime is expected to differ from the originals' legacy cpu-eager lane even before "
        "physics enters. Missing values are `n/a`.",
        "",
        "| entry | metric | 3.11 value | 3.9 value |",
        "|---|---|---|---|",
    ]
    for entry, step_id, name, old, new in rows:
        lines.append(f"| {entry} | {step_id.split('.', 1)[1]}: {name} | {old} | {new} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--quarantine-root", type=Path, default=QUARANTINE)
    parser.add_argument("--original-root", type=Path, default=ORIGINAL)
    args = parser.parse_args(argv)
    ledger = _load(args.root / "ledger.json") or {"steps": {}}
    text = render(rows_for(ledger, args.quarantine_root, args.original_root), args.root)
    print(text)
    args.root.mkdir(parents=True, exist_ok=True)
    tmp = args.root / "comparison.md.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(args.root / "comparison.md")
    # Per-entry summary for the tranche status page (tools/tranche_status.py).
    summary: dict[str, Any] = {}
    for entry, step_id, name, old, new in rows_for(ledger, args.quarantine_root, args.original_root):
        if new == "n/a":
            continue
        item = summary.setdefault(entry, {"compared": 0, "changed": 0, "rows": []})
        item["compared"] += 1
        item["changed"] += int(old != new)
        item["rows"].append({"metric": f"{step_id.split('.', 1)[1]}: {name}", "mujoco_3_11": old, "mujoco_3_9": new})
    for item in summary.values():
        item["verdict_3_9"] = f"{item['changed']} of {item['compared']} reported numbers differ from 3.11"
    tmp = args.root / "comparison.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    tmp.replace(args.root / "comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
