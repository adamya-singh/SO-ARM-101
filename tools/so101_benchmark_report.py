"""SO-101 benchmark report (2026-11-06, handoff step 3): each SO-101 scene next to the SO-100 baselines.

Reads, per scene, the teacher certification summary, the ACT fine-tune's result.json (execute 90 and 30) and the CNN's
augmentation.json, plus the same files of the SO-100 baselines (retry-tranche ACT fine-tune 20/30, CNN 4800 6/30),
and the closure / correction in mm from each run's held-out telemetry. Pre-registered reading: a scene matches if its
ACT fine-tune is within +-4/30 of 20/30 and its CNN within +-4/30 of 6/30.

  PYTHONNOUSERSITE=1 ~/miniforge3/envs/lerobot/bin/python tools/so101_benchmark_report.py --root <benchmark dir> --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
E = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
MODELS = ROOT / "simulation_code/model"
BASELINES = dict(act_ft=E / "retry_tranche_20261004/act_ft", cnn=E / "augmentation_shift12_4800_20260912")
SCENES = {"so100": "bench_pick_replace_v1", "so101": "bench_pick_replace_v1_so101", "so101_menagerie": "bench_pick_replace_v1_so101_menagerie"}
TOLERANCE = 4


def _act(run: Path) -> dict:
    path = run / "result.json"
    if not path.exists():
        return {"status": "missing"}
    r = json.loads(path.read_text())
    return {e: dict(successes=v.get("successes"), rollouts=v.get("rollouts"), lateral_rms_mm=(v.get("closure") or {}).get("lateral_rms_mm"),
                    not_descended=(v.get("closure") or {}).get("not_descended"), categories=v.get("categories"))
            for e, v in (r.get("rollouts") or {}).items()}


def _cnn(run: Path, scene_xml: Path) -> dict:
    path = run / "augmentation.json"
    if not path.exists():
        return {"status": "missing"}
    point = json.loads(path.read_text())["points"][0]
    success = point.get("heldout_success") or {}
    from so_arm101_v2.simulation.closure import correction_summary
    telemetry = sorted(glob.glob(str(run / "evaluations/**/telemetry/*.json"), recursive=True))
    correction = {k: v for k, v in correction_summary(scene_xml, telemetry).items() if k != "rows"} if telemetry else {}
    return dict(successes=success.get("successes"), rollouts=success.get("rollouts"), heldout_start_90=point.get("heldout_start_90"),
                lateral_rms_mm_250=correction.get("lateral_rms_mm_250"), not_descended=correction.get("not_descended"))


def _certification(scene_xml: Path) -> dict:
    from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash
    from so_arm101_v2.simulation.arm_profiles import grasp_detector_version, scene_arm_profile
    from so_arm101_v2.simulation.bench import certification_suite
    tag = f"{scene_dependency_hash(scene_xml)[:16]}-{grasp_detector_version(scene_arm_profile(scene_xml))}"
    suite_id = certification_suite(scene_bench_config(scene_xml), repeats=3).suite_id   # the current suite, not an older one
    reports = glob.glob(str(E.parent / "teacher_certification" / tag / "preflight" / suite_id / "evaluation.json"))
    if not reports:
        return {"status": "missing", "tag": tag}
    r = json.loads(Path(reports[0]).read_text())
    return dict(tag=tag, successes=sum(bool(x["success"] and not x.get("invalidated")) for x in r["rollouts"]), rollouts=len(r["rollouts"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="benchmark directory with <scene>/act_ft and <scene>/cnn_s202")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    report: dict = {}
    for scene, folder in SCENES.items():
        xml = MODELS / folder / "scene_bench_pick_replace_v1.xml"
        act_dir, cnn_dir = (BASELINES["act_ft"], BASELINES["cnn"]) if scene == "so100" else (args.root / scene / "act_ft", args.root / scene / "cnn_s202")
        report[scene] = dict(teacher=_certification(xml), act_ft=_act(act_dir), cnn_s202=_cnn(cnn_dir, xml))
    base_act = (report["so100"]["act_ft"].get("90") or {}).get("successes")
    base_cnn = report["so100"]["cnn_s202"].get("successes")
    for scene in ("so101", "so101_menagerie"):
        act = (report[scene]["act_ft"].get("90") or {}).get("successes")
        cnn = report[scene]["cnn_s202"].get("successes")
        report[scene]["verdict"] = dict(
            act_within_tolerance=None if act is None or base_act is None else abs(act - base_act) <= TOLERANCE,
            cnn_within_tolerance=None if cnn is None or base_cnn is None else abs(cnn - base_cnn) <= TOLERANCE)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    for scene, r in report.items():
        act90 = r["act_ft"].get("90") or {}
        print(f"{scene:16s} teacher {r['teacher'].get('successes')}/{r['teacher'].get('rollouts')}  ACT FT exec90 {act90.get('successes')}/{act90.get('rollouts')} "
              f"(rms {act90.get('lateral_rms_mm')})  CNN {r['cnn_s202'].get('successes')}/{r['cnn_s202'].get('rollouts')}  {r.get('verdict', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
