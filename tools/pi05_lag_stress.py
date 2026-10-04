"""Does a pi0.5 replanning schedule survive real servo lag? Held-out rollouts with and without the lag stress.

For each variant (re-read schedule or lag compensation of ``Pi05ChunkPolicy``) the held-out placements are rolled out
on the certified scene and with a measured real servo response in front of the actuators
(``simulation/lag_stress.py``). Reported per variant and condition: successes, failure categories, how far the
arm is from the survey pose when the survey frame is read (step 90), whether the cube is inside that frame,
and the closure error in millimetres. One model on the GPU, rollouts in-process. Simulation only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BENCH = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1"
DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
DEFAULT_CHECKPOINT = BENCH / "experiments/pi05_lora16_10k_20261002/model/pi05.pt"
DEFAULT_SUITE = BENCH / "experiments/scaling_ladder_20260910/heldout_suite.path"

VARIANTS = {
    "exec30_measured": dict(execute=30),
    "exec90_measured": dict(execute=90),
    "exec30_first90": dict(execute=30, first_execute=90),
    "exec30_lagcomp": dict(execute=30, servo_response="fit_20261004"),
    "exec30_adaptive": dict(execute=30, servo_response="adaptive"),
    "exec30_assist": dict(execute=30, assist=True),
    "exec90_assist": dict(execute=90, assist=True),
    "exec30_assist_adaptive": dict(execute=30, assist=True, servo_response="adaptive"),
    "exec90_assist_adaptive": dict(execute=90, assist=True, servo_response="adaptive"),
    "exec30_assist_gravity": dict(execute=30, assist=dict(friction_scale=0.0, integral_gain=0.0)),
    "exec90_assist_gravity": dict(execute=90, assist=dict(friction_scale=0.0, integral_gain=0.0)),
    "exec30_assist_gravity_integral": dict(execute=30, assist=dict(friction_scale=0.0)),
    "exec90_assist_gravity_integral": dict(execute=90, assist=dict(friction_scale=0.0)),
    "exec30_assist_gravity_friction": dict(execute=30, assist=dict(integral_gain=0.0)),
    "exec90_assist_gravity_friction": dict(execute=90, assist=dict(integral_gain=0.0)),
    "exec30_hold2": dict(execute=30, hold=2),
    "exec30_hold3": dict(execute=30, hold=3),
    "exec90_hold2": dict(execute=90, hold=2),
    "exec90_hold3": dict(execute=90, hold=3),
}


def survey_rows(report_rows, model: Path, bench) -> dict:
    """At the survey read (before action 91): distance from the viewing pose and whether the cube is in the frame."""
    from so_arm101_v2.physical.placement import cube_visible_at
    step = bench.observation_steps
    viewing = np.asarray(bench.viewing_qpos, dtype=np.float64)[:5]
    out = []
    for r in report_rows:
        rows = json.loads(Path(r["telemetry_path"]).read_text())["rows"]
        if len(rows) < step:
            continue
        qpos = np.asarray(rows[step - 1]["robot_qpos"], dtype=np.float64)
        visible = cube_visible_at(model, bench, qpos, rows[step - 1]["cube_position"], margin_px=0.0)["visible"]
        out.append(dict(scenario_id=r["scenario_id"], max_abs_rad=float(np.max(np.abs(qpos[:5] - viewing))),
                        elbow_rad=float(qpos[2] - viewing[2]), cube_in_frame=bool(visible)))
    return dict(rows=out, max_abs_rad_median=float(np.median([o["max_abs_rad"] for o in out])) if out else None,
                elbow_rad_median=float(np.median([o["elbow_rad"] for o in out])) if out else None,
                cube_in_frame=int(sum(o["cube_in_frame"] for o in out)), count=len(out))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE, help="a suite.json, or a *.path file naming one")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--variants", nargs="+", default=["exec30_measured", "exec90_measured", "exec30_first90", "exec30_lagcomp"], choices=list(VARIANTS))
    p.add_argument("--conditions", nargs="+", default=["lag", "nominal", "lag0910"], choices=["nominal", "lag", "lag0910"],
                   help="nominal = the certified scene; lag = the 2026-10-04 servo fit (elbow 0.16); lag0910 = the 2026-09-10 fit (elbow 0.45)")
    p.add_argument("--pairs", nargs="+", default=None, metavar="CONDITION/VARIANT", help="run exactly these, in this order")
    p.add_argument("--limit", type=int, default=None, help="first N scenarios only")
    args = p.parse_args(argv)

    from so_arm101_v2.contracts.bench import scene_bench_config
    from so_arm101_v2.learning.pi05 import PromptTokenizer, load_checkpoint
    from so_arm101_v2.simulation.closure import closure_summary
    from so_arm101_v2.simulation.lag_stress import REAL_SERVO_RESPONSE_20260910, REAL_SERVO_RESPONSE_20261004, LagStressPolicy, TimeDilationPolicy
    from so_arm101_v2.simulation.pi05_policy import Pi05ChunkPolicy
    from so_arm101_v2.simulation.rollout import evaluate_closed_loop
    from so_arm101_v2.simulation.suites import load_suite_from_path

    suite_path = args.suite_path
    if suite_path.suffix == ".path":
        suite_path = Path(suite_path.read_text().strip())
    suite = load_suite_from_path(suite_path)
    scenarios = suite.scenarios if args.limit is None else suite.scenarios[: args.limit]
    suite = dataclasses.replace(suite, suite_id=f"{suite.suite_id}_x1", scenarios=scenarios, repeats=1)   # repeats are identical
    responses = dict(nominal=None, lag=REAL_SERVO_RESPONSE_20261004, lag0910=REAL_SERVO_RESPONSE_20260910)
    bench = scene_bench_config(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else dict(
        tool="pi05_lag_stress", checkpoint=str(args.checkpoint), suite=suite.suite_id, lag_responses={k: v for k, v in responses.items() if v is not None}, results={})

    model, config, stats, _payload = load_checkpoint(args.checkpoint)
    prompt = PromptTokenizer(config.tokenizer, config.tokenizer_max_length)
    pairs = [tuple(pair.split("/")) for pair in args.pairs] if args.pairs else [(c, v) for c in args.conditions for v in args.variants]
    for condition, variant in pairs:
        if condition not in responses or variant not in VARIANTS:
            raise SystemExit(f"unknown pair {condition}/{variant}")
    for condition, variant in pairs:
        if True:
            key = f"{condition}/{variant}"
            if key in summary["results"]:
                continue

            def factory(variant=variant, condition=condition):
                options = dict(VARIANTS[variant])
                if options.get("servo_response") == "fit_20261004":       # the compensation always assumes the 2026-10-04 elbow
                    options["servo_response"] = REAL_SERVO_RESPONSE_20261004
                hold = int(options.pop("hold", 1))
                assisted = options.pop("assist", False)
                policy = Pi05ChunkPolicy(model, config, stats, prompt, clamp_channels=(5,), tag="pi05", **options)
                if assisted:
                    # The runner's servo assist, built as the runner builds it: the certified scene's mass model and the
                    # fitted servo resource. Meaningful with --model set to the measured-servo scene (the arm's stand-in).
                    from so_arm101_v2.physical.servo_assist import AssistedPolicy, AssistTuning, ServoAssist
                    tuning = AssistTuning(**assisted) if isinstance(assisted, dict) else AssistTuning()
                    policy = AssistedPolicy(policy, ServoAssist(DEFAULT_MODEL, bench, tuning=tuning))
                if hold > 1:
                    policy = TimeDilationPolicy(policy, hold)
                return policy if responses[condition] is None else LagStressPolicy(policy, responses[condition])

            started = time.time()
            result = evaluate_closed_loop(args.model, suite, {variant: factory}, args.output_dir / condition / variant,
                                          environment_proven=True, record_video=False, workers=1)
            report = json.loads(Path(result.report_json).read_text())
            rows = [r for r in report["rollouts"] if r["policy_id"] == variant]
            ok = {r["scenario_id"]: bool(r["success"] and not r["invalidated"]) for r in rows}
            categories: dict[str, int] = {}
            for r in rows:
                categories[r["failure_category"]] = categories.get(r["failure_category"], 0) + 1
            closure = closure_summary(args.model, [r["telemetry_path"] for r in rows if r.get("telemetry_path")])
            summary["results"][key] = dict(
                successes=int(sum(ok.values())), rollouts=len(rows), per_scenario=ok, categories=categories,
                survey=survey_rows(rows, args.model, bench), closure={k: v for k, v in closure.items() if k != "rows"},
                seconds=round(time.time() - started, 1), report=str(result.report_json))
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            s = summary["results"][key]
            print(f"LAGSTRESS {key}: {s['successes']}/{s['rollouts']} {categories} survey elbow {s['survey']['elbow_rad_median']:+.3f} rad, "
                  f"cube in survey frame {s['survey']['cube_in_frame']}/{s['survey']['count']} ({s['seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
