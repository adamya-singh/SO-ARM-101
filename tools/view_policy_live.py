"""Watch a retry-tranche policy drive the bench arm live in the MuJoCo viewer on the held-out placements, and record it.

Generalises ``view_pi05_live.py`` to every model the retry tranche trains (pi0.5, SmolVLA, ACT, fine-tuned or RL). Each
episode is set up the way ``evaluate_closed_loop`` sets it up (held-out suite scenario: placement, appearance, the
contract's stopping rule) and the policy is the same ``ResidualChunkPolicy`` the held-out evaluation scored, so what you
see is what the 30-rollout evaluation counted. With ``--record`` every control step is also rendered offscreen from the
viewer's current camera (so the video follows whatever view you set on screen), with the wrist frame the policy last read
in the bottom-left corner and a status line, and written to one MP4 for all the poses shown.

Usage (from the SO-ARM-101 repo root, in a terminal with a display; pi0.5 needs the lerobot-pi05 env, ACT/SmolVLA lerobot):
  PYTHONNOUSERSITE=1 ~/miniforge3/envs/lerobot-pi05/bin/python tools/view_policy_live.py --kind pi05 \
      --checkpoint artifacts/.../retry_tranche_20261004/rl_pi05/model/pi05_rl.pt --label "pi0.5 ReinFlow" --record out.mp4 --once
Without --once it cycles through the poses until the MuJoCo window is closed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
DATA = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/placements4800_20260912"


def recorded_outcomes(run_dir: Path, execute: int) -> dict[str, str]:
    """The held-out evaluation's recorded failure category per pose (repeat 0), from rl_train's or vla_finetune's progress."""
    progress = run_dir / "progress.json"
    if not progress.exists():
        return {}
    state = json.loads(progress.read_text())
    outcomes = state.get("heldout") or state.get("rollouts") or {}  # rl_train | vla_finetune
    rows = (outcomes.get(str(execute)) or {}).get("rollout_rows", [])
    return {r["scenario_id"]: r["failure_category"] for r in rows if int(r.get("repeat", 0)) == 0}


def load_model(kind: str, checkpoint: Path):
    from so_arm101_v2.learning.vla_models import ACTModel, Pi05Model, SmolVLAModel
    return {"pi05": Pi05Model, "smolvla": SmolVLAModel, "act": ACTModel}[kind].from_checkpoint(checkpoint)


class Recorder:
    """Offscreen frames from the viewer's camera, wrist inset and status text, piped to ffmpeg (H.264)."""

    def __init__(self, path: Path, model, width: int, height: int, fps: int) -> None:
        import imageio_ffmpeg
        import mujoco
        self.mujoco = mujoco
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.width, self.height = width, height
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps, codec="libx264", quality=None,
                                                  output_params=["-crf", "26", "-preset", "medium", "-pix_fmt", "yuv420p",
                                                                 "-movflags", "+faststart"], macro_block_size=8)
        self.writer.send(None)
        self.frames = 0
        self.lut = (255.0 * (np.arange(256) / 255.0) ** 0.6).astype(np.uint8)

    def frame(self, data, cam, wrist: np.ndarray | None, lines: tuple[str, ...]) -> None:
        import cv2
        self.renderer.update_scene(data, camera=cam)
        img = np.ascontiguousarray(self.renderer.render())
        # Brighten the video only (gamma 0.6): the randomized dark materials are hard to read in an MP4. The policy's own
        # wrist frame comes from adapter.observation and is never touched by this.
        img = self.lut[img]
        if wrist is not None:
            inset = cv2.resize(np.asarray(wrist, np.uint8), (self.height // 3, self.height // 3), interpolation=cv2.INTER_NEAREST)
            y0 = self.height - inset.shape[0] - 10
            img[y0:y0 + inset.shape[0], 10:10 + inset.shape[1]] = inset
            cv2.rectangle(img, (9, y0 - 1), (10 + inset.shape[1], y0 + inset.shape[0]), (255, 255, 255), 1)
            cv2.putText(img, "wrist camera (last read)", (12, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (self.width, 22 * len(lines) + 12), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (10, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        self.writer.send(img)
        self.frames += 1

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", required=True, choices=("pi05", "smolvla", "act"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--label", default=None, help="name shown on screen and in the video (default: the run directory's name)")
    p.add_argument("--execute", type=int, default=90, help="commands run per camera read (90 = the RL training schedule)")
    p.add_argument("--steps", type=int, default=None, help="denoising steps for flow models (default: the model's)")
    p.add_argument("--poses", type=int, nargs="*", default=None, help="held-out pose indices (default all 10)")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier on screen (1.0 = real time, 30 Hz)")
    p.add_argument("--record", type=Path, default=None, help="write an MP4 of what is shown")
    p.add_argument("--size", type=int, nargs=2, default=(960, 540), metavar=("W", "H"))
    p.add_argument("--once", action="store_true", help="show each pose once, then exit (default: cycle until the window closes)")
    p.add_argument("--summary", type=Path, default=None, help="write per-pose live outcomes as JSON")
    args = p.parse_args(argv)
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("no display: run this from a terminal with DISPLAY set (WSLg: echo $DISPLAY prints :0)")
        return 1

    import mujoco
    import mujoco.viewer
    import torch
    from so_arm101_v2.contracts import PickPlaceEvaluationState, evaluate_pick_place_step, load_pick_place_contract
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.residual_policy import ResidualChunkPolicy
    from so_arm101_v2.simulation.suites import load_suite_from_path

    run_dir = args.checkpoint.resolve().parent.parent
    label = args.label or run_dir.name
    suite = load_suite_from_path(Path((DATA / "heldout_suite.path").read_text().strip()))
    scenarios = [suite.scenarios[i] for i in (args.poses if args.poses else range(len(suite.scenarios)))]
    recorded = recorded_outcomes(run_dir, args.execute)
    print(f"loading {args.checkpoint} ...", flush=True)
    model = load_model(args.kind, args.checkpoint)
    policy = ResidualChunkPolicy(model, execute=args.execute, num_steps=args.steps, tag=label)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.kind == "smolvla" else None
    adapter = MujocoTaskAdapter(MODEL)
    contract = load_pick_place_contract(suite.task_contract, bench_config=adapter.bench)
    recorder = Recorder(args.record, adapter.model, *args.size, fps=30) if args.record else None
    print(f"{label}: {len(scenarios)} held-out placements, execute {args.execute}; close the MuJoCo window to exit", flush=True)

    live: dict[str, str] = {}
    with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = (0.0, 0.18, 0.06)  # side-on: arm and cube both in view for every placement
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.62, 180.0, -24.0
        k = 0
        while viewer.is_running() and not (args.once and k >= len(scenarios)):
            scenario = scenarios[k % len(scenarios)]
            k += 1
            adapter.reset(scenario)
            policy.reset(adapter)
            viewer.sync()
            state = PickPlaceEvaluationState()
            evaluation = None
            max_lift, last_read, wrist = 0.0, 0, None
            want = recorded.get(scenario.scenario_id, "n/a")
            print(f"\n=== {scenario.scenario_id} (evaluation recorded: {want}) ===", flush=True)
            for action in range(contract.max_actions):
                if not viewer.is_running():
                    break
                started = time.time()
                looking = policy.needs_frame
                raw, _, current = adapter.observation(render_pixels=True)
                if looking:
                    last_read, wrist = action, np.ascontiguousarray(raw, dtype=np.uint8)
                    frame = np.repeat(np.repeat(wrist, 2, axis=0), 2, axis=1)
                    viewer.set_images((mujoco.MjrRect(8, 8, frame.shape[1], frame.shape[0]), frame))
                lines = (f"{label}   {scenario.scenario_id}   step {action + 1}/{contract.max_actions}   lift {1000 * max_lift:4.1f} mm",
                         f"re-plans every {args.execute} steps (last camera read: step {last_read})   held-out eval recorded: {want}")
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, lines[0], lines[1]))
                if amp is not None:
                    with amp:
                        command = policy.predict(raw, current, adapter)
                else:
                    command = policy.predict(raw, current, adapter)
                command = adapter.apply_policy_command(command)
                adapter.advance_control_period()
                measurement, _ = adapter.pick_place_measurement(command, footprint_edge_margin_m=contract.placement.footprint_edge_margin_m)
                state, evaluation = evaluate_pick_place_step(contract, measurement, state)
                viewer.sync()
                max_lift = max(max_lift, measurement.pickup.cube_height_gain_m)
                if recorder is not None:
                    with viewer.lock():
                        cam = mujoco.MjvCamera()
                        cam.type, cam.lookat[:], cam.distance = viewer.cam.type, viewer.cam.lookat, viewer.cam.distance
                        cam.azimuth, cam.elevation = viewer.cam.azimuth, viewer.cam.elevation
                    recorder.frame(adapter.data, cam, wrist, lines)
                for event in list(evaluation.pickup_events) + list(evaluation.events):
                    print(f"  step {action + 1:3d}: {event.value}", flush=True)
                if evaluation.terminated or evaluation.truncated:
                    break
                budget = (1.0 / 30.0) / max(args.speed, 1e-3) - (time.time() - started)
                if budget > 0:
                    time.sleep(budget)
            if evaluation is not None:
                outcome = "SUCCESS" if evaluation.success and not evaluation.invalidated else (
                    "safety invalidation" if evaluation.invalidated else "failed")
                live[scenario.scenario_id] = outcome
                print(f"  -> {outcome}; max lift {1000 * max_lift:.1f} mm", flush=True)
                if recorder is not None:  # hold the last frame for a second with the verdict on it
                    for _ in range(30):
                        recorder.frame(adapter.data, cam, wrist, (lines[0], f"-> {outcome}   (held-out eval recorded: {want})"))
            end = time.time()
            while viewer.is_running() and time.time() - end < 1.5:
                viewer.sync(); time.sleep(0.05)
    if recorder is not None:
        recorder.close()
        print(f"video: {args.record} ({recorder.frames} frames)", flush=True)
    if args.summary:
        args.summary.write_text(json.dumps(dict(label=label, checkpoint=str(args.checkpoint), execute=args.execute, live=live,
                                                recorded=recorded, successes=sum(v == "SUCCESS" for v in live.values()),
                                                poses=len(live)), indent=1))
    adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
