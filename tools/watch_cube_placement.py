"""Where is the cube on the bench right now? Read-only: one camera frame + the measured joints, back-projected.

Reads the six joints over the bus (read-only, torque untouched), grabs one
wrist-camera frame, resamples it as the policy would, finds the white towel
and the dark cube in it, and back-projects the cube's top face through the
calibrated lens and the simulated camera pose at the measured joints onto
the bench plane. Prints the cube's bench coordinates and its offset from the
task pose (square centre 8.5 in forward of the base front edge). With
--until-within it polls until the cube is inside that tolerance (exit 0) so
a trial can start automatically after the bench is fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so_arm101_v2.contracts.bench import scene_bench_config  # noqa: E402
from so_arm101_v2.contracts.physical_io import connect_read_only, disconnect_read_only, read_measured_act  # noqa: E402
from so_arm101_v2.physical.camera import FrameGrabber  # noqa: E402
from so_arm101_v2.physical.lerobot_backend import fast_area_resampler  # noqa: E402
from so_arm101_v2.physical.placement import check_cube_placement  # noqa: E402

DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


def measure(model_path, port, camera_device, camera_size, tolerance_mm) -> tuple[dict, np.ndarray]:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    bench = scene_bench_config(model_path)
    lens = bench.lens_model
    robot = SO101Follower(SO101FollowerConfig(id="None", port=port, cameras={}, max_relative_target=20.0, use_degrees=False))
    connect_read_only(robot)
    try:
        current = read_measured_act(robot)
    finally:
        disconnect_read_only(robot)
    grabber = FrameGrabber(camera_device, width=int(camera_size[0]), height=int(camera_size[1]))
    try:
        seq = -1
        for _ in range(3):
            seq, _stamp, frame = grabber.wait_for_new(seq)
        observation = fast_area_resampler(lens, source_size=(frame.shape[1], frame.shape[0]))(np.array(frame, copy=True))
    finally:
        grabber.close()
    result = check_cube_placement(observation, bench, model_path, current, tolerance_mm=tolerance_mm)
    result.update(measured_act=[round(float(v), 4) for v in current], at=time.time())
    return result, observation


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--robot-port", default="/dev/ttyACM0")
    p.add_argument("--camera-device", type=int, default=0)
    p.add_argument("--camera-size", type=int, nargs=2, default=(1280, 720))
    p.add_argument("--until-within", type=float, default=None, help="poll until the cube is within this many mm of the task pose; exit 0 then")
    p.add_argument("--tolerance-mm", type=float, default=15.0)
    p.add_argument("--interval", type=float, default=45.0)
    p.add_argument("--save-observation", type=Path, default=None)
    args = p.parse_args(argv)
    while True:
        try:
            result, observation = measure(args.model, args.robot_port, args.camera_device, args.camera_size, args.tolerance_mm)
        except Exception as exc:  # the runner may hold the port/camera; report and keep polling
            result, observation = dict(found=False, reason=f"{type(exc).__name__}: {exc}", at=time.time()), None
        print(json.dumps(result), flush=True)
        if args.save_observation is not None and observation is not None:
            from PIL import Image
            Image.fromarray(observation).save(args.save_observation)
        if args.until_within is None:
            return 0 if result.get("found") else 1
        if result.get("found") and result.get("ok") and result["distance_mm"] <= args.until_within:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
