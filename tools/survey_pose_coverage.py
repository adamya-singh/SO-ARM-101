"""Which part of the placement rectangle can the wrist camera see from the reset pose and from the survey (viewing) pose?

Read-only: forward kinematics of the simulated arm at the two poses, the
calibrated lens, and the four top-face corners of a cube at each sample point
of the rectangle (`so_arm101_v2.physical.placement.cube_visible_at`). Writes a
JSON record and a coverage PNG under the inspection directory for the review
record. Nothing touches hardware.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash  # noqa: E402
from so_arm101_v2.contracts.placement import PlacementRegime  # noqa: E402
from so_arm101_v2.physical.placement import cube_visible_at  # noqa: E402

DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


def coverage(model_path, bench, regime: PlacementRegime, qpos, *, nx: int, ny: int, yaws=(0.0,)) -> dict:
    xs = np.linspace(regime.x_range_m[0], regime.x_range_m[1], nx)
    ys = np.linspace(regime.y_range_m[0], regime.y_range_m[1], ny)
    grid = np.zeros((ny, nx), dtype=bool)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            grid[j, i] = all(cube_visible_at(model_path, bench, qpos, bench.cube_center_at((x, y)), yaw_rad=yaw,
                                             margin_px=regime.survey_visibility_margin_px)["visible"] for yaw in yaws)
    return dict(xs=[round(float(v), 4) for v in xs], ys=[round(float(v), 4) for v in ys], visible=grid.tolist(),
                visible_fraction=round(float(grid.mean()), 4), visible_count=int(grid.sum()), total=int(grid.size))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--grid", type=int, nargs=2, default=(15, 11), metavar=("NX", "NY"))
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/inspection")
    args = p.parse_args(argv)
    bench = scene_bench_config(args.model)
    regime = bench.placement_regime or PlacementRegime()
    record = dict(model=str(args.model), scene_dependencies_sha256=scene_dependency_hash(args.model), regime=regime.identity(),
                  reset_qpos=[float(v) for v in bench.reset_qpos], viewing_qpos=[float(v) for v in bench.viewing_qpos],
                  survey_pose_differs_from_reset=bool(np.max(np.abs(np.asarray(bench.viewing_qpos) - np.asarray(bench.reset_qpos))) > 1e-6),
                  poses={})
    for name, qpos in (("reset", bench.reset_qpos), ("survey", bench.viewing_qpos)):
        record["poses"][name] = coverage(args.model, bench, regime, qpos, nx=args.grid[0], ny=args.grid[1], yaws=(0.0, np.deg2rad(45.0)))
        print(f"{name}: {record['poses'][name]['visible_count']}/{record['poses'][name]['total']} rectangle points visible (yaw 0 and 45 deg)", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = record["scene_dependencies_sha256"][:8]
    (args.output_dir / f"survey_pose_coverage_{tag}.json").write_text(json.dumps(record, indent=2) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, name in zip(axes, ("reset", "survey")):
            g = record["poses"][name]
            ax.imshow(np.array(g["visible"]), origin="lower", extent=(g["xs"][0] * 1000, g["xs"][-1] * 1000, g["ys"][0] * 1000, g["ys"][-1] * 1000),
                      cmap="Greens", vmin=0, vmax=1, aspect="equal")
            ax.set_title(f"{name} pose: {g['visible_count']}/{g['total']} visible")
            ax.set_xlabel("x (mm, lateral)"); ax.set_ylabel("y (mm, forward of world origin)")
            ax.plot([bench.square_center_xy[0] * 1000], [bench.square_center_xy[1] * 1000], "r+", ms=12)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"survey_pose_coverage_{tag}.png", dpi=110)
    except ImportError:
        pass
    print(json.dumps({k: v for k, v in record.items() if k != "poses"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
