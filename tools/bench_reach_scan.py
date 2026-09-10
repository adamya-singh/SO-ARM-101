"""Where in the placement rectangle can the teacher complete the task? A grid of full teacher episodes (read-only).

For every grid point (square centre) and yaw, the square, towel and cube are
placed there and the privileged teacher runs one full episode through the
same admission path as suite generation (`screen_scenario`). Writes a JSON
record and a reach-map PNG under the inspection directory. Nothing touches
hardware. Used to justify the certification placements and any teacher change.
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

from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash  # noqa: E402
from so_arm101_v2.contracts.placement import PlacementRegime  # noqa: E402
from so_arm101_v2.simulation.bench import bench_suite, screen_scenario  # noqa: E402

DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--grid", type=int, nargs=2, default=(7, 5), metavar=("NX", "NY"))
    p.add_argument("--yaws-deg", type=float, nargs="+", default=[0.0, 45.0, -45.0])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/inspection")
    p.add_argument("--tag", default=None)
    args = p.parse_args(argv)
    bench = scene_bench_config(args.model)
    if bench.placement is None:
        raise SystemExit("the bench config needs a placement regime (tools/prepare_bench_scene.py --placement default)")
    regime = PlacementRegime.from_mapping(bench.placement)
    xs = np.linspace(regime.x_range_m[0], regime.x_range_m[1], args.grid[0])
    ys = np.linspace(regime.y_range_m[0], regime.y_range_m[1], args.grid[1])
    points = [(float(x), float(y), float(np.deg2rad(w))) for w in args.yaws_deg for y in ys for x in xs]
    suite = bench_suite(bench, [(0, 0)] * len(points), label="reach_scan", repeats=1,
                        square_centers=[(x, y) for x, y, _ in points], square_yaws=[w for _, _, w in points])
    started = time.time()
    outcomes = {}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(screen_scenario, str(args.model), s): (pt, s) for pt, s in zip(points, suite.scenarios)}
        for future in futures:
            (x, y, w), scenario = futures[future]
            ok, reason = future.result()
            outcomes[scenario.scenario_id] = dict(x=round(x, 4), y=round(y, 4), yaw_deg=round(float(np.degrees(w)), 1), ok=bool(ok), reason=reason)
            print(f"x={x:+.3f} y={y:.3f} yaw={np.degrees(w):+.0f}: {'ok' if ok else reason[:60]}", flush=True)
    record = dict(model=str(args.model), scene_dependencies_sha256=scene_dependency_hash(args.model), regime=regime.identity(),
                  grid=dict(xs=[round(float(v), 4) for v in xs], ys=[round(float(v), 4) for v in ys], yaws_deg=args.yaws_deg),
                  outcomes=outcomes, seconds=round(time.time() - started, 1),
                  reachable_fraction={str(w): round(float(np.mean([o["ok"] for o in outcomes.values() if o["yaw_deg"] == round(w, 1)])), 3) for w in args.yaws_deg})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or record["scene_dependencies_sha256"][:8]
    (args.output_dir / f"reach_scan_{tag}.json").write_text(json.dumps(record, indent=2) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(args.yaws_deg), figsize=(4.5 * len(args.yaws_deg), 4.5), squeeze=False)
        for ax, w in zip(axes[0], args.yaws_deg):
            grid = np.array([[outcomes[s.scenario_id]["ok"] for s in suite.scenarios if outcomes[s.scenario_id]["yaw_deg"] == round(w, 1) and abs(outcomes[s.scenario_id]["y"] - round(float(y), 4)) < 1e-6]
                             for y in ys], dtype=float)
            ax.imshow(grid, origin="lower", extent=(xs[0] * 1000, xs[-1] * 1000, ys[0] * 1000, ys[-1] * 1000), cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
            ax.set_title(f"yaw {w:+.0f} deg: {record['reachable_fraction'][str(w)]:.0%} reachable")
            ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
            ax.plot([bench.square_center_xy[0] * 1000], [bench.square_center_xy[1] * 1000], "b+", ms=12)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"reach_scan_{tag}.png", dpi=110)
    except ImportError:
        pass
    print(json.dumps({k: v for k, v in record.items() if k != "outcomes"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
