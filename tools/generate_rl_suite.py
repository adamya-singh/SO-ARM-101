"""Screen the RL training and monitoring suites in parallel and write their index (retry tranche, 2026-10-04).

One process per seed: training shards on seeds 30-37 (--per-shard poses each) and the monitoring suite on
seed 21 (--monitor poses), all with appearance and placement randomization, every pose admitted only if the
privileged teacher completes the full episode and the cube is visible from the survey pose (the capture
rules). The index asserts the pools are disjoint from the held-out suite (seed 8) and the two captures'
suites (seeds 12, 14).

  PYTHONNOUSERSITE=1 MUJOCO_GL=egl ~/miniforge3/envs/lerobot/bin/python tools/generate_rl_suite.py \
      --output-dir artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/retry_tranche_20261004/rl_suites
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
EXPERIMENTS = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"


def _screen(job):
    model, seed, count, output_dir = job
    from so_arm101_v2.contracts.bench import scene_bench_config
    from so_arm101_v2.simulation.bench import generate_bench_suite
    config = scene_bench_config(Path(model))
    started = time.time()
    _, path = generate_bench_suite(Path(model), config, seed=seed, count=count, repeats=1, output_dir=Path(output_dir),
                                   randomize_appearance=True, randomize_placement=True)
    return seed, str(path), round(time.time() - started, 1)


def main(argv=None) -> int:
    from so_arm101_v2.data._serialization import write_immutable_json
    from so_arm101_v2.rl.scenarios import MONITOR_SEED, ORDER_SEED, TRAIN_SEEDS, ScenarioPool, assert_disjoint
    from so_arm101_v2.simulation.suites import load_suite_from_path

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--per-shard", type=int, default=200)
    p.add_argument("--shards", type=int, default=len(TRAIN_SEEDS))
    p.add_argument("--monitor", type=int, default=20)
    p.add_argument("--workers", type=int, default=9)
    p.add_argument("--reserved-suites", type=Path, nargs="*", default=[
        EXPERIMENTS / "scaling_ladder_20260910/heldout_suite.path", EXPERIMENTS / "scaling_ladder_20260910/train_suite.path",
        EXPERIMENTS / "placements4800_20260912/train_suite.path"])
    args = p.parse_args(argv)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(args.model), seed, args.per_shard, str(out)) for seed in TRAIN_SEEDS[: args.shards]]
    jobs.append((str(args.model), MONITOR_SEED, args.monitor, str(out)))
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for seed, path, seconds in pool.map(_screen, jobs):
            results[seed] = path
            print(f"RL SUITE seed {seed}: {path} ({seconds} s)", flush=True)
    train_paths = [results[s] for s in TRAIN_SEEDS[: args.shards]]
    reserved = []
    for pointer in args.reserved_suites:
        target = Path(pointer.read_text().strip()) if pointer.suffix == ".path" else pointer
        reserved.extend(load_suite_from_path(target).scenarios)
    train = [s for path in train_paths for s in load_suite_from_path(path).scenarios]
    monitor = list(load_suite_from_path(results[MONITOR_SEED]).scenarios)
    assert_disjoint(train, reserved + monitor)
    assert_disjoint(monitor, reserved)
    pool_ = ScenarioPool(train)
    index = dict(train_suites=train_paths, monitor_suite=results[MONITOR_SEED], order_seed=ORDER_SEED, order_sha256=pool_.order_sha256,
                 train_poses=len(pool_), monitor_poses=len(monitor), reserved_suites=[str(p) for p in args.reserved_suites],
                 reserved_poses=len(reserved), model=str(args.model))
    write_immutable_json(out / "rl_suites.json", index)
    print(json.dumps({k: v for k, v in index.items() if k not in ("train_suites",)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
