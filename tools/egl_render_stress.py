"""Stress-test the RL workers' wrist render on WSL: GPU (EGL via Mesa d3d12) vs Mesa's software rasteriser (2026-11-06).

October's RL runs switched the env workers to software rendering because "WSL's d3d12 EGL path hung inside renders
every few minutes with 16-20 contexts beside pi0.5" (commit 2c8573f). This reproduces the workers' render pattern —
N processes, each with its own adapter and EGL context, rendering the lens wrist observation between control periods —
optionally beside a CUDA load standing in for the policy, and records per-render times and every stall. A worker
whose render takes longer than --stall-s dumps its Python stack (faulthandler) into the output directory, which shows
where it is stuck (eglMakeCurrent, the render, the pixel read-back).

  PYTHONNOUSERSITE=1 MUJOCO_GL=egl ~/miniforge3/envs/lerobot/bin/python tools/egl_render_stress.py \\
      --render gpu --workers 20 --cuda-load act --seconds 300 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_SCENE = ROOT / "simulation_code/model/bench_pick_replace_v1_servo/scene_bench_pick_replace_v1.xml"
SOFTWARE_RENDER_ENV = {"GALLIUM_DRIVER": "llvmpipe", "LIBGL_ALWAYS_SOFTWARE": "1", "LP_NUM_THREADS": "1"}


def _worker(index: int, scene: str, render: str, seconds: float, periods_per_frame: int, stall_s: float, out: str, beats, frames):
    os.environ.setdefault("MUJOCO_GL", "egl")
    if render == "cpu":
        os.environ.update(SOFTWARE_RENDER_ENV)
    import faulthandler
    import numpy as np
    stack_file = open(Path(out) / f"stack_worker{index:02d}.txt", "w")
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import certification_suite
    adapter = MujocoTaskAdapter(scene)
    scenarios = certification_suite(adapter.bench, repeats=1).scenarios
    times = []
    end = time.time() + seconds
    k = 0
    while time.time() < end:
        adapter.reset(scenarios[k % len(scenarios)])
        for _ in range(480 // periods_per_frame):
            hold = adapter.current_act()
            for _ in range(periods_per_frame):
                adapter.apply_policy_command(hold)
                adapter.advance_control_period()
            faulthandler.dump_traceback_later(stall_s, repeat=False, file=stack_file)
            t0 = time.perf_counter()
            adapter.render_wrist_observation()
            times.append(time.perf_counter() - t0)
            faulthandler.cancel_dump_traceback_later()
            beats[index] = time.time()
            frames[index] += 1
            if time.time() >= end:
                break
        k += 1
    adapter.close()
    t = np.asarray(times)
    (Path(out) / f"worker{index:02d}.json").write_text(json.dumps(dict(
        frames=len(t), median_ms=float(np.median(t) * 1000), p95_ms=float(np.quantile(t, 0.95) * 1000),
        max_ms=float(t.max() * 1000), over_1s=int((t > 1.0).sum()), over_stall=int((t > stall_s).sum()))))


def _cuda_load(kind: str, seconds: float):
    """Stand-in for the policy on the GPU: ACT-sized inference batches (or a heavy matmul loop)."""
    import torch
    device = torch.device("cuda")
    end = time.time() + seconds
    if kind == "matmul":
        a = torch.randn(4096, 4096, device=device)
        while time.time() < end:
            (a @ a).sum().item()
        return
    net = torch.nn.Sequential(*[torch.nn.Linear(512, 512) for _ in range(12)]).to(device)
    images = torch.randn(20, 3, 256, 256, device=device)
    conv = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 7, 2, 3), torch.nn.ReLU(), torch.nn.Conv2d(64, 128, 3, 2, 1),
                               torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(128, 512)).to(device)
    with torch.no_grad():
        while time.time() < end:
            net(conv(images)).sum().item()
            time.sleep(0.05)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    p.add_argument("--render", choices=("gpu", "cpu"), required=True)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--periods-per-frame", type=int, default=90, help="control periods simulated between renders (RL: one frame per 90-step chunk)")
    p.add_argument("--cuda-load", choices=("none", "act", "matmul"), default="none")
    p.add_argument("--stall-s", type=float, default=20.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    beats = ctx.Array("d", [time.time()] * args.workers, lock=False)
    frames = ctx.Array("i", [0] * args.workers, lock=False)
    procs = [ctx.Process(target=_worker, args=(i, str(args.scene), args.render, args.seconds, args.periods_per_frame, args.stall_s,
                                                str(args.out), beats, frames)) for i in range(args.workers)]
    load = ctx.Process(target=_cuda_load, args=(args.cuda_load, args.seconds + 30)) if args.cuda_load != "none" else None
    started = time.time()
    if load:
        load.start()
    for proc in procs:
        proc.start()
    stalls, longest = [], [0.0] * args.workers
    while any(proc.is_alive() for proc in procs):
        time.sleep(1.0)
        now = time.time()
        for i, proc in enumerate(procs):
            if proc.is_alive() and frames[i] > 0:
                gap = now - beats[i]
                longest[i] = max(longest[i], gap)
        if now - started > args.seconds + 300:      # a worker hung for good: record and stop
            stalls += [i for i, proc in enumerate(procs) if proc.is_alive()]
            for proc in procs:
                proc.kill()
            break
    if load:
        load.kill()
    workers = {}
    for i in range(args.workers):
        path = args.out / f"worker{i:02d}.json"
        workers[i] = json.loads(path.read_text()) if path.exists() else {"status": "hung_or_crashed"}
    dumps = [f.name for f in args.out.glob("stack_worker*.txt") if f.stat().st_size > 0]
    total = sum(w.get("frames", 0) for w in workers.values())
    summary = dict(render=args.render, workers=args.workers, cuda_load=args.cuda_load, seconds=args.seconds, frames=total,
                   frames_per_s=round(total / args.seconds, 2),
                   median_ms=sorted(w["median_ms"] for w in workers.values() if "median_ms" in w)[len(workers) // 2] if total else None,
                   max_ms=max((w.get("max_ms", 0) for w in workers.values()), default=None),
                   renders_over_1s=sum(w.get("over_1s", 0) for w in workers.values()),
                   renders_over_stall=sum(w.get("over_stall", 0) for w in workers.values()),
                   hung_workers=stalls, longest_gap_s=round(max(longest), 1), stack_dumps=dumps)
    (args.out / "summary.json").write_text(json.dumps(dict(summary=summary, workers=workers), indent=2) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
