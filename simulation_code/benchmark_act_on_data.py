#!/usr/bin/env python3
"""Benchmark ACT offline training throughput for this SO-101 dataset."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "train" / "act_benchmarks"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_ROOT / "benchmark_summary.json"
TRAIN_SCRIPT = SCRIPT_DIR / "train_act_on_data.py"

METRIC_RE = re.compile(
    r"step:(?P<step>\S+)\s+smpl:(?P<samples>\S+).*?"
    r"loss:(?P<loss>[-+0-9.eE]+).*?"
    r"updt_s:(?P<update_s>[-+0-9.eE]+)\s+data_s:(?P<data_s>[-+0-9.eE]+)"
)


@dataclass
class TrialResult:
    batch_size: int
    num_workers: int
    steps: int
    returncode: int
    elapsed_s: float
    samples_per_s: float
    update_s: float | None
    data_s: float | None
    max_gpu_mem_mib: int | None
    status: str
    output_dir: str
    log_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ACT training throughput")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64, 96, 128])
    parser.add_argument("--worker-counts", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-outputs", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str], log_path: Path) -> tuple[int, float]:
    start = time.perf_counter()
    with log_path.open("w") as log_file:
        proc = subprocess.run(cmd, cwd=SCRIPT_DIR, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    return proc.returncode, time.perf_counter() - start


def _max_gpu_mem_mib() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return max(int(line.strip()) for line in result.stdout.splitlines() if line.strip())
    except (OSError, ValueError):
        return None
    return None


def _parse_log(log_path: Path) -> tuple[float | None, float | None, str]:
    text = log_path.read_text(errors="replace")
    status = "ok"
    lowered = text.lower()
    if "out of memory" in lowered or "cuda error: out of memory" in lowered:
        status = "oom"
    elif "operation not supported" in lowered or "dataloader worker" in lowered:
        status = "worker_failed"
    elif "traceback" in lowered or "error" in lowered:
        status = "failed"

    matches = list(METRIC_RE.finditer(text))
    if not matches:
        return None, None, status
    last = matches[-1]
    return float(last.group("update_s")), float(last.group("data_s")), status


def run_trial(batch_size: int, num_workers: int, steps: int, output_root: Path, device: str) -> TrialResult:
    trial_name = f"bs{batch_size}_nw{num_workers}"
    output_dir = output_root / trial_name
    log_path = output_root / f"{trial_name}.log"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--performance-profile",
        "fast",
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--output-dir",
        str(output_dir),
        "--job-name",
        f"act_benchmark_{trial_name}",
        "--device",
        device,
        "--no-wandb",
        "--log-freq",
        str(steps),
        "--eval-freq",
        "0",
    ]

    before_mem = _max_gpu_mem_mib()
    returncode, elapsed_s = _run(cmd, log_path)
    after_mem = _max_gpu_mem_mib()
    update_s, data_s, status = _parse_log(log_path)
    if returncode != 0 and status == "ok":
        status = "failed"

    samples_per_s = 0.0
    if returncode == 0 and elapsed_s > 0:
        samples_per_s = (batch_size * steps) / elapsed_s

    max_mem = None
    if before_mem is not None or after_mem is not None:
        max_mem = max(value for value in (before_mem, after_mem) if value is not None)

    return TrialResult(
        batch_size=batch_size,
        num_workers=num_workers,
        steps=steps,
        returncode=returncode,
        elapsed_s=elapsed_s,
        samples_per_s=samples_per_s,
        update_s=update_s,
        data_s=data_s,
        max_gpu_mem_mib=max_mem,
        status=status,
        output_dir=str(output_dir),
        log_path=str(log_path),
    )


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.summary_path = args.summary_path.resolve()
    results: list[TrialResult] = []

    best_batch: int | None = None
    for batch_size in args.batch_sizes:
        result = run_trial(batch_size, 0, args.steps, args.output_root, args.device)
        results.append(result)
        print(
            f"batch={batch_size} workers=0 status={result.status} "
            f"samples_s={result.samples_per_s:.1f} elapsed_s={result.elapsed_s:.1f}",
            flush=True,
        )
        if result.status == "oom":
            break
        if result.returncode == 0:
            best_batch = batch_size

    if best_batch is not None:
        for workers in args.worker_counts:
            result = run_trial(best_batch, workers, args.steps, args.output_root, args.device)
            results.append(result)
            print(
                f"batch={best_batch} workers={workers} status={result.status} "
                f"samples_s={result.samples_per_s:.1f} elapsed_s={result.elapsed_s:.1f}",
                flush=True,
            )

    successful = [r for r in results if r.returncode == 0]
    successful.sort(key=lambda r: r.samples_per_s, reverse=True)
    summary = {
        "best": asdict(successful[0]) if successful else None,
        "results": [asdict(result) for result in results],
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    if not args.keep_outputs:
        for result in results:
            path = Path(result.output_dir)
            if path.exists():
                shutil.rmtree(path)

    print(f"summary={args.summary_path}")
    if successful:
        best = successful[0]
        print(
            f"best batch={best.batch_size} workers={best.num_workers} "
            f"samples_s={best.samples_per_s:.1f} elapsed_s={best.elapsed_s:.1f}"
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
