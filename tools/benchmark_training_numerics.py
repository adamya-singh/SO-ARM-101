"""Benchmark the chunked training step across numerics regimes.

Measures steps/second for {cpu-eager, cpu-compile, gpu-eager, gpu-compile}
x {width 256, 512} on a real capture manifest, using the production
noise-penalty recipe.  Results are wall-clock telemetry, NOT evidence
artifacts: JSON goes to outputs/benchmarks/, never artifacts/.

Usage:
    PYTHONNOUSERSITE=1 python tools/benchmark_training_numerics.py \
        --manifest artifacts/.../oracle/fixed_pick_place_v3/<digest>/manifest.json \
        --json outputs/benchmarks/numerics_regime_v2/<name>.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--measure-steps", type=int, default=200)
    parser.add_argument("--widths", type=int, nargs="+", default=[256, 512])
    parser.add_argument(
        "--configs", nargs="+",
        default=["cpu-eager", "cpu-compile", "gpu-eager", "gpu-compile"],
    )
    return parser.parse_args()


def _spec_for(config_name: str):
    from so_arm101_v2.learning.numerics import PINNED_NUMERICS_V2, resolve_numerics

    if config_name == "cpu-eager":
        return None
    if config_name == "cpu-compile":
        return resolve_numerics("cpu", compile=True)
    if config_name == "gpu-eager":
        return replace(PINNED_NUMERICS_V2, compile=None)
    if config_name == "gpu-compile":
        return PINNED_NUMERICS_V2
    raise ValueError(config_name)


def _measure(manifest_path: Path, width: int, config_name: str,
             warmup_steps: int, measure_steps: int) -> dict:
    import torch

    from so_arm101_v2.learning.chunked import (
        ChunkedCloneConfig,
        _chunked_step_loss,
        _feasible_decode_constants,
        build_chunked_clone_model,
        build_chunked_targets,
    )
    from so_arm101_v2.learning.numerics import apply_numerics, noise_generator
    from so_arm101_v2.learning.oracle_distillation import (
        OracleCloneKind,
        build_oracle_features,
    )
    from so_arm101_v2.simulation.oracle import load_oracle_demonstrations
    from so_arm101_v2.learning.tiny_model import normalize_act

    spec = _spec_for(config_name)
    if spec is not None and spec.device == "cuda" and not torch.cuda.is_available():
        return {"config": config_name, "width": width, "skipped": "no cuda"}

    config = ChunkedCloneConfig(
        chunk_horizon=90, hidden_width=width,
        saturation_mode="noise_penalty_v1",
        decoder_eta=1.0, margin_act=0.002, noise_sigma=0.05, penalty_weight=1.0,
    )
    manifest, arrays = load_oracle_demonstrations(manifest_path)
    episode_lengths = [int(item["rows"]) for item in manifest["episodes"]]
    _, extras_mean, extras_std = build_oracle_features(OracleCloneKind.PHASE_STATE, arrays)
    features_np, _, _ = build_oracle_features(
        OracleCloneKind.PHASE_STATE, arrays,
        extras_mean=extras_mean, extras_std=extras_std,
    )
    targets_np = build_chunked_targets(arrays, 90, episode_lengths=episode_lengths)
    rows = int(features_np.shape[0])

    device = apply_numerics(torch, spec, seed=config.seed)
    features = torch.from_numpy(features_np).to(device)
    targets = torch.from_numpy(targets_np.reshape(rows, -1)).to(device)
    current_norm = torch.from_numpy(
        normalize_act(np.asarray(arrays["current_act"], dtype=np.float32))
    ).to(device)
    model = build_chunked_clone_model(features_np.shape[1], width, 90).to(device)
    low_norm, high_norm, delta_norm = _feasible_decode_constants(config.margin_act)
    penalty_constants = (
        torch.from_numpy(low_norm).to(device),
        torch.from_numpy(high_norm).to(device),
        (torch.from_numpy(delta_norm) * float(config.decoder_eta)).to(device),
    )
    step_loss = _chunked_step_loss
    if spec is not None and spec.compile == "inductor":
        step_loss = torch.compile(
            _chunked_step_loss, mode=spec.compile_mode, fullgraph=True, dynamic=False,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = noise_generator(torch, config.seed)

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(features.shape, generator=generator)
        noisy = features + noise.to(device) * float(config.noise_sigma)
        loss, _ = step_loss(
            model, features, targets, current_norm, noisy,
            None, penalty_constants, float(config.penalty_weight), 90, rows,
        )
        loss.backward()
        optimizer.step()

    compile_start = time.perf_counter()
    for _ in range(warmup_steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - compile_start

    start = time.perf_counter()
    for _ in range(measure_steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    seconds_per_step = elapsed / measure_steps
    return {
        "config": config_name,
        "width": width,
        "rows": rows,
        "regime": spec.regime if spec is not None else "legacy_cpu_eager",
        "warmup_steps": warmup_steps,
        "warmup_seconds": round(warmup_seconds, 3),
        "measured_steps": measure_steps,
        "seconds_per_step": seconds_per_step,
        "steps_per_second": 1.0 / seconds_per_step,
        "extrapolated_30k_minutes": seconds_per_step * 30_000 / 60,
        "extrapolated_90k_minutes": seconds_per_step * 90_000 / 60,
    }


def main() -> int:
    args = _parse()
    results = []
    for config_name in args.configs:
        for width in args.widths:
            print(f"measuring {config_name} width={width} ...", flush=True)
            row = _measure(args.manifest, width, config_name,
                           args.warmup_steps, args.measure_steps)
            results.append(row)
            if "seconds_per_step" in row:
                print(f"  {row['seconds_per_step']*1000:.2f} ms/step "
                      f"({row['steps_per_second']:.1f} steps/s; "
                      f"90k ~ {row['extrapolated_90k_minutes']:.1f} min)", flush=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "benchmark": "chunked_noise_penalty_step",
        "manifest": str(args.manifest),
        "results": results,
    }, indent=2))
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
