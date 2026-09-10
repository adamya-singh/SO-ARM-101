"""Train-versus-held-out loss of a vision checkpoint at the chunk boundaries (the generalisation diagnostic of 2026-09-10).

The closed-loop evaluation says whether a policy succeeds; this says why a
policy that fits its training capture fails on new placements. For a
checkpoint and two frames captures (its own training capture and a capture
of unseen placements, e.g. the held-out suite run through the teacher with
``store_frames``), it reports the mean chunk-target MSE per chunk start:
start 0 (the survey move), start 90 (the first descent chunk, the frame the
policy must read the square's position from), later boundaries, and all rows.
A held-out / train ratio far above 1 at start 90 is memorisation of the
training placements; both high and close together is capacity or ambiguity.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so_arm101_v2.learning.chunked import build_chunked_targets  # noqa: E402
from so_arm101_v2.learning.tiny_model import normalize_act  # noqa: E402
from so_arm101_v2.learning.vision import build_vision_chunked_model, load_vision_frames  # noqa: E402

BUCKETS = {
    "start_0": lambda ai: ai == 0,
    "start_90": lambda ai: ai == 90,
    "starts_180_270_360": lambda ai: (ai % 90 == 0) & (ai >= 180) & (ai <= 360),
    "all_rows": lambda ai: np.ones_like(ai, dtype=bool),
}


def boundary_losses(model, manifest_path, *, device, samples: int = 600, seed: int = 0, episode_limit: int | None = None) -> dict[str, float]:
    """Mean chunk-target MSE per chunk-start bucket; ``episode_limit`` scores only the first N episodes (a training prefix)."""
    import torch
    manifest, frames, arrays = load_vision_frames(manifest_path)
    episode_lengths = [int(e["rows"]) for e in manifest["episodes"]]
    targets = build_chunked_targets(arrays, model.chunk_horizon, episode_lengths=episode_lengths).reshape(frames.shape[0], -1)
    state = np.concatenate([normalize_act(np.asarray(arrays["current_act"], np.float32)),
                            np.asarray(arrays["progress"], np.float32)[:, None]], axis=1).astype(np.float32)
    action_index = np.asarray(arrays["action_index"])
    within = np.ones(frames.shape[0], dtype=bool)
    if episode_limit is not None:
        within = np.arange(frames.shape[0]) < int(sum(episode_lengths[:int(episode_limit)]))
    rng = np.random.default_rng(seed)
    out = {}
    with torch.inference_mode():
        for name, predicate in BUCKETS.items():
            index = np.flatnonzero(predicate(action_index) & within)
            pick = np.sort(rng.choice(index, min(samples, index.shape[0]), replace=False))
            images = torch.from_numpy(np.ascontiguousarray(frames[pick])).to(device).permute(0, 3, 1, 2).float().div_(255.0)
            prediction = model(images, torch.from_numpy(state[pick]).to(device)).cpu().numpy()
            out[name] = float(((prediction - targets[pick]) ** 2).mean())
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, default=None, help="default: the capture manifest under the checkpoint's experiment directory")
    p.add_argument("--heldout-manifest", type=Path, required=True, help="a frames capture of unseen placements (teacher, store_frames)")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)
    import torch
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_vision_chunked_model(int(checkpoint["hidden_width"]), int(checkpoint["chunk_horizon"]), encoder=str(checkpoint.get("encoder", "v1")))
    model.load_state_dict(checkpoint["state_dict"])
    model.chunk_horizon = int(checkpoint["chunk_horizon"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    train_manifest = args.train_manifest
    if train_manifest is None:
        experiment = args.checkpoint.resolve().parents[3]
        candidates = glob.glob(str(experiment / "capture/oracle/*/*/manifest.json"))
        if len(candidates) != 1:
            raise SystemExit(f"could not find exactly one capture manifest under {experiment}; pass --train-manifest")
        train_manifest = Path(candidates[0])
    train = boundary_losses(model, train_manifest, device=device)
    heldout = boundary_losses(model, args.heldout_manifest, device=device)
    record = dict(checkpoint=str(args.checkpoint), train_manifest=str(train_manifest), heldout_manifest=str(args.heldout_manifest),
                  parameters=int(sum(p.numel() for p in model.parameters())), encoder=str(checkpoint.get("encoder", "v1")),
                  train=train, heldout=heldout, ratio={k: round(heldout[k] / max(train[k], 1e-12), 1) for k in train})
    for k in train:
        print(f"{k:20s} train {train[k]:.2e}  held-out {heldout[k]:.2e}  ratio {record['ratio'][k]:.1f}x")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
