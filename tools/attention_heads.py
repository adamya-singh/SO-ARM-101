"""What each attention head of an ACT policy does (2026-11-25, notes/attention-heads-20261125.md).

Runs a finished ACT checkpoint on held-out frames (teacher-driven, open loop) and writes to --output-dir:

  heads.json   per head: where its attention goes (state token, style token, image patches, the napkin and cube),
               and how far the 90-step plan moves when the head's output is replaced by its mean over the frames
  block.json   the plan's error when attention may not read a group of tokens (state, style, napkin patches, ...)
  maps.npz     per-frame attention maps behind the figures
  overlay_*.png, dec_cross_bystep.png, attn_*.png   figures

The ACT has 48 heads: 4 encoder layers x 8, and one decoder layer with 8 self-attention and 8 cross-attention heads.
Encoder tokens are [style (the CVAE latent, zero at run time), state, 64 image patches in row order].

  PYTHONNOUSERSITE=1 PYTHONPATH=src:$HOME/projects/lerobot-fork/src ~/miniforge3/envs/lerobot/bin/python \\
      tools/attention_heads.py --output-dir artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/attention_heads_20261125
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EXPERIMENTS = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
DEFAULT_RUN = EXPERIMENTS / "high_lift_20261112/act_ft"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PHASES = ["survey 0-120", "descend 120-240", "close 240-300", "lift 300-360", "lower 360-420", "release 420-480"]
PHASE_EDGES = [120, 240, 300, 360, 420]
ENCODER = ["enc0", "enc1", "enc2", "enc3"]
READERS = ENCODER + ["dec_cross"]


class HookedMHA(nn.Module):
    """nn.MultiheadAttention (eval mode, batch_first=False) written out so its heads can be read, replaced and blinded."""

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.mha = mha
        self.heads = mha.num_heads
        self.dim = mha.embed_dim // mha.num_heads
        self.record = False
        self.attn: list[torch.Tensor] = []       # per batch (B, H, L, S)
        self.sum_out = None                      # (H, L, D) running sum of head outputs
        self.count = 0
        self.mean_out = None                     # (H, L, D)
        self.ablate: tuple[int, ...] = ()        # heads whose output is replaced by mean_out
        self.block = None                        # (B, S) bool: keys no query may read (the weights renormalise)

    def forward(self, query, key, value, key_padding_mask=None, **_):
        m, H, D = self.mha, self.heads, self.dim
        wq, wk, wv = m.in_proj_weight.chunk(3, dim=0)
        bq, bk, bv = m.in_proj_bias.chunk(3, dim=0)
        L, B, _ = query.shape
        S = key.shape[0]
        q = (query @ wq.T + bq).view(L, B, H, D).permute(1, 2, 0, 3)        # (B, H, L, D)
        k = (key @ wk.T + bk).view(S, B, H, D).permute(1, 2, 0, 3)
        v = (value @ wv.T + bv).view(S, B, H, D).permute(1, 2, 0, 3)
        logits = q @ k.transpose(-1, -2) / math.sqrt(D)
        if self.block is not None:
            logits = logits.masked_fill(self.block[:, None, None, :], float("-inf"))
        a = torch.softmax(logits, dim=-1)                                    # (B, H, L, S)
        out = a @ v                                                          # (B, H, L, D)
        if self.record:
            self.attn.append(a.float().cpu())
            s = out.sum(0)
            self.sum_out = s if self.sum_out is None else self.sum_out + s
            self.count += B
        if self.ablate:
            out = out.clone()
            for h in self.ablate:
                out[:, h] = self.mean_out[h]
        out = out.permute(2, 0, 1, 3).reshape(L, B, H * D)
        return m.out_proj(out), None


def hook_act(act: nn.Module) -> dict[str, HookedMHA]:
    """Swap every attention module of a lerobot ACT (not the training-only VAE encoder) for a HookedMHA."""
    if len(act.encoder.layers) != 4 or len(act.decoder.layers) != 1:
        raise ValueError("this analysis is written for the 4-layer encoder, 1-layer decoder ACT")
    hooks: dict[str, HookedMHA] = {}
    for i, layer in enumerate(act.encoder.layers):
        layer.self_attn = hooks[f"enc{i}"] = HookedMHA(layer.self_attn)
    dec = act.decoder.layers[0]
    dec.self_attn = hooks["dec_self"] = HookedMHA(dec.self_attn)
    dec.multihead_attn = hooks["dec_cross"] = HookedMHA(dec.multihead_attn)
    return hooks


def napkin_masks(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per 32-px patch (8, 8) area fractions: the napkin's convex hull (the target area) and the dark area inside it (the cube).

    The napkin is the largest bright region of a simulated frame (the ground is dark in every appearance draw).
    """
    import cv2
    lum = frame.astype(np.float32).mean(-1)
    top = np.percentile(lum, 99.5)
    bright = (lum > 0.45 * top).astype(np.uint8) if top > 40 else np.zeros(lum.shape, np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    hull = np.zeros_like(bright)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[big, cv2.CC_STAT_AREA] >= 60:
            pts = np.argwhere(lab == big)[:, ::-1].astype(np.int32)
            cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
            bright = (lab == big).astype(np.uint8)
    cube = hull & (1 - bright)
    pool = lambda m: m.reshape(8, 32, 8, 32).mean((1, 3)).astype(np.float32)
    return pool(hull), pool(cube)


def region_lift(weight: np.ndarray, region: np.ndarray) -> float:
    """Share of image attention on a region divided by the region's share of the image (1 = no preference)."""
    ok = region.sum(1) > 0
    if not ok.any():
        return float("nan")
    return float((weight[ok] * region[ok]).sum() / weight[ok].sum() / (region[ok].sum() / (64 * ok.sum())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN, help="a vla_finetune ACT run directory (identity.json, model/act.pt)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=6, help="use every stride-th held-out frame")
    ap.add_argument("--batch", type=int, default=120)
    ap.add_argument("--figure-frame", type=int, nargs=2, default=(0, 150), metavar=("EPISODE", "STEP"), help="frame of the attn_*.png strips")
    args = ap.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from so_arm101_v2.learning.pi05 import CaptureSource
    from so_arm101_v2.learning.vla_models import HORIZON, load_model

    identity = json.loads((args.run / "identity.json").read_text())
    source = CaptureSource.load(identity["heldout_manifest"], name="heldout")
    rows = source.rows[source.action_index[source.rows] % args.stride == 0]
    step = source.action_index[rows]
    episode = np.asarray(source.episode_of_row)[rows]
    frames, state = source.images(rows), source.state(rows)
    targets = source.targets(rows, HORIZON)                                   # (N, 90, 6) residual chunk, normalised-act units
    N = rows.shape[0]
    phase = np.digitize(step, PHASE_EDGES)
    var_joint = targets.reshape(-1, 6).var(0)
    print(f"{N} frames, {len(set(episode.tolist()))} episodes, stride {args.stride}", flush=True)

    model = load_model(args.run / "model/act.pt", "cuda")
    act = model.policy.model
    act.eval()

    hooks: dict[str, HookedMHA] = {}

    def predict(blocks: dict | None = None) -> np.ndarray:
        chunks = []
        with torch.no_grad():
            for i in range(0, N, args.batch):
                sl = slice(i, i + args.batch)
                for name, hook in hooks.items():
                    hook.block = blocks[name][sl] if blocks and name in blocks else None
                chunks.append(model.stats.unnorm_action(model.mean(frames[sl], state[sl]).float()).cpu().numpy())
        for hook in hooks.values():
            hook.block = None
        return np.concatenate(chunks)

    native = predict()
    hooks.update(hook_act(act))
    for hook in hooks.values():
        hook.record = True
    clean = predict()
    for hook in hooks.values():
        hook.record = False
        hook.mean_out = (hook.sum_out / hook.count).detach()
    parity = float(np.abs(clean - native).max())
    base = (clean - targets) ** 2                                             # (N, 90, 6)
    start90 = step == 90
    print(f"hooked vs native max abs diff {parity:.2e}; MSE all frames {base.mean():.3e}, chunk start 90 {base[start90].mean():.3e}", flush=True)

    def summarise(pred: np.ndarray) -> dict:
        e, move = (pred - targets) ** 2, (pred - clean) ** 2
        return dict(
            ratio=float(e.mean() / base.mean()),                              # error against the teacher, times the intact model's
            effect=float(np.mean([move[..., j].mean() / var_joint[j] for j in range(6)])),   # plan movement / plan variance, mean over joints
            ratio_by_joint=[float(e[..., j].mean() / base[..., j].mean()) for j in range(6)],
            ratio_by_phase=[float(e[phase == p].mean() / base[phase == p].mean()) for p in range(6)],
            ratio_by_third=[float(e[:, s:s + 30].mean() / base[:, s:s + 30].mean()) for s in (0, 30, 60)],
            effect_by_joint=[float(move[..., j].mean() / var_joint[j]) for j in range(6)],
            change_by_phase=[float(move[phase == p].mean()) for p in range(6)],
            change_by_third=[float(move[:, s:s + 30].mean()) for s in (0, 30, 60)])

    # ---------------------------------------------------------------- where the attention goes
    hull = np.zeros((N, 64), np.float32)
    cube = np.zeros((N, 64), np.float32)
    for i in range(N):
        a, b = napkin_masks(frames[i])
        hull[i], cube[i] = a.ravel(), b.ravel()
    yy, xx = np.divmod(np.arange(64), 8)
    dist = np.hypot(yy[:, None] - yy[None, :], xx[:, None] - xx[None, :]).astype(np.float32)

    heads: list[dict] = []
    maps: dict[str, np.ndarray] = {}
    for name, hook in hooks.items():
        A = torch.cat(hook.attn).numpy()                                      # (N, H, L, S)
        hook.attn = []
        mean = A.mean(0)
        overlap = np.minimum(A, mean[None]).sum(-1)                           # 1 = the same pattern in every frame
        spread = np.exp(-(A * np.log(A + 1e-12)).sum(-1))                     # effective number of tokens read
        for h in range(A.shape[1]):
            a = A[:, h]
            row = dict(module=name, head=h, consistency=float(overlap[:, h].mean()), effective_keys=float(spread[:, h].mean()))
            if name in ENCODER:
                img_q = a[:, 2:, :]                                           # what the image patches read
                ii = img_q[:, :, 2:]
                ii_n = ii / ii.sum(-1, keepdims=True)
                keyw = ii.mean(1)                                             # (N, 64) how much each patch is read
                row.update(img_to_style=float(img_q[:, :, 0].mean()), img_to_state=float(img_q[:, :, 1].mean()),
                           img_to_img=float(ii.sum(-1).mean()), state_to_img=float(a[:, 1, 2:].sum(-1).mean()),
                           style_to_img=float(a[:, 0, 2:].sum(-1).mean()), img_distance_patches=float((ii_n * dist[None]).sum(-1).mean()),
                           lift_napkin=region_lift(keyw, hull), lift_cube=region_lift(keyw, cube))
                maps[f"{name}_h{h}_keyw"] = a.mean(1)[:, 2:].astype(np.float16)
            elif name == "dec_cross":
                keyw = a[:, :, 2:].mean(1)
                thirds = [a[:, s:s + 30].mean(1) for s in (0, 30, 60)]        # per third of the plan
                row.update(to_style=float(a[:, :, 0].mean()), to_state=float(a[:, :, 1].mean()), to_img=float(a[:, :, 2:].sum(-1).mean()),
                           lift_napkin=region_lift(keyw, hull), lift_cube=region_lift(keyw, cube),
                           thirds_to_state=[float(t[:, 1].mean()) for t in thirds], thirds_to_style=[float(t[:, 0].mean()) for t in thirds])
                maps[f"{name}_h{h}_keyw"] = keyw.astype(np.float16)
                maps[f"{name}_h{h}_bystep"] = a.mean(0).astype(np.float32)   # (90, 66)
            heads.append(row)
        del A

    # A one-layer decoder's self-attention reads the decoder input, which is zeros: its output cannot depend on anything.
    mo = hooks["dec_self"].mean_out
    dec_self_spread = float((mo - mo.mean(1, keepdim=True)).abs().max())

    # ---------------------------------------------------------------- remove one head, then one module, at a time
    for row in heads:
        hook = hooks[row["module"]]
        hook.ablate = (row["head"],)
        row["ablation"] = summarise(predict())
        hook.ablate = ()
        print(f"{row['module']} head {row['head']}: effect {row['ablation']['effect']:.4f}, error x{row['ablation']['ratio']:.2f}", flush=True)
    modules = {}
    for name, hook in hooks.items():
        hook.ablate = tuple(range(hook.heads))
        modules[name] = summarise(predict())
        hook.ablate = ()
        print(f"all of {name}: effect {modules[name]['effect']:.4f}, error x{modules[name]['ratio']:.2f}", flush=True)

    strongest = max(heads, key=lambda r: r["ablation"]["effect"])
    (out_dir / "heads.json").write_text(json.dumps(dict(
        run=str(args.run), frames=N, episodes=len(set(episode.tolist())), stride=args.stride, parity_max_abs=parity,
        baseline_mse=float(base.mean()), baseline_start90=float(base[start90].mean()), target_var_by_joint=var_joint.tolist(),
        joints=JOINTS, phases=PHASES, dec_self_spread=dec_self_spread, mean_napkin_area=float(hull.mean()), mean_cube_area=float(cube.mean()),
        strongest_head=dict(module=strongest["module"], head=strongest["head"], effect=strongest["ablation"]["effect"]),
        largest_single_head_error_change=float(max(abs(r["ablation"]["ratio"] - 1) for r in heads)),
        heads=heads, module_ablation=modules), indent=1))
    np.savez_compressed(out_dir / "maps.npz", step=step, episode=episode, rows=rows, hull=hull, cube=cube, **maps)

    # ---------------------------------------------------------------- hide groups of tokens from attention
    napkin = hull > 0.2
    rng = np.random.default_rng(0)
    control = np.zeros_like(napkin)                                           # as many patches, drawn from the rest of the image
    for i in range(N):
        free = np.flatnonzero(~napkin[i])
        control[i, rng.choice(free, size=min(int(napkin[i].sum()), free.size), replace=False)] = True

    def keys(style: bool = False, state: bool = False, image: np.ndarray | None = None) -> torch.Tensor:
        m = np.zeros((N, 66), bool)
        m[:, 0], m[:, 1] = style, state
        if image is not None:
            m[:, 2:] = image
        return torch.as_tensor(m, device="cuda")

    tests = {
        "decoder cannot read the style token": {"dec_cross": keys(style=True)},
        "decoder cannot read the state token": {"dec_cross": keys(state=True)},
        "decoder cannot read style or state": {"dec_cross": keys(style=True, state=True)},
        "decoder cannot read any image token": {"dec_cross": keys(image=np.ones((N, 64), bool))},
        "encoder cannot read the state token": {n: keys(state=True) for n in ENCODER},
        "nothing can read the state token": {n: keys(state=True) for n in READERS},
        "nothing can read the style token": {n: keys(style=True) for n in READERS},
        "nothing can read napkin patches": {n: keys(image=napkin) for n in READERS},
        "nothing can read control patches": {n: keys(image=control) for n in READERS},
    }
    for i, name in enumerate(ENCODER):
        tests[f"encoder layer {i} cannot read the state token"] = {name: keys(state=True)}
        tests[f"encoder layer {i} cannot read napkin patches"] = {name: keys(image=napkin)}
    results = {}
    for name, blocks in tests.items():
        results[name] = summarise(predict(blocks))
        print(f"{name:48s} error x{results[name]['ratio']:6.2f}", flush=True)
    (out_dir / "block.json").write_text(json.dumps(dict(
        frames=N, napkin_patches_per_frame=float(napkin.sum(1).mean()), baseline_mse=float(base.mean()), joints=JOINTS, phases=PHASES,
        tests=results), indent=1))

    figures(out_dir, frames, maps, step, episode, tuple(args.figure_frame))
    print("wrote", out_dir, flush=True)


def figures(out_dir: Path, frames: np.ndarray, maps: dict, step: np.ndarray, episode: np.ndarray, strip_frame: tuple[int, int]) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from PIL import Image

    at = lambda e, s: int(np.flatnonzero((episode == e) & (step == s))[0])
    pick = [at(e, s) for e in (0, 7) for s in (0, 150, 300)]
    for mod in READERS:
        fig, ax = plt.subplots(8, 6, figsize=(9, 12))
        for h in range(8):
            kw = maps[f"{mod}_h{h}_keyw"].astype(np.float32)
            for c, i in enumerate(pick):
                heat = cv2.resize(kw[i].reshape(8, 8) / kw[i].max(), (256, 256), interpolation=cv2.INTER_NEAREST)
                a = ax[h, c]
                a.imshow(frames[i].astype(np.float32).mean(-1) / 255, cmap="gray", vmin=0, vmax=1)
                a.imshow(heat, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
                a.set_xticks([]); a.set_yticks([])
                if c == 0:
                    a.set_ylabel(f"head {h}", fontsize=9)
                if h == 0:
                    a.set_title(f"episode {episode[i]} step {step[i]}", fontsize=8)
        fig.suptitle(f"{mod}: which patches each head reads (bright = read most)", fontsize=11)
        fig.tight_layout(); fig.savefig(out_dir / f"overlay_{mod}.png", dpi=70); plt.close(fig)

    fig, ax = plt.subplots(2, 4, figsize=(12, 5))
    for h in range(8):
        b = maps[f"dec_cross_h{h}_bystep"]
        a = ax[h // 4, h % 4]
        a.plot(b[:, 0], label="style"); a.plot(b[:, 1], label="state"); a.plot(b[:, 2:].sum(1), label="image (all 64)")
        a.set_title(f"decoder cross-attention head {h}", fontsize=9); a.set_ylim(0, 1); a.set_xlabel("plan step")
    ax[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out_dir / "dec_cross_bystep.png", dpi=80); plt.close(fig)

    # one strip of 8 tiles per module on a single frame (the presentation figure)
    i = at(*strip_frame)
    gray = np.repeat(frames[i].astype(np.float32).mean(-1, keepdims=True) / 255, 3, -1)
    T, G = 192, 20
    for mod in READERS:
        strip = np.zeros((T, 8 * T + 7 * G, 4), np.float32)
        for h in range(8):
            kw = maps[f"{mod}_h{h}_keyw"][i].astype(np.float32).reshape(8, 8)
            heat = np.kron(kw / kw.max(), np.ones((32, 32), np.float32))
            alpha = (0.30 + 0.62 * heat)[..., None]
            tile = (1 - alpha) * gray * 0.9 + alpha * cm.inferno(heat)[..., :3]
            tile = np.asarray(Image.fromarray((tile * 255).astype(np.uint8)).resize((T, T), Image.NEAREST), np.float32) / 255
            x = h * (T + G)
            strip[:, x:x + T, :3], strip[:, x:x + T, 3] = tile, 1
        Image.fromarray((strip * 255).astype(np.uint8), "RGBA").save(out_dir / f"attn_{mod}.png")


if __name__ == "__main__":
    main()
