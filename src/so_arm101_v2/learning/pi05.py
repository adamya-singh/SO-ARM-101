"""pi0.5 (lerobot ``PI05Policy``) fine-tuned on the bench pick-and-replace captures (2026-10-02).

Same task contract as the vision chunked clone (``learning/vision.py``) so every number is comparable
with the CNN runs: one 256x256 wrist frame, state = ``normalize_act(current_act)`` + progress, target =
the 90-step residual chunk ``normalize_act(executed[t + j]) - normalize_act(current[t])`` with
repeat-last padding inside the episode, random-shift augmentation, held-out start-90 MSE in the same
normalised units (one unit = pi rad, so the working policy sits at 2.5e-6).

pi0.5 specifics:
- the frame is shifted at 256 px (the CNN recipe) and resized to SigLIP's 224 px;
- state enters through the prompt as 256-bin tokens after quantile normalisation (pi0.5's own scheme);
  only the 7 real dimensions are written (no 32-dim zero padding: fewer tokens, same information);
- actions are quantile-normalised per channel to [-1, 1] for the flow-matching loss and mapped back
  to residual units for every reported number;
- memory on a 24 GB card: bf16 weights, per-layer gradient checkpointing, the unused expert lm_head
  removed, fp32 master copies of the trainable weights only (bf16 cannot hold a 2.5e-5 x update),
  and the PaliGemma backbone either frozen (scope ``expert``) or adapted with LoRA (``expert_lora``).
- the 14.5 GB fp32 checkpoint is streamed tensor by tensor onto the GPU: the host has 15 GB of RAM.

Runs in the ``lerobot-pi05`` conda env (transformers ``fix/lerobot_openpi`` branch, which pi0.5's
adaRMS layers require); the project's ``lerobot`` env keeps stock transformers.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from so_arm101_v2.learning.tiny_model import normalize_act

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOKENIZER = ROOT / "artifacts/pretrained/paligemma_tokenizer"   # sha256-verified copy of google/paligemma-3b-pt-224's tokenizer
TASK_PROMPT = "pick up the cube and put it back on the square"
STATE_DIM = 7
ACTION_DIM = 6
IMAGE_KEY = "observation.images.base_0_rgb"
SCOPES = ("expert", "expert_lora")
LORA_TARGETS = re.compile(
    r"paligemma_with_expert\.paligemma\.model\.(language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
    r"|vision_tower\.vision_model\.encoder\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|mlp\.fc[12])"
    r"|multi_modal_projector\.linear)$"
)


@dataclass(frozen=True)
class Pi05FinetuneConfig:
    pretrained: str = "lerobot/pi05_base"
    tokenizer: str = str(DEFAULT_TOKENIZER)
    train_scope: str = "expert_lora"
    lora_rank: int = 16
    lora_alpha: float = 16.0
    batch_size: int = 32
    micro_batch: int = 16
    max_steps: int = 20000
    learning_rate: float = 2.5e-5        # pi0.5 preset (expert, projections)
    lora_learning_rate: float = 1e-4
    warmup_steps: int = 1000
    final_lr_fraction: float = 0.1       # cosine to 2.5e-6, the pi0.5 preset's decay_lr
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    random_shift: int = 12
    chunk_horizon: int = 90
    tokenizer_max_length: int = 64
    num_inference_steps: int = 10
    gradient_checkpointing: bool = True
    seed: int = 202

    def __post_init__(self) -> None:
        if self.train_scope not in SCOPES:
            raise ValueError(f"train_scope must be one of {SCOPES}")
        if self.batch_size % self.micro_batch:
            raise ValueError("batch_size must be a multiple of micro_batch")


def _torch():
    import torch
    return torch


# ---------------------------------------------------------------- model


def _lora_class():
    torch = _torch()

    class _LoRALinear(torch.nn.Module):
        def __init__(self, base: Any, rank: int, alpha: float) -> None:
            super().__init__()
            self.base = base
            self.scale = alpha / rank
            self.lora_a = torch.nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
            self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype, device=base.weight.device))
            torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        @property
        def weight(self) -> Any:   # pi0.5 / SigLIP code reads q_proj.weight.dtype
            return self.base.weight

        @property
        def bias(self) -> Any:
            return self.base.bias

        def forward(self, x: Any) -> Any:
            return self.base(x) + (x.to(self.lora_a.dtype) @ self.lora_a.t() @ self.lora_b.t()).to(x.dtype) * self.scale

    return _LoRALinear


def apply_lora(model: Any, rank: int, alpha: float) -> int:
    """Wrap every LORA_TARGETS linear of ``model`` (a PI05Policy) with a rank-``rank`` adapter; returns the count."""
    cls = _lora_class()
    # The expert reads only the backbone's keys/values, so the last language layer's o_proj and MLP never reach the loss.
    last = int(model.model.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers) - 1
    dead = re.compile(rf"language_model\.layers\.{last}\.(self_attn\.o_proj|mlp\.)")
    targets = [name for name, module in model.named_modules() if LORA_TARGETS.search(name) and not dead.search(name)]
    for name in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child, cls(getattr(parent, child), rank, alpha))
    if not targets:
        raise RuntimeError("no LoRA target modules matched; the pi0.5 module names changed")
    return len(targets)


def make_pi05_config(config: Pi05FinetuneConfig, device: str = "cuda") -> Any:
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    cfg = PI05Config(
        dtype="bfloat16",
        chunk_size=config.chunk_horizon,
        n_action_steps=config.chunk_horizon,
        max_state_dim=32,
        max_action_dim=32,
        num_inference_steps=config.num_inference_steps,
        tokenizer_max_length=config.tokenizer_max_length,
        gradient_checkpointing=config.gradient_checkpointing,
        train_expert_only=True,          # freezes PaliGemma; LoRA adapters (if any) are added after
        device=device,
        normalization_mapping={"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.QUANTILES,
                               "ACTION": NormalizationMode.QUANTILES},
    )
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    return cfg


def _resolve_pretrained_file(pretrained: str) -> Path:
    path = Path(pretrained)
    if path.is_dir():
        return path / "model.safetensors"
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(pretrained, "model.safetensors"))


def build_policy(config: Pi05FinetuneConfig, *, device: str = "cuda", load_pretrained: bool = True) -> tuple[Any, dict[str, Any]]:
    """PI05Policy on ``device`` with the pretrained weights streamed in, expert lm_head removed, LoRA applied per scope."""
    torch = _torch()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from safetensors import safe_open

    cfg = make_pi05_config(config, device)
    with torch.device(device):
        policy = PI05Policy(cfg)
    expert = policy.model.paligemma_with_expert.gemma_expert
    expert.lm_head = None   # 0.26B parameters pi0.5 never uses (the expert decodes through action_out_proj)
    info: dict[str, Any] = {"loaded": 0, "skipped": [], "missing": []}
    if load_pretrained:
        path = _resolve_pretrained_file(config.pretrained)
        params = dict(policy.named_parameters())
        params.update(dict(policy.named_buffers()))
        seen = set()
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            fixed = policy._fix_pytorch_state_dict_keys({k: k for k in keys}, policy.config)   # key -> original key
            for new_key, original in fixed.items():
                target = new_key if new_key.startswith("model.") else f"model.{new_key}"
                if target.endswith("paligemma.lm_head.weight") and target not in params:
                    # PaliGemma ties lm_head to the token embedding; the checkpoint stores it under lm_head only.
                    target = target.replace("paligemma.lm_head.weight", "paligemma.model.language_model.embed_tokens.weight")
                if target not in params:
                    info["skipped"].append(target)
                    continue
                tensor = handle.get_tensor(original)
                dest = params[target]
                if tuple(tensor.shape) != tuple(dest.shape):
                    raise ValueError(f"shape mismatch for {target}: {tuple(tensor.shape)} vs {tuple(dest.shape)}")
                with torch.no_grad():
                    dest.copy_(tensor.to(device=dest.device, dtype=dest.dtype))
                seen.add(target)
                info["loaded"] += 1
        info["missing"] = sorted(name for name, _ in policy.named_parameters() if name not in seen)
        if info["missing"]:
            raise RuntimeError(f"pretrained checkpoint left parameters uninitialised: {info['missing'][:8]}")
        info["skipped"] = [k for k in info["skipped"] if "gemma_expert.lm_head" not in k]
    if config.train_scope == "expert_lora":
        info["lora_modules"] = apply_lora(policy, config.lora_rank, config.lora_alpha)
    for name, param in policy.named_parameters():
        param.requires_grad = trainable_name(name, config.train_scope)
    info["parameters_total"] = int(sum(p.numel() for p in policy.parameters()))
    info["parameters_trainable"] = int(sum(p.numel() for p in policy.parameters() if p.requires_grad))
    return policy, info


def trainable_name(name: str, scope: str) -> bool:
    if ".lora_" in name:
        return scope == "expert_lora"
    if ".paligemma." in name:
        return False
    return True   # gemma_expert, action_in_proj, action_out_proj, time_mlp_in/out


# ---------------------------------------------------------------- data


@dataclass
class CaptureSource:
    """Arrays needed for pi0.5 samples from one frames capture, with the memmapped frames."""

    name: str
    manifest_path: Path
    frames: Any
    current_norm: np.ndarray       # (N, 6) float32
    executed_norm: np.ndarray      # (N, 6) float32
    progress: np.ndarray           # (N,) float32
    action_index: np.ndarray       # (N,) int64
    last_index: np.ndarray         # (N,) int64, last row of each row's episode
    episode_of_row: np.ndarray     # (N,) int32
    scenario_ids: list[str]
    rows: np.ndarray               # candidate sample rows (stored frames, inside the episode limit)

    @classmethod
    def load(cls, manifest_path: str | Path, *, name: str, episode_limit: int | None = None, frame_stride: int | None = None) -> "CaptureSource":
        from so_arm101_v2.learning.vision import load_vision_frames

        manifest, frames, arrays = load_vision_frames(manifest_path)
        lengths = [int(e["rows"]) for e in manifest["episodes"]]
        scenario_ids = [str(e["scenario_id"]) for e in manifest["episodes"]]
        del manifest
        starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        episode_of_row = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
        last_index = (starts[1:] - 1)[episode_of_row]
        action_index = np.asarray(arrays["action_index"], dtype=np.int64)
        stored = getattr(frames, "stored_mask", np.ones(frames.shape[0], dtype=bool))
        keep = stored.copy()
        if frame_stride:
            keep &= action_index % int(frame_stride) == 0
        if episode_limit is not None:
            keep &= episode_of_row < int(episode_limit)
        source = cls(
            name=name, manifest_path=Path(manifest_path), frames=frames,
            current_norm=normalize_act(np.asarray(arrays["current_act"], np.float32)).astype(np.float32),
            executed_norm=normalize_act(np.asarray(arrays["executed_act"], np.float32)).astype(np.float32),
            progress=np.asarray(arrays["progress"], np.float32), action_index=action_index,
            last_index=last_index, episode_of_row=episode_of_row, scenario_ids=scenario_ids,
            rows=np.flatnonzero(keep).astype(np.int64),
        )
        del arrays
        return source

    def state(self, rows: np.ndarray) -> np.ndarray:
        return np.concatenate([self.current_norm[rows], self.progress[rows, None]], axis=1).astype(np.float32)

    def targets(self, rows: np.ndarray, horizon: int) -> np.ndarray:
        """Residual chunk (len(rows), horizon, 6), identical to ``build_chunked_targets``."""
        index = np.minimum(rows[:, None] + np.arange(horizon)[None, :], self.last_index[rows][:, None])
        return (self.executed_norm[index] - self.current_norm[rows][:, None, :]).astype(np.float32)

    def images(self, rows: np.ndarray) -> np.ndarray:
        order = np.argsort(rows, kind="stable")
        block = np.asarray(self.frames[rows[order]])
        out = np.empty_like(block)
        out[order] = block
        return out

    def rows_at(self, action_index: int | Iterable[int]) -> np.ndarray:
        wanted = np.atleast_1d(np.asarray(list(action_index) if not isinstance(action_index, int) else [action_index]))
        return self.rows[np.isin(self.action_index[self.rows], wanted)]


@dataclass
class QuantileStats:
    state_q01: np.ndarray
    state_q99: np.ndarray
    action_q01: np.ndarray
    action_q99: np.ndarray

    @classmethod
    def from_sources(cls, sources: list[CaptureSource], horizon: int, *, samples: int = 40000, seed: int = 0) -> "QuantileStats":
        rng = np.random.default_rng(seed)
        states, actions = [], []
        per = samples // len(sources)
        for source in sources:
            rows = np.sort(rng.choice(source.rows, min(per, source.rows.shape[0]), replace=False))
            states.append(source.state(rows))
            actions.append(source.targets(rows, horizon).reshape(-1, ACTION_DIM))
        s = np.concatenate(states); a = np.concatenate(actions)
        return cls(np.quantile(s, 0.01, axis=0), np.quantile(s, 0.99, axis=0), np.quantile(a, 0.01, axis=0), np.quantile(a, 0.99, axis=0))

    def to_json(self) -> dict[str, list[float]]:
        return {k: [float(x) for x in v] for k, v in asdict(self).items()}

    @classmethod
    def from_json(cls, payload: dict[str, list[float]]) -> "QuantileStats":
        return cls(**{k: np.asarray(v, np.float32) for k, v in payload.items()})

    def norm_state(self, state: np.ndarray) -> np.ndarray:
        span = np.maximum(self.state_q99 - self.state_q01, 1e-8)
        return np.clip(2.0 * (state - self.state_q01) / span - 1.0, -1.0, 1.0)

    def norm_action(self, action: Any) -> Any:
        q01, span = self._action_tensors(action)
        return 2.0 * (action - q01) / span - 1.0

    def unnorm_action(self, action: Any) -> Any:
        q01, span = self._action_tensors(action)
        return (action + 1.0) * span / 2.0 + q01

    def _action_tensors(self, like: Any) -> tuple[Any, Any]:
        torch = _torch()
        q01 = torch.as_tensor(self.action_q01, dtype=like.dtype, device=like.device)
        span = torch.as_tensor(np.maximum(self.action_q99 - self.action_q01, 1e-8), dtype=like.dtype, device=like.device)
        return q01, span


class PromptTokenizer:
    """pi0.5's state-in-prompt: ``Task: <task>, State: <256-bin tokens>;\\nAction: `` (openpi PaligemmaTokenizer)."""

    def __init__(self, path: str, max_length: int) -> None:
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.max_length = int(max_length)
        self.bins = np.linspace(-1, 1, 256 + 1)[:-1]

    def prompts(self, norm_state: np.ndarray) -> list[str]:
        binned = np.digitize(norm_state, bins=self.bins) - 1
        return [f"Task: {TASK_PROMPT}, State: {' '.join(map(str, row))};\nAction: " for row in binned]

    def __call__(self, norm_state: np.ndarray, device: Any) -> tuple[Any, Any]:
        out = self.tokenizer(self.prompts(norm_state), max_length=self.max_length, truncation=True,
                             padding="max_length", padding_side="right", return_tensors="pt")
        if int(out["attention_mask"].sum(dim=1).max()) >= self.max_length:
            raise ValueError("prompt filled tokenizer_max_length; raise it so no state token is truncated")
        return out["input_ids"].to(device), out["attention_mask"].to(device=device, dtype=_torch().bool)


def prepare_images(frames_uint8: np.ndarray, device: Any, *, shift: int = 0, generator: Any = None) -> Any:
    """(B, 256, 256, 3) uint8 -> (B, 3, 224, 224) float in [0, 1]; random shift at 256 px first (the CNN recipe)."""
    torch = _torch()
    from so_arm101_v2.learning.vision import random_shift_images
    images = torch.from_numpy(np.ascontiguousarray(frames_uint8)).to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    if shift > 0:
        images = random_shift_images(images, shift, generator)
    return torch.nn.functional.interpolate(images, size=(224, 224), mode="bilinear", antialias=True, align_corners=False).clamp_(0.0, 1.0)


# ---------------------------------------------------------------- forward / sampling


def flow_loss(policy: Any, images: Any, tokens: Any, masks: Any, norm_actions: Any) -> Any:
    """Flow-matching MSE per (sample, step, channel) on the real 6 channels: (B, H, 6)."""
    torch = _torch()
    actions = torch.nn.functional.pad(norm_actions, (0, policy.config.max_action_dim - ACTION_DIM))
    img_mask = torch.ones(images.shape[0], dtype=torch.bool, device=images.device)
    losses = policy.model.forward([images], [img_mask], tokens, masks, actions)
    return losses[:, :, :ACTION_DIM]


def sample_residual(policy: Any, stats: QuantileStats, images: Any, tokens: Any, masks: Any, *, seed: int | None = 0, num_steps: int | None = None) -> Any:
    """Predicted residual chunk in normalised-act units (B, H, 6); fixed-seed noise makes scoring deterministic."""
    torch = _torch()
    bsize = images.shape[0]
    shape = (bsize, policy.config.chunk_size, policy.config.max_action_dim)
    generator = None if seed is None else torch.Generator(device=images.device).manual_seed(int(seed))
    noise = torch.randn(shape, generator=generator, device=images.device, dtype=torch.float32)
    img_mask = torch.ones(bsize, dtype=torch.bool, device=images.device)
    with torch.inference_mode():
        actions = policy.model.sample_actions([images], [img_mask], tokens, masks, noise=noise, num_steps=num_steps)
    return stats.unnorm_action(actions[:, :, :ACTION_DIM].float())


# ---------------------------------------------------------------- scoring


class Pi05Scorer:
    """Held-out start-0 / start-90 / starts-180-360 and train start-90 chunk MSE, in the CNN runs' units.

    The held-out capture stores every row; its chunk starts (multiples of 90 up to 360) are scored exhaustively
    (10 poses each); ``train_samples`` start-90 rows of the training data give the memorisation ratio.
    """

    BUCKETS = {"start_0": (0,), "start_90": (90,), "starts_180_270_360": (180, 270, 360)}

    def __init__(self, heldout: CaptureSource, train_sources: list[CaptureSource], *, horizon: int, train_samples: int = 200, seed: int = 0) -> None:
        self.horizon = horizon
        self.heldout: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
        for bucket, starts in self.BUCKETS.items():
            rows = np.flatnonzero(np.isin(heldout.action_index, starts))
            self.heldout[bucket] = (heldout.images(rows), heldout.state(rows), heldout.targets(rows, horizon),
                                    [heldout.scenario_ids[int(heldout.episode_of_row[r])] for r in rows])
        rng = np.random.default_rng(seed)
        per = max(1, train_samples // len(train_sources))
        imgs, states, targets = [], [], []
        for source in train_sources:
            rows = source.rows_at(90)
            rows = np.sort(rng.choice(rows, min(per, rows.shape[0]), replace=False))
            imgs.append(source.images(rows)); states.append(source.state(rows)); targets.append(source.targets(rows, horizon))
        self.train = (np.concatenate(imgs), np.concatenate(states), np.concatenate(targets))

    def _errors(self, policy: Any, stats: QuantileStats, prompt: PromptTokenizer, images: np.ndarray, state: np.ndarray,
                targets: np.ndarray, device: Any, batch: int = 16) -> np.ndarray:
        out = []
        for start in range(0, images.shape[0], batch):
            sl = slice(start, start + batch)
            tokens, masks = prompt(stats.norm_state(state[sl]), device)
            pred = sample_residual(policy, stats, prepare_images(images[sl], device), tokens, masks, seed=start).cpu().numpy()
            out.append(((pred - targets[sl]) ** 2).mean(axis=(1, 2)))
        return np.concatenate(out)

    def score(self, policy: Any, stats: QuantileStats, prompt: PromptTokenizer, device: Any, threshold: float = 2.5e-4) -> dict[str, Any]:
        was_training = policy.training
        policy.eval()
        try:
            result: dict[str, Any] = {}
            for bucket, (images, state, targets, poses) in self.heldout.items():
                errors = self._errors(policy, stats, prompt, images, state, targets, device)
                result[f"heldout_{bucket}"] = float(errors.mean())
                if bucket == "start_90":
                    per_pose = {pose: float(e) for pose, e in zip(poses, errors)}
                    values = np.asarray(list(per_pose.values()))
                    result.update(per_pose=per_pose, median=float(np.median(values)), max=float(values.max()),
                                  poses_under_threshold=int((values <= threshold).sum()), poses=int(values.shape[0]))
            train_errors = self._errors(policy, stats, prompt, *self.train, device)
            result["train_start_90"] = float(train_errors.mean())
            result["ratio_start_90"] = result["heldout_start_90"] / max(result["train_start_90"], 1e-12)
            result["heldout_start_90_deg_rms"] = math.sqrt(result["heldout_start_90"]) * 180.0
            return result
        finally:
            policy.train(was_training)


# ---------------------------------------------------------------- checkpoints


def save_checkpoint(path: Path, policy: Any, config: Pi05FinetuneConfig, stats: QuantileStats, extra: dict[str, Any]) -> None:
    """Trainable tensors only (expert, projections, LoRA) + config + quantile stats; the base comes from ``config.pretrained``."""
    torch = _torch()
    state = {name: p.detach().cpu() for name, p in policy.named_parameters() if p.requires_grad}
    payload = {"schema_version": 1, "model_kind": "pi05_residual_h%d" % config.chunk_horizon, "config": asdict(config),
               "stats": stats.to_json(), "state_dict": state, **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, *, device: str = "cuda") -> tuple[Any, Pi05FinetuneConfig, QuantileStats, dict[str, Any]]:
    torch = _torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or not str(payload.get("model_kind", "")).startswith("pi05_residual_h"):
        raise ValueError("not a pi0.5 residual checkpoint")
    config = Pi05FinetuneConfig(**payload["config"])
    policy, _ = build_policy(config, device=device)
    params = dict(policy.named_parameters())
    missing = [k for k in payload["state_dict"] if k not in params]
    if missing:
        raise ValueError(f"checkpoint tensors not in the model: {missing[:5]}")
    with torch.no_grad():
        for name, tensor in payload["state_dict"].items():
            params[name].copy_(tensor.to(device=params[name].device, dtype=params[name].dtype))
    policy.eval()
    return policy, config, QuantileStats.from_json(payload["stats"]), payload


def config_json(config: Pi05FinetuneConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True)
