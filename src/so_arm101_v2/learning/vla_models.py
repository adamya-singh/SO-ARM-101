"""One interface over the three pretrained policies of the retry tranche (2026-10-04): pi0.5, SmolVLA, ACT.

Every model is trained and judged on the contract of the CNN runs and the pi0.5 fine-tune (learning/pi05.py):
one 256x256 wrist frame, state = ``normalize_act(current)`` + progress (7 dims), target = the 90-step residual
chunk, random shift 12 px, held-out start-90 MSE in normalised-act units. All three share pi0.5's quantile
normalisation (state and residual action mapped per channel to [-1, 1]), computed from the captures and
stored in the checkpoint: the same numbers enter training, scoring, rollouts and RL (the processor-contract
lesson of notes/smolvla-processor-contract-fix.md: one normalisation, written once).

The models are used directly, not through lerobot's processor pipelines: images, prompt tokens and padded
state are built here.

Interface (``ResidualModel``):
- ``loss(images, state, targets, generator)`` -> (scalar, per-joint (6,)) supervised loss for one micro-batch
- ``sample(images, state, seed, num_steps)`` -> (B, 90, 6) residual chunk, numpy, deterministic for a seed
- ``trainable(scope)`` -> named parameters for a fine-tune scope or the RL scope ``rl_heads``
- flow models (pi0.5, SmolVLA) also expose ``prefix(images, state)`` (no grad: VLM keys/values) and
  ``velocity(ctx, x_t, t)`` -> (velocity (B, H, 6), features (B, H, F)) for ReinFlow (rl/reinflow.py);
- ACT exposes ``mean(images, state)`` (with grad) for the Gaussian PPO policy (rl/act_ppo.py).

pi0.5 runs in the ``lerobot-pi05`` env, SmolVLA and ACT in ``lerobot``; imports are lazy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from so_arm101_v2.learning.pi05 import ACTION_DIM, STATE_DIM, TASK_PROMPT, QuantileStats

HORIZON = 90
KINDS = ("pi05", "smolvla", "act")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _torch():
    import torch
    return torch


def frames_to_tensor(frames_uint8: np.ndarray, device: Any, *, shift: int = 0, generator: Any = None) -> Any:
    """(B, 256, 256, 3) uint8 -> (B, 3, 256, 256) float in [0, 1], random shift at 256 px (the CNN recipe)."""
    torch = _torch()
    from so_arm101_v2.learning.vision import random_shift_images
    images = torch.from_numpy(np.ascontiguousarray(frames_uint8)).to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    if shift > 0:
        images = random_shift_images(images, shift, generator)
    return images


@dataclass(frozen=True)
class VLAConfig:
    """Fine-tune settings shared by the three models; per-model fields are ignored by the others."""

    kind: str
    pretrained: str = ""
    train_scope: str = "default"
    batch_size: int = 32
    micro_batch: int = 16
    max_steps: int = 30000
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    warmup_steps: int = 1000
    final_lr_fraction: float = 0.1
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    random_shift: int = 12
    chunk_horizon: int = HORIZON
    num_inference_steps: int = 10
    seed: int = 202
    # SmolVLA
    tokenizer_max_length: int = 48
    # ACT
    dim_model: int = 512
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    use_vae: bool = True
    kl_weight: float = 10.0
    dropout: float = 0.1
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.batch_size % self.micro_batch:
            raise ValueError("batch_size must be a multiple of micro_batch")
        if self.chunk_horizon != HORIZON:
            raise ValueError("the tranche uses the 90-step residual target for every model")


DEFAULTS = {
    "smolvla": dict(pretrained="lerobot/smolvla_base", learning_rate=1e-4, warmup_steps=1000, weight_decay=1e-10, grad_clip=10.0),
    "act": dict(pretrained="resnet18_imagenet", learning_rate=1e-4, backbone_learning_rate=1e-5, weight_decay=1e-4, grad_clip=10.0,
                micro_batch=64, batch_size=64, warmup_steps=500),
}


class MasterWeights:
    """fp32 master copies for trainable parameters stored in a lower precision (pi0.5's bf16 weights)."""

    def __init__(self, params: list[Any]) -> None:
        self.params = params
        self.master = [p.detach().float().clone() if p.dtype != _torch().float32 else p for p in params]
        for m, p in zip(self.master, params):
            if m is not p:
                m.requires_grad_(True)

    def pull_grads(self) -> None:
        for p, m in zip(self.params, self.master):
            if m is p:
                continue
            m.grad = None if p.grad is None else p.grad.float()
            p.grad = None

    def push(self) -> None:
        torch = _torch()
        with torch.no_grad():
            for p, m in zip(self.params, self.master):
                if m is not p:
                    p.copy_(m)


class ResidualModel:
    kind = "base"
    flow = False

    def __init__(self, policy: Any, config: VLAConfig, stats: QuantileStats, device: Any) -> None:
        self.policy, self.config, self.stats, self.device = policy, config, stats, device

    # ---------------------------------------------------------------- shared
    def norm_state_tensor(self, state: np.ndarray) -> Any:
        torch = _torch()
        return torch.as_tensor(self.stats.norm_state(np.asarray(state, np.float32)), dtype=torch.float32, device=self.device)

    def norm_targets(self, targets: np.ndarray) -> Any:
        torch = _torch()
        return self.stats.norm_action(torch.as_tensor(targets, dtype=torch.float32, device=self.device))

    def named_trainable(self, scope: str) -> list[tuple[str, Any]]:
        raise NotImplementedError

    def set_scope(self, scope: str) -> list[tuple[str, Any]]:
        chosen = {n for n, _ in self.named_trainable(scope)}
        for name, param in self.policy.named_parameters():
            param.requires_grad = name in chosen
        return [(n, p) for n, p in self.policy.named_parameters() if p.requires_grad]

    def train(self, mode: bool = True) -> None:
        self.policy.train(mode)

    def eval(self) -> None:
        self.policy.eval()

    def save(self, path: Path, extra: dict[str, Any], *, names: set[str] | None = None) -> None:
        """Trainable tensors (or ``names``) + config + stats; the base comes from ``config.pretrained``."""
        torch = _torch()
        state = {n: p.detach().cpu() for n, p in self.policy.named_parameters() if (n in names if names is not None else p.requires_grad)}
        buffers = {n: b.detach().cpu() for n, b in self.policy.named_buffers()} if self.kind == "act" else {}
        payload = {"schema_version": 1, "model_kind": f"{self.kind}_residual_h{HORIZON}", "config": asdict(self.config),
                   "stats": self.stats.to_json(), "state_dict": state, "buffers": buffers, **extra}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def load_state(self, state_dict: dict[str, Any], buffers: dict[str, Any] | None = None) -> None:
        torch = _torch()
        params = dict(self.policy.named_parameters())
        missing = [k for k in state_dict if k not in params]
        if missing:
            raise ValueError(f"checkpoint tensors not in the model: {missing[:5]}")
        with torch.no_grad():
            for name, tensor in state_dict.items():
                params[name].copy_(tensor.to(device=params[name].device, dtype=params[name].dtype))
            if buffers:
                own = dict(self.policy.named_buffers())
                for name, tensor in buffers.items():
                    if name in own:
                        own[name].copy_(tensor.to(device=own[name].device, dtype=own[name].dtype))

    # ---------------------------------------------------------------- flow sampling shared by pi0.5 and SmolVLA
    def sample(self, frames: np.ndarray, state: np.ndarray, *, seed: int | None = 0, num_steps: int | None = None) -> np.ndarray:
        torch = _torch()
        steps = int(num_steps or self.config.num_inference_steps)
        with torch.inference_mode():
            ctx = self.prefix(frames, state)
            bsize = ctx["bsize"]
            generator = None if seed is None else torch.Generator(device=self.device).manual_seed(int(seed))
            x = torch.randn((bsize, HORIZON, self.action_width), generator=generator, device=self.device, dtype=torch.float32)
            dt = -1.0 / steps
            for k in range(steps):
                t = torch.full((bsize,), 1.0 + k * dt, dtype=torch.float32, device=self.device)
                v, _ = self.velocity(ctx, x, t)
                x = x + dt * v
            return self.stats.unnorm_action(x[:, :, :ACTION_DIM].float()).cpu().numpy()

    def _velocity_with_features(self, call: Any) -> tuple[Any, Any]:
        """Run ``call`` (which ends in ``action_out_proj``) and capture that layer's input: the expert features."""
        captured: dict[str, Any] = {}

        def hook(module: Any, inputs: tuple[Any, ...]) -> None:
            captured["features"] = inputs[0]

        handle = self.out_proj.register_forward_pre_hook(hook)
        try:
            velocity = call()
        finally:
            handle.remove()
        return velocity, captured["features"]


# ---------------------------------------------------------------------------------------------------- pi0.5


class Pi05Model(ResidualModel):
    kind = "pi05"
    flow = True

    def __init__(self, policy: Any, pi05_config: Any, stats: QuantileStats, device: Any) -> None:
        from so_arm101_v2.learning.pi05 import PromptTokenizer
        config = VLAConfig(kind="pi05", pretrained=pi05_config.pretrained, num_inference_steps=pi05_config.num_inference_steps,
                           random_shift=pi05_config.random_shift, seed=pi05_config.seed, extra=dict(pi05=asdict(pi05_config)))
        super().__init__(policy, config, stats, device)
        self.pi05_config = pi05_config
        self.prompt = PromptTokenizer(pi05_config.tokenizer, pi05_config.tokenizer_max_length)
        self.action_width = int(policy.config.max_action_dim)
        self.out_proj = policy.model.action_out_proj

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cuda") -> "Pi05Model":
        from so_arm101_v2.learning.pi05 import load_checkpoint
        policy, config, stats, _ = load_checkpoint(path, device=device)
        return cls(policy, config, stats, _torch().device(device))

    def _inputs(self, frames: np.ndarray, state: np.ndarray, *, shift: int = 0, generator: Any = None) -> tuple[Any, Any, Any]:
        from so_arm101_v2.learning.pi05 import prepare_images
        images = prepare_images(frames, self.device, shift=shift, generator=generator)
        tokens, masks = self.prompt(self.stats.norm_state(np.asarray(state, np.float32)), self.device)
        return images, tokens, masks

    def loss(self, frames, state, targets, generator=None):
        from so_arm101_v2.learning.pi05 import flow_loss
        images, tokens, masks = self._inputs(frames, state, shift=self.config.random_shift, generator=generator)
        losses = flow_loss(self.policy, images, tokens, masks, self.norm_targets(targets))
        return losses.mean(), losses.detach().mean(dim=(0, 1))

    def prefix(self, frames: np.ndarray, state: np.ndarray) -> dict[str, Any]:
        torch = _torch()
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        model = self.policy.model
        images, tokens, masks = self._inputs(frames, state)
        img_mask = torch.ones(images.shape[0], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix([images], [img_mask], tokens, masks)
            att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past_key_values = model.paligemma_with_expert.forward(
                attention_mask=model._prepare_attention_masks_4d(att_2d), position_ids=position_ids,  # noqa: SLF001
                past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)
        return dict(prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values, bsize=int(images.shape[0]))

    def velocity(self, ctx: dict[str, Any], x_t: Any, t: Any) -> tuple[Any, Any]:
        model = self.policy.model
        return self._velocity_with_features(
            lambda: model.denoise_step(prefix_pad_masks=ctx["prefix_pad_masks"], past_key_values=ctx["past_key_values"], x_t=x_t, timestep=t))

    def named_trainable(self, scope: str) -> list[tuple[str, Any]]:
        if scope == "rl_out_head":
            return [(n, p) for n, p in self.policy.named_parameters() if n.startswith("model.action_out_proj.")]
        if scope != "rl_heads":
            raise ValueError("pi0.5 fine-tunes run through tools/pi05_finetune.py; only the RL scopes are defined here")
        heads = ("model.action_in_proj.", "model.action_out_proj.", "model.time_mlp_in.", "model.time_mlp_out.")
        return [(n, p) for n, p in self.policy.named_parameters() if n.startswith(heads)]

    def save(self, path: Path, extra: dict[str, Any], *, names: set[str] | None = None) -> None:
        """A pi0.5 checkpoint loadable by learning/pi05.load_checkpoint: every tensor the fine-tune trained, updated."""
        from so_arm101_v2.learning.pi05 import save_checkpoint, trainable_name
        keep = {n for n, _ in self.policy.named_parameters() if trainable_name(n, self.pi05_config.train_scope)}
        flags = {n: p.requires_grad for n, p in self.policy.named_parameters()}
        for n, p in self.policy.named_parameters():
            p.requires_grad = n in keep
        try:
            save_checkpoint(Path(path), self.policy, self.pi05_config, self.stats, extra)
        finally:
            for n, p in self.policy.named_parameters():
                p.requires_grad = flags[n]


# ---------------------------------------------------------------------------------------------------- SmolVLA


class SmolVLAModel(ResidualModel):
    kind = "smolvla"
    flow = True
    IMAGE_KEY = "observation.images.wrist"

    def __init__(self, policy: Any, config: VLAConfig, stats: QuantileStats, device: Any) -> None:
        super().__init__(policy, config, stats, device)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
        self.action_width = int(policy.config.max_action_dim)
        self.out_proj = policy.model.action_out_proj
        self._tokens = None

    @staticmethod
    def make_policy_config(config: VLAConfig, device: str) -> Any:
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        cfg = SmolVLAConfig(chunk_size=HORIZON, n_action_steps=HORIZON, num_steps=config.num_inference_steps,
                            tokenizer_max_length=config.tokenizer_max_length, device=device, load_vlm_weights=False,
                            freeze_vision_encoder=True, train_expert_only=True, train_state_proj=True,
                            normalization_mapping={"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.IDENTITY,
                                                   "ACTION": NormalizationMode.IDENTITY})
        cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
                              SmolVLAModel.IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))}
        cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
        return cfg

    @classmethod
    def build(cls, config: VLAConfig, stats: QuantileStats, device: str = "cuda", *, load_pretrained: bool = True) -> tuple["SmolVLAModel", dict]:
        torch = _torch()
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        cfg = cls.make_policy_config(config, device)
        info: dict[str, Any] = {}
        if load_pretrained:
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open
            policy = SmolVLAPolicy.from_pretrained(config.pretrained, config=cfg)
            path = hf_hub_download(config.pretrained, "model.safetensors")
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
            own = {n for n, _ in policy.named_parameters()}
            info["missing"] = sorted(own - keys)
            info["unexpected"] = sorted(keys - own)
            # The fork's ReinFlow noise_mlp is the only parameter the base checkpoint may lack; the tranche uses its
            # own noise head (rl/reinflow.py), so the fork's stays untrained and outside every optimiser.
            if [k for k in info["missing"] if not k.startswith("model.noise_mlp.")]:
                raise RuntimeError(f"smolvla_base left parameters uninitialised: {info['missing'][:8]}")
        else:
            with torch.device(device):
                policy = SmolVLAPolicy(cfg)
        policy.to(device)
        model = cls(policy, config, stats, torch.device(device))
        model.set_scope(config.train_scope)
        info["parameters_total"] = int(sum(p.numel() for p in policy.parameters()))
        info["parameters_trainable"] = int(sum(p.numel() for p in policy.parameters() if p.requires_grad))
        return model, info

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cuda") -> "SmolVLAModel":
        torch = _torch()
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1 or payload.get("model_kind") != f"smolvla_residual_h{HORIZON}":
            raise ValueError("not a SmolVLA residual checkpoint")
        config = VLAConfig(**payload["config"])
        model, _ = cls.build(config, QuantileStats.from_json(payload["stats"]), device)
        model.load_state(payload["state_dict"])
        model.eval()
        return model

    def named_trainable(self, scope: str) -> list[tuple[str, Any]]:
        params = list(self.policy.named_parameters())
        if scope in ("default", "expert"):
            # lerobot's SmolVLA recipe: frozen VLM (train_expert_only), trained expert + projections + state_proj.
            return [(n, p) for n, p in params if not n.startswith("model.vlm_with_expert.vlm.") and not n.startswith("model.noise_mlp.")]
        if scope == "rl_out_head":
            return [(n, p) for n, p in params if n.startswith("model.action_out_proj.")]
        if scope == "rl_heads":
            # Legacy rl_stable_heads (README "Legacy RL stack status"): action in/out projections and time MLPs.
            heads = ("model.action_in_proj.", "model.action_out_proj.", "model.action_time_mlp_in.", "model.action_time_mlp_out.")
            return [(n, p) for n, p in params if n.startswith(heads)]
        raise ValueError(f"unknown SmolVLA scope {scope!r}")

    def _images(self, frames: np.ndarray, *, shift: int = 0, generator: Any = None) -> Any:
        torch = _torch()
        images = frames_to_tensor(frames, self.device, shift=shift, generator=generator)
        size = self.policy.config.resize_imgs_with_padding
        if size is not None:
            images = torch.nn.functional.interpolate(images, size=tuple(size), mode="bilinear", align_corners=False)   # square frames: no pad
        return images * 2.0 - 1.0

    def _language(self, bsize: int) -> tuple[Any, Any]:
        torch = _torch()
        if self._tokens is None:
            out = self.tokenizer([TASK_PROMPT + "\n"], padding="max_length", max_length=self.config.tokenizer_max_length,
                                 truncation=True, return_tensors="pt")
            self._tokens = (out["input_ids"].to(self.device), out["attention_mask"].to(self.device, dtype=torch.bool))
        tokens, masks = self._tokens
        return tokens.expand(bsize, -1), masks.expand(bsize, -1)

    def _state(self, state: np.ndarray) -> Any:
        torch = _torch()
        s = self.norm_state_tensor(state)
        return torch.nn.functional.pad(s, (0, self.policy.config.max_state_dim - STATE_DIM))

    def loss(self, frames, state, targets, generator=None):
        torch = _torch()
        images = self._images(frames, shift=self.config.random_shift, generator=generator)
        bsize = images.shape[0]
        tokens, masks = self._language(bsize)
        actions = torch.nn.functional.pad(self.norm_targets(targets), (0, self.action_width - ACTION_DIM))
        img_mask = torch.ones(bsize, dtype=torch.bool, device=self.device)
        losses = self.policy.model.forward([images], [img_mask], tokens, masks, self._state(state), actions)[:, :, :ACTION_DIM]
        return losses.mean(), losses.detach().mean(dim=(0, 1))

    def prefix(self, frames: np.ndarray, state: np.ndarray) -> dict[str, Any]:
        torch = _torch()
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        model = self.policy.model
        images = self._images(frames)
        bsize = images.shape[0]
        tokens, masks = self._language(bsize)
        img_mask = torch.ones(bsize, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix([images], [img_mask], tokens, masks, state=self._state(state))
            att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, past_key_values = model.vlm_with_expert.forward(attention_mask=att_2d, position_ids=position_ids, past_key_values=None,
                                                               inputs_embeds=[prefix_embs, None], use_cache=model.config.use_cache,
                                                               fill_kv_cache=True)
        return dict(prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values, bsize=int(bsize))

    def velocity(self, ctx: dict[str, Any], x_t: Any, t: Any) -> tuple[Any, Any]:
        model = self.policy.model
        return self._velocity_with_features(lambda: model.denoise_step(ctx["prefix_pad_masks"], ctx["past_key_values"], x_t, t))


# ---------------------------------------------------------------------------------------------------- ACT


class ACTModel(ResidualModel):
    kind = "act"
    flow = False
    IMAGE_KEY = "observation.images.wrist"

    @staticmethod
    def make_policy_config(config: VLAConfig, device: str, *, pretrained_backbone: bool) -> Any:
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        cfg = ACTConfig(chunk_size=HORIZON, n_action_steps=HORIZON, dim_model=config.dim_model, n_encoder_layers=config.n_encoder_layers,
                        n_decoder_layers=config.n_decoder_layers, use_vae=config.use_vae, kl_weight=config.kl_weight, dropout=config.dropout,
                        pretrained_backbone_weights=("ResNet18_Weights.IMAGENET1K_V1" if pretrained_backbone else None), device=device,
                        normalization_mapping={"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.IDENTITY,
                                               "ACTION": NormalizationMode.IDENTITY})
        cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
                              ACTModel.IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))}
        cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
        return cfg

    @classmethod
    def build(cls, config: VLAConfig, stats: QuantileStats, device: str = "cuda", *, load_pretrained: bool = True) -> tuple["ACTModel", dict]:
        torch = _torch()
        from lerobot.policies.act.modeling_act import ACTPolicy
        cfg = cls.make_policy_config(config, device, pretrained_backbone=load_pretrained)
        policy = ACTPolicy(cfg).to(device)
        model = cls(policy, config, stats, torch.device(device))
        model.set_scope(config.train_scope)
        info = {"parameters_total": int(sum(p.numel() for p in policy.parameters())),
                "parameters_trainable": int(sum(p.numel() for p in policy.parameters() if p.requires_grad))}
        return model, info

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cuda") -> "ACTModel":
        torch = _torch()
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1 or payload.get("model_kind") != f"act_residual_h{HORIZON}":
            raise ValueError("not an ACT residual checkpoint")
        config = VLAConfig(**payload["config"])
        model, _ = cls.build(config, QuantileStats.from_json(payload["stats"]), device, load_pretrained=False)
        model.load_state(payload["state_dict"], payload.get("buffers"))
        model.eval()
        return model

    def named_trainable(self, scope: str) -> list[tuple[str, Any]]:
        params = list(self.policy.named_parameters())
        if scope in ("default", "all"):
            return params
        if scope == "rl_heads":
            # Legacy ACTGaussianPPOPolicy trained the whole ACT at a small rate; the decoder + action head is the
            # stable subset (the backbone and encoder carry the perception the fine-tune learned).
            return [(n, p) for n, p in params if n.startswith(("model.decoder.", "model.action_head.", "model.decoder_pos_embed."))]
        raise ValueError(f"unknown ACT scope {scope!r}")

    def param_groups(self, named: list[tuple[str, Any]]) -> list[dict[str, Any]]:
        backbone = [p for n, p in named if n.startswith("model.backbone.")]
        rest = [p for n, p in named if not n.startswith("model.backbone.")]
        groups = [dict(params=rest, lr=self.config.learning_rate, base_lr=self.config.learning_rate)]
        if backbone:
            groups.append(dict(params=backbone, lr=self.config.backbone_learning_rate, base_lr=self.config.backbone_learning_rate))
        return groups

    def _batch(self, frames: np.ndarray, state: np.ndarray, *, shift: int = 0, generator: Any = None) -> dict[str, Any]:
        torch = _torch()
        images = frames_to_tensor(frames, self.device, shift=shift, generator=generator)
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        return {"observation.state": self.norm_state_tensor(state), "observation.images": [(images - mean) / std]}

    def loss(self, frames, state, targets, generator=None):
        torch = _torch()
        batch = self._batch(frames, state, shift=self.config.random_shift, generator=generator)
        actions = self.norm_targets(targets)
        batch["action"] = actions
        batch["action_is_pad"] = torch.zeros(actions.shape[:2], dtype=torch.bool, device=self.device)
        actions_hat, (mu, log_sigma_x2) = self.policy.model(batch)
        l1 = torch.nn.functional.l1_loss(actions_hat, actions, reduction="none")
        loss = l1.mean()
        if self.config.use_vae and mu is not None:
            loss = loss + self.config.kl_weight * (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
        return loss, l1.detach().mean(dim=(0, 1))

    def mean(self, frames: np.ndarray, state: np.ndarray) -> Any:
        """Deterministic chunk (latent zero) in normalised residual units, (B, H, 6); with grad when enabled."""
        was = self.policy.model.training
        self.policy.model.eval()        # latent zero and no dropout: the same chunk at rollout and in the PPO ratio
        try:
            return self.policy.model(self._batch(frames, state))[0]
        finally:
            self.policy.model.train(was)

    def sample(self, frames: np.ndarray, state: np.ndarray, *, seed: int | None = 0, num_steps: int | None = None) -> np.ndarray:
        torch = _torch()
        del seed, num_steps
        with torch.inference_mode():
            return self.stats.unnorm_action(self.mean(frames, state).float()).cpu().numpy()


# ---------------------------------------------------------------------------------------------------- loading


def load_model(path: str | Path, device: str = "cuda") -> ResidualModel:
    """Any tranche checkpoint (pi0.5 fine-tune or RL, SmolVLA, ACT) by its ``model_kind``."""
    torch = _torch()
    kind = str(torch.load(Path(path), map_location="cpu", weights_only=False, mmap=True).get("model_kind", ""))
    if kind.startswith("pi05_residual"):
        return Pi05Model.from_checkpoint(path, device)
    if kind.startswith("smolvla_residual"):
        return SmolVLAModel.from_checkpoint(path, device)
    if kind.startswith("act_residual"):
        return ACTModel.from_checkpoint(path, device)
    raise ValueError(f"unknown checkpoint kind {kind!r}")


def build_model(config: VLAConfig, stats: QuantileStats, device: str = "cuda", *, load_pretrained: bool = True) -> tuple[ResidualModel, dict]:
    if config.kind == "smolvla":
        return SmolVLAModel.build(config, stats, device, load_pretrained=load_pretrained)
    if config.kind == "act":
        return ACTModel.build(config, stats, device, load_pretrained=load_pretrained)
    raise ValueError("pi0.5 fine-tunes use tools/pi05_finetune.py; load its checkpoint with Pi05Model.from_checkpoint")


# ---------------------------------------------------------------------------------------------------- scoring


class HeldoutScorer:
    """pi0.5's held-out scorer (learning/pi05.Pi05Scorer) for any ResidualModel: same buckets, same units."""

    def __init__(self, heldout: Any, train_sources: list[Any], *, horizon: int = HORIZON, train_samples: int = 200, seed: int = 0) -> None:
        from so_arm101_v2.learning.pi05 import Pi05Scorer
        self._inner = Pi05Scorer(heldout, train_sources, horizon=horizon, train_samples=train_samples, seed=seed)

    def score(self, model: ResidualModel, threshold: float = 2.5e-4, batch: int = 16) -> dict[str, Any]:
        inner = self._inner

        def errors(_policy, _stats, _prompt, images, state, targets, _device, batch=batch):
            out = []
            for start in range(0, images.shape[0], batch):
                sl = slice(start, start + batch)
                pred = model.sample(images[sl], state[sl], seed=start)
                out.append(((pred - targets[sl]) ** 2).mean(axis=(1, 2)))
            return np.concatenate(out)

        original = inner._errors
        inner._errors = errors   # type: ignore[method-assign]
        try:
            return inner.score(model.policy, model.stats, None, model.device, threshold=threshold)
        finally:
            inner._errors = original   # type: ignore[method-assign]


__all__ = ["ACTModel", "DEFAULTS", "HORIZON", "HeldoutScorer", "KINDS", "MasterWeights", "Pi05Model", "ResidualModel", "SmolVLAModel",
           "VLAConfig", "build_model", "frames_to_tensor", "load_model"]
