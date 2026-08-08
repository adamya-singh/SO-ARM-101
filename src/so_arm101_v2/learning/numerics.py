"""Numerics regime descriptors: pinned, identity-bearing training numerics.

A regime describes every environment property that can change training
floating-point bits.  ``None`` throughout this codebase means the legacy
CPU-eager regime whose digests are pinned by existing artifacts; a
``NumericsSpec`` means regime v2 (GPU or CPU inductor).  The descriptor
enters the run identity as a conditional top-level ``"numerics"`` key so
every legacy digest stays byte-identical, exactly like the correction and
saturation conditional keys.

Regime v2 runs are bitwise-reproducible *within* a fingerprint; the
``require_numerics`` guard turns any environment drift (torch, CUDA, GPU
model, driver) into a loud error instead of a silent numerics fork — the
same philosophy as the mujoco 3.9.0 pin in ``simulation/cli.py``.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

# Sentinel for keyword defaults: "resolve the best available regime".
AUTO = "auto"

_CUBLAS_WORKSPACE = ":4096:8"
_INDUCTOR_CACHE_DIR = os.path.expanduser("~/.cache/so_arm101_v2/torchinductor")
_ENV_OVERRIDE = "SO_ARM101_V2_NUMERICS"


@dataclass(frozen=True)
class NumericsSpec:
    device: str                        # "cuda" or "cpu"
    gpu: str | None                    # device name, None for cpu specs
    torch_version: str                 # e.g. "2.7.1+cu126"
    cuda_version: str | None           # torch.version.cuda, None for cpu specs
    driver_version: str | None         # nvidia-smi driver, None for cpu specs
    compile: str | None = "inductor"   # None = eager on the given device
    compile_mode: str = "default"      # never "max-autotune": autotune selects
                                       # kernels by measured runtime, which is
                                       # not reproducible
    tf32: bool = False
    deterministic_algorithms: bool = True
    cublas_workspace: str = _CUBLAS_WORKSPACE
    noise_stream: str = "cpu_generator_v1"

    def __post_init__(self) -> None:
        if self.device not in ("cuda", "cpu"):
            raise ValueError("numerics device must be 'cuda' or 'cpu'")
        if self.compile not in (None, "inductor"):
            raise ValueError("numerics compile must be None or 'inductor'")
        if self.compile_mode != "default":
            raise ValueError("numerics compile_mode is pinned to 'default'")
        if self.tf32:
            raise ValueError("tf32 must stay off in every numerics regime")
        if self.noise_stream != "cpu_generator_v1":
            raise ValueError("unsupported noise_stream")
        if self.device == "cuda" and (
            not self.gpu or not self.cuda_version or not self.driver_version
        ):
            raise ValueError("cuda numerics require gpu, cuda and driver versions")

    @property
    def regime(self) -> str:
        return f"{self.device}_{'inductor' if self.compile else 'eager'}_v2"


# The one blessed v2 GPU environment on this machine; every identity field is
# exact-match guarded at apply time.  A driver or torch upgrade must fork
# digests loudly, never drift silently.
PINNED_NUMERICS_V2 = NumericsSpec(
    device="cuda",
    gpu="NVIDIA GeForce RTX 3090",
    torch_version="2.7.1+cu126",
    cuda_version="12.6",
    driver_version="610.47",
)


def numerics_identity(spec: NumericsSpec) -> dict[str, Any]:
    """JSON-safe identity descriptor; key order is irrelevant (canonical hash)."""
    return {
        "regime": spec.regime,
        "device": spec.device,
        "gpu": spec.gpu,
        "torch": spec.torch_version,
        "cuda": spec.cuda_version,
        "driver": spec.driver_version,
        "compile": spec.compile,
        "compile_mode": spec.compile_mode,
        "tf32": spec.tf32,
        "deterministic_algorithms": spec.deterministic_algorithms,
        "cublas_workspace": spec.cublas_workspace,
        "noise_stream": spec.noise_stream,
    }


def _driver_version() -> str | None:
    # WSL2 exposes the Windows driver through the /usr/lib/wsl/lib nvidia-smi
    # shim; /proc/driver/nvidia/version is unreliable under GPU-PV.
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return output[0].strip() if output else None
    except Exception:
        return None


def resolve_default_numerics() -> NumericsSpec | None:
    """Library default: the pinned v2 GPU regime when CUDA is live, else legacy."""
    if os.environ.get(_ENV_OVERRIDE, "auto") == "legacy":
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return PINNED_NUMERICS_V2


def resolve_numerics(device: str = "auto", *, compile: bool = True) -> NumericsSpec | None:
    """CLI-facing resolver.

    'auto'  -> pinned v2 GPU regime when CUDA is live, else legacy (None);
    'cuda'  -> pinned v2 (hard error later if the environment mismatches);
    'cpu'   -> legacy (None) when compile is disabled, else the cpu-inductor
               probe regime (used by the compile-parity validation lane).
    """
    if device == "auto":
        spec = resolve_default_numerics()
        if spec is not None and not compile:
            spec = replace(spec, compile=None)
        return spec
    if device == "cuda":
        spec = PINNED_NUMERICS_V2
        return replace(spec, compile=None) if not compile else spec
    if device == "cpu":
        if not compile:
            return None
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("cpu-inductor numerics require torch") from exc
        return NumericsSpec(
            # str(): torch.__version__ is a TorchVersion instance whose class
            # would poison weights_only=True checkpoint loading if pickled.
            device="cpu", gpu=None, torch_version=str(torch.__version__),
            cuda_version=None, driver_version=None, compile="inductor",
        )
    raise ValueError(f"unknown numerics device {device!r}")


def require_numerics(spec: NumericsSpec) -> None:
    """Error unless the live environment matches the spec exactly."""
    import torch

    problems: list[str] = []
    if torch.__version__ != spec.torch_version:
        problems.append(f"torch {torch.__version__} != {spec.torch_version}")
    if spec.device == "cuda":
        if not torch.cuda.is_available():
            problems.append("CUDA unavailable")
        else:
            if torch.version.cuda != spec.cuda_version:
                problems.append(f"cuda {torch.version.cuda} != {spec.cuda_version}")
            name = torch.cuda.get_device_name(0)
            if name != spec.gpu:
                problems.append(f"gpu {name!r} != {spec.gpu!r}")
            driver = _driver_version()
            if driver is None:
                problems.append("driver version unresolvable (nvidia-smi failed); "
                                "regime-v2 evidence requires a provable driver")
            elif driver != spec.driver_version:
                problems.append(f"driver {driver} != {spec.driver_version}")
    if problems:
        raise RuntimeError(
            "live environment does not match the pinned numerics regime "
            f"({spec.regime}): " + "; ".join(problems) +
            " — update PINNED_NUMERICS_V2 deliberately (forking digests) or "
            "run with --legacy-numerics; see notes/numerics-regime-v2.md"
        )


def apply_numerics(torch: Any, spec: NumericsSpec | None, *, seed: int) -> Any:
    """Seed + configure the process for the regime; returns the torch.device.

    Legacy (``spec is None``) reproduces the historical setup lines exactly,
    in the historical order — byte-identical trainings.
    """
    if spec is None:
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        np.random.seed(seed)
        return torch.device("cpu")
    require_numerics(spec)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", spec.cublas_workspace)
    # WSL2 wipes /tmp per boot; keep compiled kernels across sessions.
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _INDUCTOR_CACHE_DIR)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    np.random.seed(seed)
    if spec.device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = spec.tf32
        torch.backends.cudnn.allow_tf32 = spec.tf32
        torch.backends.cudnn.benchmark = False
    if spec.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    return torch.device(spec.device)


def noise_generator(torch: Any, seed: int) -> Any:
    """cpu_generator_v1: a dedicated CPU generator, so the noise stream is
    device-invariant and insulated from any global-RNG consumers."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def cpu_state_dict(model: Any) -> dict[str, Any]:
    """State dict of the original module (compile-wrapper unwrapped), on CPU."""
    original = getattr(model, "_orig_mod", model)
    return {name: value.detach().cpu() for name, value in original.state_dict().items()}


__all__ = [
    "AUTO",
    "NumericsSpec",
    "PINNED_NUMERICS_V2",
    "apply_numerics",
    "cpu_state_dict",
    "noise_generator",
    "numerics_identity",
    "require_numerics",
    "resolve_default_numerics",
    "resolve_numerics",
]
