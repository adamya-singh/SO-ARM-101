"""Versioned physical-to-MuJoCo joint maps.

Two maps exist and both are explicit:

``LEGACY_JOINT_MAP`` is the June 2026 affine endpoint mapping (``coordinates``
module functions, bit-identical). It assumed each LeRobot calibrated tick range
spans the same physical angle as the model joint range. That assumption was
measured false on 2026-09-07: spans differ by up to 37 percent and the
shoulder-lift, elbow and wrist-roll zeros were each a quarter turn off. The
legacy lane (25 mm cube, every August artifact) keeps this map so its evidence
stays reproducible.

``measured_20260908`` (``load_joint_map``; ``measured_20260907`` kept as its
predecessor with the hand-held zeros) uses the exact 4096 ticks/turn
encoder scale and per-joint zero offsets read from the arm posed by hand at
the model's zero configuration. The bench scene binds to it through
``BenchConfig.joint_map``. The gripper channel keeps the legacy affine map in
both, because the model cannot represent touching jaws.

ACT units stay defined as servo-normalized units (act = normalized/100*pi for
body joints, normalized/100*1.7 for the gripper); only the ACT<->MuJoCo leg
differs between maps.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from so_arm101_v2.data.resources import load_json_resource, read_resource_bytes

from . import coordinates as C
from .physical import act_to_physical_normalized, physical_normalized_to_act

KNOWN_JOINT_MAPS = ("legacy_affine_v1", "measured_20260907", "measured_20260908")
_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class JointMap:
    name: str
    resource_sha256: str | None
    mujoco_low: np.ndarray
    mujoco_high: np.ndarray
    _zero_ticks: np.ndarray | None = None
    _sign: np.ndarray | None = None
    _range_min: np.ndarray | None = None
    _range_max: np.ndarray | None = None
    _ticks_per_turn: int = 4096

    @property
    def legacy(self) -> bool:
        return self._zero_ticks is None

    def act_to_mujoco(self, values: Any) -> np.ndarray:
        if self.legacy:
            return C.act_to_mujoco_qpos(values)
        act = np.asarray(values, dtype=np.float64)
        legacy = C.act_to_mujoco_qpos(act)  # gripper channel
        normalized = np.asarray(act_to_physical_normalized(act), dtype=np.float64)
        ticks = self._range_min + (normalized[..., :5] + 100.0) / 200.0 * (self._range_max - self._range_min)
        body = self._sign * (ticks - self._zero_ticks) * (_TWO_PI / self._ticks_per_turn)
        result = np.concatenate([body, legacy[..., 5:].astype(np.float64)], axis=-1)
        return np.asarray(result, dtype=np.float32)

    def mujoco_to_act(self, values: Any) -> np.ndarray:
        if self.legacy:
            return C.mujoco_qpos_to_act(values)
        qpos = C._as_joint_values(values, "MuJoCo qpos").astype(np.float64)
        legacy = C.mujoco_qpos_to_act(qpos)  # gripper channel
        ticks = self._zero_ticks + qpos[..., :5] / self._sign * (self._ticks_per_turn / _TWO_PI)
        normalized = (ticks - self._range_min) / (self._range_max - self._range_min) * 200.0 - 100.0
        full = np.concatenate([normalized, np.zeros_like(normalized[..., :1])], axis=-1)
        act = np.asarray(physical_normalized_to_act(full), dtype=np.float64)
        result = np.concatenate([act[..., :5], legacy[..., 5:].astype(np.float64)], axis=-1)
        return np.asarray(result, dtype=np.float32)

    def clip_mujoco(self, values: Any) -> tuple[np.ndarray, np.ndarray]:
        source = C._as_joint_values(values, "MuJoCo targets")
        clipped = np.clip(source, self.mujoco_low, self.mujoco_high).astype(np.float32, copy=False)
        return clipped, np.not_equal(clipped, source)

    def provenance(self) -> dict[str, Any]:
        return {"name": self.name, "resource_sha256": self.resource_sha256}


LEGACY_JOINT_MAP = JointMap(
    name="legacy_affine_v1", resource_sha256=None,
    mujoco_low=C.MUJOCO_JOINT_LOW, mujoco_high=C.MUJOCO_JOINT_HIGH,
)

_CACHE: dict[str, JointMap] = {}


def load_joint_map(name: str = "measured_20260908") -> JointMap:
    """Return a named joint map; ``legacy_affine_v1`` or a measured resource."""
    if name == "legacy_affine_v1":
        return LEGACY_JOINT_MAP
    if name not in KNOWN_JOINT_MAPS:
        raise ValueError(f"unknown joint map {name!r}; expected one of {KNOWN_JOINT_MAPS}")
    if name in _CACHE:
        return _CACHE[name]
    resource_name = f"physical_joint_map_{name.split('_', 1)[1]}.json"
    raw = load_json_resource(resource_name)
    if raw["name"] != name or raw["gripper"] != "legacy_affine_endpoints":
        raise ValueError(f"joint map resource {resource_name} does not describe {name!r}")
    joints = raw["joints"]
    names = C.JOINT_NAMES[:5]
    if tuple(joints) != names:
        raise ValueError("joint map resource must list the five body joints in recording order")
    if any(joints[n]["drive_mode"] != 0 for n in names):
        raise ValueError("joint map assumes drive_mode 0 for every body joint (matches the pinned calibration)")
    zero = np.array([joints[n]["model_zero_ticks"] for n in names], dtype=np.float64)
    sign = np.array([joints[n]["sign"] for n in names], dtype=np.float64)
    if not np.all(np.abs(sign) == 1.0):
        raise ValueError("joint map signs must be +1 or -1")
    rmin = np.array([joints[n]["range_min"] for n in names], dtype=np.float64)
    rmax = np.array([joints[n]["range_max"] for n in names], dtype=np.float64)
    # Envelope: the physical +-100 normalized range through the map, intersected
    # with the Menagerie mechanical limits; gripper from the legacy map.
    partial = JointMap(name=name, resource_sha256=None, mujoco_low=C.MUJOCO_JOINT_LOW, mujoco_high=C.MUJOCO_JOINT_HIGH,
                       _zero_ticks=zero, _sign=sign, _range_min=rmin, _range_max=rmax, _ticks_per_turn=int(raw["ticks_per_turn"]))
    ends = np.stack([partial.act_to_mujoco(physical_normalized_to_act([-100] * 5 + [0])),
                     partial.act_to_mujoco(physical_normalized_to_act([100] * 5 + [100]))]).astype(np.float64)
    low = np.maximum(ends.min(axis=0), C._MENAGERIE_MECHANICAL_LOW).astype(np.float32)
    high = np.minimum(ends.max(axis=0), C._MENAGERIE_MECHANICAL_HIGH).astype(np.float32)
    low[5], high[5] = C.MUJOCO_JOINT_LOW[5], C.MUJOCO_JOINT_HIGH[5]
    low.setflags(write=False); high.setflags(write=False)
    result = JointMap(name=name, resource_sha256=hashlib.sha256(read_resource_bytes(resource_name)).hexdigest(),
                      mujoco_low=low, mujoco_high=high, _zero_ticks=zero, _sign=sign, _range_min=rmin, _range_max=rmax,
                      _ticks_per_turn=int(raw["ticks_per_turn"]))
    _CACHE[name] = result
    return result


__all__ = ["JointMap", "KNOWN_JOINT_MAPS", "LEGACY_JOINT_MAP", "load_joint_map"]
