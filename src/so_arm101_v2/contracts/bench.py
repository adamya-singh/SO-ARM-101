"""Versioned bench geometry and extra physical limits, without renormalization."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .appearance import AppearanceRegime
from .joint_map import KNOWN_JOINT_MAPS, load_joint_map
from .lens import LensModel
from .physical import physical_normalized_to_act


@dataclass(frozen=True)
class BenchConfig:
    schema_version: int = 1
    task_id: str = "bench_pick_replace_v1"
    shoulder_floor: float = -92.0
    cube_edge_m: float = 0.020
    square_edge_m: float = 0.0508
    distance_reference: str = "base_front_edge"
    measured_forward_distance_m: float = 0.2159
    base_front_edge_y_m: float = 0.0646353
    square_center_xy: tuple[float, float] = (0.0, 0.2805353)
    square_thickness_m: float = 0.001
    reset_physical: tuple[float, ...] | None = None
    reset_evidence_sha256: str | None = None
    viewing_qpos: tuple[float, ...] | None = None
    observation_steps: int = 90
    approach_pitch_deg: float = 50.0
    grasp_pad: int = 1
    # Teacher tuning that changes the captured data, so it is versioned and
    # hashed here rather than living as code defaults.
    depth_lead_m: float = 0.006
    grasp_offset_m: float = 0.0085
    # Wrist camera in the gripper frame (MuJoCo camera convention), from the
    # 2026-09-08 hand-eye calibration; None = keep the Menagerie mount camera.
    camera_pos: tuple | None = None
    camera_quat_wxyz: tuple | None = None
    camera_fovy_deg: float | None = None
    # Physical-to-MuJoCo joint map. The legacy affine map put three joints a
    # quarter turn off; the bench lane uses the encoder-anchored measured map.
    joint_map: str = "measured_20260908b"
    # Physical lens model (contracts/lens.py): when set, the simulator renders
    # a wider pinhole (fovy = lens.render_fovy_deg) and resamples it into the
    # observation as the real lens would image it; None = plain 256x256 pinhole.
    lens: dict | None = None
    # Appearance randomization regime (contracts/appearance.py): ranges for the per-scenario
    # look drawn at suite generation; None = every scenario renders the pristine scene.
    appearance: dict | None = None

    def __post_init__(self):
        if self.schema_version != 1 or self.task_id != "bench_pick_replace_v1":
            raise ValueError("unsupported bench configuration")
        if self.shoulder_floor != -92.0:
            raise ValueError("bench shoulder floor must be -92 calibrated units")
        if (self.distance_reference != "base_front_edge"
                or self.measured_forward_distance_m != 0.2159
                or self.base_front_edge_y_m != 0.0646353):
            raise ValueError("bench distance must use the measured base-front-edge reference")
        if (self.cube_edge_m, self.square_edge_m, self.square_center_xy,
                self.square_thickness_m) != (0.020, 0.0508, (0.0, 0.2805353), 0.001):
            raise ValueError("bench geometry differs from the agreed task")
        if self.observation_steps != 90:
            raise ValueError("bench observation prefix must align with the first H90 image refresh")
        # 0 deg = jaws pointing straight down; 90 would be horizontal.
        if not np.isfinite(self.approach_pitch_deg) or not 0 <= self.approach_pitch_deg <= 85:
            raise ValueError("invalid teacher approach pitch")
        if self.joint_map not in KNOWN_JOINT_MAPS or self.joint_map == "legacy_affine_v1":
            raise ValueError("bench requires a measured joint map (the legacy affine map misplaces the arm)")
        if self.grasp_pad not in (1, 2, 3, 4):
            raise ValueError("unsupported grasp pad")
        for name in ("depth_lead_m", "grasp_offset_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or not -0.02 <= value <= 0.02:
                raise ValueError(f"bench {name} must be a finite offset within 20 mm")
        if self.lens is not None:
            lens = self.lens_model
            if self.camera_fovy_deg is None or abs(float(self.camera_fovy_deg) - lens.fovy_deg) > 0.05:
                raise ValueError("camera_fovy_deg must equal the lens model's vertical field of view")
            if not lens.fovy_deg <= lens.render_fovy_deg <= 120.0:
                raise ValueError("lens render fovy must cover the physical field of view and stay below 120 deg")
            if abs(lens.render_size[0] * 9 - lens.render_size[1] * 16) > 16:
                raise ValueError("lens render size must be 16:9 like the physical frame")
            object.__setattr__(self, "lens", lens.identity())
        if self.appearance is not None:
            object.__setattr__(self, "appearance", AppearanceRegime.from_mapping(self.appearance).identity())
        if self.reset_physical is not None:
            p = np.asarray(self.reset_physical, dtype=np.float64)
            if p.shape != (6,) or not np.isfinite(p).all() or p[1] < self.shoulder_floor:
                raise ValueError("invalid physical reset or shoulder below floor")
            self.validate_qpos(self.joint_map_object.act_to_mujoco(physical_normalized_to_act(p)))
        if self.viewing_qpos is not None:
            self.validate_qpos(self.viewing_qpos)

    @property
    def joint_map_object(self):
        return load_joint_map(self.joint_map)

    @property
    def lens_model(self):
        return None if self.lens is None else LensModel.from_mapping(self.lens)

    @property
    def appearance_regime(self):
        return None if self.appearance is None else AppearanceRegime.from_mapping(self.appearance)

    @property
    def mujoco_low(self):
        low = self.joint_map_object.mujoco_low.copy()
        p = np.zeros(6)
        p[1] = self.shoulder_floor
        low[1] = max(low[1], self.joint_map_object.act_to_mujoco(physical_normalized_to_act(p))[1])
        return low

    @property
    def mujoco_high(self):
        return self.joint_map_object.mujoco_high.copy()

    def validate_qpos(self, qpos, tolerance=1e-5):
        q = np.asarray(qpos, dtype=np.float64)
        if (q.shape != (6,) or not np.isfinite(q).all()
                or np.any(q < self.mujoco_low - tolerance) or np.any(q > self.mujoco_high + tolerance)):
            raise ValueError("bench pose exceeds effective joint bounds; do not clip")

    @property
    def reset_qpos(self):
        if self.reset_physical is None:
            raise ValueError("bench reset has not been measured and verified")
        return self.joint_map_object.act_to_mujoco(physical_normalized_to_act(self.reset_physical))

    @property
    def cube_center(self):
        return (*self.square_center_xy, self.square_thickness_m + self.cube_edge_m / 2)

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text())
        for key in ("square_center_xy", "reset_physical", "viewing_qpos"):
            if raw.get(key) is not None:
                raw[key] = tuple(raw[key])
        return cls(**raw)


def scene_bench_config(model_path):
    """Bench scenes require a sibling configuration; legacy scenes stay unchanged."""
    path = Path(model_path)
    if path.name != "scene_bench_pick_replace_v1.xml":
        return None
    return BenchConfig.load(path.with_name("bench_config.json"))


def scene_dependency_hash(model_path):
    """Hash all XML includes/assets plus the bench config, independent of location."""
    import xml.etree.ElementTree as ET
    root = Path(model_path).resolve()
    seen = set()
    digest = hashlib.sha256()

    def visit(path, meshdir=""):
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        data = path.read_bytes()
        digest.update(hashlib.sha256(data).digest())
        if path.suffix == ".xml":
            node = ET.fromstring(data)
            compiler = node.find("compiler")
            if compiler is not None:
                meshdir = compiler.get("meshdir", meshdir)
            for element in node.iter():
                name = element.get("file")
                if name:
                    relative = Path(meshdir) / name if element.tag == "mesh" else Path(name)
                    visit(path.parent / relative, meshdir)

    visit(root)
    if scene_bench_config(root) is not None:
        visit(root.with_name("bench_config.json"))
    return digest.hexdigest()
