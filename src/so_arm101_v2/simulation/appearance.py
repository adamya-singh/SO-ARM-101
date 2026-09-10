"""Apply a resolved appearance draw to a MuJoCo model (render-side fields only).

The numpy regime and resolver live in ``contracts.appearance``; this module is
the MuJoCo side: resolve the scene's appearance slots by name, snapshot the
pristine values so any draw can be undone exactly, and write a draw into the
model. Nothing here touches physics: no ``geom_size`` of a colliding geom, no
poses of physical bodies, no contact parameters. The towel is a visual-only
body the scene generator appends last (``tools/prepare_bench_scene.py``).

Texture *binding* (``mat_texid``, ``mat_texrepeat``, ``mat_texuniform``) and
texel data are baked into a render context when it is created, so the caller
(``MujocoTaskAdapter``) closes its renderers before ``apply_appearance`` /
``restore`` and lets them be recreated. Light and camera derived quantities
are refreshed by the ``mj_forward`` the adapter's reset already runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from so_arm101_v2.contracts.appearance import AppearanceParams, MATERIAL_NAMES, ground_texture, skybox_texture

GROUND_TEXTURE_NAME = "ground_speckle"
TOWEL_BODY_NAME = "towel_visual"
TOWEL_GEOM_NAME = "towel_visual_geom"
NAPKIN_GEOM_NAME = "napkin"
WRIST_CAMERA_NAME = "wrist_camera"
GROUND_MATERIAL_NAME = "groundplane"
MAX_RENDER_FOVY_DEG = 119.0


def _mujoco() -> Any:
    import mujoco
    return mujoco


@dataclass(frozen=True)
class SceneAppearanceIds:
    materials: dict[str, int]
    napkin_geom: int
    wrist_camera: int
    light: int
    skybox_texture: int          # -1 when the scene has no skybox
    ground_texture: int          # -1 when the scene lacks the slot
    towel_body: int              # -1 when the scene lacks the slot
    towel_geom: int

    @property
    def has_slots(self) -> bool:
        return self.ground_texture >= 0 and self.towel_body >= 0 and self.towel_geom >= 0


def resolve_scene_ids(model: Any) -> SceneAppearanceIds:
    mujoco = _mujoco()
    name2id = lambda kind, name: int(mujoco.mj_name2id(model, kind, name))
    materials = {name: name2id(mujoco.mjtObj.mjOBJ_MATERIAL, name) for name in MATERIAL_NAMES}
    missing = [name for name, index in materials.items() if index < 0]
    if missing:
        raise ValueError(f"scene lacks appearance materials {missing}")
    napkin = name2id(mujoco.mjtObj.mjOBJ_GEOM, NAPKIN_GEOM_NAME)
    camera = name2id(mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA_NAME)
    if napkin < 0 or camera < 0 or model.nlight < 1:
        raise ValueError("scene lacks the napkin geom, the wrist camera or a light")
    skybox = -1
    for index in range(model.ntex):
        if int(model.tex_type[index]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
            skybox = index
            break
    return SceneAppearanceIds(
        materials=materials, napkin_geom=napkin, wrist_camera=camera, light=0, skybox_texture=skybox,
        ground_texture=name2id(mujoco.mjtObj.mjOBJ_TEXTURE, GROUND_TEXTURE_NAME),
        towel_body=name2id(mujoco.mjtObj.mjOBJ_BODY, TOWEL_BODY_NAME),
        towel_geom=name2id(mujoco.mjtObj.mjOBJ_GEOM, TOWEL_GEOM_NAME),
    )


_MODEL_ARRAYS = ("mat_rgba", "mat_specular", "mat_shininess", "mat_reflectance", "mat_emission", "mat_texid",
                 "mat_texrepeat", "mat_texuniform", "geom_rgba", "geom_size", "body_pos", "body_quat",
                 "light_dir", "light_diffuse", "light_ambient", "light_specular", "light_castshadow", "light_active",
                 "cam_pos", "cam_quat", "cam_fovy", "tex_data")
_HEADLIGHT_FIELDS = ("ambient", "diffuse", "specular", "active")


@dataclass(frozen=True)
class ModelAppearanceSnapshot:
    """Exact copies of every model field the applier may write; ``restore`` undoes any draw."""

    arrays: dict[str, np.ndarray]
    headlight: dict[str, Any]

    @classmethod
    def capture(cls, model: Any) -> "ModelAppearanceSnapshot":
        arrays = {name: np.array(getattr(model, name), copy=True) for name in _MODEL_ARRAYS}
        headlight = {name: np.array(getattr(model.vis.headlight, name), copy=True) for name in _HEADLIGHT_FIELDS}
        return cls(arrays=arrays, headlight=headlight)

    def restore(self, model: Any) -> None:
        for name, value in self.arrays.items():
            getattr(model, name)[...] = value
        for name, value in self.headlight.items():
            setattr(model.vis.headlight, name, value)

    def matches(self, model: Any) -> bool:
        return all(np.array_equal(getattr(model, name), value) for name, value in self.arrays.items()) and all(
            np.array_equal(getattr(model.vis.headlight, name), value) for name, value in self.headlight.items())


def _write_texture(model: Any, texture_id: int, pixels: np.ndarray) -> None:
    width, height, channels = int(model.tex_width[texture_id]), int(model.tex_height[texture_id]), int(model.tex_nchannel[texture_id])
    if pixels.shape != (height, width, channels) or pixels.dtype != np.uint8:
        raise ValueError(f"texture {texture_id} expects uint8 {height}x{width}x{channels}, got {pixels.shape} {pixels.dtype}")
    start = int(model.tex_adr[texture_id])
    model.tex_data[start:start + pixels.size] = pixels.reshape(-1)


def apply_appearance(model: Any, params: AppearanceParams, ids: SceneAppearanceIds, pristine: ModelAppearanceSnapshot) -> None:
    """Write ``params`` into ``model`` (starting from the pristine snapshot). Renderers must be recreated afterwards."""
    mujoco = _mujoco()
    pristine.restore(model)
    if (params.towel is not None or params.ground_texture is not None) and not ids.has_slots:
        raise ValueError("scene lacks the appearance slots (ground_speckle texture, towel_visual body); "
                         "regenerate it with tools/prepare_bench_scene.py")
    # Headlight and the key light.
    model.vis.headlight.ambient = np.full(3, params.headlight_ambient, dtype=np.float32)
    model.vis.headlight.diffuse = np.full(3, params.headlight_diffuse, dtype=np.float32)
    model.vis.headlight.specular = np.full(3, params.headlight_specular, dtype=np.float32)
    light = ids.light
    model.light_dir[light] = np.asarray(params.light_dir, dtype=np.float64)
    model.light_diffuse[light] = np.full(3, params.light_diffuse, dtype=np.float32)
    model.light_ambient[light] = np.full(3, params.light_ambient, dtype=np.float32)
    model.light_specular[light] = np.full(3, params.light_specular, dtype=np.float32)
    model.light_castshadow[light] = bool(params.castshadow)
    # Materials.
    for name, rgb, specular, shininess, reflectance in params.materials:
        index = ids.materials[name]
        model.mat_rgba[index, :3] = np.asarray(rgb, dtype=np.float32)
        model.mat_specular[index] = float(specular)
        model.mat_shininess[index] = float(shininess)
        model.mat_reflectance[index] = float(reflectance)
    model.geom_rgba[ids.napkin_geom, :3] = np.asarray(params.napkin_rgb, dtype=np.float32)
    # Towel: visual-only body under the napkin, sized relative to the square.
    if params.towel is not None:
        scale_x, scale_y, yaw, rgb = params.towel
        pristine_size = pristine.arrays["geom_size"][ids.towel_geom]
        model.geom_size[ids.towel_geom, 0] = float(pristine_size[0] * scale_x)
        model.geom_size[ids.towel_geom, 1] = float(pristine_size[1] * scale_y)
        model.body_quat[ids.towel_body] = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float64)
        model.geom_rgba[ids.towel_geom] = np.array([*rgb, 1.0], dtype=np.float32)
    # Ground texture: darkness baked into the texels, material set to white so the texture shows as drawn.
    if params.ground_texture is not None:
        texture_seed, contrast, repeat_per_m = params.ground_texture
        ground = ids.materials[GROUND_MATERIAL_NAME]
        base_rgb = np.array(model.mat_rgba[ground, :3], copy=True)
        width, height = int(model.tex_width[ids.ground_texture]), int(model.tex_height[ids.ground_texture])
        _write_texture(model, ids.ground_texture, ground_texture(texture_seed, width, height, contrast=contrast, base_rgb=base_rgb))
        model.mat_texid[ground, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = ids.ground_texture
        model.mat_texuniform[ground] = True
        model.mat_texrepeat[ground] = (float(repeat_per_m), float(repeat_per_m))
        model.mat_rgba[ground, :3] = 1.0
    # Skybox recolour ("off" is a per-renderer flag the adapter sets at render time).
    if params.skybox != "off" and ids.skybox_texture >= 0:
        top, bottom = params.skybox
        width, height = int(model.tex_width[ids.skybox_texture]), int(model.tex_height[ids.skybox_texture])
        _write_texture(model, ids.skybox_texture, skybox_texture(top, bottom, width, height))
    # Camera nuisance: position offset, small rotation composed onto the calibrated mount, fovy scale.
    camera = ids.wrist_camera
    model.cam_pos[camera] = pristine.arrays["cam_pos"][camera] + np.asarray(params.camera_pos_offset, dtype=np.float64)
    jitter = np.zeros(4)
    mujoco.mju_euler2Quat(jitter, np.deg2rad(np.asarray(params.camera_euler_deg, dtype=np.float64)), "xyz")
    composed = np.zeros(4)
    mujoco.mju_mulQuat(composed, pristine.arrays["cam_quat"][camera], jitter)
    model.cam_quat[camera] = composed
    model.cam_fovy[camera] = min(float(pristine.arrays["cam_fovy"][camera]) * float(params.fovy_scale), MAX_RENDER_FOVY_DEG)


__all__ = ["GROUND_TEXTURE_NAME", "ModelAppearanceSnapshot", "SceneAppearanceIds", "TOWEL_BODY_NAME", "TOWEL_GEOM_NAME",
           "apply_appearance", "resolve_scene_ids"]
