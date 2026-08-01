"""Strict 25 mm cube opposing-face grasp detector ported from legacy evidence."""

from __future__ import annotations

from typing import Any

import numpy as np


FACE_GRASP_MIN_FORCE = 0.1
FACE_GRASP_CUBE_HALF_EXTENT = 0.0125
FACE_GRASP_CORNER_MARGIN = 0.002
FACE_GRASP_NORMAL_COSINE = float(np.cos(np.deg2rad(25.0)))
FACE_GRASP_SURFACE_TOLERANCE = 0.002


def evaluate_face_grasp_contacts(
    contacts: list[dict[str, Any]],
    jaw_axis_local: np.ndarray,
    *,
    min_force: float = FACE_GRASP_MIN_FORCE,
    cube_half_extent: float = FACE_GRASP_CUBE_HALF_EXTENT,
    corner_margin: float = FACE_GRASP_CORNER_MARGIN,
    normal_cosine: float = FACE_GRASP_NORMAL_COSINE,
    surface_tolerance: float = FACE_GRASP_SURFACE_TOLERANCE,
) -> tuple[bool, float, dict[str, Any]]:
    jaw_axis = np.asarray(jaw_axis_local, dtype=np.float64)
    jaw_axis_norm = float(np.linalg.norm(jaw_axis))
    empty = {
        "face_alignment": 0.0, "face_opposition": 0.0,
        "face_jaw_axis_alignment": 0.0, "face_corner_rejection": False,
        "face_corner_rejection_count": 0, "interior_face_contact": False,
        "interior_face_contact_score": 0.0, "interior_face_contact_count": 0,
        "face_contact_quality": 0.0, "fixed_interior_face_contact": False,
        "moving_interior_face_contact": False, "bilateral_interior_face_contact": False,
        "bilateral_opposition_quality": 0.0,
    }
    if jaw_axis.shape != (3,) or jaw_axis_norm <= 1e-9:
        return False, 0.0, empty
    jaw_axis /= jaw_axis_norm
    usable: dict[str, list[dict[str, Any]]] = {"fixed": [], "moving": []}
    corner_rejections = 0
    best_face_alignment = best_jaw_alignment = best_contact_quality = 0.0
    for contact in contacts:
        side = contact.get("side")
        if side not in usable:
            continue
        position = np.asarray(contact["position_local"], dtype=np.float64)
        normal = np.asarray(contact["normal_local"], dtype=np.float64)
        normal_norm = float(np.linalg.norm(normal))
        if position.shape != (3,) or normal.shape != (3,) or normal_norm <= 1e-9:
            continue
        normal /= normal_norm
        if float(np.dot(normal, position)) < 0.0:
            normal = -normal
        face_axis = int(np.argmax(np.abs(position)))
        face_sign = 1.0 if position[face_axis] >= 0 else -1.0
        expected = np.zeros(3, dtype=np.float64)
        expected[face_axis] = face_sign
        face_alignment = float(np.dot(normal, expected))
        jaw_alignment = float(abs(np.dot(normal, jaw_axis)))
        best_face_alignment = max(best_face_alignment, face_alignment)
        best_jaw_alignment = max(best_jaw_alignment, jaw_alignment)
        tangential = np.delete(np.abs(position), face_axis)
        edge_clearance = cube_half_extent - float(np.max(tangential))
        quality = float(
            np.clip(face_alignment, 0.0, 1.0)
            * np.clip(jaw_alignment, 0.0, 1.0)
            * np.clip(edge_clearance / max(corner_margin, 1e-9), 0.0, 1.0)
        )
        best_contact_quality = max(best_contact_quality, quality)
        if not bool(np.all(tangential <= cube_half_extent - corner_margin)):
            corner_rejections += 1
            continue
        on_surface = abs(abs(float(position[face_axis])) - cube_half_extent) <= surface_tolerance
        if not on_surface or face_alignment < normal_cosine or jaw_alignment < normal_cosine:
            continue
        usable[side].append({
            "normal": normal, "axis": face_axis, "sign": face_sign,
            "force": max(0.0, float(contact.get("force", 0.0))),
            "face_alignment": face_alignment, "jaw_alignment": jaw_alignment,
        })
    best_pair: tuple[Any, ...] | None = None
    best_opposition = 0.0
    for fixed in usable["fixed"]:
        for moving in usable["moving"]:
            opposition = float(-np.dot(fixed["normal"], moving["normal"]))
            best_opposition = max(best_opposition, opposition)
            if fixed["axis"] != moving["axis"] or fixed["sign"] != -moving["sign"] or opposition < normal_cosine:
                continue
            score = min(fixed["face_alignment"], moving["face_alignment"], fixed["jaw_alignment"], moving["jaw_alignment"], opposition)
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, fixed["axis"], fixed["sign"], moving["sign"], min(fixed["face_alignment"], moving["face_alignment"]), min(fixed["jaw_alignment"], moving["jaw_alignment"]), opposition)
    fixed_force = moving_force = 0.0
    if best_pair is not None:
        _, axis, fixed_sign, moving_sign, _, _, _ = best_pair
        fixed_force = sum(item["force"] for item in usable["fixed"] if item["axis"] == axis and item["sign"] == fixed_sign)
        moving_force = sum(item["force"] for item in usable["moving"] if item["axis"] == axis and item["sign"] == moving_sign)
    gripped = best_pair is not None and fixed_force > min_force and moving_force > min_force
    interior = usable["fixed"] + usable["moving"]
    diagnostics = {
        "face_alignment": float(best_pair[4] if best_pair else best_face_alignment),
        "face_opposition": float(best_pair[6] if best_pair else best_opposition),
        "face_jaw_axis_alignment": float(best_pair[5] if best_pair else best_jaw_alignment),
        "face_corner_rejection": corner_rejections > 0,
        "face_corner_rejection_count": corner_rejections,
        "interior_face_contact": bool(interior),
        "interior_face_contact_score": float(max((min(item["face_alignment"], item["jaw_alignment"], np.clip(item["force"] / min_force, 0, 1)) for item in interior), default=0.0)),
        "interior_face_contact_count": len(interior),
        "face_contact_quality": best_contact_quality,
        "fixed_interior_face_contact": bool(usable["fixed"]),
        "moving_interior_face_contact": bool(usable["moving"]),
        "bilateral_interior_face_contact": bool(usable["fixed"] and usable["moving"]),
        "bilateral_opposition_quality": best_opposition,
    }
    return gripped, min(fixed_force, moving_force) if gripped else 0.0, diagnostics


def check_block_face_gripped(model: Any, data: Any, block_name: str = "red_block") -> tuple[bool, float, dict[str, Any]]:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MuJoCo grasp detection requires the 'sim' extra") from exc
    block_id = model.body(block_name).id
    fixed_id = model.body("gripper").id
    moving_id = model.body("moving_jaw_so101_v1").id
    block_pos = data.body(block_name).xpos.copy()
    rotation = np.asarray(data.body(block_name).xmat).reshape(3, 3)
    jaw_axis_local = rotation.T @ (data.site("moving_jaw_tip").xpos - data.site("fixed_jaw_tip").xpos)
    contacts: list[dict[str, Any]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if block_id not in bodies:
            continue
        side = "fixed" if fixed_id in bodies else "moving" if moving_id in bodies else None
        if side is None:
            continue
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, wrench)
        contacts.append({
            "side": side,
            "position_local": rotation.T @ (np.asarray(contact.pos) - block_pos),
            "normal_local": rotation.T @ np.asarray(contact.frame[:3], dtype=np.float64),
            "force": float(np.linalg.norm(wrench[:3])),
        })
    return evaluate_face_grasp_contacts(contacts, jaw_axis_local)


__all__ = ["check_block_face_gripped", "evaluate_face_grasp_contacts"]
