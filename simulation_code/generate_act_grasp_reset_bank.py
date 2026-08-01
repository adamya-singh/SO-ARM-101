#!/usr/bin/env python3
"""Generate curriculum resets proven by the production environment dynamics."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

from act_coordinate_utils import MUJOCO_JOINT_HIGH, MUJOCO_JOINT_LOW, act_to_mujoco_qpos
from so101_gym_env import SO101PickPlaceEnv
from so101_mujoco_utils import check_block_face_gripped


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR.parent / "imitation-learning/datasets/so101_pickplace_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states-per-stage", type=int, default=32)
    parser.add_argument("--seed", type=int, default=260713)
    parser.add_argument("--max-attempts", type=int, default=4096)
    return parser.parse_args()


def load_demo_states(dataset: Path) -> np.ndarray:
    encoded = []
    for path in sorted(glob.glob(str(dataset / "data/**/*.parquet"), recursive=True)):
        encoded.extend(pq.read_table(path, columns=["observation.state"]).column(0).to_pylist())
    if not encoded:
        raise RuntimeError(f"No demonstration parquet rows found under {dataset}")
    return act_to_mujoco_qpos(np.asarray(encoded, dtype=np.float32))


def jaw_tips(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> np.ndarray:
    data.qpos[:6] = qpos
    mujoco.mj_forward(model, data)
    return np.concatenate(
        [data.site("fixed_jaw_tip").xpos.copy(), data.site("moving_jaw_tip").xpos.copy()]
    )


def solve_tip_translation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    result = qpos.astype(np.float64).copy()
    target = jaw_tips(model, data, result) + np.tile(translation, 2)
    for _ in range(600):
        current = jaw_tips(model, data, result)
        error = target - current
        if float(np.linalg.norm(error)) < 1e-5:
            break
        jacobian = np.zeros((6, 5), dtype=np.float64)
        for joint in range(5):
            shifted = result.copy()
            shifted[joint] += 1e-4
            jacobian[:, joint] = (jaw_tips(model, data, shifted) - current) / 1e-4
        step = np.linalg.solve(
            jacobian.T @ jacobian + 1e-3 * np.eye(5), jacobian.T @ error
        )
        result[:5] = np.clip(
            result[:5] + np.clip(step, -0.03, 0.03),
            MUJOCO_JOINT_LOW[:5],
            MUJOCO_JOINT_HIGH[:5],
        )
    return result


def solve_grasp_control(
    model: mujoco.MjModel, data: mujoco.MjData, demo_states: np.ndarray
) -> np.ndarray:
    target = np.asarray([0.0, 0.2249, 0.0160, -0.0045, 0.2608, 0.0160])
    sampled = demo_states[::5]
    scores = np.linalg.norm(
        np.stack([jaw_tips(model, data, qpos) for qpos in sampled]) - target, axis=1
    )
    control = sampled[int(np.argmin(scores))].astype(np.float64)
    control[5] = MUJOCO_JOINT_LOW[5]
    for _ in range(600):
        current = jaw_tips(model, data, control)
        error = target - current
        jacobian = np.zeros((6, 5), dtype=np.float64)
        for joint in range(5):
            shifted = control.copy()
            shifted[joint] += 1e-4
            jacobian[:, joint] = (jaw_tips(model, data, shifted) - current) / 1e-4
        step = np.linalg.solve(
            jacobian.T @ jacobian + 1e-3 * np.eye(5), jacobian.T @ error
        )
        control[:5] = np.clip(
            control[:5] + np.clip(step, -0.03, 0.03),
            MUJOCO_JOINT_LOW[:5],
            MUJOCO_JOINT_HIGH[:5],
        )
    return control


def arm_equilibrium(env: SO101PickPlaceEnv, control: np.ndarray) -> np.ndarray:
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[:6] = control
    env.data.qpos[6:9] = [0.5, 0.5, 0.0125]
    env.data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]
    env.data.ctrl[:6] = control
    mujoco.mj_forward(env.model, env.data)
    for _ in range(500):
        mujoco.mj_step(env.model, env.data)
    return env.data.qpos[:6].copy()


def make_candidate_pair(
    env: SO101PickPlaceEnv,
    ik_data: mujoco.MjData,
    base_control: np.ndarray,
    rng: np.random.Generator,
    pair_id: int,
) -> tuple[dict, dict] | None:
    grasp_ctrl = base_control.copy()
    grasp_ctrl[:5] += rng.normal(0.0, [0.0010, 0.0015, 0.0015, 0.0015, 0.0010])
    grasp_ctrl[:5] = np.clip(
        grasp_ctrl[:5], MUJOCO_JOINT_LOW[:5], MUJOCO_JOINT_HIGH[:5]
    )
    grasp_ctrl[5] = MUJOCO_JOINT_LOW[5]
    equilibrium = arm_equilibrium(env, grasp_ctrl)
    initial_qpos = equilibrium.copy()
    initial_qpos[5] = -0.160 + float(rng.uniform(-0.002, 0.002))

    env.data.qpos[:6] = initial_qpos
    mujoco.mj_forward(env.model, env.data)
    fixed_tip = env.data.site("fixed_jaw_tip").xpos.copy()
    moving_tip = env.data.site("moving_jaw_tip").xpos.copy()
    block_pos = 0.5 * (fixed_tip + moving_tip)
    block_pos[:2] += rng.normal(0.0, [0.00035, 0.00035])
    block_pos[2] = max(0.0125, float(0.5 * (fixed_tip[2] + moving_tip[2])))

    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[:6] = initial_qpos
    env.data.qpos[6:9] = block_pos
    env.data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[:] = 0.0
    env.data.ctrl[:6] = grasp_ctrl
    mujoco.mj_forward(env.model, env.data)
    strict_streak = 0
    for _ in range(250):
        mujoco.mj_step(env.model, env.data)
        strict, _, _ = check_block_face_gripped(env.model, env.data)
        strict_streak = strict_streak + 1 if strict else 0
    if strict_streak < 5:
        return None

    settled_qpos = env.data.qpos[:6].copy()
    settled_block = env.data.body("red_block").xpos.copy()
    settled_quat = env.data.qpos[9:13].copy()
    # Command 18 mm at the unloaded kinematic model so the gravity-loaded arm
    # realizes at least 12 mm and the block clears the 10 mm success threshold.
    lift_ctrl = solve_tip_translation(
        env.model, ik_data, grasp_ctrl, np.asarray([0.0, 0.0, 0.018])
    )
    lift_ctrl[5] = grasp_ctrl[5]
    grasped = {
        "pair_id": pair_id,
        "stage": "grasped",
        "robot_qpos": settled_qpos.tolist(),
        "robot_ctrl": grasp_ctrl.tolist(),
        "grasp_ctrl": grasp_ctrl.tolist(),
        "lift_ctrl": lift_ctrl.tolist(),
        "block_pos": settled_block.tolist(),
        "block_quat": settled_quat.tolist(),
    }
    grasp_proof = env.prove_curriculum_reset(grasped)
    if not grasp_proof["valid"]:
        return None
    grasped["proof"] = grasp_proof

    open_qpos = settled_qpos.copy()
    open_qpos[5] = 0.05
    open_ctrl = grasp_ctrl.copy()
    open_ctrl[5] = 0.05
    pregrasp = {
        "pair_id": pair_id,
        "stage": "pregrasp",
        "robot_qpos": open_qpos.tolist(),
        "robot_ctrl": open_ctrl.tolist(),
        "grasp_ctrl": grasp_ctrl.tolist(),
        "lift_ctrl": lift_ctrl.tolist(),
        "block_pos": settled_block.tolist(),
        "block_quat": settled_quat.tolist(),
    }
    pregrasp_proof = env.prove_curriculum_reset(pregrasp)
    if not pregrasp_proof["valid"]:
        return None
    pregrasp["proof"] = pregrasp_proof
    return grasped, pregrasp


def main() -> int:
    args = parse_args()
    if args.states_per_stage < 32:
        raise ValueError("--states-per-stage must be at least 32")
    rng = np.random.default_rng(args.seed)
    env = SO101PickPlaceEnv(
        render_mode=None,
        image_size=64,
        randomize_block=False,
        randomize_appearance=False,
        reward_kwargs={"strict_transition": True},
    )
    ik_data = mujoco.MjData(env.model)
    demos = load_demo_states(args.dataset)
    base_control = solve_grasp_control(env.model, ik_data, demos)
    print(f"Base grasp control: {base_control.tolist()}", flush=True)

    grasped = []
    pregrasp = []
    for attempt in range(args.max_attempts):
        if len(grasped) >= args.states_per_stage:
            break
        pair = make_candidate_pair(env, ik_data, base_control, rng, len(grasped))
        if pair is None:
            continue
        grasp_state, pregrasp_state = pair
        grasped.append(grasp_state)
        pregrasp.append(pregrasp_state)
        print(
            f"validated pair {len(grasped)}/{args.states_per_stage}: "
            f"grasp lift={grasp_state['proof']['height_gain']:.4f}m, "
            f"pregrasp lift={pregrasp_state['proof']['height_gain']:.4f}m",
            flush=True,
        )
    env.close()
    if len(grasped) < args.states_per_stage:
        raise RuntimeError(
            f"Only produced {len(grasped)} fully dynamic pairs after {args.max_attempts} attempts"
        )

    payload = {
        "schema_version": 2,
        "seed": args.seed,
        "dataset": str(args.dataset.resolve()),
        "model": str((SCRIPT_DIR / "model/scene.xml").resolve()),
        "validation": {
            "authority": "SO101PickPlaceEnv.prove_curriculum_reset",
            "minimum_final_strict_grasp_steps": 5,
            "minimum_dynamic_block_lift_gain": 0.010,
            "maximum_lateral_displacement": 0.005,
            "block_teleportation": False,
            "states_per_stage": args.states_per_stage,
        },
        "contact_calibration": {
            "bounded_grid": {
                "pad_half_thickness_m": [0.002, 0.00225, 0.0025],
                "sliding_friction": [1.0, 1.5, 2.0],
            },
            "selected_pad_half_thickness_m": 0.002,
            "selected_sliding_friction": 1.0,
            "selection_rule": "smallest pad and lowest friction passing paired dynamic proof",
        },
        "pregrasp": pregrasp,
        "grasped": grasped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(pregrasp)} pregrasp and {len(grasped)} grasped states to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
