"""Optional passive MuJoCo sample playback; never steps physics or hardware."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from so_arm101_v2.contracts import JOINT_NAMES, act_to_mujoco_qpos, clip_mujoco_qpos
from so_arm101_v2.data import FutureStateSample


@dataclass
class _SamplePlaybackController:
    trajectory: list[np.ndarray]
    playing: bool = False
    index: int = 0

    def handle_key(self, keycode: int) -> np.ndarray | None:
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if key == "C":
            self.playing = False
            self.index = 0
            return self.trajectory[0]
        if key == "T":
            self.playing = False
            self.index = len(self.trajectory) - 1
            return self.trajectory[-1]
        if keycode == 32:
            self.playing = not self.playing
        return None

    def advance(self) -> np.ndarray:
        self.index = (self.index + 1) % len(self.trajectory)
        return self.trajectory[self.index]


def _mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MuJoCo viewing requires the 'visualize' extra") from exc
    return mujoco


def apply_mujoco_pose(model: Any, data: Any, qpos: np.ndarray) -> None:
    """Assign one six-joint mechanical pose through named qpos addresses."""
    mujoco = _mujoco()
    values = np.asarray(qpos, dtype=np.float32)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("MuJoCo pose must be one finite six-joint vector")
    for name, value in zip(JOINT_NAMES, values):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint {name!r}")
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)
    mujoco.mj_forward(model, data)


def launch_mujoco_sample_viewer(
    sample: FutureStateSample,
    model_path: str | Path,
    *,
    playback_fps: float = 5.0,
) -> None:
    """Open a passive viewer with current/target/recorded-trajectory controls."""
    if not np.isfinite(playback_fps) or playback_fps <= 0:
        raise ValueError("playback_fps must be positive and finite")
    mujoco = _mujoco()
    try:
        import mujoco.viewer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("this MuJoCo installation has no passive viewer") from exc
    path = Path(model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MuJoCo model does not exist: {path}")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    trajectory = [
        clip_mujoco_qpos(act_to_mujoco_qpos(state))[0]
        for state in sample.observed_states
    ]
    apply_mujoco_pose(model, data, trajectory[0])
    controller = _SamplePlaybackController(trajectory)

    def on_key(keycode: int) -> None:
        selected = controller.handle_key(keycode)
        if selected is not None:
            apply_mujoco_pose(model, data, selected)

    print("MuJoCo sample viewer: C=current, T=target, Space=play/pause trajectory")
    period = 1.0 / playback_fps
    next_frame = time.monotonic() + period
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        try:
            while viewer.is_running():
                now = time.monotonic()
                if controller.playing and now >= next_frame:
                    with viewer.lock():
                        apply_mujoco_pose(model, data, controller.advance())
                    next_frame = now + period
                viewer.sync()
                time.sleep(0.01)
        except KeyboardInterrupt:
            return


__all__ = ["apply_mujoco_pose", "launch_mujoco_sample_viewer"]
