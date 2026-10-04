"""Rendering the wrist frame only at chunk boundaries leaves a pick-place rollout identical (2026-10-04)."""

import json
from pathlib import Path

import numpy as np

from so_arm101_v2.contracts.bench import scene_bench_config
from so_arm101_v2.simulation.bench import bench_suite
from so_arm101_v2.simulation.rollout import evaluate_closed_loop

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


class ImageChunkPolicy:
    """A 30-step chunk whose commands depend on the frame, so a blank frame at a boundary would change the run."""

    requires_pixels = True

    def __init__(self) -> None:
        self._buffer: list[np.ndarray] = []
        self.frames_used = 0

    def reset(self, adapter=None) -> None:
        self._buffer = []
        self.frames_used = 0

    @property
    def needs_frame(self) -> bool:
        return not self._buffer

    def predict(self, image, current_act, adapter=None):
        if not self._buffer:
            self.frames_used += 1
            level = float(np.asarray(image, np.float32).mean()) / 255.0
            assert level > 0.0, "a chunk boundary received a blank frame"
            delta = np.zeros(6, np.float32)
            delta[1] = 0.002 * level
            self._buffer = [np.asarray(current_act, np.float32) + delta * (k + 1) for k in range(30)]
        return self._buffer.pop(0)


class Ungated:
    """The same policy without ``needs_frame``: every step is rendered, as before the change."""

    requires_pixels = True

    def __init__(self) -> None:
        self.inner = ImageChunkPolicy()

    def reset(self, adapter=None) -> None:
        self.inner.reset(adapter)

    def predict(self, image, current_act, adapter=None):
        return self.inner.predict(image, current_act, adapter)


def _rows(result):
    report = json.loads(Path(result.report_json).read_text())
    (rollout,) = report["rollouts"]
    return json.loads(Path(rollout["telemetry_path"]).read_text())["rows"]


def test_frame_gated_rollout_matches_every_frame_rollout(tmp_path):
    config = scene_bench_config(SCENE)
    suite = bench_suite(config, [(0.0, 0.0)], label="frame_skip_test", repeats=1)
    gated = evaluate_closed_loop(SCENE, suite, {"chunk": ImageChunkPolicy}, tmp_path / "gated", environment_proven=True, record_video=False, workers=1)
    every = evaluate_closed_loop(SCENE, suite, {"chunk": Ungated}, tmp_path / "every", environment_proven=True, record_video=False, workers=1)
    a, b = _rows(gated), _rows(every)
    assert len(a) == len(b) > 30
    for ra, rb in zip(a, b):
        assert ra["executed_act"] == rb["executed_act"]
        assert ra["robot_qpos"] == rb["robot_qpos"]
        assert ra["cube_position"] == rb["cube_position"]
