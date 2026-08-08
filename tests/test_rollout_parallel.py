"""Process-parallel rollout execution: parity, ordering, and failure modes."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.simulation import load_simulation_suite
from so_arm101_v2.simulation.policy_specs import PolicySpec
from so_arm101_v2.simulation.rollout import evaluate_closed_loop

SCENE = REPOSITORY_ROOT / "simulation_code/model/menagerie_so_arm100/scene_v2.xml"

_PATH_FIELDS = ("telemetry_path", "wrist_video_path", "overview_video_path")


def _narrow_suite():
    suite = load_simulation_suite("fixed_pickup_contract_v1")
    return replace(suite, scenarios=suite.scenarios[:1], repeats=2)


def _comparable(rollouts) -> list[dict]:
    rows = []
    for item in rollouts:
        record = asdict(item)
        for field in _PATH_FIELDS:
            record.pop(field, None)
        rows.append(record)
    return rows


def _telemetry_hashes(destination: Path) -> dict[str, str]:
    telemetry_dir = destination / "telemetry"
    return {
        item.name: json.loads(item.read_text())["content_sha256"]
        for item in sorted(telemetry_dir.iterdir())
    }


def _renderer_available() -> bool:
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(SCENE))
        renderer = mujoco.Renderer(model, height=256, width=256)
        renderer.close()
        return True
    except Exception:
        return False


def test_parallel_rollouts_match_sequential_bytes(tmp_path: Path) -> None:
    suite = _narrow_suite()
    policies = {"current_pose": PolicySpec(kind="current_pose")}
    sequential = evaluate_closed_loop(
        SCENE, suite, policies, tmp_path / "sequential",
        environment_proven=True, record_video=False, workers=1,
    )
    parallel = evaluate_closed_loop(
        SCENE, suite, policies, tmp_path / "parallel",
        environment_proven=True, record_video=False, workers=2,
    )
    assert _comparable(parallel.rollouts) == _comparable(sequential.rollouts)
    assert [item.repeat for item in parallel.rollouts] == [0, 1]
    assert parallel.deterministic and sequential.deterministic
    destination = "policies/" + suite.suite_id
    assert _telemetry_hashes(tmp_path / "parallel" / destination) == _telemetry_hashes(
        tmp_path / "sequential" / destination
    )


def test_bare_callable_with_workers_raises(tmp_path: Path) -> None:
    suite = _narrow_suite()
    with pytest.raises(ValueError, match="requires PolicySpec"):
        evaluate_closed_loop(
            SCENE, suite, {"legacy": lambda: None}, tmp_path,
            environment_proven=True, record_video=False, workers=2,
        )
    assert not (tmp_path / "policies" / suite.suite_id / "evaluation.json").exists()


def test_worker_failure_propagates_and_writes_no_report(tmp_path: Path) -> None:
    suite = _narrow_suite()
    policies = {
        "broken": PolicySpec(
            kind="chunked_clone", checkpoint=str(tmp_path / "missing.pt")
        )
    }
    with pytest.raises(Exception):
        evaluate_closed_loop(
            SCENE, suite, policies, tmp_path,
            environment_proven=True, record_video=False, workers=2,
        )
    assert not (tmp_path / "policies" / suite.suite_id / "evaluation.json").exists()


@pytest.mark.skipif(not _renderer_available(), reason="offscreen MuJoCo rendering unavailable")
def test_video_toggle_leaves_telemetry_identical_for_pixel_free_policy(tmp_path: Path) -> None:
    suite = _narrow_suite()
    policies = {"current_pose": PolicySpec(kind="current_pose")}
    with_video = evaluate_closed_loop(
        SCENE, suite, policies, tmp_path / "video",
        environment_proven=True, record_video=True, workers=1,
    )
    without_video = evaluate_closed_loop(
        SCENE, suite, policies, tmp_path / "novideo",
        environment_proven=True, record_video=False, workers=1,
    )
    destination = "policies/" + suite.suite_id
    assert _telemetry_hashes(tmp_path / "video" / destination) == _telemetry_hashes(
        tmp_path / "novideo" / destination
    )
    assert with_video.rollouts[0].wrist_video_path is not None
    assert without_video.rollouts[0].wrist_video_path is None


@pytest.mark.skipif(not _renderer_available(), reason="offscreen MuJoCo rendering unavailable")
def test_pixel_hungry_policy_still_gets_real_frames_without_video(tmp_path: Path) -> None:
    from so_arm101_v2.simulation.rollout import _BLANK_WRIST_IMAGE

    frames: list[np.ndarray] = []

    class PixelProbePolicy:
        requires_pixels = True

        def predict(self, image, current_act, adapter=None):
            frames.append(image)
            return current_act.copy()

    suite = replace(_narrow_suite(), repeats=1)
    evaluate_closed_loop(
        SCENE, suite, {"probe": PixelProbePolicy}, tmp_path,
        environment_proven=True, record_video=False, workers=1,
    )
    assert frames
    assert all(frame is not None and frame.shape == (256, 256, 3) for frame in frames)
    assert any(frame is not _BLANK_WRIST_IMAGE for frame in frames)


def test_pixel_free_policy_receives_blank_frames_without_video(tmp_path: Path) -> None:
    from so_arm101_v2.simulation.rollout import _BLANK_WRIST_IMAGE

    frames: list = []

    class PixelFreeProbePolicy:
        requires_pixels = False

        def predict(self, image, current_act, adapter=None):
            frames.append(image)
            return current_act.copy()

    suite = replace(_narrow_suite(), repeats=1)
    evaluate_closed_loop(
        SCENE, suite, {"probe": PixelFreeProbePolicy}, tmp_path,
        environment_proven=True, record_video=False, workers=1,
    )
    assert frames
    assert all(frame is _BLANK_WRIST_IMAGE for frame in frames)
