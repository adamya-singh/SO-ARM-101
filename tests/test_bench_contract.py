from pathlib import Path
from dataclasses import replace
import json
from unittest.mock import Mock
import numpy as np
import pytest

from so_arm101_v2.contracts.bench import BenchConfig, scene_dependency_hash
from so_arm101_v2.contracts.physical import evaluate_physical_command, physical_normalized_to_act, load_physical_calibration
from so_arm101_v2.contracts.physical_io import connect_read_only, disconnect_read_only, read_measured_act


def test_reset_validation_under_the_measured_joint_map():
    # Under the legacy affine map the gravity-rest elbow (100 normalized) fell
    # outside the model; the measured map places it at the model's 92.5 deg
    # limit, so the natural rest is a valid reset and no elbow staging is
    # required to enter the model's range.
    BenchConfig(reset_physical=(0, -90, 100, 43, -1, 13))
    with pytest.raises(ValueError, match="shoulder"):
        BenchConfig(reset_physical=(0, -93, 90, 43, -1, 13))
    with pytest.raises(ValueError, match="bounds"):
        # shoulder +100 normalized is +104 deg in the model, past its +10 deg limit
        BenchConfig(reset_physical=(0, 100, 90, 43, -1, 13))
    with pytest.raises(ValueError, match="measured joint map"):
        BenchConfig(joint_map="legacy_affine_v1")
    BenchConfig(reset_physical=(0, -90, 90, 43, -1, 13))


def test_shoulder_floor_includes_quantization():
    current = physical_normalized_to_act([0, -90, 90, 43, -1, 13])
    for shoulder in (-92.1, -92):
        target = physical_normalized_to_act([0, shoulder, 90, 43, -1, 13])
        assert evaluate_physical_command(current, target, shoulder_floor=-92).physical_clip_mask[1]
    cal = load_physical_calibration().joints[1]
    safe_tick = int(np.ceil(cal.range_min + (8 / 200) * (cal.range_max - cal.range_min)))
    safe_value = (safe_tick - cal.range_min) / (cal.range_max - cal.range_min) * 200 - 100
    target = physical_normalized_to_act([0, safe_value + 0.01, 90, 43, -1, 13])
    assert not evaluate_physical_command(current, target, shoulder_floor=-92).physical_clip_mask[1]


def test_read_only_connection_never_configures_or_changes_torque():
    robot = Mock()
    robot.bus.is_calibrated = True
    robot.bus.is_connected = True
    connect_read_only(robot)
    disconnect_read_only(robot)
    robot.configure.assert_not_called()
    robot.connect.assert_not_called()
    robot.bus.enable_torque.assert_not_called()
    robot.bus.disable_torque.assert_not_called()
    robot.bus.disconnect.assert_called_once_with(disable_torque=False)


def test_stale_and_nonfinite_feedback(monkeypatch):
    import so_arm101_v2.contracts.physical_io as module
    from so_arm101_v2.contracts.coordinates import JOINT_NAMES
    robot = Mock()
    robot.bus.sync_read.return_value = dict.fromkeys(JOINT_NAMES, 0)
    times = iter([0, 0.2])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(times))
    with pytest.raises(RuntimeError, match='stale'):
        read_measured_act(robot)
    monkeypatch.setattr(module.time, 'monotonic', lambda: 0)
    robot.bus.sync_read.return_value['elbow_flex'] = float('nan')
    with pytest.raises(RuntimeError, match='nonfinite'):
        read_measured_act(robot)


def test_asset_hash_changes_with_included_mesh(tmp_path):
    (tmp_path/'asset.stl').write_bytes(b'original')
    (tmp_path/'arm.xml').write_text('<mujoco><asset><mesh file="asset.stl"/></asset></mujoco>')
    p=tmp_path/'scene.xml'
    p.write_text('<mujoco><include file="arm.xml"/></mujoco>')
    original=scene_dependency_hash(p)
    (tmp_path/'asset.stl').write_bytes(b'changed')
    assert scene_dependency_hash(p) != original


def test_elbow_preparation_is_inward_only_and_never_relaxes_shoulder():
    from tools.prepare_physical_elbow import next_elbow_command
    current=physical_normalized_to_act([0,-90,99.7,43,-1,13])
    target=next_elbow_command(current,99.7)
    assert 99.5 < target < 99.7
    for _ in range(150):
        target=next_elbow_command(current,99.7,target)
    assert 89.69 < target < 89.71  # capped ten units ahead of measured state
    with pytest.raises(RuntimeError,match='shoulder'):
        next_elbow_command(physical_normalized_to_act([0,-92.1,99.7,43,-1,13]),99.7)
    with pytest.raises(RuntimeError,match='unexpected'):
        next_elbow_command(physical_normalized_to_act([0,-90,100,43,-1,13]),98)


def test_cube_already_on_square_does_not_count_as_pickup():
    from so_arm101_v2.contracts.pick_place import load_pick_place_contract, PickPlaceMeasurement, PickPlaceEvaluationState, evaluate_pick_place_step
    from so_arm101_v2.contracts.task import TaskMeasurement
    config=BenchConfig(reset_physical=(0,-90,90,43,-1,13))
    contract=load_pick_place_contract('bench_pick_replace_v1',bench_config=config)
    assert contract.pickup.object.edge_length_m == .020
    measurement=PickPlaceMeasurement(TaskMeasurement(.1,False,False,False,0),True,0,0,0,1.2)
    state=PickPlaceEvaluationState()
    for _ in range(30):
        state,result=evaluate_pick_place_step(contract,measurement,state)
        assert not result.success and not result.pickup_completed and result.settled_frames == 0


def test_bench_scene_twenty_mm_footprint_and_support(tmp_path):
    pytest.importorskip('mujoco')
    from tools.prepare_bench_scene import prepare
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    config=BenchConfig(reset_physical=(0,-90,90,43,-1,13))
    path=prepare(tmp_path,config)
    a=MujocoTaskAdapter(path)
    try:
        scenario=bench_suite(config,[(.01,0)],label='test',repeats=1).scenarios[0]
        a.reset(scenario)
        measurement,_=a.pick_place_measurement()
        assert measurement.cube_footprint_inside
        assert abs(measurement.cube_support_error_m)<.001
        assert a.model.geom('red_block_geom').size.tolist() == [.01]*3
    finally:a.close()


@pytest.mark.parametrize('prefetch', ['0','1'])
def test_vision_checkpoint_resume_matches_uninterrupted(tmp_path, monkeypatch, prefetch):
    torch=pytest.importorskip('torch')
    torch.set_num_threads(1)
    from test_vision_lane import _write_tiny_frames_manifest
    from so_arm101_v2.learning.vision import train_vision_chunked, VisionChunkedConfig
    monkeypatch.setenv('SO_ARM101_V2_PREFETCH',prefetch)
    manifest=_write_tiny_frames_manifest(tmp_path,rows=8)
    config=VisionChunkedConfig(chunk_horizon=2,max_steps=7,batch_size=2)
    full=train_vision_chunked(manifest,tmp_path/'full',config=config,numerics=None)
    scratch=tmp_path/'scratch.pt'
    with pytest.raises(InterruptedError):
        train_vision_chunked(manifest,tmp_path/'resumed',config=config,numerics=None,
            scratch_checkpoint=scratch,checkpoint_interval=2,stop_after_steps=3)
    resumed=train_vision_chunked(manifest,tmp_path/'resumed',config=config,numerics=None,
        scratch_checkpoint=scratch,checkpoint_interval=2)
    assert full.checkpoint.read_bytes() == resumed.checkpoint.read_bytes()
    assert json.loads(full.report_json.read_text()) == json.loads(resumed.report_json.read_text())


def test_bench_distance_uses_front_edge_not_rotation_origin():
    config = BenchConfig()
    assert config.distance_reference == "base_front_edge"
    assert config.square_center_xy[1] == pytest.approx(
        config.base_front_edge_y_m + 8.5 * .0254)
    with pytest.raises(ValueError, match="geometry"):
        replace(config, square_center_xy=(0., .2159))


def test_pinned_calibration_check_refuses_drift():
    from types import SimpleNamespace
    from so_arm101_v2.contracts.physical_io import assert_pinned_calibration
    expected = load_physical_calibration()
    live = {j.name: SimpleNamespace(id=j.motor_id, drive_mode=j.drive_mode, homing_offset=j.homing_offset,
                                    range_min=j.range_min, range_max=j.range_max) for j in expected.joints}
    robot = Mock()
    robot.calibration = live
    assert_pinned_calibration(robot)
    live['shoulder_lift'].range_min += 1
    with pytest.raises(RuntimeError, match='shoulder_lift'):
        assert_pinned_calibration(robot)
    live['shoulder_lift'].range_min -= 1
    del live['gripper']
    with pytest.raises(RuntimeError, match='gripper'):
        assert_pinned_calibration(robot)


def test_bench_verification_requires_hashed_review_images(tmp_path):
    import hashlib
    from so_arm101_v2.simulation.bench import load_bench_verification
    (tmp_path / 'phys.png').write_bytes(b'physical')
    (tmp_path / 'sim.png').write_bytes(b'simulated')
    review = dict(physical_wrist_image='phys.png', physical_wrist_image_sha256=hashlib.sha256(b'physical').hexdigest(),
                  simulated_wrist_image='sim.png', simulated_wrist_image_sha256=hashlib.sha256(b'simulated').hexdigest(),
                  reviewed_at='2026-09-06T18:00:00+00:00', reviewer='bench operator', notes='cube centered in both frames')
    good = dict(scene_dependencies_sha256='scene', reset_evidence_sha256='reset', camera_review=review)
    path = tmp_path / 'verification.json'
    path.write_text(json.dumps(good))
    assert load_bench_verification(path, scene_hash='scene', reset_evidence_sha256='reset')['camera_review'] == review
    bad_records = [
        dict(good, camera_review=None, camera_review_passed=True),          # the old bare boolean
        dict(good, scene_dependencies_sha256='other'),                        # stale scene
        dict(good, reset_evidence_sha256='other'),                            # stale reset
        dict(good, camera_review=dict(review, notes='')),                     # no notes
        dict(good, camera_review=dict(review, reviewed_at='yesterday')),      # bad timestamp
        dict(good, camera_review=dict(review, simulated_wrist_image='gone.png')),
    ]
    for record in bad_records:
        path.write_text(json.dumps(record))
        with pytest.raises(RuntimeError):
            load_bench_verification(path, scene_hash='scene', reset_evidence_sha256='reset')
    path.write_text(json.dumps(good))
    with pytest.raises(RuntimeError, match='active bench config'):
        load_bench_verification(path, scene_hash='scene', reset_evidence_sha256=None)
    (tmp_path / 'sim.png').write_bytes(b'regenerated')
    with pytest.raises(RuntimeError, match='hash'):
        load_bench_verification(path, scene_hash='scene', reset_evidence_sha256='reset')


def test_active_bench_teacher_registers_strict_grasp_and_completes_nominal():
    """Regression pin for the 20 mm teacher: the tip-station grasp must stay strict."""
    pytest.importorskip('mujoco')
    from pathlib import Path
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    from so_arm101_v2.simulation.privileged import PrivilegedStagedController
    from so_arm101_v2.contracts.pick_place import load_pick_place_contract, PickPlaceEvaluationState, evaluate_pick_place_step
    scene = Path(__file__).resolve().parents[1] / 'simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml'
    adapter = MujocoTaskAdapter(scene)
    try:
        # Pad 1 pinches a 20 mm cube at qpos +0.055 with ample closing travel; pad 4
        # would pinch at -0.152, 0.02 rad from the mechanical limit.
        assert adapter.bench.grasp_pad == 1
        adapter.reset(bench_suite(adapter.bench, [(0, 0)], label='test', repeats=1).scenarios[0])
        controller = PrivilegedStagedController()
        controller.reset(adapter)
        assert controller.depth_lead_m == adapter.bench.depth_lead_m
        contract = load_pick_place_contract('bench_pick_replace_v1', bench_config=adapter.bench)
        state = PickPlaceEvaluationState()
        strict = safety = 0
        for _ in range(contract.max_actions):
            command = adapter.apply_policy_command(controller.predict(None, adapter.current_act(), adapter))
            adapter.advance_control_period()
            measurement, _ = adapter.pick_place_measurement(command)
            state, result = evaluate_pick_place_step(contract, measurement, state)
            strict += measurement.pickup.strict_bilateral_grasp
            safety += (measurement.pickup.unsafe_contact + measurement.pickup.command_bound_violation
                       + measurement.pickup.delta_limiter_activated + measurement.pickup.nonfinite_command)
            if result.terminated or result.truncated or result.invalidated:
                break
        assert result.success and not result.invalidated
        assert strict >= 100 and safety == 0
    finally:
        adapter.close()


def test_bench_config_rejects_unbounded_teacher_offsets():
    with pytest.raises(ValueError, match='depth_lead_m'):
        BenchConfig(depth_lead_m=0.05)
    with pytest.raises(ValueError, match='grasp_offset_m'):
        BenchConfig(grasp_offset_m=float('nan'))
    assert BenchConfig(grasp_pad=2).grasp_pad == 2


@pytest.mark.parametrize('qgrip', [-0.15, -0.06, 0.0, 0.055, 0.4])
def test_pad_axis_tracks_pad_normal_where_tip_sites_do_not(qgrip):
    """The detector's jaw axis must follow the pad closing direction across the whole closure range."""
    mujoco = pytest.importorskip('mujoco')
    from pathlib import Path
    from so_arm101_v2.simulation.contact import jaw_closing_axis
    scene = Path(__file__).resolve().parents[1] / 'simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml'
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    data.qpos[model.joint('gripper').qposadr[0]] = qgrip
    mujoco.mj_forward(model, data)
    pad_normal = data.geom_xmat[model.geom('fixed_jaw_pad_4').id].reshape(3, 3)[:, 0]
    moving_normal = data.geom_xmat[model.geom('moving_jaw_pad_4').id].reshape(3, 3)[:, 0]
    tilt = np.degrees(np.arccos(abs(float(np.dot(pad_normal, moving_normal)))))
    new = np.degrees(np.arccos(abs(float(np.dot(jaw_closing_axis(model, data, 'pad_normals'), pad_normal)))))
    old = np.degrees(np.arccos(abs(float(np.dot(jaw_closing_axis(model, data, 'tip_sites'), pad_normal)))))
    assert abs(new - tilt / 2) < 0.5, (new, tilt)   # bisector: half the moving-jaw tilt, zero for parallel pads
    if qgrip <= 0.06:                      # the whole 20 mm and 25 mm pinch range
        assert new <= 5.0, new
        assert old >= 22.0, old            # the legacy reference was never near the pad normal there
    with pytest.raises(ValueError, match='jaw axis mode'):
        jaw_closing_axis(model, data, 'elsewhere')


def test_prefix_success_reads_viewing_pose_at_first_image_refresh(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    from run_bench_pipeline import prefix_success, PREFIX_TOLERANCE_RAD
    base = BenchConfig(reset_physical=(0, -90, 90, 43, -1, 13))
    viewing = [float(v) for v in base.reset_qpos]
    bench = BenchConfig(reset_physical=(0, -90, 90, 43, -1, 13), viewing_qpos=tuple(viewing))
    good = [dict(robot_qpos=viewing[:2] + [viewing[2] - PREFIX_TOLERANCE_RAD / 2] + viewing[3:5] + [0.5])] * 90
    bad = [dict(robot_qpos=viewing[:2] + [viewing[2] - 2 * PREFIX_TOLERANCE_RAD] + viewing[3:])] * 90
    short = good[:40]
    paths = {}
    for name, rows in (('good', good), ('bad', bad), ('short', short)):
        paths[name] = tmp_path / f'{name}.json'
        paths[name].write_text(json.dumps(dict(rows=rows)))
    report = dict(rollouts=[dict(policy_id='vision', telemetry_path=str(paths['good'])),
                            dict(policy_id='vision', telemetry_path=str(paths['bad'])),
                            dict(policy_id='vision', telemetry_path=str(paths['short'])),
                            dict(policy_id='other', telemetry_path=str(paths['good']))])
    assert prefix_success(report, bench, 90) == {'vision': dict(prefix_ok=1, rollouts=3), 'other': dict(prefix_ok=1, rollouts=1)}


def test_pipeline_rehearsal_refuses_unlabelled_output_and_requires_verification(tmp_path):
    import os, subprocess, sys
    root = Path(__file__).resolve().parents[1]
    model = root / 'simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml'
    tool = root / 'tools/run_bench_pipeline.py'
    env = {**os.environ, 'PYTHONPATH': str(root / 'src'), 'PYTHONNOUSERSITE': '1', 'MUJOCO_GL': 'egl'}
    plain = subprocess.run([sys.executable, str(tool), '--model', str(model), '--output-dir', str(tmp_path / 'not_labelled'), '--rehearsal'],
                           capture_output=True, text=True, env=env)
    assert plain.returncode != 0 and 'rehearsal' in (plain.stderr + plain.stdout)
    assert not (tmp_path / 'not_labelled' / 'experiment.json').exists()
    missing = subprocess.run([sys.executable, str(tool), '--model', str(model), '--output-dir', str(tmp_path / 'real')],
                             capture_output=True, text=True, env=env)
    assert missing.returncode != 0 and '--verification is required' in (missing.stderr + missing.stdout)
