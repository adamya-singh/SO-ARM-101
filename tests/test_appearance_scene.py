"""Appearance slots in the generated scene and the adapter's randomized/pristine render paths (EGL)."""
from __future__ import annotations

from dataclasses import replace
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts.appearance import AppearanceRegime, photometric_lut, resolve_appearance
from so_arm101_v2.contracts.bench import scene_bench_config

LIVE_SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
SLOT_TEXTURE, SLOT_BODY = "ground_speckle", "towel_visual"


def _strip_slots(xml_text: str) -> str:
    root = ET.fromstring(xml_text)
    assets, world = root.find("asset"), root.find("worldbody")
    assets.remove(assets.find(f"texture[@name='{SLOT_TEXTURE}']"))
    world.remove(world.find(f"body[@name='{SLOT_BODY}']"))
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def _prepare_pair(tmp_path, appearance=None):
    """A freshly generated scene (with slots) and a sibling copy with the slots stripped out."""
    pytest.importorskip("mujoco")
    from tools.prepare_bench_scene import prepare
    config = scene_bench_config(LIVE_SCENE)
    if appearance is not None:
        config = replace(config, appearance=appearance.identity())
    slotted = prepare(tmp_path / "slotted", config)
    stripped_dir = tmp_path / "stripped"
    shutil.copytree(tmp_path / "slotted", stripped_dir)
    stripped = stripped_dir / slotted.name
    stripped.write_text(_strip_slots(slotted.read_text()))
    return slotted, stripped


def _raster_tolerance(params) -> int:
    """EGL renders jitter by ±1 grey level; the photometric LUT (gain/gamma) can amplify that to its largest step."""
    return max(1, int(np.diff(photometric_lut(params.photometric).astype(int), axis=1).max()))


def _nominal(adapter, seed=None):
    from so_arm101_v2.simulation.bench import bench_suite
    scenario = bench_suite(adapter.bench, [(0, 0)], label="test", repeats=1).scenarios[0]
    if seed is None:
        return scenario
    return replace(scenario, scenario_id=f"{scenario.scenario_id}_a{seed}", fixed_appearance=False, appearance_seed=seed)


def test_slots_do_not_shift_ids_extent_or_the_pristine_render(tmp_path):
    import mujoco
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    slotted, stripped = _prepare_pair(tmp_path)
    a, b = mujoco.MjModel.from_xml_path(str(slotted)), mujoco.MjModel.from_xml_path(str(stripped))
    assert a.nbody == b.nbody + 1 and a.ngeom == b.ngeom + 1 and a.ntex == b.ntex + 1
    assert (a.nq, a.nv, a.nu, a.nmat) == (b.nq, b.nv, b.nu, b.nmat)
    assert a.stat.extent == b.stat.extent and np.array_equal(a.stat.center, b.stat.center)
    for kind, count in ((mujoco.mjtObj.mjOBJ_BODY, b.nbody), (mujoco.mjtObj.mjOBJ_GEOM, b.ngeom)):
        for index in range(count):
            assert mujoco.mj_id2name(a, kind, index) == mujoco.mj_id2name(b, kind, index)
    towel = a.geom(f"{SLOT_BODY}_geom")
    assert a.geom_contype[towel.id] == 0 and a.geom_conaffinity[towel.id] == 0 and a.geom_rgba[towel.id, 3] == 0
    assert a.body_mass[a.body(SLOT_BODY).id] == 0 and a.mat_texid[a.material("groundplane").id].max() == -1
    first = MujocoTaskAdapter(slotted)
    second = MujocoTaskAdapter(stripped)
    try:
        first.reset(_nominal(first)); second.reset(_nominal(second))
        assert first.appearance_record is None
        one, two = first.render_wrist_observation(), second.render_wrist_observation()
        assert np.abs(one.astype(int) - two.astype(int)).max() <= 1
        assert np.array_equal(first.mujoco_qpos(), second.mujoco_qpos())
        assert np.array_equal(first.data.body("red_block").xpos, second.data.body("red_block").xpos)
    finally:
        first.close(); second.close()


def test_randomized_resets_change_the_look_and_restore_exactly(tmp_path):
    import mujoco
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    regime = AppearanceRegime()
    slotted, stripped = _prepare_pair(tmp_path, regime)
    adapter = MujocoTaskAdapter(slotted)
    reference = MujocoTaskAdapter(stripped)
    try:
        reference.reset(_nominal(reference))
        pristine = reference.render_wrist_observation()
        adapter.reset(_nominal(adapter))
        assert np.abs(adapter.render_wrist_observation().astype(int) - pristine.astype(int)).max() <= 1
        adapter.reset(_nominal(adapter, 11))
        record = adapter.appearance_record
        assert record["seed"] == 11 and record["resolver"] == "bench_appearance_v1" and record["params"] == resolve_appearance(regime, 11).as_record()
        seed11 = adapter.render_wrist_observation()
        tolerance = _raster_tolerance(adapter.appearance_params)
        assert np.abs(seed11.astype(int) - adapter.render_wrist_observation().astype(int)).max() <= tolerance  # idempotent at a fixed step
        assert np.abs(seed11.astype(int) - pristine.astype(int)).mean() > 5
        adapter.reset(_nominal(adapter, 12))
        seed12 = adapter.render_wrist_observation()
        assert np.abs(seed12.astype(int) - seed11.astype(int)).mean() > 5
        # Physics untouched by the draw: identical settled state to the pristine reference.
        assert np.array_equal(adapter.mujoco_qpos(), reference.mujoco_qpos())
        assert np.array_equal(adapter.data.body("red_block").xpos, reference.data.body("red_block").xpos)
        assert np.array_equal(adapter.model.geom_size[adapter.model.geom("napkin").id], reference.model.geom_size[reference.model.geom("napkin").id])
        # Back to pristine: the model fields are restored exactly and the render matches within raster jitter.
        adapter.reset(_nominal(adapter))
        assert adapter._appearance_pristine.matches(adapter.model) and adapter.appearance_record is None
        assert np.abs(adapter.render_wrist_observation().astype(int) - pristine.astype(int)).max() <= 1
        # The same seed reproduces the same look (fresh render context; amplified raster jitter only).
        adapter.reset(_nominal(adapter, 11))
        assert np.abs(adapter.render_wrist_observation().astype(int) - seed11.astype(int)).max() <= tolerance
        # Skybox flag follows the draw; an "off" seed exists within the first 40 seeds.
        off = next(s for s in range(40) if resolve_appearance(regime, s).skybox == "off")
        adapter.reset(_nominal(adapter, off))
        adapter.render_wrist_observation()
        assert adapter.lens_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] == 0
        on = next(s for s in range(40) if resolve_appearance(regime, s).skybox != "off")
        adapter.reset(_nominal(adapter, on))
        adapter.render_wrist_observation()
        assert adapter.lens_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] == 1
    finally:
        adapter.close(); reference.close()


def test_seeded_scenarios_are_refused_without_a_regime_or_slots(tmp_path):
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    slotted, stripped = _prepare_pair(tmp_path)
    adapter = MujocoTaskAdapter(slotted)
    try:
        with pytest.raises(ValueError, match="no appearance regime"):
            adapter.reset(_nominal(adapter, 3))
    finally:
        adapter.close()
    _, stripped_with_regime = _prepare_pair(tmp_path / "regime", AppearanceRegime())
    adapter = MujocoTaskAdapter(stripped_with_regime)
    try:
        towel_seed = next(s for s in range(40) if resolve_appearance(AppearanceRegime(), s).towel is not None)
        with pytest.raises(ValueError, match="appearance slots"):
            adapter.reset(_nominal(adapter, towel_seed))
    finally:
        adapter.close()


def test_privileged_teacher_is_bit_identical_under_a_random_appearance(tmp_path):
    """No silent physics change: the pixel-free teacher's commands and measurements match the stripped scene exactly."""
    from so_arm101_v2.contracts.pick_place import PickPlaceEvaluationState, evaluate_pick_place_step, load_pick_place_contract
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.privileged import PrivilegedStagedController
    slotted, stripped = _prepare_pair(tmp_path, AppearanceRegime())

    def run(scene, seed):
        adapter = MujocoTaskAdapter(scene)
        try:
            adapter.reset(_nominal(adapter, seed))
            controller = PrivilegedStagedController()
            controller.reset(adapter)
            contract = load_pick_place_contract("bench_pick_replace_v1", bench_config=adapter.bench)
            state = PickPlaceEvaluationState()
            acts, cube = [], []
            for _ in range(contract.max_actions):
                command = adapter.apply_policy_command(controller.predict(None, adapter.current_act(), adapter))
                adapter.advance_control_period()
                measurement, _ = adapter.pick_place_measurement(command)
                state, result = evaluate_pick_place_step(contract, measurement, state)
                acts.append(np.array(adapter.current_act(), copy=True)); cube.append(np.array(adapter.data.body("red_block").xpos, copy=True))
                if result.terminated or result.truncated or result.invalidated:
                    break
            return result.success, np.stack(acts), np.stack(cube)
        finally:
            adapter.close()

    success_a, acts_a, cube_a = run(slotted, 21)
    success_b, acts_b, cube_b = run(stripped, None)
    assert success_a and success_b
    assert acts_a.shape == acts_b.shape and np.array_equal(acts_a, acts_b) and np.array_equal(cube_a, cube_b)
