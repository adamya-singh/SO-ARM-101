"""Counterfactual feature probes (2026-11-18, notes/feature-probes-20261118.md).

A success count cannot tell a policy that looks from one that remembers: both pick the cube up when nothing changes
after the first frame. A probe changes the world in the middle of an episode, where the two must behave differently,
and compares the episode with the same episode left alone (the simulator is deterministic, so the pair differs only
by the change). Each probe is tied to one feature a closed-loop policy would need:

  shift_placement  square and cube move together after the survey frame   -> does the reach follow later frames?
  shift_cube       the cube alone moves on its square after the survey    -> is it the cube or the square it follows?
  drop_cube        the cube is put back on its square once it is lifted   -> does it notice an empty gripper and retry?
  move_square      the square moves while the cube is in the air          -> does the set-down follow the square?
  swap_elbow       the elbow servo changes mid-episode: the measured soft -> does it cope with a servo that changed?
                   one becomes the stiff nominal one, or the reverse
  shift_and_swap   shift_placement and swap_elbow at the same action       -> do the two features work together?
                   (2026-11-19, notes/recursive-generalization-20261119.md); its follow gain is read against the
                   swap_elbow episode, so it is the effect of the shift on an arm whose elbow already changed

The scripted teacher gives two reference controls. It is open-loop, so its plain episode must show a follow gain of
zero; planned with the change known in advance (``teacher_informed``) it must show a gain of one and still succeed.
If the controls do not read zero and one the probe is wrong, not the policy.

The rollout loop is ``rollout._run_pick_place_rollout`` with the probe applied before the frame of its action is
read (tests/test_probes.py pins the unprobed loop to the certified one). It keeps stepping after the task evaluator
has finished, so a late reaction (a retry) is still seen; the contract outcome is the one at its first terminal step.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import zlib

import numpy as np

from so_arm101_v2.contracts.coordinates import JOINT_NAMES

PROBE_NAMES = ("control", "shift_placement", "shift_cube", "drop_cube", "move_square", "swap_elbow", "shift_and_swap")
GAIN_REFERENCE = {"shift_and_swap": "swap_elbow"}     # probe -> the episode its follow gain is measured against (default: control)
CLOSURE_STEP = 250            # the row just before the jaws close (simulation/closure.py)
LIFT_TRIGGER_M = 0.015        # drop_cube / move_square fire when the cube has risen this far
REGRASP_HEIGHT_M = 0.010      # a second lift after the drop counts from this height
REGRASP_DELAY = 10            # actions after the drop before a lift counts as a new one
STIFF_ELBOW = (50.0, 4.66067, 0.1)   # kp, kv, frictionloss of the nominal SO-101 elbow (next_round_20261108 stiff_elbow scene)
SOFT_ELBOW = (6.481, 0.9333, 0.3)    # the measured elbow (servo_model_20261108_so101, the so101_servo scenes)
ELBOW = JOINT_NAMES.index("elbow_flex")
FLAG_STRICT, FLAG_LIMIT, FLAG_CLIP, FLAG_UNSAFE = 1, 2, 4, 8


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    at_action: int = 100          # shift_* and swap_elbow: applied before the frame of this action is read
    magnitude_m: float = 0.020    # shift_placement / move_square; shift_cube uses ``cube_magnitude_m``
    cube_magnitude_m: float = 0.010
    seed: int = 0

    def __post_init__(self) -> None:
        if self.name not in PROBE_NAMES:
            raise ValueError(f"unknown probe {self.name!r}; choose from {PROBE_NAMES}")
        if not 0 <= int(self.at_action) < 480:
            raise ValueError("at_action must lie inside the 480-action episode")
        if not 0.0 < float(self.magnitude_m) <= 0.05 or not 0.0 < float(self.cube_magnitude_m) <= 0.02:
            raise ValueError("probe magnitudes out of range")


def probe_direction(seed: int, scenario_id: str, draw: int = 0) -> np.ndarray:
    """Unit world-xy direction, a pure function of (seed, scenario id, draw)."""
    rng = np.random.default_rng([int(seed), zlib.crc32(str(scenario_id).encode()), int(draw)])
    angle = rng.uniform(-np.pi, np.pi)
    return np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)


def cube_shift_on_square(adapter: Any, spec: ProbeSpec, scenario_id: str, skip: int = 0) -> np.ndarray:
    """A ``cube_magnitude_m`` shift that keeps the whole cube on its square (2 mm inside the edge); ``skip`` passes over
    that many valid draws (the shift plan tries the next one when the teacher cannot reach the first)."""
    bench = adapter.bench
    limit = bench.square_edge_m / 2 - bench.cube_edge_m / 2 - 0.002
    centre = np.asarray(adapter.square_center_xy, dtype=np.float64)
    yaw = float(adapter.square_yaw_rad)
    rot = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])     # world -> square frame
    local = rot @ (np.asarray(adapter.data.body("red_block").xpos[:2], dtype=np.float64) - centre)
    for draw in range(64):
        delta = probe_direction(spec.seed, scenario_id, draw) * spec.cube_magnitude_m
        if np.all(np.abs(local + rot @ delta) <= limit):
            if skip == 0:
                return delta
            skip -= 1
    inward = -local / (np.linalg.norm(local) + 1e-12)      # always valid: towards the square's centre
    return rot.T @ inward * spec.cube_magnitude_m


def planned_shift(adapter: Any, spec: ProbeSpec, scenario_id: str, draw: int = 0) -> np.ndarray | None:
    """The world-xy displacement the probe will apply (None for probes that displace nothing), for candidate ``draw``."""
    if spec.name in ("shift_placement", "move_square", "shift_and_swap"):
        return probe_direction(spec.seed, scenario_id, draw) * spec.magnitude_m
    if spec.name == "shift_cube":
        return cube_shift_on_square(adapter, spec, scenario_id, skip=draw)
    return None


def _forward(adapter: Any) -> None:
    import mujoco
    mujoco.mj_forward(adapter.model, adapter.data)


def shift_cube(adapter: Any, delta_xy: Sequence[float]) -> None:
    start = adapter._block_qpos
    adapter.data.qpos[start:start + 2] += np.asarray(delta_xy, dtype=np.float64)
    _forward(adapter)


def shift_square(adapter: Any, delta_xy: Sequence[float]) -> None:
    """Move the square (napkin geom) and the towel under it; the adapter rewrites both at the next reset."""
    delta = np.asarray(delta_xy, dtype=np.float64)
    adapter.model.geom_pos[adapter._napkin_geom, :2] += delta
    if adapter._towel_body >= 0:
        adapter.model.body_pos[adapter._towel_body, :2] += delta
    adapter.square_center_xy = (float(adapter.square_center_xy[0] + delta[0]), float(adapter.square_center_xy[1] + delta[1]))
    _forward(adapter)


def drop_cube(adapter: Any, pose7: Sequence[float]) -> None:
    """Put the cube back where the episode started, at rest (it leaves the closed jaws)."""
    start, dof = adapter._block_qpos, adapter._block_dof
    adapter.data.qpos[start:start + 7] = np.asarray(pose7, dtype=np.float64)
    adapter.data.qvel[dof:dof + 6] = 0.0
    _forward(adapter)


def swap_elbow(adapter: Any) -> str:
    """Replace the elbow's position servo by the other kind: a soft scene elbow becomes the stiff nominal one, a stiff
    one the measured soft one. ``adapter.reset`` restores the scene's own (``_physics_applied``). Returns the new kind."""
    actuator, dof = adapter._actuator_ids[ELBOW], adapter._joint_dof[ELBOW]
    kind = "stiff" if float(adapter.model.actuator_gainprm[actuator, 0]) < 20.0 else "soft"
    kp, kv, friction = STIFF_ELBOW if kind == "stiff" else SOFT_ELBOW
    adapter.model.actuator_gainprm[actuator, 0] = kp
    adapter.model.actuator_biasprm[actuator, 1] = -kp
    adapter.model.actuator_biasprm[actuator, 2] = -kv
    adapter.model.dof_frictionloss[dof] = friction
    adapter._physics_applied = True
    return kind


@contextmanager
def shifted_world(adapter: Any, cube_delta: Sequence[float] | None, square_delta: Sequence[float] | None):
    """Temporarily displace the cube and/or the square (for planning the informed teacher), then put them back."""
    start = adapter._block_qpos
    qpos = adapter.data.qpos[start:start + 7].copy()
    napkin = adapter.model.geom_pos[adapter._napkin_geom].copy()
    towel = adapter.model.body_pos[adapter._towel_body].copy() if adapter._towel_body >= 0 else None
    centre = adapter.square_center_xy
    try:
        if square_delta is not None:
            shift_square(adapter, square_delta)
        if cube_delta is not None:
            shift_cube(adapter, cube_delta)
        yield
    finally:
        adapter.data.qpos[start:start + 7] = qpos
        adapter.model.geom_pos[adapter._napkin_geom] = napkin
        if towel is not None:
            adapter.model.body_pos[adapter._towel_body] = towel
        adapter.square_center_xy = centre
        _forward(adapter)


class TeacherPolicy:
    """The scripted teacher as a probe control. ``informed`` plans with the probe's displacement known in advance:

    - ``reach``: the whole plan is made for the displaced cube (and square), so the reach must land on it;
    - ``place``: the plan is the ordinary one through the lift, then the displaced plan's waypoints from the hold at
      the top of the lift on, so the set-down must land on the moved square.
    """

    requires_pixels = False

    def __init__(self, informed: str | None = None, shift_xy: Sequence[float] | None = None, cube_only: bool = False) -> None:
        if informed not in (None, "reach", "place"):
            raise ValueError("informed must be None, 'reach' or 'place'")
        self.informed, self.shift_xy, self.cube_only = informed, shift_xy, bool(cube_only)
        self.controller = None

    def reset(self, adapter: Any) -> None:
        from so_arm101_v2.simulation.privileged import PrivilegedStagedController
        self.controller = PrivilegedStagedController()
        self.controller.reset(adapter)
        if self.informed is None or self.shift_xy is None:
            return
        other = PrivilegedStagedController()
        with shifted_world(adapter, self.shift_xy, None if self.cube_only else self.shift_xy):
            other.reset(adapter)
        if self.informed == "reach":
            self.controller = other
            return
        if tuple(other.boundaries) != tuple(self.controller.boundaries):
            raise RuntimeError("informed teacher: the displaced plan has a different stage schedule")
        first = len(self.controller.waypoints) - 6     # lift (hold), traverse, set down, released, released (hold), retreat
        self.controller.waypoints[first:] = other.waypoints[first:]

    def predict(self, image: Any, current_act: np.ndarray, adapter: Any | None = None) -> np.ndarray:
        return self.controller.predict(image, current_act, adapter)


def run_probe_rollout(adapter: Any, contract: Any, scenario: Any, policy_factory: Callable[[np.ndarray | None], Any],
                      spec: ProbeSpec, shift: Sequence[float] | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """One episode with ``spec`` applied; returns (summary row, per-action trace). ``shift`` is the displacement from the
    shift plan (``plan_shifts``); without one the first candidate is used, which the teacher may not be able to reach."""
    from so_arm101_v2.contracts.pick_place import PickPlaceEvaluationState, evaluate_pick_place_step
    from so_arm101_v2.simulation.rollout import _BLANK_WRIST_IMAGE

    adapter.reset(scenario)
    displaces = spec.name in ("shift_placement", "shift_cube", "move_square", "shift_and_swap")
    shift = (np.asarray(shift, dtype=np.float64) if shift is not None else planned_shift(adapter, spec, scenario.scenario_id)) if displaces else None
    policy = policy_factory(shift)
    reset = getattr(policy, "reset", None)
    if callable(reset):
        try:
            reset(adapter)
        except TypeError:
            reset()
    physics = adapter.physics_record
    start = adapter._block_qpos
    cube_start = adapter.data.qpos[start:start + 7].copy()
    render_wrist = bool(getattr(policy, "requires_pixels", True))
    frame_gated = render_wrist and hasattr(policy, "needs_frame")
    steps = int(contract.max_actions)
    trace = dict(jaw=np.zeros((steps, 3), np.float32), cube=np.zeros((steps, 3), np.float32), gripper=np.zeros(steps, np.float32),
                 requested=np.zeros((steps, 6), np.float32), height=np.zeros(steps, np.float32), flags=np.zeros(steps, np.uint8))
    state = PickPlaceEvaluationState()
    outcome: dict[str, Any] | None = None
    event_action: int | None = None
    footprint_inside = False
    for action in range(steps):
        if event_action is None and spec.name != "control":
            lifted = float(adapter.data.body("red_block").xpos[2] - adapter.initial_cube_height) >= LIFT_TRIGGER_M
            if spec.name in ("shift_placement", "shift_cube", "swap_elbow", "shift_and_swap") and action == spec.at_action:
                if spec.name in ("swap_elbow", "shift_and_swap"):
                    swap_elbow(adapter)
                if spec.name != "swap_elbow":
                    if spec.name != "shift_cube":
                        shift_square(adapter, shift)
                    shift_cube(adapter, shift)
                event_action = action
            elif spec.name == "drop_cube" and lifted:
                drop_cube(adapter, cube_start)
                event_action = action
            elif spec.name == "move_square" and lifted:
                shift_square(adapter, shift)
                event_action = action
        render_now = render_wrist and (not frame_gated or bool(policy.needs_frame))
        boundary = physics is not None and (bool(policy.needs_frame) if hasattr(policy, "needs_frame") else action % 90 == 0)
        raw, _, current = adapter.observation(render_pixels=render_now)
        if raw is None:
            raw = _BLANK_WRIST_IMAGE
        requested = policy.predict(raw, current, adapter)
        if boundary:
            adapter.inference_pause(action)
        command = adapter.apply_policy_command(requested)
        adapter.advance_control_period()
        measurement, _ = adapter.pick_place_measurement(command, footprint_edge_margin_m=contract.placement.footprint_edge_margin_m)
        pickup = measurement.pickup
        footprint_inside = bool(measurement.cube_footprint_inside)
        if outcome is None:
            state, evaluation = evaluate_pick_place_step(contract, measurement, state)
            if evaluation.terminated or evaluation.truncated:
                outcome = dict(success=bool(evaluation.success), pickup_completed=bool(evaluation.pickup_completed),
                               invalidated=bool(evaluation.invalidated), contract_actions=int(evaluation.actions_evaluated))
        trace["jaw"][action] = 0.5 * (adapter.data.site("fixed_jaw_tip").xpos + adapter.data.site("moving_jaw_tip").xpos)
        trace["cube"][action] = adapter.data.body("red_block").xpos
        trace["gripper"][action] = adapter.current_act()[5]
        trace["requested"][action] = command.requested_act
        trace["height"][action] = pickup.cube_height_gain_m
        trace["flags"][action] = (FLAG_STRICT * bool(pickup.strict_bilateral_grasp) + FLAG_LIMIT * bool(pickup.delta_limiter_activated)
                                  + FLAG_CLIP * bool(pickup.command_bound_violation) + FLAG_UNSAFE * bool(pickup.unsafe_contact))
    if outcome is None:
        raise RuntimeError("probe rollout produced no contract outcome")
    after = None if event_action is None else trace["height"][min(event_action + REGRASP_DELAY, steps - 1):]
    row = dict(scenario_id=scenario.scenario_id, probe=spec.name, **outcome, event_action=event_action,
               shift_xy_m=None if shift is None else [float(shift[0]), float(shift[1])],
               max_height_gain_m=float(trace["height"].max()),
               max_height_gain_after_event_m=None if after is None else float(after.max()),
               final_footprint_inside=footprint_inside,
               limit_frames=int(np.sum(trace["flags"] & FLAG_LIMIT > 0)), clip_frames=int(np.sum(trace["flags"] & FLAG_CLIP > 0)),
               unsafe_frames=int(np.sum(trace["flags"] & FLAG_UNSAFE > 0)))
    return row, trace


# ---------------------------------------------------------------------------------------------------------------------
# Running a cell: one policy on one scene, every probe on every scenario.

_ADAPTERS: dict[str, Any] = {}
_POLICIES: dict[tuple, Any] = {}


def _adapter(model_path: str) -> Any:
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    if model_path not in _ADAPTERS:
        _ADAPTERS[model_path] = MujocoTaskAdapter(model_path)
    return _ADAPTERS[model_path]


def policy_factory(kind: str, checkpoint: str | None, execute: int, spec: ProbeSpec, model: Any = None) -> Callable[[np.ndarray | None], Any]:
    """``kind``: teacher, teacher_informed, vision_chunked (CNN checkpoint) or vla (a loaded residual model)."""
    if kind == "teacher":
        return lambda shift: TeacherPolicy()
    if kind == "teacher_informed":
        informed = "place" if spec.name == "move_square" else "reach" if spec.name in ("shift_placement", "shift_cube", "shift_and_swap") else None
        return lambda shift: TeacherPolicy(informed, shift, cube_only=spec.name == "shift_cube")
    if kind == "vision_chunked":
        from so_arm101_v2.simulation.policy_specs import PolicySpec
        options = (("black_image", False), ("clamp_channels", (5,))) + ((("execute", int(execute)),) if execute != 90 else ())
        policy_spec = PolicySpec(kind="vision_chunked", checkpoint=str(checkpoint), options=options)

        def build(shift):
            key = (kind, checkpoint, execute)
            if key not in _POLICIES:        # one network per worker process; reset() clears its episode state
                _POLICIES[key] = policy_spec.build()
            return _POLICIES[key]
        return build
    if kind == "vla":
        from so_arm101_v2.simulation.residual_policy import ResidualChunkPolicy
        if model is None:
            raise ValueError("kind 'vla' needs the loaded model")
        return lambda shift: ResidualChunkPolicy(model, execute=execute)
    raise ValueError(f"unknown policy kind {kind!r}")


def _rollout_task(task: tuple) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model_path, contract_name, scenario, kind, checkpoint, execute, spec, shift = task
    from so_arm101_v2.contracts.pick_place import load_pick_place_contract
    adapter = _adapter(model_path)
    contract = load_pick_place_contract(contract_name, bench_config=adapter.bench)
    return run_probe_rollout(adapter, contract, scenario, policy_factory(kind, checkpoint, execute, spec), spec, shift)


def shift_key(spec_name: str) -> str | None:
    """The shift-plan entry a probe uses: move_square displaces the square by the same vector as shift_placement."""
    return {"shift_placement": "placement", "move_square": "placement", "shift_cube": "cube", "shift_and_swap": "placement"}.get(spec_name)


def _plan_task(task: tuple) -> tuple[str, dict[str, Any]]:
    """Teacher admission for one scenario's displacements: the first candidate the informed teacher completes.

    ``placement`` must work twice with the same vector: as a reach (cube and square displaced after the survey frame)
    and as a set-down (square displaced during the lift). ``cube`` is the cube alone on its square. A scenario with no
    admissible candidate gets None and is left out of that probe: a displacement the teacher cannot follow would test
    reachability, not looking."""
    model_path, contract_name, scenario, seed, magnitude_m, cube_magnitude_m, at_action, max_draws = task
    from so_arm101_v2.contracts.pick_place import load_pick_place_contract
    adapter = _adapter(model_path)
    contract = load_pick_place_contract(contract_name, bench_config=adapter.bench)

    def completes(name: str, shift: np.ndarray) -> bool:
        spec = ProbeSpec(name, at_action=at_action, magnitude_m=magnitude_m, cube_magnitude_m=cube_magnitude_m, seed=seed)
        try:
            row, _ = run_probe_rollout(adapter, contract, scenario, policy_factory("teacher_informed", None, 90, spec), spec, shift)
        except (RuntimeError, ValueError):       # the displaced plan has no IK solution
            return False
        return bool(row["success"])

    entry: dict[str, Any] = dict(placement=None, cube=None, placement_draw=None, cube_draw=None)
    for draw in range(max_draws):
        adapter.reset(scenario)
        shift = planned_shift(adapter, ProbeSpec("shift_placement", magnitude_m=magnitude_m, seed=seed), scenario.scenario_id, draw)
        if completes("shift_placement", shift) and completes("move_square", shift):
            entry.update(placement=[float(shift[0]), float(shift[1])], placement_draw=draw)
            break
    for draw in range(max_draws):
        adapter.reset(scenario)
        shift = planned_shift(adapter, ProbeSpec("shift_cube", cube_magnitude_m=cube_magnitude_m, seed=seed), scenario.scenario_id, draw)
        if completes("shift_cube", shift):
            entry.update(cube=[float(shift[0]), float(shift[1])], cube_draw=draw)
            break
    return scenario.scenario_id, entry


def plan_shifts(model_path: str | Path, suite: Any, *, seed: int = 0, magnitude_m: float = 0.020, cube_magnitude_m: float = 0.010,
                at_action: int = 100, max_draws: int = 12, workers: int = 1) -> dict[str, dict[str, Any]]:
    """Scenario id -> {placement, cube}: the displacement every policy gets on this scene (teacher-admitted)."""
    tasks = [(str(Path(model_path).resolve()), suite.task_contract, scenario, seed, magnitude_m, cube_magnitude_m, at_action, max_draws)
             for scenario in suite.scenarios]
    if workers <= 1:
        return dict(_plan_task(task) for task in tasks)
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(int(workers), len(tasks)), mp_context=multiprocessing.get_context("spawn")) as pool:
        return dict(pool.map(_plan_task, tasks))


def run_probe(model_path: str | Path, suite: Any, kind: str, checkpoint: str | None, execute: int, spec: ProbeSpec, *,
              workers: int = 1, model: Any = None, plan: Mapping[str, Mapping[str, Any]] | None = None,
              ) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Every scenario of ``suite`` under ``spec``; rows in suite order, traces stacked on a leading scenario axis.

    With a shift ``plan`` (``plan_shifts``) a displacing probe uses the planned vector and skips scenarios that have
    none; the caller pairs the rows with the control by scenario id."""
    model_path = str(Path(model_path).resolve())
    key = shift_key(spec.name)
    shifts = {s.scenario_id: (None if plan is None or key is None else plan[s.scenario_id][key]) for s in suite.scenarios}
    scenarios = [s for s in suite.scenarios if plan is None or key is None or shifts[s.scenario_id] is not None]
    if not scenarios:
        raise RuntimeError(f"the shift plan admits no scenario for {spec.name}")
    if kind == "vla":
        from so_arm101_v2.contracts.pick_place import load_pick_place_contract
        adapter = _adapter(model_path)
        contract = load_pick_place_contract(suite.task_contract, bench_config=adapter.bench)
        factory = policy_factory(kind, checkpoint, execute, spec, model)
        results = [run_probe_rollout(adapter, contract, scenario, factory, spec, shifts[scenario.scenario_id]) for scenario in scenarios]
    else:
        tasks = [(model_path, suite.task_contract, scenario, kind, checkpoint, execute, spec, shifts[scenario.scenario_id]) for scenario in scenarios]
        if workers <= 1:
            results = [_rollout_task(task) for task in tasks]
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            from so_arm101_v2.simulation.rollout import _rollout_worker_init
            with ProcessPoolExecutor(min(int(workers), len(tasks)), mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_rollout_worker_init) as pool:
                results = list(pool.map(_rollout_task, tasks))
    rows = [row for row, _ in results]
    traces = {name: np.stack([trace[name] for _, trace in results]) for name in results[0][1]}
    return rows, traces


# ---------------------------------------------------------------------------------------------------------------------
# Paired metrics: the probed episode against the same episode left alone.

def _interval(values: np.ndarray, seed: int = 0, resamples: int = 2000) -> list[float] | None:
    """95% bootstrap interval of the mean over scenarios."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _gain(displacement: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Per-scenario projection of an observed displacement onto the applied shift: 0 = ignored, 1 = followed."""
    return np.sum(displacement * shift, axis=1) / np.sum(shift * shift, axis=1)


def paired_metrics(spec: ProbeSpec, control_rows: Sequence[Mapping[str, Any]], control: Mapping[str, np.ndarray],
                   probe_rows: Sequence[Mapping[str, Any]], probe: Mapping[str, np.ndarray],
                   reference: tuple[Sequence[Mapping[str, Any]], Mapping[str, np.ndarray]] | None = None) -> dict[str, Any]:
    """Summary of one probe against its control. Rates are over all scenarios unless a key says otherwise.
    ``reference`` (rows, trace) replaces the control for the follow gain only (``GAIN_REFERENCE``)."""
    index = {r["scenario_id"]: i for i, r in enumerate(control_rows)}
    if any(r["scenario_id"] not in index for r in probe_rows):
        raise ValueError("every probed scenario needs its control episode")
    take = [index[r["scenario_id"]] for r in probe_rows]      # the probe may cover a subset (the shift plan's admissions)
    control_rows = [control_rows[i] for i in take]
    control = {name: values[take] for name, values in control.items()}
    n = len(probe_rows)
    picked_c = np.array([bool(r["pickup_completed"]) for r in control_rows])
    picked_p = np.array([bool(r["pickup_completed"]) for r in probe_rows])
    success_c = np.array([bool(r["success"]) for r in control_rows])
    success_p = np.array([bool(r["success"]) for r in probe_rows])
    fired = np.array([r["event_action"] is not None for r in probe_rows])
    events = np.array([r["event_action"] if r["event_action"] is not None else 0 for r in probe_rows])
    steps = probe["requested"].shape[1]
    after = np.arange(steps)[None, :] >= events[:, None]
    deviation = np.abs(probe["requested"] - control["requested"]).max(axis=2)        # (scenario, action), act units
    reaction = np.where(after, deviation, 0.0).max(axis=1)
    out: dict[str, Any] = dict(
        probe=asdict(spec), scenarios=n, fired=int(fired.sum()),
        control_pickups=int(picked_c.sum()), control_successes=int(success_c.sum()),
        probe_pickups=int(picked_p.sum()), probe_successes=int(success_p.sum()),
        pickup_retention=float(picked_p.sum() / picked_c.sum()) if picked_c.sum() else None,
        # Largest change in any commanded joint after the probe, against the unprobed episode: 0 = the policy did
        # exactly what it would have done anyway.
        reaction_act_median=float(np.median(reaction[fired])) if fired.any() else None,
        probe_limit_frames=int(sum(r["limit_frames"] for r in probe_rows)), control_limit_frames=int(sum(r["limit_frames"] for r in control_rows)),
    )
    if spec.name in ("shift_placement", "shift_cube", "shift_and_swap"):
        shift = np.array([r["shift_xy_m"] for r in probe_rows], dtype=np.float64)
        base_jaw = control["jaw"]
        if spec.name in GAIN_REFERENCE:
            if reference is None:
                raise ValueError(f"{spec.name} needs its {GAIN_REFERENCE[spec.name]} episodes as the gain reference")
            where = {r["scenario_id"]: i for i, r in enumerate(reference[0])}
            base_jaw = reference[1]["jaw"][[where[r["scenario_id"]] for r in probe_rows]]
            out["gain_reference"] = GAIN_REFERENCE[spec.name]
        moved = (probe["jaw"][:, CLOSURE_STEP, :2] - base_jaw[:, CLOSURE_STEP, :2]).astype(np.float64)
        gain = _gain(moved, shift)
        out.update(follow_gain_mean=float(gain.mean()), follow_gain_median=float(np.median(gain)), follow_gain_ci95=_interval(gain),
                   follow_gain_where_control_picked=_mean(gain[picked_c]),
                   residual_mm_median=float(np.median(np.linalg.norm(moved - shift, axis=1)) * 1000.0))
    elif spec.name == "move_square":
        keep = fired & picked_c
        shift = np.array([r["shift_xy_m"] for r in probe_rows], dtype=np.float64)
        moved = (probe["cube"][:, -1, :2] - control["cube"][:, -1, :2]).astype(np.float64)
        gain = _gain(moved, shift)
        out.update(compared=int(keep.sum()), follow_gain_mean=_mean(gain[keep]),
                   follow_gain_median=float(np.median(gain[keep])) if keep.any() else None, follow_gain_ci95=_interval(gain[keep]),
                   on_moved_square=int(sum(bool(r["final_footprint_inside"]) for r, k in zip(probe_rows, keep) if k)))
    elif spec.name == "drop_cube":
        regrasp = np.array([r["max_height_gain_after_event_m"] is not None and r["max_height_gain_after_event_m"] >= REGRASP_HEIGHT_M
                            for r in probe_rows])
        out.update(regrasps=int(regrasp[fired].sum()), regrasp_rate=float(regrasp[fired].mean()) if fired.any() else None)
    elif spec.name == "swap_elbow":
        moved = (probe["jaw"][:, CLOSURE_STEP] - control["jaw"][:, CLOSURE_STEP]).astype(np.float64) * 1000.0
        out.update(closure_change_mm_median=float(np.median(np.linalg.norm(moved, axis=1))),
                   closure_height_change_mm_median=float(np.median(moved[:, 2])),
                   closure_lateral_change_mm_median=float(np.median(np.linalg.norm(moved[:, :2], axis=1))))
    return out


__all__ = ["CLOSURE_STEP", "GAIN_REFERENCE", "PROBE_NAMES", "ProbeSpec", "STIFF_ELBOW", "TeacherPolicy", "cube_shift_on_square", "drop_cube",
           "SOFT_ELBOW", "paired_metrics", "plan_shifts", "planned_shift", "policy_factory", "probe_direction", "run_probe", "run_probe_rollout",
           "shift_cube", "shift_key", "shift_square", "shifted_world", "swap_elbow"]
