"""Bench suite construction and deterministic full-episode admission."""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from so_arm101_v2.contracts.appearance import APPEARANCE_RESOLVER_VERSION, appearance_seeds
from so_arm101_v2.contracts.placement import PLACEMENT_RESOLVER_VERSION, PlacementRegime, placement_draws
from so_arm101_v2.contracts.bench import BenchConfig, scene_dependency_hash
from so_arm101_v2.contracts.pick_place import load_pick_place_contract, evaluate_pick_place_step, PickPlaceEvaluationState
from so_arm101_v2.data._serialization import content_sha256, write_immutable_json
from .adapter import MujocoTaskAdapter
from .privileged import PrivilegedStagedController
from .suites import SimulationScenario, SimulationSuite, scenario_record, suite_payload


def load_bench_verification(path, *, scene_hash, reset_evidence_sha256):
    """Load the reset/camera review record that gates the pipeline.

    The record must name the physical and simulated wrist images that were
    compared (paths absolute or relative to the record), carry their SHA-256
    digests, a reviewer, an ISO timestamp and notes, and match the active
    scene and reset evidence. A bare boolean is refused.
    """
    import hashlib
    import json
    from datetime import datetime
    path = Path(path)
    raw = json.loads(path.read_text())
    if raw.get("scene_dependencies_sha256") != scene_hash:
        raise RuntimeError("verification scene hash is stale; re-review after any scene change")
    if reset_evidence_sha256 is None or raw.get("reset_evidence_sha256") != reset_evidence_sha256:
        raise RuntimeError("verification reset evidence does not match the active bench config")
    review = raw.get("camera_review")
    if not isinstance(review, dict):
        raise RuntimeError("verification lacks an artifact-backed camera_review record")
    for key in ("physical_wrist_image", "simulated_wrist_image"):
        image = Path(str(review.get(key, "")))
        if not str(image):
            raise RuntimeError(f"camera review lacks {key}")
        if not image.is_absolute():
            image = path.parent / image
        if not image.is_file():
            raise RuntimeError(f"camera review {key} is missing: {image}")
        if review.get(f"{key}_sha256") != hashlib.sha256(image.read_bytes()).hexdigest():
            raise RuntimeError(f"camera review {key} hash does not match the file on disk")
    try:
        datetime.fromisoformat(str(review["reviewed_at"]))
    except (KeyError, ValueError) as exc:
        raise RuntimeError("camera review needs an ISO reviewed_at timestamp") from exc
    if not str(review.get("reviewer", "")).strip() or not str(review.get("notes", "")).strip():
        raise RuntimeError("camera review needs a reviewer and notes")
    return raw


# Teacher certification poses. Fixed-square configs keep the original five cube offsets (nominal + four 10 mm
# axis offsets, 15/15). With a placement regime the certification spans the teacher-reachable rectangle
# (reach scan 2026-09-10) with yawed squares: ten placements x three repeats = 30/30.
CERTIFICATION_OFFSETS = [(0, 0), (.01, 0), (-.01, 0), (0, .01), (0, -.01)]
CERTIFICATION_PLACEMENTS = [   # (square x, square y, yaw deg)
    (0.0, 0.2805353, 0.0), (0.0, 0.2805353, 45.0), (0.0, 0.179, -30.0), (0.119, 0.179, 0.0), (-0.119, 0.179, 45.0),
    (0.119, 0.242, -45.0), (-0.119, 0.242, 20.0), (0.13, 0.14, 0.0), (-0.12, 0.15, -20.0), (0.059, 0.29, 30.0),
]


def certification_suite(config, *, repeats=3):
    """The certification suite for this config: placement set when a regime is present, the legacy offsets otherwise."""
    if config.placement is None:
        return bench_suite(config, CERTIFICATION_OFFSETS, label='certification', repeats=repeats)
    return bench_suite(config, [(0, 0)] * len(CERTIFICATION_PLACEMENTS), label='certification', repeats=repeats,
                       square_centers=[(x, y) for x, y, _ in CERTIFICATION_PLACEMENTS],
                       square_yaws=[float(np.deg2rad(w)) for _, _, w in CERTIFICATION_PLACEMENTS])


def bench_suite(config, offsets, *, label, repeats, appearance_seeds=None, square_centers=None, square_yaws=None):
    """Scenarios at the reset pose with cube offsets from the square centre.

    ``appearance_seeds`` (one per offset) makes each look randomized; ``square_centers`` /
    ``square_yaws`` (one per offset) place and yaw the square itself (placement regime
    required); the cube sits at the square centre plus the offset, rotated with the square.
    """
    if appearance_seeds is not None and len(appearance_seeds)!=len(offsets):
        raise ValueError('one appearance seed per offset is required')
    if appearance_seeds is not None and config.appearance is None:
        raise ValueError('appearance seeds require an appearance regime in the bench config')
    if (square_centers is not None or square_yaws is not None) and config.placement is None:
        raise ValueError('square placements require a placement regime in the bench config')
    if square_centers is not None and len(square_centers)!=len(offsets):
        raise ValueError('one square centre per offset is required')
    if square_yaws is not None and len(square_yaws)!=len(offsets):
        raise ValueError('one square yaw per offset is required')
    from so_arm101_v2.contracts.placement import yaw_quaternion
    scenarios=[]
    for i,(dx,dy) in enumerate(offsets):
        centre=tuple(float(v) for v in (square_centers[i] if square_centers is not None else config.square_center_xy))
        yaw=float(square_yaws[i]) if square_yaws is not None else 0.0
        c,s=np.cos(yaw),np.sin(yaw)
        cube=(centre[0]+c*dx-s*dy,centre[1]+s*dx+c*dy,config.cube_center[2])   # the offset rotates with the square
        placed=square_centers is not None or square_yaws is not None
        scenarios.append(SimulationScenario('nominal' if i==0 and dx==dy==0 and not placed else f'pose_{i:03d}',
            tuple(float(v) for v in cube),tuple(float(v) for v in config.reset_qpos),yaw_quaternion(yaw),
            fixed_appearance=appearance_seeds is None,
            appearance_seed=None if appearance_seeds is None else int(appearance_seeds[i]),
            square_center_xy=centre if square_centers is not None else None,square_yaw_rad=yaw))
    scenarios=tuple(scenarios)
    digest=content_sha256(dict(config=asdict(config),scenarios=[scenario_record(s) for s in scenarios]))[:12]
    return SimulationSuite(f'bench_pick_replace_v1_{label}_{digest}',1,config.task_id,repeats,False,scenarios)


def appearance_product(config, suite, *, per_scenario, seed, label):
    """Every pose of ``suite`` under ``per_scenario`` appearance draws: unique scenarios, repeats 1.

    Poses are not re-screened (physics is identical under a draw); the seeds come
    from stream 1 of the appearance seed generator so they never coincide with a
    training suite drawn from stream 0 with the same seed.
    """
    if config.appearance is None:
        raise ValueError('appearance_product requires an appearance regime in the bench config')
    seeds=appearance_seeds(seed,per_scenario*len(suite.scenarios),stream=1)
    scenarios=tuple(replace(s,scenario_id=f'{s.scenario_id}_a{k}',fixed_appearance=False,
        appearance_seed=seeds[i*per_scenario+k])
        for i,s in enumerate(suite.scenarios) for k in range(per_scenario))
    digest=content_sha256(dict(config=asdict(config),source=suite.suite_id,scenarios=[scenario_record(s) for s in scenarios]))[:12]
    return SimulationSuite(f'bench_pick_replace_v1_{label}_{digest}',1,config.task_id,1,False,scenarios)


def screen_scenario(model_path,scenario):
    """Full-episode teacher admission; placement scenarios must also be visible from the survey (viewing) pose."""
    adapter=MujocoTaskAdapter(model_path)
    try:
        contract=load_pick_place_contract('bench_pick_replace_v1',bench_config=adapter.bench)
        bench=adapter.bench
        if bench.placement is not None and scenario.square_center_xy is not None:
            from so_arm101_v2.physical.placement import cube_visible_at
            regime=PlacementRegime.from_mapping(bench.placement)
            visible=cube_visible_at(model_path,bench,bench.viewing_qpos,scenario.cube_position_m,yaw_rad=scenario.square_yaw_rad,
                margin_px=regime.survey_visibility_margin_px)
            if not visible['visible']:return False,'not_visible_from_survey'
        adapter.reset(scenario)
        controller=PrivilegedStagedController()
        controller.reset(adapter)
        state=PickPlaceEvaluationState()
        for _ in range(contract.max_actions):
            current=adapter.current_act()
            requested=controller.predict(None,current,adapter)
            command=adapter.apply_policy_command(requested)
            adapter.advance_control_period()
            measurement,_=adapter.pick_place_measurement(command)
            state,result=evaluate_pick_place_step(contract,measurement,state)
            if result.invalidated:return False,'safety_invalidation'
            if result.terminated or result.truncated:
                return result.success,'success' if result.success else 'incomplete'
        return False,'incomplete'
    except (RuntimeError,ValueError) as exc:
        return False,str(exc)
    finally:adapter.close()


def generate_bench_suite(model_path,config,*,seed,count,repeats,output_dir,randomize_appearance=False,randomize_placement=False):
    """Screen ``count`` poses (offset stream ``seed``); with ``randomize_appearance`` each accepted pose also gets an
    appearance seed from a separate stream, so the accepted poses equal the fixed-appearance suite's. With
    ``randomize_placement`` every candidate also gets a square centre and yaw from the placement stream (stream 2),
    and the teacher screen (plus the survey-pose visibility check) decides which placements are admitted."""
    from collections import Counter
    if randomize_appearance and config.appearance is None:
        raise ValueError('randomize_appearance requires an appearance regime in the bench config')
    if randomize_placement and config.placement is None:
        raise ValueError('randomize_placement requires a placement regime in the bench config')
    regime=PlacementRegime.from_mapping(config.placement) if randomize_placement else None
    rng=np.random.default_rng(seed)
    draws=placement_draws(seed,count*100,regime) if randomize_placement else None
    cube_offset=regime.cube_offset_m if randomize_placement else .010
    offsets=[];centres=[];yaws=[];rejected=[]
    reasons=Counter()
    for attempt in range(count*100):
        offset=tuple(float(v) for v in rng.uniform(-cube_offset,cube_offset,size=2))
        centre=(draws[attempt][0],draws[attempt][1]) if randomize_placement else None
        yaw=draws[attempt][2] if randomize_placement else None
        candidate=bench_suite(config,[offset],label='candidate',repeats=1,
            square_centers=None if centre is None else [centre],square_yaws=None if yaw is None else [yaw]).scenarios[0]
        ok,reason=screen_scenario(model_path,candidate)
        if ok:
            offsets.append(offset)
            if randomize_placement:centres.append(centre);yaws.append(yaw)
        else:
            reasons[reason]+=1
            if randomize_placement and len(rejected)<2000:rejected.append([round(centre[0],4),round(centre[1],4),round(float(np.degrees(yaw)),1),reason[:40]])
        if len(offsets)==count:break
        if attempt%20==0:print(f'SCREEN seed={seed} attempts={attempt+1} accepted={len(offsets)}/{count}',flush=True)
    if len(offsets)!=count:
        raise RuntimeError(f'cannot fill bench coverage: {len(offsets)}/{count}, rejections={dict(reasons)}')
    seeds=appearance_seeds(seed,count,stream=0) if randomize_appearance else None
    suite=bench_suite(config,offsets,label=f'seed{seed}_n{count}'+('_appearance' if randomize_appearance else '')+('_placement' if randomize_placement else ''),
        repeats=repeats,appearance_seeds=seeds,square_centers=centres if randomize_placement else None,square_yaws=yaws if randomize_placement else None)
    payload=suite_payload(suite)
    payload['generator']=dict(seed=seed,requested=count,attempts=attempt+1,rejections=dict(reasons),
        offset_range_m=[-.010,.010],scene_dependencies_sha256=scene_dependency_hash(model_path))
    if randomize_appearance:
        payload['generator']['appearance']=dict(regime=dict(config.appearance),resolver=APPEARANCE_RESOLVER_VERSION,
            seed_stream=dict(seed=seed,stream=0),seeds=list(seeds))
    if randomize_placement:
        accepted=np.asarray(centres)
        payload['generator']['placement']=dict(regime=dict(config.placement),resolver=PLACEMENT_RESOLVER_VERSION,
            seed_stream=dict(seed=seed,stream=2),cube_offset_range_m=[-cube_offset,cube_offset],
            accepted_bbox_m=dict(x=[round(float(accepted[:,0].min()),4),round(float(accepted[:,0].max()),4)],
                                 y=[round(float(accepted[:,1].min()),4),round(float(accepted[:,1].max()),4)]),
            rejections_by_reason=dict(reasons),rejected_placements=rejected)
    payload['content_sha256']=content_sha256(payload)
    path=Path(output_dir)/suite.suite_id/'suite.json'
    write_immutable_json(path,payload)
    return suite,path
