"""Bench suite construction and deterministic full-episode admission."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
from so_arm101_v2.contracts.bench import BenchConfig, scene_dependency_hash
from so_arm101_v2.contracts.pick_place import load_pick_place_contract, evaluate_pick_place_step, PickPlaceEvaluationState
from so_arm101_v2.data._serialization import content_sha256, write_immutable_json
from .adapter import MujocoTaskAdapter
from .privileged import PrivilegedStagedController
from .suites import SimulationScenario, SimulationSuite, suite_payload


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


def bench_suite(config, offsets, *, label, repeats):
    scenarios=tuple(SimulationScenario('nominal' if i==0 and dx==dy==0 else f'pose_{i:03d}',
        (config.square_center_xy[0]+dx,config.square_center_xy[1]+dy,config.cube_center[2]),
        tuple(float(v) for v in config.reset_qpos),(1.,0.,0.,0.))
        for i,(dx,dy) in enumerate(offsets))
    digest=content_sha256(dict(config=asdict(config),scenarios=[asdict(s) for s in scenarios]))[:12]
    return SimulationSuite(f'bench_pick_replace_v1_{label}_{digest}',1,config.task_id,repeats,False,scenarios)


def screen_scenario(model_path,scenario):
    adapter=MujocoTaskAdapter(model_path)
    try:
        contract=load_pick_place_contract('bench_pick_replace_v1',bench_config=adapter.bench)
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


def generate_bench_suite(model_path,config,*,seed,count,repeats,output_dir):
    from collections import Counter
    rng=np.random.default_rng(seed)
    offsets=[]
    reasons=Counter()
    for attempt in range(count*100):
        offset=tuple(float(v) for v in rng.uniform(-.010,.010,size=2))
        candidate=bench_suite(config,[offset],label='candidate',repeats=1).scenarios[0]
        ok,reason=screen_scenario(model_path,candidate)
        if ok:offsets.append(offset)
        else:reasons[reason]+=1
        if len(offsets)==count:break
        if attempt%20==0:print(f'SCREEN seed={seed} attempts={attempt+1} accepted={len(offsets)}/{count}',flush=True)
    if len(offsets)!=count:
        raise RuntimeError(f'cannot fill bench coverage: {len(offsets)}/{count}, rejections={dict(reasons)}')
    suite=bench_suite(config,offsets,label=f'seed{seed}_n{count}',repeats=repeats)
    payload=suite_payload(suite)
    payload['generator']=dict(seed=seed,requested=count,attempts=attempt+1,rejections=dict(reasons),
        offset_range_m=[-.010,.010],scene_dependencies_sha256=scene_dependency_hash(model_path))
    payload['content_sha256']=content_sha256(payload)
    path=Path(output_dir)/suite.suite_id/'suite.json'
    write_immutable_json(path,payload)
    return suite,path
