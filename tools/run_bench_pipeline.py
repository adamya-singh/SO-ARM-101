"""Gated bench oracle -> fresh data -> seed-202 vision -> closed-loop eval.

Run through tsp with PYTHONNOUSERSITE=1 MUJOCO_GL=egl. This tool never
connects to hardware. Its progress.json is the monitoring source of truth.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash
from so_arm101_v2.data._serialization import content_sha256, write_immutable_json
from so_arm101_v2.simulation.bench import bench_suite, generate_bench_suite, load_bench_verification
from so_arm101_v2.simulation.suites import load_suite_from_path, suite_payload
from so_arm101_v2.simulation.rollout import run_simulation_preflight, evaluate_closed_loop
from so_arm101_v2.simulation.oracle import capture_oracle_demonstrations
from so_arm101_v2.simulation.policy_specs import PolicySpec


def atomic_json(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(payload,indent=2,default=str)+'\n')
    os.replace(temp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--verification',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--stop-after',choices=['preflight','capture','train'],default='train')
    args=p.parse_args()
    import mujoco
    if mujoco.__version__!='3.9.0':raise RuntimeError('bench requires MuJoCo 3.9.0')
    bench=scene_bench_config(args.model)
    if bench is None or bench.viewing_qpos is None:raise RuntimeError('missing verified bench reset/viewing pose')
    bench.reset_qpos
    scene_hash=scene_dependency_hash(args.model)
    # Artifact-backed review record: named, hashed wrist images plus reviewer,
    # timestamp and notes. Refuses a bare boolean or a stale scene/reset.
    verification=load_bench_verification(args.verification,scene_hash=scene_hash,
        reset_evidence_sha256=bench.reset_evidence_sha256)
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    identity=dict(scene_dependencies_sha256=scene_hash,bench=asdict(bench),
        verification=verification,training_seed=202,max_steps=120000)
    write_immutable_json(root/'experiment.json',identity)
    state=dict(status='running',phase='preflight',pid=os.getpid(),started_at=time.time(),
        output_dir=str(root),scene_dependencies_sha256=scene_hash)
    def progress(**values):
        state.update(values);state['updated_at']=time.time()
        atomic_json(root/'progress.json',state)
    def assert_scene():
        if scene_dependency_hash(args.model)!=scene_hash:raise RuntimeError('scene changed during experiment')
    def preflight(suite):
        path=root/'simulation'/'preflight'/suite.suite_id/'evaluation.json'
        if not path.exists():
            result=run_simulation_preflight(args.model,root/'simulation',suite=suite,record_video=False,workers=1)
            path=result.report_json
        report=json.loads(path.read_text())
        if not report.get('environment_proven') or not report.get('deterministic'):
            raise RuntimeError(f'oracle gate failed: {path}')
        return path
    tracker=None
    try:
        progress()
        stage=bench_suite(bench,[(0,0),(.01,0),(-.01,0),(0,.01),(0,-.01)],label='certification',repeats=3)
        preflight(stage)
        assert_scene()
        if args.stop_after=='preflight':
            progress(status='complete',phase='preflight_complete');return 0
        suites={}
        for label,seed,count,repeats in [('train',12,400,1),('heldout',8,10,3)]:
            progress(phase=f'screen_{label}')
            pointer=root/f'{label}_suite.path'
            if pointer.exists():
                path=Path(pointer.read_text().strip());suite=load_suite_from_path(path)
            else:
                suite,path=generate_bench_suite(args.model,bench,seed=seed,count=count,repeats=repeats,output_dir=root/'suites')
                pointer.write_text(str(path.resolve())+'\n')
            suites[label]=suite
            preflight(suite)
        assert_scene()
        progress(phase='capture')
        pointer=root/'capture_manifest.path'
        if pointer.exists():manifest=Path(pointer.read_text().strip())
        else:
            capture=capture_oracle_demonstrations(args.model,suites['train'],preflight(suites['train']),
                root/'capture',scenario='all',record_video=False,teacher_horizon=480,store_frames=True)
            manifest=capture.manifest
            pointer.write_text(str(manifest.resolve())+'\n')
        progress(capture_manifest=str(manifest))
        assert_scene()
        if args.stop_after=='capture':progress(status='complete',phase='capture_complete');return 0
        import wandb
        from so_arm101_v2.learning.vision import train_vision_chunked,VisionChunkedConfig
        config=VisionChunkedConfig(seed=202,max_steps=120000)
        id_file=root/'wandb.json'
        run_id=json.loads(id_file.read_text())['id'] if id_file.exists() else wandb.util.generate_id()
        atomic_json(id_file,dict(id=run_id))
        tracker=wandb.init(project='so-arm101-v2-scaling',group='bench-pick-replace-20260906',
            name='bench-pick-replace-v1-s202-120k',id=run_id,resume='allow',mode='online',
            config={**identity,'training':asdict(config),'capture_manifest':str(manifest)})
        atomic_json(id_file,dict(id=tracker.id,url=tracker.url))
        print(f'WANDB_URL {tracker.url}',flush=True)
        progress(phase='training_initialization',wandb_url=tracker.url,wandb_id=tracker.id)
        scratch=root/'scratch'/'vision.pt'
        history=[]
        telemetry=(root/'training.jsonl').open('a')
        prior_progress=json.loads((root/'training_clock.json').read_text()) if (root/'training_clock.json').exists() else {}
        def on_loss(step,loss):
            now=time.time()
            if not prior_progress:
                prior_progress.update(first_optimizer_step_at=now)
                atomic_json(root/'training_clock.json',prior_progress)
            history.append((now,step))
            history[:]=history[-20:]
            rate=(step-history[0][1])/max(now-history[0][0],1e-6)
            values=dict(step=step,loss=loss,timestamp=now,steps_per_second=rate,
                first_optimizer_step_at=prior_progress['first_optimizer_step_at'],
                checkpoint_at=scratch.stat().st_mtime if scratch.exists() else None)
            telemetry.write(json.dumps(values)+'\n');telemetry.flush()
            progress(phase='training',**values)
            try:tracker.log({'train/batch_normalized_mse':loss,'train/steps_per_second':rate},step=step)
            except Exception as exc:progress(tracking_error=str(exc))
            print(f'TRAIN step={step} loss={loss:.8g} steps_per_second={rate:.2f}',flush=True)
        try:
            trained=train_vision_chunked(manifest,root,config=config,on_loss=on_loss,
                scratch_checkpoint=scratch,checkpoint_interval=5000)
        finally:telemetry.close()
        progress(phase='evaluation',checkpoint=str(trained.checkpoint))
        assert_scene()
        nominal=bench_suite(bench,[(0,0)],label='nominal_eval',repeats=3)
        for label,suite in [('nominal',nominal),('heldout',suites['heldout'])]:
            policies={}
            for black in (False,True):
                name='vision_black' if black else 'vision'
                policies[name]=PolicySpec(kind='vision_chunked',checkpoint=str(trained.checkpoint),
                    options=(('clamp_channels',(5,)),('black_image',black)))
            result=evaluate_closed_loop(args.model,suite,policies,root/'evaluations'/label,
                environment_proven=True,record_video=True,workers=1,provenance=identity)
            report=json.loads(result.report_json.read_text())
            for name in policies:
                rows=[r for r in report['rollouts'] if r['policy_id']==name]
                tracker.summary[f'{label}/{name}/successes']=sum(r['success'] and not r['invalidated'] for r in rows)
                tracker.summary[f'{label}/{name}/safety_frames']=sum(sum(r[k] for k in
                    ('clipping_frames','limiting_frames','nonfinite_frames','unsafe_contact_frames')) for r in rows)
            tracker.summary[f'{label}/report']=str(result.report_json)
        tracker.finish();tracker=None
        progress(status='complete',phase='complete')
        return 0
    except BaseException as exc:
        progress(status='failed',error=str(exc),traceback=traceback.format_exc())
        if tracker is not None:tracker.finish(exit_code=1)
        raise

if __name__=='__main__':raise SystemExit(main())
