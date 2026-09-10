"""Gated bench oracle -> fresh data -> seed-202 vision -> closed-loop eval.

Run through tsp with PYTHONNOUSERSITE=1 MUJOCO_GL=egl (see
simulation_code/queue_bench_pipeline.sh). This tool never connects to hardware.
Its progress.json is the monitoring source of truth; tools/bench_health_check.py
reads it.

Gates, in order: MuJoCo 3.9.0; verified reset and artifact-backed camera review
(load_bench_verification); teacher certification 15/15 on this scene; screened
training and held-out suites with their own preflights; then capture, one vision
run and evaluation. A failed gate raises before anything downstream runs.

--rehearsal is a loudly labelled dry run of the whole chain at toy scale: it
skips the camera-review gate, forces the output directory under a "rehearsal"
folder, screens a handful of poses, trains for a few dozen steps, and logs to
W&B offline. Rehearsal artifacts are never eligible as data or results.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from so_arm101_v2.contracts.appearance import APPEARANCE_RESOLVER_VERSION
from so_arm101_v2.contracts.placement import PLACEMENT_RESOLVER_VERSION
from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash
from so_arm101_v2.data._serialization import content_sha256, write_immutable_json
from so_arm101_v2.physical.dry_pass import REAL_FRAME_GATE_VERSION, load_boundary_frame
from so_arm101_v2.simulation.bench import CERTIFICATION_PLACEMENTS, appearance_product, bench_suite, certification_suite, generate_bench_suite, load_bench_verification
from so_arm101_v2.simulation.contact import GRASP_DETECTOR_VERSION
from so_arm101_v2.simulation.suites import load_suite_from_path
from so_arm101_v2.simulation.rollout import run_simulation_preflight, evaluate_closed_loop
from so_arm101_v2.simulation.oracle import capture_oracle_demonstrations
from so_arm101_v2.simulation.policy_specs import PolicySpec

PREFIX_TOLERANCE_RAD = 0.05
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_FRAME_EPISODES = [ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/physical/episode_02_20260909']
HELDOUT_APPEARANCE_PER_POSE = 3


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(payload, indent=2, default=str) + '\n')
    os.replace(temp, path)


def prefix_success(report: dict, bench, observation_steps: int) -> dict:
    """Per policy: fraction of rollouts whose arm is at the viewing pose when the first image refreshes."""
    viewing = np.asarray(bench.viewing_qpos, dtype=np.float64)[:5]
    result = {}
    for rollout in report['rollouts']:
        path = rollout.get('telemetry_path')
        if not path or not Path(path).exists():
            continue
        rows = json.loads(Path(path).read_text())['rows']
        if len(rows) < observation_steps:
            ok = False
        else:
            qpos = np.asarray(rows[observation_steps - 1]['robot_qpos'], dtype=np.float64)[:5]
            ok = bool(np.max(np.abs(qpos - viewing)) <= PREFIX_TOLERANCE_RAD)
        bucket = result.setdefault(rollout['policy_id'], dict(prefix_ok=0, rollouts=0))
        bucket['rollouts'] += 1; bucket['prefix_ok'] += ok
    return result


def success_by_region(rows: list, suite, bench) -> dict:
    """Successes per (y band, x band) of the square centre: near/mid/far forward, left/centre/right lateral."""
    regime = bench.placement_regime
    by_id = {s.scenario_id: s for s in suite.scenarios}
    result = {}
    for row in rows:
        scenario = by_id.get(row['scenario_id'])
        centre = scenario.square_center_xy if scenario is not None and scenario.square_center_xy is not None else bench.square_center_xy
        fy = (centre[1] - regime.y_range_m[0]) / (regime.y_range_m[1] - regime.y_range_m[0])
        fx = (centre[0] - regime.x_range_m[0]) / (regime.x_range_m[1] - regime.x_range_m[0])
        band = ('near', 'mid', 'far')[min(2, int(fy * 3))] + '/' + ('left', 'centre', 'right')[min(2, int(fx * 3))]
        bucket = result.setdefault(band, dict(successes=0, rollouts=0))
        bucket['rollouts'] += 1
        bucket['successes'] += int(bool(row['success']) and not row['invalidated'])
    return dict(sorted(result.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--verification', type=Path, help='artifact-backed reset/camera review record (required unless --rehearsal)')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--stop-after', choices=['preflight', 'screen', 'capture', 'train'], default='train')
    p.add_argument('--rehearsal', action='store_true', help='toy-scale dry run; skips the camera gate; output must be under a rehearsal/ folder')
    p.add_argument('--workers', type=int, default=None, help='process-parallel preflights, capture and evaluation (digest-neutral; default auto, 1 = sequential)')
    p.add_argument('--appearance', action='store_true',
                   help='the bench_appearance recipe: training poses drawn with per-scenario appearance seeds, the held-out poses also evaluated '
                        'under 3 appearance draws each, and the offline real-frame gate run on the recorded physical reset frame(s). '
                        'Requires (and is required by) an appearance block in bench_config.json.')
    p.add_argument('--placement', action='store_true',
                   help='the bench_placement recipe: training and held-out squares (with the cube) drawn anywhere in the placement '
                        'rectangle with yaw, screened by the teacher and the survey-pose visibility check; certification over the '
                        'placement set; success reported by region. Requires (and is required by) a placement block in bench_config.json.')
    p.add_argument('--train-count', type=int, default=400, help='screened training poses (the registered recipe uses 400; the placement recipe needs more coverage)')
    p.add_argument('--max-steps', type=int, default=120000, help='optimizer steps for the single training run (registered recipe: 120000)')
    p.add_argument('--frame-store', choices=['zlib', 'gpu', 'off'], default='zlib',
                   help="where training reads frames: 'zlib' = lossless RAM cache (skipped when it would exceed half of RAM), "
                        "'gpu' = the (strided) frames as one uint8 tensor on the training device, 'off' = the memmap")
    p.add_argument('--frame-stride', type=int, default=1,
                   help='train on every N-th row of each episode (1 = the historical recipe; 3 with --frame-store gpu fits a '
                        '400-episode capture on a 24 GB GPU and removes the disk from the training loop)')
    p.add_argument('--real-frame-episode', type=Path, action='append', default=None,
                   help='recorded physical episode directory for the real-frame gate (repeatable; default physical/episode_02_20260909)')
    args = p.parse_args()
    import mujoco
    if mujoco.__version__ != '3.9.0':
        raise RuntimeError('bench requires MuJoCo 3.9.0')
    bench = scene_bench_config(args.model)
    if bench is None or bench.viewing_qpos is None:
        raise RuntimeError('missing verified bench reset/viewing pose')
    bench.reset_qpos
    if bool(args.appearance) != (bench.appearance is not None):
        raise RuntimeError('--appearance must be passed exactly when bench_config.json carries an appearance regime '
                           f'(flag={bool(args.appearance)}, regime={"present" if bench.appearance is not None else "absent"})')
    if bool(args.placement) != (bench.placement is not None):
        raise RuntimeError('--placement must be passed exactly when bench_config.json carries a placement regime '
                           f'(flag={bool(args.placement)}, regime={"present" if bench.placement is not None else "absent"})')
    scene_hash = scene_dependency_hash(args.model)
    real_frame_episodes = [Path(p) for p in (args.real_frame_episode or DEFAULT_REAL_FRAME_EPISODES)] if args.appearance else []
    real_frames = []
    for episode_dir in real_frame_episodes:
        if not episode_dir.exists():
            if args.rehearsal:
                continue
            raise RuntimeError(f'real-frame episode directory missing: {episode_dir}')
        _image, _anchor, evidence = load_boundary_frame(episode_dir, step=0)   # hash-verified now, so the identity is honest
        real_frames.append(evidence)
    root = args.output_dir.resolve()
    if args.rehearsal:
        if 'rehearsal' not in root.parts:
            raise RuntimeError('--rehearsal output must live under a directory named "rehearsal"')
        verification = dict(rehearsal=True, camera_review='SKIPPED: rehearsal dry run, not a verified experiment')
        counts = dict(train=(12, 3, 1), heldout=(8, 2, 1)); max_steps, checkpoint_interval = 30, 10
        wandb_mode, group, run_name = 'offline', 'bench-rehearsal', 'bench-rehearsal'
        print('REHEARSAL MODE: camera gate skipped, toy scale, W&B offline. Nothing here is a result.', flush=True)
    else:
        if args.verification is None:
            p.error('--verification is required outside --rehearsal')
        # Artifact-backed review record: named, hashed wrist images plus reviewer,
        # timestamp and notes. Refuses a bare boolean or a stale scene/reset.
        verification = load_bench_verification(args.verification, scene_hash=scene_hash,
                                               reset_evidence_sha256=bench.reset_evidence_sha256)
        counts = dict(train=(12, int(args.train_count), 1), heldout=(8, 10, 3)); max_steps, checkpoint_interval = int(args.max_steps), 5000
        wandb_mode, group, run_name = 'online', 'bench-pick-replace-20260906', 'bench-pick-replace-v1-s202-120k' + ('-appearance' if args.appearance else '') + ('-placement' if args.placement else '')
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(scene_dependencies_sha256=scene_hash, grasp_detector=GRASP_DETECTOR_VERSION, bench=asdict(bench),
                    verification=verification, training_seed=202, max_steps=max_steps, rehearsal=bool(args.rehearsal),
                    suite_seeds={k: v[0] for k, v in counts.items()}, suite_counts={k: v[1] for k, v in counts.items()})
    if args.appearance:
        identity['recipe'] = 'bench_appearance_v1'
        identity['appearance'] = dict(regime=dict(bench.appearance), resolver=APPEARANCE_RESOLVER_VERSION,
                                      train_suite='per-pose appearance seeds (stream 0)',
                                      heldout_appearance=dict(per_pose=HELDOUT_APPEARANCE_PER_POSE, seed=counts['heldout'][0], stream=1))
        identity['real_frame_gate'] = dict(gate=REAL_FRAME_GATE_VERSION, frames=real_frames)
    else:
        identity['recipe'] = 'fixed_appearance'
    if args.placement:
        identity['recipe'] += '+bench_placement_v1'
        identity['placement'] = dict(regime=dict(bench.placement), resolver=PLACEMENT_RESOLVER_VERSION,
                                     train_suite='square centre + yaw per pose (stream 2), teacher + survey-visibility screened',
                                     heldout_suite='same, seed 8', certification=CERTIFICATION_PLACEMENTS,
                                     viewing_qpos=[float(v) for v in bench.viewing_qpos])
    write_immutable_json(root / 'experiment.json', identity)
    started = time.time()
    state = dict(status='running', phase='preflight', pid=os.getpid(), started_at=started, output_dir=str(root),
                 scene_dependencies_sha256=scene_hash, max_steps=max_steps, rehearsal=bool(args.rehearsal))

    def progress(**values):
        state.update(values); state['updated_at'] = time.time(); state['elapsed_s'] = round(time.time() - started, 1)
        atomic_json(root / 'progress.json', state)

    def assert_scene():
        if scene_dependency_hash(args.model) != scene_hash:
            raise RuntimeError('scene changed during experiment')

    def preflight(suite):
        path = root / 'simulation' / 'preflight' / suite.suite_id / 'evaluation.json'
        if not path.exists():
            result = run_simulation_preflight(args.model, root / 'simulation', suite=suite, record_video=False, workers=args.workers)
            path = Path(result.report_json)
        report = json.loads(path.read_text())
        if not report.get('environment_proven') or not report.get('deterministic'):
            raise RuntimeError(f'oracle gate failed: {path}')
        return path

    tracker = None
    try:
        progress()
        # W&B is initialised first so a connectivity problem fails the run before
        # hours of screening, and so every phase is visible on the run page.
        import wandb
        id_file = root / 'wandb.json'
        run_id = json.loads(id_file.read_text())['id'] if id_file.exists() else wandb.util.generate_id()
        atomic_json(id_file, dict(id=run_id, mode=wandb_mode))
        tracker = wandb.init(project='so-arm101-v2-scaling', group=group, name=run_name, id=run_id, resume='allow',
                             mode=wandb_mode, config=identity, settings=wandb.Settings(init_timeout=120))
        atomic_json(id_file, dict(id=tracker.id, url=tracker.url, mode=wandb_mode))
        print(f'WANDB_URL {tracker.url}', flush=True)
        progress(wandb_url=tracker.url, wandb_id=tracker.id, tracking_last_ok=time.time())

        def track(payload, step=None):
            try:
                tracker.log(payload, step=step); state['tracking_last_ok'] = time.time(); state.pop('tracking_error', None)
            except Exception as exc:  # tracking must never stop the experiment
                state['tracking_error'] = str(exc)

        stage = certification_suite(bench, repeats=3)
        preflight(stage); assert_scene(); track({'phase/certification_passed': 1})
        if args.stop_after == 'preflight':
            progress(status='complete', phase='preflight_complete'); return 0
        suites = {}; provenance = dict(suites={})
        for label, (seed, count, repeats) in counts.items():
            progress(phase=f'screen_{label}')
            pointer = root / f'{label}_suite.path'
            if pointer.exists():
                path = Path(pointer.read_text().strip()); suite = load_suite_from_path(path)
            else:
                suite, path = generate_bench_suite(args.model, bench, seed=seed, count=count, repeats=repeats, output_dir=root / 'suites',
                                                   randomize_appearance=bool(args.appearance) and label == 'train',
                                                   randomize_placement=bool(args.placement))
                pointer.write_text(str(path.resolve()) + '\n')
            suites[label] = suite
            provenance['suites'][label] = dict(suite_id=suite.suite_id, path=str(path), preflight=str(preflight(suite)))
        assert_scene()
        if args.stop_after == 'screen':
            atomic_json(root / 'provenance.json', provenance); progress(status='complete', phase='screen_complete'); return 0
        progress(phase='capture')
        pointer = root / 'capture_manifest.path'
        if pointer.exists():
            manifest = Path(pointer.read_text().strip())
        else:
            capture = capture_oracle_demonstrations(args.model, suites['train'], provenance['suites']['train']['preflight'],
                                                    root / 'capture', scenario='all', record_video=False, teacher_horizon=480, store_frames=True,
                                                    workers=args.workers)
            manifest = Path(capture.manifest); pointer.write_text(str(manifest.resolve()) + '\n')
        provenance['capture_manifest'] = str(manifest)
        provenance['capture_manifest_sha256'] = content_sha256(json.loads(manifest.read_text()))
        atomic_json(root / 'provenance.json', provenance)
        tracker.config.update(dict(capture_manifest=str(manifest), suites=provenance['suites']), allow_val_change=True)
        progress(capture_manifest=str(manifest)); assert_scene()
        if args.stop_after == 'capture':
            progress(status='complete', phase='capture_complete'); return 0

        from so_arm101_v2.learning.vision import train_vision_chunked, VisionChunkedConfig
        os.environ['SO_ARM101_V2_FRAME_CACHE'] = args.frame_store
        config = VisionChunkedConfig(seed=202, max_steps=max_steps, frame_stride=int(args.frame_stride))
        identity['training_frame_store'] = args.frame_store
        tracker.config.update(dict(training=asdict(config)), allow_val_change=True)
        progress(phase='training_initialization')
        scratch = root / 'scratch' / 'vision.pt'
        history = []
        telemetry = (root / 'training.jsonl').open('a')
        clock_path = root / 'training_clock.json'
        clock = json.loads(clock_path.read_text()) if clock_path.exists() else {}

        def on_loss(step, loss):
            now = time.time()
            if 'first_optimizer_step_at' not in clock:
                clock['first_optimizer_step_at'] = now; atomic_json(clock_path, clock)
            history.append((now, step)); history[:] = history[-20:]
            rate = (step - history[0][1]) / max(now - history[0][0], 1e-6)
            checkpoint_age = (now - scratch.stat().st_mtime) if scratch.exists() else None
            values = dict(step=step, loss=loss, timestamp=now, steps_per_second=rate,
                          first_optimizer_step_at=clock['first_optimizer_step_at'],
                          training_elapsed_s=now - clock['first_optimizer_step_at'], checkpoint_age_s=checkpoint_age)
            telemetry.write(json.dumps(values) + '\n'); telemetry.flush()
            progress(phase='training', **values)
            track({'train/batch_normalized_mse': loss, 'train/steps_per_second': rate,
                   'train/elapsed_s': values['training_elapsed_s'],
                   'train/checkpoint_age_s': checkpoint_age if checkpoint_age is not None else -1,
                   'train/tracking_ok': 0 if state.get('tracking_error') else 1}, step=step)
            if step % 100 == 0 or step == max_steps:
                print(f'TRAIN step={step} loss={loss:.8g} steps_per_second={rate:.2f}', flush=True)
        try:
            trained = train_vision_chunked(manifest, root, config=config, on_loss=on_loss,
                                           scratch_checkpoint=scratch, checkpoint_interval=checkpoint_interval)
        finally:
            telemetry.close()
        progress(phase='evaluation', checkpoint=str(trained.checkpoint)); assert_scene()
        nominal = bench_suite(bench, [(0, 0)], label='nominal_eval', repeats=3)
        summary = {}
        evaluations = [('nominal', nominal), ('heldout', suites['heldout'])]
        if args.appearance:
            evaluations.append(('heldout_appearance', appearance_product(bench, suites['heldout'], per_scenario=HELDOUT_APPEARANCE_PER_POSE,
                                                                         seed=counts['heldout'][0], label='heldout_appearance')))
        for label, suite in evaluations:
            policies = {}
            for black in (False, True):
                name = 'vision_black' if black else 'vision'
                policies[name] = PolicySpec(kind='vision_chunked', checkpoint=str(trained.checkpoint),
                                            options=(('clamp_channels', (5,)), ('black_image', black)))
            result = evaluate_closed_loop(args.model, suite, policies, root / 'evaluations' / label,
                                          environment_proven=True, record_video=True, workers=args.workers, provenance=identity)
            report = json.loads(Path(result.report_json).read_text())
            prefix = prefix_success(report, bench, bench.observation_steps)
            for name in policies:
                rows = [r for r in report['rollouts'] if r['policy_id'] == name]
                entry = dict(successes=sum(r['success'] and not r['invalidated'] for r in rows), rollouts=len(rows),
                             safety_frames=sum(sum(r[k] for k in ('clipping_frames', 'limiting_frames', 'nonfinite_frames', 'unsafe_contact_frames')) for r in rows),
                             prefix_ok=prefix.get(name, {}).get('prefix_ok'), report=str(result.report_json))
                if args.placement:
                    entry['success_by_region'] = success_by_region(rows, suite, bench)
                summary[f'{label}/{name}'] = entry
                for key, value in entry.items():
                    if key not in ('report', 'success_by_region'):
                        tracker.summary[f'{label}/{name}/{key}'] = value
            tracker.summary[f'{label}/report'] = str(result.report_json)
        if args.appearance and real_frame_episodes and any(p.exists() for p in real_frame_episodes):
            # Offline real-frame gate on the new policy: recorded, never raised (the run must complete for comparison).
            progress(phase='real_frame_check')
            from check_policy_on_real_frames import check_policy_on_real_frames
            from so_arm101_v2.simulation.vision_policy import VisionChunkedPolicy
            policy = VisionChunkedPolicy(trained.checkpoint, black_image=False, clamp_channels=(5,))
            check = check_policy_on_real_frames(policy, args.model, bench, [p for p in real_frame_episodes if p.exists()],
                                                perturb=16, regime=bench.appearance_regime)
            check.update(checkpoint=str(trained.checkpoint))
            atomic_json(root / 'real_frame_check.json', check)
            summary['real_frames/reset'] = dict(passed=check['passed'],
                                                frames={f['label']: dict(passed=f['passed'], reasons=f['reasons'],
                                                                         shoulder_units=f['dry_pass']['max_abs_delta_from_start_units']['shoulder_lift'],
                                                                         elbow_units=f['dry_pass']['max_abs_delta_from_start_units']['elbow_flex'],
                                                                         holds=f['dry_pass']['holds_in_dry_chunk'],
                                                                         photometric_pass_fraction=f.get('photometric_sweep', {}).get('pass_fraction'))
                                                        for f in check['frames']},
                                                sim_reference_passed=check['sim_reference']['check']['passed'], report=str(root / 'real_frame_check.json'))
            tracker.summary['real_frames/reset/passed'] = int(check['passed'])
            for name, frame in summary['real_frames/reset']['frames'].items():
                for key in ('shoulder_units', 'elbow_units', 'holds'):
                    tracker.summary[f'real_frames/{name}/{key}'] = frame[key]
        atomic_json(root / 'evaluation_summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
        tracker.finish(); tracker = None
        progress(status='complete', phase='complete', evaluation_summary=summary)
        return 0
    except BaseException as exc:
        progress(status='failed', error=str(exc), traceback=traceback.format_exc())
        if tracker is not None:
            tracker.finish(exit_code=1)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
