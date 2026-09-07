"""Health check for a running bench pipeline experiment (read-only).

Reads progress.json, training.jsonl, training_clock.json and wandb.json from
the experiment output directory and reports: process alive, phase, step
advance since the previous check, finite loss and trend, throughput against
the same run's post-warmup baseline, GPU memory, scratch-checkpoint age, W&B
sync state and estimated remaining time. Exit status 0 = healthy, 1 = warning
(stalled or degraded), 2 = failed/complete/required action. It never changes
the recipe or the run.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def gpu_memory() -> str:
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout.strip()
        used, total, util = [v.strip() for v in out.split(',')]
        return f'{used}/{total} MiB, {util}% util'
    except Exception as exc:  # nvidia-smi missing or busy
        return f'unavailable ({exc.__class__.__name__})'


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir', type=Path, required=True)
    p.add_argument('--state', type=Path, default=None, help='where to remember the previous check (default: <experiment>/health_state.json)')
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--stall-seconds', type=float, default=900)
    args = p.parse_args(argv)
    root = args.experiment_dir
    state_path = args.state or root / 'health_state.json'
    now = time.time()
    progress = json.loads((root / 'progress.json').read_text()) if (root / 'progress.json').exists() else {}
    clock = json.loads((root / 'training_clock.json').read_text()) if (root / 'training_clock.json').exists() else {}
    wandb_info = json.loads((root / 'wandb.json').read_text()) if (root / 'wandb.json').exists() else {}
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    rows = []
    if (root / 'training.jsonl').exists():
        with (root / 'training.jsonl').open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    lines = []
    status = 0
    phase = progress.get('phase'); run_status = progress.get('status')
    lines.append(f"status={run_status} phase={phase} pid={progress.get('pid')} alive={alive(progress.get('pid'))}")
    lines.append(f"experiment started {(now - progress['started_at'])/60:.1f} min ago; progress.json updated {(now - progress.get('updated_at', now)):.0f} s ago"
                 if progress.get('started_at') else 'no progress.json yet')
    if run_status == 'failed':
        lines.append(f"FAILED: {progress.get('error')}"); status = 2
    elif run_status == 'complete':
        lines.append('COMPLETE'); status = 2
    elif not alive(progress.get('pid')):
        lines.append('process NOT alive but status not final: required action'); status = 2
    if rows:
        last = rows[-1]; step = last['step']; loss = last['loss']
        first_at = clock.get('first_optimizer_step_at')
        lines.append(f"training: step {step} (of {progress.get('max_steps', '?')}), loss {loss:.4g}, first optimizer step {(now - first_at)/60:.1f} min ago" if first_at else f"training: step {step}, loss {loss:.4g}")
        if not math.isfinite(loss):
            lines.append('loss NONFINITE'); status = 2
        recent = [r['loss'] for r in rows[-50:]]
        earlier = [r['loss'] for r in rows[-500:-450]] if len(rows) >= 500 else []
        if earlier:
            lines.append(f"loss trend: mean(last 50) {sum(recent)/len(recent):.4g} vs mean(450-500 steps ago) {sum(earlier)/len(earlier):.4g}")
        post = [r['steps_per_second'] for r in rows if r['step'] > args.warmup_steps]
        if post:
            baseline = sorted(post)[len(post)//2]; current = last['steps_per_second']
            lines.append(f"throughput: {current:.2f} steps/s now vs post-warmup median {baseline:.2f}")
            if current < 0.5 * baseline:
                lines.append('throughput DEGRADED below half of baseline'); status = max(status, 1)
            remaining = max(0, int(progress.get('max_steps', 0)) - step) if progress.get('max_steps') else None
            if remaining is not None and current > 0:
                lines.append(f"estimated remaining: {remaining / current / 3600:.2f} h")
        if previous.get('step') is not None and previous['step'] == step and now - previous['checked_at'] > args.stall_seconds:
            lines.append(f"STALLED: step unchanged since previous check {(now - previous['checked_at'])/60:.0f} min ago"); status = max(status, 1)
        scratch = root / 'scratch' / 'vision.pt'
        lines.append(f"scratch checkpoint age: {(now - scratch.stat().st_mtime)/60:.1f} min" if scratch.exists() else 'scratch checkpoint: none yet')
    lines.append(f"gpu: {gpu_memory()}")
    lines.append(f"wandb: url={wandb_info.get('url')} tracking_error={progress.get('tracking_error')} last_ok={progress.get('tracking_last_ok')}")
    state_path.write_text(json.dumps(dict(checked_at=now, step=rows[-1]['step'] if rows else None, status=status)))
    print('\n'.join(lines))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
