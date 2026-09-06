# Physical Smoke Runbook (bench procedure)

Updated 2026-09-06. **Full-task physical replay is not ready or authorized by the current bench preparation approval.** The previous v3 trajectory examples used a different cube, location and reset. Do not use them on the new bench. Read the [bench setup/runbook](bench-pick-replace-v1.md) first.

## Current hardware and preparation

The serial adapter was `/dev/ttyACM0` (CH343, `1a86:55d3`); camera `/dev/video0` (`0c45:6366`) was returned from Windows to WSL after focusing. Device enumeration can change: inspect before connecting. Live calibration is `~/.cache/huggingface/lerobot/calibration/robots/so_follower/None.json`, pinned against `src/so_arm101_v2/data/resources/physical_inference_calibration_20260620.json`. Do not recalibrate automatically.

The user approved and completed a separate elbow-only preparation. Last stable measured shoulder/elbow: −90.8363 / 91.2613 calibrated units. **Elbow torque was left enabled to prevent gravity collapse.** Reconnecting or ending a session does not establish current pose; use fresh read-only measurements. Do not disable torque without support or assume the current pose matches the saved reset.

The elbow staging tool defaults to read-only dry run. Motion requires `--enable-motion`, a new log path and on-site confirmation; it is separate from trajectory replay. See its implementation and the bench runbook for command-floor/feedback-stop details. The shoulder floor remains −92 at every stage.

## Replay modes and required evidence

`tools/replay_physical_trajectory.py` supports offline dry run, read-only preflight and explicit motion. Its hardware paths require the active `--bench-config`; legacy simulation artifacts remain usable offline but are not approved bench trajectories.

Before physical execution, complete the bench teacher/camera gates, capture a fresh matching trajectory, finish the outstanding replay review items in the bench runbook, and obtain explicit confirmation for the particular reviewed motion. The replay claims its exclusive log and checks the live calibration against the pinned contract before it opens the bus; a failed connection removes the unused log.

For a future verified capture, substitute its actual path for `<verified-bench-manifest>`:

```bash
# Offline only: no hardware connection or motor commands.
PYTHONNOUSERSITE=1 /home/win10ubuntu/miniforge3/envs/lerobot/bin/python \
  tools/replay_physical_trajectory.py \
  --capture-manifest <verified-bench-manifest> --scenario nominal \
  --bench-config simulation_code/model/bench_pick_replace_v1/bench_config.json

# Read-only bus connection, calibration/read checks and first-target gating.
PYTHONNOUSERSITE=1 /home/win10ubuntu/miniforge3/envs/lerobot/bin/python \
  tools/replay_physical_trajectory.py \
  --capture-manifest <verified-bench-manifest> --scenario nominal \
  --bench-config simulation_code/model/bench_pick_replace_v1/bench_config.json \
  --robot-port /dev/ttyACM0 --preflight-only
```

These are templates, not commands with an existing approved bench manifest. Preflight **requires hardware**; offline dry run does not. Read-only connection bypasses motor configuration and does not change torque or calibration. A refused first command means stop and resolve the mismatch; never skip it or widen the limits.

## Motion behavior and exit semantics

Motion requires `--enable-motion`, an exclusive `--log <new.csv>` and explicit interactive confirmation. Each step uses fresh finite feedback, the configured relative limit, the shoulder floor including integer tick conversion, and all existing bounds. A refusal, stale observation or communication failure aborts the trajectory; it does not skip ahead. Logs distinguish measured, requested and actually sent positions.

Ctrl-C stops issuing subsequent trajectory commands and closes serial while preserving torque. **It does not command a mechanical stop or remove motor power:** the servo may continue toward its last goal. Support the arm before removing power. This replaces earlier inaccurate descriptions of disconnect/hold behavior.

Future physical-test evidence should retain the command CSV, real video, calibration/config/trajectory identities, safety events and observations of lag, slip or other sim-to-real differences. Record results in `notes/vision-rung-notebook.md`; do not infer physical readiness from training loss or old v3 simulation success.
