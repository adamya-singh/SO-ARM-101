# Physical Smoke Runbook (bench procedure)

Updated 2026-09-08. **Full-task physical replay is not ready or authorized.** The bench sim run is in progress (see the bench runbook status); no learned policy exists yet. The previous v3 trajectory examples used a different cube, location and reset. Do not use them on the new bench. Read the [bench setup/runbook](bench-pick-replace-v1.md) first.

## Current hardware and preparation

The serial adapter was `/dev/ttyACM0` (CH343, `1a86:55d3`); camera `/dev/video0` (`0c45:6366`) was returned from Windows to WSL after focusing. Device enumeration can change: inspect before connecting. Live calibration is `~/.cache/huggingface/lerobot/calibration/robots/so_follower/None.json`, pinned against `src/so_arm101_v2/data/resources/physical_inference_calibration_20260620.json`. Do not recalibrate automatically.

The arm is at gravity rest with torque off (power-cycled 2026-09-06; photographed 2026-09-08, `readme-assets/bench-rest-side-20260908.jpg`). Under the corrected joint map `measured_20260908b` the rest pose is inside the simulator's range, so the elbow-only staging is no longer required for range; the gravity-rest **shoulder reads −92.08, just below the −92 floor**, and a small reviewed shoulder lift is required before any episode. Reconnecting or ending a session does not establish current pose; use fresh read-only measurements (`tools/read_joint_reference.py`) and never assume the current pose matches the saved reset.

The elbow staging tool (`tools/prepare_physical_elbow.py`) defaults to read-only dry run and is kept for reference; it moves only the elbow. Any shoulder lift needs its own reviewed tool. Motion always requires `--enable-motion`, a new log path and on-site confirmation. The shoulder floor remains −92 at every stage.

**Camera at deployment:** the simulator trained on a pinhole wrist camera (fovy 44.0°, calibrated 2026-09-08). Real frames must be undistorted with `artifacts/so_arm101_v2/bench_pick_replace_v1/camera_calibration/camera_intrinsics.json` (`selected`) and resized exactly as the capture pipeline does before they reach a learned policy. This is not yet implemented in the replay/inference tools.

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
