# Physical Smoke Runbook (bench procedure)

Updated 2026-09-09. **A physical inference runner now exists (`tools/run_physical_episode.py`); no physical episode has been run yet and none is authorized until its preflight passes and the user confirms each motion on site.** The lens-scene policy (run `iziftplw`, nominal 3/3, held-out 30/30) is the candidate. The previous v3 trajectory examples used a different cube, location and reset. Do not use them on the new bench. Read the [bench setup/runbook](bench-pick-replace-v1.md) first.

## Current hardware and preparation

The serial adapter was `/dev/ttyACM0` (CH343, `1a86:55d3`); camera `/dev/video0` (`0c45:6366`) was returned from Windows to WSL after focusing. Device enumeration can change: inspect before connecting. Live calibration is `~/.cache/huggingface/lerobot/calibration/robots/so_follower/None.json`, pinned against `src/so_arm101_v2/data/resources/physical_inference_calibration_20260620.json`. Do not recalibrate automatically.

The arm is at gravity rest with torque off (power-cycled 2026-09-06; photographed 2026-09-08, `readme-assets/bench-rest-side-20260908.jpg`). Under the corrected joint map `measured_20260908b` the rest pose is inside the simulator's range, so the elbow-only staging is no longer required for range; the gravity-rest **shoulder reads −92.08, just below the −92 floor**, and a small reviewed shoulder lift is required before any episode. Reconnecting or ending a session does not establish current pose; use fresh read-only measurements (`tools/read_joint_reference.py`) and never assume the current pose matches the saved reset.

The elbow staging tool (`tools/prepare_physical_elbow.py`) defaults to read-only dry run and is kept for reference; it moves only the elbow. Any shoulder lift needs its own reviewed tool. Motion always requires `--enable-motion`, a new log path and on-site confirmation. The shoulder floor remains −92 at every stage.

**Camera at deployment (updated 2026-09-09):** policies trained on the lens-matched scene (`7c765d4b…` and later) expect the raw 1920×1080 MJPEG frame, **channel order flipped BGR→RGB** (OpenCV delivers BGR; the policy trained on MuJoCo's RGB renders), passed through `LensModel.real_operator()` from `src/so_arm101_v2/contracts/lens.py` (an exact area filter to 256×256; no undistortion, no crop), with the lens block taken from the scene's `bench_config.json`. In the control loop the runner computes that filter with `cv2.resize` in INTER_AREA mode, which `verify_fast_resampler` proves bit-identical on live frames at preflight (it refuses to run otherwise). Policies from the earlier pinhole scene would need undistortion and are superseded.

## Physical inference runner (`tools/run_physical_episode.py`, 2026-09-09)

One control loop (`src/so_arm101_v2/physical/runner.py`) serves two backends. The simulator backend drives a `MujocoTaskAdapter` through it and raises if the runner's gate ever differs from the adapter's; with pinned frames it reproduces `evaluate_closed_loop` row for row, and with live renders it reproduces the stored nominal rollout of run `iziftplw` exactly (`tests/test_physical_runner_parity.py`). The real backend (`physical/lerobot_backend.py`) reads joints with `read_measured_act` (100 ms staleness rule), takes the newest camera frame only at chunk boundaries (steps 0, 90, …; it must be ≤ 100 ms old and unused), gates every command with the shared bench rule (`bench_hold_decision`: any clip/limiter mask holds the current pose; 15 consecutive holds abort), sends through `robot.send_action` and aborts if the driver modifies a command, and keeps absolute 30 Hz deadlines (re-anchors after a large overrun; any step over 100 ms or three consecutive overruns aborts). Torque is never disabled; the bus is opened read-only (`robot.connect()`/`disconnect()` are never called).

Modes, in the order to use them:

```bash
# 1. No hardware: the same tool path on the simulator, same evidence layout; must succeed.
PYTHONNOUSERSITE=1 MUJOCO_GL=egl /home/win10ubuntu/miniforge3/envs/lerobot/bin/python \
  tools/run_physical_episode.py --sim-rehearsal --run-dir artifacts/so_arm101_v2/bench_pick_replace_v1/rehearsal/<new>

# 2. Hardware, read-only (torque off): pinned calibration, fresh pose, camera rate >= 25 fps, resampler proof on
#    live frames, pan-sign check (you rotate the base by hand the way the preview shows), policy dry pass.
PYTHONNOUSERSITE=1 /home/win10ubuntu/miniforge3/envs/lerobot/bin/python \
  tools/run_physical_episode.py --preflight-only --run-dir artifacts/so_arm101_v2/bench_pick_replace_v1/physical/<new>

# 3. Motion: confirmation -> torque on at the present pose -> gated approach to the recorded reset (0.5 units per
#    step, gravity lead cap 10 units, feedback stop, 45 s timeout; this is the shoulder lift above the -92 floor)
#    -> second confirmation -> the 16 s episode at 30 Hz with the camera recorded.
PYTHONNOUSERSITE=1 /home/win10ubuntu/miniforge3/envs/lerobot/bin/python \
  tools/run_physical_episode.py --enable-motion --run-dir artifacts/so_arm101_v2/bench_pick_replace_v1/physical/<new>
```

Preconditions: usbipd 5-4 (serial) and 5-3 (camera) attached, `/dev/ttyACM0` and `/dev/video0` present; cube on the square, mousepad and lighting as in the reference frames; arm at gravity rest, power on, torque off; a hand near the power switch. The approach starts with the shoulder 0.08 units below the floor, which the gate tolerates because it clips targets, not the current pose; the runner refuses to start the episode unless the shoulder is above the floor and the pose is within 0.35 ACT of the recorded reset.

Evidence per run directory: `run.json` (status, checkpoint and report hashes, scene hash, bench and lens blocks, calibration pin with the live file's hash, camera properties and measured rate, resampler proof, pan-sign result, policy dry pass, approach plan and result, confirmations with timestamps, start-pose delta, timing, holds, prefix check at step 90, file hashes), `steps.csv` (measured / policy / requested / executed / sent / returned values per step with hold reasons and latencies), `approach.csv`, `boundaries/` (raw and observation PNGs at every chunk boundary with hashes), `camera.mp4` with `camera_frames.csv`, `preflight/`. Record the outcome in `notes/vision-rung-notebook.md`.

Ctrl-C during motion stops issuing commands and leaves torque on; it is not a mechanical stop. Support the arm before removing power.

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
