# Physical Smoke Runbook (bench procedure)

Updated 2026-09-10. **Two physical attempts were run on 2026-09-09 (`physical/episode_01`, `episode_02`; see `notes/vision-rung-notebook.md`): the runner, timing, gate and safety stack worked; the lens policy (run `iziftplw`) retracted on the real reset frame and is not a candidate for another trial.** The next candidate is the policy from the appearance-randomized run (gate 6 in the bench runbook), and only if it passes the offline real-frame gate. The preflight now records `Present_Voltage` (the stock 5 V adapter reads 5.3–5.4 V; refusal only below 4.8 V), and the runner refuses the episode if the real-frame gate fails on a fresh frame at the reset pose. The previous v3 trajectory examples used a different cube, location and reset. Do not use them on the new bench. Read the [bench setup/runbook](bench-pick-replace-v1.md) first.

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

# 2. Hardware, read-only (torque off): pinned calibration, fresh pose, servo voltage >= 4.8 V (stock 5 V adapter; refuses a brown-out),
#    camera rate >= 25 fps, resampler proof on live frames, optional pan-sign check (--pan-check; verified 2026-09-09),
#    policy dry pass (informational at gravity rest; the real-frame gate applies here only if already at the reset).
PYTHONNOUSERSITE=1 /home/win10ubuntu/miniforge3/envs/lerobot/bin/python \
  tools/run_physical_episode.py --preflight-only --run-dir artifacts/so_arm101_v2/bench_pick_replace_v1/physical/<new>

# 3. Motion: confirmation -> torque on at the present pose -> gated approach to the recorded reset (0.5 units per
#    step, gravity lead cap 10 units, feedback stop, 45 s timeout; this is the shoulder lift above the -92 floor)
#    -> real-frame gate on a fresh frame at the reset (hold-like chunk: shoulder/elbow <= 3 units, no holds; refuses the
#       episode otherwise, torque kept) -> second confirmation -> the 16 s episode at 30 Hz with the camera recorded.
#    Add --yes to auto-confirm both prompts (bench owner's standing authorization of 2026-09-10, recorded in run.json).
PYTHONNOUSERSITE=1 /home/win10ubuntu/miniforge3/envs/lerobot/bin/python \
  tools/run_physical_episode.py --enable-motion --run-dir artifacts/so_arm101_v2/bench_pick_replace_v1/physical/<new>
```

**Servo supply voltage (measured 2026-09-09, gated 2026-09-10):** with torque on, every servo reported `Present_Voltage` 5.3–5.4 V. That is the stock SO-101 5 V adapter under load, which this arm has always run on (user, 2026-09-10); it is not a fault. The gripper's input-voltage flag and the elbow's ~4-unit gravity sag seen on 2026-09-09 are observations to keep in mind (the approach ramp now overshoots up to 6 units to cover the sag). The preflight reads every motor's `Present_Voltage` (raw 0.1 V units) right after the read-only connect, records it in `run.json` (`servo_voltage`) and refuses only a brown-out below 4.8 V (`MIN_SERVO_VOLTAGE_V`).

**Real-frame gate (2026-09-10):** `so_arm101_v2.physical.dry_pass.check_reset_frame` runs the network once on a frame at the reset pose and walks the first chunk through the bench gate; it passes only if shoulder_lift and elbow_flex move at most 3 units, no command is held and the shoulder never goes below the floor. Offline: `tools/check_policy_on_real_frames.py` on `physical/episode_02_20260909` (the lens policy fails it: shoulder 25 units, elbow 21, 23 holds; its simulated reset chunk passes at 0.47 / 0.45). Live: the runner grabs a fresh frame after the approach and refuses the episode on failure (`run.json` `real_frame_check`, `preflight/real_frame_gate_reset.png`).

**Relative limit on real servos (2026-09-10, `physical/episode_09`):** the runner's gate is `bench_clip_decision`: on a relative-limit-only mask it sends the rate-limited target (20 units per step) and continues; range and floor clips still hold and count toward the 15-hold abort. `run_episode(rate_limit_continues=False)` restores the hold-on-any-mask rule.

**Cube placement (found 2026-09-10, `physical/episode_04`):** the task pose is the square's **centre** 8.5 in (215.9 mm) forward of the base front edge, cube centred on it. On 2026-09-10 the towel's near edge was at that mark, putting the cube ~40 mm too far and ~11 mm to the side; the policy (trained on ±10 mm) executed the whole plan and closed on nothing. Before a trial, check the reset frame: the white square should sit with its bottom edge about 45 % down the 256-px observation (rows ~46–112 in the simulated reference `inspection/sim_reset_observation_fcead5c7.png`), not touching the top edge.

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
