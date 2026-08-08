# Physical Smoke Runbook (bench procedure)

Goal: the earliest end-to-end touch of the real SO-101 arm — replay a
recorded trajectory through the v2 safety contract at 30 Hz. This is a smoke
test, not a promotion: success = the arm tracks the trajectory with zero
safety-gate refusals, and the run is logged for the eventual sim-to-real
comparison.

Tool: `tools/replay_physical_trajectory.py`. Dry-run is the default; motion
requires `--enable-motion` **and** a bare-Enter confirmation at the prompt.
Every command is checked with `evaluate_physical_command` before sending —
a fired mask HOLDS the arm (logs the reason) instead of sending. The sim and
servo coordinate systems are formally the same (pinned by
`tests/test_calibration_integrity.py`), so no unit conversion beyond the
contract's own `act_to_physical_normalized` is involved.

## 0. Attach the hardware (WSL2)

In Windows PowerShell (admin), with the arm and camera plugged in:

```
usbipd list
usbipd bind --busid <SERIAL_BUSID>     # SO-101 serial adapter: CH343, VID:PID 1A86:55D3
usbipd attach --wsl --busid <SERIAL_BUSID>
```

(The wrist camera is `0C45:6366` — not needed for replay.) Then in WSL,
verify `/dev/ttyUSB0` exists and is readable (`sudo usermod -aG dialout` +
re-login if not).

## 1. Verify calibration

The live LeRobot calibration must exist:
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/None.json`
(semantically identical to the pinned contract copy
`src/so_arm101_v2/data/resources/physical_inference_calibration_20260620.json`
— verified 2026-08-06). If missing, run the LeRobot calibration flow first.

## 2. Dry run (no hardware contact)

```bash
PYTHONNOUSERSITE=1 /home/win10ubuntu/miniforge3/envs/lerobot/bin/python \
  tools/replay_physical_trajectory.py \
  --capture-manifest artifacts/so_arm101_v2/oracle_distillation/oracle/fixed_pick_place_v3/9f54b9e66c884855/manifest.json \
  --scenario nominal
```

Expected: `offline gating: all steps clean` then `DRY RUN complete`. Every
step of the trajectory is validated against the safety contract offline
first — a refusal here means do not proceed.

## 3. Preflight (connects, reads one observation, no motion)

```bash
PYTHONNOUSERSITE=1 /home/win10ubuntu/miniforge3/envs/lerobot/bin/python \
  tools/replay_physical_trajectory.py \
  --capture-manifest artifacts/so_arm101_v2/oracle_distillation/oracle/fixed_pick_place_v3/9f54b9e66c884855/manifest.json \
  --scenario nominal --preflight-only
```

Check the printed current pose is sane (all six values finite, inside
[-π, π] / [0, 1.7]).

## 4. Stage the bench

- Clear the workspace. First run: **no cube** — replay in the air.
- Move the arm by hand near the trajectory's start pose (the tool refuses if
  the start differs by > 0.35 ACT ≈ 20°; the sim start pose is the v3
  nominal `robot_qpos_mujoco`).
- Know your abort: **Ctrl-C** stops the loop and disconnects cleanly; the
  servos hold their last position. Keep a hand near the power switch for the
  first run.

## 5. Motion run

```bash
PYTHONNOUSERSITE=1 /home/win10ubuntu/miniforge3/envs/lerobot/bin/python \
  tools/replay_physical_trajectory.py \
  --capture-manifest artifacts/so_arm101_v2/oracle_distillation/oracle/fixed_pick_place_v3/9f54b9e66c884855/manifest.json \
  --scenario nominal --enable-motion \
  --log imitation-learning/outputs/replay_smoke_$(date +%Y%m%d_%H%M%S).csv
```

Confirm with a bare Enter. The 480-step replay takes 16 s.

Success criteria: the arm executes the full pick-place motion (reach, close,
lift, carry, set down, open, retreat, hold), **zero HOLD lines** in the
output, and the log CSV shows no `hold_reason` entries.

Second run: place a 25 mm cube at the sim-nominal position (x=0, y=0.30 m
from the base, on the table plane) and repeat — a physical grasp on replay
would be a genuine (if lucky) first physical pick.

## 6. Record for sim-to-real

Keep: the log CSV, a phone video, and notes on visible divergence (servo
lag, oscillation, gripper slip). These are the inputs to the eventual
sim-to-real gap analysis; file them in `imitation-learning/outputs/` and add
an entry to `notes/vision-rung-notebook.md`.

## Not attached?

`--preflight-only` and the dry run work without hardware (dry run needs no
device at all). The tool refuses cleanly if `/dev/ttyUSB0` is absent.
