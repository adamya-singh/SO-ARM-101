# ACT Coordinate Contract

## Root cause

The physical dataset does not contain MuJoCo mechanical radians. LeRobot reads
each calibrated body motor in `[-100, 100]`, then
`record_single_arm.py` stores `motor_value / 100 * pi`. The gripper is read in
`[0, 100]` and stored as `motor_value / 100 * 1.7`. These are calibrated
motor-range encodings shaped like radians, not angles that can be sent directly
to MuJoCo.

The old simulation path treated those values as mechanical radians and clipped
them to the MJCF limits. For example, the lead-3 checkpoint's wrist-flex mean
near `2.849` was interpreted as 2.849 mechanical radians and clipped to the
MuJoCo upper limit of `1.6581`.

This is not a degree/radian arithmetic error. It is a coordinate-frame and
scale mismatch.

## Current physical calibration

Physical ACT inference on this machine resolves:

`~/.cache/huggingface/lerobot/calibration/robots/so_follower/None.json`

The file was modified on 2026-06-20 and is the authoritative current deployment
calibration. An immutable copy is stored beside the dataset metadata at
`imitation-learning/datasets/so101_pickplace_v1/meta/physical_inference_calibration_20260620.json`.
Its source SHA-256 is
`00c6940f682c47448c484ee727117d28439744cc4fbc93a60f9b492ce2892ca8`.

The file confirms the expected six-joint order, `drive_mode=0` for every motor,
body-joint normalization to `[-100, 100]`, and gripper normalization to
`[0, 100]`. Raw servo tick endpoints are used by LeRobot before dataset
encoding; they cancel from the endpoint-to-endpoint simulation transform.

The full machine-readable contract is
`imitation-learning/datasets/so101_pickplace_v1/meta/act_coordinate_contract.json`.

## Simulation adapter

`simulation_code/act_coordinate_utils.py` performs one affine transform per
joint:

```text
MuJoCo state: mechanical MJCF low..high -> ACT dataset low..high
ACT action:   ACT dataset low..high     -> mechanical MJCF low..high
```

For the five body joints, ACT dataset endpoints are `[-pi, pi]`. For the
gripper they are `[0, 1.7]`. The transform is applied before checkpoint state
normalization and after checkpoint action denormalization.

The same implementation is used by supervised ACT simulation inference, ACT
PPO training, PPO inference, and the throughput benchmark. PPO uses the
differentiable PyTorch transform and checkpoint normalization statistics.

## Controlled lead-3 comparison

Measured on 2026-07-10 with the lead-3 `026020` checkpoint, fixed block,
seed `2606`, deterministic actions, three 150-step episodes, and the current
camera/wrist geometry:

- Legacy direct-coordinate path: 435 of 450 steps (96.7%) clipped at least one
  joint; mean return `-54.909`; mean final distance `0.1823 m`; `0/3` success.
- Correct affine adapter: 0 of 450 steps clipped; mean return `-24.906`; mean
  final distance `0.1452 m`; `0/3` success.

The adapter removes the systematic clipping and materially improves return and
final distance, but it does not solve pickup by itself. The remaining failure
can still include visual distribution shift, policy quality, and residual
simulation-to-real mismatch. The earlier upright-hover screenshot remains
useful evidence of the broken legacy path, not the corrected baseline.

## Verification

Run:

```bash
cd /home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code
python -m unittest discover -s tests -p 'test_*.py' -v
```

The tests cover endpoint and midpoint mapping, batched round trips, current
physical-calibration endpoints, representative wrist-flex saturation,
differentiability, and calibration-file integrity.
