# Teleop Gripper / Keyboard Notes

Date: 2026-01-24

## What we implemented
- Updated `imitation-learning/record_single_arm.py` to support gripper stepping by keyboard:
  - Press `1` to open the gripper by 25% of its normalized range.
  - Press `2` to close the gripper by 25%.
- Kept arm joints limp by disabling torque for all motors, then enabling torque only on the gripper.
- Initialized gripper target from `Present_Position` (normalized 0-100).
- Writes use `bus.write("Goal_Position", "gripper", target)`.
- Added controls text for `1`/`2` in the UI.

## Relevant lerobot APIs (local fork)
- Per-motor torque control exists:
  - `FeetechMotorsBus.disable_torque(motors=...)`
  - `FeetechMotorsBus.enable_torque(motors=...)`
- Single-motor write:
  - `MotorsBus.write("Goal_Position", motor_name, value, normalize=True)`
- Gripper norm range is 0-100:
  - SO-101 follower uses `MotorNormMode.RANGE_0_100` for gripper.
- Present position read:
  - `bus.read("Present_Position", "gripper")` returns normalized value by default.

## Issue observed
- Gripper briefly responds to `1` then stops responding. Needs further debugging.

## Teleop scripts copied in for testing
From XLeRobot (SO100/SO101 keyboard control examples):
- `imitation-learning/0_so100_keyboard_joint_control.py`
- `imitation-learning/1_so100_keyboard_ee_control.py`

## Other notes
- Generic `lerobot_teleoperate.py` exists but `teleop.type=keyboard` only returns pressed keys and does not map to SO-101 joint actions by default.

