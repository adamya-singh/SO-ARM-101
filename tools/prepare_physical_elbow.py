"""Supervised elbow-only staging from gravity rest into the validated sim range.

Dry-run by default. No other motor is enabled or commanded. The sole
out-of-sim-range exception is a decreasing elbow target between the measured
gravity-rest position and the 85-unit command floor, stopping once the measured
elbow is at or below 92 units; it is never used by replay.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import time
import numpy as np
from so_arm101_v2.contracts.physical import act_to_physical_normalized, physical_normalized_to_act, evaluate_physical_command
from so_arm101_v2.contracts.coordinates import act_to_mujoco_qpos
from so_arm101_v2.contracts.bench import BenchConfig
from so_arm101_v2.contracts.physical_io import assert_pinned_calibration, connect_read_only, disconnect_read_only, read_measured_act

ELBOW_TARGET = 85.0  # command floor; stop once the MEASURED elbow is <=92
MAX_STEP = 0.1


def next_elbow_command(current_act, initial_elbow, previous_target=None):
    physical=act_to_physical_normalized(current_act)
    if not np.isfinite(physical).all() or physical[1] < -92:
        raise RuntimeError('shoulder floor or nonfinite observation; stop preparation')
    if not ELBOW_TARGET <= initial_elbow <= 100:
        raise RuntimeError('elbow start is outside the reviewed preparation interval')
    if physical[2] > initial_elbow + 0.5 or physical[2] < ELBOW_TARGET - 1:
        raise RuntimeError('unexpected elbow motion')
    target=physical.copy()
    prior=float(physical[2]) if previous_target is None else previous_target
    target[2]=max(ELBOW_TARGET, float(physical[2])-10.0, prior-MAX_STEP)
    evaluation=evaluate_physical_command(current_act, physical_normalized_to_act(target),
        shoulder_floor=-92, max_relative_target=10.01)
    if evaluation.act_clip_mask.any() or evaluation.physical_clip_mask.any() or evaluation.relative_limit_mask.any():
        raise RuntimeError('preparation command violates physical safety contract')
    mask=evaluation.mujoco_clip_mask.copy()
    mask[2]=False  # Only the elbow may traverse inward from its observed rest.
    if mask.any():
        raise RuntimeError('a non-elbow joint is outside the simulator bounds')
    if target[2] > physical[2]:
        raise RuntimeError('preparation must never increase elbow flexion')
    return float(target[2])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',default='/dev/ttyACM0')
    p.add_argument('--enable-motion',action='store_true')
    p.add_argument('--log',type=Path)
    a=p.parse_args(argv)
    if a.enable_motion and a.log is None:p.error('motion requires --log')
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    robot=SO101Follower(SO101FollowerConfig(id='None',port=a.port,cameras={},use_degrees=False))
    assert_pinned_calibration(robot)
    handle=None
    try:
        connect_read_only(robot)
        current=read_measured_act(robot)
        physical=act_to_physical_normalized(current)
        initial=float(physical[2])
        if 85 <= initial <= 92:
            BenchConfig().validate_qpos(act_to_mujoco_qpos(current))
            print('elbow already prepared')
            return 0
        next_elbow_command(current,initial)
        print(f'Elbow-only preparation: measured {initial:.3f} -> <=92; command floor {ELBOW_TARGET}; shoulder {physical[1]:.3f}; floor -92',flush=True)
        if not a.enable_motion:
            print('DRY RUN: no torque changes or motor commands')
            return 0
        if input('Support the arm, keep the workspace clear. Press Enter to prepare only the elbow; anything else aborts: ').strip():
            return 0
        a.log.parent.mkdir(parents=True,exist_ok=True)
        handle=a.log.open('x',newline='')
        writer=csv.writer(handle);writer.writerow(['monotonic','measured_elbow','target_elbow','shoulder'])
        current=read_measured_act(robot)
        initial=float(act_to_physical_normalized(current)[2])
        next_elbow_command(current,initial)
        holding = robot.bus.read('Torque_Enable','elbow_flex',normalize=False)
        prior = float(robot.bus.read('Goal_Position','elbow_flex')) if holding else initial
        prior = max(ELBOW_TARGET,min(initial,prior))
        robot.bus.write('Goal_Position','elbow_flex',prior)
        robot.bus.enable_torque('elbow_flex')
        previous_target=prior
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            start=time.monotonic()
            current=read_measured_act(robot)
            physical=act_to_physical_normalized(current)
            if ELBOW_TARGET - 0.5 <= float(physical[2]) <= 92.0:
                BenchConfig().validate_qpos(act_to_mujoco_qpos(current))
                print('Elbow prepared; torque remains enabled to prevent gravity collapse. Capture the reset now.')
                return 0
            target=next_elbow_command(current,initial,previous_target)
            robot.bus.write('Goal_Position','elbow_flex',target)
            previous_target=target
            writer.writerow([start,float(physical[2]),target,float(physical[1])]);handle.flush()
            time.sleep(max(0,1/30-(time.monotonic()-start)))
        raise RuntimeError('elbow preparation timed out; no further commands sent')
    finally:
        if handle:handle.close()
        disconnect_read_only(robot)

if __name__ == '__main__':raise SystemExit(main())
