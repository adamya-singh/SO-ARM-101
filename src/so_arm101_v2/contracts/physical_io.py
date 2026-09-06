"""Read-only hardware inspection and measured-feedback replay primitives."""
from __future__ import annotations

import time
import numpy as np

from .coordinates import JOINT_NAMES
from .physical import physical_normalized_to_act


def read_measured_act(robot, *, max_read_seconds=0.1):
    start = time.monotonic()
    positions = robot.bus.sync_read("Present_Position")
    if time.monotonic() - start > max_read_seconds:
        raise RuntimeError("stale observation: serial read exceeded deadline")
    state = np.asarray([positions[name] for name in JOINT_NAMES], dtype=np.float32)
    if not np.isfinite(state).all():
        raise RuntimeError("nonfinite measured joint position")
    return physical_normalized_to_act(state)


def assert_pinned_calibration(robot):
    """Refuse when the live calibration file differs from the pinned physical contract."""
    from .physical import load_physical_calibration
    expected = load_physical_calibration()
    for item in expected.joints:
        live = robot.calibration.get(item.name) if hasattr(robot.calibration, "get") else None
        if live is None or live.id != item.motor_id or any(
            getattr(live, key) != getattr(item, key)
            for key in ("drive_mode", "homing_offset", "range_min", "range_max")
        ):
            raise RuntimeError(f"live calibration for {item.name} differs from the pinned physical contract")


def connect_read_only(robot):
    """Open serial and validate; never configure, calibrate, or change torque."""
    robot.bus.connect()
    if not robot.bus.is_calibrated:
        robot.bus.disconnect(disable_torque=False)
        raise RuntimeError("motor calibration does not match file; refusing to write calibration")


def disconnect_read_only(robot):
    if robot.bus.is_connected:
        robot.bus.disconnect(disable_torque=False)
