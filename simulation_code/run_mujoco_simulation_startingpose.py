import os
import time


def _configure_wsl_viewer_gl():
    """Avoid WSLg MIT-SHM / BadDrawable crashes in the MuJoCo GLFW viewer."""
    try:
        with open("/proc/version", encoding="utf-8") as f:
            is_wsl = "microsoft" in f.read().lower()
    except OSError:
        is_wsl = False

    if not is_wsl:
        return

    # Prefer X11 over Wayland; WSLg's Wayland path often breaks GLFW shared-memory blits.
    os.environ.pop("WAYLAND_DISPLAY", None)
    # Use Mesa's D3D12 backend (WSLg GPU) instead of software GL.
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
    os.environ.setdefault("MUJOCO_GL", "glfw")
    # Drop the slow software fallback if a previous run set it.
    os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)


_configure_wsl_viewer_gl()

import mujoco
import mujoco.viewer
from so101_mujoco_utils import set_initial_pose, send_position_command

m = mujoco.MjModel.from_xml_path('model/scene.xml')
d = mujoco.MjData(m)


starting_position = {
	'shoulder_pan': 0.06, #degrees
        'shoulder_lift': -100.21,
        'elbow_flex': 89.95,
        'wrist_flex': 66.46,
        'wrist_roll': -84.04,  # 90 degrees right from the original 5.96-degree pose
        'gripper': 1.0,  #0-100 range for open and closed
}

set_initial_pose(d, starting_position)

with mujoco.viewer.launch_passive(m, d) as viewer:
    #close the viewer automatically after 30 wall-seconds
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        send_position_command(d, starting_position)

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics
        mujoco.mj_step(m, d)

        # pick up changes to the physics state, apply peturbations, update options from GUI
        viewer.sync()

        # rudimentary time keeping, will drift relative to wall clock
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
