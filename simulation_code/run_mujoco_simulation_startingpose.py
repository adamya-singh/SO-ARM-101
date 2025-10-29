import time
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
        'wrist_roll': 5.96,
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
