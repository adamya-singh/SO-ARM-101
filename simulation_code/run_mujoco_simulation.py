import time
import mujoco
import mujoco.viewer
#from so101_mujoco_utils import set_initial_pose, send_position_command
from so101_mujoco_utils import *

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

all_zeros_position = {
    'shoulder_pan': 0.0,   # in degrees
    'shoulder_lift': 0.0,
    'elbow_flex': 0.0,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': 0.0           # 0-100 range
}

set_initial_pose(d, starting_position) #set initial pose before starting sim viewer

with mujoco.viewer.launch_passive(m, d) as viewer:
    #close the viewer automatically after 30 wall-seconds
    start = time.time()
    hold_position(m, d, viewer, 2) #hold starting position for 2 seconds

    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        move_to_pose(m, d, viewer, all_zeros_position, 2)
        move_to_pose(m, d, viewer, starting_position, 2)

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics
        mujoco.mj_step(m, d)

        # pick up changes to the physics state, apply peturbations, update options from GUI
        viewer.sync()

        # rudimentary time keeping, will drift relative to wall clock
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
