import time
import mujoco

def convert_to_dictionary(qpos):
    return {
            'shoulder_pan': qpos[0]*180.0/3.12159,  # convert radians(mujoco) to degrees(SO101)
            'shoulder_lift': qpos[1]*180.0/3.12159,
            'elbow_flex': qpos[2]*180.0/3.12159,
            'wrist_flex': qpos[3]*180.0/3.12159,
            'wrist_roll': qpos[4]*180.0/3.12159,
            'gripper': qpos[5]*100/3.12159,
        }

def convert_to_list(dictionary):
    return [
            dictionary['shoulder_pan']*3.14159/180.0,   # convert degrees(SO101) to radians(mujoco)
            dictionary['shoulder_lift']*3.14159/180.0,
            dictionary['elbow_flex']*3.14159/180.0,
            dictionary['wrist_flex']*3.14159/180.0,
            dictionary['wrist_roll']*3.14159/180.0,
            dictionary['gripper']*3.14159/100.0,
        ]

def set_initial_pose(d, position_dict):
    pos = convert_to_list(position_dict)
    d.qpos = pos

def send_position_command(d, position_dict):
    pos = convert_to_list(position_dict)
    d.ctrl = pos
