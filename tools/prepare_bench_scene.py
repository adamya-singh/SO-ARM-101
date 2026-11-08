"""Build the versioned 20 mm XYZ bench scene without modifying legacy XML."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from so_arm101_v2.contracts.appearance import AppearanceRegime
from so_arm101_v2.contracts.bench import BenchConfig
from so_arm101_v2.contracts.lens import LensModel
from so_arm101_v2.contracts.placement import PlacementRegime

# Raised survey pose for the observation prefix (2026-09-10): joint deltas from the reset pose, MuJoCo radians.
# Found by a read-only search: sees 119/121 sample points of the teacher-reachable rectangle with a 20 px margin.
SURVEY_POSE_DELTA_RAD = (0.0, 0.40, -0.60, 0.40, 0.0, 0.0)

ROOT = Path(__file__).resolve().parents[1]
# Names shared with so_arm101_v2.simulation.appearance (the applier resolves them by name).
APPEARANCE_TEXTURE = 'ground_speckle'
APPEARANCE_TEXTURE_SIZE = 256
APPEARANCE_TOWEL_BODY = 'towel_visual'
APPEARANCE_TOWEL_GEOM = 'towel_visual_geom'
APPEARANCE_TOWEL_HALF_THICKNESS_M = 0.0002

SERVO_CLASSES=('Rotation','Pitch','Elbow','Wrist_Pitch','Wrist_Roll','Jaw')   # joint order of JOINT_NAMES
JOINT_ORDER=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
# The bench's actuator and joint settings (Menagerie trs_so_arm100 defaults), applied to the SO-101 with --dynamics bench
# so that only the arm's geometry and kinematics change (handoff step 3, 2026-11-06).
BENCH_DYNAMICS=dict(joint=dict(damping='0',frictionloss='0.1',armature='0.1'),
                    position=dict(kp='50',dampratio='1',forcerange='-3.5 3.5'),
                    option=dict(cone='elliptic',impratio='10'))


def _so101_arm(source: Path, dynamics: str):
    """The vendored Menagerie SO-101 (tools/vendor_menagerie_so101.py) with the chosen dynamics."""
    arm = ET.parse(source / 'so101_bench.xml').getroot()
    if dynamics == 'bench':
        option = arm.find('option')
        for key in list(option.attrib):
            del option.attrib[key]
        option.attrib.update(BENCH_DYNAMICS['option'])
        sts = arm.find(".//default[@class='sts3215']")
        sts.find('joint').attrib.clear(); sts.find('joint').attrib.update(BENCH_DYNAMICS['joint'])
        sts.find('position').attrib.clear(); sts.find('position').attrib.update(BENCH_DYNAMICS['position'])
    elif dynamics != 'menagerie':
        raise ValueError(f'unknown dynamics {dynamics!r}')
    return arm


def prepare(destination: Path, config: BenchConfig, *, arm_model: str = 'so100', dynamics: str = 'bench'):
    import os
    destination.mkdir(parents=True, exist_ok=True)
    if arm_model == 'so100':
        if dynamics != 'bench':
            raise ValueError('the SO-100 scene has only the bench dynamics')
        source = ROOT / 'simulation_code/model/menagerie_so_arm100'
        mesh_root = source / 'assets'
        # Include-expanded XML retains the original body's frames, inertia,
        # collision decomposition, and camera calibration.
        arm = ET.parse(source / 'so_arm101_v2.xml').getroot()
    elif arm_model == 'so101':
        source = ROOT / 'simulation_code/model/menagerie_so101'
        mesh_root = ROOT / 'simulation_code/model/assets'
        arm = _so101_arm(source, dynamics)
    else:
        raise ValueError(f'unknown arm {arm_model!r}')
    arm.set('model', 'bench_pick_replace_v1')
    arm.find('compiler').set('meshdir', os.path.relpath(mesh_root, destination))
    source = ROOT / 'simulation_code/model/menagerie_so_arm100'   # the world (scene_v2.xml) is shared by both arms
    baseline = ET.parse(source / 'scene_v2.xml').getroot()
    arm.append(baseline.find('visual'))
    assets = arm.find('asset')
    assets.find("material[@name='white']").set('rgba','0.07 0.07 0.07 1')
    for element in baseline.find('asset'):
        assets.append(element)
    # The 3d-printing folder sat beside the repo until the 2026-11 move into ~/projects put it inside the repo.
    cube_relative = '3d-printing/oak-d-lite-mount/things-I-used/XYZ 20mm Calibration Cube - 1278865/files/xyzCalibration_cube.stl'
    cube_source = next((base / cube_relative for base in (ROOT.parent, ROOT) if (base / cube_relative).is_file()), ROOT.parent / cube_relative)
    cube_dest = destination / 'xyzCalibration_cube.stl'
    if cube_source.is_file():
        shutil.copyfile(cube_source, cube_dest)
    elif not cube_dest.is_file():
        raise FileNotFoundError(f'XYZ cube STL not found at {cube_source} and no copy exists in {destination}')
    ET.SubElement(assets, 'mesh', name='xyz_cube_visual',
        file=os.path.relpath(destination / 'xyzCalibration_cube.stl', mesh_root),
        scale='0.001 0.001 0.001')
    ET.SubElement(assets, 'material', name='black_pla', rgba='0.025 0.025 0.025 1',
        specular='0', shininess='0')
    # Appearance slot 1 (always present so the scene hash does not depend on whether a regime is on):
    # an UNBOUND flat texture the adapter fills with procedural speckle and binds to the ground
    # material at reset (texture binding is baked into a render context, so the adapter recreates
    # its renderers on an appearance transition). Unbound, it is invisible.
    ET.SubElement(assets, 'texture', name=APPEARANCE_TEXTURE, type='2d', builtin='flat', rgb1='1 1 1',
        width=str(APPEARANCE_TEXTURE_SIZE), height=str(APPEARANCE_TEXTURE_SIZE))
    world = arm.find('worldbody')
    for element in baseline.find('worldbody'):
        world.append(element)
    square=world.find("geom[@name='napkin']")
    square.set('pos', f'{config.square_center_xy[0]} {config.square_center_xy[1]} {config.square_thickness_m/2}')
    square.set('rgba','1 1 1 1')
    body=world.find("body[@name='red_block']")
    body.set('pos',' '.join(map(str,config.cube_center)))
    geom=body.find("geom[@name='red_block_geom']")
    geom.set('size','0.01 0.01 0.01')
    geom.set('rgba','0.025 0.025 0.025 0')
    ET.SubElement(body,'geom',name='xyz_visual',type='mesh',mesh='xyz_cube_visual',
        pos='-0.01 -0.01 -0.01',material='black_pla',contype='0',conaffinity='0',density='0',group='2')
    # Appearance slot 2: a visual-only 'towel' body (the real folded paper towel is larger than the
    # square) appended LAST so no existing body/geom id shifts. Invisible (alpha 0) until an
    # appearance draw sizes, yaws and colours it; its top sits 0.5 mm inside the napkin slab so the
    # napkin always wins the depth test. contype/conaffinity 0 and density 0: no contacts, no mass.
    towel_half_thickness = APPEARANCE_TOWEL_HALF_THICKNESS_M
    towel_z = config.square_thickness_m - 0.0005 - towel_half_thickness
    towel = ET.SubElement(world, 'body', name=APPEARANCE_TOWEL_BODY,
        pos=f'{config.square_center_xy[0]} {config.square_center_xy[1]} {towel_z:.6f}')
    ET.SubElement(towel, 'geom', name=APPEARANCE_TOWEL_GEOM, type='box',
        size=f'{config.square_edge_m / 2} {config.square_edge_m / 2} {towel_half_thickness}',
        rgba='1 1 1 0', contype='0', conaffinity='0', density='0', group='2')
    if arm_model == 'so100':
        pitch=arm.find("default/default/default[@class='Pitch']/joint")
        pitch.set('range',f'{float(config.mujoco_low[1])} 0.174')
    else:
        # Shoulder floor on the SO-101: its joints and actuators carry explicit ranges (no class inheritrange).
        joint=arm.find(".//joint[@name='shoulder_lift']")
        actuator=arm.find(".//actuator/position[@name='shoulder_lift']")
        joint.set('range',f"{float(config.mujoco_low[1])} {joint.get('range').split()[1]}")
        actuator.set('ctrlrange',f"{float(config.mujoco_low[1])} {actuator.get('ctrlrange').split()[1]}")
    if config.servo is not None and arm_model == 'so101':
        # Per-joint servo model on the SO-101: written on each actuator and joint (kv replaces dampratio).
        for index,name in enumerate(JOINT_ORDER):
            act=arm.find(f".//actuator/position[@name='{name}']")
            act.set('kp',repr(float(config.servo['kp'][index]))); act.set('kv',repr(float(config.servo['kv'][index])))
            arm.find(f".//joint[@name='{name}']").set('frictionloss',repr(float(config.servo['frictionloss'][index])))
        sts=arm.find(".//default[@class='sts3215']/position")
        sts.attrib.pop('dampratio',None); sts.attrib.pop('kp',None); sts.attrib.pop('kv',None)
    elif config.servo is not None:
        # Measured servo model (2026-10-04): every joint class gets its own actuator stiffness and damping and its own
        # dry friction. The shared default's dampratio goes (kv and dampratio cannot both be set on one actuator).
        shared=arm.find("default/default[@class='so_arm100']/position")
        if shared is None or shared.get('dampratio') is None:
            raise RuntimeError('unexpected Menagerie actuator default; cannot write the servo block')
        del shared.attrib['dampratio']
        del shared.attrib['kp']
        for index,name in enumerate(SERVO_CLASSES):
            default=arm.find(f"default/default/default[@class='{name}']")
            default.find('joint').set('frictionloss',repr(float(config.servo['frictionloss'][index])))
            ET.SubElement(default,'position',kp=repr(float(config.servo['kp'][index])),kv=repr(float(config.servo['kv'][index])))
    # Calibrated wrist camera (2026-09-08 hand-eye): pose in the gripper frame and the lens's vertical field of view.
    if config.camera_pos is not None:
        cam = arm.find(".//camera[@name='wrist_camera']")
        cam.set('pos', ' '.join(f'{float(v):.7f}' for v in config.camera_pos))
        cam.set('quat', ' '.join(f'{float(v):.7f}' for v in config.camera_quat_wxyz))
        cam.set('fovy', f'{float(config.camera_fovy_deg):.2f}')
    lens = config.lens_model
    if lens is not None:
        # The simulator renders a wider pinhole that the lens operator resamples into the
        # observation; the offscreen framebuffer must hold that render (default 640x480).
        cam = arm.find(".//camera[@name='wrist_camera']")
        cam.set('fovy', f'{float(lens.render_fovy_deg):.2f}')
        glob = arm.find('visual/global')
        glob.set('offwidth', str(max(int(lens.render_size[0]), 640)))
        glob.set('offheight', str(max(int(lens.render_size[1]), 480)))
    ET.indent(arm)
    scene=destination/'scene_bench_pick_replace_v1.xml'
    ET.ElementTree(arm).write(scene,encoding='unicode')
    payload=asdict(config)
    if payload.get('servo') is None:
        payload.pop('servo',None)     # a scene without a servo block keeps the bench_config.json bytes (and hash) it always had
    (destination/'bench_config.json').write_text(json.dumps(payload,indent=2)+'\n')
    return scene

def main():
    p=argparse.ArgumentParser(description=__doc__)
    # Required: the active scene carries a measured reset, a viewing pose and
    # a tuned teacher and a calibrated camera; bare BenchConfig() defaults would
    # silently overwrite them.
    p.add_argument('--bench-config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT/'simulation_code/model/bench_pick_replace_v1')
    p.add_argument('--intrinsics',type=Path,default=None,help='camera_intrinsics.json: (re)build the lens block from its selected model and set camera_fovy_deg from it')
    p.add_argument('--render-size',type=int,nargs=2,default=(1600,900),metavar=('W','H'),help='pinhole render size for the lens path (16:9)')
    p.add_argument('--render-fovy',type=float,default=None,help='pinhole render fovy; default = smallest fovy covering the whole undistorted frame + 5%% margin')
    p.add_argument('--appearance',choices=['keep','none','default'],default='keep',
        help='appearance randomization regime written into bench_config.json: keep the config\'s block, remove it, or write the default bench_appearance v1 regime')
    p.add_argument('--placement',choices=['keep','none','default'],default='keep',
        help='placement randomization regime written into bench_config.json: keep, remove, or write the default bench_placement v1 rectangle (14 x 10 in, near edge 2 in from the base front edge)')
    p.add_argument('--servo',default='keep',
        help="'keep' (default), 'none' (the Menagerie actuators), or a tools/fit_servo_model.py record: its fitted "
             "per-joint stiffness, damping and dry friction are written into the scene and the bench config")
    p.add_argument('--servo-compensation',action='append',default=[],metavar='KEY=VALUE',
        help='with --servo <fit>: override a teacher compensation setting (contracts/bench.py SERVO_COMPENSATION_V1), e.g. '
             'direction_slew=0.2 (2026-11-08: certifies the SO-101 servo scene 30/30)')
    p.add_argument('--arm',choices=['so100','so101'],default='so100',
        help="arm model: the Menagerie SO-100 (default, the scenes so far) or the Menagerie SO-101 "
             "(simulation_code/model/menagerie_so101); so101 also converts a measured_20260908b config to measured_20260908b_so101")
    p.add_argument('--dynamics',choices=['bench','menagerie'],default='bench',
        help="SO-101 only: the bench's actuator/joint/solver settings (default) or Menagerie's as shipped")
    p.add_argument('--survey-pose',choices=['keep','default','reset'],default='keep',
        help='viewing_qpos: keep, write the default raised survey pose (reset + shoulder_lift 0.40, elbow -0.60, wrist_flex 0.40 rad; camera 235 mm up, sees the reachable rectangle), or set it back to the reset pose')
    a=p.parse_args()
    config=BenchConfig.load(a.bench_config)
    if a.appearance=='none':
        config=replace(config,appearance=None)
    elif a.appearance=='default':
        config=replace(config,appearance=AppearanceRegime().identity())
    if a.placement=='none':
        config=replace(config,placement=None)
    elif a.placement=='default':
        config=replace(config,placement=PlacementRegime().identity())
    if a.servo=='none':
        config=replace(config,servo=None)
    elif a.servo!='keep':
        fit_path=Path(a.servo)
        fitted=json.loads(fit_path.read_text())['fitted']
        import hashlib
        config=replace(config,servo=dict(name='bench_servo',version=1,kp=[float(v) for v in fitted['kp']],kv=[float(v) for v in fitted['kv']],
            frictionloss=[float(v) for v in fitted['frictionloss']],
            source=f'{fit_path.name} sha256 {hashlib.sha256(fit_path.read_bytes()).hexdigest()}',
            **({'compensation':{k:float(v) for k,v in (item.split('=',1) for item in a.servo_compensation)}} if a.servo_compensation else {})))
    if a.survey_pose=='default':
        config=replace(config,viewing_qpos=tuple(float(q+d) for q,d in zip(config.reset_qpos,SURVEY_POSE_DELTA_RAD)))
    elif a.survey_pose=='reset':
        config=replace(config,viewing_qpos=tuple(float(q) for q in config.reset_qpos))
    if a.intrinsics is not None:
        probe=LensModel.from_intrinsics_file(a.intrinsics,render_fovy_deg=90.0,render_size=tuple(a.render_size))
        fovy=a.render_fovy if a.render_fovy is not None else round(probe.required_render_fovy_deg(),2)
        lens=LensModel.from_intrinsics_file(a.intrinsics,render_fovy_deg=fovy,render_size=tuple(a.render_size))
        lens.sim_operator()  # coverage guard: fails here rather than at capture time
        config=replace(config,lens=lens.identity(),camera_fovy_deg=round(lens.fovy_deg,2))
    if a.arm=='so101' and config.joint_map=='measured_20260908b':
        # Same physical arm, the SO-101's joint conventions: the derived map and the survey pose converted exactly
        # (q_so101 = SIGN * (q_so100 - OFFSET); SIGN is its own inverse).
        import numpy as np
        from so_arm101_v2.contracts.coordinates import MODEL_CONVERSION_OFFSET, MODEL_CONVERSION_SIGN
        viewing=None if config.viewing_qpos is None else tuple(float(v) for v in
            MODEL_CONVERSION_SIGN.astype(np.float64)*(np.asarray(config.viewing_qpos,dtype=np.float64)-MODEL_CONVERSION_OFFSET.astype(np.float64)))
        config=replace(config,joint_map='measured_20260908b_so101',viewing_qpos=viewing)
    print(prepare(a.output_dir,config,arm_model=a.arm,dynamics=a.dynamics))

if __name__ == '__main__':main()
