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

ROOT = Path(__file__).resolve().parents[1]
# Names shared with so_arm101_v2.simulation.appearance (the applier resolves them by name).
APPEARANCE_TEXTURE = 'ground_speckle'
APPEARANCE_TEXTURE_SIZE = 256
APPEARANCE_TOWEL_BODY = 'towel_visual'
APPEARANCE_TOWEL_GEOM = 'towel_visual_geom'
APPEARANCE_TOWEL_HALF_THICKNESS_M = 0.0002

def prepare(destination: Path, config: BenchConfig):
    source = ROOT / 'simulation_code/model/menagerie_so_arm100'
    destination.mkdir(parents=True, exist_ok=True)
    # Include-expanded XML retains the original body's frames, inertia,
    # collision decomposition, and camera calibration.
    arm = ET.parse(source / 'so_arm101_v2.xml').getroot()
    arm.set('model', 'bench_pick_replace_v1')
    import os
    arm.find('compiler').set('meshdir', os.path.relpath(source / 'assets', destination))
    baseline = ET.parse(source / 'scene_v2.xml').getroot()
    arm.append(baseline.find('visual'))
    assets = arm.find('asset')
    assets.find("material[@name='white']").set('rgba','0.07 0.07 0.07 1')
    for element in baseline.find('asset'):
        assets.append(element)
    cube_source = ROOT.parent / '3d-printing/oak-d-lite-mount/things-I-used/XYZ 20mm Calibration Cube - 1278865/files/xyzCalibration_cube.stl'
    cube_dest = destination / 'xyzCalibration_cube.stl'
    if cube_source.is_file():
        shutil.copyfile(cube_source, cube_dest)
    elif not cube_dest.is_file():
        raise FileNotFoundError(f'XYZ cube STL not found at {cube_source} and no copy exists in {destination}')
    ET.SubElement(assets, 'mesh', name='xyz_cube_visual',
        file=os.path.relpath(destination / 'xyzCalibration_cube.stl', source / 'assets'),
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
    pitch=arm.find("default/default/default[@class='Pitch']/joint")
    pitch.set('range',f'{float(config.mujoco_low[1])} 0.174')
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
    (destination/'bench_config.json').write_text(json.dumps(asdict(config),indent=2)+'\n')
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
    a=p.parse_args()
    config=BenchConfig.load(a.bench_config)
    if a.appearance=='none':
        config=replace(config,appearance=None)
    elif a.appearance=='default':
        config=replace(config,appearance=AppearanceRegime().identity())
    if a.intrinsics is not None:
        probe=LensModel.from_intrinsics_file(a.intrinsics,render_fovy_deg=90.0,render_size=tuple(a.render_size))
        fovy=a.render_fovy if a.render_fovy is not None else round(probe.required_render_fovy_deg(),2)
        lens=LensModel.from_intrinsics_file(a.intrinsics,render_fovy_deg=fovy,render_size=tuple(a.render_size))
        lens.sim_operator()  # coverage guard: fails here rather than at capture time
        config=replace(config,lens=lens.identity(),camera_fovy_deg=round(lens.fovy_deg,2))
    print(prepare(a.output_dir,config))

if __name__ == '__main__':main()
