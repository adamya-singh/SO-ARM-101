"""Build the versioned 20 mm XYZ bench scene without modifying legacy XML."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from so_arm101_v2.contracts.bench import BenchConfig

ROOT = Path(__file__).resolve().parents[1]

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
    pitch=arm.find("default/default/default[@class='Pitch']/joint")
    pitch.set('range',f'{float(config.mujoco_low[1])} 0.174')
    ET.indent(arm)
    scene=destination/'scene_bench_pick_replace_v1.xml'
    ET.ElementTree(arm).write(scene,encoding='unicode')
    (destination/'bench_config.json').write_text(json.dumps(asdict(config),indent=2)+'\n')
    return scene

def main():
    p=argparse.ArgumentParser(description=__doc__)
    # Required: the active scene carries a measured reset, a viewing pose and
    # a tuned teacher (78 deg / pad 4); bare BenchConfig() defaults would
    # silently overwrite them.
    p.add_argument('--bench-config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT/'simulation_code/model/bench_pick_replace_v1')
    a=p.parse_args()
    print(prepare(a.output_dir,BenchConfig.load(a.bench_config)))

if __name__ == '__main__':main()
