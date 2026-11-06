"""Derive the SO-101 bench joint map from the measured SO-100 one (2026-11-06, handoff step 3).

``measured_20260908b`` maps servo ticks to the Menagerie SO-100's joint angles. The Menagerie SO-101 (vendored with
its base yawed 90 deg) uses the legacy so101_new_calib conventions, and the repo converts those to the SO-100 with
q_so100 = SIGN * q_so101 + OFFSET (contracts/coordinates.py). With q_so100 = s * (ticks - z) * 2pi/4096 that gives,
exactly, s_so101 = SIGN * s and z_so101 = z + OFFSET / (s * 2pi/4096) for the five body joints. No new measurement:
same encoder readings, same physical arm, another model's angle convention. The gripper keeps the legacy affine map.

  PYTHONNOUSERSITE=1 ~/miniforge3/envs/lerobot/bin/python tools/derive_joint_map_so101.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESOURCES = ROOT / "src/so_arm101_v2/data/resources"
SOURCE = RESOURCES / "physical_joint_map_20260908b.json"
TARGET = RESOURCES / "physical_joint_map_20260908b_so101.json"


def derive() -> dict:
    from so_arm101_v2.contracts.coordinates import JOINT_NAMES, MODEL_CONVERSION_OFFSET, MODEL_CONVERSION_SIGN
    raw = json.loads(SOURCE.read_text())
    k = 2.0 * np.pi / int(raw["ticks_per_turn"])
    joints = {}
    for index, name in enumerate(JOINT_NAMES[:5]):
        source = raw["joints"][name]
        sign = int(source["sign"]) * int(MODEL_CONVERSION_SIGN[index])
        zero = float(source["model_zero_ticks"]) + float(MODEL_CONVERSION_OFFSET[index]) / (int(source["sign"]) * k)
        joints[name] = {**source, "model_zero_ticks": round(zero, 6), "sign": sign,
                        "verification": f"derived from measured_20260908b ({source['model_zero_ticks']}, sign {source['sign']}) by the exact "
                                        f"SO-100 <-> SO-101 convention change; " + source["verification"]}
    return {
        "name": "measured_20260908b_so101",
        "schema_version": 1,
        "ticks_per_turn": raw["ticks_per_turn"],
        "model": "menagerie_so101",
        "description": ("measured_20260908b expressed in the Menagerie SO-101's joint conventions (legacy so101_new_calib "
                        "zeros, base yawed 90 deg; simulation_code/model/menagerie_so101). Derived, not measured: "
                        "tools/derive_joint_map_so101.py. The physical arm and its encoder readings are the same."),
        "derived_from": {"name": raw["name"], "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "gripper": raw["gripper"],
        "joints": joints,
        "references": raw.get("references", {}),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    text = json.dumps(derive(), indent=2) + "\n"
    if args.check:
        same = TARGET.is_file() and TARGET.read_text() == text
        print("current" if same else "differs")
        return 0 if same else 1
    TARGET.write_text(text)
    print(TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
