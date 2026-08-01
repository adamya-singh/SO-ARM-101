# 2026-08-01 v2 Sim Lane Migration to MuJoCo Menagerie trs_so_arm100

## Status

**COMPLETE: the preflight PASSES on the Menagerie model** —
`environment_proven: true`, 15/15 scenario rollouts succeed deterministically
(strict bilateral face grasp sustained through the 1 s hold at ~29.8 mm height
gain, zero clipping, zero delta-limiter hits, zero unsafe contacts), and the
full test suite is green (84/84). This is the first time the Phase-4
"prove the simulator" gate of the rebuild plan has ever passed in this repo.
Passing report: `artifacts/so_arm101_v2/simulation/preflight/fixed_pickup_contract_v1/evaluation.json`
(the legacy-model failed run is archived alongside with a `_legacy_model_failed_20260801` suffix).

## What changed

- Vendored Menagerie `trs_so_arm100` (Apache-2.0, commit `71f066ad`) into
  `simulation_code/model/menagerie_so_arm100/` with a pristine `upstream/`
  copy and a patched `so_arm101_v2.xml` (renames to LeRobot joint/body names,
  base-alignment transform, keyframes removed, wrist camera + jaw-tip sites
  FK-transplanted from the legacy model). Runtime entry: `scene_v2.xml`.
  Patch inventory: `VENDOR.md`.
- Coordinate contract (`contracts/coordinates.py`): the ACT endpoint affine
  now composes with a numerically derived per-joint conversion
  `q_menagerie = SIGN * q_legacy + OFFSET`, `SIGN = [-1,1,1,1,1,1]`,
  `OFFSET = [0, -pi/2, +pi/2, 0, -0.04867319, 0]`. Derived from physical
  invariants only (axis lines, next-joint axes, distal COMs); validated at
  0.27 mm axis-line / 0.00 deg axis / 6.3 mm distal-COM residuals over 120
  random poses (the COM residual is the real SO-100 v1.3 vs SO-101 CAD
  difference). Permanently re-verified by `tests/test_model_conversion.py`.
  New resource: `data/resources/simulation_model_contract_v2.json`
  (SHA-pinned). The byte-pinned `act_coordinate_contract.json` is untouched.
- Task/suite contracts: `fixed_cube_pickup_v1.json` bumped to
  `contract_version: 2` with the converted reset pose
  `(-0.0010472, -3.31, 3.13, 1.1599458, -1.5154479, 0.0314159)` and new
  safety bounds (converted calibrated envelope intersected with Menagerie
  mechanical ranges); both simulation suites updated.
- Strict grasp detector: `FACE_GRASP_CORNER_MARGIN` 4 mm -> 2 mm. The old
  margin was tuned to the legacy blade jaws; the Menagerie fingertip pads
  (community-standard) legitimately contact within ~2.5 mm of a 25 mm cube's
  edges when gripping flat. All other strictness (face normals within 25 deg,
  jaw-axis cone, opposition, bilateral min force, surface tolerance) is
  unchanged, and near-edge pinches (<2 mm) are still rejected.
- CLI defaults and tests point at the new scene. Legacy `scene.xml` and the
  whole legacy stack are untouched.

## Verified behavior on the new model

- Wedge test (cube placed at the pad-4 pocket, gripper commanded closed):
  **strict bilateral grasp for 269 consecutive frames at 56.5 N** and the cube
  held indefinitely - the first strict grasp ever achieved in this repo, and
  confirmation that the Menagerie pads + soft contact give a real, stable,
  detector-valid grip. The pad-4 pair is parallel (0.0 deg) at a 25.2 mm gap
  at gripper qpos ~0.0 - the gripper is designed for exactly this object size.
- Wrist-camera view at the converted reset pose matches the legacy render
  almost exactly (same jaw framing, cube position).

## How the final gap was closed (resolved 2026-08-01)

The earlier "floor-pick gap" turned out to be misdiagnosed. Live viewer
inspection (user observation) showed the FIXED jaw descending on top of the
cube instead of beside it. Root cause: a sign error in the descent clearance
shift - shifting the pocket TARGET along the pad normal displaces the cube
(which is fixed in the world) TOWARD the fixed pad, the exact opposite of the
intent. Every earlier parameter sweep ran with the inverted sign, which is why
none of them worked and why the failure was misattributed to bill-underside
floor geometry.

Two-line fix in `simulation/privileged.py`:

1. `mouth_shift = -approach_clearance_m * mouth_normal` (sign flipped): the
   descent now runs with the cube displaced toward the open moving-jaw side,
   and the fixed pad clears it cleanly.
2. `depth_lead_m = 0.006`: the grip patch was landing on the cube's far-edge
   band (corner-rejected by the strict detector at 59 N of real force); a
   6 mm depth lead centers the pad patch on the face. Sweep showed success at
   4-8 mm of lead with 6 mm giving the widest strict-frame margin.

Lesson recorded: when a controller fails geometrically, watch it in the
viewer before sweeping parameters - one human observation localized in
seconds what blind sweeps mislocalized for hours.

## Post-migration fixes

- **Camera-mount visual pose (2026-08-01, verified live).** The mount mesh
  initially rendered floating and rotated ~90 deg while the camera itself was
  correct. Cause: MuJoCo's compiler re-centers mesh assets and folds that
  transform into the geom pose, so reading a mesh geom's *runtime* world pose
  and writing it back into a new XML double-applies the re-centering. Plain
  frames (cameras, sites) are unaffected, which is why only the mount broke.
  Fix: transform the legacy XML's *declared* local pose through the rigid
  old-gripper -> new-gripper body transform instead (`pos="0.0174898
  -0.0377991 0.0258225" quat="0 -0.7071068 0 0.7071068"`); the same math
  reproduces the known-good camera pose to 7 decimals. Rule for future
  transplants: never round-trip mesh geom poses through runtime world frames.

## Tools

- `tools/view_privileged_live.py` - drives the preflight controller and
  scenarios in the interactive MuJoCo viewer in real time (WSLg works; do not
  set `MUJOCO_GL=egl` for this). From the repo root:
  `PYTHONPATH=src python tools/view_privileged_live.py [--scenario N] [--speed X]`

## Files

- Model: `simulation_code/model/menagerie_so_arm100/` (VENDOR.md, LICENSE,
  upstream/, assets/, so_arm101_v2.xml, scene_v2.xml)
- Code: `src/so_arm101_v2/contracts/coordinates.py`, `contracts/task.py`,
  `simulation/privileged.py`, `simulation/contact.py`, both CLIs
- Resources: `simulation_model_contract_v2.json` (new),
  `fixed_cube_pickup_v1.json`, both suite JSONs
- Tests: `test_model_conversion.py` (new), `test_coordinates.py`,
  `test_calibration_integrity.py`, `test_task_contract.py`,
  `test_full_dataset_and_simulation.py`, `test_visualization_and_learning.py`

## Post-pass tuning (2026-08-01)

- `grasp_height_m` 0.017 -> 0.021: eliminates all jaw-floor contact during the
  pick (143 frames/episode -> 0) while keeping 5/5 scenario success.
- `lift_command_m` -> 0.031 with a faster lift stage: the cube tops out at
  ~29.8 mm (target 3 cm). Note the episode terminates on success, so the
  measured apex depends on lift speed as well as the commanded height;
  realized gain ~= command - 1.3 mm with the current stage timing.
