# Vendored MuJoCo Menagerie trs_so_arm100

- Upstream: https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100
- Commit: 71f066ad0be9cd271f7ed58c030243ef157af9f4 (also in `.upstream_commit`)
- License: Apache-2.0 (see `LICENSE`, verbatim upstream copy)
- `upstream/` holds the pristine `so_arm100.xml` and `scene.xml`; they are loaded
  only by tests (`upstream/assets` is a symlink to `../assets`). The runtime
  entry point is `scene_v2.xml`, which includes the patched arm
  `so_arm101_v2.xml`.
- `assets/` is the unmodified upstream STL set plus one addition:
  `so101_camera_wrist_mount.stl`, copied from `simulation_code/model/assets/`
  (this repo's SO-101 wrist camera adapter; not part of Menagerie).

## Patch inventory (`so_arm101_v2.xml` vs `upstream/so_arm100.xml`)

1. Joints and actuators renamed to this repo's LeRobot convention:
   Rotation→shoulder_pan, Pitch→shoulder_lift, Elbow→elbow_flex,
   Wrist_Pitch→wrist_flex, Wrist_Roll→wrist_roll, Jaw→gripper.
2. Bodies renamed: Fixed_Jaw→gripper, Moving_Jaw→moving_jaw_so101_v1
   (names the v2 adapter and strict grasp detector resolve). Geom names,
   including `fixed_jaw_pad_1..4` / `moving_jaw_pad_1..4`, are untouched.
3. `Base` body pos/quat set to the numerically derived base-alignment
   transform so the arm occupies the same world workspace as the legacy
   `so101_new_calib.xml` (cube at (0, 0.3, 0.0125), fixed cameras).
4. `home` / `rest` keyframes removed (the runtime scene adds a freejoint cube,
   which changes nq and would make the keyframes fail to compile).
5. Added: sites `fixed_jaw_tip` (in `gripper`) and `moving_jaw_tip` (in
   `moving_jaw_so101_v1`); body `wrist_camera_mount` with camera
   `wrist_camera` (fovy 72) and the SO-101 mount mesh visual. Camera/site
   poses were FK-transplanted from the legacy model; the mount MESH geom pose
   was derived by transforming the legacy XML's declared local pose through
   the old-gripper -> new-gripper rigid transform (runtime world poses of mesh
   geoms must not be written back into XML: the compiler's mesh re-centering
   would be applied twice). See
   `src/so_arm101_v2/data/resources/simulation_model_contract_v2.json`.
6. Actuator gains kept at Menagerie defaults (kp=50, dampratio=1, forcerange
   ±3.5) — part of the community-validated grasping behavior. The legacy model
   used kp=17.8, forcerange ±3.35; reset-settle behavior was revalidated after
   the swap.

## Joint-convention conversion

The legacy model uses LeRobot-calibrated zeros; upstream uses CAD zeros.
The per-joint conversion q_menagerie = SIGN * q_legacy + OFFSET and its
derivation residuals are recorded in
`src/so_arm101_v2/data/resources/simulation_model_contract_v2.json` and
permanently re-verified by `tests/test_model_conversion.py`.

Known limitation: upstream is SO-ARM100 v1.3 CAD; the physical arm is an
SO-101. Kinematics are near-identical; FK residuals between the two models are
recorded in the model contract JSON.
