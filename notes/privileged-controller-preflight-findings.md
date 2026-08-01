# 2026-08-01 Privileged-Controller Preflight: Findings

> **Resolution update (later the same day):** the v2 lane was migrated to the
> MuJoCo Menagerie `trs_so_arm100` model, which fixes the collision-geometry
> defects described below the community-standard way (decomposed jaw collision
> meshes + fingertip pad primitives). The hand-made `scene_v2_collision*.xml`
> experiments referenced in this note are superseded and retained only as
> evidence. See `notes/menagerie-model-migration.md` for the migration record
> and current status.

## Status

The v2 simulation preflight (`so-arm101-v2-sim preflight`) was run for the first
time and **fails**, and after deep investigation the failure is not a controller
bug: **a strict opposing-face grasp of the 25 mm cube has been geometrically
impossible in every simulator configuration this repository has ever used.**
This retroactively explains the entire RL history (corner pinches everywhere,
~zero interior-face contacts across tens of thousands of episodes, 12,000-episode
runs with zero strict grasps).

## Root causes, in the order discovered

1. **Convex-hull collision fills the jaw mouth (production model).**
   `wrist_roll_follower_so101_v1` is a C-shaped bracket plus the fixed jaw in a
   single mesh. MuJoCo collides mesh geoms by convex hull, and the hull bridges
   the C, occupying the space where the cube must sit (measured ~15 mm phantom
   penetration at any grasp pose, cube ejected on contact). The moving jaw mesh
   (horn + curved neck + spade) has the same defect: its hull belly hangs over
   the mouth and rides the cube's top during descent.
   Fix implemented: `simulation_code/model/so101_new_calib_v2_collision.xml` +
   `scene_v2_collision.xml` replace both collision meshes with box
   decompositions derived from the mesh vertex extents (visual meshes
   unchanged; production `scene.xml` untouched).

2. **The jaw-tip sites are not on the jaws.** `fixed_jaw_tip` sits ~15-20 mm off
   the physical blade (outside the mesh bounding box); `moving_jaw_tip` is
   27 mm short of the real blade tip. Every tip-midpoint-based grasp target in
   the legacy stack inherited this bias.

3. **The July reset bank does not reproduce on the production model.** The
   bank-recipe replay (tip-pair IK, teleported block, closed ctrl, 250 steps)
   produces zero strict-grasp frames on both `scene.xml` and the v2 model, at
   the bank's own location and at the contract location. The bank's 64
   "validated" states evidently depended on the fingertip-pad collision model
   that was later reverted (campaign `excluded_jobs` 28/29). The strict-
   transition v1 failure note ("curriculum states lost strict grasp during
   production reset settling") was the same phenomenon.

4. **The jaws are a scissor, not a parallel gripper, at cube scale.** With the
   faithful box decomposition, the pad surfaces are parallel only at gripper
   ~0.0 rad where the pad gap is 1.7-7.4 mm. At a 25 mm gap the pad planes meet
   at ~26-34 degrees. A rigid 25 mm cube pinched by a 26-degree wedge contacts
   at edges and is expelled (watermelon-seed effect) - exactly the corner-pinch
   behavior every experiment observed. Additionally, at every reachable pose
   family the moving jaw's distal tip must pass below floor level to reach a
   25 mm gap on the segments that face the cube (the moving jaw is much longer
   than the fixed jaw), so floor-level grasps jam or sweep the cube away.
   The real hardware presumably succeeds via serrated pad surfaces (mean facet
   tilt ~17 deg on the fixed pad), print compliance, and friction - none of
   which rigid convex geoms reproduce.

5. **Bevelled pad plates make the cube holdable but not yet "strict".** Adding
   2 mm-thick pad-plate geoms bevelled 16 deg on each jaw
   (`*_v2_collision_test.xml`) makes the pads parallel (0.9 deg) at a 24 mm gap;
   a wedged cube is then retained and lifted (held at z = 53-81 mm in teleport
   tests - the first time any simulator config here has held the cube at all).
   But box-box contacts land as edge contacts within the detector's 4 mm corner
   margin, so `strict_bilateral_grasp` still never fires, and the floor-approach
   targeting must be re-derived against the pad-plate pockets.

## Controller state

`src/so_arm101_v2/simulation/privileged.py` was rewritten around pad-frame IK
(targets the fixed-pad collision geom, not the bogus tip sites; normal/pitch
constraints; laddered shifted descent + slide; slow two-stage close; rigid
tip-translation lift). Against the v2 model it produces converged IK
(<1 mm / <4 deg), zero command clipping, zero delta-limiter hits, and clean
descent - the remaining failure is entirely the grasp-contact physics above.

## Decision needed (Phase 4 gate, per the rebuild plan)

The preflight is doing its job: the environment cannot currently be solved even
with privileged access. Options, roughly in order of fidelity:

1. **Contact-model calibration (recommended).** Keep the v2 box decomposition,
   add the bevelled pad plates, and calibrate pad size/bevel/friction/solref
   plus a squeeze protocol until the *privileged* controller passes; document
   it as the July pad calibration did (it chose 2 mm pads, friction 1.0, for
   the same reason). This is an effective-contact model for compliant serrated
   pads, not a cheat.
2. **True mesh decomposition.** Run CoACD/VHACD on the two jaw STLs and use the
   convex pieces as collision geoms; re-measure whether real pad surfaces
   grip a 25 mm cube. Higher fidelity, more work, may still need friction/
   softness tuning.
3. **Contract revision.** If the gripper (parallel gap ~2-7 mm) is simply not a
   25 mm-cube parallel gripper, change the object (smaller cube, or a
   compressible/soft body) or relax the strict detector's dual 25-degree cone +
   4 mm corner margin to something a perfect rigid parallel grip can satisfy.
   The scenario distance (y = 0.30) is also at the edge of the dexterous
   workspace - the demos grasped near y = 0.24.

## Artifacts

- Failed first preflight: `artifacts/so_arm101_v2/simulation/fixed_pickup_contract_v1/`
  (15/15 deterministic, zero clipping, zero success; videos show the old
  site-based controller bumping the cube and never grasping).
- v2 collision model: `simulation_code/model/scene_v2_collision.xml`,
  `so101_new_calib_v2_collision.xml`.
- Pad-plate experiment: `scene_v2_collision_test.xml`,
  `so101_new_calib_v2_collision_test.xml` (16-degree bevelled 2 mm pads).
