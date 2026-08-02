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

## Napkin place phase (2026-08-01)

The physical dataset episodes end by placing the cube on a 2 in x 2 in napkin
a few inches from the cube; the sim now matches:

- `scene_v2.xml` adds a static 50.8 mm x 50.8 mm x 1 mm `napkin` geom at
  (0.08, 0.30) - it enters both the physics and every camera view, closing a
  visual domain gap for future imitation learning.
- The privileged controller reads the napkin geom's pose and, after the lift,
  traverses over it, sets the cube down, opens, and retreats (place-phase IK
  solves run with loosened orientation tolerances/weights - carrying does not
  need grasp-grade wrist orientation). All five suite scenarios finish with
  the cube at rest on the napkin.
- The preserved v2 task contract still terminates on pickup success, so its
  contract-evaluated episodes and preflight were unchanged (re-certified after
  the scene change: environment_proven=true, 15/15, tests 84/84 at that
  migration checkpoint). A separate full-task v3 contract was subsequently
  added and is documented below; v2 semantics and artifacts remain unchanged.
- `tools/view_privileged_live.py` plays the full pick-and-place by default;
  `--stop-on-success` reproduces the contract's early termination.

## Full-task v3 contract and oracle distillation (2026-08-01)

The napkin behavior is now independently certified without changing the
pickup-v2 contract or its artifacts:

- `fixed_cube_pick_place_v3` extends the earlier strict pickup requirement with
  napkin-local cube-footprint containment, support height, release, linear and
  angular rest thresholds for ten consecutive frames, a 40 mm retreat, and
  permanent safety invalidation. Its 480-action budget lets the 450-action
  controller complete and hold its final pose.
- The `fixed_pick_place_v3` privileged preflight passes 15/15 deterministically
  with zero clipping, limiting, nonfinite commands, or unsafe contacts. Report:
  `artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json`.
- The capture tool recorded one content-addressed nominal demonstration with
  exactly 450 aligned pre-action state / requested action / executed action /
  post-action measurement rows, plus wrist and overview videos. Manifest:
  `artifacts/so_arm101_v2/oracle_distillation/oracle/fixed_pick_place_v3/
  5de8ab6ee95e2500/manifest.json`.
- The first deliberately small `phase_state` residual MLP did **not** pass the
  near-exact offline gate after its fixed 10,000 steps. It reached normalized
  delta MSE `4.305080802e-05` and maximum ACT error `0.04552662373`; the gates
  were `1e-6` and `0.01`. Its worst error is shoulder lift at row 446 in the
  five-frame retreat transition, while all predicted training commands remain
  inside the unchanged safety path. This is a valid Karpathy-style stop: feedback-state,
  multi-scenario cloning, and learned closed-loop evaluation were not run.

### Smooth-retreat controlled follow-up

The five-frame final retreat was then tested as a specific causal hypothesis.
It was lengthened to 26 actions while preserving 16 actions for gripper
opening. A first rebalance that shortened opening to 10 actions was rejected:
servo lag activated the real delta limiter at action 411 and invalidated all
15 preflight rollouts. With the 16-action opening restored, the smooth
controller passed the v3 preflight 15/15 deterministically with zero clipping,
limiting, nonfinite commands, or unsafe contacts. The current preflight report
at the path above is this smooth-retreat certificate.

The new nominal capture contains the same 450 aligned rows and is stored at:

`artifacts/so_arm101_v2/oracle_distillation/oracle/fixed_pick_place_v3/
9164a76699186c34/manifest.json`

The unchanged 128-wide, seed-101 `phase_state` clone improved materially but
still failed its predefined offline gate:

| Measurement | Five-frame retreat | Smooth retreat | Gate |
|---|---:|---:|---:|
| Normalized delta MSE | `4.305080802e-05` | `2.328354094e-05` | `<=1e-6` |
| Maximum ACT error | `0.04552662373` | `0.01958596706` | `<=0.01` |
| Worst point | row 446, retreat | row 147, mid-descent | n/a |
| Training safety violations | 0 | 0 | 0 |

This removes retreat as the dominant error but leaves a whole-trajectory
optimization or capacity floor. Diagnostic:
`artifacts/so_arm101_v2/oracle_distillation/models/phase_state/
0db4db27b04f7b1c/report.html`.

A late 10x learning-rate-reduction diagnostic was also reported at
`1.99e-05` MSE and `0.0201` maximum error, so simple decay did not clear either
gate. No immutable artifact for that diagnostic is present; it must not be
treated as promotable evidence. The checked-in trainer still uses fixed-rate
Adam.

Commands from the repository root (inside the `lerobot` environment):

```bash
PYTHONPATH=src python -m so_arm101_v2.simulation.cli preflight \
  --suite fixed_pick_place_v3 --no-video
PYTHONPATH=src python -m so_arm101_v2.simulation.cli capture-oracle \
  --suite fixed_pick_place_v3 --scenario nominal
PYTHONPATH=src python -m so_arm101_v2.learning.cli distill-oracle \
  --manifest artifacts/so_arm101_v2/oracle_distillation/oracle/fixed_pick_place_v3/9164a76699186c34/manifest.json \
  --model-kind phase_state --seed 101 --max-steps 10000 \
  --hidden-width 128 --training-rows memorization32

PYTHONPATH=src python -m so_arm101_v2.learning.cli distill-oracle \
  --manifest artifacts/so_arm101_v2/oracle_distillation/oracle/fixed_pick_place_v3/9164a76699186c34/manifest.json \
  --model-kind phase_state --seed 101 --max-steps 10000 \
  --hidden-width 128 --training-rows full

PYTHONPATH=src python -m so_arm101_v2.learning.cli distill-oracle \
  --manifest artifacts/so_arm101_v2/oracle_distillation/oracle/fixed_pick_place_v3/9164a76699186c34/manifest.json \
  --model-kind phase_state --seed 101 --max-steps 10000 \
  --hidden-width 256 --training-rows full
```

These policies use privileged simulator state and are diagnostics only. Wrist
images are recorded for inspection but are not model inputs.

### Controlled capacity gate

The planned capacity diagnostic was executed without changing the oracle,
features, target, optimizer, learning rate, seed, step budget, or thresholds:

| Run | Rows | Width | MSE | Maximum ACT error | Safety violations | Result |
|---|---:|---:|---:|---:|---:|---|
| Fixed memorization set | 32 | 128 | `9.945671309e-07` | `0.004400968552` | 0 | pass, diagnostic only |
| Full parity control | 450 | 128 | `2.328354094e-05` | `0.01958596706` | 0 | fail, exact prior replay |
| Full capacity test | 450 | 256 | `4.330015145e-06` | `0.009032011032` | 0 | fail MSE gate |

The 32 rows are selected only after computing normalization on all 450 source
rows. The subset passed at step 7,577 and is structurally prohibited from
closed-loop promotion. The width-128 full replay matched the old loss trace and
metrics exactly, establishing that width was the only optimization change in
the 256 run. Width 256 cleared the maximum-error threshold but missed the
`1e-6` MSE threshold by 4.33x. Its immutable report is
`artifacts/so_arm101_v2/oracle_distillation/models/phase_state/
6fc677db0c3ae755/report.json`.

The offline report therefore has `closed_loop_eligible: false`; no nominal
MuJoCo clone evaluation was run. Feedback state, five-scenario training,
vision, ACT, physical deployment, RL, and additional capacity or optimizer
changes remain blocked pending a new controlled plan.

The two prerequisite reports are
`artifacts/so_arm101_v2/oracle_distillation/models/phase_state/
2215e6361027023e/report.json` (fixed-32 pass) and
`artifacts/so_arm101_v2/oracle_distillation/models/phase_state/
5b9ff38c61eb8673/report.json` (exact width-128/full parity replay). The source
suite passed 107 tests at that capacity-gate checkpoint.

### Closed-loop diagnosis and phase-wide recovery follow-up (2026-08-02)

The later width-256 scheduled run passed the full-450 offline gate at step
28,718 (`9.999720305e-07` normalized MSE, `0.007196128` maximum ACT error,
zero safety violations), but its three nominal autonomous repeats all failed
pickup. This is the original offline/closed-loop discrepancy; model artifact:
`artifacts/so_arm101_v2/oracle_distillation/models/phase_state/
d5f96d397bd9b915/`, evaluation artifact:
`artifacts/so_arm101_v2/oracle_distillation/clone_evaluations/
943cf536710e3d84/`.

The aligned diagnosis found small joint error from the first action and a cube
trajectory split at shared first contact, action 194. The decisive fixed-action
test inferred all 450 clone actions on teacher states and replayed them without
feedback; that sequence completed pick-place safely. This establishes
covariate shift/feedback compounding on the demonstrated system. It does not
identify one scalar error threshold as the cause. Immutable evidence is under
`artifacts/so_arm101_v2/oracle_distillation/diagnostics/
first_divergence_v1/` and `fixed_action_replay_v1/`.

The controlled data follow-up added eight physical MuJoCo recovery rows at
approach (70), first contact (195), seating (205), closure (255), lift (315),
transport (365), placement (386), and release (419). Each was generated by a
`0.01`-ACT perturbation to the prior command, reproduced exactly twice, labeled
by the unchanged phase oracle, passed unchanged through the safety layer, and
validated by a complete safe oracle suffix. The manifest also records changes
to velocity, cube orientation/velocity, and other quantities omitted from the
10-input student; these omissions limit the generality of the labels but did
not create a demonstrated contradiction for the eight accepted states.
Recovery artifact: `artifacts/so_arm101_v2/oracle_distillation/recovery/
0570ec8c0d37002f/`.

With only those rows added, the otherwise unchanged 458-row run passed offline
at step 24,691 (`9.999772601e-07` MSE, `0.005619988` maximum ACT error, zero
training-command safety violations). Autonomous control regressed: all 15
standard rollouts failed, with 291 clipped frames and 13 unsafe-contact frames
in each nominal repeat. Exact handoffs at the labeled anchors also fell from
`3/8` successes for the old clone to `1/8` for the augmented clone. Artifacts:
`artifacts/so_arm101_v2/oracle_distillation/models/phase_state/
c3ca76dc2c0fa42d/`, `clone_evaluations/e67ba4433d3aa98c/`, and
`recovery_evaluations/{4e5e4f1252d582ae,b0e6e56a018c1d7f}/`.

Established conclusion: eight isolated equal-weight anchors are insufficient
and can destabilize off-table predictions despite excellent finite-table fit.
Hypotheses about hidden contact-state aliasing remain unproven. The next
controlled test is to reuse the same immutable rows at relative recovery-loss
weights `0.10`, `0.25`, and `0.50`, reject saturation using a dense pre-rollout
command-bound scan, and require nominal `3/3` with zero safety events before
broader evaluation. No larger model, new feature, vision, ACT, RL, or physical
deployment is authorized by these results.

The complete source suite after this follow-up passes 114 tests.
