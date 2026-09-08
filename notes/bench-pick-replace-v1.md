# Bench lift-and-replace, September 2026

This is the governing record for the active `bench_pick_replace_v1`
experiment: physical setup, recorded hardware state, active scene and
teacher, evidence inventory, gates, open work, and resume commands. It
absorbed the 2026-09-06 assistant handoff note, which has been deleted.

## Status (2026-09-08)

**All gates passed; the single pre-registered run is in progress.** Launched
2026-09-08 03:03 local on the persistent queue (`tsp` job 0) as W&B run
`bench-pick-replace-v1-s202-120k`, id `tinmahze`, project
`so-arm101-v2-scaling`:
https://wandb.ai/7adamyasingh-rutgers-university/so-arm101-v2-scaling/runs/tinmahze.
Experiment directory
`artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/seed202_120k_20260908/`
(`progress.json` is the monitoring source of truth; `experiment.json` is the
immutable identity; `provenance.json` records suite ids, preflights and the
capture digest). Scene `6477c4bd…`, joint map `measured_20260908b`, detector
`pad_normals_v2`, calibrated camera (fovy 44.0°).

- Gate 1 (reset + camera review): **closed 2026-09-08**, user sign-off in
  `inspection/camera_review_20260908.json`.
- Gate 2 (teacher certification): **15/15, deterministic, zero safety
  invalidation** on the corrected arm (scene `6477c4bd…`):
  `artifacts/so_arm101_v2/bench_pick_replace_v1/teacher_certification/6477c4bda5b92eaa-pad_normals_v2/preflight/bench_pick_replace_v1_certification_5ef75fc85458/evaluation.json`.
  Earlier certifications (`512f897d…`, `f198fce2…`, `c2053419…`) used
  superseded joint maps or the old detector and are historical only.
- Full test suite: **259 passed** at launch.
- Monitoring: a detached watcher writes `health_20min.txt` twenty minutes
  after the first optimizer step; the assistant session checks the run
  twice an hour (`tools/bench_health_check.py`), reporting only on stall,
  failure or completion. No automatic recipe changes.
- The August results elsewhere in this repository concern the legacy fixed
  25 mm cube task and establish nothing about bench readiness; the run above
  is the first bench evidence and is **exploratory** (single seed).

**Deployment reminder.** The simulator renders a pinhole camera. Real wrist
frames carry strong barrel distortion (k1 ≈ −0.57) and must be undistorted
with `camera_calibration/camera_intrinsics.json` (`selected` model) before
the policy sees them, then resized to 256×256 the same way the capture does.
This step is not yet implemented in any physical inference path.

## Physical setup

![Unaltered physical setup](../readme-assets/bench-setup-20260906-original.jpg)

The book was used for spacing because no ruler was available. The user
measured **8.5 inches (215.9 mm)** from the front edge of the arm base to the square center, centered
forward. The model’s base mesh front edge is at world Y = 64.6353 mm
(compiled mesh/geom world transforms; base bounds min
[-0.0554625, -0.0312647, -0.0024], max [0.0554625, 0.0646353, 0.08109645]),
so the square center is at world Y = 280.5353 mm. This mesh-derived offset
is recorded separately from the user’s measured distance; the photo is not
used to estimate it. This supersedes the earlier 7.5 inch measurement and
the mistaken reading of 215.9 mm from the rotation origin. Verify the
mesh/base correspondence when completing camera alignment. The cube is a
black **20 mm PLA XYZ calibration cube** (letters X/Y/Z on three faces;
existing STL for visuals, separate 20 mm box collision), initially on a
**50.8 mm square** over a black mousepad. The same square is the
replacement target. Remove the book before task trials. Cube mass and
friction and the 1 mm square thickness remain simulation assumptions, not
physical measurements.

The README uses a separately annotated copy; do not use generative image
output to infer geometry. The unaltered photo above is the reference.
Annotation was made with the built-in imagegen tool with the prompt:
“Preserve the photograph, every object, pose, scale, camera perspective,
lighting and existing book text. Add only a high-contrast caption in unused
mousepad area: The book was used as a spacing guide because no ruler was
available. The square is 8.5 inches forward of the arm base (measured).”
Its generic “arm base” caption means the base **front edge**.

## Hardware, runtime, and last observed state

- Git root `/home/win10ubuntu/dev/robotic-arm/SO-ARM-101`; the outer
  `robotic-arm` directory is not a Git repository. Hardware driver source is
  in the sibling `../lerobot-fork`.
- Python `/home/win10ubuntu/miniforge3/envs/lerobot/bin/python` with
  `PYTHONNOUSERSITE=1`; `MUJOCO_GL=egl` for rendering. MuJoCo pinned to
  **3.9.0**. Preserve the numerics regime; do not upgrade dependencies.
- GPU RTX 3090 (24 GB); about 559 GB disk free on 2026-09-06. Recheck before
  capture.
- Serial: `/dev/ttyACM0`, CH343 `1a86:55d3`, serial `5A68009601`. Camera:
  `/dev/video0`, `0c45:6366`, USB bus `5-3` (`/dev/video1` is metadata).
  Device enumeration can change; inspect before connecting.
- Windows USB forwarding: `/mnt/c/Program Files/usbipd-win/usbipd.exe` (not
  on PATH); `attach --wsl --busid 5-3` / `detach --busid 5-3` for the
  camera. Detach for lens adjustment in Windows and reattach only when the
  user is done. Camera works as MJPEG 1920×1080 at 30 Hz; uncompressed
  1080p over forwarding produced corrupt green frames.
- Live calibration `~/.cache/huggingface/lerobot/calibration/robots/so_follower/None.json`,
  checked against the pinned
  `src/so_arm101_v2/data/resources/physical_inference_calibration_20260620.json`.
  Never recalibrate automatically.
- Read-only inspection uses bus connect/read/disconnect with
  `disable_torque=False`, bypassing `robot.connect()`, which configures
  motors and changes torque.

**Last observed arm state (2026-09-08): torque off, arm at gravity rest**
(shoulder −92.08, elbow 100.0, wrist flex ≈ 38–44, roll ≈ −2 normalized;
the user turned the roll by hand during calibration and returned it). The
side photo `readme-assets/bench-rest-side-20260908.jpg` shows this pose:
upper arm horizontal backward, forearm horizontal forward stacked on it,
gripper folded down. Under the corrected joint map this rest pose is inside
the model (elbow at its calibrated maximum = model 173.5°), so the elbow no
longer needs staging for range; the shoulder at rest sits 0.08 below the −92
floor and must be lifted a few units before any physical episode. Never
assume the current pose from any file; measure read-only first
(`tools/read_joint_reference.py`). Joint order: shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, wrist_roll, gripper.

## Rest pose and elbow preparation

**Superseded for range on 2026-09-08.** The paragraphs below describe the
elbow staging performed under the June affine joint map, which placed the
gravity-rest elbow outside the model. Under `measured_20260908b` the same
rest reading (100.0 units) maps to the model's elbow maximum, inside range,
so staging is no longer required for range. What still binds is the
shoulder: gravity rest reads −92.08, below the −92 floor, so a small
reviewed shoulder lift (no tool exists yet) precedes any physical episode.
Kept for history and because the staging tool's safety pattern (read-only
default, per-step floor check, feedback stop) is the template for that
shoulder tool.

The observed gravity-rest elbow is about 99.73 calibrated units, which the
June affine map sent to 3.256 rad, beyond the Menagerie model's 3.14-rad
elbow maximum. This is not a
reason to silently clip a reset, recalibrate the arm, or widen model limits.
The user requested an explicit preparation step before every run instead.
The user confirmed being beside the arm with a clear workspace and approved
elbow-only preparation, which was performed successfully. **That approval
did not extend to a full-task physical test.**

`tools/prepare_physical_elbow.py` defaults to a read-only dry run. With
`--enable-motion --log <new.csv>` it requires an interactive confirmation,
then enables and commands **only the elbow**, moving inward in at most 0.1
calibrated-unit increments, stopping when measured elbow position is at most
92 units. A command floor of 85 and maximum command lead of 10 units
allow for observed gravity tracking error. The successful preparation
settled at 91.26 measured units, with shoulder lift at −90.84. Every observation checks the
−92 shoulder floor. The temporary elbow-only interval from observed gravity
rest into the validated range is confined to this preparation tool; learned control and replay
retain all model limits. A timeout or unexpected motion stops further
commands. Torque stays enabled afterward to prevent gravity collapse.
Earlier tiny-increment and goal-90 trials stalled on gravity tracking
error; their logs (`artifacts/so_arm101_v2/bench_pick_replace_v1/
elbow-preparation-20260906*.csv`) are evidence, not commands to repeat.

Capture the prepared pose using `tools/read_bench_pose.py --output-dir
artifacts/so_arm101_v2/bench_pick_replace_v1/reset_inspections`. It preserves
20 normalized/raw-tick samples, serial identity, timestamp and calibration
hash. Invalid or moving snapshots produce evidence but no usable config.
No task motion is implicit in recording a reset.

### Recorded reset

Stable 20-sample evidence:
`artifacts/so_arm101_v2/bench_pick_replace_v1/reset_inspections/9ed7de5997d0db89/reset_evidence.json`,
content digest `9ed7de5997d0db89014ff2af0e177cd3e0a1780fab247b0eac6c26342b28d8bc`
(the repository's canonical `content_sha256`, not the file byte hash;
re-verified 2026-09-06).

```
normalized physical reset (calibrated units):
[0.49315068493149283, -90.83629893238434, 91.26126126126127,
 43.64351245085189, -1.1965811965811923, 13.533834586466165]

MuJoCo reset (rad, gripper in model units):
[-0.009467720985412598, -3.156188726425171, 3.1131114959716797,
 0.7236365079879761, -0.03340867906808853, 0.08529800176620483]
```

The sibling `bench_config.json` inside that evidence directory is a
historical snapshot with the earlier distance interpretation. **Do not
overwrite historical evidence or copy that geometry into the active
scene.** The active configuration is
`simulation_code/model/bench_pick_replace_v1/bench_config.json`.

## Joint map: physical servo units to MuJoCo (corrected 2026-09-07)

The user compared the sim reset with the physical arm in the live viewer and
found the wrist roll a quarter turn off. Two read-only encoder recordings
then fixed the whole map (`tools/read_joint_reference.py`, torque off, arm
posed by hand):

- `artifacts/so_arm101_v2/bench_pick_replace_v1/joint_references/2026-09-07T133225_wrist_roll_jaws_horizontal_rest_396aa222.json`:
  gravity rest, jaws horizontal. Earlier the user turned the roll by hand
  from 2023 to 3089 ticks (+93.7°); the jaws then opened vertically with the
  moving finger on top, matching the old sim reset. That pins the roll offset
  and its direction.
- `artifacts/so_arm101_v2/bench_pick_replace_v1/joint_references/2026-09-07T133609_full_model_zero_reference_hand_held_25d117ec.json`:
  the model's all-zero configuration held by hand (arm straight out
  horizontally forward, jaws opening vertically). Ticks: pan 1957, lift 2003,
  elbow 2114, wrist flex 2031, roll 3113.

The legacy map (`act_coordinate_contract.json`, June 2026) assumed each
calibrated tick range spans the model joint range; the spans differ by up to
37 % and the shoulder-lift, elbow and wrist-roll zeros were each ~90° off.
The new map `measured_20260907` (`contracts/joint_map.py`,
`data/resources/physical_joint_map_20260907.json`) uses the exact 4096
ticks/turn scale and the reference ticks as zero offsets; the gripper keeps
the legacy affine endpoints because the model cannot represent touching jaws.
Signs: roll verified by the user's turn; shoulder, elbow and wrist flex
consistent with the rest pose; pan carried over from the legacy FK match and
not yet verified physically. The hand-held reference is good to roughly ±3°.

| Rest pose (model degrees) | pan | lift | elbow | wrist flex | roll |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy map | −0.5 | −180.8 | 178.4 | 41.5 | −1.9 |
| measured map | 0.1 | −85.3 | 92.5 | 41.6 | −99.5 |

Consequences: the sim rest pose now matches the setup photo (upper arm
vertical, forearm horizontal forward, gripper pitched 42° down over the
square, jaws horizontal). The physical joint ranges in model terms are pan
±80°, lift −85° (the −92 floor) to +10°, elbow −10° to 92.5° (the calibrated
maximum is the gravity rest), wrist flex ±95°, roll −160° to +86°. The
gravity-rest elbow is therefore inside the model and needs no staging; the
shoulder at rest reads −92.08, 0.08 below the floor, so a small shoulder lift
is still needed before any physical episode. **Every legacy sim trajectory
folded the shoulder to about −180°, a pose the physical arm cannot reach, so
the August results say nothing about physical feasibility.** The legacy lane
keeps `legacy_affine_v1` so its artifacts stay reproducible; the bench scene
binds to the measured map through `BenchConfig.joint_map`, which is hashed
into the scene dependencies and every capture identity.

## Camera calibration and joint-zero correction (2026-09-08)

The mount is the official SO-ARM101 small camera mount, i.e. the very part
the Menagerie model already places on the gripper, so the camera position was
known and only the lens and the arm's joint zeros were in question.

**Lens.** 29 hand-held views of a checkerboard shown 1:1 on an iPhone 15 Plus
(160 px squares = 8.835 mm at 460 ppi) plus 9 flat-on-table views:
`cv2.calibrateCamera`, principal point fixed at the image centre, five
distortion coefficients: **fovy 44.0°, fovx 71.5°, 1.9 px RMS**, strong barrel
distortion (k1 −0.57). The 103° figure found online for this module family is
wrong for this lens; the sim had been using 72°. Record:
`artifacts/so_arm101_v2/bench_pick_replace_v1/camera_calibration/camera_intrinsics.json`.
MuJoCo renders a pinhole, so physical frames must be undistorted with these
coefficients before the policy sees them.

**Joint zeros.** With the phone flat on the mousepad and the arm posed by hand
(torque off), 9 frames were kept in which all 98 corners of an asymmetric
8×15-square board were detected together with a read-only joint reading
(`tools/capture_checkerboard.py`, `tools/checkerboard_assist.py`; the first
symmetric board flipped its corner order between frames and had to be
replaced). Per-frame PnP gives the camera pose to ~1 px. Relative rotation
angles from the encoders matched the camera's to 2.5°, so scales and signs were
right; the camera's height above the board disagreed with the arm model by up
to 10 cm, so the shape of the arm was wrong. Deriving the board's world pose
from each frame through the forward kinematics and the mount camera, the nine
poses agree to **21 mm** (tilt 3.5°) only when the shoulder-lift, elbow and
wrist-flex zeros are shifted by **−72.3°, +60.0°, +20.0°**; with the 2026-09-07
zeros they scatter by 61 mm. The user confirmed the cause: the hand-held
reference pose of 2026-09-07 had the forearm level but the upper arm raised
and the elbow bent, not the straight horizontal arm the model zero means
(`inspection/reference_pose_hypothesis.png`). A bounded pixel refinement
(camera within 10 mm of the mount, zeros within 6°) settles at **73 px RMS**
over 882 corners, about 1.6 cm at working distance, with the camera 1 cm from
the Menagerie position and pitched 5° from its orientation; the board lands
at (−0.010, 0.269, 0.017), where the phone lay. The "board rotated 180°"
conclusion of 2026-09-07 was an artifact of the wrong arm shape and is
withdrawn: the board is mounted as the model assumes.

**Photo anchor (same day).** The user then supplied a side photo of the arm
at rest: upper arm horizontal pointing backward, forearm horizontal pointing
forward stacked on it, gripper folded down. Under the board-only zeros the
sim's upper-arm bar was tilted 21°, and the board data is indifferent along
the shoulder/elbow trade-off (scatter 19 mm at −72/+60 vs 23 mm at −93/+81;
pixel RMS 72 vs 78). The photo breaks the tie: zeros **−93.25°, +81.0°,
+20.0°** vs the 2026-09-07 map make both printed bars flat (the bar-to-axis
offsets of 14° and 2° are accounted for from the mesh) and put the board at
table height where the board-only fit had it floating 27 mm up. Extra wrist
flex beyond +20° worsens the board fit, so the gripper keeps +20°.

Outputs: joint map `measured_20260908b` (`physical_joint_map_20260908b.json`,
same scale and signs as 20260907, three zeros corrected; `measured_20260908`
kept as the board-only intermediate; rest pose now lift −178.6°, elbow
173.5°, wrist flex 61.6°: upper arm flat backward over the base, forearm flat
forward on top of it, gripper pitched down), `BenchConfig.camera_*` written
into the scene's `wrist_camera` by `tools/prepare_bench_scene.py`, hand-eye
record `camera_calibration/zero_and_camera_refine.json`, and the comparison
`inspection/calibrated_rest_compare.png` (physical rest frame vs the calibrated
sim view at the same joints) and `inspection/calibrated_rest_side_views.png`
(sim arm at the rest joints, to hold against the user's side photo, to be
filed as `readme-assets/bench-rest-side-20260908.jpg`). Teacher re-certified
on the corrected arm: 15/15, deterministic, zero safety invalidation (scene
`6477c4bd…`).

Remaining residual (~1.6 cm) comes from the intrinsics' limited coverage, the
±1 unit joint readings of hand-held poses, and the real napkin being a folded
paper towel larger than the 50.8 mm square in the model.

## Camera and viewing pose

H90 inference sees a fresh image every 90 actions. The shared observation
prefix moves to a verified viewing pose for 60 actions, holds to action 90,
then starts cube-conditioned behavior. Preserve the total 480-action budget
and final hold tail; any resulting safety or success failure blocks capture.
Reset, viewing, approach and grasp frames must be inspected before the large
dataset is captured. A successful physics simulation does not validate the
real camera mount.

The viewing pose is only a **candidate**: the prepared reset with the elbow
changed to 2.8 rad (`viewing_qpos` in the active config). It has not been
verified as a safe, useful view, and moving the physical arm to it needs
its own reviewed motion scope and on-site confirmation.

### Corrected-distance renders and the measured camera mismatch (2026-09-06)

`tools/render_bench_views.py` renders the sim wrist camera (and the side
camera) at the recorded reset and the candidate viewing pose at 1920×1080,
writes side-by-side and 50 % blend composites against the physical frame,
and a `render_manifest_512f897d.json` with every SHA-256. Files carry the
scene-hash tag `512f897d`; the older `sim_prepared_wrist.png` and
`sim_viewing_wrist.png` are superseded.

**The reset-pose comparison does not match, and the gap is in the camera
model, not the bench geometry.** In `physical_prepared_wrist.png` the white
square (a folded paper napkin) spans 574 px wide with its top edge cropped at
the frame top, centred at pixel (952, 184), and the two jaw fingers fill the
bottom third of the frame. Projecting the sim square through the sim wrist
camera at the same reset pose (`compare_reset_sim_projection_over_physical_512f897d.png`,
green outline) gives a 139 px wide quad centred at (955, 482): **4.1× smaller
and about 300 px lower**, with a different in-plane rotation. Horizontal
centring agrees, so the base-to-square line is right; scale and pitch are not.

Two effects explain the size of the gap. The sim `wrist_camera` inherits the
Menagerie mount (`fovy=72`, 68 mm above the gripper body origin). A 1920×1080
webcam with a typical ~60° horizontal field of view has a *vertical* field of
about 36°, which alone halves the apparent size. The remaining ~2× means the
physical camera sits closer to the square, i.e. the 3D-printed mount
(`3d-printing/oak-d-lite-mount`) places the camera further forward and pitched
further down than the Menagerie mount; the large fingers in the physical
frame agree. Neither number is measured yet.

**The joint-map correction (see the Joint map section) changed the sim arm
pose; the comparison was redone on 2026-09-07** with
`tools/compare_bench_camera.py` (`camera_comparison_f198fce2.json`,
`compare_reset_sim_projection_over_physical_f198fce2.png`). At the corrected
reset the sim wrist camera sits 24 cm above the table, 6 cm beyond the
square centre, looking down steeply, and projects the square 218 px wide
centred at (1019, 1126), i.e. **below the bottom edge of the frame**, while
the physical frame shows it 574 px wide at the top (952, 184). Horizontal
centring still agrees; the size ratio fell from 4.1× to 2.6×; the pitch
disagreement is now the other way. So the arm pose is right and the
remaining gap is the camera model: field of view and mount pose.

Separately, the physical camera mount differs from the Menagerie mount: in
the setup photo the camera board sits beside the jaws along the jaw-opening
axis, whereas the model camera sits perpendicular to it (68 mm "above" the
gripper body). No roll value reconciles both the jaw orientation and the
camera side, so the camera pose fit (gate 1b) remains a separate step.

What gate 1 now requires: (a) the physical camera's field of view, from the
module datasheet or one photograph of a ruler at a measured distance;
(b) three to five physical wrist frames at read-only recorded arm poses with
the square and cube in view; (c) fit the sim camera pose in the gripper frame
plus `fovy` to the detected square corners and jaw-tip features, regenerate
the scene (a camera-only change, versioned through the scene hash), re-render,
and only then write the review record. Training on the current renders would
teach the policy a viewpoint the real camera never produces.

Inspection images under `artifacts/so_arm101_v2/bench_pick_replace_v1/inspection/`:

| File | Meaning |
| --- | --- |
| `physical_prepared_wrist.png` | Good focused physical frame after elbow preparation; cube near upper center, partly cropped |
| `sim_prepared_wrist.png`, `sim_viewing_wrist.png` | **Superseded** (256×256, pre-correction). Use the `*_512f897d.png` renders |
| `sim_{reset,viewing}_{wrist_camera,camera_side}_512f897d.png`, `compare_*_512f897d.png`, `render_manifest_512f897d.json` | Corrected-distance renders and comparison composites, 2026-09-06; the `*_BRIGHTENED_review_only.png` is a gamma-lifted copy for viewing, never for training |
| `compare_reset_sim_projection_over_physical_512f897d.png` | Red: detected physical square; green: sim square projected at the same pose; cyan: sim cube top. The 4.1× mismatch evidence |
| `physical_reset_wrist.png` | Corrupt raw frame, not evidence |
| `physical_reset_wrist_mjpg.png` | Hand occlusion, not evidence |
| `physical_reset_wrist_focused.png` | Not described in the handoff; treat as inspection only, not verification evidence |
| `diagnostic_unsettled_*` | Out-of-model natural-rest FK diagnostics, not accepted reset images |

## Active scene and teacher

Active model: `simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml`,
generated by `tools/prepare_bench_scene.py` from the active
`bench_config.json` (black XYZ visual mesh, separate box collision, white
square, black floor and arm). Teacher tuning lives in the bench config so it
is hashed into every capture identity: **grasp pad 1, approach pitch 5°
(near vertical), depth lead 0 mm, grasp height offset 3 mm, joint map
`measured_20260907`** (active since 2026-09-07). The `BenchConfig` code
defaults differ (50°, pad 1, +6 mm, +8.5 mm); the scene builder therefore
requires `--bench-config` and must never be run with bare defaults.

### Why the grasp moved from pad 4 to pad 1, and the detector fix (2026-09-06)

The 25 mm legacy grasp closed the pad-4 pocket from a 25.2 mm gap at joint
angle 0. Measured on the bench scene, the jaw pads pinch a 20 mm cube at these
joint angles (gripper qpos, rad): pad 4 at −0.152, pad 3 at −0.063, pad 2 at
−0.003, pad 1 at +0.055. The strict detector's jaw-axis reference used to be
the line between the `fixed_jaw_tip` and `moving_jaw_tip` sites, which
carries an 18.6 mm vertical offset, so that line sat 25.4° off the pad normal
at joint angle 0 and rotated further as the jaw closed (27.3° at −0.06, 30.7°
at −0.15) against a cos 25° = 0.906 threshold. A parallel face pinch at pads 2
to 4 was therefore rejected regardless of contact quality, and the pad-4
teacher stalled at qpos −0.067 when moving pad 3 reached the cube first,
lifting it by edge contacts (the "jaw-axis alignment 0.888" symptom). The
legacy 25 mm grasp had passed that reference by 0.002.

**Detector fix (user-approved, `GRASP_DETECTOR_VERSION = pad_normals_v2`).**
`jaw_closing_axis` now returns the bisector of the fixed and moving pad-4
inward normals, the direction the jaws actually close along: exactly the pad
normal for parallel pads and half the moving-jaw tilt otherwise (at most 4.9°
at the closed limit). The legacy `tip_sites` mode is kept only to reproduce
historical measurements. Legacy check on `fixed_pick_place_v3` (25 mm, five
scenarios): strict-frame counts identical under both references (48/63/48/
50/47), all rollout fields identical, gate still proven and deterministic; the
tip-site reference's minimum cosine there was 0.908, the new reference's 1.0.
The comparison report is under
`artifacts/so_arm101_v2/simulation/detector_pad_axis_check/`. The version tag
is folded into bench capture identity and the certification directory key so
evidence from the old detector cannot be mistaken for current certification;
the earlier `512f897da336467a/` tree is that superseded evidence.

The teacher stays at pad 1 on its own merits: the cube pinches at +0.055 with
0.23 rad of closing travel to spare and a 3° moving-pad tilt, whereas pad 4
would pinch at −0.152, only 0.02 rad from the mechanical limit.

### Teacher scan and certification

Scan over pad ∈ {1}, pitch ∈ {78, 65, 50}, height offset ∈ {−4…+4 mm},
depth lead ∈ {−3, 0, +3 mm}, run on the nominal pose and then on all five
certification poses (`tools/bench_grasp_scan.py` records the older
pad-4 depth scan; the pad-station scans ran from scratch scripts and are
summarised here):

| Config (pad 1, 78°) | 5-pose passes | Min strict frames | Safety | Corner rejections |
| --- | ---: | ---: | ---: | ---: |
| offset 0, lead 0 (**chosen**) | 5/5 | 151 | 0 | 0 |
| offset −4…+2 mm, lead 0 | 5/5 each | 151 | 0 | 0 |
| offset +4 mm, lead 0 | 5/5 | 99 | 0 | 151 to 331 |
| lead −3 mm | 5/5 | 81 to 99 | 0 | 63 to 154 |
| lead +3 mm | 0 to 4/5 | 0 | 0 | IK failure or edge contacts |

Pitch 65° also passed 5/5 at lead 0 but with pad-edge rejections; 78° was
kept. Certification (`tools/certify_bench_teacher.py`): nominal plus four
±10 mm axis offsets, three repeats each, through the unchanged preflight
machinery → **15/15 successes, deterministic, `environment_proven: true`**,
scene dependencies `512f897da336467ac2b83de5e4b5adc641db350627c776511bfdff6caffa5375`,
detector `pad_normals_v2` (first passed under the tip-site detector the same
day with the same 15/15; re-certified after the fix).
A nominal regression pin lives in `tests/test_bench_contract.py`.

### Retune under the physical joint ranges (2026-09-07)

With the measured joint map the 78° near-horizontal approach became
infeasible: the pre-grasp point is reachable in position (4 mm) but the wrist
cannot bring the gripper back to horizontal at that reach within ±95°, and
the shoulder cannot lean past +10°. The legacy approach only worked because
the wrong map let the sim fold the shoulder to −180°. `tools/bench_teacher_scan.py`
over pitch {0, 5, 8, 10, 12, 15}° × height offset {0…8 mm} on the five
certification poses:

| pitch | offset 0 | 2 mm | 4 mm | 6 mm | 8 mm |
| ---: | --- | --- | --- | --- | --- |
| 0° | 5/5, 755 corner rej. | **5/5 clean** | **5/5 clean** | 4/5 | 0/5 |
| 5° | 4/5 | **5/5 clean** | **5/5 clean** | 5/5, corner rej. | 0/5 |
| 8° to 15° | IK failures on the y−10 mm pose or all poses | | | | |

Chosen: pitch 5°, offset 3 mm (centre of the clean window). Certification:
15/15, deterministic, zero safety invalidation, 151 strict frames per
episode, scene dependencies
`f198fce23f1920008910f92e80f2bd1f9bd06d2b680a491fbe7f651c763c59e8`.

Stage boundaries (actions) are unchanged:
`(0,60,90,135,160,185,205,225,260,280,299,334,369,374,399,419,424,450)`.
The full task completes at about action 431.

The earlier pad-4 preflight report at
`artifacts/so_arm101_v2/bench_pick_replace_v1/teacher_diagnostics/9c4cff24f78b/preflight/bench_pick_replace_v1_diagnostic_59b6ba96b593/evaluation.json`
used the OLD incorrect distance and older code. It is superseded.

## Software inventory (reviewed 2026-09-06)

- `src/so_arm101_v2/contracts/bench.py`: versioned bench reset and
  geometry, shoulder-floor intersection, strict reset rejection, dependency
  hashing; geometry measured from the base front edge.
- `contracts/physical.py`: shoulder floor and post-integer-tick conversion
  checks without renormalization.
- `contracts/physical_io.py`: read-only bus connect/disconnect, finite and
  fresh measured observations (100 ms read deadline).
- `contracts/task.py`, `contracts/pick_place.py`: bench pickup and
  full-task contracts; starting on the destination cannot succeed without
  grasp, lift and hold first.
- `simulation/adapter.py`, `simulation/contact.py`: derived 20 mm cube
  geometry, bench safety intersection and refusal; legacy scene behavior
  preserved conditionally. `contact.py` jaw axis = pad-normal bisector
  (`pad_normals_v2`); `tip_sites` legacy mode retained for reproduction.
- `simulation/bench.py`: seeded suite construction and teacher screening
  with rejection accounting.
- `simulation/oracle.py`, `simulation/rollout.py`: new task support and
  bench/dependency provenance.
- `learning/vision.py`: optional atomic scratch checkpoints holding model,
  optimizer, RNG, step, and pending minibatch indices; interrupted versus
  uninterrupted training is bitwise equivalent with prefetch on and off;
  final artifacts stay immutable. Review fix: resume loads weights into the
  unwrapped module, so it also works under the default torch.compile
  regime (the equivalence test runs eager only).
- `tools/read_bench_pose.py`, `tools/prepare_physical_elbow.py`,
  `tools/prepare_bench_scene.py`, `tools/bench_grasp_scan.py`,
  `tools/certify_bench_teacher.py`: described above.
- `tools/replay_physical_trajectory.py`: bench config required on hardware
  paths, first-target check, fresh feedback before each command, configured
  relative limit, measured/requested/sent logging, abort on refusal instead
  of skipping, read-only preflight. Review fixes: the exclusive log is
  claimed and the live calibration checked against the pinned contract
  before the bus opens; a failed connection removes the unused log; default
  port is now `/dev/ttyACM0`. Ordering is pinned by
  `tests/test_physical_replay.py`.
- `tools/run_bench_pipeline.py`: draft gate chain, local progress and
  telemetry, online W&B initialization, single vision recipe, evaluations.
  Arguments `--model`, `--verification`, `--output-dir`, optional
  `--stop-after preflight|screen|capture|train`, and `--rehearsal`. The
  verification file must be an artifact-backed camera review record (see
  Run gates). W&B is initialised first (120 s init timeout) so a tracking
  problem fails before hours of screening; every phase, throughput, elapsed
  time, checkpoint age and tracking state go to W&B and `progress.json`;
  `provenance.json` records suite ids, preflight reports and the capture
  manifest digest; the evaluation summary reports success, safety frames
  and observation-prefix success (arm within 0.05 rad of the viewing pose
  at the first image refresh; the certified teacher scores 15/15).
  **Rehearsed end to end 2026-09-06** in `--rehearsal` mode (camera gate
  skipped and labelled, 3+2 screened poses, 30 training steps, W&B
  offline, output forced under `rehearsal/`): certification → screening →
  preflights → capture with frames → training with scratch checkpoints →
  nominal and held-out evaluation with black-image ablation all ran;
  `artifacts/so_arm101_v2/bench_pick_replace_v1/rehearsal/20260906_200755/`.
  Rehearsal numbers are meaningless by construction. The real run still
  refuses to start without the review record.
- `simulation_code/queue_bench_pipeline.sh`: enqueues the pipeline (or a
  rehearsal) on the persistent `tsp` queue with a label and a queue log;
  the pipeline's internal gate order is the dependency chain.
- `tools/bench_health_check.py`: read-only health report from an experiment
  directory (process alive, phase, step advance, finite loss and trend,
  throughput vs the run's post-warmup median, GPU memory, checkpoint age,
  W&B state, ETA); exit 0 healthy, 1 warning, 2 failed/complete/action.
- `tools/render_bench_views.py`: the camera-review renders above.
- `contracts/joint_map.py` + `data/resources/physical_joint_map_20260907.json`:
  versioned physical-to-MuJoCo joint maps; `tools/read_joint_reference.py`
  records read-only encoder references; `tools/bench_teacher_scan.py`
  is the durable teacher scan; `tools/compare_bench_camera.py` the
  pixel-space camera comparison.
- `tests/test_bench_contract.py`: shoulder and tick boundaries, stale and
  nonfinite observations, read-only I/O, reset rejection, geometry and
  provenance, already-on-square prevention, elbow staging, exact resume,
  front-edge distance, pinned-calibration drift, verification record.
- `tools/prepare_bench_scene.py` now requires `--bench-config`, so the
  active scene can no longer be regenerated from bare defaults by accident.
  `contracts/physical_io.py` gained the shared `assert_pinned_calibration`
  used by the replay and elbow tools.

## Run gates

1. Verified prepared reset and matched physical/simulated wrist view,
   recorded as a verification JSON next to the images: the active
   `scene_dependencies_sha256` and `reset_evidence_sha256`, plus a
   `camera_review` object naming `physical_wrist_image` and
   `simulated_wrist_image` (paths relative to the record), their SHA-256
   digests, `reviewer`, ISO `reviewed_at`, and `notes`. The pipeline
   refuses anything less; there is no boolean to set.
2. Oracle certification: nominal plus four ±10 mm axis offsets, three
   repeats each, 15/15 complete episodes with zero safety invalidation. If
   the full task cannot fit safely within 480 actions, stop before data
   generation and report the failed gate.
3. Teacher-screen 400 training poses (seed 12) and 10 held-out poses (seed 8,
   three repeats each), independent uniform XY offsets within ±10 mm of the
   square center, fixed square and initial cube orientation. Record
   rejections and stop if coverage cannot be filled. Capture fresh frames
   and labels; no reuse of old data. Provenance hashes cover task, scene
   with dependencies, reset, safety config, teacher config and datasets.
4. Exactly one vision run: seed 202, H90, hidden width 256, batch 64, Adam
   1e−3, cosine_floor_v1, 120,000 steps, existing gripper clamp, pinned
   numerics, deterministic prefetch. No extra seeds or architecture search.
   Atomic scratch checkpoints every 5,000 optimizer steps must resume
   exactly, including prefetched minibatch order.
5. Nominal and held-out closed-loop evaluation with videos and black-image
   ablation; report task success, safety, and observation-prefix success.
   A single run is exploratory, not a promotion claim.

Full physical testing, additional seeds, and arbitrary workspace placement
are later work.

## Implementation and review items (all pre-launch items closed 2026-09-08)

1. *Done 2026-09-08.* Camera review record: the user confirmed
   `inspection/calibrated_rest_compare.png` ("basically matches"; the real
   frame crops the top of the cube and napkin that the sim shows), recorded in
   `inspection/camera_review_20260908.json`, which `load_bench_verification`
   accepts. Physical frames must be undistorted at inference with the
   calibrated coefficients. The rest-pose side photo is to be filed as
   `readme-assets/bench-rest-side-20260908.jpg` by the user.
2. *Done 2026-09-06.* Teacher retuned to the pad-1 tip station; the common
   prefix is unchanged and no privileged cube coordinates reach the
   learned policy.
3. *Done 2026-09-06.* Gate 2 certified 15/15 with zero safety invalidation
   (`tools/certify_bench_teacher.py`).
4. *Done 2026-09-06.* Replay code review: log claimed and calibration
   checked before hardware contact, unused log removed on failed
   connection, measured/requested/sent distinction and the actual relative
   limit preserved, finite and stale feedback refused. LeRobot's integer
   tick truncation was confirmed against the driver source, so the
   shoulder-floor tick check matches what goes on the wire.
5. *Done 2026-09-06.* Artifact-backed review record required;
   `provenance.json` and prefix-success reporting added; rehearsed.
6. *Done 2026-09-06.* `simulation_code/queue_bench_pipeline.sh` enqueues on
   `tsp`; gate order is enforced inside the pipeline, so a failed gate
   raises before capture, training or evaluation.
7. *Done 2026-09-06 at rehearsal scale.* Elapsed time, throughput,
   checkpoint age and tracking state logged locally and to W&B; W&B is
   probed first with a 120 s timeout. Still unexercised: online W&B on this
   machine, and resume at the real 38 GB frame scale (the equivalence test
   covers the mechanism, not the scale).
8. Re-run tests after code changes and proceed only through passing gates.
9. **Physical preparation before any episode (open):** the gravity-rest
   shoulder reads −92.08, below the −92 floor, so a reviewed shoulder lift
   of a few units is needed to enter the contract; the elbow no longer
   needs staging under `measured_20260908b`. Also verify the shoulder-pan
   sign physically (one small hand rotation, read-only). Do not rewrite
   legacy results or decisions to imply they apply to this task.
10. **Deployment path (open):** undistort real frames with the calibrated
    intrinsics before the policy; implement and test this in the physical
    inference/replay path before the first learned-policy trial.
11. **After the run (open):** read `evaluation_summary.json` (successes,
    safety frames, prefix success, black-image ablation) on the nominal and
    held-out suites; a single seed is exploratory. If a promotion claim is
    wanted, pre-register a gate first. Next levers if it under-performs:
    more data (the engine is proven) or more compute per seed; the
    calibration residual (~1.6 cm at working distance) is a known
    sim-to-real gap.

## Tracking and monitoring

**Live run (2026-09-08):** W&B project `so-arm101-v2-scaling`, run
`bench-pick-replace-v1-s202-120k` id `tinmahze`,
https://wandb.ai/7adamyasingh-rutgers-university/so-arm101-v2-scaling/runs/tinmahze;
queue `tsp` job 0 (`tsp -l`; stdout in the file `tsp -l` names); output
`artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/seed202_120k_20260908/`.
Monitoring in place: `watch_first_step.sh` (detached, pid in
`watch_first_step.log`) writes `health_20min.txt` twenty minutes after
`training_clock.json` appears; the assistant session's scheduled check runs
`tools/bench_health_check.py` at :23 and :53 and reports once after the
20-minute check, then only stall/failure/completion. Delete the schedule
when the run finishes.

The pipeline writes `training_clock.json`, `training.jsonl`,
`progress.json` and `wandb.json` to its experiment output directory.

The first health check is due **20 minutes after the first optimizer
step**, not after preparation begins: process alive, steps advancing,
finite loss and trend, throughput against the same run's post-warmup
baseline, GPU memory, checkpoint freshness, W&B sync, estimated remaining
time. Send one early report with the run link, then notify only on stall,
failure, completion, or required action, and disable the watcher after the
final report. No automatic recipe changes and no competence inference from
loss. The check is implemented as `tools/bench_health_check.py`; schedule it (or
run it by hand) once `training_clock.json` appears. The user asked for an
assistant-thread heartbeat; if the assistant cannot create that automation
natively, say so and arrange monitoring explicitly rather than claiming a
watcher exists.

## Resume commands

From the Git root; none of these actuates hardware.

```bash
cd /home/win10ubuntu/dev/robotic-arm/SO-ARM-101
git status --short

# Regenerate only after deliberately reviewing/editing the active config.
PYTHONNOUSERSITE=1 /home/win10ubuntu/miniforge3/envs/lerobot/bin/python \
  tools/prepare_bench_scene.py \
  --bench-config simulation_code/model/bench_pick_replace_v1/bench_config.json

# Reproduce the bounded simulation-only depth-lead diagnostic.
PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  /home/win10ubuntu/miniforge3/envs/lerobot/bin/python tools/bench_grasp_scan.py

# Full test suite (259 passed on 2026-09-08 at launch).
PYTHONNOUSERSITE=1 /home/win10ubuntu/miniforge3/envs/lerobot/bin/python -m pytest -q tests

# Run status (read-only).
tsp -l
PYTHONNOUSERSITE=1 /home/win10ubuntu/miniforge3/envs/lerobot/bin/python tools/bench_health_check.py \
  --experiment-dir artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/seed202_120k_20260908

# Sim/real comparison at the rest pose (needs the camera on /dev/video0).
PYTHONNOUSERSITE=1 MUJOCO_GL=egl /home/win10ubuntu/miniforge3/envs/lerobot/bin/python \
  tools/compare_bench_camera.py --reference artifacts/so_arm101_v2/bench_pick_replace_v1/camera_references/<latest>.json
```
