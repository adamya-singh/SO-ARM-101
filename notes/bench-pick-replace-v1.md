# Bench lift-and-replace, September 2026

This is the governing record for the active `bench_pick_replace_v1`
experiment: physical setup, recorded hardware state, active scene and
teacher, evidence inventory, gates, open work, and resume commands. It
absorbed the 2026-09-06 assistant handoff note, which has been deleted.

## Status (2026-09-06)

Teacher certified in simulation; camera unverified. **No fresh dataset capture,
training run, W&B run or URL, or scheduled watcher exists.** The September
software and its review fixes are committed as one tranche on `master`
(after `640aa76`, 2026-08-08), with the documentation in a second commit.

- Targeted regression set: **54 passed** (command under Resume commands).
  This covers software contracts and exact resume; it does not certify the
  teacher, the camera view, or any physical execution.
- **Teacher certification passed 2026-09-06 (gate 2): 15/15 complete
  lift-and-replace episodes, deterministic, zero safety invalidation**, with
  the grasp moved to the pad-1 tip station (see Active scene and teacher for
  why pads 2 to 4 cannot register a strict grasp on a 20 mm cube). Report:
  `artifacts/so_arm101_v2/bench_pick_replace_v1/teacher_certification/512f897da336467a/preflight/bench_pick_replace_v1_certification_e920b408fb5a/evaluation.json`.
  The success criterion was not changed.
- Camera and viewing pose are unverified. No camera-review approval
  artifact exists.
- The August results elsewhere in this repository concern the legacy fixed
  25 mm cube task and establish nothing about bench readiness.

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

**Last observed arm state (2026-09-06): elbow torque was deliberately left
ENABLED to hold the prepared pose.** No other joint was commanded or
enabled by the preparation tool. This is a past observation, not a fresh
measurement. Do not disable torque without supporting the arm, and do not
rerun motion merely to resume work. Joint order: shoulder_pan,
shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.

## Rest pose and elbow preparation

The observed gravity-rest elbow is about 99.73 calibrated units, mapping to
3.256 rad, beyond the Menagerie model's 3.14-rad elbow maximum. This is not a
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

Inspection images under `artifacts/so_arm101_v2/bench_pick_replace_v1/inspection/`:

| File | Meaning |
| --- | --- |
| `physical_prepared_wrist.png` | Good focused physical frame after elbow preparation; cube near upper center, partly cropped |
| `sim_prepared_wrist.png`, `sim_viewing_wrist.png` | **Stale**, rendered before the front-edge correction. Regenerate and compare; never mark these as aligned |
| `physical_reset_wrist.png` | Corrupt raw frame, not evidence |
| `physical_reset_wrist_mjpg.png` | Hand occlusion, not evidence |
| `physical_reset_wrist_focused.png` | Not described in the handoff; treat as inspection only, not verification evidence |
| `diagnostic_unsettled_*` | Out-of-model natural-rest FK diagnostics, not accepted reset images |

## Active scene and teacher

Active model: `simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml`,
generated by `tools/prepare_bench_scene.py` from the active
`bench_config.json` (black XYZ visual mesh, separate box collision, white
square, black floor and arm). Teacher tuning lives in the bench config so it
is hashed into every capture identity: **grasp pad 1, approach pitch 78°,
depth lead 0 mm, grasp height offset 0 mm** (active since 2026-09-06). The
`BenchConfig` code defaults differ (50°, pad 1, +6 mm, +8.5 mm); the scene
builder therefore requires `--bench-config` and must never be run with bare
defaults.

### Why the grasp moved from pad 4 to pad 1 (2026-09-06)

The 25 mm legacy grasp closed the pad-4 pocket from a 25.2 mm gap at joint
angle 0. Measured on the bench scene, the jaw pads pinch a 20 mm cube at these
joint angles (gripper qpos, rad): pad 4 at −0.152, pad 3 at −0.063, pad 2 at
−0.003, pad 1 at +0.055. The strict detector's jaw-axis reference is the line
between the `fixed_jaw_tip` and `moving_jaw_tip` sites, which carries an
18.6 mm vertical offset, so that line sits 25.4° off the pad normal at joint
angle 0 and rotates further as the jaw closes (27.3° at −0.06, 30.7° at
−0.15). Its cosine threshold is cos 25° = 0.906. Consequently a parallel face
pinch at pads 2 to 4 is rejected by the unchanged detector regardless of
contact quality, and the earlier pad-4 teacher stalled at qpos −0.067 when
moving pad 3 reached the cube first, lifting it by edge contacts (the
"jaw-axis alignment 0.888" symptom). At pad 1 the cube pinches at +0.055
where the reference is 24.0° off (cosine 0.913), inside the threshold by
about 1°.

The detector was **not** changed. This is a known fragility to record: the
jaw-axis reference measures tip-site geometry, not the pad closing
direction, and it now holds by a 0.007 cosine margin. Whether to replace the
reference with the pad-to-pad closing axis is a detector decision for the
user, separate from teacher tuning.

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
scene dependencies `512f897da336467ac2b83de5e4b5adc641db350627c776511bfdff6caffa5375`.
A nominal regression pin lives in `tests/test_bench_contract.py`.

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
  preserved conditionally.
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
  `--stop-after preflight|capture|train`. The verification file must be an
  artifact-backed camera review record (see Run gates). **Never run
  end-to-end; not launch-ready.** No verification file or experiment output
  directory exists yet.
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

## Open implementation and review items before launch

1. Finish the viewing-pose and reset camera comparison at the corrected
   distance: regenerate the stale sim renders, capture reset, viewing,
   approach and grasp evidence. A rendered image never proves the real
   camera mount.
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
5. *Partly done 2026-09-06.* The pipeline now requires the artifact-backed
   camera review record described under Run gates. Still open:
   task/teacher/dataset provenance review and prefix-success reporting in
   the evaluation summary.
6. Queue orchestration: the draft pipeline orders its gates internally but
   is not wired into the existing persistent job queue with explicit
   dependencies. Inspect the queue tooling before submitting. Failed gates
   must prevent capture, training and evaluation.
7. Telemetry: log elapsed time, throughput, checkpoint age and tracking
   state to W&B and local state. Review W&B connectivity failure behavior,
   actual recipe defaults, numerics and prefetch pinning, and resume at
   real scale. No expensive run has exercised this pipeline.
8. Re-run tests after code changes and proceed only through passing gates.
   Do not rewrite legacy results or decisions to imply they apply to this
   task.

## Tracking and monitoring

W&B project `so-arm101-v2-scaling`, distinct bench group and name. Persist
the actual returned run ID and URL; report a link only after the run has
started. The draft pipeline writes `training_clock.json`, `training.jsonl`,
`progress.json` and `wandb.json` to its experiment output directory.

The first health check is due **20 minutes after the first optimizer
step**, not after preparation begins: process alive, steps advancing,
finite loss and trend, throughput against the same run's post-warmup
baseline, GPU memory, checkpoint freshness, W&B sync, estimated remaining
time. Send one early report with the run link, then notify only on stall,
failure, completion, or required action, and disable the watcher after the
final report. No automatic recipe changes and no competence inference from
loss. The user asked for an assistant-thread heartbeat; if the assistant
cannot create that automation natively, say so and arrange monitoring
explicitly rather than claiming a watcher exists.

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

# Targeted regression set (54 passed on 2026-09-06).
PYTHONNOUSERSITE=1 /home/win10ubuntu/miniforge3/envs/lerobot/bin/python -m pytest -q \
  tests/test_bench_contract.py tests/test_physical_replay.py \
  tests/test_physical_commands.py tests/test_calibration_integrity.py \
  tests/test_task_contract.py tests/test_pick_place_v3.py tests/test_vision_lane.py
```
