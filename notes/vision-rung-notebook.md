# Vision Rung Notebook (running log, exploratory lane)

> **Current state (2026-09-08):** see the last entry, "Bench pick-replace
> v1: first run launched". The September 6 and 7 entries below are the
> trail that led there and contain claims that were later corrected in
> place (camera field of view, joint zeros, the 180° board claim).

## September 6: bench adaptation in progress

The [bench runbook](bench-pick-replace-v1.md) records the complete current plan, evidence and remaining implementation work. New task: black 20 mm XYZ cube, 50.8 mm white square, 8.5 inches from the base **front edge**, shoulder floor −92 calibrated units. Stable prepared physical reset captured; scene corrected to Y=0.2805353 m. Latest software regression set: 54 passed. Nominal teacher lifts ~29 mm but fails strict opposing-face grasp; an explicit stage schedule removed the earlier nominal release limiter event. Camera comparison is still pending. **No new dataset capture, training run, W&B URL or watcher exists.** August results below remain historical; they are not bench certification.

## September 6 (later): bench teacher certified at the pad-1 tip station

The strict-grasp failure was geometric, not tuning. The detector's jaw-axis reference (tip-site line, 18.6 mm vertical offset) is 25.4° off the pad normal at joint angle 0 and rotates further as the jaw closes, so a 20 mm cube pinched at pads 2 to 4 can never satisfy the unchanged 25° criterion; pad 1 pinches at +0.055 rad where the reference is 24.0° off. Teacher moved to pad 1 (78°, zero depth lead, zero height offset, all now versioned in `bench_config.json`). Certification: **15/15 complete lift-and-replace episodes, deterministic, zero safety invalidation**. The user then approved correcting the detector: the jaw axis is now the pad-normal bisector (`pad_normals_v2`, half the moving-jaw tilt at most), legacy `tip_sites` mode kept for reproduction. Legacy 25 mm gate re-run: identical strict-frame counts and rollout fields, still proven and deterministic. Bench teacher re-certified 15/15 under the new detector; it stays at pad 1 for closing-travel margin. Details in the [bench runbook](bench-pick-replace-v1.md). Camera comparison still pending; still no dataset, training run, or W&B run.

## September 6 (evening): pipeline rehearsed; camera mismatch measured

The bench pipeline ran end to end in a labelled `--rehearsal` mode (toy scale, camera gate skipped and recorded as skipped, W&B offline). Corrected-distance wrist renders were produced and compared with the physical reset frame: the sim square projects **4.1× smaller and ~300 px lower** than the detected physical square at the same recorded pose, horizontally centred in both. Diagnosis: the sim camera keeps the Menagerie mount (`fovy=72`) while the real webcam likely has ~36° vertical field and a different 3D-printed mount pose. Gate 1 now means calibrating the sim camera model to physical frames, not eyeballing a pair of images. The user power-cycled the arm; elbow prep must be redone before any capture. Still no dataset, training run or W&B run.

## September 8: lens calibrated, joint zeros corrected from a flat checkerboard, camera in the scene

Lens: 44.0° vertical field of view, strong barrel distortion, 1.9 px RMS from 38 phone-checkerboard views (the sim had 72°). Hand-eye on 9 flat-board frames with simultaneous joint readings: rotations matched the encoders to 2.5° but heights were off by up to 10 cm, and the nine derived board poses agree (21 mm) only with the shoulder-lift/elbow/wrist-flex zeros shifted by −72.3°/+60.0°/+20.0°: the 2026-09-07 hand-held reference had the upper arm raised and the elbow bent (user confirmed from a render). Board-only zeros gave `measured_20260908`; the user's side photo of the rest pose (both bars flat) then anchored the shoulder/elbow pair the board could not separate: `measured_20260908b` (−93.25/+81.0/+20.0 vs 09-07), rest = upper arm flat backward, forearm flat forward, gripper down; camera = official mount ±1 cm, ~5° tweak, fovy 44; 78 px RMS. The 'board rotated 180°' claim is withdrawn. Teacher re-certified 15/15 on the corrected arm. Gate 1 now waits on the user's sign-off of `inspection/calibrated_rest_compare.png`.

## September 7: joint map corrected from encoder references; teacher re-certified top-down

The user spotted in the live viewer that the sim wrist roll was a quarter turn off; two read-only encoder recordings (rest, and the model's all-zero pose held by hand) showed the June affine joint map had shoulder-lift, elbow and wrist-roll zeros each ~90° off and spans off by up to 37 %. New versioned map `measured_20260907` (4096 ticks/turn, reference zeros); legacy lane keeps `legacy_affine_v1`. Under real joint ranges the 78° approach is infeasible (the legacy sim folded the shoulder to −180°, impossible physically), so the teacher moved to a 5° near-vertical approach with a 3 mm height offset: **15/15 certified, deterministic, zero safety**. Camera comparison redone on the corrected arm: sim square now projects off the bottom of the frame vs the top physically (2.6× size ratio) — the mount pose and field of view remain the gate-1b measurement. Still no dataset, training run or W&B run.

## Methodology for this rung (recorded 2026-08-06)

The vision rung runs in **exploratory mode**, a deliberate recalibration now
that trainings cost minutes instead of hours:

- **Still on, automatically:** content-addressed artifacts, immutable
  writes, numerics-regime fingerprints, the mujoco 3.9.0 pin, deterministic
  seeds. Nothing in this lane can silently corrupt or overwrite evidence.
- **Relaxed:** individual experiments need no pre-registered proposal. Look,
  tweak, rerun. Negative results get a notebook entry, not an addendum.
- **Unchanged for promotions:** any claim that advances the ladder (e.g.
  "the vision policy passes Stage A robustly") requires a pre-registered
  gate with a prediction on record, exactly like every prior rung.

Rationale: the pre-registration ceremony was calibrated to 9-hour runs where
a silent mistake cost days. At ~minutes per experiment the binding risk is
silent corruption (still guarded) rather than wasted compute, and the vision
rung's early phase needs iteration speed more than adjudication.

## Baseline context

State-policy baseline (privileged inputs, horizon-aligned 480 capture,
cosine recipe): 9/15, 9/15, 12/15 across seeds 101/202/303, failures =
gripper floor overshoot (see `notes/gripper-clamp-proposal.md` for the
in-flight fix). The vision policy replaces the privileged `cube_position[3]`
(and its capture-statistics normalization) with the wrist camera frame;
inputs are pixels + `normalize_act(current_act)[6]` + the open-loop progress
clock. Expectation for v0, recorded loosely: **worse than the state policy**;
the purpose is to measure the gap, verify the pipeline end to end, and check
via the black-image ablation that the policy actually uses pixels (the
teacher is open-loop, so image-conditioned BC learns pose-from-view — the
ablation distinguishes that from memorizing the clock).

## Experiments

### v0 (2026-08-06): pipeline verified; policy is bad in the expected way

Capture `oracle/fixed_pick_place_v3/691a1c59a5ffef65` (2,400 rows + 450 MB
frames sidecar; scalar npz byte-identical to the frameless capture — frame
storage provably did not perturb physics). Recipe: conv trunk (512-d) +
[current_act, progress] state branch, zero-init H90 head, minibatch 64,
20k steps, cosine LR, regime-v2 GPU. Evals: 15-rollout Stage A.

| Run | Successes | Safety frames | Failure mix |
| --- | ---: | ---: | --- |
| seed101 | 0/15 | 259 | 9 safety_invalidation, 6 pickup_incomplete |
| seed202 | 0/15 | 441 | 15 safety_invalidation |
| seed303 | 3/15 | 198 | 12 safety_invalidation |
| seed101 **black-image** | 0/15 | 303 | 15 safety_invalidation |

Reading: v0 is far below the state-policy baseline (9-12/15) — expected and
fine; the rung's pipeline (frames capture → minibatch training → live-frame
closed-loop eval) works end to end. The black-image ablation shows *partial*
pixel dependence: with pixels the failure mix changes qualitatively
(pickup_incomplete appears; 259 vs 303 frames), but the gap is small — v0
leans heavily on the progress clock. Obvious v1 levers, in order: (1) carry
the validated **gripper output clamp** into `VisionChunkedPolicy` (much of
the violation mass is presumably the same floor overshoot the state lane
had); (2) more steps / larger batch; (3) the randomized-capture data engine
(more visual diversity should force pixel reliance). No gate, no claim.


## Randomization engine (2026-08-06): screening hardened twice

The data engine's admission rule evolved through two real failures:

1. **Unscreened sampling** (rnd2): sampled poses just outside IK tolerance —
   preflight aborted with `privileged IK failed: position=0.002184m`.
   Fix: screen candidates with the teacher at generation time.
2. **IK-only screening** (rnd3): every plan solved, but preflight still
   returned `environment_proven=false` — 4/25 train and 2/10 eval scenarios
   failed the *execution* (scattered mid-region poses: 2 safety_invalidation
   with 1 unsafe-contact frame each, 4 pickup_incomplete). Plan solvability
   is not execution success.
3. **Full-episode screening** (rnd4, current): the generator now runs the
   complete 480-action privileged rollout per candidate and admits only
   poses the teacher executes to contract success with zero safety events —
   exactly what preflight + capture demand. Suite ids are content-distinct
   (scenario-hash suffix), so each screening regime lives in its own
   immutable namespace: train `random_pick_place_v3_seed7_n25_2b99d8ab`
   (suites/e7663bab273d7514), held-out eval
   `random_pick_place_v3_seed8_n10_9d8c330a` (suites/10e6a56077e91ce4).

Consequence worth stating: the engine's distribution is now "poses the
teacher can solve", not "poses in the region" — generalization numbers are
conditioned on teacher competence, and region edges the teacher can't handle
are invisible to the student. Fine for v1; revisit if we ever need coverage
beyond the teacher.

### randomized v1 (2026-08-06): first held-out generalization numbers

Loop: capture 25 full-episode-screened random poses with frames
(`oracle/random_pick_place_v3_seed7_n25_2b99d8ab/4f4b6f9171270e44`, 12,000
rows, zero skips) → train both families on it → evaluate on the held-out
10-scenario × 3-repeat suite (`…seed8_n10_9d8c330a`, different generator
seed). Evaluations under `randomized_v1/randomized_explorations/`
(state+clamp `fd18a476910d7732`, vision `6ecb2a418dfe405e`).

| Policy | Held-out successes | Safety frames | Per-scenario shape |
| --- | ---: | ---: | --- |
| state+clamp (w512/90k, mse 5.9e-07) | **18/30** | 57 | perfectly bimodal: 6 scenarios 3/3 with **zero** safety frames, 4 scenarios 0/3 deterministic |
| vision (20k steps, mse 1.7e-05) | 0/30 | 848 | 7 scenarios safety_invalidation, 3 pickup_incomplete |

State+clamp failure anatomy (each mode distinct, each deterministic across
repeats): `random_003` (x+0.013 y0.256) pickup_incomplete with zero safety
frames; `random_007` (y0.289) 4 limiting_frames; `random_008` (y0.249)
3 clipping_frames — clipping, i.e. a **non-gripper** channel, outside the
clamp's scope; `random_009` (y0.245, nearest pose) 12 unsafe_contact_frames.
The two nearest-y poses (0.245/0.249) both fail — the training region's
y∈[0.24,0.31] near-edge is under-covered by 25 poses.

Reading: the state policy genuinely interpolates across cube poses (60% on
poses it never saw, and *when it succeeds it is perfectly clean*), so the
randomized data engine works end to end. The gripper clamp does not cover
out-of-distribution failure modes (limiting/clipping/contact are all
non-gripper-floor mechanisms). Vision at v1 scale (20k steps, 25 episodes)
is still far from lift-off — consistent with v0; pixels need much more data
and the clamp carried into `VisionChunkedPolicy`. Obvious v2 levers:
(1) scale capture to 100+ episodes (engine is proven, ~25 min/25 eps),
(2) clamp in the vision policy, (3) denser near-edge sampling.

### Data-scaling curve (2026-08-07): coverage solves state; vision needs compute scaled with data

One sweep (`simulation_code/queue_scaling_curve_20260807.sh` + n400 retry),
~15 h: per tier, generate a full-episode-screened suite (seeds 9-12),
preflight, capture with frames, train both families, evaluate on the
**frozen held-out benchmark** (`…seed8_n10_9d8c330a`, 30 rollouts —
identical across all tiers, tuned on never). Vision now carries the gripper
clamp (`VisionChunkedPolicy(clamp_channels=(5,))`, id suffix
`.gripper_clamp_v1`). State recipe fixed (w512/90k/cosine); vision fixed at
20k steps except the probe. SWEEP_LOG at
`artifacts/so_arm101_v2/randomized_scaling/SWEEP_LOG.txt`.

| Train episodes | state+clamp | vision+clamp (20k) | safety frames (state / vision) |
| ---: | ---: | ---: | --- |
| 25 (v1, 2026-08-06) | 18/30 | 0/30 (unclamped) | 57 / 848 |
| 50 | 27/30 | 6/30 | **0** / 396 |
| 100 | 27/30 | 14/30 | 9 / 31 |
| 200 | 27/30 | 9/30 | 6 / 9 |
| 400 | **30/30** | 18/30 | **0** / 642 |
| 200 @ 60k steps (probe) | — | **24/30** | — / 51 |

Readings:
1. **State: data coverage was the whole story.** 18 → 27 (plateau through
   50-200, three different single-scenario failures) → **30/30 with zero
   safety frames at 400 episodes** — the first perfect held-out score in
   the project. No recipe change, no new mechanism; just more teacher data.
2. **Vision: steps must scale with data.** At fixed 20k steps the curve is
   non-monotone (0→6→14→9→18) because epochs shrink as data grows — a
   classic compute/data confound. The probe isolates it: same n200 data,
   3× steps → 9/30 becomes **24/30** (train MSE 2.0e-5 → 2.9e-6). The
   fixed-steps column understates vision badly; the recipe rule going
   forward is compute scaled with data (next: 60-120k steps on n400).
3. **Undertrained vision is unsafe vision.** Vision safety frames collapse
   with data at matched epochs (848→396→31→9) but blow back up when
   undertrained (642 at n400/20k). The clamp caps the gripper floor only;
   the rest of safety comes from actually fitting the teacher.
4. Infra: frames sidecars past RAM size broke the publish
   (`MemoryError` at 38 GB); fixed with `write_immutable_file` (streaming
   atomic publish, same conflict semantics; `test_serialization_atomic.py`).

Candidate next moves: (a) vision at scaled compute on n400 — if it
approaches state's 30/30, the deployable-inputs policy is real; (b) the
state+clamp 30/30 is a promotion-shaped claim: if we want it on record, it
gets a pre-registered gate (exploratory numbers stay exploratory);
(c) physical smoke replay unchanged, awaiting bench.

### Vision n400 @120k, 3 seeds (2026-08-08): deployable inputs reach a perfect held-out score

Epoch-matched compute on the n400 capture (120k steps ≈ the 60k/n200
probe's ~40 epochs), seeds 101/202/303, frozen held-out benchmark, clamp
carried. wandb project `so-arm101-v2-scaling`, group
`vision-n400-120k-20260807`.

| Seed | Held-out | Safety frames | Failure mix | Train MSE |
| ---: | ---: | ---: | --- | ---: |
| 101 | 24/30 | **0** | 6 pickup_incomplete | 2.03e-6 |
| 202 | **30/30** | **0** | — | 2.18e-6 |
| 303 | 24/30 | 3 | 3 pickup_incomplete, 3 safety_invalidation | 2.21e-6 |

Mean 26/30 (87%), and **seed 202 is the first perfect held-out score from
the deployable-inputs policy** — wrist pixels + proprioception + clock,
no privileged state. The rung's core question (can pixels replace the
privileged cube position?) is answered yes at n400 scale with matched
compute. Seed spread (24-30) says the recipe is not yet seed-robust — a
promotion-grade claim needs either more data/compute or a robustness pass;
this stays exploratory.

Infra shipped mid-sweep — **deterministic prefetching** (`vision.py`
`_read_frame_rows` + reader pool): the 38 GB n400 sidecar exceeds this
machine's 32 GB RAM, so per-step random reads were disk-bound (GPU 20%
utilized, ~11k steps/h). Reader threads now do only the raw uint8 memmap
reads (sorted-gather, 6 workers, 8 batches ahead); index draws and all
float ops stay on the main thread in step order → **bitwise identical**
(pinned: same digest, loss trace, and checkpoint sha with
`SO_ARM101_V2_PREFETCH=0`, catalogued with its depth/worker knobs in
`notes/environment-switches.md`). Measured **3.2×** (35.3k steps/h, GPU 44%).
Seed 101 ran pre-fix (~11 h); seeds 202/303 ran ~3.5 h each. Known gap
noted for later: training is not resumable mid-run (checkpoint only at
completion); worth a scratch-checkpoint mechanism before longer runs.

### Bench pick-replace v1: first run launched (2026-09-08)

Task changed to the physical bench (`bench_pick_replace_v1`: 20 mm cube on a
2 in white square 8.5 in from the base front edge). Everything above this
entry ran on the legacy 25 mm cube task with an arm model the real robot
cannot match; none of it transfers. What changed before launch, all in
`notes/bench-pick-replace-v1.md`:

- **Grasp detector** `pad_normals_v2`: jaw axis is the pad-normal bisector,
  not the tip-site line (25.4° off). Legacy 25 mm gate re-verified unchanged.
- **Joint map** `measured_20260908b`: tick-anchored zeros, shoulder/elbow
  pinned by a side photo of the rest pose. The June map had three zeros a
  quarter turn off and spans up to 37 % off.
- **Camera** calibrated from a phone checkerboard: fovy 44.0°, k1 −0.57.
  Mount pose taken from the official SO-ARM101 part (same as the Menagerie
  mount) and refined ±10 mm by a hand-eye fit; ~1.6 cm residual at working
  distance. Sim renders a pinhole, so real frames need undistortion.
- **Teacher** near-vertical (5°) approach at the jaw tips, 15/15 zero-safety
  certification on scene `6477c4bd…`.
- Gate 1 closed by user sign-off on the rest-pose sim/real comparison.

Run: seed 202, 120k steps, H90, w256, batch 64, clamp, cosine_floor_v1,
scratch checkpoints every 5k. wandb `so-arm101-v2-scaling` run `tinmahze`
(`bench-pick-replace-v1-s202-120k`), output
`artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/seed202_120k_20260908/`.

**Result (2026-09-08, run finished 08:35 local, 5.5 h wall including a
2 h screen):** `evaluation_summary.json`

| Suite | Policy | Success | Safety frames | Prefix ok |
| --- | --- | ---: | ---: | ---: |
| nominal (3 repeats) | vision | **3/3** | 0 | 3/3 |
| nominal | vision, black image | 0/3 | 204 | 0/3 |
| held-out (10 poses × 3) | vision | **27/30** | 0 | 30/30 |
| held-out | vision, black image | 0/30 | 2055 | 0/30 |

Zero safety frames across all 33 vision rollouts. The three failures are one
held-out pose (`pose_006`, cube 6 mm left and 6 mm short of nominal),
deterministic across repeats: `pickup_incomplete`, 28 mm height gain, no
strict grasp at lift, released and timed out. The black-image ablation
collapses to 0 with heavy safety invalidation, so the policy is using the
pixels. Final train MSE ~1e-6. Single seed, exploratory: no promotion
claim. Next levers, per the pre-registration: more data or a second seed
before any recipe change.

### Lens recalibration with edge coverage (2026-09-09)

Why: the 2026-09-08 intrinsics were fit from board views concentrated in the
upper-middle of the frame; the resulting 5-coefficient model's distorted radius
peaks at 929 px from the centre while the frame corners lie 1101 px out, so the
outer ~11 % of the frame had no invertible model. A full-frame simulated lens
(the user's choice: the policy sees the whole 1920×1080 frame squashed to
256×256) needs a model that reaches the corners with evidence behind it.

Session: guided capture (`tools/capture_checkerboard.py --guided --live-dir`,
new this day) with a live tkinter preview highlighting the target cell; the
assistant watched the snapshot feed and narrated poses. 17 new views at the left
and right edges, the four corners and the bottom-left/right; the bottom-centre
cells `Bl`/`Br` are permanently filled by the fixed gripper jaw and were dropped
(`--ignore-cells Bl,Br`). Tilted passes were skipped: the kept frames already
carry 11–29° of incidental tilt.

Refit on 46 views (29 old 7×16 board + 17 new 7×14 board, object points per
view): outer annulus (>800 px) now holds 355 corners at 1.1–1.3 px RMS.
Selected `pinhole_rational_8coef_free_principal_point`: RMS 1.651 px, fovy
44.85°, fovx 72.5°, principal point (867.5, 531.9), valid radius 1873 px vs
corner radius 1185 px. The fixed-principal-point rational model (1.72 px,
fovy 44.01) and the fisheye (1.656 px) also cover the frame; both free models
put the principal point ~70–90 px left of centre, so the offset is treated as
real. The centre-only file is kept as `camera_intrinsics_20260908_centre.json`.

### Lens-matched simulator and the second bench run (2026-09-09)

Decision: reproduce the real lens in the simulator (option 2) with the policy
seeing the full 1920×1080 frame squashed to 256×256; same recipe as the first
run. What landed, each committed as its own tranche:

- `contracts/lens.py`: one `LensModel` for both sides. Sim renders a 90.34°
  pinhole at 1600×900 (the smallest symmetric field covering the undistorted
  frame plus 5 %) and resamples it through the rational lens with one fixed
  sparse operator (Newton inverse, coverage guard, ~21 taps/pixel, 28 ms per
  frame); real frames get the exact area filter. Bench config carries the
  block, so the scene hash changed to `7c765d4b…`.
- Gate 1 re-signed by the user on the observation pair and the projection
  overlay (`readme-assets/bench-lens-review-*-20260909.png`): "the review
  images look good for now". The corrected principal point removed a ~100 px
  horizontal offset; the ~140 px vertical residual from 2026-09-08 remains and
  a pitch refit was declined for now. Teacher 15/15 on the new scene.
- Capture is process-parallel (scenario slots in one memmap, order-preserving
  assembly; physics bytes identical). Finding: EGL rendering jitters by one
  grey level on a few pixels run to run even sequentially, so frame digests
  were never reproducible; tests now say so.
- Training was disk-bound: raw random reads from the 37.7 GB sidecar cap at
  ~13 batches/s on this WSL disk while the GPU step is 5.3 ms. Prefetch/upload
  knobs gave 9.5 → 10.6 steps/s; the lossless in-RAM frame cache (42×, 0.89 GB,
  53 s to build) gives **57.8 steps/s**, bit-identical. Pipeline rehearsal on
  the lens scene passed end to end.

Run queued 2026-09-09 as `experiments/seed202_120k_lens_20260909` with
`--workers 10`; W&B run `iziftplw`: https://wandb.ai/7adamyasingh-rutgers-university/so-arm101-v2-scaling/runs/iziftplw.

| Suite | Policy | Success | Safety frames | Prefix ok |
| --- | --- | ---: | ---: | ---: |
| nominal (3 repeats) | vision | **3/3** | 0 | 3/3 |
| nominal | vision, black image | 0/3 | 48 | 0/3 |
| held-out (10 poses × 3) | vision | **30/30** | 0 | 30/30 |
| held-out | vision, black image | 0/30 | 492 | 0/30 |

Wall clock 94 min total (capture ~20 min with 10 workers, training 29.5 min
at ~68 steps/s with the frame cache, evaluations ~5 min) versus 5.5 h for the
first run. Held-out is perfect where the pinhole-scene run scored 27/30; the
only held-out pose that failed before (`pose_006`) succeeds in all three
repeats. One seed, so the 27 → 30 difference is suggestive, not established.
Zero safety frames across all 33 vision rollouts; the black-image ablation
collapses to 0, so the policy relies on the pixels.

### Physical inference runner built (2026-09-09, evening)

`tools/run_physical_episode.py` with `so_arm101_v2.physical`: one control
loop shared by a simulator backend and the real-arm backend. The simulator
backend cross-checks the runner's gate against the adapter on every step and
reproduces the stored nominal rollout of run `iziftplw` exactly with live
renders (427 actions, success, max |Δ executed| = 0). The real backend holds
on refusal like the bench rule (abort after 15 consecutive holds), converts
BGR→RGB before the area filter (nothing on the live camera path did this
before; it would have swapped red and blue silently), proves the fast
resampler bit-identical at preflight, keeps absolute 30 Hz deadlines, and
never disables torque. The approach to the reset pose (shoulder lift
included) is a phase inside the tool, and the pan-sign check is a read-only
preflight step. Tool-level rehearsal on the simulator:
`artifacts/so_arm101_v2/bench_pick_replace_v1/rehearsal/physical_runner_20260909/`.
No physical episode has been run.

### First physical attempt (2026-09-09, evening): approach stalled, supply at 5.4 V

Preflight passed on the hardware except the timed pan-sign check (the base was
not rotated in its window; a manual two-point check then confirmed the sign,
`physical/pan_sign_check_20260909.json`). The camera's 1080p mode streams at
3–6 fps over USB-over-IP; 720p at 29 fps has the same field of view (zero
pixel shift after resampling), so the runner captures at 720p. In the first
motion attempt (`physical/episode_01_20260909`, user-confirmed) pan, shoulder,
wrist and gripper reached the reset in 4 s but the elbow stalled 4 units short
under gravity and the approach timed out. Cause found afterwards: all servos
report a 5.3–5.4 V supply (the stock 5 V adapter under load, normal for this
arm as the user confirmed on 2026-09-10); the gripper had flagged an input
voltage error. The approach ramp now lets commands pass the reset by up to 6
units; the supply reading turned out to be normal for this arm (see the
2026-09-10 entry), so the next attempt waits on the appearance-randomized
policy passing the offline real-frame gate, not on a hardware change.

### Second physical attempt (2026-09-09, evening): approach succeeded, policy's first chunk is off-distribution

`physical/episode_02_20260909` (user-confirmed twice; supply still 5.4 V). The
approach with the overshoot ramp reached the recorded reset in 0.6 s (max
residual 0.72 units, start delta 0.023 ACT). The episode ran 72 actions at a
clean 33 ms/step (0 overruns, camera 29 fps) and aborted on 15 consecutive
holds: `physical_clip:shoulder_lift`, the policy's commands had drifted the
shoulder below the −92 floor. In simulation the first 90-action chunk is a
hold at the reset pose (shoulder command stays within −91.0…−90.2); on the
real first frame the same network, anchored on the same measured pose, emits
shoulder −94.5…−65.8, elbow 69.7…96.0, wrist 39.8…47.4: the arm retracted and
stopped, exactly as the user saw. Simple photometric edits of the real frame
(gray, gain/bias, contrast stretch) do not recover a hold, and even adding +25
brightness to the *simulated* reset frame breaks the chunk (shoulder
−93…−76). The policy is brittle to appearance: it was trained on one exact
rendered look (background level ~10, flat lighting, exact towel size) and the
real frame differs in background level (~30), texture, towel size and shape,
cube face shading, and the mousepad edge. Comparison image:
`physical/episode_02_20260909/real_vs_sim_reset_observation.png`. Runner,
timing, gate and safety all behaved as designed; the gap is visual domain
transfer, to be addressed in the capture (appearance randomization) before
another trial.

### Appearance randomization tranche (2026-09-10): regime, slots, real-frame gate, rehearsal

Response to the second physical attempt. `contracts/appearance.py` defines a
versioned regime (stored in `bench_config.json`, hashed into the scene) and a
pure resolver from a per-scenario seed: lights and headlight, material albedo
as grey × drawn tint (neutral looks dominate), a ground speckle texture with
drawn contrast and tile repeat, a visual-only towel under the square (0.9–1.6×,
±10°), skybox off or recoloured, wrist-camera pose/fovy nuisance, and
photometric gain/gamma/balance/noise/blur applied to the 256×256 observation
after the lens operator, keyed on (seed, control step). `simulation/appearance.py`
applies a draw to the MuJoCo model from an exact pristine snapshot; the adapter
recreates its renderers on a transition (texture binding is baked into a render
context). The scene gained two render-only slots; its pristine reset and viewing
observations are byte-identical to the signed 2026-09-09 review images, the
teacher is bit-identical under a draw and re-certified 15/15 on the new hash
`fcead5c7…`. Fourteen draws next to the real reset observation:
`readme-assets/bench-appearance-review-sheet-20260910.png` (brightness spread
18–93 around the real 44; neutral greys with occasional tints; towel doubling;
shadows; sky patches).

Offline real-frame gate (`physical/dry_pass.py`, `tools/check_policy_on_real_frames.py`):
the first chunk on a reset-pose frame must be hold-like (shoulder/elbow ≤ 3
units, no holds, above the floor). Calibrated on the lens policy with the
episode_02 frame: FAIL (shoulder 25.15, elbow 21.07, 23 holds, min shoulder
−94.49) versus PASS on the simulated reset chunk (0.47 / 0.45). The pipeline
records it after evaluation; the physical runner runs it live at the reset pose
and refuses the episode on failure, and records the servo voltage (the stock
5 V adapter, 5.3–5.4 V under load, is normal for this arm; refusal only below
4.8 V).
Pipeline rehearsed end to end with `--appearance` (toy scale, offline W&B).
Full suite 318 passed. The third run was queued after the bench owner
confirmed the camera review on the regenerated scene (images identical).

### Third bench run, appearance recipe (2026-09-10): held-out 30/30 and the real-frame gate passes

W&B `ytn3eygr`, `experiments/seed202_120k_appearance_20260910`. Nominal 3/3,
held-out (fixed look) 30/30 with zero safety frames, held-out under appearance
29/30 (one rollout with 50 safety frames), prefix 63/63. Black-image ablation:
3/3 nominal, 12/30 held-out, 12/30 held-out appearance, each with 672 safety
frames: the pixels are used, but appearance randomization also made the
policy's proprioceptive prior strong enough to complete some ±10 mm poses
blind (the earlier runs scored 0/33). Offline real-frame gate on the
recorded real reset frame: PASS, shoulder 0.70 units, elbow 0.61, zero holds,
min shoulder −91.03, 16/16 photometric perturbations pass; the simulated
reference chunk moves 0.22 / 0.25. Training ran at ~10.6 steps/s (263 min
wall): the photometric noise defeats the zlib frame cache, so the next run
needs an uncompressed in-RAM or GPU-resident frame store. Per the user's
standing authorization of 2026-09-10, the live trial follows immediately
(`tools/run_physical_episode.py --enable-motion --yes`).

### Live attempts 3 and 4 (2026-09-10): the plan executes on the arm; the cube is 40 mm from where the task puts it

Under the user's standing authorization the runner now auto-confirms
(`--yes`). Attempt 3 (`physical/episode_03_20260910`) was refused by the live
real-frame gate on a graze: the chunk moved the shoulder 1.2 units and
touched −92.12, one hold. The gate became `real_frame_gate_v2` (up to 3 holds,
0.5 units of floor margin; the calibration failure sits at 23 holds and 2.5
units). Attempt 4 (`physical/episode_04_20260910`) then ran the whole plan at
30 Hz with zero overruns: 90-step observation prefix at the reset (prefix
check passed, 0.017 rad), lift to the viewing pose, descent, gripper open at
~130, close at ~280, and the release at ~412, where the policy's 50-unit
gripper jump against the 20-unit relative limiter produced 15 consecutive
holds and the abort. The gripper closed to 0.55 units: it closed on nothing,
and the video shows the cube still on the towel beside the jaws
(`analysis/video_sheet.png`, `analysis/joints.png`).

Why: the same checkpoint run through the tool's simulator backend
(`rehearsal/appearance_policy_nominal_20260910`, success, 431 actions) ends
its descent with the square between the jaws; on the bench the arm stops
short with the cube far ahead and to one side, and the policy's servoing
bends the trajectory (pan −5 vs −1.4 units, shoulder 8 units less forward,
elbow 8 units more flexed) without reaching it
(`analysis/real_vs_sim_boundaries.png`, `analysis/real_vs_sim_joints.png`).
Back-projecting the reset frame through the calibrated lens and the
simulated camera pose (`analysis/cube_placement_vs_nominal.png`): the
towel's near edge lies at y = 280 mm, exactly where the task puts the
**centre** of the square (8.5 in from the base front edge), and the cube's
top face is at y ≈ 319 mm (its near edge 305 vs 270.5 nominal), x ≈ +11 mm:
the cube is about **40 mm farther from the base and 11 mm to the side** of
the pose the policy trained around (±10 mm). The same back-projection on the
simulated frame reproduces the square's near edge to 0.2 mm, so this is
placement, not a camera or joint-map error; the "sim square ~140 px lower
than the real towel" residual accepted in the camera reviews was this
placement all along. The physical fix is to move the towel so its centre,
not its near edge, sits 8.5 in from the base front edge, with the cube at
the towel's centre; the runner and the policy need no change.

Attempt 5 (`physical/episode_05_20260910`, after adding a cube-placement gate
to the runner: the reset-pose frame is back-projected and the episode is
refused beyond 15 mm with a move instruction): the approach returned the arm
from episode 4's end pose to the reset in 7 s (residual < 1 unit), then the
real-frame gate refused on the margin (shoulder 3.07 units, min −93.1): the
gate frame shows the towel bunched at the top-left with no cube on it, so
episode 4's gripper had pushed the towel and the cube off the task area. The
bench needs a physical reset (towel centred on the 8.5 in mark, cube at the
towel's centre); `tools/watch_cube_placement.py --until-within 15` polls the
camera from the parked reset pose and the next attempt starts automatically
when the cube is back in range.

### First successful live episode on the physical arm (2026-09-10, `physical/episode_10_20260910`)

Attempts 5 to 8 were refused before the episode (towel bunched and cube off
the area after episode 4; then the cube 46, 18 and 25 mm beyond the task pose
as the user re-placed it; the runner now carries a cube-placement gate that
back-projects the reset-pose frame and prints how far to move the cube).
Attempt 9 ran with the placement tolerance widened to 35 mm (cube 19 mm
beyond) and exposed a runner rule that cannot work on real servos: the lift
chunk climbed ~4 units per step, the servo lagged, and once the gap crossed
the 20-unit relative limit the hold-on-any-mask rule froze the arm while the
chunk's targets kept advancing, so the abort was guaranteed (16 holds, step
129). Fix: `bench_clip_decision`, the runner's gate since then, rate-limits
and continues on a relative-limit-only mask (range and floor clips still
hold; the simulator never triggers the limit, so scored rollouts are
unchanged).

Attempt 10 (cube read 27 mm beyond the task pose, 8 mm right) then completed
all 480 actions at 30 Hz with zero holds and zero overruns: the observation
prefix (prefix check 0.026 rad), the lift to the viewing pose, the descent,
gripper closed at ~270 and stopped at 6 units by the cube (closing on
nothing reaches 0.5), a lift of 18.6 mm by forward kinematics of the jaw
tips (51 to 70 mm) held for ~4.7 s, release at ~407 (five rate-limited steps,
no holds), retreat. The cube ended on the towel within 3 mm of where it
started (back-projected 301.7 to 300.1 mm forward, 7.6 to 4.6 mm lateral).
Evidence: `run.json`, `steps.csv`, `boundaries/`, `camera.mp4`,
`analysis/joints.png`, `analysis/video_sheet.png`. The policy is the
appearance-randomized checkpoint of run `ytn3eygr`
(`models/vision_h90/96bc418efb97e58c/model.pt`).

Open point recorded for later: the placement gate's camera reading may
carry a bias of a few degrees of pitch (the grasp succeeded with the cube
read ~20-27 mm beyond the task pose); a ruler measurement of the cube's
distance from the base front edge would settle it, after which either the
tolerance or the camera pitch in the scene config should be corrected.

### Episodes 11-13 (2026-09-10): the policy has no tolerance for placement error; next tranche = randomize placement

After the success the user asked for another run. Attempts 11 and 12 were
refused by the placement gate: from the reset pose the camera read the cube
44 mm beyond the task pose although episode 10 had just set it down within
3 mm of where it grasped it, and from the end pose the same cube read 300 mm
forward (reset pose: 325 mm). The reading is pose-dependent, so the
camera/joint-map geometry carries a bias of ~25 mm between those two poses;
the gate is now optional (`--no-cube-gate`, recorded only). Episode 13 then
ran the whole plan (480 actions, zero holds, zero overruns) in a much
brighter daylight scene and closed beside the cube (gripper 0.55; the jaws
closed at the towel's near edge with the cube ahead and to the right).

By forward kinematics of the jaw tips at the closing step: episode 10
(success) closed at (8, 290) mm, episode 13 (miss) at (1, 284) mm, episode 4
(miss) at (-13, 283) mm, the simulator at (-1, 288) mm. The three real
descents land within about 10 mm of the simulator's grasp point, and with
the cube some 20 mm beyond the task pose (camera reading, biased) success
hinges on that last 10 mm. The user's conclusion, adopted: the policy has no
tolerance for placement error because it trained on ±10 mm cube offsets.
Also recorded: the real jaws close ~15 mm higher than the simulator's (z 50-51
vs 35 mm) in every real episode, a systematic joint-map offset the placement
tranche should measure.

Next tranche (user, 2026-09-10): **randomize placement** in the capture:
widen the cube offset distribution well beyond ±10 mm (teacher screening at
the wider range; the certification offsets may need to grow), keep the
appearance regime, and retrain; the placement gate then becomes a soft
report. The arm was parked at the reset pose (`physical/reset_pose_20260910c`).

### Placement randomization tranche implemented (2026-09-10, later)

Square + cube anywhere in the user's 14 × 10 in rectangle (near edge 2 in from
the base front edge), yaw ±45°, cube jitter ±10 mm kept. Read-only
measurements first: the teacher reaches about half of the rectangle (not the
far strip beyond ~9.5 in nor the near-centre pocket); no joint-feasible pose
sees the whole rectangle, but a raised survey pose (reset + 0.40/−0.60/0.40
rad on shoulder/elbow/wrist) sees the reachable part, and the teacher's
existing 60-step look-up stage reaches it for free (horizon stays 480).
Implemented: placement regime block in the scene config; per-scenario square
centre and yaw moved with the napkin, towel and cube at every reset (the
napkin's compiler same-rotation flag had to be cleared for yaw to take
effect); teacher targets rotated with the cube yaw with a retry over the
equivalent grasp yaws (+45° reach rose from 31 % to 49 % of the grid, matching
−45° and 0°); certification over ten placements 30/30; survey-visibility
screening; pipeline `--placement` with success by region; real-frame gate v3
(track the simulated survey chunk); region-aware placement report. Reach map
and coverage map in `inspection/` (`reach_scan_92f07142_retry.png`,
`survey_pose_coverage_92f07142.png`).

### Fourth run (2026-09-10, W&B `8qj8795i`): placement recipe underfits on 400 episodes

Pipeline with `--appearance --placement --frame-store gpu --frame-stride 3`:
screening accepted 400/~580 training placements and 10 held-out (rejections
recorded), capture 400/400, training 120k steps in 14 min at 144 steps/s with
the frames on the GPU (16.9 GB used), evaluation and the real-frame check in
98 min end to end. Result: nominal 0/3 (pickup incomplete after a 30 mm
lift), held-out 3/30 (one pose 3/3; 18 safety invalidations from the cube
being nudged and tilted at 3-5 mm height, 9 incomplete pickups), held-out
appearance 0/30; prefix 30/30, so the survey move is learned. Full-data
normalized MSE 7.7e-6 against 1.3e-6 for the fixed-square run: the same 400
episodes now cover positions × yaw × appearance and the network underfits.
Frame stride is not the cause: only chunk starts at multiples of 90 occur at
inference and all of them are multiples of the stride. Next: 1200
placements, stride 9, 240k steps (same GPU footprint, ~28 min of training).

### Analysis of run 4 (2026-09-10): memorisation on the placement axis, not a saturated loss

Loss curves (batch loss, smoothed): lens run 7.9e-5 at 10k, 6.9e-6 at 60k,
9.4e-7 at 120k; appearance run 8.4e-5, 7.9e-6, 1.3e-6; placement run 2.4e-4,
2.8e-5, 7.7e-6. All three fall to the end because the cosine schedule is
still annealing (the last 20k steps buy 1.2-1.4x); the placement run is ~6x
higher at every stage and drops less from 60k to 120k (3.6x vs 6-7x), i.e.
it converges to a higher floor rather than lagging. Per-start-step loss of
the run-4 checkpoint on its training capture: start 0 (survey move) 3.0e-6,
start 90 (first descent chunk from the survey frame) 1.4e-5, later boundary
starts 5.6e-6, in-between starts 5-11e-6. The 30/30 fixed-square policy sits
at 2.5e-6 / 1.3e-6 at the same points, so the first descent chunk is 5.7x
worse and the rest 4.4x. Held-out placements (10 poses captured with the
teacher, `rehearsal/analysis_run4_heldout_capture`): start 0 1.2e-5, **start 90
1.9e-3 (136x the training loss)**, later boundaries 1.8e-5, all rows 1.9e-4
(24x). Reading: the network fits the 400 training placements well and does
not generalise the one mapping that matters, survey frame -> where to descend.
That is memorisation, not a saturated loss, so longer training alone would
not help; more placements (run 5, 1200) attacks it directly. The network is
also small for the job: 352k parameters, of which the image encoder holds
8k (three convolutions, 8/16/32 channels, stride 8 first layer); at the
survey pose the cube spans 12-25 px in the 256-px observation, i.e. two or
three 8-px patches, so sub-patch localisation relies on the 344k-parameter
head reading a 4x4x32 map. A finer, wider encoder is the next lever if run 5
narrows but does not close the gap.

### W&B chart order and a per-run generalisation phase (2026-09-10)

User request: every run's W&B page should show the charts from most to least
important. W&B sorts panel sections alphabetically by the prefix before the
slash, so the pipeline and the scaling ladder now log under numbered
sections: `01_outcome/` (success rates, safety frames, real-frame gate),
`02_generalisation/` (train vs held-out loss at chunk start 90 and the
ratio), `03_training/` (batch loss), `04_throughput/` (steps/s, elapsed,
checkpoint age), `05_phases/` (gate flags). To make section 02 exist for
every run, the pipeline gained a `heldout_fit` phase between training and
evaluation: the held-out suite is captured with frames once per (scene hash,
suite id) into `heldout_fit_captures/` and reused, and the checkpoint is
scored with `tools/heldout_fit.py` (seconds). The run 4 analysis had to build
this by hand; from run 6 on the memorisation-versus-capacity reading is
available before a single rollout is spent. Run 5 (already training under
the old names) is scored by hand the same way when it finishes.

### Run 5 (2026-09-10): 3x the placements, same result, larger gap

`experiments/seed202_120k_placement_20260910b`, W&B `pfavtk49`, 247 min
wall (capture 1200 placements ~150 min, training 240k steps ~31 min at ~134
steps/s with the GPU frame store at stride 9). Closed loop: nominal 0/3,
held-out 3/30 (far/centre 3/3, all other regions 0/27), held-out appearance
3/30, prefix 63/63, 846 safety frames on held-out (18 safety invalidations,
9 incomplete pickups). Real-frame gate: PASS on episodes 02 and 10, 16/16
photometric perturbations each, so the survey move itself transfers.

Train vs held-out at the chunk boundaries (the run-4 method, same held-out
capture): start 0 6.5e-7 / 1.4e-6 (2x), **start 90 7.0e-6 / 2.9e-3 (417x)**,
later boundaries 3.2e-6 / 1.6e-5 (5x). Run 4 had 1.4e-5 / 1.9e-3 (136x).
Reading: the training loss halved with 3x the placements and 2x the steps,
the held-out loss at the one chunk that must read the square's position from
the survey frame stayed at 2-3e-3. A learner that generalised would have
moved that number with 3x the unique scenes; this one memorises 1200 images
as easily as 400. So the data axis is exhausted for this architecture and the
question is capacity/representation (the 8k-parameter, stride-8 encoder
reading a 12-25 px cube) or an ambiguity in the inputs. The scaling ladder
(running as tsp job 4: encoders v1/v2/v3 at 300/600/1200/2400 placements,
steps sweep at (1200, v2)) separates the two: if v2/v3 bend the held-out
curve down with data, the encoder was the bottleneck; if every encoder is
flat at every size, the survey frame does not determine the descent well
enough and the next lever is input augmentation (random shifts, which force
the network to localise instead of memorise) or a richer observation.
Decision: no live trial with this checkpoint; `ytn3eygr` stays the live
policy. W&B outcome and generalisation sections back-filled at step 240001.

### Scaling ladder (2026-09-11): data, encoder and steps all fail the same way; the bottleneck is perception, not scale

`experiments/scaling_ladder_20260910` (tools/scaling_ladder.py, W&B
`scaling-ladder-92f07142` id xygyhacp; the first attempt died at ENOSPC after
its capture, see commit 796e352). One screened capture of 2400 placements
(seed 12, 8 workers, 3.5 h; 226 GB frames sidecar), data sizes as prefixes,
frame stride 18 (all inference-time chunk starts kept), hidden width 512,
120k steps; scored on a 10-placement held-out capture (seed 8) with
`tools/heldout_fit.py`; closed-loop rollouts (30) on the frontier.

Held-out MSE at chunk start 90 (survey frame -> first descent chunk), train
in brackets, ratio held-out/train:

| placements | v1 (0.83M) | v2 (2.7M) | v3 (5.0M) |
|---|---|---|---|
| 300 | 7.5e-3 (3.6e-7) 21000x | 5.1e-3 (1.5e-7) 34000x | 6.2e-3 (1.3e-7) 50000x |
| 600 | 5.1e-3 (8.4e-7) 6000x | 5.6e-3 (3.9e-7) 14000x | 3.9e-3 (3.0e-7) 13000x |
| 1200 | 2.9e-3 (1.9e-6) 1550x | 4.2e-3 (8.0e-7) 5250x | 4.6e-3 (6.8e-7) 6800x |
| 2400 | 2.3e-3 (5.4e-6) 418x, **0/30** | 3.2e-3 (1.9e-6) 1745x, **6/30** | 2.3e-3 (1.4e-6) 1580x, **3/30** |

Steps sweep at (1200 placements, v2): 60k 3.6e-3 (0/30), 120k 4.2e-3, 240k
3.6e-3 (0/30), 480k 5.6e-3 (**9/30**); train falls 1.9e-6 -> 3.3e-7 over the
same range. Power-law fit L = E + A/N^a + B/D^b: a = 0.05 (the grid's lower
bound: no dependence on parameters), b = 0.40, E = 5.3e-5 (20x the working
fixed-square policy's 2.5e-6); halving the held-out loss by data alone needs
5.7x the placements, reaching the ~2.5e-4 per-pose level at which rollouts
succeed needs ~250x (600k placements).

Decomposition per held-out pose (`analysis_per_pose.txt`,
`analysis_lookup_baseline.txt`): every frontier checkpoint's mean is set by
two or three poses (pose_000 at x = -0.146 m: 7e-3 to 3.8e-2; pose_006:
4e-3 to 9e-3) while the poses it gets right sit at 4e-5 to 2e-4, and every
closed-loop success happened at a pose with per-pose error <= 2.5e-4. A
nearest-training-placement lookup (predict the held-out chunk with the chunk
of the closest of the 2400 training placements, 2-9 mm away, 36-54 training
placements within 20 mm of every held-out pose) scores 4.0e-4 mean, 2.6e-5 at
pose_000: the network is 5x worse than a lookup table on average and 300x
worse at its worst pose although a near-identical placement (2 mm away) is in
its training set. The training density is not the problem; reading the
square's position out of the survey frame is. (The held-out scenarios also
carry their own appearance draws; a memorised image->chunk map breaks under
both.)

Ruled out: more placements (b = 0.4 with a 5e-5 floor), larger encoders
(a = 0; v2/v3 memorise harder), more steps (flat held-out, deeper fit).
Pipeline checks: stride 18 keeps every inference-time start (multiples of
90), the GPU store is bitwise-neutral by test, one run produced all 15 points
(no resume). The 480k point's 9/30 with the worst mean loss shows the mean
is a poor judge: watch the per-pose median and the count of poses under
2.5e-4.

Decision (ordered levers): (1) make memorisation impossible for the same
network: random-shift/crop augmentation of the training frames (a pure
training change, ~20 min GPU at 1200 placements), judged by per-pose
held-out start-90 (median, poses under 2.5e-4) and 30 rollouts; (2) if that
is not enough, split perception from control: a square/cube localiser
trained on the same captures (labels = scenario placements, heavy
augmentation, deployable because it reads the camera image) whose (x, y,
yaw) estimate conditions the chunk policy; the lookup baseline shows the
control side is then easy (the fixed-position policies reach 30/30), and the
localiser can be validated on the recorded real frames against the ruler;
(3) data only where the localiser needs it. The number to watch: per-pose
held-out start-90 loss, median under 2.5e-4 on all 10 poses.

### Random-shift augmentation (2026-09-12): the memorisation gap closes 350x, the tail failures become near-misses, closed loop stays at 9/30

`experiments/augmentation_20260912` (tools/augmentation_run.py, W&B
`random-shift-92f07142` id 8okj7i9y, tsp job 6, 1 h 25 min wall). Lever 1
of the ladder decision, one change: each training frame is replicate-padded
by `s` pixels and cropped back to 256x256 at a per-sample random offset
(`VisionChunkedConfig.random_shift`, commit 6b4d936). Everything else is the
ladder's sweep point (its 2400-placement capture as a 1200-placement prefix,
v2 encoder, hidden 512, 120k steps, stride 18, seed 202); the ladder's
unaugmented point is the baseline, re-scored per pose and given the 30
rollouts it did not have. Evaluation frames are never shifted.

| shift (px) | train start 90 | held-out start 90 | ratio | per-pose median | per-pose max | poses <= 2.5e-4 | rollouts | safety frames | train min |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 8.0e-7 | 4.2e-3 | 5250x | 1.1e-3 | 1.8e-2 | 0/10 | **0/30** | 525 | 20 |
| 4 | 8.0e-6 | 1.3e-3 | 168x | 1.3e-3 | 4.4e-3 | 1/10 | **9/30** | 279 | 25 |
| 12 | 4.1e-5 | 6.2e-4 | 15x | 2.4e-4 | 2.5e-3 | 5/10 | **9/30** | 193 | 26 |

Per pose (held-out start-90 MSE; wins out of 3 at shift 0/4/12):

| pose | shift 0 | shift 4 | shift 12 | wins |
|---|---|---|---|---|
| pose_000 (x -0.146) | 1.8e-2 | 2.1e-3 | 1.0e-3 | 0/0/0 |
| pose_001 | 8.5e-4 | 1.1e-4 | 9.1e-5 | 0/0/3 |
| pose_002 | 4.2e-4 | 2.8e-4 | 1.5e-4 | 0/3/0 |
| pose_003 | 2.8e-3 | 1.3e-3 | 9.0e-4 | 0/0/0 |
| pose_004 | 7.4e-3 | 7.3e-4 | 6.2e-5 | 0/3/3 |
| pose_005 | 4.9e-4 | 1.3e-3 | 2.3e-4 | 0/0/0 |
| pose_006 | 8.6e-3 | 4.4e-3 | 9.8e-4 | 0/0/0 |
| pose_007 | 6.9e-4 | 5.4e-4 | 3.6e-5 | 0/0/3 |
| pose_008 | 1.0e-3 | 1.3e-3 | 2.5e-3 | 0/3/0 |
| pose_009 | 1.2e-3 | 1.4e-3 | 2.6e-4 | 0/0/0 |

Readings, in the skill's order:

- **Loss curves.** Batch loss at 10k/60k/120k: shift 0 2.5e-4 / 6.4e-6 /
  6.4e-7; shift 4 1.8e-4 / 1.4e-5 / 2.7e-6; shift 12 3.5e-4 / 3.4e-5 /
  1.0e-5. The augmented runs are still descending at 120k (2.4x and 3.4x in
  the last half versus 1.9x for the baseline's annealing tail): with the
  hash-table shortcut removed the network has not finished fitting.
- **Decomposition.** All of the held-out error is still at chunk start 90
  (survey frame -> first descent chunk). Starts 180/270/360 are 1e-7 to 1e-4
  for every pose and every shift (`analysis_later_chunks.txt`); shift did not
  move error downstream.
- **Train versus held-out.** The ratio at start 90 falls 5250x -> 168x ->
  15x. Held-out loss now follows train loss: the network reads the survey
  frame instead of memorising it. The mean held-out loss (6.2e-4) is now
  close to the 1200-prefix lookup baseline (6.0e-4 mean, 3.7e-4 median) and
  the per-pose median (2.4e-4) is at the level where the ladder's successes
  occurred, with five poses under it (none before).
- **Where the failures went** (`analysis_rollouts.txt`, repeat 0 per pose,
  categories from the pickup events): shift 0 = 5 collisions (unsafe contact
  on descent, cube pushed 7-26 mm), 3 misses (no lift), 2 lifted-no-strict-
  grasp, 0 successes. Shift 4 = 2 collisions, 2 misses, 3 lifted-no-strict-
  grasp, 3 successes. Shift 12 = 1 collision, 2 misses, 3 lifted-no-strict-
  grasp, 1 strict-then-lost (pose_002: strict grasp at step 259, unsafe
  contact 312, grasp loss 316), 3 successes. "Lifted-no-strict-grasp" is a
  cube caught by an edge or corner (interior face count 1-2, corner
  rejections 1-2, opposition quality 0) and carried 20-30 mm, then dropped at
  the release: the descent found the cube but landed a few millimetres or
  degrees off a pad-centred grasp. So the gross failures of the baseline
  (collide / miss) became precision failures, and the successes are the poses
  whose descent was already within pad tolerance. The 2.5e-4 threshold is not
  sharp: pose_008 won at 1.3e-3 (shift 4), pose_002/005/009 lost at 1.5e-4 to
  2.6e-4 (shift 12); a chunk MSE averages 90 steps x 6 channels, and a few
  millimetres at closure are a small part of it.
- **Pipeline.** Shift 0 leaves every identity and digest unchanged (test);
  the augmentation is a pure gather of the padded image (no interpolation,
  pinned by test); the shift draws come from a separate seeded generator so
  the minibatch stream and the prefetch equivalence are untouched; the
  baseline re-scored at the ladder's value (4.18e-3) before anything else ran.

Ruled in: input augmentation is the lever the ladder said it was; with it,
data and steps start to matter again (held-out tracks train at 15x). Ruled
out: the 9/30 plateau is not a plateau of the same kind as the ladder's
(the ladder's 9/30 came with a 5e-3 mean and a memorised map; this one comes
with a 6e-4 mean and per-pose errors within a factor of 2-4 of the working
fixed-square policy on half the poses).

Decision (ordered): (1) keep shift 12 and give it what the ratio now
permits: 240k-480k steps (the curve is still descending; the baseline's
steps sweep was flat only because it was memorising) and the full 2400
placements (data exponent should now exceed the ladder's 0.4); one run of
each, judged by per-pose median and the count of lifted-no-strict-grasp
rollouts turning into successes; (2) the square localiser stays queued
behind it, and the "lifted-no-strict" profile is its argument: the network
knows where the cube is to within a cube width, and the last millimetres
are a precision problem a dedicated (x, y, yaw) head is built for; (3) shift
16-24 only if (1) stalls, since 12 px is already half a cube width. Number
to watch: per-pose held-out start-90 median under 2.5e-4 with 8/10 poses
under it, and rollouts above 15/30. Still no live-trial candidate;
`ytn3eygr` remains the live policy.

### Shift-12 follow-ups (2026-09-12): steps bring memorisation back, placements do not; 2400 placements + shift 12 = 15/30

Two runs of `tools/augmentation_run.py`, one change each from the shift-12
point above (1200 placements, v2, 120k steps, stride 18, seed 202), each
with the ladder's unaugmented point at the same settings as its baseline.
`experiments/augmentation_shift12_480k_20260912` (W&B l5c6y603; held-out
curve from `tools/heldout_watch.py` in W&B kspx4lgi) and
`experiments/augmentation_shift12_2400_20260912` (W&B u883j0tu, curve in the
run). Both runs now log held-out start-90 loss every 5000 steps
(`heldout_curve.jsonl`; trainer `on_checkpoint` observer, commit 01977c6).

| placements / steps / shift | train 90 | held-out 90 | ratio | per-pose median | poses <= 2.5e-4 | rollouts | safety | failures (repeat 0) |
|---|---|---|---|---|---|---|---|---|
| 1200 / 120k / 0 | 8.0e-7 | 4.2e-3 | 5250x | 1.1e-3 | 0/10 | 0/30 | 525 | 5 collide, 3 miss, 2 edge-lift |
| 1200 / 120k / 12 | 4.1e-5 | 6.2e-4 | 15x | 2.4e-4 | 5/10 | 9/30 | 193 | 1 collide, 2 miss, 3 edge-lift, 1 lost |
| 1200 / 480k / 0 | 3.3e-7 | 5.6e-3 | 16800x | 1.0e-3 | 4/10 | 9/30 | 279 | 3 collide, 3 miss, 1 edge-lift |
| 1200 / 480k / 12 | 1.4e-5 | 1.1e-3 | 78x | 1.1e-3 | 1/10 | **0/30** | 382 | 3 collide, 2 miss, 5 edge-lift |
| 2400 / 120k / 0 | 1.8e-6 | 3.2e-3 | 1744x | 1.0e-3 | 2/10 | 6/30 | 228 | 2 collide, 4 miss, 2 edge-lift |
| 2400 / 120k / 12 | 9.8e-5 | 5.7e-4 | **6x** | 3.0e-4 | 3/10 | **15/30** | 261 | 1 collide, 2 miss, 2 edge-lift |

Held-out start-90 during training (every 5000 steps; "train" is a fixed
200-row sample of the training prefix):

| step | 1200 pl., 480k: held-out / train / ratio | 2400 pl., 120k: held-out / train / ratio |
|---|---|---|
| 25k | (watcher attached at 95k) | 9.7e-4 / 1.1e-3 / 1x |
| 65k | | 7.3e-4 / 2.5e-4 / 3x |
| 95k | 1.4e-3 / 1.9e-4 / 7x | 6.0e-4 / 1.2e-4 / 5x |
| 120k | 1.4e-3 / 1.8e-4 / 8x | 5.7e-4 / 1.0e-4 / 5x (end) |
| 215k | 1.0e-3 / 7.2e-5 / 14x | |
| 275k | 8.9e-4 / 4.2e-5 / 21x (best) | |
| 335k | 9.2e-4 / 3.1e-5 / 30x | |
| 480k | 1.1e-3 / 1.4e-5 / 77x (end) | |

Readings:

- **Steps (1200 placements, shift 12, 480k).** Held-out flattens at ~1e-3
  from 150k on and never reaches the 120k run's 6.2e-4 at any checkpoint
  (best 8.5e-4 at 260k), while the train sample falls 13x and the ratio
  climbs 7x -> 77x. A 12 px shift leaves 625 offsets per frame; given 4x
  the steps the network memorises those too. Steps are ruled out under
  augmentation as they were without it; 120k with the cosine anneal is the
  right budget. Closed loop 0/30 with the same per-pose median as the
  shift-4 point that scored 9/30: with 10 poses x 3 repeats, a single
  run's rollout count moves by +-9 between checkpoints of similar loss.
  Treat single-run rollout counts accordingly.
- **Placements (2400, shift 12, 120k).** The ratio stays at 1-5x for the
  whole run (held-out tracks train from the first checkpoint), the final
  held-out mean is the lowest of any point (5.7e-4), the worst ladder pose
  (pose_000, x = -0.146) drops to 1.8e-4 from 1.8e-2, and closed loop is
  15/30 (5 poses of 10, all three repeats each) against 6/30 for the same
  capture unaugmented and 9/30 for the 1200-placement augmented point. The
  mean and median are within noise of the 1200-placement augmented point,
  so the loss alone does not prove the data effect; the ratio (15x -> 6x)
  and the doubled rollouts do. Failures are again 2 edge-lifts, 2 misses,
  1 collision: precision at closure, not gross localisation.
- **Later chunks** are unaffected in every run (same as before; not
  re-tabulated).

Ruled in: with memorisation blocked, unique placements are the axis that
pays (ratio 15x -> 6x, rollouts 9 -> 15 of 30) and the held-out curve is
still descending at 120k on 2400 placements with the ratio at 5x. Ruled
out: steps (memorisation returns), and reading a single run's rollout count
as a stable level.

Decision (ordered): (1) measure the noise before spending on levers: repeat
2400 / 120k / shift 12 with seeds 101 and 303 (25 min each + rollouts); if
the three seeds hold 12/30 or better the data effect is real and 4800
placements (a 7 h CPU capture, overnight) is the next data point; (2) the
square localiser stays next in line for the closure-precision failures,
now with a capture of 2400 labelled placements to train it on; (3) shift 12
at 240k on 2400 placements only if the seed repeats show the 120k curve
still falling with the ratio under 10x. Number to watch: per-pose held-out
start-90 median across seeds and rollouts >= 15/30 on all three. Live
policy unchanged (`ytn3eygr`); the 2400/shift-12 checkpoint is the first
placement-general candidate worth a real-frame gate check.

### Seed repeats (2026-09-12 evening): 2400 / 120k / shift 12 at seeds 101 and 303; loss reproduces, rollouts spread 6-15

`experiments/augmentation_shift12_2400_seed101_20260912` (W&B qcxtwbdg) and
`..._seed303_20260912` (W&B qiz5n9g3), `analysis_seeds.txt` beside the latter.
Same capture, same recipe, only the training seed differs (minibatch order,
initialisation, shift offsets).

| seed | train 90 | held-out 90 | ratio | per-pose median | poses <= 2.5e-4 | rollouts | safety frames |
|---|---|---|---|---|---|---|---|
| 202 | 9.8e-5 | 5.7e-4 | 5.8x | 3.0e-4 | 3/10 | 15/30 | 261 |
| 101 | 1.3e-4 | 5.8e-4 | 4.4x | 4.0e-4 | 2/10 | 6/30 | 455 |
| 303 | 1.1e-4 | 3.4e-4 | 3.3x | 2.6e-4 | 5/10 | 12/30 | 601 |
| mean +- sd | | 5.0e-4 +- 1.3e-4 | | | | 11.0 +- 4.6 | |

Reading: the loss-side result is stable across seeds (every seed under
6e-4 held-out and under 6x ratio, against 6.2e-4 / 15x for 1200 placements
and 3.2e-3 / 1744x unaugmented on this capture), so "placements pay once
memorisation is blocked" holds. The closed-loop count is not stable: 15, 6,
12 of 30, sd 4.6, with pose wins that move between seeds (five poses won by
at least two seeds: 001, 002, 004, 006, 007; 000, 005, 008, 009 lost by all
three). Per pose the loss also moves 2-5x between seeds at the same
placement (pose_003: 6.6e-4 / 1.4e-3 / 2.4e-4), so the per-pose profile of
one run is a noisy estimate too. The decision rule set above (all three
seeds >= 12/30) is not met; the mean (11/30) is above every earlier
configuration (1200-placement shift 12: 9/30 single seed; unaugmented 2400:
6/30) but the spread means a single 30-rollout evaluation cannot rank
configurations closer than ~10 successes apart. Consequences: (1) judge
configurations by held-out loss across seeds (sd 1.3e-4 here) and by
rollouts pooled over seeds (33/90 for this recipe), not by one run's
count; (2) the 4800-placement run (queued) is judged the same way: its loss
must land under 3.4e-4 to be distinguishable from this recipe's best seed;
(3) the closure-precision failures (edge-caught lifts) persist in every
seed, which keeps the square localiser as the lever after data.
