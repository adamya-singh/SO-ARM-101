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
`--workers 10`; W&B id in that directory's `wandb.json`. Result: pending.
