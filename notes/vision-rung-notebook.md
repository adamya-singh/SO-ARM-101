# Vision Rung Notebook (running log, exploratory lane)

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
