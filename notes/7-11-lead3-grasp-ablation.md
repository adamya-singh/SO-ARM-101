# 2026-07-11 Lead-3 Grasp-Focused PPO Ablation

## Goal

Start fresh from the lead-3 supervised `026020` checkpoint and determine which
optimizer, exploration, and grasp-quality reward settings most reliably turn
the policy's useful approach behavior into jaw-centered grasp and lift.

The study is staged. Phase one screens six settings for 100 updates at seeds
17 and 71. After comparing complete W&B curves, phase two replicates the top
two settings for 190 updates at seeds 83 and 101 and evaluates fixed milestone
snapshots. Final confirmation uses 30 deterministic fixed, 30 stochastic fixed,
and 30 stochastic narrow-block episodes per finalist.

## Reward Profiles

- `baseline`: no overrides; numerically preserves the current reward defaults.
- `jaw_quality`: raises positive pregrasp alignment, aligned-close, and
  jaw-centered-contact shaping and extends centered-contact memory.
- `jaw_quality_rebalanced`: keeps the positive quality changes but reduces
  generic bilateral/persistent-grasp reward and modestly strengthens side-push
  and displacement penalties.

The profiles are selected with `train_act_in_sim.py --reward-profile ...` and
are stored in W&B and checkpoint configuration.

## Launchers

- Phase one: `simulation_code/queue_act_grasp_phase1_20260711.sh`
- Phase two: `simulation_code/queue_act_grasp_phase2_20260711.sh`
- Final evaluation: `simulation_code/queue_act_grasp_final_eval_20260711.sh`

Every launcher validates required artifacts, limits task spooler to one GPU
job, and supports `DRY_RUN=1`. Phase two requires `WINNER1` and `WINNER2` IDs;
final evaluation requires two checkpoint paths and their reward profiles.

## Selection Rules

At the phase-one gate, rank profiles by cross-seed success/lift presence,
worst-seed lift and micro-lift counts, micro-lift-to-grasp conversion, centered
contact/close quality, side pushing, displacement, action clipping, and PPO
health. Raw grasp duration or return alone cannot select a winner.

The independent evaluator writes JSON containing per-episode and aggregate
sustained-grasp rate, strict-lift rate, success, height gain, displacement, and
action clipping. The final setting is chosen by sustained-grasp episode rate,
then strict lift/success, with height, displacement, and clipping as
tie-breakers.

## Phase-One Live Inspection

User side-by-side stochastic live simulation compared:

- A baseline, seed 17, episode 99; and
- E jaw-quality-rebalanced, seed 71, episode 79.

The A checkpoint appeared to be trying to align around the block and establish
a grasp before lifting. Its alignment and grasp were not yet correct, but the
behavior looked close to the intended sequence. The E checkpoint appeared more
clumsy: it sometimes missed the block entirely and sometimes shook or disturbed
the block instead of forming a controlled grasp.

This observation favors A qualitatively and raises concern that E's additional
training grasp/micro-lift activity includes unstable contact rather than better
grasp quality. It is user visual evidence from selected checkpoints, not an
independent success-rate measurement; the planned replicated and headless
evaluation remains necessary.

## Phase-One Results

All 12 runs completed 100 updates. The figures below were aggregated from all
100 W&B history rows in each run, not from only the final summary. A
success-bearing worker chunk is recovered from the logged mean success rate
across the 12 parallel workers. These are on-policy training-rollout metrics;
they are not independent checkpoint evaluations.

Common conditions were the lead-3 supervised `026020` initialization, fixed
appearance and block placement, actor learning rate `1e-6`, critic learning
rate `5e-5`, and initial log standard deviation `-2.0`, except for the change
named in each setting:

- A: unchanged baseline and the control for all comparisons.
- B: actor learning rate reduced to `5e-7`.
- C: lower exploration, with initial log standard deviation `-2.356675`
  (action standard deviation about `0.095` instead of `0.135`).
- D: baseline optimizer and exploration with the `jaw_quality` reward profile.
- E: baseline optimizer and exploration with the
  `jaw_quality_rebalanced` reward profile.
- F: the `jaw_quality` profile combined with C's lower exploration.

| Setting | Seed | W&B | Success chunks | Lift steps | Micro lifts | Grasp steps | Maximum height gain |
|---|---:|---|---:|---:|---:|---:|---:|
| A | 17 | [`kdwzzqh7`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/kdwzzqh7) | 2 | 34 | 511 | 1,457 | 11.4mm |
| A | 71 | [`5x3y7onh`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/5x3y7onh) | 1 | 22 | 274 | 652 | 10.2mm |
| B | 17 | [`svxstwyd`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/svxstwyd) | 3 | 16 | 71 | 266 | 12.4mm |
| B | 71 | [`yeyzhlqb`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/yeyzhlqb) | 1 | 6 | 60 | 173 | 10.5mm |
| C | 17 | [`bx0nm3q9`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/bx0nm3q9) | 1 | 5 | 163 | 392 | 10.1mm |
| C | 71 | [`rkn4swif`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/rkn4swif) | 5 | 38 | 2,448 | 7,849 | 10.4mm |
| D | 17 | [`n2l20yhk`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/n2l20yhk) | 2 | 15 | 251 | 448 | 10.3mm |
| D | 71 | [`pjph7x1l`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/pjph7x1l) | 2 | 9 | 67 | 156 | 10.7mm |
| E | 17 | [`ejh08us4`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/ejh08us4) | 1 | 14 | 64 | 270 | 10.6mm |
| E | 71 | [`kima1cyb`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/kima1cyb) | 2 | 27 | 1,019 | 2,776 | 10.9mm |
| F | 17 | [`r3aznu1j`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/r3aznu1j) | 2 | 3 | 70 | 205 | 11.2mm |
| F | 71 | [`u8fliart`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/u8fliart) | 1 | 22 | 1,161 | 5,004 | 10.5mm |

### Interpretation

- A had the strongest worst-seed lift, micro-lift, and grasp counts. Both seeds
  improved late, so A is the most reproducible phase-one setting in this
  two-seed sample.
- B produced much less grasp and lift activity at both seeds. Its isolated
  seed-17 success and 12.4mm height maximum do not outweigh the consistently
  slower learning, but they show that it was not completely inactive.
- C seed 71 was the strongest individual training run by grasp, micro-lift,
  lift, success, and late return. C seed 17 had only five lift steps. C
  therefore has the highest observed upside but no evidence yet that the
  improvement reliably comes from lower exploration rather than seed
  variation.
- D increased the logged aligned-close reward, as intended, but reduced grasp,
  micro-lift, and lift counts relative to matched A seeds. Its two success
  chunks at each seed are encouraging but too sparse to establish an
  improvement.
- E also improved aligned-close measurements. Seed 71 developed strong late
  grasp and micro-lift activity, while seed 17 did not. It is a plausible
  reward-profile candidate, but the two seeds do not establish a consistent
  advantage over A.
- F behaved similarly to C in its seed sensitivity. Seed 71 accumulated many
  grasp steps but only matched A seed 71's 22 lift steps, while seed 17 had
  three lift steps. This suggests poor grasp-to-lift conversion, not a clear
  improvement.

Reward-component magnitudes are not directly comparable between profiles when
the profile changes their weights. Return and raw grasp duration were therefore
not used alone to name a winner. Approximate KL and action clipping remained
small in every run, so there is no obvious PPO-instability explanation for the
cross-seed differences.

### Provisional Checkpoint Candidates

- Most reproducible setting: A. Its strongest training-associated snapshot is
  [`A seed 17 ep0099`](../simulation_code/outputs/train/act_lead3_grasp_phase1_20260711/A_baseline_lr1em6_stdm2.0_s17/act_sim_ppo_checkpoint_ep0099.pt).
  Updates 90-99 contained 14 lift steps, 267 micro lifts, 798 grasp steps, and
  two success-bearing chunks.
- Strongest individual run: C seed 71. Its provisional snapshot is
  [`ep0089`](../simulation_code/outputs/train/act_lead3_grasp_phase1_20260711/C_baseline_lr1em6_stdm2.356675_s71/act_sim_ppo_checkpoint_ep0089.pt).
  Updates 80-89 contained nine lift steps, 507 micro lifts, 1,502 grasp steps,
  and two success-bearing chunks. The following ten updates had higher grasp
  duration and return but only two lift steps and no success-bearing chunks.

These labels are provisional. A rollout window measures the policies used to
collect training data around a snapshot; it does not directly evaluate the
saved checkpoint. Neither candidate has yet passed the planned deterministic
and stochastic independent evaluation, so this phase does not establish a
new final policy.

### Live Inspection of A and C

User live inspection of the provisional fixed-block stochastic candidates
found that neither produced the intended face-to-face grasp:

- `A seed 17 ep0099` contacted the block at its corners with a relatively loose
  grip. It shook the block substantially and sometimes rolled it rather than
  securing it for a lift.
- `C seed 71 ep0089` repeatedly pinched the block at its corners. It appeared
  to be learning contact and closure, but did not form a secure face grip or
  visibly attempt a lift.

This qualitative result changes the interpretation of the training counters.
The current `gripped` signal requires sufficient force from both gripper bodies
but does not require contact on opposing block faces or exclude corners.
Likewise, `jaw_centering_score` measures the jaw-tip midpoint relative to the
block center rather than the locations and normals of the contacts. A corner
pinch can therefore count as a centered grasp and accumulate grasp-persistence
reward. The high grasp and micro-lift totals, especially for C seed 71, are
evidence of interaction under the present detector, not evidence of a
lift-ready grasp.

Consequently, A remains the most reproducible setting only by the current
training metrics, and C remains the peak training run only by those same
metrics. Neither is a validated policy candidate after live inspection.
Continuing either checkpoint unchanged risks reinforcing corner pinching,
shaking, or rolling. Before phase-two training, the grasp-quality detector and
reward should require opposing face contacts away from corners and should use
contact locations/normals; candidate checkpoints should then be reevaluated
with explicit face-grasp and lift-attempt metrics.

## A Seed-17 Continuation to Episode 149

On 2026-07-12, A seed 17 was resumed unchanged from `ep0099` for 50 additional
updates. The continuation reused W&B run
[`kdwzzqh7`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/kdwzzqh7)
and completed at episode 149 with 107,968 environment steps and 3,600 rollout
chunks. The saved continuation snapshot is
[`ep0149`](../simulation_code/outputs/train/act_lead3_grasp_phase1_20260711/A_baseline_lr1em6_stdm2.0_s17/act_sim_ppo_checkpoint_ep0149.pt).

User live inspection of stochastic, fixed-block `ep0149` inference found that
the policy was trying to align more than `ep0099`. This is a useful qualitative
improvement in the approach/pregrasp phase. It still did not begin lifting,
however, so the extra training has not demonstrated grasp-to-lift conversion or
a successful policy. The next behavioral objective is a deliberate upward arm
motion after a stable face grasp; alignment alone should not be treated as the
new best-policy criterion. This observation is qualitative and does not replace
the planned independent checkpoint evaluation.

## Strict Face-Grasp Follow-up

On 2026-07-12, the environment was instrumented with a geometry-aware detector
that separately records legacy force closure, interior jaw-to-face contact,
bilateral interior contact, and a strict opposing-face grasp. Strict reward
profiles G through P then tested progressively stronger curricula: strict
gating, pregrasp alignment, potential-difference shaping, single-jaw interior
contact, graded contact quality, and bilateral opposition. The strict detector
was unit-tested against valid opposing contacts, corner pinches, one-sided
contacts, and misaligned normals. Re-evaluation also confirmed that the old A
and C corner-pinching behavior is rejected.

None of the strict screens produced a strict face-grasp step. In the final O/P
screen, each profile was trained for 100 updates at seeds 17 and 71:

- O recorded 3 and 4 single-jaw interior-contact steps, 0 bilateral interior
  steps, 0 strict face-grasp steps, and 0 lift steps.
- P recorded 4 and 5 single-jaw interior-contact steps, 0 bilateral interior
  steps, 0 strict face-grasp steps, and 0 lift steps.
- All four runs still accumulated thousands of corner-only contact steps
  (3,443 to 5,998). The new continuous contact-quality reward provided a
  learning signal, but did not convert one-sided or corner contact into an
  opposing-face pair.

Because the predeclared advancement gate required reproducible strict
face-grasp activity, no profile qualified for long replication. To check
whether training-rollout transients hid useful checkpoint behavior, two
diagnostic snapshots were nevertheless evaluated for 30 episodes in each of
deterministic fixed-block, stochastic fixed-block, and stochastic narrow-reset
modes:

- M seed 17 `ep0059`, selected from the screen's highest single-jaw interior
  window, produced 0 strict grasps, 0 interior-face contacts, 0 strict lifts,
  and 0 successes in all 90 evaluation episodes.
- P seed 71 `ep0079`, selected from P's highest single-jaw interior window,
  produced 0 strict grasps, 0 bilateral contacts, 0 strict lifts, and 0
  successes in all 90 evaluation episodes. It produced one isolated
  single-jaw interior-contact step in stochastic narrow evaluation.

User live inspection of stochastic fixed-block inference confirmed the
quantitative failure for both snapshots. M `ep0059` and P `ep0079` mostly
hovered around the cube, occasionally poked it, and otherwise barely interacted
with it. Neither showed a credible grasp attempt, secure contact, or transition
toward lifting. They should not be described as usable policies or practical
"best" candidates; they were only the least-bad diagnostic snapshots selected
for failure analysis.

The unbiased conclusion is that no new policy was found. More PPO updates on
these checkpoints are not justified by the evidence: exploration repeatedly
finds corner contact but almost never reaches even one interior face, and never
reaches a bilateral opposing pair. The next experiment should change the
initial-state or action curriculum so the policy can experience bilateral
face contact before asking PPO to discover the entire sequence from the current
start distribution. The legacy A/C policies and all G-P profiles remain
diagnostic artifacts, not deployment candidates.

## 2026-07-12 Reward Rollback

Strict G-P reward shaping was retired after live inspection showed hovering
and poking instead of credible grasp attempts. Training and evaluation again
use the legacy force-based grasp and lift reward behavior. The geometry-aware
face-grasp detector remains enabled as telemetry only, including face
alignment/opposition, interior and bilateral contacts, corner rejection, and
contact quality. All historical G-P checkpoints and results remain failures
and are not deployment candidates.
