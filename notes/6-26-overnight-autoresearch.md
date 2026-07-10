# 2026-06-26 Overnight ACT Sim PPO Autoresearch

Goal: restart from the last promising ACT sim checkpoint and run focused
ablations until lift-related reward starts improving and ideally produces lift
successes.

## Baseline

- Starting checkpoint:
  `outputs/train/act_sim_ppo_contact_commit_v5_ep150/act_sim_ppo_checkpoint_snapshot_20260625_130447.pt`
- Reason: v5 episode-90 preserved the best observed approach and visible
  grip-attempt behavior before later continuations collapsed into local minima.
- Rejected continuation: `act_sim_ppo_pregrasp_guard_v8_overnight` degraded into
  air-waving behavior and should not be used as a baseline.

## Monitoring Priorities

- Primary: `rollout/reward_components/lift_progress_reward`,
  `rollout/lift_steps`, `rollout/reward_components/lift_bonus_reward`, and
  success/lift events.
- Health: rollout return, `rollout/contact_steps`, `rollout/grasp_steps`,
  `train/approx_kl`, `train/clip_fraction`, policy loss spikes, and obvious live
  behavior regressions.
- Stop or pivot if contact/grasp disappear, KL/clip fraction becomes unstable,
  or live behavior collapses into retreat/scrunching/air-waving.
- Stop a run rather than letting it continue if
  `rollout/reward_components/lift_progress_reward` flatlines at `0` for a
  sustained window; note the failure mode and move to the next ablation.

## Run Log

### Run 1: v5 episode-90 continuation, conservative short gate

- Status: stopped early.
- W&B: [`d2ej7q0a`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/d2ej7q0a)
- Output directory:
  `outputs/train/act_sim_ppo_v5ep90_lift_progress_gate_r1`
- Hypothesis: a short continuation from v5 episode-90 with the current
  post-rollback reward stack can show whether lift-progress reward is reachable
  before longer overnight training overfits into a bad posture.
- Command summary: 12 parallel envs, 2 rollout chunks per env, 1 PPO epoch,
  minibatch 64, chunk size 30, fixed curriculum block, snapshots every 25
  updates.
- Started at local W&B run directory:
  `simulation_code/wandb/run-20260626_015041-d2ej7q0a`.
- Early W&B metrics at updates `91`-`95`: lift-progress reward is present
  (`0.0` to `2.93`), contact is intermittently high (`0` to `253` steps),
  grasp appears on the stronger updates (`68` to `126` steps), but
  `lift_steps`, `lift_bonus_reward`, and success are still zero. PPO health is
  stable so far with `train/clip_fraction = 0`.
- Updates `96`-`105`: lift-progress reward has not flatlined; recent values are
  about `0.17` to `3.41`. Contact remains frequent on strong updates (`260` to
  `305` steps), grasp still appears intermittently (`23` to `135` steps), and
  `train/clip_fraction` remains `0`. Still no lift steps or successes.
- Updates `106`-`112`: lift-progress has faded but is not yet a zero flatline
  (`2.88`, `2.04`, `1.35`, `0.94`, `0.55`, `0.38`, `0.02`). Contact/grasp are
  weakening in the same window. Watch the next synced window closely; stop and
  pivot if lift-progress stays at `0`.
- Updates `113`-`120`: not a strict zero flatline, but the run is fading badly.
  Lift-progress alternates between small values and zero
  (`0.22`, `0`, `0.73`, `0`, `0.78`, `0`, `0.62`, `0`), contact is often zero,
  and grasp has effectively disappeared. Stopped rather than spending the full
  300-update budget.

### Run 2: v5 episode-90, lower actor LR preservation test

- Status: running.
- W&B: [`vvlrtwjl`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/vvlrtwjl)
- Output directory:
  `outputs/train/act_sim_ppo_v5ep90_lift_progress_low_lr_r2`
- Hypothesis: Run 1 washed out the fragile contact/grasp behavior while still
  producing intermittent lift-progress reward. Lowering the actor LR may preserve
  the v5 episode-90 behavior longer and let the critic/reward signal shape
  toward lift without collapsing grasp.
- Command summary: restart from the same v5 episode-90 snapshot, `policy-lr=3e-6`,
  `critic-lr=5e-5`, 12 parallel envs, 2 rollout chunks per env, 1 PPO epoch,
  minibatch 64, chunk size 30, fixed curriculum block, snapshots every 25
  updates.
- Started at local W&B run directory:
  `simulation_code/wandb/run-20260626_015902-vvlrtwjl`.
- Early W&B metrics at updates `91`-`93`: update `91` is strong
  (`lift_progress_reward=3.59`, `contact_steps=244`, `grasp_steps=135`,
  return `10.25`), followed by a weak update `92` and partial recovery at
  update `93` (`lift_progress_reward=1.06`, `contact_steps=142`). No lift steps
  or successes yet; `train/clip_fraction` remains `0`.
- Updates `94`-`100`: stopped early. The lower actor LR did not preserve grasp;
  after update `93`, `grasp_steps` stayed at `0`, contact alternated between
  weak and absent, and lift-progress alternated between small values and zero.
  This looked worse than Run 1 despite stable PPO metrics.

### Run 3: v5 episode-90, smaller parallel batch credit assignment test

- Status: running.
- W&B: [`x586ixom`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/x586ixom)
- Output directory:
  `outputs/train/act_sim_ppo_v5ep90_lift_progress_smallbatch_r3`
- Hypothesis: 12 parallel envs may average sparse good contact/grasp trajectories
  with many failed trajectories, making the update drift away from the fragile
  good behavior. Use fewer envs and smaller minibatches to see whether sparse
  lift-progress/contact-grasp signal can steer the policy before it fades.
- Command summary: restart from the same v5 episode-90 snapshot, 4 parallel envs,
  4 rollout chunks per env, minibatch 16, 1 PPO epoch, default actor/critic LR,
  chunk size 30, fixed curriculum block, snapshots every 25 updates.
- Started at local W&B run directory:
  `simulation_code/wandb/run-20260626_020321-x586ixom`.
- Updates `91`-`96`: stopped early. Smaller batch did not help; lift-progress
  fell from `2.82` to near zero by update `96`, and grasp disappeared after
  update `92`. This suggests the fragile v5 behavior is not being preserved by
  PPO sampling/batch changes alone.

### Run 4: grasp/lift reward-weight ablation

- Status: running.
- W&B: [`1mgcefpa`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/1mgcefpa)
- Output directory:
  `outputs/train/act_sim_ppo_v5ep90_grasp_lift_reward_r4`
- Code change: strengthen grasp persistence and lift incentives in
  `simulation_code/so101_mujoco_utils.py` for this ablation.
- Reward profile: `bilateral_grasp_bonus=1.10`,
  `grasp_persistence_reward=0.45`, gripped lift-progress cap/scale `2.0` /
  `30.0`, contact-only lift-progress cap/scale `0.35` / `8.0`,
  `lift_bonus=2.0`, `success_lift_bonus=6.0`.
- Hypothesis: all first three continuations lose grasp before lift can emerge.
  Boosting grasp persistence and early lift-progress reward may keep the policy
  in the grasp/contact phase long enough to discover actual block height gain.
- Started at local W&B run directory:
  `simulation_code/wandb/run-20260626_020544-1mgcefpa`.
- Early W&B metrics at updates `91`-`93`: best early signal so far. Update
  `91` has `lift_progress_reward=4.05`, `contact_steps=220`,
  `grasp_steps=72`, and strong grasp reward components; update `93` recovers
  with `lift_progress_reward=2.70`, `contact_steps=203`, `grasp_steps=66`.
  No lift steps or successes yet, and `train/clip_fraction` remains `0`.
- Updates `94`-`103`: keep running. The signal alternates with failed rollouts,
  but strong updates are getting stronger: update `101` reached
  `lift_progress_reward=5.02`, `contact_steps=189`, `grasp_steps=102`, and
  update `103` reached `lift_progress_reward=8.55`, `contact_steps=249`,
  `grasp_steps=97`. Still no lift steps or successes, but this is the first run
  where the target lift-progress component is rising instead of fading.
- Updates `104`-`118`: keep running. Lift-progress no longer flatlines and
  remains frequently positive after the first window, with peaks at update `105`
  (`9.19`) and continued nonzero signal through update `118`. Actual
  `lift_steps` and success remain `0`. Grasp weakened after update `105`, but
  still recovers intermittently, including update `117` with `grasp_steps=58`.
- Updates `119`-`132`: strongest branch so far. Returns are often positive,
  lift-progress stays sustained with peaks at update `121` (`12.30`) and `122`
  (`13.70`), contact is frequently `300`-`460` steps, and grasp often exceeds
  `100` steps. Still no `lift_steps`, `lift_bonus_reward`, or success, so keep
  watching for whether this becomes real lift or just height-gain shaping.
- Checkpoints written: `act_sim_ppo_checkpoint.pt` and
  `act_sim_ppo_checkpoint_ep0115.pt`.
- Updates `133`-`148`: still no real lift, but contact/grasp remain strong.
  Examples: update `134` has `contact_steps=480`, `grasp_steps=209`; update
  `148` has `contact_steps=464`, `grasp_steps=224`, and return `26.39`.
  `lift_progress_reward` remains positive but not enough to trigger
  `lift_steps`/`lift_bonus_reward`. Snapshot `act_sim_ppo_checkpoint_ep0140.pt`
  has been written.
- Updates `149`-`162`: dense lift-progress is high and sustained, with multiple
  updates in the `9`-`14` range and strong contact/grasp, but still no
  `lift_steps`, `lift_bonus_reward`, or success. This may be learning small
  height gains below the lift threshold. Give it one more window; if no lift
  events appear, consider a staged low-lift-threshold curriculum run instead of
  just continuing dense shaping.
- Updates `163`-`177`: stopped. The run faded after the strong mid-window:
  lift-progress dropped to small values, `grasp_steps` went to `0`, and no
  `lift_steps` or successes appeared. The reward boost produced the best dense
  signal so far, but did not convert into real threshold-crossing lift.

### Run 5: low-lift-threshold curriculum with Run 4 reward boost

- Status: running.
- W&B: [`46qbw7r2`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/46qbw7r2)
- Output directory:
  `outputs/train/act_sim_ppo_v5ep90_low_lift_threshold_r5`
- Code change: keep the Run 4 grasp/lift reward boost and lower the curriculum
  thresholds so partial real lifts become terminal/rewarded sooner.
- Reward profile: `lift_bonus_threshold=0.025`, `lift_threshold=0.045`,
  plus the Run 4 grasp/lift reward weights.
- Hypothesis: Run 4 learns small height gains but never crosses the original
  lift thresholds. A lower threshold can turn those partial lifts into sparse
  events, creating a curriculum before returning to the original threshold.
- Started at local W&B run directory:
  `simulation_code/wandb/run-20260626_022700-46qbw7r2`.
- Early W&B metrics at updates `91`-`93`: nonzero lift-progress and contact,
  but no low-threshold lift events yet. Update `91` has
  `lift_progress_reward=3.46`, `contact_steps=194`, `grasp_steps=73`; update
  `93` has `lift_progress_reward=1.76`, `contact_steps=153`,
  `grasp_steps=11`. `lift_steps`, `lift_bonus_reward`, and success remain `0`.
- Updates `94`-`108`: no low-threshold lift events yet. The run nearly
  flatlined from `94`-`104`, then recovered at `105`-`108` with
  `lift_progress_reward` up to `4.27`, contact up to `302`, and some grasp.
  Give it one more short window; stop if the recovery fails to become lift
  events or collapses back to zero.
- Updates `109`-`125`: stopped. No low-threshold lift events appeared, and the
  run was weaker than Run 4. Starting the low-threshold curriculum from raw v5
  did not preserve the high-contact/high-grasp state.

### Run 6: low-threshold curriculum from Run 4 ep140 snapshot

- Status: running.
- W&B: [`lj6i0xlg`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/lj6i0xlg)
- Output directory:
  `outputs/train/act_sim_ppo_r4ep140_low_lift_threshold_r6`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_v5ep90_grasp_lift_reward_r4/act_sim_ppo_checkpoint_ep0140.pt`
- Code profile: same low-threshold reward profile as Run 5.
- Hypothesis: Run 4 ep140 already has strong contact/grasp and sustained dense
  lift-progress. Applying the lower lift threshold from that stronger policy may
  convert partial height gains into actual sparse lift events better than
  restarting from raw v5.
- Started at local W&B run directory:
  `simulation_code/wandb/run-20260626_023608-lj6i0xlg`.
- Early W&B metrics at updates `141`-`143`: healthier than Run 5, as expected
  from the Run 4 ep140 handoff. Lift-progress is sustained (`6.09`, `2.88`,
  `4.10`), contact is `223`-`315`, and grasp is `36`-`133`. Still no
  low-threshold `lift_steps`, `lift_bonus_reward`, or success.
- Updates `144`-`158`: keep running. Run 6 stays healthier than Run 5, with
  continued nonzero lift-progress and intermittent strong grasp/contact. Best
  recent update is `158`: `lift_progress_reward=10.23`, `contact_steps=317`,
  `grasp_steps=132`. Still no low-threshold lift events.
- Updates `159`-`173`: still no low-threshold lift events. There was a strong
  burst at update `166` (`return=33.47`, `contact_steps=468`,
  `grasp_steps=282`, `lift_progress_reward=8.31`), but `lift_steps`,
  `lift_bonus_reward`, and success remain `0`. Snapshot
  `act_sim_ppo_checkpoint_ep0165.pt` has been written. If the next window fades
  without lift, pivot to a shaped upward-motion reward while grasped.
- Updates `174`-`187`: stopped. The run hit the flatline rule: after update
  `180`, lift-progress stayed at or near `0`, contact/grasp vanished, and no
  low-threshold lift events appeared.

### Run 7: upward-motion shaping from Run 6 ep165 snapshot

- Status: running.
- W&B: [`8rvtoq08`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/8rvtoq08)
- Output directory:
  `outputs/train/act_sim_ppo_r6ep165_grasp_lift_motion_r7`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r4ep140_low_lift_threshold_r6/act_sim_ppo_checkpoint_ep0165.pt`
- Code change: add `grasp_lift_motion_reward`, paid only when currently or
  previously gripped and the gripper moves upward. This is meant to encourage
  the actual lift action after the policy reaches the strong contact/grasp
  state seen at Run 6 update `166`.
- Reward profile: Run 6 low-threshold profile plus
  `grasp_lift_motion_reward_scale=12.0`,
  `grasp_lift_motion_reward_cap=0.35`.
- Started at local W&B run directory:
  `simulation_code/wandb/run-20260626_024840-8rvtoq08`.
- Early W&B metrics at updates `166`-`168`: the new upward-motion reward is
  active. `grasp_lift_motion_reward` is `3.01`, `1.25`, `2.68`; contact/grasp
  are strong (`contact_steps=232`-`404`, `grasp_steps=177`-`365`), and returns
  are high. Still no `lift_steps`, `lift_bonus_reward`, or success.
- Updates `169`-`182`: strongest run so far, but still no sparse lift events.
  Returns stay frequently high, contact/grasp remain strong, and
  `lift_progress_reward` reaches `16.59` at update `181`.
  `grasp_lift_motion_reward` is active early, then smaller later. Keep running
  toward the first snapshot, but this may still be exploiting dense shaping
  without actually crossing even the lowered lift threshold.
- Updates `183`-`198`: still no `lift_steps`, `lift_bonus_reward`, or success,
  but dense metrics remain very strong. Update `191` reached
  `lift_progress_reward=23.37`, `contact_steps=440`, `grasp_steps=332`; updates
  `194`-`198` keep high contact/grasp and nonzero upward-motion reward. Snapshot
  `act_sim_ppo_checkpoint_ep0190.pt` has been written. Keep running while
  healthy, but this likely needs visual inspection or max-block-height logging
  to determine whether it is lifting slightly below threshold or reward hacking.
- Updates `199`-`212`: still no sparse lift events. The dense signal remained
  strong through update `207` (`lift_progress_reward=22.42`,
  `contact_steps=460`, `grasp_steps=267`), then weakened after `210`. Continue
  while not flatlined, but next instrumentation should log max block height or
  visually inspect the ep190/current checkpoint.
- Updates `213`-`227`: still no sparse lift events. The run is not flatlined,
  but contact/grasp are less consistently strong than the best mid-window.
  Snapshots through `act_sim_ppo_checkpoint_ep0215.pt` have been written.
  Continue while nonzero, but this branch increasingly looks like dense
  shaping without thresholded lift.
- Updates `228`-`242`: still no `lift_steps`, `lift_bonus_reward`, or success.
  The run continues to produce intermittent strong windows, such as update `228`
  (`return=35.20`, `contact_steps=420`, `grasp_steps=252`) and update `240`
  (`return=31.61`, `contact_steps=305`, `grasp_steps=241`), but it has not
  crossed the lowered lift threshold.
- Updates `243`-`256`: still no sparse lift events. Dense reward remains
  nonzero and the run has not flatlined, but it is not converting into
  `lift_steps`. Snapshot `act_sim_ppo_checkpoint_ep0240.pt` has been written.
- Updates `257`-`282`: still no sparse lift events. The run continues to produce
  intermittent nonzero lift-progress and some strong contact/grasp windows, but
  it remains a dense-shaping result rather than a lift-success result. Snapshot
  `act_sim_ppo_checkpoint_ep0265.pt` has been written.
- Final updates `283`-`299`: finished with no `lift_steps`, `lift_bonus_reward`,
  or success. The run faded late, with lift-progress near zero across several
  updates. Final checkpoint and snapshots through
  `act_sim_ppo_checkpoint_ep0290.pt` are available. Conclusion: upward-motion
  shaping improved dense reward and contact/grasp but still did not produce
  thresholded lift, even with the lowered lift threshold.

### Run 8: max-height telemetry from Run 7 ep190 snapshot

- Status: preparing.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/b6l2ran9
- Output directory:
  `outputs/train/act_sim_ppo_r7ep190_height_telemetry_r8`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r6ep165_grasp_lift_motion_r7/act_sim_ppo_checkpoint_ep0190.pt`
- Code change: add `rollout/max_block_height_gain` telemetry so the next run can
  distinguish true sub-threshold block lift from dense reward hacking.
- Hypothesis: Run 7 may be producing small real height gains below the threshold,
  or may be exploiting contact/gripper motion without moving the block. Height
  telemetry is needed before more reward changes.
- Update `191`: `max_block_height_gain=0.0053m`, `lift_progress_reward=7.72`,
  `contact_steps=344`, `grasp_steps=202`, `lift_steps=0`, success `0`. This
  suggests the policy is producing small real block movement, but only about 5mm,
  far below the low-threshold lift target (`0.045m`) and lift bonus threshold
  (`0.025m`).
- Updates `191`-`201`: stopped as a diagnostic run, not because of
  lift-progress flatline. `lift_progress_reward` stayed active (`4.85`-`20.85`),
  but `lift_steps` and success stayed `0`. `max_block_height_gain` peaked at
  `0.0085m` and was usually `0.005`-`0.007m`. Conclusion: the policy is creating
  real but tiny block-height gains, far below even the previously lowered
  threshold. Next branch should turn this into a curriculum target rather than
  keeping the `0.045m` success threshold immediately.

### Run 9: ultra-low lift threshold curriculum

- Status: stopped; invalid threshold semantics.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/vjlocldd
- Output directory:
  `outputs/train/act_sim_ppo_r7ep190_ultralow_lift_threshold_r9`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r6ep165_grasp_lift_motion_r7/act_sim_ppo_checkpoint_ep0190.pt`
- Code change: lower `lift_bonus_threshold` from `0.025m` to `0.006m` and
  `lift_threshold` from `0.045m` to `0.008m`.
- Hypothesis: the current policy already reaches about `5`-`8.5mm` block-height
  gain. Treating that as an explicit curriculum success should expose sparse
  lift/success signals and give PPO a cleaner stepping stone before restoring
  higher thresholds.
- Early result: stopped immediately. It showed `success=2` and `lift_steps=24`,
  but with `contact_steps=0`, `grasp_steps=0`, `lift_progress_reward=0`, and
  `max_block_height_gain=0`. This exposed a reward bug/semantics mismatch:
  sparse `block_lifted` and `done` thresholds were compared against absolute
  block `z`, not `block_height_gain`, so thresholds below the resting block
  height create false successes.

### Run 10: gain-based ultra-low lift threshold curriculum

- Status: preparing.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/rx8y9w3n
- Output directory:
  `outputs/train/act_sim_ppo_r7ep190_gain_lift_threshold_r10`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r6ep165_grasp_lift_motion_r7/act_sim_ppo_checkpoint_ep0190.pt`
- Code change: compare sparse `block_lifted` and `done` thresholds against
  `block_height_gain`, not absolute block `z`; keep `lift_bonus_threshold=0.006m`
  and `lift_threshold=0.008m`.
- Hypothesis: with gain-based sparse checks, the observed `5`-`8.5mm` true height
  gains can become legitimate curriculum lift events without false positives at
  reset.
- Update `191`: valid first row after the gain-based fix:
  `contact_steps=355`, `grasp_steps=234`, `lift_progress_reward=5.36`,
  `max_block_height_gain=0.0038m`, `lift_steps=0`, success `0`. No false
  successes at reset.
- Updates `192`-`197`: first legitimate gain-based sparse lift event. Update
  `192` reached `max_block_height_gain=0.00826m`, `lift_steps=3`,
  `lift_bonus_reward=6`, `success_lift_bonus=6`, and `success=0.0833` with
  strong contact/grasp (`348`/`219`). Later updates faded, though update `197`
  still crossed the bonus threshold (`lift_steps=6`, `max_block_height_gain=0.0065m`)
  without full success. Continue at least toward the next snapshot while
  monitoring for a zero flatline.
- Updates `198`-`207`: legitimate curriculum lift events are recurring. After
  weak updates `198`-`200`, update `201` crossed the bonus threshold
  (`lift_steps=8`, `max_block_height_gain=0.0070m`), updates `203`-`204` had
  substantial bonus-threshold lift steps (`23` and `70`), and updates `206`-`207`
  again reached full success (`success=0.0833`) around `0.0081m` max height gain.
  This is the first branch with real sparse lift/success after correcting the
  threshold semantics.
- Updates `208`-`217`: stopped after snapshot `act_sim_ppo_checkpoint_ep0215.pt`
  was written. The curriculum signal persisted: updates `208`-`209` had success,
  updates `210`-`215` kept bonus-threshold lift steps, update `216` reached
  `success=0.1667`, and update `217` reached `success=0.0833`. Best max height
  in this window was `0.00939m`. Conclusion: gain-based sparse checks plus an
  `8mm` success target successfully produce real curriculum lift events.

### Run 11: gain-based 1cm lift threshold curriculum

- Status: preparing.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/1lpe58fa
- Output directory:
  `outputs/train/act_sim_ppo_r10ep215_gain_1cm_threshold_r11`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r7ep190_gain_lift_threshold_r10/act_sim_ppo_checkpoint_ep0215.pt`
- Code change: keep gain-based sparse lift checks, raise `lift_bonus_threshold`
  from `0.006m` to `0.008m`, and raise `lift_threshold` from `0.008m` to
  `0.010m`.
- Hypothesis: Run 10 repeatedly reached `8`-`9.4mm`; using that as the bonus
  threshold and `10mm` as the success threshold should encourage the next step
  of vertical block motion without jumping too far.
- Updates `216`-`218`: no 1cm success yet, but the raised bonus threshold is
  already active. Update `216` reached `max_block_height_gain=0.00870m` with
  `lift_steps=6`; update `218` reached `0.00909m` with `lift_steps=4` and strong
  contact/grasp (`388`/`182`). Continue; lift-progress is not flatlined.
- Updates `219`-`228`: still no 1cm success. The run repeatedly crosses the
  `8mm` bonus threshold (`lift_steps=3` at `220`, `7` at `221`, `16` at `222`)
  and peaks near `0.00908m`, but it is not yet extending to `0.010m`. A couple
  updates had `lift_progress_reward=0`, but not a sustained flatline; continue
  toward the ep240 snapshot.
- Updates `229`-`241`: snapshot `act_sim_ppo_checkpoint_ep0240.pt` written. Still
  no 1cm success, but the ceiling is rising: update `238` reached
  `max_block_height_gain=0.00948m`, and update `240` reached `0.00967m` with
  `lift_steps=7`, `contact_steps=232`, and `grasp_steps=135`. Continue briefly;
  it is within about `0.33mm` of the 1cm threshold.
- Updates `242`-`252`: stopped. No 1cm success appeared after the ep240 snapshot,
  and the post-snapshot window regressed below the `8mm` bonus threshold despite
  nonzero lift-progress. Use ep240, the best checkpoint from this run, for an
  intermediate `9.5mm` bridge target rather than continuing the fading branch.

### Run 12: gain-based 9.5mm bridge threshold

- Status: preparing.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/7gv1cduw
- Output directory:
  `outputs/train/act_sim_ppo_r11ep240_gain_9p5mm_threshold_r12`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r10ep215_gain_1cm_threshold_r11/act_sim_ppo_checkpoint_ep0240.pt`
- Code change: keep gain-based sparse lift checks, set
  `lift_bonus_threshold=0.0085m` and `lift_threshold=0.0095m`.
- Hypothesis: Run 11 reached `9.67mm` once but could not consolidate a `10mm`
  target. A `9.5mm` bridge should turn the best observed behavior into sparse
  success while still pushing above the previous `8mm` curriculum.
- Update `241`: immediate legitimate bridge success. `max_block_height_gain`
  reached `0.01014m`, so the policy actually crossed `1cm` even though the run's
  success threshold is `9.5mm`. Metrics: `lift_steps=2`, `success=0.0833`,
  `lift_progress_reward=8.97`, `contact_steps=293`, `grasp_steps=124`. Continue
  to see whether it repeats and to save a snapshot.
- Updates `242`-`253`: stopped. The 1cm crossing did not repeat; the branch kept
  nonzero lift-progress but fell below the `9.5mm` success threshold and did not
  reach an early snapshot. Treat the update `241` success as evidence that the
  Run 11 ep240 policy distribution can cross 1cm, but PPO updates from this
  branch did not preserve it.

### Run 13: 9.5mm bridge with lower LR preservation

- Status: preparing.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/mnqa5xu3
- Output directory:
  `outputs/train/act_sim_ppo_r11ep240_gain_9p5mm_low_lr_r13`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r10ep215_gain_1cm_threshold_r11/act_sim_ppo_checkpoint_ep0240.pt`
- Code profile: same gain-based `lift_bonus_threshold=0.0085m` and
  `lift_threshold=0.0095m` as Run 12.
- Command change: lower `policy-lr` to `3e-6`, lower `critic-lr` to `5e-5`, and
  snapshot every `5` updates so early successes are not overwritten before being
  captured.
- Hypothesis: Run 12 proved the ep240 policy can cross 1cm, but standard PPO
  updates quickly moved away from it. Gentler updates may preserve and consolidate
  the crossing.
- Updates `241`-`243`: lower LR is preserving bonus-threshold lift but has not
  repeated the `9.5mm` success yet. Update `241` reached `0.00895m` with
  `lift_steps=4`; update `243` reached `0.00871m` with `lift_steps=2` and strong
  contact/grasp. Continue to the first frequent snapshot point.
- Updates `244`-`258`: stopped after snapshot `act_sim_ppo_checkpoint_ep0255.pt`
  was written. Run 13 repeated the bridge/near-1cm success: update `253` reached
  `max_block_height_gain=0.01002m`, `lift_steps=7`, `success=0.0833`, and update
  `256` reached `0.00971m` with `success=0.0833`. This is the best preserved
  checkpoint so far for lifting above the 9.5mm bridge and occasionally crossing
  1cm.

### Run 14: gain-based strict 1cm threshold

- Status: preparing.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/k9yofm31
- Output directory:
  `outputs/train/act_sim_ppo_r13ep255_gain_1cm_threshold_r14`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r11ep240_gain_9p5mm_low_lr_r13/act_sim_ppo_checkpoint_ep0255.pt`
- Code change: keep gain-based sparse lift checks, set
  `lift_bonus_threshold=0.0095m` and `lift_threshold=0.010m`.
- Command profile: keep Run 13's lower `policy-lr=3e-6`, `critic-lr=5e-5`, and
  frequent snapshots.
- Hypothesis: Run 13 ep255 has now captured repeated 9.5mm successes and one
  genuine 1cm crossing. Promoting 1cm to the sparse success threshold should
  consolidate that behavior if the lower LR preserves it.
- Updates `256`-`258`: no strict 1cm success yet. Best early row is update `256`
  with `max_block_height_gain=0.00927m`, `contact_steps=205`,
  `grasp_steps=104`, and `lift_progress_reward=8.92`, just below the `9.5mm`
  bonus threshold. Continue through the first snapshot window.
- Updates `259`-`265`: stopped. No strict 1cm success and no `9.5mm` bonus
  crossings. The run stayed active, but the strict bonus threshold was too high
  for this checkpoint; best max height remained `0.00927m`. Next branch should
  keep `1cm` as success but lower the bonus bridge to `9mm`.

### Run 15: gain-based 9mm bonus, strict 1cm success

- Status: preparing.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/o05pn6ke
- Output directory:
  `outputs/train/act_sim_ppo_r13ep255_gain_9mm_bonus_1cm_success_r15`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r11ep240_gain_9p5mm_low_lr_r13/act_sim_ppo_checkpoint_ep0255.pt`
- Code change: keep `lift_threshold=0.010m`, lower `lift_bonus_threshold` from
  `0.0095m` to `0.0090m`.
- Command profile: keep lower `policy-lr=3e-6`, `critic-lr=5e-5`, and frequent
  snapshots.
- Hypothesis: Run 14 frequently reached about `9.1`-`9.3mm` but received no
  sparse lift bonus. A `9mm` bridge may reinforce those near-misses while keeping
  the sparse success target at `1cm`.
- Updates `256`-`258`: no strict 1cm success yet, but the bridge is working.
  Update `256` reached `max_block_height_gain=0.00983m`, `lift_steps=3`,
  `lift_progress_reward=10.17`, `contact_steps=240`, and `grasp_steps=131`. This
  is within `0.17mm` of the `1cm` success threshold.
- Updates `259`-`266`: stopped. The near-1cm first row did not repeat, and the
  run faded below the `9mm` bonus bridge. Conclusion: the bridge helps identify
  near-misses, but sparse rewards may still be too weak/rare to consolidate the
  1cm behavior from the Run 13 ep255 checkpoint.

### Run 16: stronger sparse 1cm consolidation

- Status: preparing.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/iuevxmno
- Output directory:
  `outputs/train/act_sim_ppo_r13ep255_gain_9mm_bonus_1cm_strong_sparse_r16`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r11ep240_gain_9p5mm_low_lr_r13/act_sim_ppo_checkpoint_ep0255.pt`
- Code profile: keep `lift_bonus_threshold=0.0090m` and
  `lift_threshold=0.010m`, increase `lift_bonus` from `2.0` to `3.0`, and
  increase `success_lift_bonus` from `6.0` to `12.0`.
- Command profile: keep lower `policy-lr=3e-6`, `critic-lr=5e-5`, and frequent
  snapshots.
- Hypothesis: Run 15 reached `9.83mm` immediately but did not consolidate. A
  stronger sparse reward at the same thresholds may make rare 9mm/1cm crossings
  dominate the PPO update enough to preserve them.
- Updates `256`-`258`: no 9mm/1cm event yet. Best early max height is
  `0.00829m` at update `256` with strong contact/grasp and
  `lift_progress_reward=8.88`. Continue to first snapshot window unless
  lift-progress flatlines.
- Updates `259`-`266`: stopped. Stronger sparse bonuses did not help from this
  checkpoint; the run never crossed the `9mm` bridge in the first snapshot
  window. Best max height was `0.00853m`, despite repeated high dense
  lift-progress updates. The best useful handoff remains Run 13 ep255, while the
  best measured height event remains Run 12 update `241` / Run 13 update `253`
  at about `1cm`.

## Next Overnight Prep

- Added stochastic PPO inference support to `run_act_ppo_sim_inference.py`:
  use `--stochastic` to sample from the PPO policy distribution instead of
  showing only the deterministic mean action. The script now also reports
  `max_block_height_gain`, matching the gain metric used by training.
- Smoke test command from Run 13 ep255:
  `run_act_ppo_sim_inference.py --resume outputs/train/act_sim_ppo_r11ep240_gain_9p5mm_low_lr_r13/act_sim_ppo_checkpoint_ep0255.pt --init-checkpoint outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model --episodes 1 --max-steps-per-episode 60 --chunk-size 30 --steps-per-action 1 --headless --stochastic --seed 13`
- Smoke test result: loaded successfully and reached
  `max_block_height_gain=0.0088m` in 60 steps with contact/grasp present. This is
  below strict success, but it confirms stochastic inference is exercising the
  same kind of sampled behavior as PPO training.
- Next overnight-ready code profile: gain-based sparse lift checks,
  `lift_bonus_threshold=0.0090m`, `lift_threshold=0.010m`, default sparse bonuses
  (`lift_bonus=2.0`, `success_lift_bonus=6.0`).
- Next starting checkpoint:
  `outputs/train/act_sim_ppo_r11ep240_gain_9p5mm_low_lr_r13/act_sim_ppo_checkpoint_ep0255.pt`.
- Recommended next ladder after repeatable 1cm: `0.010m -> 0.015m -> 0.020m ->
  0.030m -> 0.045m`.

### Run 17: overnight 1cm consolidation from Run 13 ep255

- Status: finished.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/st01r7h2
- Output directory:
  `outputs/train/act_sim_ppo_r13ep255_gain_9mm_1cm_overnight_r17`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r11ep240_gain_9p5mm_low_lr_r13/act_sim_ppo_checkpoint_ep0255.pt`
- Code profile: gain-based sparse lift checks, `lift_bonus_threshold=0.0090m`,
  `lift_threshold=0.010m`, default sparse bonuses (`lift_bonus=2.0`,
  `success_lift_bonus=6.0`).
- Command profile: lower `policy-lr=3e-6`, `critic-lr=5e-5`, 12 parallel envs,
  2 rollout chunks per env, 1 PPO epoch, minibatch 64, chunk size 30,
  `max_steps_per_episode=150`, snapshots every 10 updates, target episode/update
  `900`.
- Hypothesis: Run 13 ep255 is the best checkpoint with repeated 9.5mm/near-1cm
  events. A longer, lower-LR overnight continuation with the 9mm bridge should
  consolidate repeatable 1cm lift before raising the curriculum threshold.
- Monitoring rule: if `rollout/reward_components/lift_progress_reward` flatlines
  at `0` for a sustained window, stop, note the failure, and pivot rather than
  spending the overnight budget.
- Final W&B summary: completed through update `899` with snapshots through
  `act_sim_ppo_checkpoint_ep0895.pt`. The strongest metric window was around
  updates `869`-`871`, with max height gains of about `0.0107`-`0.0116m` and
  sparse success `0.0833`. The final updates regressed to dense lift-progress
  without sparse lift/success, so the latest checkpoint is not the best visual
  policy.
- Best live-inspected checkpoint:
  `outputs/train/act_sim_ppo_r13ep255_gain_9mm_1cm_overnight_r17/act_sim_ppo_checkpoint_ep0875.pt`.
  User observation from live stochastic sim inference: this snapshot is much
  better than the final checkpoint. It approaches the block, tries to grasp, and
  starts to attempt lift. The remaining blocker is not lack of lift intent; it is
  gripper/block alignment. The gripper does not consistently orient around the
  block in a way that permits a proper grasp, so the block is nudged/partially
  lifted rather than securely picked up.
- Next research direction: use Run 17 `ep0875` as the preferred handoff and focus
  on pre-grasp alignment/orientation shaping before increasing lift thresholds.
  Candidate signals: reward closing only when the gripper is centered around the
  block, penalize lifting/closing while horizontally or yaw-misaligned, add a
  gripper-facing/block-corridor alignment bonus, or add short supervised/RL
  warm-start clips for correct gripper pose at contact.

### Run 18: pre-grasp alignment shaping from Run 17 ep875

- Status: stopped early; alignment penalties too strong.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/qhl5d2ki
- Output directory:
  `outputs/train/act_sim_ppo_r17ep875_pregrasp_alignment_r18`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r13ep255_gain_9mm_1cm_overnight_r17/act_sim_ppo_checkpoint_ep0875.pt`
- Code profile: keep gain-based sparse lift checks with `lift_bonus_threshold=0.0090m`
  and `lift_threshold=0.010m`; add pre-grasp alignment score/reward, aligned-close
  reward, misaligned-close penalty, and misaligned-lift penalty.
- Command profile: lower `policy-lr=3e-6`, `critic-lr=5e-5`, 12 parallel envs,
  2 rollout chunks per env, 1 PPO epoch, minibatch 64, chunk size 30,
  `max_steps_per_episode=150`, snapshots every 10 updates, target update `1100`.
- Hypothesis: Run 17 ep875 already tries to grasp and lift but fails to align the
  gripper around the block. Alignment shaping should improve centered approach
  and close timing before raising lift thresholds beyond the 1cm curriculum.
- Updates `876`-`878`: stopped quickly. The new metrics logged correctly, but the
  reward balance was too punitive: `pregrasp_alignment_reward` stayed at `0`
  while `misaligned_lift_penalty` became large, including about `-30` at update
  `877`. This risked training away from the useful Run 17 behavior, so the run
  was stopped and the alignment penalties were softened before relaunch.

### Run 19: softened pre-grasp alignment shaping from Run 17 ep875

- Status: stopped early; alignment reward still inactive.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/8vpl5xyl
- Output directory:
  `outputs/train/act_sim_ppo_r17ep875_pregrasp_alignment_soft_r19`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r13ep255_gain_9mm_1cm_overnight_r17/act_sim_ppo_checkpoint_ep0875.pt`
- Code profile: same alignment features as Run 18, but softened:
  `pregrasp_alignment_reward_scale=0.12`, `aligned_close_reward=0.08`,
  `misaligned_close_penalty=-0.03`, `misaligned_lift_penalty=-0.02`, and
  pre-grasp aligned threshold lowered to `0.25`.
- Hypothesis: the softer version should provide a positive alignment gradient
  before contact without overwhelming the existing grasp/lift behavior from Run
  17 ep875.
- Updates `876`-`878`: healthier than Run 18. The positive alignment reward still
  has not fired, but penalties are much smaller and the policy preserved useful
  lift behavior: update `877` reached `max_block_height_gain=0.00970m`,
  `lift_steps=4`, `contact_steps=471`, `grasp_steps=97`, and
  `lift_progress_reward=17.40`. Continue to the first snapshot and watch whether
  pre-grasp alignment reward begins to appear.
- Updates `879`-`887`: stopped after first snapshot
  `act_sim_ppo_checkpoint_ep0885.pt`. The run preserved strong contact/grasp and
  high dense lift-progress, but `pregrasp_alignment_reward` never activated. This
  made it a contact/lift continuation rather than a real alignment run. The
  alignment reward threshold/conditions were loosened before relaunch.

### Run 20: partial pre-grasp alignment shaping from Run 17 ep875

- Status: running.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/gy45tl7j
- Output directory:
  `outputs/train/act_sim_ppo_r17ep875_pregrasp_alignment_partial_r20`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r13ep255_gain_9mm_1cm_overnight_r17/act_sim_ppo_checkpoint_ep0875.pt`
- Code profile: same alignment features, but reward any positive pre-grasp
  alignment score before contact and lower the recent-alignment threshold to
  `0.05`. Keep softened penalties from Run 19.
- Hypothesis: partial alignment reward should finally create a positive W&B
  signal before contact, while preserving the useful grasp/lift behavior from
  Run 17 ep875.
- Updates `876`-`878`: the positive alignment signal is now active. Updates
  `876` and `877` logged nonzero `pregrasp_alignment_reward`, while preserving
  contact/grasp and lift-progress. Best early height was
  `max_block_height_gain=0.00853m` at update `876`. Continue to the first
  snapshot to see whether the alignment signal grows without collapsing grasp.
- Updates `879`-`888`: first snapshot
  `act_sim_ppo_checkpoint_ep0885.pt` written. The run preserved a 9mm lift-bonus
  event at update `884` (`lift_steps=2`, `max_block_height_gain=0.00901m`) and
  finally produced a clearer positive alignment/close signal at update `885`
  (`pregrasp_alignment_reward=0.0218`, `aligned_close_reward=0.0047`). This is
  the first alignment branch where the intended positive shaping is actually
  visible in W&B.
- Updates `889`-`892`: stopped. After the first snapshot, alignment reward
  disappeared again, misaligned-close penalties stayed high, and grasp/lift
  metrics weakened. A short stochastic smoke check from `ep0885` loaded
  successfully but was visually/metric weak (`max_block_height_gain=0.0033m`,
  low contact/grasp). Conclusion: the geometric alignment signal is wired and can
  activate, but it is too sparse/poorly targeted to improve the policy reliably.
  Do not continue this branch as-is; the next alignment attempt should likely use
  a better grasp-pose target, e.g. explicit jaw-tip sites or demonstration-derived
  pregrasp poses, rather than only gripper-frame centering.

### Run 21: jaw-centered grasp reward from Run 17 ep875

- Status: stopped after short ablation; useful grasp/contact improvement, no lift
  conversion.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/lbwdw5md
- Output directory:
  `outputs/train/act_sim_ppo_r17ep875_jaw_centered_grasp_r21`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r13ep255_gain_9mm_1cm_overnight_r17/act_sim_ppo_checkpoint_ep0875.pt`
- Code profile: added non-physical MuJoCo diagnostic sites `fixed_jaw_tip` and
  `moving_jaw_tip`; replaced the failed body-frame geometric alignment profile
  with jaw-gap centering diagnostics/reward (`jaw_centering_score`,
  `jaw_lateral_error`, `jaw_depth_error`, `jaw_gap_width`). Closing is rewarded
  only when the block is recently centered between the jaws, side-pushing contact
  is penalized when jaw centering is poor, and sparse lift bonuses/success are
  gated on `gripped` or recent jaw-centered alignment/contact. The 1cm curriculum
  remains active with `lift_bonus_threshold=0.0090m` and `lift_threshold=0.010m`.
- Command profile: lower `policy-lr=3e-6`, `critic-lr=5e-5`, 12 parallel envs,
  2 rollout chunks per env, 1 PPO epoch, minibatch 64, chunk size 30,
  `max_steps_per_episode=150`, snapshots every 10 updates, target update `980`.
- Pre-launch validation: XML/model load succeeds and exposes `fixed_jaw_tip`,
  `moving_jaw_tip`, and `gripperframe`; Python syntax checks pass. A 1-episode
  stochastic smoke test from Run 17 `ep0875` completed successfully with
  `max_block_height_gain=0.0020m`, `contact_count=30`, `grasp_count=0`.
- Stop rules: stop quickly if `lift_progress_reward` flatlines at `0`,
  contact/grasp collapse for several updates, or the new `jaw_centering_score` /
  `pregrasp_alignment_reward` pathway stays zero through the first snapshot
  window.
- Outcome through final checkpoint `act_sim_ppo_checkpoint.pt` (`start_episode=911`
  when reloaded): not a flatline. W&B showed repeated high-return updates with
  stronger contact/grasp than Run 17, including update `902` with
  `contact_steps=640`, `grasp_steps=542`, and `lift_progress_reward=12.80`.
  The best height spike was update `894` with
  `max_block_height_gain=0.01037m`, but gated `lift_bonus_reward` stayed `0`,
  indicating the lift event did not coincide with the grasp-ready gate. A
  3-episode stochastic inference smoke from the checkpoint had mean return
  `86.06`, mean `contact_count=82.3`, mean `grasp_count=57.3`, but only
  `mean_max_block_height_gain=0.0043m` and `lift_count=0`. This is a good handoff
  for the grasp-gated lift ablation, not a final policy.
- User live-inspection update: this policy is also good visually. It approaches
  the arm/block quickly and tries to grasp, but most attempts still smash into
  the block and shove it around rather than wrapping the gripper around it for a
  secure pickup. Treat this as evidence that the approach/grasp-attempt timing is
  improving, while the remaining blocker is contact pose quality and impact
  control.

### Run 22: recent jaw-centered contact lift gate from Run 21 ep910

- Status: stopped after short ablation; denser gated lift signal, but no sparse
  lift conversion and weaker inference than Run 21.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/vyyza8mf
- Output directory:
  `outputs/train/act_sim_ppo_r21ep910_grasp_gated_lift_r22`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r17ep875_jaw_centered_grasp_r21/act_sim_ppo_checkpoint.pt`
- Code profile: preserve the Run 21 jaw-site centering reward and add
  `recent_jaw_centered_contact_steps`. Contact that is jaw-centered now opens a
  short lift-ready window, receives `jaw_centered_contact_reward`, and permits
  stronger contact-only lift-progress shaping plus gated sparse lift bonuses.
  This should convert Run 21's improved contact/grasp into height gain without
  rewarding side-pushing.
- Command profile: lower `policy-lr=3e-6`, `critic-lr=5e-5`, 12 parallel envs,
  2 rollout chunks per env, 1 PPO epoch, minibatch 64, chunk size 30,
  `max_steps_per_episode=150`, snapshots every 10 updates, target update `960`.
- Pre-launch validation: Python syntax/lints passed. A 1-episode stochastic
  smoke under the new reward path loaded the Run 21 checkpoint and completed with
  `return=26.19`, `contact_count=29`, `grasp_count=18`,
  `max_block_height_gain=0.0062m`, and `lift_count=0`.
- Stop rules: stop if lift-progress flatlines at `0`, if contact/grasp collapse,
  or if `jaw_centered_contact_reward`/`recent_jaw_centered_contact_steps` stay
  zero through the first few updates.
- Outcome through final checkpoint `act_sim_ppo_checkpoint.pt` (`start_episode=921`
  when reloaded): the new contact-memory pathway activated, e.g. update `911`
  logged `jaw_centered_contact_reward=0.31` and
  `recent_jaw_centered_contact_steps=720`. Dense lift-progress became very large
  at several updates (`51.10` at update `912`, `34.22` at `914`, `30.65` at
  `915`), and contact/grasp stayed strong early. However, `lift_bonus_reward`,
  `lift_steps`, and success remained `0`; max height mostly stayed below the
  9mm bonus threshold, peaking around `0.00836m` in the monitored window. A
  3-episode stochastic inference smoke from the final checkpoint was weaker than
  Run 21 (`mean_return=17.49`, `mean_contact_count=58.3`,
  `mean_grasp_count=16.0`, `mean_max_block_height_gain=0.0042m`,
  `lift_count=0`). Do not start the height ladder yet; 1cm success is not
  repeatable. Prefer Run 21's checkpoint as the current jaw-reward handoff if
  continuing this branch.

### Run 23: randomized block PPO from supervised ACT pretrain

- Status: stopped early; default LR trained away from the small initial contact
  signal.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/78o3y4to
- Output directory:
  `outputs/train/act_sim_ppo_pretrain_random_block_r23`
- Starting checkpoint:
  `outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model`
- Code profile: current ACT PPO trainer/reward stack, no PPO resume, randomized
  block reset enabled with `block_dist_range=0.22 0.26` and
  `block_angle_range=-10 10`.
- Command profile: default `policy-lr=1e-5`, default `critic-lr=1e-4`, 12
  parallel envs, 2 rollout chunks per env, 1 PPO epoch, minibatch 64, chunk size
  30, `max_steps_per_episode=150`, target update `300`.
- Hypothesis: because the physical ACT training data varies the block placement
  between episodes, starting sim PPO directly from the supervised pretrain with
  randomized block placement may better match the pretrain distribution than
  fixed/curriculum-block PPO.
- Outcome: stopped at update `22`. The first few updates had small contact and
  lift-progress signal (`contact_steps=83` and
  `lift_progress_reward=1.05` at update `1`), but updates `3` through `20`
  flatlined with `contact_steps=0`, `grasp_steps=0`, `lift_steps=0`,
  `max_block_height_gain≈0.00007m`, and `lift_progress_reward=0`. PPO metrics
  did not show a KL/clip blowup (`train/clip_fraction=0`), so this looked like
  behavioral collapse rather than optimizer instability.

### Run 24: low-LR randomized block PPO from supervised ACT pretrain

- Status: stopped after useful mid-run behavior faded.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/4ai70olk
- Output directory:
  `outputs/train/act_sim_ppo_pretrain_random_block_low_lr_r24`
- Starting checkpoint:
  `outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model`
- Code profile: same randomized-block profile as Run 23.
- Command change: lowered actor/critic learning rates to the preservation profile
  used in later ACT PPO branches: `policy-lr=3e-6`, `critic-lr=5e-5`.
- Early result through update `82`: the lower-LR version avoided the immediate
  flatline. It has recurring contact, intermittent grasp, and small real block
  height gains. Best early values include `grasp_steps=5`,
  `max_block_height_gain=0.00874m`, and
  `lift_progress_reward=1.21`. No `lift_steps` or success yet. Continue watching
  for whether contact/grasp consolidates or whether the branch fades into dense
  shaping without lift.
- Outcome: stopped at update `184` after a strong mid-run window faded. The best
  measured height event was update `149`, with
  `max_block_height_gain=0.01012m`, `contact_steps=235`,
  `grasp_steps=17`, and `lift_progress_reward=5.71`, but `lift_steps` and
  success stayed `0`. The strongest grasp/contact row was update `141`
  (`return=10.42`, `contact_steps=254`, `grasp_steps=103`,
  `lift_progress_reward=8.43`). After update `170`, grasp mostly disappeared and
  lift-progress weakened, so the run was stopped to preserve the better
  snapshots. Best handoffs: `act_sim_ppo_checkpoint_ep0149.pt` for the 1cm
  height event, and `act_sim_ppo_checkpoint_ep0124.pt` / `ep0149.pt` for the
  strong contact/grasp window.

### Run 25: conservative continuation from Run 24 ep149

- Status: completed; strongest randomized-block contact/grasp window so far, but
  no sparse lift/success.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/09100viw
- Output directory:
  `outputs/train/act_sim_ppo_r24ep149_random_block_conserve_r25`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_pretrain_random_block_low_lr_r24/act_sim_ppo_checkpoint_ep0149.pt`
- Code profile: same randomized-block profile as Run 24.
- Command change: lowered actor LR again to `policy-lr=1e-6` while keeping
  `critic-lr=5e-5`, with snapshots every 10 updates.
- Hypothesis: Run 24 ep149 had the best measured height event but later drifted.
  A more conservative continuation might preserve the 1cm/high-grasp behavior
  instead of training away from it.
- Outcome: completed through update `229`. The run preserved and improved
  contact/grasp strength, but did not improve height beyond Run 24 and still did
  not produce sparse `lift_steps` or success. Best row was update `183`:
  `return=12.58`, `contact_steps=350`, `grasp_steps=106`,
  `lift_progress_reward=14.92`, and `max_block_height_gain=0.00732m`. The best
  measured height in Run 25 was `0.00886m`, below Run 24's 1cm spike. Useful
  handoff for contact/grasp behavior: `act_sim_ppo_checkpoint_ep0189.pt`.
  Useful handoff for height remains Run 24 `act_sim_ppo_checkpoint_ep0149.pt`.
- User live-inspection update: `act_sim_ppo_checkpoint_ep0189.pt` looks good in
  live MuJoCo. The policy appears to be trying to grasp and lift, contacts the
  block more gently than the earlier "slam into the block" failures, and is
  starting to use the end of the gripper in a more promising way. Treat this as
  the current best qualitative randomized-block checkpoint for grasp/lift-attempt
  behavior, even though the logged sparse lift/success metrics are still zero.
- Interpretation: randomized-block PPO from the supervised ACT pretrain works
  much better with the lower-LR preservation profile than with default LR. The
  policy can learn strong contact/grasp under randomized block positions, but the
  current sparse lift gate still does not credit even some real height-gain
  events. Next branch should either inspect the Run 24 ep149 / Run 25 ep189
  rollouts visually or adjust the lift-credit gate so true block-height gain with
  strong contact/grasp becomes a sparse curriculum signal.

### Run 26: micro-lift long continuation from Run 25 ep189

- Status: running; overnight babysitting target is to continue/pivot until 5am
  local time with the best documented lift/success checkpoint preserved.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/7blyw1gb
- Output directory:
  `outputs/train/act_sim_ppo_r25ep189_micro_lift_long_r26`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r24ep149_random_block_conserve_r25/act_sim_ppo_checkpoint_ep0189.pt`
- Code profile: added a micro-lift curriculum before the existing strict
  lift/success gates. The new gate uses `micro_lift_bonus_threshold=0.0050`,
  `micro_lift_bonus=0.8`, and requires strong manipulation context via current
  or previous grasp, contacted `contact_lift_ready`, or sustained jaw-centered
  contact. The existing `lift_bonus_threshold=0.0090` and
  `lift_threshold=0.010` remain unchanged.
- Command summary: `policy-lr=5e-7`, `critic-lr=5e-5`, 12 parallel envs, 2
  rollout chunks per env, 1 PPO epoch, minibatch 64, chunk size 30, 150 max
  steps per episode, randomized block reset with distance range `0.22 0.26` and
  angle range `-10 10`, 600 configured episodes, snapshots every 10 updates.
- Validation before launch: `py_compile` passed for `so101_mujoco_utils.py`,
  `so101_gym_env.py`, `train_act_in_sim.py`, and
  `run_act_ppo_sim_inference.py`. A one-episode stochastic smoke test from Run
  25 ep189 loaded successfully and reported the new `mean_micro_lift_count`
  field; that short rollout had contact but no micro-lift.
- Early babysitting through update `221`: keep running. Contact/grasp are still
  active, micro-lift events have appeared, and PPO health is stable. The first
  32 synced updates averaged `contact_steps=106.8`, `grasp_steps=11.5`,
  `max_block_height_gain=0.0036m`, `lift_progress_reward=1.43`,
  `micro_lift_bonus=0.575`, `micro_lifted=0.72`, `train/clip_fraction=0`, and
  no sparse `lift_steps`. Strong early rows include update `198`
  (`max_block_height_gain=0.00517m`, `micro_lifted=10`) and update `221`
  (`contact_steps=223`, `grasp_steps=61`, `max_block_height_gain=0.00580m`,
  `micro_lifted=5`). Continue unless lift-progress/micro-lift flatline for a
  sustained window, contact/grasp collapse, or optimizer metrics spike.
- Updates `222`-`241`: keep running. This is the first R26 window to show a
  sparse lift event: update `224` reached `lift_steps=1` and
  `max_block_height_gain=0.00906m`, just over the strict bonus threshold. The
  strongest nearby row was update `238` with `return=11.52`,
  `contact_steps=327`, `grasp_steps=93`, `lift_progress_reward=11.21`,
  `micro_lifted=11`, and `max_block_height_gain=0.00891m`. Last-20 averages
  improved to `contact_steps=160.8`, `grasp_steps=30.6`,
  `max_block_height_gain=0.0061m`, `micro_lifted=2.30`, and
  `train/clip_fraction=0`. Snapshot `act_sim_ppo_checkpoint_ep0239.pt` is the
  nearest preserved checkpoint to this strong micro-lift/near-lift window.
- Updates `242`-`262`: watch closely, but keep running. The sparse lift event
  has not repeated yet, and the latest-10 averages weakened
  (`contact_steps=101.0`, `grasp_steps=15.9`, `max_block_height_gain=0.0045m`),
  but the latest-20 window still has active manipulation signal:
  `contact_steps=117.8`, `grasp_steps=25.0`, `max_block_height_gain=0.0049m`,
  `micro_lifted=4.40`, `lift_progress_reward=3.11`, and stable
  `train/clip_fraction=0`. Update `258` is the most important recent row:
  `max_block_height_gain=0.00848m` with `micro_lifted=9`, showing the policy can
  still reach near-lift height after the update-224 sparse lift. Stop or pivot
  if the next sustained window loses this micro-lift/height signal.
- Updates `263`-`282`: fading but not stopped yet. No additional sparse
  `lift_steps` appeared, and latest-20 grasp weakened to `9.3` average steps,
  but contact remains active (`128.1` average steps), max height still reaches
  `0.0078m`, and micro-lift still appears intermittently. Update `248` remains
  the best dense micro-lift row so far (`return=19.80`, `contact_steps=303`,
  `grasp_steps=140`, `lift_progress_reward=16.39`, `micro_lifted=40`,
  `max_block_height_gain=0.00849m`). Continue for one more close window; if
  contact/grasp/micro-lift continue trending downward, stop R26 and pivot from
  the best preserved snapshot rather than training away from the useful window.
- Updates `283`-`301`: recovered enough to keep running. The latest-20 window is
  still below the `238`/`248` peak, but update `296` fired
  `micro_lifted=49` with `micro_lift_bonus=39.2`,
  `lift_progress_reward=11.14`, `contact_steps=150`, `grasp_steps=54`, and
  `max_block_height_gain=0.00784m`. Latest-10 averages also improved over the
  prior watch window (`grasp_steps=20.2`, `micro_lifted=6.90`,
  `lift_progress_reward=3.54`, `train/clip_fraction=0`). No repeated sparse
  lift or success yet, but the micro-lift curriculum is still active, so do not
  stop R26 here.
- Final/abort result: the shell task ended around update `324` with
  `exit_code=unknown` and no Python traceback. Treat the final checkpoint as
  degraded relative to the mid-run snapshots: the final synced window through
  update `322` had mostly zero `grasp_steps`, zero `micro_lifted`, zero
  `lift_steps`, and no success despite occasional high-return rows such as
  update `313`. Useful outputs were preserved by snapshots: update `224` was
  the only sparse lift row (`lift_steps=1`, `lift_bonus_reward=2`,
  `max_block_height_gain=0.00906m`), while update `248` was the best dense
  micro-lift/contact row (`return=19.80`, `contact_steps=303`,
  `grasp_steps=140`, `lift_progress_reward=16.39`, `micro_lifted=40`,
  `max_block_height_gain=0.00849m`). Use
  `act_sim_ppo_checkpoint_ep0249.pt` as the next handoff because it is the
  nearest preserved checkpoint to the strongest dense micro-lift row; keep
  `act_sim_ppo_checkpoint_ep0229.pt` as the nearest sparse-lift backup.

### Run 27: lower-drift micro-lift consolidation from Run 26 ep249

- Status: running.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/fjvp1tkd
- Output directory:
  `outputs/train/act_sim_ppo_r26ep249_micro_lift_consolidate_r27`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r25ep189_micro_lift_long_r26/act_sim_ppo_checkpoint_ep0249.pt`
- Hypothesis: Run 26 discovered a useful micro-lift/contact window but then
  drifted away. Restart from the preserved peak and reduce actor LR again to
  preserve fine-grained manipulation while the critic/reward signal attempts to
  consolidate repeated `5-9mm` lifts into sparse lift events.
- Command profile: same randomized-block micro-lift reward as Run 26,
  `policy-lr=2e-7`, `critic-lr=5e-5`, 12 parallel envs, 2 rollout chunks per
  env, 1 PPO epoch, minibatch 64, chunk size 30, 150 max steps, randomized block
  distance range `0.22 0.26`, angle range `-10 10`, snapshots every 10 updates.
- Early babysitting through update `268`: mixed but not stopped. The first
  synced window preserved near-lift height at update `260`
  (`return=6.41`, `contact_steps=151`, `grasp_steps=59`,
  `max_block_height_gain=0.00868m`, `micro_lifted=6`) and had another useful
  contact/grasp row at update `258` (`contact_steps=186`, `grasp_steps=44`,
  `max_block_height_gain=0.00756m`, `micro_lifted=8`). However, averages are
  weaker than Run 26's peak: latest-19 `return=-4.44`, `contact_steps=87.5`,
  `grasp_steps=15.5`, `max_block_height_gain=0.0057m`, `micro_lifted=3.11`,
  no `lift_steps`, and `train/clip_fraction=0`. Continue one more window; stop
  if the near-lift/micro-lift signal does not recover.
- Stopped after update `288`. The lower-actor-LR continuation did not preserve
  the Run 26 peak: latest-20 averaged `return=-9.30`, `contact_steps=94.0`,
  `grasp_steps=13.3`, `micro_lifted=0.40`, and no `lift_steps`. Best R27 rows
  remained early and weaker than the R26 handoff (`ep260` at
  `max_block_height_gain=0.00868m`, but only `micro_lifted=6`; `ep255` had
  `micro_lifted=16` with `max_block_height_gain=0.00784m`). Do not use R27 as
  the next handoff.

### Run 28: sparse-lift backup consolidation from Run 26 ep229

- Status: running.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/lfezt6q4
- Output directory:
  `outputs/train/act_sim_ppo_r26ep229_sparse_lift_preserve_r28`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r25ep189_micro_lift_long_r26/act_sim_ppo_checkpoint_ep0229.pt`
- Hypothesis: R26 ep249 captured the best dense micro-lift row, but its
  continuation faded. The earlier ep229 snapshot is closest to the only sparse
  lift event at update `224`, so it may preserve a more threshold-adjacent
  behavior. Use an ultra-conservative actor LR and frequent snapshots to avoid
  training through any recovered sparse lift.
- Command profile: same randomized-block micro-lift reward, `policy-lr=1e-7`,
  `critic-lr=3e-5`, 12 parallel envs, 2 rollout chunks per env, 1 PPO epoch,
  minibatch 64, chunk size 30, 150 max steps, randomized block distance range
  `0.22 0.26`, angle range `-10 10`, snapshots every 5 updates.
- Early babysitting through update `248`: healthiest post-R26 branch so far.
  Latest-19 averages are strongly better than R27: `return=2.44`,
  `contact_steps=221.4`, `grasp_steps=46.8`,
  `max_block_height_gain=0.0067m`, `lift_progress_reward=6.82`,
  `micro_lifted=4.11`, and `train/clip_fraction=0`. Important rows:
  update `241` reached `max_block_height_gain=0.00908m`, update `245` reached
  `return=17.25`, `grasp_steps=116`, `micro_lifted=19`, update `246` reached
  `contact_steps=485`, `grasp_steps=117`, `lift_progress_reward=19.86`, and
  update `248` reached `return=14.22`, `contact_steps=313`,
  `grasp_steps=116`, `micro_lifted=9`. No sparse `lift_steps` yet, but this is
  the strongest consolidation signal after R26; keep running and preserve
  frequent snapshots.
- Updates `249`-`268`: keep running; R28 has now produced the strongest lift
  signal of the overnight window. Latest-20 averages stayed healthy:
  `return=1.26`, `contact_steps=200.8`, `grasp_steps=49.9`,
  `max_block_height_gain=0.0076m`, `lift_progress_reward=5.51`,
  `micro_lifted=3.60`, `lift_steps=0.15`, and `train/clip_fraction=0`.
  Important rows: update `251` reached `max_block_height_gain=0.00980m` with
  `contact_steps=350`, `grasp_steps=86`, `micro_lifted=15`; update `256`
  reached `0.00917m` with `contact_steps=370`, `grasp_steps=105`; update `262`
  produced sparse lift (`lift_steps=3`, `lift_bonus_reward=6`,
  `max_block_height_gain=0.00913m`); update `263` reached `0.00961m` with
  `contact_steps=269`, `grasp_steps=91`; and update `268` remained strong at
  `0.00853m`, `contact_steps=305`, `grasp_steps=70`. Treat this as the current
  best branch and keep it running unless these lift/contact signals collapse.
- Updates `269`-`288`: keep running, but watch for fade. R28 produced the best
  sparse lift/success-adjacent signal so far at update `276`:
  `return=10.08`, `contact_steps=341`, `grasp_steps=94`, `lift_steps=1`,
  `lift_bonus_reward=2`, `success_lift_bonus=6`, and
  `max_block_height_gain=0.01227m`. Update `278` was also strong without sparse
  lift (`return=15.43`, `contact_steps=313`, `grasp_steps=128`,
  `max_block_height_gain=0.01174m`), and update `281` reached `0.00913m`.
  Latest-20 averages now include `lift_steps=0.05`, `success_lift_bonus=0.30`,
  and `max_block_height_gain=0.0067m`; latest-10 is softer and micro-lift is
  zero, so preserve snapshots around this window and stop if the next window
  loses contact/grasp/lift-height entirely.
- Updates `289`-`308`: R28 repeated the strict lift/success signal, but the
  latest rows are fading. Update `296` reached `return=10.09`,
  `contact_steps=205`, `grasp_steps=80`, `lift_steps=2`,
  `lift_bonus_reward=4`, `success_lift_bonus=6`, and
  `max_block_height_gain=0.01138m`. This is the second success-bonus event after
  update `276`, so R28 is now the best training branch by lift/success metrics.
  The latest-10 window weakened (`contact_steps=89.2`, `grasp_steps=9.2`,
  `lift_progress_reward=0.55`, `micro_lifted=0`), though update `304` still
  reached `0.00953m` and update `308` recovered some contact/grasp. Continue one
  more close window; if it keeps fading, stop to preserve the best snapshots
  around `ep0279` and `ep0299`.
- Updates `309`-`327`: recovered; keep running. R28 now has repeated strict
  lift events across several windows. Update `319` reached `lift_steps=2`,
  `lift_bonus_reward=4`, `micro_lifted=17`, and
  `max_block_height_gain=0.00930m`; update `321` reached `return=12.33`,
  `contact_steps=263`, `grasp_steps=71`, `lift_steps=2`,
  `lift_bonus_reward=4`, `success_lift_bonus=6`, `micro_lifted=18`, and
  `max_block_height_gain=0.01143m`. Latest-10 averages improved to
  `lift_steps=0.40`, `success_lift_bonus=0.60`, `micro_lifted=3.50`, with
  stable `train/clip_fraction=0`. This is the clearest repeatable
  lift/success-signal branch so far; continue training and preserve snapshots.
- Stopped after the post-`327` window faded. Latest-20 before stop had
  `grasp_steps=1.9`, `lift_steps=0`, `micro_lifted=0.05`, and no success-bonus
  events; latest-10 had almost no grasp or height progress. This looks like the
  same post-peak drift pattern as R26/R27, not optimizer instability
  (`train/clip_fraction=0`). Best R28 snapshots to preserve:
  `act_sim_ppo_checkpoint_ep0279.pt` near the first success-bonus lift,
  `act_sim_ppo_checkpoint_ep0299.pt` near the second success-bonus lift, and
  `act_sim_ppo_checkpoint_ep0324.pt` nearest the strongest repeated late
  success-bonus window around update `321`. Use `ep0324` for the next
  stabilization attempt.

### Run 29: ultra-low-LR stabilization from Run 28 ep324

- Status: running.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/9kawslr5
- Output directory:
  `outputs/train/act_sim_ppo_r28ep324_lift_stabilize_r29`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r26ep229_sparse_lift_preserve_r28/act_sim_ppo_checkpoint_ep0324.pt`
- Hypothesis: R28 produced repeated strict lift/success-bonus events but then
  drifted. Restart from the nearest checkpoint after the update-321
  success-bonus row and reduce actor/critic LR again to test whether repeated
  lift can be stabilized without erasing the contact/grasp behavior.
- Command profile: same randomized-block micro-lift reward, `policy-lr=5e-8`,
  `critic-lr=1e-5`, 12 parallel envs, 2 rollout chunks per env, 1 PPO epoch,
  minibatch 64, chunk size 30, 150 max steps, randomized block distance range
  `0.22 0.26`, angle range `-10 10`, snapshots every 5 updates.

## Current Overnight Best Snapshot Comparison

- Headless stochastic inference comparison, 5 episodes each, 150 max steps,
  seed `2606`: R28 `ep0279` is the best of the first candidate set despite no
  evaluation successes. It had the strongest mean return/contact/grasp and was
  the only tested checkpoint with nonzero mean micro-lift count:
  `mean_return=38.69`, `mean_contact_count=58.4`, `mean_grasp_count=23.4`,
  `mean_max_block_height_gain=0.0050m`, `mean_micro_lift_count=2.6`,
  `success_rate=0`.
- Weaker evaluated candidates: R28 `ep0299` had `mean_grasp_count=0.4` and
  `mean_max_block_height_gain=0.0028m`; R28 `ep0324` had
  `mean_grasp_count=2.4` and `mean_max_block_height_gain=0.0032m`; R29
  `ep0354` had better height (`0.0047m`) but no grasp. For live inspection and
  further handoff, prefer
  `outputs/train/act_sim_ppo_r26ep229_sparse_lift_preserve_r28/act_sim_ppo_checkpoint_ep0279.pt`
  unless the next comparison finds an earlier R28 snapshot with better
  lift/success behavior.
- Additional 5-episode headless comparison: R28 `ep0264` had better average
  height (`0.0058m`) but poor return/grasp (`mean_return=-3.86`,
  `mean_grasp_count=4.8`). R28 `ep0274` is also a top candidate:
  `mean_return=35.71`, `mean_contact_count=56.6`, `mean_grasp_count=26.2`,
  `mean_max_block_height_gain=0.0049m`, and one episode reached
  `0.0095m`. Current top two for live inspection are R28 `ep0274` and R28
  `ep0279`; `ep0279` has slightly better mean return/micro-lift, while `ep0274`
  has slightly better mean grasp.

### Run 32: ultra-low-LR continuation from R28 ep274

- Status: running.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/0xb9h2xb
- Output directory:
  `outputs/train/act_sim_ppo_r28ep274_lift_stabilize_r32`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r26ep229_sparse_lift_preserve_r28/act_sim_ppo_checkpoint_ep0274.pt`
- Hypothesis: R28 ep274 is the other top headless-evaluated snapshot, with
  stronger mean grasp than ep279 and a 9.5mm height episode. Prior
  continuations from later snapshots drifted, so use an even smaller actor and
  critic LR to see whether the policy can retain contact/grasp while nudging
  toward repeated sparse lift.
- Command profile: same randomized-block micro-lift reward, `policy-lr=2e-8`,
  `critic-lr=5e-6`, 12 parallel envs, 2 rollout chunks per env, 1 PPO epoch,
  minibatch 64, chunk size 30, 150 max steps, randomized block distance range
  `0.22 0.26`, angle range `-10 10`, snapshots every 5 updates.
- Early babysitting through update `294`: weak, but not stopped yet. R32 has not
  produced sparse `lift_steps` or success-bonus, and latest-20 averages are poor
  (`return=-3.81`, `contact_steps=94.3`, `grasp_steps=15.6`,
  `micro_lifted=0`). It did still reach near/over threshold height multiple
  times: update `281` hit `max_block_height_gain=0.01119m`, update `292` hit
  `0.00908m`, and update `293` hit `0.00995m`. Give it one more short window;
  stop if these height events do not convert into contact/grasp/lift signal.
- Updates `295`-`314`: recovered; keep running. Latest-10 became much stronger:
  `return=3.85`, `contact_steps=260.3`, `grasp_steps=68.7`,
  `max_block_height_gain=0.0065m`, `lift_progress_reward=3.39`, with stable
  `train/clip_fraction=0`. Update `303` produced a sparse lift row
  (`lift_steps=1`, `lift_bonus_reward=2`, `max_block_height_gain=0.00921m`).
  Update `311` was the best manipulation row: `return=19.55`,
  `contact_steps=414`, `grasp_steps=187`, and
  `max_block_height_gain=0.00996m`; update `313` stayed strong with
  `contact_steps=446`, `grasp_steps=100`, `micro_lifted=9`. No success-bonus
  yet, but this is healthier than the first R32 window, so continue.
- Updates `315`-`334`: keep running. R32 is now very strong on contact/grasp and
  dense lift-progress, though it has not converted into repeated sparse lift or
  success. Latest-20 averages: `return=12.75`, `contact_steps=335.6`,
  `grasp_steps=116.3`, `max_block_height_gain=0.0067m`,
  `lift_progress_reward=8.51`, `micro_lifted=9.30`, `train/clip_fraction=0`.
  Important rows: update `329` reached `contact_steps=599`, `grasp_steps=248`,
  and `max_block_height_gain=0.00810m`; update `331` reached
  `return=32.53`, `contact_steps=431`, `grasp_steps=254`, `micro_lifted=29`;
  update `333` reached `return=45.71`, `contact_steps=524`,
  `grasp_steps=289`, `lift_progress_reward=40.38`, and `micro_lifted=102`.
  Continue while this strong manipulation state persists; stop if it turns into
  dense-reward exploitation without height/lift progress for a sustained window.
- Updates `335`-`353`: still healthy contact/grasp, but watch for dense-only
  exploitation. Latest-20 averages remain strong (`return=6.66`,
  `contact_steps=302.2`, `grasp_steps=77.9`, `lift_progress_reward=4.43`), but
  sparse lift has not repeated after update `303`. Height remains moderate
  (`max_block_height_gain` avg `0.0049m`, max `0.0084m`). Keep R32 running for
  one more window because contact/grasp are stable and useful; stop if the next
  window does not recover toward `9-10mm` height or sparse lift.
- Stopped after update `374`. R32 became a strong dense contact/grasp run but
  did not convert to repeated sparse lift or success. Later windows had high
  return/contact/grasp (`contact_steps` often `300+`, `grasp_steps` often
  `80+`) but height settled around `5-7mm` and `lift_steps` did not repeat after
  update `303`. Preserve R32 snapshots as possible grasp-quality references,
  but for lift/success prefer R28 `ep0274`/`ep0279`.

### Run 33: strict-lift-emphasis reward from R28 ep274

- Status: running.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/8j9zjtja
- Output directory:
  `outputs/train/act_sim_ppo_r28ep274_strict_lift_r33`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r26ep229_sparse_lift_preserve_r28/act_sim_ppo_checkpoint_ep0274.pt`
- Code profile: lower micro-lift shaping and raise strict lift/success rewards
  to discourage dense contact-only exploitation. Change defaults to
  `micro_lift_bonus=0.4`, `lift_bonus=4.0`, and `success_lift_bonus=10.0` while
  leaving thresholds unchanged.
- Hypothesis: R32 showed the policy can hold strong contact/grasp, but dense
  shaping did not drive it above the strict lift threshold often enough. Lowering
  micro-lift reward and increasing strict lift/success rewards should make the
  update prefer actual pickup events over prolonged contact.
- Stopped after update `293`. Starting strict-lift emphasis directly from R28
  `ep0274` was too early/fragile: latest-19 averaged `return=-1.57`,
  `contact_steps=108.8`, `grasp_steps=17.4`, no `lift_steps`, no success-bonus,
  and `max_block_height_gain=0.0054m`. Do not use R33 as a handoff. Next test:
  apply the same strict-lift reward to an already-established R32 dense
  contact/grasp snapshot.

### Run 34: strict-lift reward from R32 dense-grasp ep334

- Status: running.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/6a6et3em
- Output directory:
  `outputs/train/act_sim_ppo_r32ep334_strict_lift_r34`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r28ep274_lift_stabilize_r32/act_sim_ppo_checkpoint_ep0334.pt`
- Code profile: same strict-lift reward defaults as R33
  (`micro_lift_bonus=0.4`, `lift_bonus=4.0`, `success_lift_bonus=10.0`).
- Hypothesis: R32 ep334 already has strong contact/grasp and dense lift-progress
  (`ep333` had very high contact/grasp/micro-lift). Starting strict-lift
  training from this established manipulation state may convert contact/grasp
  into actual lift without first relearning grasp.
- Early babysitting through update `353`: healthy contact/grasp under strict
  reward, but no sparse lift yet. Latest-19 averages are strong:
  `return=10.45`, `contact_steps=304.7`, `grasp_steps=99.3`,
  `max_block_height_gain=0.0054m`, `lift_progress_reward=5.74`,
  `train/clip_fraction=0`. Important rows include update `351`
  (`return=27.71`, `contact_steps=466`, `grasp_steps=228`,
  `lift_progress_reward=17.25`) and update `350`
  (`max_block_height_gain=0.00826m`). Continue because strict reward preserved
  the manipulation state better than R33; watch for conversion to `9mm+` height
  and sparse lift.
- Stopped after update `372`. R34 preserved contact/grasp under strict reward
  (`contact_steps` around `300`, `grasp_steps` around `80`) but did not convert
  to sparse lift or success; latest max height was below the strict threshold.
  The strict reward from a dense-grasp snapshot is stable, but needs more
  exploration to escape contact-only behavior.

### Run 35: strict-lift exploratory continuation from R32 ep334

- Status: running.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/m87o69tj
- Output directory:
  `outputs/train/act_sim_ppo_r32ep334_strict_lift_explore_r35`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r28ep274_lift_stabilize_r32/act_sim_ppo_checkpoint_ep0334.pt`
- Code profile: same strict-lift reward defaults as R33/R34
  (`micro_lift_bonus=0.4`, `lift_bonus=4.0`, `success_lift_bonus=10.0`).
- Hypothesis: R34 was stable but conservative. Increasing actor/critic LR from
  the same dense-grasp checkpoint may add enough exploration to discover upward
  lift while the strict reward profile prioritizes threshold crossings.
- Command profile: `policy-lr=2e-7`, `critic-lr=2e-5`, 12 parallel envs, 2
  rollout chunks per env, 1 PPO epoch, minibatch 64, chunk size 30, 150 max
  steps, randomized block distance range `0.22 0.26`, angle range `-10 10`,
  snapshots every 5 updates.
- Early babysitting through update `353`: healthier than R34 for exploration,
  but no sparse lift yet. Latest-19 averages: `return=10.86`,
  `contact_steps=332.1`, `grasp_steps=111.7`,
  `max_block_height_gain=0.0067m`, `lift_progress_reward=4.93`,
  `train/clip_fraction=0`. Important rows: update `346` reached
  `return=32.42`, `contact_steps=580`, `grasp_steps=281`; update `348` reached
  `max_block_height_gain=0.00976m` with active contact/grasp; update `351`
  reached `contact_steps=524`, `grasp_steps=209`. Keep running to see whether
  higher LR converts the near-threshold row into `lift_bonus_reward` or
  success-bonus.
- Updates `354`-`378`: converted to sparse lift; keep running. Latest-20
  averages: `return=17.27`, `contact_steps=340.5`, `grasp_steps=154.6`,
  `lift_steps=0.10`, `max_block_height_gain=0.0067m`,
  `lift_bonus_reward=0.40`, and stable `train/clip_fraction=0`. Important rows:
  update `366` reached `max_block_height_gain=0.01028m`, update `368` produced
  `lift_steps=2`, `lift_bonus_reward=8`, `contact_steps=387`,
  `grasp_steps=164`, `micro_lifted=43`, and
  `max_block_height_gain=0.00967m`; update `371` reached `return=38.55`,
  `contact_steps=515`, `grasp_steps=302`. No success-bonus yet, but R35 is the
  strongest strict-reward follow-up and should continue unless lift/contact
  signals fade.
- Updates `379`-`397`: watch closely. Contact/grasp stayed strong
  (`latest-20 contact_steps=344.3`, `grasp_steps=129.6`), but sparse lift did
  not repeat after update `368`, and max height stayed below the strict lift
  threshold in the latest window. PPO health remains stable
  (`train/clip_fraction=0`). Continue one more short window; stop if sparse lift
  or 9mm+ height does not recover, preserving the R35 `ep0369`/`ep0374`
  snapshots around the strict-lift event.
- Stopped after update `416`. R35 did not repeat the update-368 lift and latest
  heights drifted below threshold, but it preserved stable contact/grasp and the
  strict-lift event snapshot. Preserve `act_sim_ppo_checkpoint_ep0369.pt` as the
  nearest checkpoint to the R35 lift event (`lift_steps=2`,
  `lift_bonus_reward=8`, `max_block_height_gain=0.00967m`). Evaluate it against
  the R28 top candidates before deciding whether to resume more training.
- Headless stochastic eval of R35 `ep0369` (5 episodes, seed `2606`): high
  contact/grasp and return, but no evaluation lift. Metrics:
  `mean_return=69.56`, `mean_contact_count=90.8`, `mean_grasp_count=46.4`,
  `mean_max_block_height_gain=0.0020m`, `mean_lift_count=0`,
  `success_rate=0`. R35 `ep0369` is strong for grasp/contact but does not beat
  R28 `ep0279` on height/micro-lift in inference. Use it only if stabilizing the
  strict-lift event is the immediate goal.

### Run 36: strict-lift stabilization from R35 ep369

- Status: running.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/g5werco1
- Output directory:
  `outputs/train/act_sim_ppo_r35ep369_strict_lift_stabilize_r36`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r32ep334_strict_lift_explore_r35/act_sim_ppo_checkpoint_ep0369.pt`
- Code profile: same strict-lift reward defaults
  (`micro_lift_bonus=0.4`, `lift_bonus=4.0`, `success_lift_bonus=10.0`).
- Hypothesis: R35 found a strict lift event at update `368` but did not repeat
  it. Restart from the nearest preserved snapshot with lower LR to test whether
  the event can be stabilized rather than explored through.
- Command profile: `policy-lr=5e-8`, `critic-lr=1e-5`, 12 parallel envs, 2
  rollout chunks per env, 1 PPO epoch, minibatch 64, chunk size 30, 150 max
  steps, randomized block distance range `0.22 0.26`, angle range `-10 10`,
  snapshots every 5 updates.
- Early babysitting through update `388`: stable but no repeated sparse lift.
  Latest-19 averages: `return=11.48`, `contact_steps=275.2`,
  `grasp_steps=111.1`, `max_block_height_gain=0.0051m`,
  `lift_progress_reward=7.53`, `micro_lifted=2.84`, `train/clip_fraction=0`.
  No `lift_steps` yet; keep one more window because contact/grasp and micro-lift
  are active.
- Stopped after update `408`. R36 collapsed after the first stable window:
  latest-10 averaged `return=-5.96`, `contact_steps=44.9`, `grasp_steps=0.7`,
  `max_block_height_gain=0.0009m`, no lift, and no micro-lift. Do not use R36
  as a handoff.

### Run 37: R28 ep279 lift-success continuation with restored micro-lift profile

- Status: stopped/rejected; do not use as a handoff.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/h1r2g7m8
- Output directory:
  `outputs/train/act_sim_ppo_r28ep279_lift_success_r37`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r26ep229_sparse_lift_preserve_r28/act_sim_ppo_checkpoint_ep0279.pt`
- Code profile: restored the R28 micro-lift reward defaults
  (`micro_lift_bonus=0.8`, `lift_bonus=2.0`, `success_lift_bonus=6.0`) and
  kept thresholds unchanged (`micro_lift_bonus_threshold=0.0050`,
  `lift_bonus_threshold=0.0090`, `lift_threshold=0.010`). Added two extra
  diagnostics/reward components: `grasped_vertical_lift_reward` for positive
  block height delta under `micro_lift_ready`/`lift_grasp_ready`, and
  `lift_side_push_penalty` for horizontal displacement during sub-threshold
  lift attempts.
- Baseline eval before launch, narrowed randomized block reset
  (`block-dist-range 0.23 0.25`, `block-angle-range -5 5`, 5 stochastic
  episodes, seed `2606`):
  - R28 `ep0279`: `success_rate=0`, `mean_return=6.17`,
    `mean_contact_count=66.8`, `mean_grasp_count=27.4`,
    `mean_max_block_height_gain=0.0048m`, `mean_micro_lift_count=0`,
    `mean_grasped_vertical_lift_reward=0.204`, and
    `mean_lift_side_push_penalty=-0.014`.
  - R28 `ep0274`: `success_rate=0`, `mean_return=-16.80`,
    `mean_contact_count=74.0`, `mean_grasp_count=21.8`,
    `mean_max_block_height_gain=0.0071m`, `mean_micro_lift_count=0.4`,
    `mean_grasped_vertical_lift_reward=0.203`, and
    `mean_lift_side_push_penalty=-0.105`.
  R28 `ep0279` remained the cleaner default handoff because it kept better
  return/grasp balance with far less side-push penalty.
- Command profile: `policy-lr=5e-8`, `critic-lr=1e-5`, 12 parallel envs,
  2 rollout chunks per env, 1 PPO epoch, minibatch 64, chunk size 30,
  150 max steps, narrowed randomized block distance range `0.23 0.25`,
  angle range `-5 5`, snapshots every 5 updates.
- Babysitting result: update `286` produced the intended sparse-lift/success
  signal (`lift_steps=2`, `lift_bonus_reward=4`, `success_lift_bonus=6`,
  `micro_lifted=3`, `max_block_height_gain=0.01327m`), but the policy then
  collapsed quickly. Latest-10 before stop averaged `return=-7.16`,
  `contact_steps=2.4`, `grasp_steps=0`, `lift_steps=0`,
  `max_block_height_gain=0.0008m`, zero micro-lift, and stable
  `train/clip_fraction=0`; this was another behavior-erasure/post-peak drift
  failure, not optimizer instability.
- Snapshot tournament:
  - R37 `ep0284`: `success_rate=0`, `mean_return=-53.74`,
    `mean_contact_count=30.6`, `mean_grasp_count=0`,
    `mean_max_block_height_gain=0.0027m`, and no micro-lift/lift.
  - R37 `ep0289`: `success_rate=0`, `mean_return=-59.13`,
    `mean_contact_count=4.8`, `mean_grasp_count=0`,
    `mean_max_block_height_gain=0.0001m`, and no micro-lift/lift.
  Both snapshots are worse than the R28 `ep0279` baseline, so R37 does not
  supersede the current best policy.

### Run 38: R28 ep274 ultra-low-LR lift-success continuation

- Status: stopped/rejected as a new handoff; useful as evidence that R28
  `ep0274` is the more robust headless-eval seed but PPO still erases behavior.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/6ln49hnx
- Output directory:
  `outputs/train/act_sim_ppo_r28ep274_lift_success_r38`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r26ep229_sparse_lift_preserve_r28/act_sim_ppo_checkpoint_ep0274.pt`
- Pre-launch 20-episode narrowed-randomization tournament
  (`block-dist-range 0.23 0.25`, `block-angle-range -5 5`, stochastic,
  seed `2606`):
  - R28 `ep0279`: `success_rate=0`, `mean_return=-15.22`,
    `mean_contact_count=55.5`, `mean_grasp_count=18.1`,
    `mean_max_block_height_gain=0.0042m`, `mean_micro_lift_count=0.1`,
    `mean_lift_side_push_penalty=-0.296`.
  - R28 `ep0274`: `success_rate=0`, `mean_return=7.47`,
    `mean_contact_count=74.8`, `mean_grasp_count=26.1`,
    `mean_max_block_height_gain=0.0053m`, `mean_micro_lift_count=0.4`,
    `mean_lift_side_push_penalty=-0.038`.
  This made `ep0274` the better headless-eval seed under the narrowed task,
  though `ep0279` remains the user-validated live demo policy until a later
  checkpoint beats it visually.
- Command profile: `policy-lr=1e-8`, `critic-lr=5e-6`, 12 parallel envs,
  2 rollout chunks per env, 1 PPO epoch, minibatch 64, chunk size 30,
  150 max steps, narrowed randomized block distance range `0.23 0.25`,
  angle range `-5 5`, snapshots every 5 updates.
- Babysitting result: R38 preserved behavior longer than R37 and produced two
  sparse-lift/success rows:
  - update `306`: `lift_steps=6`, `lift_bonus_reward=12`,
    `success_lift_bonus=6`, `micro_lifted=17`,
    `max_block_height_gain=0.01002m`.
  - update `316`: `lift_steps=2`, `lift_bonus_reward=4`,
    `success_lift_bonus=6`, `micro_lifted=4`,
    `max_block_height_gain=0.01023m`.
  It then collapsed like previous continuations. Latest-10 before stop averaged
  `return=-4.79`, `contact_steps=14.1`, `grasp_steps=0`, `lift_steps=0`,
  `max_block_height_gain=0.00087m`, and zero micro-lift, with stable
  `train/clip_fraction=0`.
- Snapshot tournament:
  - R38 `ep0309` (nearest after update `306`): `success_rate=0`,
    `mean_return=-16.22`, `mean_contact_count=35.6`,
    `mean_grasp_count=2.4`, `mean_max_block_height_gain=0.0049m`,
    `mean_micro_lift_count=1.0`, no strict lift.
  - R38 `ep0319` (nearest after update `316`): `success_rate=0`,
    `mean_return=-36.19`, zero contact/grasp/lift, and
    `mean_max_block_height_gain=0.0001m`.
  R38 produced stronger training lift rows than R37, but saved snapshots did
  not beat the original R28 `ep0274`/`ep0279` policies in stochastic eval.

### Run 39: per-update snapshots from R28 ep274

- Status: stopped/completed; useful strict-lift artifact, but not a replacement
  for R28 as the general policy.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/ccrz0cbj
- Output directory:
  `outputs/train/act_sim_ppo_r28ep274_per_update_lift_r39`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r26ep229_sparse_lift_preserve_r28/act_sim_ppo_checkpoint_ep0274.pt`
- Motivation: R38 produced real training lift/success rows, but `snapshot-every=5`
  missed the narrow useful windows. R39 repeated the R38 settings with
  `snapshot-every=1` and a short budget to preserve exact per-update policies
  around any lift row.
- Command profile: `policy-lr=1e-8`, `critic-lr=5e-6`, 12 parallel envs,
  2 rollout chunks per env, 1 PPO epoch, minibatch 64, chunk size 30,
  150 max steps, narrowed randomized block distance range `0.23 0.25`,
  angle range `-5 5`, snapshots every update, episodes through `318`.
- Training result: update `298` produced a sparse-lift row
  (`lift_steps=2`, `lift_bonus_reward=4`, `max_block_height_gain=0.00904m`,
  `contact_steps=301`, `grasp_steps=91`) without success-bonus. R39 did not
  repeat the stronger R38 success-bonus rows, but it preserved exact snapshots
  around the lift event.
- Snapshot tournament:
  - R39 `ep0298`, 5 episodes: `success_rate=0`, `mean_return=-56.31`,
    `mean_contact_count=20.0`, `mean_grasp_count=3.4`,
    `mean_max_block_height_gain=0.0040m`, `mean_lift_count=0.6`.
    One episode reproduced strict lift (`lift_count=3`,
    `max_block_height_gain=0.0092m`), making this the first saved R37-R39
    artifact with nonzero strict lift in stochastic eval.
  - R39 `ep0300`, 5 episodes: `success_rate=0`, `mean_return=-48.89`,
    `mean_grasp_count=1.6`, no lift.
  - R39 `ep0311`, 5 episodes: `success_rate=0`, `mean_return=-40.02`,
    `mean_grasp_count=1.8`, no lift.
  - R39 `ep0298`, 20 episodes: `success_rate=0`, `mean_return=-53.43`,
    `mean_contact_count=21.9`, `mean_grasp_count=3.8`,
    `mean_max_block_height_gain=0.0044m`, `mean_micro_lift_count=0.2`,
    `mean_lift_count=0.1`. Lift reproduced in 1/20 episodes, but contact/grasp
    quality was far worse than R28 `ep0274`/`ep0279`.
- Interpretation: per-update snapshots solved the capture problem, not the
  generalization problem. R39 `ep0298` is the best saved strict-lift artifact
  from the R37-R39 family, but it is too brittle and low-quality for demo or
  handoff. Keep R28 `ep0279` as the live demo policy and R28 `ep0274` as the
  best narrowed headless seed.

### Run 40: low-exploration R28 continuation from ep274

- Status: rejected as a PPO continuation; useful diagnostic.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/pxv4ttnv
- Output directory:
  `outputs/train/act_sim_ppo_r28ep274_lowstd_lift_r40`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r26ep229_sparse_lift_preserve_r28/act_sim_ppo_checkpoint_ep0274.pt`
- Code change: added `--resume-log-std-offset` to
  `train_act_in_sim.py`, applied after checkpoint resume. R40 used
  `--resume-log-std-offset -0.693147`, halving the learned PPO action std
  while preserving the resumed mean policy.
- Motivation: pre-launch eval showed untouched R28 `ep0274` with
  `--stochastic-scale 0.5` was the best immediate lift probe:
  `mean_return=8.27`, `mean_contact_count=84.0`, `mean_grasp_count=32.4`,
  `mean_max_block_height_gain=0.0065m`, and `mean_lift_count=0.2` over
  5 episodes. Deterministic R28 `ep0274` had better return/contact but weaker
  height (`mean_max_block_height_gain=0.0018m`), and R39 `ep0298` remained too
  low-contact.
- Command profile: same R39 narrowed randomization and per-update snapshots,
  `policy-lr=1e-8`, `critic-lr=5e-6`, 12 parallel envs, 2 rollout chunks per
  env, 1 PPO epoch, minibatch 64, chunk size 30, 150 max steps, distance range
  `0.23 0.25`, angle range `-5 5`.
- Babysitting result: early lower-noise rows were healthier than R39. Update
  `278` reached `contact_steps=602`, `grasp_steps=208`,
  `micro_lifted=1`, and `max_block_height_gain=0.01528m`; update `281`
  reached `contact_steps=455`, `grasp_steps=104`, and
  `max_block_height_gain=0.00981m`. The run then faded quickly; latest-5 at
  the first check had `grasp_steps=0`, `lift_steps=0`, and was stopped to
  preserve snapshots.
- Snapshot evals with `--stochastic --stochastic-scale 0.5`, 5 episodes:
  - `ep0275`: `success_rate=0`, `mean_return=51.90`,
    `mean_contact_count=94.6`, `mean_grasp_count=50.6`,
    `mean_max_block_height_gain=0.0029m`, no lift.
  - `ep0276`: `success_rate=0`, `mean_return=23.00`,
    `mean_contact_count=88.8`, `mean_grasp_count=32.8`,
    `mean_max_block_height_gain=0.0025m`, no lift.
  - `ep0278`: `success_rate=0`, `mean_return=-32.91`,
    `mean_contact_count=65.2`, `mean_grasp_count=1.8`,
    `mean_max_block_height_gain=0.0014m`, no lift.
  - `ep0281`: `success_rate=0`, `mean_return=-47.55`,
    `mean_contact_count=9.0`, `mean_grasp_count=0.0`,
    `mean_max_block_height_gain=0.0004m`, no lift.
- Interpretation: lower exploration noise is useful at inference and in early
  rollout rows, but even tiny PPO continuation still corrupts the transferable
  grasp/lift behavior. The strongest current operating policy remains untouched
  R28 `ep0274`/`ep0279`, with lower-noise stochastic inference worth using as
  a comparison mode. The next credible path is supervised correction data
  (teleop/scripted lift demonstrations through `record_dataset.py` and
  `train_act_on_data.py`) or another non-PPO objective, not another blind
  continuation run from R28.

### Run 41: narrowed low-std sibling from R26 ep229

- Status: stopped/rejected as a replacement for R28.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/dxvnn5m2
- Output directory:
  `outputs/train/act_sim_ppo_r26ep229_narrow_lowstd_perupdate_r41`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r25ep189_micro_lift_long_r26/act_sim_ppo_checkpoint_ep0229.pt`
- Motivation: the user asked to start from the same policy that produced the
  current best R28, rather than continuing from R28 itself. R41 is a sibling of
  R28 from R26 `ep0229`, with narrowed randomized placement and per-update
  snapshots so short useful windows would be preserved.
- Command profile: current R28 micro-lift reward profile with
  `grasped_vertical_lift_reward` and `lift_side_push_penalty` telemetry,
  `policy-lr=5e-8`, `critic-lr=1e-5`, 12 parallel envs, 2 rollout chunks per
  env, 1 PPO epoch, minibatch 64, chunk size 30, 150 max steps, narrowed block
  distance range `0.23 0.25`, angle range `-5 5`, per-update snapshots, and
  `--resume-log-std-offset -0.356675` (`log_std_mean=-2.3567`, about 0.7x the
  resumed action std).
- Babysitting result: stopped after update `250`. The first rows were mixed but
  briefly promising by compact return (`ep0230` return `9.73`, `ep0233` return
  `16.07`, `ep0235` return `4.30`), then the branch degraded into mostly
  negative returns with no success rows. Final W&B summary at update `250`:
  `return=-0.98`, `contact_steps=136`, `grasp_steps=5`, `lift_steps=0`,
  `max_block_height_gain=0.00521m`, `micro_lift_bonus=0`,
  `grasped_vertical_lift_reward=0.190`, `lift_side_push_penalty=0`,
  `lift_bonus_reward=0`, `success_lift_bonus=0`, `train/clip_fraction=0`, and
  `train/log_std_mean=-2.3567`. This looked like the familiar behavior-drift
  failure rather than optimizer instability.
- Focused 5-episode narrowed stochastic evals (`block-dist-range 0.23 0.25`,
  `block-angle-range -5 5`, 150 max steps, seed `2606`):
  - R41 `ep0230`: `success_rate=0`, `mean_return=11.33`,
    `mean_contact_count=54.0`, `mean_grasp_count=11.6`,
    `mean_max_block_height_gain=0.0048m`, `mean_lift_count=0`,
    `mean_grasped_vertical_lift_reward=0.119`,
    `mean_lift_side_push_penalty=-0.063`.
  - R41 `ep0233`: `success_rate=0`, `mean_return=-5.52`,
    `mean_contact_count=21.6`, `mean_grasp_count=2.8`,
    `mean_max_block_height_gain=0.0042m`, no lift.
  - R41 `ep0250`: `success_rate=0`, `mean_return=-22.03`,
    `mean_contact_count=12.6`, `mean_grasp_count=0`,
    `mean_max_block_height_gain=0.0021m`, no lift.
  - R28 `ep0279` baseline under the same narrowed stochastic protocol:
    `success_rate=0`, `mean_return=30.67`, `mean_contact_count=49.0`,
    `mean_grasp_count=19.8`, `mean_max_block_height_gain=0.0045m`, no lift.
  - R28 `ep0274` with `--stochastic-scale 0.5`: `success_rate=0`,
    `mean_return=54.14`, `mean_contact_count=83.4`,
    `mean_grasp_count=36.2`, `mean_max_block_height_gain=0.0041m`, no lift.
- Conclusion: R41 did not beat R28 `ep0279` or the lower-noise R28 `ep0274`
  evaluation mode. Its best quick-eval snapshot (`ep0230`) had comparable mean
  height to R28 `ep0279` but substantially worse return/grasp, so no 20-episode
  eval was warranted. Preserve `act_sim_ppo_checkpoint_ep0230.pt` only as the
  best R41 artifact; keep R28 `ep0279` as the live/demo best and R28 `ep0274`
  with lower-noise stochastic inference as the narrowed headless reference.

## Current Handoff Status

- Best training branch by strict lift/success-like training metrics: R28. It
  produced repeated sparse lift and success-bonus events, especially update
  `276` (`lift_steps=1`, `success_lift_bonus=6`,
  `max_block_height_gain=0.01227m`) and update `321` (`lift_steps=2`,
  `success_lift_bonus=6`, `max_block_height_gain=0.01143m`).
- Best headless stochastic inference candidate so far: R28
  `act_sim_ppo_checkpoint_ep0279.pt`, with `mean_return=38.69`,
  `mean_contact_count=58.4`, `mean_grasp_count=23.4`,
  `mean_max_block_height_gain=0.0050m`, and `mean_micro_lift_count=2.6` over
  5 episodes. R28 `ep0274` is the close alternate with slightly better grasp
  (`mean_grasp_count=26.2`) and one `0.0095m` height episode.
- User live-sim inspection on 2026-06-27: R28 `ep0279` is the best policy found
  so far qualitatively. It actively tries to interact with and grasp the block
  in simulation, enough that a video was worth sharing publicly. Treat
  `outputs/train/act_sim_ppo_r26ep229_sparse_lift_preserve_r28/act_sim_ppo_checkpoint_ep0279.pt`
  as the default "best current policy" for demos and next-stage comparisons
  unless a later checkpoint clearly improves live grasp/lift behavior.
- Best current headless-eval seed under narrowed randomized placement is R28
  `ep0274`, based on the 20-episode R38 pre-launch comparison. It had better
  mean return/contact/grasp/height than `ep0279` in that protocol, but no
  successor checkpoint from R38 has beaten it.
- Best saved strict-lift stochastic-eval artifact from the R37-R39 follow-ups
  is R39 `ep0298`, but it only lifted in 1/20 eval episodes and had poor
  contact/grasp quality. Use it for failure analysis of how lift appears, not
  as the best policy.
- Best lower-noise operating mode discovered on 2026-06-28: untouched R28
  `ep0274` with `run_act_ppo_sim_inference.py --stochastic --stochastic-scale
  0.5` under narrowed randomized placement. It reproduced strict lift in 1/5
  quick eval episodes while preserving materially better contact/grasp than
  R39 artifacts. R40 showed that continuing PPO from this lower-noise state
  still degrades quickly, so use this as an evaluation/demo variant rather than
  a PPO handoff.
- Best strict-lift-reward artifact: R35
  `act_sim_ppo_checkpoint_ep0369.pt`. It had a training lift event at update
  `368` (`lift_steps=2`, `lift_bonus_reward=8`,
  `max_block_height_gain=0.00967m`), but its 5-episode headless eval did not
  reproduce lift and averaged only `0.0020m` height gain.
- Current code profile is restored to the R28 micro-lift emphasis:
  `micro_lift_bonus=0.8`, `lift_bonus=2.0`, `success_lift_bonus=6.0`, with
  added `grasped_vertical_lift_reward` and `lift_side_push_penalty` telemetry.
  R37/R38 showed that even narrowed randomization and actor LR as low as
  `1e-8` can erase R28 behavior after briefly producing sparse-lift training
  rows. Future attempts should avoid trusting PPO training rows alone and
  should prioritize supervised correction data, scripted/teleop lift demos, or
  another non-PPO objective before more continuation PPO.
- Stopped early after update `298`. R31 had one early success-bonus row at
  update `286` (`lift_steps=1`, `success_lift_bonus=6`,
  `max_block_height_gain=0.01080m`), but latest-10 collapsed to
  `return=-5.60`, `contact_steps=27.3`, `grasp_steps=2.3`, zero lift/micro-lift,
  and weak height progress. This confirms that additional PPO from the later R28
  handoff family quickly erases the useful behavior. Current best training
  branch remains R28; compare preserved R28/R29 snapshots directly before
  launching any more continuations.
- Stopped early after update `343`. R30 was worse than the R28 source branch:
  it had one early success-bonus row at update `324`
  (`lift_steps=1`, `success_lift_bonus=6`, `max_block_height_gain=0.01037m`),
  but latest-10 quickly faded to `contact_steps=49.6`, `grasp_steps=3.9`,
  zero lift/micro-lift, and weak height progress. Do not use R30 as a handoff.

### Run 31: early R28 success-window stabilization from ep279

- Status: running.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/aes6djlr
- Output directory:
  `outputs/train/act_sim_ppo_r28ep279_lift_stabilize_r31`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r26ep229_sparse_lift_preserve_r28/act_sim_ppo_checkpoint_ep0279.pt`
- Hypothesis: R28 ep279 is closest to the first strong success-bonus event and
  still had much stronger contact/grasp than the later ep319/ep324 handoffs.
  If later handoffs were already drifting, restarting from this earlier peak may
  better preserve the successful grasp/lift behavior.
- Command profile: same randomized-block micro-lift reward, `policy-lr=5e-8`,
  `critic-lr=1e-5`, 12 parallel envs, 2 rollout chunks per env, 1 PPO epoch,
  minibatch 64, chunk size 30, 150 max steps, randomized block distance range
  `0.22 0.26`, angle range `-10 10`, snapshots every 5 updates.
- Early babysitting through update `343`: weaker than hoped, but one useful lift
  row appeared immediately. Update `326` reached `return=6.99`,
  `contact_steps=191`, `grasp_steps=45`, `lift_steps=1`,
  `lift_bonus_reward=2`, `success_lift_bonus=6`,
  `micro_lifted=6`, and `max_block_height_gain=0.01046m`. After that, latest-10
  faded to `contact_steps=74.1`, `grasp_steps=9.3`, zero lift/micro-lift, and
  `max_block_height_gain=0.0044m`. Give R29 one more close window; stop if it
  does not recover, and keep R28 `ep0319`/`ep0324` as the stronger handoff
  candidates.
- Stopped after update `363`. R29 did recover one more strict lift row at update
  `352` (`lift_steps=3`, `lift_bonus_reward=6`, `success_lift_bonus=6`,
  `max_block_height_gain=0.01058m`), but the latest-10 window then collapsed to
  `grasp_steps=0`, `lift_steps=0`, and weak height progress. Preserve
  `act_sim_ppo_checkpoint_ep0354.pt` as the nearest R29 success snapshot, but
  prefer R28 `ep0319`/`ep0324` as stronger handoffs because they had better
  contact/grasp and repeated success-bonus rows.

### Run 30: pre-fade stabilization from Run 28 ep319

- Status: running.
- W&B: https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/b4ciw8ya
- Output directory:
  `outputs/train/act_sim_ppo_r28ep319_lift_stabilize_r30`
- Starting checkpoint:
  `outputs/train/act_sim_ppo_r26ep229_sparse_lift_preserve_r28/act_sim_ppo_checkpoint_ep0319.pt`
- Hypothesis: R29 from R28 ep324 could still produce lifts, but contact/grasp
  was already fragile. R28 ep319 is immediately at the start of the late
  repeated-lift window, before the strongest update-321 success-bonus row and
  before the subsequent fade, so it may be a better stabilization handoff.
- Command profile: same randomized-block micro-lift reward, `policy-lr=5e-8`,
  `critic-lr=1e-5`, 12 parallel envs, 2 rollout chunks per env, 1 PPO epoch,
  minibatch 64, chunk size 30, 150 max steps, randomized block distance range
  `0.22 0.26`, angle range `-10 10`, snapshots every 5 updates.
