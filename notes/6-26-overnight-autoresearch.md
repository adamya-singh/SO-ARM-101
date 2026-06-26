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
