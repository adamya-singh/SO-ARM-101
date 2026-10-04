# W&B standard (2026-10-04)

These are the rules for every run this repository logs. The code that enforces them is
`src/so_arm101_v2/tracking.py`, and new tools use it.

Where the rules came from:
- The user's 2026-09-10 request: put the most important charts first, using numbered sections.
- The user's 2026-10-02 request for pi0.5: "make the wandb charts nice and organized".
- A review of the 22 runs since 09-08: the augmentation, ladder and pi0.5 runs, plus the bench runs from before
  the convention.

## Project, group, name, tags

- **Project:** always `so-arm101-v2-scaling`. The legacy projects `reinflow-smolvla` and `act-so101-sim-ppo` are
  closed.
- **Group:** the experiment family, for example `pi05`, `augmentation`, `scaling-ladder`, `smolvla`, `act`,
  `rl-pi05`, `rl-smolvla`, `rl-act` or `rerun-entries-03-07`.
- **Name:** `<family>-<key hyperparameters>-<scene hash 8>[-seed N]`, built by `tracking.run_name`. A run filter by
  family prefix then catches every run in the family.
- **Rehearsals** are named `<family>-rehearsal` and run offline. They are never quoted.
- **Tags:** the family, the scope (for example `expert_lora`), the data regime (`placements`), the shift
  (`shift12`), and the tranche tag (for example `retry-tranche-20261004`). The cross-family saved view filters on
  the tranche tag.
- **`job_type`:** one of `finetune`, `rl`, `eval`, `rerun`, `watcher` or `probe`.

## Sections (most important first)

Every key, summary keys included, sits under one of five numbered sections. W&B sorts sections alphabetically,
so the number sets the order.

| section | what goes in it |
|---|---|
| `01_outcome/` | Closed-loop results on the 30 held-out rollouts: `success_rate_*`, `successes_*`, `pickups_*`, `safety_frames_*` and `closure_*_mm_*`, plus the per-rollout table `rollouts_*` and the image `summary_figure`. RL runs also log their monitoring-suite success here during training (`monitor_success_rate` and related keys). |
| `02_generalisation/` | The held-out curve during training: `heldout_start_90`, `train_start_90`, `ratio_start_90`, `median`, `max`, `poses_under_threshold`, `pose/<id>`, `heldout_start_0` and `heldout_starts_180_270_360`. Also the constant reference lines. |
| `03_training/` | The step axes `step` and `episodes`. The loss (`loss`, `flow_loss`, per joint), `grad_norm`, `lr_*` and `epoch`. For RL: `reward_*`, `return_mean`, `kl`, `clip_fraction`, `logprob_*`, `value_loss`, `explained_variance`, `sigma_*`, `entropy` and `advantage_*`. |
| `04_throughput/` | `steps_per_s`, `samples_per_s`, `episodes_per_hour`, `env_seconds`, `policy_seconds`, `update_seconds`, `data_wait_fraction`, `gpu_peak_gb`, `host_rss_gb`, `elapsed_hours` and `eta_hours`. |
| `05_phases/` | `phase` (an index into the run's phase list), one 0/1 flag per phase, and `episodes_done`/`episodes_target` for RL. |

Keys outside these five sections are refused. `Tracker.log` raises in rehearsals and tests, and drops the key
with a printed message in real runs. Older runs keep their old names, since a logged run cannot be renamed.

## Step axes and summaries

- **Axes:** `03_training/step` is the axis for fine-tunes. RL runs declare `03_training/episodes` as their
  default axis and also log `03_training/step` as the update count.
- **Declaring sections:** every section is declared with `define_metric(f"{section}/*", step_metric=<axis>)`.
- **Summaries:** `summary="min"` or `"max"` is set only on keys the run actually logs. `Tracker.finish` checks this
  and raises in rehearsals.
  - This rule comes from pi0.5 on 2026-10-02: its run set `summary="min"` on `per_pose_median` but logged the
    value as `median`, so the summary never applied.

## Fixed names

Use the same key for the same quantity in every tool:
- `steps_per_s`, never `steps_per_second`
- `median` for the per-pose median
- `loss` for a supervised loss; a flow-matching model also logs `flow_loss`
- `_exec90` / `_exec30` as the suffix for an execution schedule, and `_K<n>` for a denoising-step count

## Reference lines

Every held-out-curve evaluation also logs these three constants from `tracking.REFERENCE_LINES`, so each chart
carries the levels it must be read against:

| key | value | meaning |
|---|---|---|
| `02_generalisation/ref_cnn4800_heldout_start_90` | 3.6e-4 | the CNN-4800 run's held-out start-90 error |
| `02_generalisation/ref_working_policy_start_90` | 2.5e-6 | the working fixed-square policy |
| `02_generalisation/ref_pose_threshold_2.5e-4` | 2.5e-4 | below this, past held-out poses succeeded |

Success counts are compared against `tracking.REFERENCES`: CNN-4800 6/30, and the pi0.5 fine-tune 12/30 at
execute 90 and 15/30 at execute 30.

## Every run also

- Passes its full identity dict as config: tool, config dataclass, manifests, suites, scene sha256, references,
  and the episode budget for RL.
- Writes the run id to `wandb.json` in its output directory (`tracking.persist_run_id`) and resumes with
  `resume="allow"`. The tranche status page reads the URL from that file.
- Never stops because of a W&B failure: every call is wrapped.
- Logs a per-rollout table and a summary figure when it finishes.
- Mirrors the held-out curve to `heldout_curve.jsonl`, and RL statistics to `metrics.jsonl`, on disk.

## Saved views (workspace as code)

`tools/wandb_workspace.py --family <family>` saves one view per family. It runs in the `lerobot-pi05` env, the
only env with `wandb_workspaces`. Each view has:
- sections in numeric order, with 04 and 05 collapsed
- log-y panels for losses and held-out curves, overlaid with the reference lines (dashed and dotted)
- an outcome section that puts success and closure millimetres side by side
- a run filter on the family's name prefix

The `tranche` family puts the runs of one tranche side by side on the episode axis. Record each saved view's URL
in the experiment's note.
