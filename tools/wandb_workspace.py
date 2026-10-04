"""Saved W&B workspace views, one per experiment family (notes/wandb-standard.md), 2026-10-04.

Generalises tools/pi05_wandb_workspace.py (2026-10-02). Every view orders the sections by number
(01 outcome first), draws held-out curves on a log axis against the reference levels they must be read
against (CNN-4800 3.6e-4 dashed, working fixed-square policy 2.5e-6 dotted, 2.5e-4 pose threshold), and
collapses the plumbing (04 throughput, 05 phases).

Families:
  pi05, smolvla, act   fine-tunes on the placement captures (x axis: training step)
  rl                   RL runs of every model (x axis: environment episodes); filter --query rl-
  tranche              every run of a tranche side by side (filter by name regex), episode axis for RL panels

Usage (needs network; each call saves a new view and prints its URL; lerobot-pi05 env has wandb_workspaces):
  ~/miniforge3/envs/lerobot-pi05/bin/python tools/wandb_workspace.py --family rl
"""
from __future__ import annotations

import argparse

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

O, G, T, P, H = "01_outcome/", "02_generalisation/", "03_training/", "04_throughput/", "05_phases/"
STEP, EPISODES = f"{T}step", f"{T}episodes"
REF_MARKS = {f"{G}ref_cnn4800_heldout_start_90": "dashed", f"{G}ref_working_policy_start_90": "dotted",
             f"{G}ref_pose_threshold_2.5e-4": "dotted"}
REF_TITLES = {f"{G}ref_cnn4800_heldout_start_90": "CNN 4800 held-out (3.6e-4)",
              f"{G}ref_working_policy_start_90": "working fixed-square policy (2.5e-6)",
              f"{G}ref_pose_threshold_2.5e-4": "level of past successes (2.5e-4)"}
TITLES = {"pi05": "pi0.5", "smolvla": "SmolVLA", "act": "ACT"}


def line(title, ys, *, x=STEP, log=False, titles=None, marks=None, smoothing=None, regex=None, y_title=None):
    return wr.LinePlot(title=title, x=x, y=ys, log_y=log, line_titles=titles, line_marks=marks,
                       title_x=("environment episodes" if x == EPISODES else "training step"), title_y=y_title, metric_regex=regex,
                       smoothing_type=("exponential" if smoothing else None), smoothing_factor=smoothing,
                       smoothing_show_original=True if smoothing else None)


def outcome_section(label: str) -> ws.Section:
    return ws.Section(name=f"01 Outcome: {label} closed loop on 10 unseen placements x 3", is_open=True, panels=[
        wr.BarPlot(title="Successes of 30 (exec 90 = CNN schedule, exec 30 = replan every 30 steps)",
                   metrics=[f"{O}successes_exec90", f"{O}successes_exec30"], orientation="v"),
        wr.BarPlot(title="Closure: lateral error median at step 250 (mm; tolerance 8, working policy 1.7, CNN-4800 5.1)",
                   metrics=[f"{O}closure_lateral_median_mm_exec90", f"{O}closure_lateral_median_mm_exec30"], orientation="v"),
        wr.BarPlot(title="Closure: rollouts inside 8 mm vs never descended",
                   metrics=[f"{O}closure_within_8mm_exec90", f"{O}closure_within_8mm_exec30",
                            f"{O}closure_not_descended_exec90", f"{O}closure_not_descended_exec30"], orientation="v"),
        wr.BarPlot(title="Pickups completed and safety frames",
                   metrics=[f"{O}pickups_exec90", f"{O}pickups_exec30", f"{O}safety_frames_exec90", f"{O}safety_frames_exec30"], orientation="v"),
        wr.MediaBrowser(title="Summary figure", media_keys=[f"{O}summary_figure"], num_columns=1),
    ])


def generalisation_section(label: str, x: str = STEP) -> ws.Section:
    return ws.Section(name="02 Generalisation: held-out start-90 chunk error", is_open=True, panels=[
        line("Held-out vs train start-90 MSE against the two reference levels (log)",
             [f"{G}heldout_start_90", f"{G}train_start_90", f"{G}ref_cnn4800_heldout_start_90", f"{G}ref_working_policy_start_90"], x=x,
             log=True, titles={f"{G}heldout_start_90": f"{label} held-out", f"{G}train_start_90": f"{label} train", **REF_TITLES}, marks=REF_MARKS,
             y_title="MSE (1 unit = pi rad)"),
        line("Per-pose median and worst pose (log)",
             [f"{G}median", f"{G}max", f"{G}ref_pose_threshold_2.5e-4", f"{G}ref_working_policy_start_90"], x=x, log=True,
             titles={f"{G}median": "per-pose median", f"{G}max": "worst pose", **REF_TITLES}, marks=REF_MARKS),
        line("Held-out error in degrees rms (joint space)", [f"{G}heldout_start_90_deg_rms"], x=x, y_title="deg rms"),
        line("Memorisation ratio: held-out / train at start 90", [f"{G}ratio_start_90"], x=x, log=True),
        line("Poses under 2.5e-4 (of 10)", [f"{G}poses_under_threshold"], x=x),
        line("Held-out by chunk start: 0 (survey move), 90 (survey frame), 180-360",
             [f"{G}heldout_start_0", f"{G}heldout_start_90", f"{G}heldout_starts_180_270_360"], x=x, log=True,
             titles={f"{G}heldout_start_0": "start 0", f"{G}heldout_start_90": "start 90", f"{G}heldout_starts_180_270_360": "starts 180-360"}),
        line("Every held-out pose at start 90 (log)", [], x=x, log=True, regex=f"{G}pose/.*"),
    ])


def plumbing(phase_title: str, x: str = STEP) -> list[ws.Section]:
    return [
        ws.Section(name="04 Throughput", is_open=False, panels=[
            line("Samples per second", [f"{P}samples_per_s"], x=x, smoothing=0.6),
            line("Episodes per hour (RL)", [f"{P}episodes_per_hour"], x=x, smoothing=0.6),
            line("Seconds per iteration: env / policy / update (RL)", [f"{P}env_seconds", f"{P}policy_seconds", f"{P}update_seconds"], x=x),
            line("Fraction of time waiting on data", [f"{P}data_wait_fraction"], x=x),
            line("GPU peak memory (GB, 24 GB card) and host RSS", [f"{P}gpu_peak_gb", f"{P}host_rss_gb"], x=x),
            line("ETA (hours)", [f"{P}eta_hours"], x=x),
        ]),
        ws.Section(name="05 Phases", is_open=False, panels=[line(phase_title, [f"{H}phase"], x=x),
                                                            line("Episodes done of the budget", [f"{H}episodes_done", f"{H}episodes_target"], x=x)]),
    ]


def finetune_view(entity: str, project: str, family: str) -> ws.Workspace:
    label = TITLES[family]
    training = ws.Section(name="03 Training", is_open=True, panels=[
        line("Loss (log, smoothed)", [f"{T}flow_loss", f"{T}loss"], log=True, smoothing=0.8),
        line("Loss per joint (log, smoothed)", [], log=True, smoothing=0.8, regex=f"{T}(flow_)?loss_.*"),
        line("Gradient norm (pre-clip)", [f"{T}grad_norm"], smoothing=0.6),
        line("Learning rate", [], log=True, regex=f"{T}lr_.*"),
        line("Epochs over the training frames", [f"{T}epoch"]),
    ])
    return ws.Workspace(entity=entity, project=project, name=f"{label} fine-tune: outcome first",
                        sections=[outcome_section(label), generalisation_section(label), training,
                                  *plumbing("Phase (0 load, 1 train, 2 score, 3 rollouts, 4 complete)")],
                        settings=ws.WorkspaceSettings(x_axis=STEP, max_runs=10, group_by_prefix="first"),
                        runset_settings=ws.RunsetSettings(query=f"{family}-"))


def rl_sections(x: str = EPISODES) -> list[ws.Section]:
    monitor = ws.Section(name="01 Outcome: monitoring suite during RL, held-out at the end", is_open=True, panels=[
        line("Monitoring-suite success rate (20 unseen placements, noise off)", [f"{O}monitor_success_rate", f"{O}ref_start_success_rate"], x=x,
             titles={f"{O}monitor_success_rate": "RL policy", f"{O}ref_start_success_rate": "fine-tune it started from"},
             marks={f"{O}ref_start_success_rate": "dashed"}),
        line("Monitoring pickups and closure inside 8 mm", [f"{O}monitor_pickup_rate", f"{O}monitor_closure_within_8mm_rate"], x=x),
        line("Training-episode success and pickup rate (exploration noise on, smoothed)", [f"{O}train_success_rate", f"{O}train_pickup_rate"], x=x, smoothing=0.8),
        line("Safety frames per episode (training)", [f"{O}train_safety_frames_mean"], x=x, smoothing=0.6),
        wr.BarPlot(title="Held-out successes of 30 at the end (exec 90 / 30)", metrics=[f"{O}successes_exec90", f"{O}successes_exec30"], orientation="v"),
        wr.BarPlot(title="Held-out closure lateral median (mm)", metrics=[f"{O}closure_lateral_median_mm_exec90", f"{O}closure_lateral_median_mm_exec30"], orientation="v"),
        wr.MediaBrowser(title="Summary figure", media_keys=[f"{O}summary_figure"], num_columns=1),
    ])
    rl = ws.Section(name="03 Training: reward, policy update, critic", is_open=True, panels=[
        line("Episode return (smoothed) and reward parts", [f"{T}return_mean", f"{T}reward_success", f"{T}reward_shaping", f"{T}reward_safety"], x=x, smoothing=0.8),
        line("KL (post-update) vs target", [f"{T}kl", f"{T}kl_target"], x=x, log=True, marks={f"{T}kl_target": "dashed"}),
        line("Clip fraction", [f"{T}clip_fraction"], x=x, smoothing=0.6),
        line("Exploration noise sigma (mean / min / max)", [f"{T}sigma_mean", f"{T}sigma_min", f"{T}sigma_max"], x=x),
        line("Log-prob per dimension (old / new)", [f"{T}logprob_old_per_dim", f"{T}logprob_new_per_dim"], x=x),
        line("Critic: value loss and explained variance", [f"{T}value_loss", f"{T}explained_variance"], x=x),
        line("Advantage mean / std", [f"{T}advantage_mean", f"{T}advantage_std"], x=x),
        line("Actor grad norm / learning rates", [f"{T}grad_norm", f"{T}lr_actor", f"{T}lr_critic"], x=x, log=True),
    ])
    return [monitor, rl, *plumbing("Phase (0 load, 1 warmup, 2 train, 3 heldout, 4 complete)", x=x)]


def rl_view(entity: str, project: str, query: str) -> ws.Workspace:
    return ws.Workspace(entity=entity, project=project, name="RL on the bench task: outcome first",
                        sections=rl_sections(), settings=ws.WorkspaceSettings(x_axis=EPISODES, max_runs=10, group_by_prefix="first"),
                        runset_settings=ws.RunsetSettings(query=query))


def tranche_view(entity: str, project: str, query: str, name: str) -> ws.Workspace:
    ft = ws.Section(name="00 Fine-tunes: held-out curve and closed loop", is_open=True, panels=[
        line("Held-out start-90 MSE (log) against the reference levels",
             [f"{G}heldout_start_90", f"{G}ref_cnn4800_heldout_start_90", f"{G}ref_working_policy_start_90"],
             log=True, titles=REF_TITLES, marks=REF_MARKS),
        wr.BarPlot(title="Held-out successes of 30, every run", metrics=[f"{O}successes_exec90", f"{O}successes_exec30"], orientation="v"),
    ])
    return ws.Workspace(entity=entity, project=project, name=name, sections=[*rl_sections()[:2], ft, *rl_sections()[2:]],
                        settings=ws.WorkspaceSettings(x_axis=EPISODES, max_runs=20, group_by_prefix="first"),
                        runset_settings=ws.RunsetSettings(query=query, regex_query=True))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=("pi05", "smolvla", "act", "rl", "tranche"))
    p.add_argument("--entity", default=None)
    p.add_argument("--project", default="so-arm101-v2-scaling")
    p.add_argument("--query", default=None, help="run-name filter (rl: 'rl-'; tranche: a regex over the tranche's run names)")
    p.add_argument("--name", default="Retry tranche 2026-10-04: every run side by side")
    args = p.parse_args(argv)
    entity = args.entity or wandb.Api().default_entity
    if args.family in TITLES:
        view = finetune_view(entity, args.project, args.family)
    elif args.family == "rl":
        view = rl_view(entity, args.project, args.query or "rl-")
    else:
        view = tranche_view(entity, args.project, args.query or "^(rl-|smolvla-|act-|rerun-)", args.name)
    print(view.save().url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
