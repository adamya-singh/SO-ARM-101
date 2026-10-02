"""Saved W&B workspace view for the pi0.5 runs (project so-arm101-v2-scaling, group pi05), 2026-10-02.

The repository convention (AGENTS.md) numbers the metric sections so the decisive charts come first; this
view goes further and composes the panels: each generalisation chart overlays the run's held-out curve on
the two reference levels it must be read against (the CNN-4800 run, 3.6e-4, and the working fixed-square
policy, 2.5e-6) on a log axis, the outcome section puts closed-loop success and closure millimetres side
by side for both execution schedules, and the plumbing (throughput, phases) sits collapsed at the bottom.

Usage (needs network; idempotent: re-running replaces nothing, it saves a new view and prints its URL):
  ~/miniforge3/envs/lerobot-pi05/bin/python tools/pi05_wandb_workspace.py [--entity E] [--project P]
"""
from __future__ import annotations

import argparse

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

G = "02_generalisation/"
O = "01_outcome/"
T = "03_training/"
X = "03_training/step"


def line(title, ys, *, log=False, titles=None, marks=None, smoothing=None, regex=None, y_title=None):
    return wr.LinePlot(title=title, x=X, y=ys, log_y=log, line_titles=titles, line_marks=marks, title_x="training step",
                       title_y=y_title, metric_regex=regex,
                       smoothing_type=("exponential" if smoothing else None), smoothing_factor=smoothing,
                       smoothing_show_original=True if smoothing else None)


def build(entity: str, project: str) -> ws.Workspace:
    ref_marks = {f"{G}ref_cnn4800_heldout_start_90": "dashed", f"{G}ref_working_policy_start_90": "dotted",
                 f"{G}ref_pose_threshold_2.5e-4": "dotted"}
    ref_titles = {f"{G}ref_cnn4800_heldout_start_90": "CNN 4800 held-out (3.6e-4)",
                  f"{G}ref_working_policy_start_90": "working fixed-square policy (2.5e-6)",
                  f"{G}ref_pose_threshold_2.5e-4": "level of past successes (2.5e-4)"}
    outcome = ws.Section(name="01 Outcome: closed loop on 10 unseen placements x 3", is_open=True, panels=[
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
    generalisation = ws.Section(name="02 Generalisation: held-out start-90 chunk error during training", is_open=True, panels=[
        line("Held-out vs train start-90 MSE against the two reference levels (log)",
             [f"{G}heldout_start_90", f"{G}train_start_90", f"{G}ref_cnn4800_heldout_start_90", f"{G}ref_working_policy_start_90"],
             log=True, titles={f"{G}heldout_start_90": "pi0.5 held-out", f"{G}train_start_90": "pi0.5 train", **ref_titles}, marks=ref_marks,
             y_title="MSE (1 unit = pi rad)"),
        line("Per-pose median and worst pose (log)",
             [f"{G}median", f"{G}max", f"{G}ref_pose_threshold_2.5e-4", f"{G}ref_working_policy_start_90"], log=True,
             titles={f"{G}median": "per-pose median", f"{G}max": "worst pose", **ref_titles}, marks=ref_marks),
        line("Held-out error in degrees rms (joint space)", [f"{G}heldout_start_90_deg_rms"], y_title="deg rms"),
        line("Memorisation ratio: held-out / train at start 90", [f"{G}ratio_start_90"], log=True),
        line("Poses under 2.5e-4 (of 10)", [f"{G}poses_under_threshold"]),
        line("Held-out by chunk start: 0 (survey move), 90 (survey frame), 180-360 (close-up and after)",
             [f"{G}heldout_start_0", f"{G}heldout_start_90", f"{G}heldout_starts_180_270_360"], log=True,
             titles={f"{G}heldout_start_0": "start 0", f"{G}heldout_start_90": "start 90", f"{G}heldout_starts_180_270_360": "starts 180-360"}),
        line("Every held-out pose at start 90 (log)", [], log=True, regex=f"{G}pose/.*"),
    ])
    training = ws.Section(name="03 Training", is_open=True, panels=[
        line("Flow-matching loss (log, smoothed)", [f"{T}flow_loss"], log=True, smoothing=0.8),
        line("Flow-matching loss per joint (log, smoothed)", [], log=True, smoothing=0.8, regex=f"{T}flow_loss_.*"),
        line("Gradient norm (pre-clip, clip at 1.0)", [f"{T}grad_norm"], smoothing=0.6),
        line("Learning rate (expert / LoRA)", [f"{T}lr_expert", f"{T}lr_lora"], log=True),
        line("Epochs over the 141k training frames", [f"{T}epoch"]),
    ])
    throughput = ws.Section(name="04 Throughput", is_open=False, panels=[
        line("Samples per second", ["04_throughput/samples_per_s"], smoothing=0.6),
        line("Fraction of time waiting on data", ["04_throughput/data_wait_fraction"]),
        line("GPU peak memory (GB, 24 GB card)", ["04_throughput/gpu_peak_gb"]),
        line("ETA (hours)", ["04_throughput/eta_hours"]),
    ])
    phases = ws.Section(name="05 Phases", is_open=False, panels=[
        line("Phase (0 load, 1 train, 2 score, 3 rollouts, 4 complete)", ["05_phases/phase"]),
    ])
    return ws.Workspace(entity=entity, project=project, name="pi0.5 fine-tune: outcome first",
                        sections=[outcome, generalisation, training, throughput, phases],
                        settings=ws.WorkspaceSettings(x_axis=X, max_runs=10, group_by_prefix="first"),
                        runset_settings=ws.RunsetSettings(query="pi05"))   # run names all start with pi05-


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default=None)
    p.add_argument("--project", default="so-arm101-v2-scaling")
    args = p.parse_args(argv)
    entity = args.entity or wandb.Api().default_entity
    view = build(entity, args.project).save()
    print(view.url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
