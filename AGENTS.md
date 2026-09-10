# Agent instructions (Codex, Claude Code, any other agent)

This repository trains vision policies for the SO-ARM-101 bench task in a MuJoCo simulator and
runs them on the physical arm. The governing runbook is `notes/bench-pick-replace-v1.md`; the
running log is `notes/vision-rung-notebook.md`.

## Skills

Skills live under `.claude/skills/<name>/SKILL.md`. They are plain Markdown procedures; an agent
that cannot load them as skills should read the file and follow it.

- `.claude/skills/experiment-analyze-explain/SKILL.md`: **experiment, analyze, explain.** Use it
  whenever an ML run underperforms its baseline, a result is surprising, or the user asks why a
  run behaved as it did or what to scale (data, model, steps). It prescribes the measurements
  (loss curves against a baseline, error decomposition along the task, train versus held-out on
  the same metric, capacity where the error concentrates, ambiguity in the data, pipeline
  parity), the decision rules, what to record, and how to explain the result to a new intern in
  plain technical English. Agents may invoke it on their own when a result would teach a new
  intern something; the user invokes it as `/experiment-analyze-explain`.

## Conventions that matter

- W&B panel order (user request 2026-09-10): log metrics under numbered section prefixes so the
  most important charts come first: `01_outcome/` (closed-loop success rates, safety frames,
  gate results), `02_generalisation/` (train vs held-out loss at chunk start 90 and the ratio),
  `03_training/` (loss), `04_throughput/` (steps/s, checkpoint age), `05_phases/` (gate flags).
  Sections sort alphabetically in the run workspace; keep the numbering on every new metric.

- Python runs with `PYTHONNOUSERSITE=1` and `MUJOCO_GL=egl` from
  `/home/win10ubuntu/miniforge3/envs/lerobot/bin/python`; MuJoCo is pinned to 3.9.0.
- `artifacts/` is git-ignored; evidence files are force-added individually.
- Commit each tranche of work as it lands (code + tests + note).
- Never weaken the strict-grasp criterion, recalibrate the servos, or widen mechanical limits;
  shoulder lift never below -92 calibrated units.
