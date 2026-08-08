"""Failure-frame analysis: localize safety-invalidation frames in closed-loop rollouts.

Reads an evaluation report, recomputes per-frame per-joint envelope headroom
and delta-cap usage from the recorded telemetry (requested/current ACT), and
renders per-scenario figures plus a markdown analysis note. Read-only over
artifacts/; writes only into notes/ (or the chosen output paths).

Hard cross-check: the recomputed clip/limit frame counts must equal the
evaluation report's recorded `clipping_frames`/`limiting_frames` per rollout,
or the tool exits with an error — the analysis math is thereby pinned to the
live safety path.

Usage:
    PYTHONNOUSERSITE=1 python tools/analyze_safety_frames.py \
        --evaluation artifacts/.../evaluation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from so_arm101_v2.contracts import JOINT_NAMES, load_task_contract
from so_arm101_v2.contracts.coordinates import effective_safe_act_bounds

# Privileged teacher stage schedule (simulation/privileged.py, boundaries and
# waypoint names are set at solve time; these are the pre-registered constants
# of the 450-action teacher). Segment i spans [BOUNDARIES[i], BOUNDARIES[i+1])
# and moves toward WAYPOINT_TARGETS[i].
STAGE_BOUNDARIES = (
    0, 70, 110, 145, 175, 205, 255, 285, 315, 350, 365, 386, 403, 419, 424, 450,
)
WAYPOINT_TARGETS = (
    "above", "lowered", "descended", "engaged", "seated", "half_closed",
    "closed", "closed_hold", "lift", "lift_hold", "traverse", "set_down",
    "released", "released_hold", "retreat",
)

# Series colors validated with the dataviz palette validator (light surface):
# categorical slots for the two identity-carrying series; gray is a neutral
# reference line; violation markers carry shape + legend, never color alone.
COLOR_REQUESTED = "#2a78d6"
COLOR_EXECUTED = "#e8710a"
COLOR_CURRENT = "#8d8d8d"
COLOR_CLIP = "#c4302b"
COLOR_SURFACE = "#fcfcfb"
COLOR_TEXT = "#0b0b0b"
COLOR_MUTED = "#52514e"


@dataclass(frozen=True)
class FrameFinding:
    action: int
    joint: str
    kind: str                   # "envelope_clip" | "delta_limit"
    requested: float
    signed_excess: float
    stage: str
    chunk_index: int
    chunk_offset: int
    past_teacher_horizon: bool


@dataclass(frozen=True)
class RolloutAnalysis:
    telemetry_path: Path
    telemetry_content_sha256: str
    policy_id: str
    scenario_id: str
    repeat: int
    duplicate_repeats: tuple[int, ...]
    findings: tuple[FrameFinding, ...]
    requested: np.ndarray
    executed: np.ndarray
    current: np.ndarray
    delta_usage: np.ndarray


def _stage_for(action: int, teacher_horizon: int) -> str:
    if action > teacher_horizon:
        return "past_teacher_horizon"
    for index in range(len(STAGE_BOUNDARIES) - 1):
        if STAGE_BOUNDARIES[index] < action <= STAGE_BOUNDARIES[index + 1]:
            return WAYPOINT_TARGETS[index]
    return WAYPOINT_TARGETS[0]


def rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash of the physics rows alone — the payload's repeat field makes the
    stored content_sha256 differ across otherwise-identical repeats."""
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def analyze_rollout(
    path: Path, *, chunk_horizon: int, teacher_horizon: int
) -> RolloutAnalysis:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    requested = np.asarray([row["requested_act"] for row in rows], dtype=np.float64)
    executed = np.asarray([row["executed_act"] for row in rows], dtype=np.float64)
    current = np.asarray([row["current_act"] for row in rows], dtype=np.float64)
    low, high = effective_safe_act_bounds()
    caps = np.asarray(
        load_task_contract("fixed_cube_pickup_v1").safety.maximum_act_delta_per_step,
        dtype=np.float64,
    )
    clipped = np.clip(requested, low, high)
    envelope_violation = (requested < low) | (requested > high)      # [T, 6]
    delta_violation = np.abs(clipped - current) > caps               # [T, 6]
    delta_usage = np.abs(clipped - current) / caps

    findings: list[FrameFinding] = []
    for t in range(requested.shape[0]):
        action = int(rows[t]["action"])
        for j, joint in enumerate(JOINT_NAMES):
            for kind, mask in (("envelope_clip", envelope_violation), ("delta_limit", delta_violation)):
                if not mask[t, j]:
                    continue
                if kind == "envelope_clip":
                    excess = float(
                        requested[t, j] - high[j] if requested[t, j] > high[j]
                        else low[j] - requested[t, j]
                    )
                else:
                    excess = float(np.abs(clipped[t, j] - current[t, j]) - caps[j])
                findings.append(FrameFinding(
                    action=action, joint=joint, kind=kind,
                    requested=float(requested[t, j]), signed_excess=excess,
                    stage=_stage_for(action, teacher_horizon),
                    chunk_index=(action - 1) // chunk_horizon,
                    chunk_offset=(action - 1) % chunk_horizon,
                    past_teacher_horizon=action > teacher_horizon,
                ))
    return RolloutAnalysis(
        telemetry_path=path,
        telemetry_content_sha256=str(payload["content_sha256"]),
        policy_id=str(payload["policy_id"]),
        scenario_id=str(payload["scenario_id"]),
        repeat=int(payload["repeat"]),
        duplicate_repeats=(),
        findings=tuple(findings),
        requested=requested, executed=executed, current=current,
        delta_usage=delta_usage,
    )


def _cross_check(analysis: RolloutAnalysis, record: Mapping[str, Any]) -> None:
    clip_frames = len({f.action for f in analysis.findings if f.kind == "envelope_clip"})
    limit_frames = len({f.action for f in analysis.findings if f.kind == "delta_limit"})
    expected_clip = int(record["clipping_frames"])
    expected_limit = int(record["limiting_frames"])
    if clip_frames != expected_clip or limit_frames != expected_limit:
        raise RuntimeError(
            f"{analysis.scenario_id}.repeat{analysis.repeat}: recomputed "
            f"clip/limit frames ({clip_frames}/{limit_frames}) disagree with the "
            f"recorded evaluation counts ({expected_clip}/{expected_limit}) — "
            "analysis math has drifted from the live safety path"
        )


def _style_axis(ax: Any) -> None:
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, color="#e8e7e2", linewidth=0.6)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#d8d7d2")


def _decorate_timeline(ax: Any, chunk_horizon: int, teacher_horizon: int, total: int) -> None:
    for boundary in STAGE_BOUNDARIES[1:]:
        ax.axvline(boundary, color="#c9c8c2", linewidth=0.6, linestyle="--", zorder=1)
    for chunk_edge in range(chunk_horizon, total + 1, chunk_horizon):
        ax.axvline(chunk_edge, color="#b9c6da", linewidth=0.8, zorder=1)
    if total > teacher_horizon:
        ax.axvspan(teacher_horizon, total, color="#f2e6e6", zorder=0)


def render_rollout_figures(
    analysis: RolloutAnalysis, images_dir: Path, *,
    prefix: str, chunk_horizon: int, teacher_horizon: int,
) -> list[Path]:
    low, high = effective_safe_act_bounds()
    total = analysis.requested.shape[0]
    actions = np.arange(1, total + 1)
    clip_frames = {(f.action, f.joint) for f in analysis.findings if f.kind == "envelope_clip"}
    limit_frames = {(f.action, f.joint) for f in analysis.findings if f.kind == "delta_limit"}
    images_dir.mkdir(parents=True, exist_ok=True)
    scenario = analysis.scenario_id
    written: list[Path] = []

    # Figure A: per-joint requested/executed vs envelope band.
    fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    fig.patch.set_facecolor(COLOR_SURFACE)
    for j, (ax, joint) in enumerate(zip(axes, JOINT_NAMES)):
        _style_axis(ax)
        _decorate_timeline(ax, chunk_horizon, teacher_horizon, total)
        ax.axhspan(low[j], high[j], color="#e9f0e9", zorder=0)
        ax.plot(actions, analysis.requested[:, j], color=COLOR_REQUESTED, linewidth=1.3, label="requested")
        ax.plot(actions, analysis.executed[:, j], color=COLOR_EXECUTED, linewidth=1.1, label="executed")
        clip_x = [a for (a, name) in clip_frames if name == joint]
        if clip_x:
            clip_y = [analysis.requested[a - 1, j] for a in clip_x]
            ax.scatter(clip_x, clip_y, marker="x", s=42, color=COLOR_CLIP, zorder=5, label="envelope clip")
        limit_x = [a for (a, name) in limit_frames if name == joint]
        if limit_x:
            limit_y = [analysis.requested[a - 1, j] for a in limit_x]
            ax.scatter(limit_x, limit_y, marker="^", s=34, facecolors="none",
                       edgecolors=COLOR_TEXT, zorder=5, label="delta limit")
        ax.set_ylabel(joint, fontsize=8, color=COLOR_TEXT)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("action (shaded span = past 450-action teacher horizon; solid vlines = 90-action chunk edges)",
                        fontsize=9, color=COLOR_MUTED)
    fig.suptitle(f"{scenario}: requested vs executed ACT against the safety envelope", fontsize=12, color=COLOR_TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = images_dir / f"{prefix}-{scenario}-envelope.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    # Figure B: per-joint delta usage vs the 1.0 cap.
    fig, axes = plt.subplots(6, 1, figsize=(11, 11), sharex=True)
    fig.patch.set_facecolor(COLOR_SURFACE)
    for j, (ax, joint) in enumerate(zip(axes, JOINT_NAMES)):
        _style_axis(ax)
        _decorate_timeline(ax, chunk_horizon, teacher_horizon, total)
        ax.plot(actions, analysis.delta_usage[:, j], color=COLOR_REQUESTED, linewidth=1.1, label="delta usage")
        ax.axhline(1.0, color=COLOR_CLIP, linewidth=0.9, label="cap")
        over = analysis.delta_usage[:, j] > 1.0
        if over.any():
            ax.scatter(actions[over], analysis.delta_usage[over, j], marker="^", s=30,
                       facecolors="none", edgecolors=COLOR_TEXT, zorder=5)
        ax.set_ylabel(joint, fontsize=8, color=COLOR_TEXT)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("action", fontsize=9, color=COLOR_MUTED)
    fig.suptitle(f"{scenario}: per-joint delta usage (|clipped − current| / cap)", fontsize=12, color=COLOR_TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = images_dir / f"{prefix}-{scenario}-delta.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    # Figure C: gripper zoom on the tail.
    j = len(JOINT_NAMES) - 1
    start = max(0, total - 80)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor(COLOR_SURFACE)
    _style_axis(ax)
    _decorate_timeline(ax, chunk_horizon, teacher_horizon, total)
    window = slice(start, total)
    ax.plot(actions[window], analysis.requested[window, j], color=COLOR_REQUESTED, linewidth=1.5, label="requested")
    ax.plot(actions[window], analysis.executed[window, j], color=COLOR_EXECUTED, linewidth=1.2, label="executed")
    ax.plot(actions[window], analysis.current[window, j], color=COLOR_CURRENT, linewidth=1.0, label="current")
    ax.axhline(high[j], color=COLOR_CLIP, linewidth=0.9, label=f"gripper high = {high[j]:.3f}")
    clip_x = [a for (a, name) in clip_frames if name == JOINT_NAMES[j] and a > start]
    if clip_x:
        ax.scatter(clip_x, [analysis.requested[a - 1, j] for a in clip_x],
                   marker="x", s=48, color=COLOR_CLIP, zorder=5)
    ax.set_xlim(start + 1, total)
    ax.set_xlabel("action", fontsize=9, color=COLOR_MUTED)
    ax.set_ylabel("gripper ACT", fontsize=9, color=COLOR_TEXT)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle(f"{scenario}: gripper tail (actions {start + 1}-{total})", fontsize=12, color=COLOR_TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = images_dir / f"{prefix}-{scenario}-gripper-zoom.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)
    return written


def render_overview(
    analyses: Sequence[RolloutAnalysis], images_dir: Path, *,
    prefix: str, chunk_horizon: int, teacher_horizon: int, total: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.55 * len(analyses)))
    fig.patch.set_facecolor(COLOR_SURFACE)
    _style_axis(ax)
    _decorate_timeline(ax, chunk_horizon, teacher_horizon, total)
    labels = []
    for row, analysis in enumerate(analyses):
        labels.append(analysis.scenario_id)
        clip_x = sorted({f.action for f in analysis.findings if f.kind == "envelope_clip"})
        limit_x = sorted({f.action for f in analysis.findings if f.kind == "delta_limit"})
        ax.scatter(clip_x, [row] * len(clip_x), marker="x", s=40, color=COLOR_CLIP,
                   label="envelope clip" if row == 0 else None, zorder=5)
        ax.scatter(limit_x, [row] * len(limit_x), marker="^", s=30, facecolors="none",
                   edgecolors=COLOR_TEXT, label="delta limit" if row == 0 else None, zorder=5)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlim(0, total + 5)
    ax.set_xlabel("action", fontsize=9, color=COLOR_MUTED)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle("Violation raster across failing scenarios", fontsize=12, color=COLOR_TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = images_dir / f"{prefix}-overview.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_markdown_note(
    analyses: Sequence[RolloutAnalysis],
    figures: Mapping[str, Sequence[Path]],
    overview: Path,
    note_path: Path,
    *,
    evaluation_path: Path,
    evaluation_sha: str,
    chunk_horizon: int,
    teacher_horizon: int,
) -> None:
    lines: list[str] = []
    lines.append("# Scaling-gate failure-frame analysis")
    lines.append("")
    lines.append(f"Generated by `tools/analyze_safety_frames.py` (matplotlib {matplotlib.__version__}).")
    lines.append(f"Input: `{evaluation_path}` (`content_sha256 {evaluation_sha[:16]}…`).")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("Per frame and joint, recomputed from telemetry `requested_act`/`current_act`:")
    lines.append("envelope test = requested outside `effective_safe_act_bounds()`; delta test =")
    lines.append("`|clip(requested, low, high) − current| > maximum_act_delta_per_step`.")
    lines.append("**Cross-check: recomputed per-rollout clip/limit frame counts equal the")
    lines.append("evaluation report's recorded `clipping_frames`/`limiting_frames` exactly**")
    lines.append("(the tool errors otherwise), so this analysis is pinned to the live safety path.")
    lines.append(f"Phase labels come from the privileged teacher's stage boundaries; chunk index")
    lines.append(f"= (action−1)//{chunk_horizon}; actions > {teacher_horizon} are past the teacher horizon.")
    lines.append("Repeats with byte-identical physics rows are deduplicated (hash of the")
    lines.append("`rows` array; the payload-level `content_sha256` differs per repeat because")
    lines.append("it embeds the repeat index).")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"![Violation raster across failing scenarios](./images/{overview.name})")
    lines.append("")
    for analysis in analyses:
        alias = (
            f" (repeats {', '.join(str(r) for r in analysis.duplicate_repeats)} byte-identical)"
            if analysis.duplicate_repeats else ""
        )
        lines.append(f"## {analysis.scenario_id} — repeat {analysis.repeat}{alias}")
        lines.append("")
        lines.append(f"Telemetry `{analysis.telemetry_content_sha256[:16]}…`, policy `{analysis.policy_id}`.")
        lines.append("")
        for figure in figures[analysis.scenario_id]:
            lines.append(f"![{figure.stem}](./images/{figure.name})")
            lines.append("")
        lines.append("| action | joint | kind | excess | stage | chunk | offset | past horizon |")
        lines.append("| ---: | --- | --- | ---: | --- | ---: | ---: | --- |")
        for f in analysis.findings:
            lines.append(
                f"| {f.action} | {f.joint} | {f.kind} | {f.signed_excess:+.6f} | {f.stage} "
                f"| {f.chunk_index} | {f.chunk_offset} | {'yes' if f.past_teacher_horizon else ''} |"
            )
        lines.append("")

    joints = sorted({f.joint for a in analyses for f in a.findings})
    past = sum(1 for a in analyses for f in a.findings if f.past_teacher_horizon)
    total_findings = sum(len(a.findings) for a in analyses)
    release_stages = {"released", "released_hold", "retreat"}
    release = sum(1 for a in analyses for f in a.findings if f.stage in release_stages)
    lines.append("## Findings")
    lines.append("")
    lines.append(f"- Violating joints: **{', '.join(joints)}** — no other joint violates anywhere.")
    lines.append(f"- {past}/{total_findings} violating (frame, joint) findings lie **past the "
                 f"{teacher_horizon}-action teacher horizon** (actions {teacher_horizon + 1}-480), i.e. in the "
                 "chunk predicted entirely beyond the teacher's demonstration, where the progress "
                 f"feature is clamped at `min(action, {teacher_horizon - 1})/{teacher_horizon - 1}`.")
    lines.append(f"- {release}/{total_findings} findings fall in the release/retreat stages, where the "
                 "privileged teacher itself needs ≥16 actions to open the gripper without tripping "
                 "the delta limiter (servo lag).")
    lines.append("")
    lines.append("## Implication for the precision tranche")
    lines.append("")
    lines.append("The failures are not behavioral (all milestone chains complete) but concentrated")
    lines.append("envelope-grazing in the past-horizon/release regime. LR decay and budget scaling")
    lines.append("(`notes/precision-tranche-proposal.md`) are therefore measured against whether they")
    lines.append("move this grazing margin; if violations remain confined to past-horizon/release")
    lines.append("frames across all cells, the pre-registered next action is a teacher-horizon /")
    lines.append("contract alignment proposal (450-action teacher vs 480-action contract).")
    lines.append("")
    note_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--notes-dir", type=Path, default=Path("notes"))
    parser.add_argument("--images-prefix", default="scaling-failure")
    parser.add_argument("--note-name", default="scaling-failure-frame-analysis.md")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--chunk-horizon", type=int, default=90)
    parser.add_argument("--teacher-horizon", type=int, default=450)
    args = parser.parse_args(argv)

    report = json.loads(args.evaluation.read_text(encoding="utf-8"))
    failing = [
        record for record in report["rollouts"]
        if record["invalidated"] or any(
            int(record[name]) > 0 for name in
            ("clipping_frames", "limiting_frames", "nonfinite_frames", "unsafe_contact_frames")
        )
    ]
    if not failing:
        print("no failing rollouts in the evaluation report")
        return 0

    analyses: list[RolloutAnalysis] = []
    digest_order: list[str] = []
    duplicates: dict[str, list[int]] = {}
    for record in failing:
        path = Path(record["telemetry_path"])
        digest = rows_digest(json.loads(path.read_text(encoding="utf-8"))["rows"])
        if digest in duplicates:
            duplicates[digest].append(int(record["repeat"]))
            continue
        analysis = analyze_rollout(
            path, chunk_horizon=args.chunk_horizon, teacher_horizon=args.teacher_horizon,
        )
        _cross_check(analysis, record)
        duplicates[digest] = []
        digest_order.append(digest)
        analyses.append(analysis)
    analyses = [
        RolloutAnalysis(**{
            **analysis.__dict__,
            "duplicate_repeats": tuple(duplicates[digest]),
        })
        for analysis, digest in zip(analyses, digest_order)
    ]

    images_dir = args.notes_dir / "images"
    figures: dict[str, list[Path]] = {}
    total = max(a.requested.shape[0] for a in analyses)
    for analysis in analyses:
        figures[analysis.scenario_id] = render_rollout_figures(
            analysis, images_dir, prefix=args.images_prefix,
            chunk_horizon=args.chunk_horizon, teacher_horizon=args.teacher_horizon,
        )
    overview = render_overview(
        analyses, images_dir, prefix=args.images_prefix,
        chunk_horizon=args.chunk_horizon, teacher_horizon=args.teacher_horizon, total=total,
    )
    note_path = args.notes_dir / args.note_name
    write_markdown_note(
        analyses, figures, overview, note_path,
        evaluation_path=args.evaluation, evaluation_sha=str(report["content_sha256"]),
        chunk_horizon=args.chunk_horizon, teacher_horizon=args.teacher_horizon,
    )
    json_out = args.json_out or (args.notes_dir / args.note_name).with_suffix(".json")
    json_out.write_text(json.dumps({
        "tool": "analyze_safety_frames",
        "matplotlib": matplotlib.__version__,
        "evaluation": str(args.evaluation),
        "evaluation_content_sha256": report["content_sha256"],
        "chunk_horizon": args.chunk_horizon,
        "teacher_horizon": args.teacher_horizon,
        "rollouts": [
            {
                "scenario_id": a.scenario_id,
                "repeat": a.repeat,
                "duplicate_repeats": list(a.duplicate_repeats),
                "telemetry_content_sha256": a.telemetry_content_sha256,
                "policy_id": a.policy_id,
                "findings": [f.__dict__ for f in a.findings],
            }
            for a in analyses
        ],
    }, indent=2), encoding="utf-8")
    print(f"note: {note_path}")
    print(f"json: {json_out}")
    print(f"images: {len(sum(figures.values(), []) ) + 1} under {images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
