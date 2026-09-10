"""Summarise one physical episode directory: joint traces, holds, boundary observations and a video contact sheet.

Read-only over the evidence written by tools/run_physical_episode.py (run.json,
steps.csv, boundaries/, camera.mp4). Writes <run-dir>/analysis/{summary.json,
joints.png, video_sheet.png} and prints the summary. Nothing here touches the
arm.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so_arm101_v2.contracts import JOINT_NAMES  # noqa: E402
from so_arm101_v2.contracts.physical import act_to_physical_normalized  # noqa: E402


def load_steps(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def summarize(run_dir: Path, *, video_samples: int = 16) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    summary: dict = dict(run_dir=str(run_dir), status=run.get("status"), reason=run.get("reason"), actions=run.get("actions"),
                         hold_frames=run.get("hold_frames"), aborted_reason=run.get("aborted_reason"), timing=run.get("timing"),
                         servo_voltage=run.get("servo_voltage", {}).get("volts"), approach=run.get("approach"),
                         real_frame_check={k: v for k, v in (run.get("real_frame_check") or {}).items() if k in ("passed", "reasons", "label")},
                         start_pose_delta_act=run.get("start_pose_delta_act"), prefix=run.get("prefix"))
    steps_path = run_dir / "steps.csv"
    if steps_path.exists():
        rows = load_steps(steps_path)
        if rows:
            measured = np.array([[float(r[f"measured_act_{n}"]) for n in JOINT_NAMES] for r in rows], dtype=np.float32)
            measured_physical = np.stack([act_to_physical_normalized(m) for m in measured])
            sent = np.array([[float(r[f"sent_physical_{n}"]) for n in JOINT_NAMES] for r in rows])
            policy = np.array([[float(r[f"policy_act_{n}"]) for n in JOINT_NAMES] for r in rows], dtype=np.float32)
            policy_physical = np.stack([act_to_physical_normalized(p) for p in policy])
            holds = [r["hold_reason"] for r in rows if r["hold_reason"]]
            reasons = Counter(h.split(":")[0] + ":" + h.split(":")[1] if ":" in h else h for h in holds)
            summary.update(
                steps=len(rows), boundaries=[int(r["step"]) for r in rows if r["boundary"] in ("1", "True", "true")],
                hold_reasons=dict(reasons.most_common()),
                first_hold_step=next((int(r["step"]) for r in rows if r["hold_reason"]), None),
                measured_physical_start=[round(float(v), 2) for v in measured_physical[0]],
                measured_physical_end=[round(float(v), 2) for v in measured_physical[-1]],
                measured_physical_min={n: round(float(v), 2) for n, v in zip(JOINT_NAMES, measured_physical.min(axis=0))},
                measured_physical_max={n: round(float(v), 2) for n, v in zip(JOINT_NAMES, measured_physical.max(axis=0))},
                policy_physical_min={n: round(float(v), 2) for n, v in zip(JOINT_NAMES, policy_physical.min(axis=0))},
                policy_physical_max={n: round(float(v), 2) for n, v in zip(JOINT_NAMES, policy_physical.max(axis=0))},
                max_abs_delta_from_start_units={n: round(float(v), 2) for n, v in zip(JOINT_NAMES, np.max(np.abs(measured_physical - measured_physical[0]), axis=0))},
                lateness_ms_max=max(float(r["lateness_ms"]) for r in rows), step_ms_max=max(float(r["step_ms"]) for r in rows),
                frame_age_ms_at_boundaries=[float(r["frame_age_ms"]) for r in rows if r["frame_age_ms"]],
            )
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
                steps = np.arange(len(rows))
                hold_steps = np.array([int(r["step"]) for r in rows if r["hold_reason"]], dtype=int)
                for index, (name, ax) in enumerate(zip(JOINT_NAMES, axes.ravel())):
                    ax.plot(steps, measured_physical[:, index], label="measured", lw=1.5)
                    ax.plot(steps, policy_physical[:, index], label="policy", lw=1, alpha=0.7)
                    ax.plot(steps, sent[:, index], label="sent", lw=1, ls="--", alpha=0.8)
                    if name == "shoulder_lift":
                        ax.axhline(run["bench"]["shoulder_floor"], color="red", lw=1, ls=":", label="floor")
                    if len(hold_steps):
                        ax.scatter(hold_steps, measured_physical[hold_steps, index], s=8, color="red", zorder=3, label="hold")
                    for b in summary["boundaries"]:
                        ax.axvline(b, color="grey", lw=0.5, alpha=0.5)
                    ax.set_title(name); ax.grid(alpha=0.3)
                axes[0, 0].legend(loc="best", fontsize=8)
                fig.suptitle(f"{run_dir.name}: {summary['status']} actions={summary['actions']} holds={summary['hold_frames']} abort={summary['aborted_reason']}")
                fig.tight_layout()
                fig.savefig(out / "joints.png", dpi=110)
                plt.close(fig)
                summary["joints_png"] = str(out / "joints.png")
            except ImportError:
                summary["joints_png"] = None
    video = run_dir / "camera.mp4"
    if video.exists():
        import cv2
        from PIL import Image
        capture = cv2.VideoCapture(str(video))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        picks = sorted(set(int(round(i)) for i in np.linspace(0, max(count - 1, 0), num=min(video_samples, max(count, 1)))))
        tiles = []
        for index in picks:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)[:, :, ::-1]
            tiles.append((index, small))
        capture.release()
        if tiles:
            columns = 4
            rows_n = (len(tiles) + columns - 1) // columns
            sheet = Image.new("RGB", (columns * 320, rows_n * 196), (20, 20, 20))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(sheet)
            for k, (index, tile) in enumerate(tiles):
                x, y = (k % columns) * 320, (k // columns) * 196
                sheet.paste(Image.fromarray(np.ascontiguousarray(tile)), (x, y + 16))
                draw.text((x + 4, y + 2), f"frame {index} (~{index / 30:.1f} s)", fill=(255, 255, 255))
            sheet.save(out / "video_sheet.png")
            summary["video_sheet_png"] = str(out / "video_sheet.png")
            summary["video_frames"] = count
    boundaries = sorted((run_dir / "boundaries").glob("step_*.obs.png")) if (run_dir / "boundaries").exists() else []
    summary["boundary_observations"] = [str(p) for p in boundaries]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--video-samples", type=int, default=16)
    args = p.parse_args(argv)
    summary = summarize(args.run_dir, video_samples=args.video_samples)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("boundary_observations",)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
