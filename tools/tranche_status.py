"""Read-only status of a tranche for its live page (retry tranche, 2026-10-04).

Reads the tranche manifest (tools/tranche_20261004.json), the task-spooler queue (``tsp -l``) and each
experiment's run directory (progress.json, wandb.json, result.json, heldout_curve.jsonl, metrics.jsonl, the
re-run ledger) and writes ``<root>/status.json``: one row per experiment with its state, phase, progress,
ETA, latest numbers and links. Changes nothing. The status page's rows are written from this file.

  python tools/tranche_status.py [--manifest tools/tranche_20261004.json] [--print]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tools/tranche_20261004.json"
FT_SOCKET = "/tmp/tsp-retry-finetunes"
CPU_SOCKET = "/tmp/tsp-retry-cpu"


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def last_jsonl(path: Path) -> dict | None:
    try:
        lines = path.read_text().strip().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def epoch(value: Any) -> float | None:
    """Seconds since the epoch from a float or an ISO-8601 string (the re-run driver writes ISO)."""
    if value is None or isinstance(value, (int, float)):
        return value
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return None


def tsp_jobs() -> dict[str, dict[str, Any]]:
    """label -> newest job {id, state, exit}. ``tsp`` is shared with other projects: only labelled jobs count."""
    import os
    out = ""
    for socket in (None, FT_SOCKET, CPU_SOCKET):       # the main queue and the fine-tune queue that shares the GPU with RL
        env = dict(os.environ)
        if socket:
            env["TS_SOCKET"] = socket
        try:
            out += subprocess.run(["tsp", "-l"], capture_output=True, text=True, timeout=20, env=env).stdout
        except (OSError, subprocess.SubprocessError):
            pass
    jobs: dict[str, dict[str, Any]] = {}
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        labels = [m for m in re.findall(r"\[([^\]]+)\]", line) if not m.isdigit()]   # "[12]&& [label]" = depends on job 12
        if not labels:
            continue
        label = re.match(r"(.*)", labels[0])
        job = dict(id=int(parts[0]), state=parts[1])
        if parts[1] == "finished" and len(parts) > 3:
            try:
                job["exit"] = int(parts[3])
            except ValueError:
                pass
        if label.group(1) not in jobs or job["id"] > jobs[label.group(1)]["id"]:
            jobs[label.group(1)] = job
    return jobs


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    return value


def finetune_row(run: Path, row: dict) -> None:
    progress = read_json(run / "progress.json") or {}
    curve = last_jsonl(run / "heldout_curve.jsonl")
    result = read_json(run / "result.json")
    target = progress.get("steps_target")
    row["progress"] = dict(done=progress.get("step", 0) if progress.get("phase") not in ("score", "rollouts", "complete") else target,
                           target=target, unit="steps")
    metrics = {}
    if curve:
        metrics.update(heldout_start_90=_round(curve.get("heldout_start_90")), ratio_start_90=_round(curve.get("ratio_start_90"), 3),
                       poses_under_threshold=curve.get("poses_under_threshold"), curve_step=curve.get("step"))
    for key, r in ((result or {}).get("rollouts") or progress.get("rollouts") or {}).items():
        if key.isdigit():
            metrics[f"successes_exec{key}"] = r.get("successes")
            metrics[f"closure_median_mm_exec{key}"] = _round((r.get("closure") or {}).get("lateral_median_mm"), 3)
        else:
            metrics[f"successes_exec{key}"] = r.get("successes")
    row["metrics"] = metrics


def rl_row(run: Path, row: dict) -> None:
    progress = read_json(run / "progress.json") or {}
    last = last_jsonl(run / "metrics.jsonl")
    row["progress"] = dict(done=progress.get("episodes_done", 0), target=progress.get("episodes_target"), unit="episodes")
    metrics: dict[str, Any] = {}
    start = progress.get("start_monitor") or {}
    history = progress.get("monitor_history") or []
    if start:
        metrics["monitor_start_success_rate"] = _round(start.get("success_rate"), 3)
    if history:
        metrics["monitor_success_rate"] = _round(history[-1].get("success_rate"), 3)
        metrics["monitor_episodes"] = history[-1].get("episodes")
        row["monitor_curve"] = [dict(episodes=h.get("episodes"), success_rate=_round(h.get("success_rate"), 3)) for h in history[-40:]]
    if last:
        train = last.get("train", {})
        metrics.update(train_success_rate=_round(train.get("success_rate"), 3), train_pickup_rate=_round(train.get("pickup_rate"), 3),
                       return_mean=_round(train.get("return_mean"), 3), kl=_round((last.get("stats") or {}).get("kl"), 3))
    for key, r in (progress.get("heldout") or {}).items():
        metrics[f"successes_exec{key}"] = r.get("successes")
        metrics[f"closure_median_mm_exec{key}"] = _round((r.get("closure") or {}).get("lateral_median_mm"), 3)
    if progress.get("aborted"):
        row["aborted"] = progress["aborted"]
    row["metrics"] = metrics


RERUN_STEPS = {"03": 5, "04": 11, "05": 2, "06": 3, "07": 1}   # tools/rerun_entries_03_07.build_steps (entry 00 feeds 03, 04, 06)
RERUN_DONE = ("completed", "skipped_by_rule")


def rerun_row(run: Path, row: dict, entry: str) -> None:
    ledger = read_json(run / "ledger.json") or {}
    steps = list((ledger.get("steps") or {}).values())
    mine = [s for s in steps if str(s.get("entry", "")) == entry]
    done = [s for s in mine if s.get("status") in RERUN_DONE]
    failed = [s for s in mine if s.get("status") == "failed"]
    total = RERUN_STEPS.get(entry)
    row["progress"] = dict(done=len(done), target=total, unit="steps")
    comparison = read_json(run / "comparison.json") or {}
    summary = comparison.get(entry) or {}
    row["metrics"] = {k: summary[k] for k in ("verdict_3_9", "changed", "compared") if k in summary}
    if failed:
        row["state_hint"] = "failed"
    elif total and len(done) >= total:
        row["state_hint"] = "done"
    elif mine:
        row["state_hint"] = "running"
    else:
        row["state_hint"] = "waiting"


def status(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    root = ROOT / manifest["root"]
    jobs = tsp_jobs()
    rows = []
    now = time.time()
    for spec in manifest["experiments"]:
        run = root / spec["dir"]
        row: dict[str, Any] = {k: spec.get(k) for k in ("id", "order", "kind", "model", "method", "title", "question", "success_means",
                                                         "start_from", "depends_on", "env", "entry")}
        row = {k: v for k, v in row.items() if v is not None}
        job = jobs.get(spec["label"])
        row["tsp_job"] = None if job is None else job["id"]
        progress = read_json(run / "progress.json") or {}
        wandb = read_json(run / "wandb.json") or {}
        row["wandb_url"] = wandb.get("url") or progress.get("wandb_url")
        if spec["kind"] == "finetune":
            finetune_row(run, row)
        elif spec["kind"] == "rl":
            rl_row(run, row)
        else:
            rerun_row(run, row, spec["entry"])
        # state: the run's own record first, then the queue
        run_status = progress.get("status")
        state = "planned"
        if job is not None:
            state = {"queued": "queued", "running": "running", "finished": "done" if job.get("exit", 0) == 0 else "failed",
                     "skipped": "failed"}.get(job["state"], job["state"])
        if spec["kind"] == "rerun":
            hint = row.pop("state_hint", None)
            if hint == "failed":
                state = "failed"
            elif hint == "done":
                state = "done"
            elif state == "running" and hint == "waiting":
                state = "queued"          # the chain is running an earlier entry
        elif run_status in ("complete", "aborted", "failed"):
            state = {"complete": "done", "aborted": "aborted", "failed": "failed"}[run_status]
            if run_status == "failed" and job is not None and job["state"] == "running":
                state = "running"          # resumed after a failure
        elif run_status == "running" and (job is None or job["state"] == "running"):
            state = "running"
        row["state"] = state
        row["phase"] = progress.get("phase")
        updated = epoch(progress.get("updated_at"))
        row["updated_at"] = updated
        row["stale"] = bool(state == "running" and updated and now - updated > 1800)
        p = row.get("progress") or {}
        started = epoch(progress.get("started_at"))
        if state == "running" and p.get("done") and p.get("target") and started:
            elapsed = now - started
            row["eta_hours"] = round(elapsed / max(p["done"], 1) * max(p["target"] - p["done"], 0) / 3600, 2)
        if progress.get("error") and state == "failed":
            row["error"] = str(progress["error"])[:300]
        rows.append(row)
    queue = sorted((r for r in rows if r.get("tsp_job") is not None), key=lambda r: r["tsp_job"])
    meta = dict(tranche=manifest["tranche"], title=manifest["title"], subtitle=manifest["subtitle"], why=manifest["why"],
                baselines=manifest["baselines"], updated_at=now,
                counts={s: sum(r["state"] == s for r in rows) for s in ("planned", "queued", "running", "done", "failed", "aborted")},
                queue=[dict(id=r["id"], tsp_job=r["tsp_job"], state=r["state"]) for r in queue])
    stage = read_json(root / "stage.json")
    if stage:
        meta["stage"] = stage.get("text")
    budget = read_json(root / "budget.json")
    if budget:
        meta["budget"] = budget
    return dict(meta=meta, experiments=sorted(rows, key=lambda r: r["order"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--print", action="store_true")
    args = p.parse_args(argv)
    payload = status(args.manifest)
    root = ROOT / json.loads(args.manifest.read_text())["root"]
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=1, default=float))
    tmp.replace(root / "status.json")
    docs = root / "status_docs"          # one JSON file per page row (the page's db documents)
    docs.mkdir(exist_ok=True)
    (docs / "meta__tranche.json").write_text(json.dumps(payload["meta"], default=float))
    for row in payload["experiments"]:
        (docs / f"experiments__{row['id']}.json").write_text(json.dumps(row, default=float))
    if args.print:
        print(json.dumps(payload, indent=1, default=float))
    else:
        print(" ".join(f"{r['id']}={r['state']}" for r in payload["experiments"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
