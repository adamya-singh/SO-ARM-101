"""The project's W&B convention in code (notes/wandb-standard.md, 2026-10-04).

Every run logs to one project under numbered sections, so the decisive charts sort first in the run
workspace: ``01_outcome`` (closed-loop success, safety, closure mm), ``02_generalisation`` (held-out vs
train), ``03_training`` (loss, RL statistics, the step axes), ``04_throughput`` and ``05_phases``.
``Tracker`` enforces it: a key outside the five sections is refused (raised in rehearsals and tests,
dropped with a message in real runs), a ``summary=`` is accepted only for a key that is later logged,
and a W&B failure never stops a run.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT = "so-arm101-v2-scaling"
SECTIONS = ("01_outcome", "02_generalisation", "03_training", "04_throughput", "05_phases")
STEP = "03_training/step"
EPISODES = "03_training/episodes"
KEY = re.compile(r"^(%s)/[A-Za-z0-9_.\-/=]+$" % "|".join(SECTIONS))
JOB_TYPES = ("finetune", "rl", "eval", "rerun", "watcher", "probe")

# Reference levels every placement number is read against (tranche_analysis_20260918, pi05_lora16_10k_20261002).
REFERENCES = dict(
    cnn4800_heldout_start_90=3.6e-4,
    working_policy_start_90=2.5e-6,
    pose_threshold=2.5e-4,
    cnn4800_successes=6,
    pi05_ft_successes_exec90=12,
    pi05_ft_successes_exec30=15,
    cnn4800_closure_lateral_median_mm=5.1,
    working_policy_closure_lateral_median_mm=1.7,
    grasp_tolerance_mm=8.0,
)
# Constant lines logged next to a held-out curve so every chart shows the levels it must be read against.
REFERENCE_LINES = {
    "02_generalisation/ref_cnn4800_heldout_start_90": REFERENCES["cnn4800_heldout_start_90"],
    "02_generalisation/ref_working_policy_start_90": REFERENCES["working_policy_start_90"],
    "02_generalisation/ref_pose_threshold_2.5e-4": REFERENCES["pose_threshold"],
}


def check_key(key: str) -> None:
    if not KEY.match(key):
        raise ValueError(f"W&B key {key!r} is not under a numbered section {SECTIONS} (notes/wandb-standard.md)")


def run_name(family: str, *parts: Any, scene_hash: str | None = None, seed: int | None = None, rehearsal: bool = False) -> str:
    """``<family>-<key hparams>-<scene8>[-seed N]``; rehearsals are ``<family>-rehearsal``."""
    if rehearsal:
        return f"{family}-rehearsal"
    name = "-".join([family, *[str(p) for p in parts if p not in (None, "")]])
    if scene_hash:
        name += f"-{scene_hash[:8]}"
    if seed is not None:
        name += f"-seed{seed}"
    return name


class Tracker:
    """A W&B run that follows the standard. ``wandb`` may be None (tests): every call is then recorded only."""

    def __init__(self, run: Any, *, strict: bool, wandb_module: Any = None) -> None:
        self.run, self.strict, self.wandb = run, bool(strict), wandb_module
        self.logged: set[str] = set()
        self.summaries: dict[str, str] = {}
        self.records: list[dict[str, Any]] = []   # what was logged, for tests and offline checks

    @property
    def id(self) -> str | None:
        return getattr(self.run, "id", None)

    @property
    def url(self) -> str | None:
        return getattr(self.run, "url", None)

    def _accept(self, key: str) -> bool:
        try:
            check_key(key)
            return True
        except ValueError as exc:
            if self.strict:
                raise
            print(f"wandb key dropped: {exc}", flush=True)
            return False

    def define(self, pattern: str, *, step_metric: str | None = STEP, summary: str | None = None) -> None:
        base = pattern.rstrip("*").rstrip("/")
        if pattern.endswith("*"):
            if base not in SECTIONS and not any(base.startswith(s + "/") for s in SECTIONS):
                raise ValueError(f"glob {pattern!r} is outside the numbered sections")
        else:
            check_key(pattern)
        if summary is not None:
            if pattern.endswith("*"):
                raise ValueError("summary= needs an exact key, not a glob")
            self.summaries[pattern] = summary
        if self.run is not None:
            try:
                kwargs = {} if step_metric is None else {"step_metric": step_metric}
                if summary is not None:
                    kwargs["summary"] = summary
                self.run.define_metric(pattern, **kwargs)
            except Exception as exc:
                print(f"wandb define_metric failed: {exc}", flush=True)

    def log(self, payload: Mapping[str, Any]) -> None:
        clean = {k: v for k, v in payload.items() if self._accept(k)}
        self.logged.update(clean)
        self.records.append({k: v for k, v in clean.items() if isinstance(v, (int, float, str, bool)) or v is None})
        if self.run is None or not clean:
            return
        try:
            self.run.log(clean)
        except Exception as exc:   # tracking must never stop the run
            print(f"wandb log failed: {exc}", flush=True)

    def summary(self, payload: Mapping[str, Any]) -> None:
        clean = {k: v for k, v in payload.items() if self._accept(k)}
        if self.run is None:
            return
        try:
            self.run.summary.update(clean)
        except Exception as exc:
            print(f"wandb summary failed: {exc}", flush=True)

    def config_update(self, payload: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        try:
            self.run.config.update(dict(payload), allow_val_change=True)
        except Exception as exc:
            print(f"wandb config update failed: {exc}", flush=True)

    def phase(self, name: str, phases: tuple[str, ...], step: int, axis: str = STEP) -> None:
        self.log({axis: step, "05_phases/phase": phases.index(name), **{f"05_phases/{p}": int(p == name) for p in phases}})

    def references(self, step: int, axis: str = STEP) -> None:
        self.log({axis: step, **REFERENCE_LINES})

    def table(self, key: str, columns: list[str], rows: Iterable[Iterable[Any]]) -> None:
        if self.wandb is None or not self._accept(key):
            return
        try:
            table = self.wandb.Table(columns=columns)
            for row in rows:
                table.add_data(*row)
            self.log({key: table})
        except Exception as exc:
            print(f"wandb table failed: {exc}", flush=True)

    def image(self, key: str, path: str | Path) -> None:
        if self.wandb is None or not self._accept(key):
            return
        try:
            self.log({key: self.wandb.Image(str(path))})
        except Exception as exc:
            print(f"wandb image failed: {exc}", flush=True)

    def unlogged_summaries(self) -> list[str]:
        """Keys given a ``summary=`` that were never logged (the pi05 per_pose_median slip of 2026-10-02)."""
        return sorted(k for k in self.summaries if k not in self.logged)

    def finish(self) -> None:
        missing = self.unlogged_summaries()
        if missing:
            message = f"summary metrics never logged: {missing}"
            if self.strict:
                raise ValueError(message)
            print(message, flush=True)
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as exc:
                print(f"wandb finish failed: {exc}", flush=True)


def init_run(*, family: str, name: str, config: Mapping[str, Any], tags: Iterable[str] = (), job_type: str,
             rehearsal: bool, run_id: str | None = None, axes: Iterable[str] = (STEP,),
             default_axis: str = STEP, directory: str | Path | None = None, enabled: bool = True) -> Tracker:
    """Start (or resume, ``run_id``) a run in ``PROJECT``, group ``family``; numbered sections plotted on ``default_axis``.

    Rehearsals run offline and are marked ``strict`` (an unnumbered key raises). ``enabled=False`` gives a
    recording-only tracker (unit tests, tools run without network).
    """
    if job_type not in JOB_TYPES:
        raise ValueError(f"job_type must be one of {JOB_TYPES}")
    if rehearsal and not name.endswith("rehearsal"):
        raise ValueError("rehearsal runs are named '*-rehearsal' so they are never quoted")
    wandb = None
    run = None
    if enabled:
        import wandb as _wandb
        wandb = _wandb
        run = wandb.init(project=PROJECT, group=family, job_type=job_type, name=name, id=run_id, resume="allow",
                         mode=("offline" if rehearsal else "online"), config=dict(config), tags=sorted(set(tags) | {family}),
                         dir=None if directory is None else str(directory), settings=wandb.Settings(init_timeout=120))
    tracker = Tracker(run, strict=rehearsal or not enabled, wandb_module=wandb)
    for axis in axes:
        check_key(axis)
        tracker.define(axis, step_metric=None)
    for section in SECTIONS:
        tracker.define(f"{section}/*", step_metric=default_axis)
    return tracker


def persist_run_id(path: str | Path, tracker: Tracker) -> None:
    """``wandb.json`` beside the run's outputs: the id resumes the run and the status page links the URL."""
    path = Path(path)
    payload = dict(id=tracker.id, url=tracker.url, project=PROJECT, updated_at=time.time())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def stored_run_id(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("id")


__all__ = ["EPISODES", "JOB_TYPES", "PROJECT", "REFERENCES", "REFERENCE_LINES", "SECTIONS", "STEP", "Tracker",
           "check_key", "init_run", "persist_run_id", "run_name", "stored_run_id"]
