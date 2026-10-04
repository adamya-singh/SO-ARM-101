"""Re-run research-log entries 03-07 under the pinned MuJoCo 3.9.0.

Entries 03 (recovery anchors), 04 (recovery weights), 05 (more inputs),
06 (DAgger) and 07 (action chunks) were produced on 2026-08-02/03 while a
stray ``pip --user`` mujoco 3.11.0 shadowed the env's 3.9.0.  Their sim
artifacts sit in
``artifacts/so_arm101_v2/quarantine_mujoco_3_11_0/oracle_distillation/`` and
were never re-run.  This driver replays the command chain each entry used,
reconstructed from the identities recorded in those quarantined reports (see
the ``source`` string on every step), into a separate artifacts root so
nothing collides with existing artifacts.

Every CLI used here takes ``--output-dir``; all of them write
content-addressed hash directories below it.  The driver therefore never
hardcodes a produced hash: it reads the path each CLI prints on stdout (with a
newest-file glob as the fallback) and records it in the ledger, and later
steps resolve ``Ref`` placeholders from those recorded outputs.

Inputs that were NOT quarantined and stay fixed (they predate the 3.11 window
or are torch-only):
- oracle manifest ``oracle/fixed_pick_place_v3/9164a76699186c34`` (content
  70fff5ec..., captured 2026-08-01 under 3.9.0), the manifest every
  quarantined report cites;
- the v3 preflight report (content 0330cc10..., 2026-08-01);
- the nominal baseline clone ``models/phase_state/d5f96d397bd9b915/model.pt``
  (sha 8ba87aae..., legacy numerics), the inducing/baseline checkpoint of
  entries 04 and 06.

Usage::

    PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=src ~/miniforge3/envs/lerobot/bin/python \
        tools/rerun_entries_03_07.py [--only 03 04] [--dry-run] [--legacy-numerics] [--wandb]

Outputs: ``<root>/ledger.json`` (argv, times, exit code, produced paths per
step, rewritten atomically after every step; completed steps are skipped on
resume), ``<root>/progress.json`` (status page), ``<root>/logs/<step>.log``
and ``<root>/oracle_distillation/...`` (the CLIs' own outputs).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "artifacts/so_arm101_v2/rerun_mujoco_3_9_0/entries_03_07"
DEFAULT_PYTHON = Path.home() / "miniforge3/envs/lerobot/bin/python"
ORIG = REPO / "artifacts/so_arm101_v2/oracle_distillation"
DEFAULT_ORACLE = ORIG / "oracle/fixed_pick_place_v3/9164a76699186c34/manifest.json"
DEFAULT_BASELINE = ORIG / "models/phase_state/d5f96d397bd9b915/model.pt"
DEFAULT_PREFLIGHT = REPO / "artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"
ENTRIES = ("00", "03", "04", "05", "06", "07")
ENTRY_TITLES = {
    "00": "shared prerequisites",
    "03": "recovery anchors",
    "04": "recovery weights",
    "05": "more inputs",
    "06": "dagger",
    "07": "action chunks",
}
WEIGHTS = ("0.10", "0.25", "0.50")
NUMERICS_ENV = "SO_ARM101_V2_NUMERICS"
SCHEMA = 1


@dataclass(frozen=True)
class Ref:
    """A placeholder for an output recorded by an earlier step."""

    step: str
    key: str = "report"

    def __str__(self) -> str:
        return f"<{self.step}:{self.key}>"


@dataclass
class Step:
    id: str
    entry: str
    module: str                      # "sim" or "learn"
    subcommand: str
    args: list[Any]
    source: str                      # which quarantined report / identity justifies this step
    accept: tuple[int, ...] = (0, 2)  # 2 = the CLI's negative verdict, still a completed step
    numerics_flag: bool = False      # accepts --legacy-numerics / --device
    workers_flag: bool = False
    fallback_glob: str | None = None  # relative to the output dir, newest file created by the step
    derive: dict[str, Callable[[Path], Path]] = field(default_factory=dict)
    when: tuple[str, str, str] | None = None  # (step, summary key, required value)

    def refs(self) -> list[Ref]:
        refs = [a for a in self.args if isinstance(a, Ref)]
        if self.when is not None:
            refs.append(Ref(self.when[0], "summary"))
        return refs


def _sibling_model(report: Path) -> Path:
    return report.parent / "model.pt"


def _correction_candidate_model(report: Path) -> Path:
    payload = json.loads(report.read_text(encoding="utf-8"))
    return Path(payload["candidate"]["training_report"]).parent / "model.pt"


def build_steps(*, out: Path, oracle: Path, baseline: Path, preflight: Path) -> list[Step]:
    """The reconstructed chains, in the order the originals ran."""
    o, b, p = str(oracle), str(baseline), str(preflight)
    od = ["--output-dir", str(out)]
    distill_common = [
        "--manifest", o, "--model-kind", "phase_state", "--seed", "101",
        "--max-steps", "30000", "--hidden-width", "256", "--lr-schedule", "decay_10k_20k",
    ]
    steps: list[Step] = [
        # ---- 00: shared prerequisites -------------------------------------------------
        Step(
            "00.prefix_reference", "00", "learn", "distill-oracle",
            ["--manifest", o, "--model-kind", "phase_state", "--seed", "101", "--max-steps", "10000",
             "--hidden-width", "256", "--lr-schedule", "fixed", *od],
            "re-trains models/phase_state/6fc677db0c3ae755 (content fde2b0ca), the fixed-LR 10k-step "
            "prefix reference every decay_10k_20k phase_state run of 03/04/06 cites; a prefix reference "
            "only validates runs of its own numerics regime, so it is rebuilt in the run's regime "
            "(a 10k-step run that never passes, exit 2, is the expected outcome)",
            numerics_flag=True, fallback_glob="models/phase_state/*/report.json",
            derive={"checkpoint": _sibling_model},
        ),
        Step(
            "00.baseline_eval", "00", "sim", "evaluate-clone",
            ["--scenario", "nominal", "--checkpoint", b, "--preflight-report", p, *od],
            "re-runs clone_evaluations/943cf536710e3d84 (content 5efae257, pre-quarantine 3.9, with video), "
            "the --baseline-evaluation of correction_gates/faf13e95 and correction_probes/63e2717b; "
            "doubles as a 3.9 regime sanity check (expect 0/3 pickup_incomplete, zero safety)",
            workers_flag=True, fallback_glob="clone_evaluations/*/policies/*/evaluation.json",
        ),
        # ---- 03: recovery anchors -----------------------------------------------------
        Step(
            "03.capture_recovery", "03", "sim", "capture-recovery", ["--oracle-manifest", o, *od],
            "recovery/0570ec8c0d37002f (8 anchors, source manifest 70fff5ec)",
            accept=(0,), fallback_glob="recovery/*/manifest.json",
        ),
        Step(
            "03.distill", "03", "learn", "distill-oracle",
            [*distill_common, "--prefix-parity-report", Ref("00.prefix_reference"),
             "--recovery-manifest", Ref("03.capture_recovery"), *od],
            "models/phase_state/c3ca76dc2c0fa42d (report cc791f98): phase_state, seed 101, width 256, "
            "30000 steps, decay_10k_20k, prefix reference fde2b0ca, recovery 0570ec8c, weight 1.0",
            numerics_flag=True, fallback_glob="models/phase_state/*/report.json",
            derive={"checkpoint": _sibling_model},
        ),
        Step(
            "03.evaluate_clone", "03", "sim", "evaluate-clone",
            ["--scenario", "all", "--no-video", "--checkpoint", Ref("03.distill", "checkpoint"),
             "--preflight-report", p, *od],
            "clone_evaluations/e67ba4433d3aa98c (suite fixed_pick_place_v3 = all 5 scenarios x 3, "
            "no videos, checkpoint 0d87c94e): the 0/15 headline",
            workers_flag=True, fallback_glob="clone_evaluations/*/policies/*/evaluation.json",
        ),
        Step(
            "03.recovery_starts", "03", "sim", "evaluate-recovery-starts",
            ["--oracle-manifest", o, "--recovery-manifest", Ref("03.capture_recovery"),
             "--checkpoint", Ref("03.distill", "checkpoint"), *od],
            "recovery_evaluations/4e5e4f1252d582ae (checkpoint 0d87c94e): 1/8 anchors",
            fallback_glob="recovery_evaluations/*/report.json",
        ),
        Step(
            "03.recovery_starts_baseline", "03", "sim", "evaluate-recovery-starts",
            ["--oracle-manifest", o, "--recovery-manifest", Ref("03.capture_recovery"),
             "--checkpoint", b, *od],
            "recovery_evaluations/b0e6e56a018c1d7f (baseline checkpoint 8ba87aae): 3/8 anchors",
            fallback_glob="recovery_evaluations/*/report.json",
        ),
        # ---- 04: recovery weights -----------------------------------------------------
        Step(
            "04.scan_w0.00", "04", "sim", "scan-clone-static",
            ["--oracle-manifest", o, "--recovery-manifest", Ref("03.capture_recovery"),
             "--checkpoint", b, *od],
            "static_command_scans/0255f8e093a6649f: weight-0 control = baseline 8ba87aae, samples/path 101",
            fallback_glob="static_command_scans/*/report.json",
        ),
        Step(
            "04.scan_w1.00", "04", "sim", "scan-clone-static",
            ["--oracle-manifest", o, "--recovery-manifest", Ref("03.capture_recovery"),
             "--checkpoint", Ref("03.distill", "checkpoint"), *od],
            "static_command_scans/f95b3aba27c4a12d: weight-1 control = entry-03 clone 0d87c94e",
            fallback_glob="static_command_scans/*/report.json",
        ),
    ]
    ablation = {
        "0.10": ("models/phase_state/f26b6c646a76823e (62d789ed)", "static_command_scans/460243a473cde400",
                 "clone_evaluations/80d1706cce5d9675"),
        "0.25": ("models/phase_state/5a402430f069faa7 (5d17b545, missed the fit bar)", None, None),
        "0.50": ("models/phase_state/f27127f973a1b048 (e228632c)", "static_command_scans/859c763c50c7b754",
                 "clone_evaluations/45084642c47d1200"),
    }
    for weight in WEIGHTS:
        model_src, scan_src, eval_src = ablation[weight]
        sid = f"04.distill_w{weight}"
        steps.append(Step(
            sid, "04", "learn", "distill-oracle",
            [*distill_common, "--prefix-parity-report", Ref("00.prefix_reference"),
             "--recovery-manifest", Ref("03.capture_recovery"), "--recovery-loss-weight", weight, *od],
            f"{model_src}: as 03.distill with recovery_loss_weight {weight} "
            "(recovery_weight_ablations/187b171bdf0fd39e candidate_weights_in_order)",
            numerics_flag=True, fallback_glob="models/phase_state/*/report.json",
            derive={"checkpoint": _sibling_model},
        ))
        rule = (sid, "passed", "true")
        rule_note = ("; only run when the offline fit passed (the ablation's unrun_by_gate rule; "
                     "0.25 missed it in the original)")
        steps.append(Step(
            f"04.scan_w{weight}", "04", "sim", "scan-clone-static",
            ["--oracle-manifest", o, "--recovery-manifest", Ref("03.capture_recovery"),
             "--checkpoint", Ref(sid, "checkpoint"), *od],
            (scan_src or "not run in the original") + rule_note,
            fallback_glob="static_command_scans/*/report.json", when=rule,
        ))
        steps.append(Step(
            f"04.eval_w{weight}", "04", "sim", "evaluate-clone",
            ["--scenario", "nominal", "--no-video", "--checkpoint", Ref(sid, "checkpoint"),
             "--preflight-report", p, *od],
            (f"{eval_src} (suite .nominal, 3 repeats, no videos)" if eval_src else "not run in the original")
            + rule_note,
            workers_flag=True, fallback_glob="clone_evaluations/*/policies/*/evaluation.json", when=rule,
        ))
    steps += [
        # ---- 05: more inputs ----------------------------------------------------------
        Step(
            "05.capture_observability", "05", "sim", "capture-observability",
            ["--oracle-manifest", o, "--recovery-manifest", Ref("03.capture_recovery"), *od],
            "observability/ec2d88b418e15186 (oracle 70fff5ec + recovery 24837909)",
            accept=(0,), fallback_glob="observability/*/manifest.json",
        ),
        Step(
            "05.observability_gate", "05", "sim", "run-observability-gate",
            ["--oracle-manifest", o, "--recovery-manifest", Ref("03.capture_recovery"),
             "--observability-manifest", Ref("05.capture_observability"),
             "--baseline-checkpoint", Ref("03.distill", "checkpoint"),
             "--baseline-evaluation", Ref("03.evaluate_clone"), "--preflight-report", p, *od],
            "observability_gates/3cdb2c3d2c467a5b: baseline checkpoint 0d87c94e (entry-03 clone), baseline "
            "evaluation 4021fe84 (= clone_evaluations/e67ba443), trains phase_dynamics / _contact / "
            "_contact_history2 internally, nominal evals with video",
            workers_flag=True, fallback_glob="observability_gates/*/report.json",
        ),
        # ---- 06: dagger ---------------------------------------------------------------
        Step(
            "06.capture_corrections", "06", "sim", "capture-corrections",
            ["--oracle-manifest", o, "--checkpoint", b, *od],
            "corrections/7feed472071fd725 (inducing checkpoint 8ba87aae, 11 sites, 4818 rows)",
            accept=(0,), fallback_glob="corrections/*/manifest.json",
        ),
        Step(
            "06.correction_gate", "06", "sim", "run-correction-gate",
            ["--oracle-manifest", o, "--correction-manifest", Ref("06.capture_corrections"),
             "--baseline-checkpoint", b, "--baseline-evaluation", Ref("00.baseline_eval"),
             "--prefix-parity-report", Ref("00.prefix_reference"), "--preflight-report", p, *od],
            "correction_gates/faf13e95b8caa2e8: baseline 8ba87aae, baseline evaluation 5efae257, prefix "
            "parity fde2b0ca; trains phase_state+corrections internally (30000 steps, decay_10k_20k)",
            workers_flag=True, fallback_glob="correction_gates/*/report.json",
            derive={"candidate_checkpoint": _correction_candidate_model},
        ),
        Step(
            "06.correction_probe", "06", "sim", "run-correction-probe",
            ["--checkpoint", Ref("06.correction_gate", "candidate_checkpoint"),
             "--baseline-evaluation", Ref("00.baseline_eval"), "--preflight-report", p, *od],
            "correction_probes/63e2717b530a2cf0: the gate's blocked candidate (120cf691 = "
            "models/phase_state/9e1bb18f724aa989) vs baseline evaluation 5efae257, with video",
            workers_flag=True, fallback_glob="correction_probes/*/report.json",
        ),
        # ---- 07: action chunks --------------------------------------------------------
        Step(
            "07.chunked_gate", "07", "sim", "run-chunked-gate",
            ["--oracle-manifest", o, "--preflight-report", p, *od],
            "chunked_gates/591f0686e94d27fd: H = 10, 30, 90 on the 450 nominal rows, seed 101, width 256, "
            "lr 1e-3, 30000 steps (fixed in code), nominal evals with video",
            numerics_flag=True, workers_flag=True, fallback_glob="chunked_gates/*/report.json",
        ),
    ]
    return steps


# ---------------------------------------------------------------------------------------------
# ledger / progress


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_ledger(root: Path) -> dict[str, Any]:
    path = root / "ledger.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": SCHEMA, "created_at": _now(), "root": str(root), "steps": {}}


DONE = ("completed", "skipped_by_rule")


def write_progress(root: Path, *, status: str, phase: str, step: int, steps_total: int,
                   current: str | None = None) -> None:
    atomic_write_json(root / "progress.json", {
        "status": status, "phase": phase, "step": step, "steps_total": steps_total,
        "updated_at": _now(), "current_step": current,
    })


# ---------------------------------------------------------------------------------------------
# command construction


def select_steps(steps: Sequence[Step], only: Iterable[str] | None, ledger: Mapping[str, Any]) -> list[Step]:
    """The steps of the requested entries plus every not-yet-completed step they depend on."""
    by_id = {s.id: s for s in steps}
    if not only:
        return [s for s in steps if s.entry in ENTRIES]
    wanted = {s.id for s in steps if s.entry in set(only)}
    frontier = list(wanted)
    while frontier:
        for ref in by_id[frontier.pop()].refs():
            if ref.step not in wanted:
                wanted.add(ref.step)
                frontier.append(ref.step)
    return [s for s in steps if s.id in wanted]


def resolve(arg: Any, ledger: Mapping[str, Any], *, strict: bool) -> str:
    if not isinstance(arg, Ref):
        return str(arg)
    record = ledger.get("steps", {}).get(arg.step, {})
    value = record.get("outputs", {}).get(arg.key)
    if value is None:
        if strict:
            raise RuntimeError(f"step needs {arg} but {arg.step} has not recorded it (status "
                               f"{record.get('status', 'not run')})")
        return str(arg)
    return str(value)


def argv_for(step: Step, ledger: Mapping[str, Any], *, python: str, legacy_numerics: bool,
             workers: int | None, strict: bool) -> list[str]:
    module = "so_arm101_v2.simulation.cli" if step.module == "sim" else "so_arm101_v2.learning.cli"
    argv = [python, "-m", module, step.subcommand, *[resolve(a, ledger, strict=strict) for a in step.args]]
    if legacy_numerics and step.numerics_flag:
        argv.append("--legacy-numerics")
    if workers is not None and step.workers_flag:
        argv += ["--workers", str(workers)]
    return argv


def child_env(legacy_numerics: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"PYTHONNOUSERSITE": "1", "MUJOCO_GL": "egl", "PYTHONPATH": str(REPO / "src")})
    # Gates that train internally (observability, correction) take no numerics flag and follow the
    # library default, so the regime is pinned through the env for every child.
    if legacy_numerics:
        env[NUMERICS_ENV] = "legacy"
    else:
        env.pop(NUMERICS_ENV, None)
    return env


# ---------------------------------------------------------------------------------------------
# running


Runner = Callable[[list[str], Mapping[str, str], Path], tuple[int, str]]


def subprocess_runner(argv: list[str], env: Mapping[str, str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(argv) + "\n")
        log.flush()
        proc = subprocess.Popen(argv, cwd=REPO, env=dict(env), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            log.write(line)
            log.flush()
            sys.stdout.write("    " + line)
        code = proc.wait()
    return code, "".join(lines)


_KV = re.compile(r"(?:^|\s)([A-Za-z_]+)=(\S+)")


def parse_stdout(stdout: str) -> tuple[list[str], dict[str, str]]:
    paths, summary = [], {}
    for line in stdout.splitlines():
        text = line.strip()
        if text.endswith(".json") and " " not in text:
            candidate = Path(text) if Path(text).is_absolute() else REPO / text
            if candidate.is_file():
                paths.append(str(candidate.resolve()))
        for key, value in _KV.findall(text):
            summary[key] = value
    return paths, summary


def discover_outputs(step: Step, stdout: str, out: Path, started: float) -> tuple[dict[str, Any], dict[str, str]]:
    paths, summary = parse_stdout(stdout)
    report: str | None = paths[0] if paths else None
    discovered_by = "stdout"
    if report is None and step.fallback_glob:
        fresh = [p for p in glob.glob(str(out / step.fallback_glob)) if os.path.getmtime(p) >= started - 1]
        if fresh:
            report = max(fresh, key=os.path.getmtime)
            discovered_by = "newest_glob"
    outputs: dict[str, Any] = {"report": report, "printed_paths": paths, "discovered_by": discovered_by}
    if report is not None:
        for key, fn in step.derive.items():
            try:
                derived = fn(Path(report))
                outputs[key] = str(derived) if derived.exists() else None
            except Exception as exc:  # recorded, the dependent step then fails loudly
                outputs[key] = None
                outputs[f"{key}_error"] = repr(exc)
    return outputs, summary


def run_chain(steps: Sequence[Step], *, root: Path, out: Path, python: str, legacy_numerics: bool,
              workers: int | None, runner: Runner, tracker: Any = None,
              on_entry_done: Callable[[str, dict], None] | None = None) -> int:
    ledger = load_ledger(root)
    ledger.update({
        "numerics": "legacy" if legacy_numerics else "default",
        "output_dir": str(out),
        "python": python,
    })
    ledger.setdefault("steps", {})
    total = len(steps)
    env = child_env(legacy_numerics)

    def done_count() -> int:
        return sum(1 for s in steps if ledger["steps"].get(s.id, {}).get("status") in DONE)

    atomic_write_json(root / "ledger.json", ledger)
    for index, step in enumerate(steps):
        phase = f"{step.entry} {ENTRY_TITLES[step.entry]}"
        record = ledger["steps"].get(step.id, {})
        if record.get("status") in DONE:
            print(f"[skip] {step.id} ({record['status']} {record.get('ended_at', '')})")
            continue
        if step.when is not None:
            gate_step, key, required = step.when
            gate = ledger["steps"].get(gate_step, {})
            if gate.get("summary", {}).get(key) != required:
                ledger["steps"][step.id] = {
                    "entry": step.entry, "status": "skipped_by_rule", "source": step.source,
                    "reason": f"{gate_step} {key}={gate.get('summary', {}).get(key)!r}, needs {required!r}",
                    "ended_at": _now(),
                }
                atomic_write_json(root / "ledger.json", ledger)
                print(f"[rule] {step.id} skipped: {ledger['steps'][step.id]['reason']}")
                continue
        write_progress(root, status="running", phase=phase, step=done_count(), steps_total=total, current=step.id)
        try:
            argv = argv_for(step, ledger, python=python, legacy_numerics=legacy_numerics,
                            workers=workers, strict=True)
        except RuntimeError as exc:
            ledger["steps"][step.id] = {"entry": step.entry, "status": "failed", "error": str(exc),
                                        "ended_at": _now()}
            atomic_write_json(root / "ledger.json", ledger)
            write_progress(root, status="failed", phase=phase, step=done_count(), steps_total=total, current=step.id)
            print(f"[fail] {step.id}: {exc}", file=sys.stderr)
            return 1
        print(f"[run ] {step.id}: {' '.join(argv)}", flush=True)
        started = time.time()
        started_at = _now()
        ledger["steps"][step.id] = {"entry": step.entry, "status": "running", "argv": argv,
                                    "started_at": started_at, "source": step.source}
        atomic_write_json(root / "ledger.json", ledger)
        code, stdout = runner(argv, env, root / "logs" / f"{step.id}.log")
        outputs, summary = discover_outputs(step, stdout, out, started)
        ok = code in step.accept and outputs.get("report") is not None and all(
            outputs.get(k) is not None for k in step.derive)
        ledger["steps"][step.id] = {
            "entry": step.entry,
            "subcommand": step.subcommand,
            "status": "completed" if ok else "failed",
            "argv": argv,
            "started_at": started_at,
            "ended_at": _now(),
            "wall_seconds": round(time.time() - started, 3),
            "exit_code": code,
            "verdict": "pass" if code == 0 else ("fail" if code == 2 else "error"),
            "outputs": outputs,
            "summary": summary,
            "source": step.source,
            "log": str(root / "logs" / f"{step.id}.log"),
        }
        atomic_write_json(root / "ledger.json", ledger)
        if tracker is not None:
            _log_step(tracker, steps, step, index, done_count(), total)
        if not ok:
            write_progress(root, status="failed", phase=phase, step=done_count(), steps_total=total, current=step.id)
            print(f"[fail] {step.id}: exit {code}, report {outputs.get('report')}", file=sys.stderr)
            return 1
        last_of_entry = all(s.entry != step.entry for s in steps[index + 1:])
        if last_of_entry and on_entry_done is not None:
            on_entry_done(step.entry, ledger)
    write_progress(root, status="completed", phase="done", step=done_count(), steps_total=total)
    return 0


# ---------------------------------------------------------------------------------------------
# W&B (optional)


def _log_step(tracker: Any, steps: Sequence[Step], step: Step, index: int, done: int, total: int) -> None:
    from so_arm101_v2 import tracking

    phases = tuple(f"entry{e}" for e in ENTRIES)
    tracker.phase(f"entry{step.entry}", phases, index + 1)
    tracker.log({tracking.STEP: index + 1, "05_phases/steps_done": done, "05_phases/steps_total": total})


def _wandb_tracker(root: Path, config: Mapping[str, Any]) -> Any:
    from so_arm101_v2 import tracking

    try:
        tracker = tracking.init_run(
            family="rerun-entries-03-07", name="rerun-entries-03-07", config=dict(config),
            tags=("rerun", "mujoco-3.9.0", "quarantine-3.11"), job_type="rerun", rehearsal=False,
            run_id=tracking.stored_run_id(root / "wandb.json"), directory=root,
        )
        tracking.persist_run_id(root / "wandb.json", tracker)
        return tracker
    except Exception as exc:  # W&B must never stop the chain
        print(f"wandb init failed, continuing without it: {exc}", flush=True)
        return None


def _entry_headlines(tracker: Any, root: Path, quarantine: Path) -> Callable[[str, dict], None]:
    def log(entry: str, ledger: dict) -> None:
        if entry == "00":
            return
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import compare_quarantine

            numbers = compare_quarantine.headline_numbers(entry, ledger, quarantine)
            payload = {f"01_outcome/entry{entry}_{k}": v for k, v in numbers.items()}
            if payload:
                tracker.log(payload)
                tracker.summary(payload)
        except Exception as exc:
            print(f"wandb headline logging failed: {exc}", flush=True)
    return log


# ---------------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="separate artifacts root; CLI outputs go to <root>/oracle_distillation")
    parser.add_argument("--only", nargs="+", choices=[e for e in ENTRIES if e != "00"],
                        help="run only these entries (plus any unfinished steps they depend on)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    parser.add_argument("--legacy-numerics", action="store_true",
                        help="the originals' cpu-eager training lane (pre-2026-08-06); default keeps the "
                             "library default regime")
    parser.add_argument("--workers", type=int, default=None, help="passthrough to rollout steps")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--oracle-manifest", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--preflight-report", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--quarantine-root", type=Path,
                        default=REPO / "artifacts/so_arm101_v2/quarantine_mujoco_3_11_0/oracle_distillation")
    parser.add_argument("--wandb", action="store_true",
                        help="one W&B run (project so-arm101-v2-scaling, group rerun-entries-03-07)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, runner: Runner | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    out = root / "oracle_distillation"
    steps = build_steps(out=out, oracle=args.oracle_manifest.resolve(),
                        baseline=args.baseline_checkpoint.resolve(), preflight=args.preflight_report.resolve())
    ledger = load_ledger(root)
    selected = select_steps(steps, args.only, ledger)
    if args.dry_run:
        print(f"# root {root}  numerics {'legacy' if args.legacy_numerics else 'default'}  "
              f"steps {len(selected)}")
        current = None
        for step in selected:
            if step.entry != current:
                current = step.entry
                print(f"\n## entry {step.entry}: {ENTRY_TITLES[step.entry]}")
            status = ledger.get("steps", {}).get(step.id, {}).get("status", "pending")
            argv_ = argv_for(step, ledger, python=args.python, legacy_numerics=args.legacy_numerics,
                             workers=args.workers, strict=False)
            when = f"  [only if {step.when[0]} {step.when[1]}={step.when[2]}]" if step.when else ""
            print(f"# {step.id} ({status}){when}\n#   source: {step.source}")
            print(" ".join(argv_))
        return 0
    for required in (args.oracle_manifest, args.baseline_checkpoint, args.preflight_report):
        if not required.is_file():
            print(f"error: missing input {required}", file=sys.stderr)
            return 1
    root.mkdir(parents=True, exist_ok=True)
    tracker = None
    on_entry_done = None
    if args.wandb:
        tracker = _wandb_tracker(root, {
            "tool": "tools/rerun_entries_03_07.py", "root": str(root),
            "numerics": "legacy" if args.legacy_numerics else "default",
            "oracle_manifest": str(args.oracle_manifest), "baseline_checkpoint": str(args.baseline_checkpoint),
            "preflight_report": str(args.preflight_report), "entries": args.only or list(ENTRIES[1:]),
            "steps": [s.id for s in selected], "mujoco_expected": "3.9.0",
        })
        if tracker is not None:
            on_entry_done = _entry_headlines(tracker, root, args.quarantine_root)
    try:
        return run_chain(selected, root=root, out=out, python=args.python, legacy_numerics=args.legacy_numerics,
                         workers=args.workers, runner=runner or subprocess_runner, tracker=tracker,
                         on_entry_done=on_entry_done)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    raise SystemExit(main())
