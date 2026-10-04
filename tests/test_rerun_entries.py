"""tools/rerun_entries_03_07.py: dry-run coverage and ledger resume, with a fake runner (no simulation)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from conftest import REPOSITORY_ROOT

sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
import compare_quarantine  # noqa: E402
import rerun_entries_03_07 as tool  # noqa: E402


class FakeRunner:
    """Writes a fake report (+ model.pt) per step and prints it like the CLIs do."""

    serial = 0

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[list[str]] = []

    def __call__(self, argv, env, log_path):
        assert env["PYTHONNOUSERSITE"] == "1" and env["MUJOCO_GL"] == "egl"
        self.calls.append(list(argv))
        FakeRunner.serial += 1
        directory = self.root / "oracle_distillation" / "fake" / str(FakeRunner.serial)
        directory.mkdir(parents=True)
        report = directory / "report.json"
        report.write_text(json.dumps({"candidate": {"training_report": str(report)}}))
        (directory / "model.pt").write_bytes(b"")
        passed = "false" if "0.25" in argv else "true"
        return (0 if passed == "true" else 2), f"{report}\npassed={passed} steps=1\n"


def subcommands(calls):
    return [call[3] for call in calls]


def test_dry_run_lists_commands_for_all_five_entries(tmp_path, capsys):
    assert tool.main(["--dry-run", "--root", str(tmp_path / "root")]) == 0
    out = capsys.readouterr().out
    commands = [line for line in out.splitlines() if " -m so_arm101_v2." in line and not line.startswith("#")]
    assert len(commands) == 24
    for entry in ("03", "04", "05", "06", "07"):
        assert f"## entry {entry}:" in out
    assert "--legacy-numerics" not in out
    assert not (tmp_path / "root").exists()  # a dry run writes nothing
    assert tool.main(["--dry-run", "--legacy-numerics", "--root", str(tmp_path / "root")]) == 0
    assert "--legacy-numerics" in capsys.readouterr().out


def test_ledger_resume_skips_completed_steps(tmp_path):
    root = tmp_path / "root"
    first = FakeRunner(root)
    assert tool.main(["--root", str(root)], runner=first) == 0
    ledger = json.loads((root / "ledger.json").read_text())
    steps = ledger["steps"]
    # 0.25 misses the fit bar, so its scan and evaluation are skipped by the rule (as in the original)
    assert steps["04.scan_w0.25"]["status"] == "skipped_by_rule"
    assert steps["04.eval_w0.25"]["status"] == "skipped_by_rule"
    assert len(first.calls) == 22
    assert all(steps[s]["status"] in tool.DONE for s in steps)
    # refs resolve to recorded outputs, never to placeholders
    probe = steps["06.correction_probe"]["argv"]
    assert probe[probe.index("--checkpoint") + 1].endswith("model.pt")
    assert not any(arg.startswith("<") for record in steps.values() for arg in record.get("argv", []))
    progress = json.loads((root / "progress.json").read_text())
    assert progress["status"] == "completed" and progress["step"] == progress["steps_total"] == 24
    assert set(progress) >= {"status", "phase", "step", "steps_total", "updated_at"}

    # everything recorded -> a resume runs nothing
    again = FakeRunner(root)
    assert tool.main(["--root", str(root)], runner=again) == 0
    assert again.calls == []

    # a failed step re-runs on resume, the completed ones around it do not
    ledger["steps"]["05.observability_gate"]["status"] = "failed"
    (root / "ledger.json").write_text(json.dumps(ledger))
    resume = FakeRunner(root)
    assert tool.main(["--root", str(root), "--only", "05"], runner=resume) == 0
    assert subcommands(resume.calls) == ["run-observability-gate"]


def test_only_pulls_in_unfinished_dependencies_and_stops_on_failure(tmp_path):
    root = tmp_path / "root"
    runner = FakeRunner(root)
    assert tool.main(["--root", str(root), "--only", "07"], runner=runner) == 0
    assert subcommands(runner.calls) == ["run-chunked-gate"]

    runner = FakeRunner(root)
    assert tool.main(["--root", str(root), "--only", "05"], runner=runner) == 0
    # 05 needs the entry-03 recovery capture, clone and evaluation, and through them the prefix reference
    assert subcommands(runner.calls) == [
        "distill-oracle", "capture-recovery", "distill-oracle", "evaluate-clone",
        "capture-observability", "run-observability-gate",
    ]

    def broken(argv, env, log_path):
        return 1, "error: boom\n"

    assert tool.main(["--root", str(root), "--only", "06"], runner=broken) == 1
    ledger = json.loads((root / "ledger.json").read_text())
    assert ledger["steps"]["00.baseline_eval"]["status"] == "failed"
    assert "06.capture_corrections" not in ledger["steps"]
    assert json.loads((root / "progress.json").read_text())["status"] == "failed"


def test_comparison_handles_missing_reports(tmp_path):
    ledger = {"steps": {"03.evaluate_clone": {"status": "completed", "outputs": {"report": None}}}}
    rows = compare_quarantine.rows_for(ledger, quarantine=tmp_path, original=tmp_path)
    assert rows and all(new == "n/a" for *_, new in rows)
    text = compare_quarantine.render(rows, tmp_path)
    assert "| entry | metric | 3.11 value | 3.9 value |" in text
