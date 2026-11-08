"""Rebase the RL scenario pool (rl_suites.json) onto an SO-101 scene's joint conventions (2026-11-08).

The retry tranche's RL suites (train seeds 30-37, monitor seed 21) were generated on the SO-100 and store the bench
reset pose as SO-100 joint angles. Every scenario starts at the bench reset pose, so on an SO-101 scene each suite is
rebased exactly as capture_placements --heldout-suite rebases the held-out suite (tools/capture_placements.rebase_suite:
the stored pose is checked to BE the physical reset pose, then rewritten in the scene's angles). Placements, appearance
and order are unchanged. Writes ``<output-dir>/rl_suites.json`` with the same fields, pointing at the rebased suites.

  PYTHONNOUSERSITE=1 ~/miniforge3/envs/lerobot/bin/python tools/rebase_rl_suites.py --model <so101 scene.xml> \\
      --rl-suites <retry_tranche_20261004/rl_suites/rl_suites.json> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--rl-suites", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    from capture_placements import rebase_suite
    from so_arm101_v2.contracts.bench import BenchConfig, scene_dependency_hash
    from so_arm101_v2.data._serialization import write_immutable_json
    from so_arm101_v2.simulation.suites import load_suite_from_path
    bench = BenchConfig.load(args.model.parent / "bench_config.json")
    root = args.output_dir.resolve()
    index = json.loads(args.rl_suites.read_text())

    def rebased(path: str) -> str:
        return str(rebase_suite(load_suite_from_path(path), Path(path), bench, root).resolve())

    out = dict(index, model=str(args.model.resolve()), scene_dependencies_sha256=scene_dependency_hash(args.model),
               rebased_from=str(args.rl_suites.resolve()), train_suites=[rebased(s) for s in index["train_suites"]],
               monitor_suite=rebased(index["monitor_suite"]))
    target = root / "rl_suites.json"
    if target.exists() and json.loads(target.read_text()) == out:
        print(f"unchanged: {target}")
        return 0
    write_immutable_json(target, out)
    print(f"wrote {target}: {len(out['train_suites'])} train suites + monitor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
