"""Pick the winner of a hyperparameter sweep of tools/vla_finetune.py runs (2026-11-18).

Each ``--run name=dir`` is a finished run directory (``result.json`` with ``final`` and ``identity.config``). The
winner is the run with the lowest held-out loss at chunk start 90 (``final.heldout_start_90``, the repo's
generalisation metric); the per-pose median and the closed-loop counts are recorded beside it, not used to choose.
Writes ``--out``: the ranking and the winner's learning rate, backbone learning rate, KL weight and steps, which the
queue scripts read. A run that did not finish is listed as missing and cannot win; no finished run is an error.

  tools/sweep_select.py --run a=<dir> --run b=<dir> --out <winner.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(name: str, directory: Path) -> dict | None:
    path = directory / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    config, final = result["identity"]["config"], result["final"]
    rollouts = {k: dict(successes=v.get("successes"), pickups=v.get("pickups"), rollouts=v.get("rollouts"),
                        lateral_median_mm=(v.get("closure") or {}).get("lateral_median_mm"))
                for k, v in (result.get("rollouts") or {}).items() if isinstance(v, dict)}
    return dict(name=name, dir=str(directory), heldout_start_90=float(final["heldout_start_90"]), median=float(final["median"]),
                ratio_start_90=final.get("ratio_start_90"), rollouts=rollouts, learning_rate=config["learning_rate"],
                backbone_learning_rate=config["backbone_learning_rate"], kl_weight=config["kl_weight"], max_steps=config["max_steps"],
                seed=config["seed"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rows, missing = [], []
    for item in args.run:
        name, directory = item.split("=", 1)
        row = summarize(name, Path(directory))
        (rows if row else missing).append(row or name)
    if not rows:
        raise SystemExit("no finished run to choose from")
    rows.sort(key=lambda r: r["heldout_start_90"])
    winner = rows[0]
    record = dict(criterion="lowest final.heldout_start_90", winner=winner, ranking=rows, missing=missing)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    for r in rows:
        print(f"{r['name']:28s} heldout90 {r['heldout_start_90']:.2e} median {r['median']:.2e} lr {r['learning_rate']:g} "
              f"backbone {r['backbone_learning_rate']:g} kl {r['kl_weight']:g} steps {r['max_steps']} rollouts {json.dumps(r['rollouts'])}")
    print(f"WINNER {winner['name']}" + (f"; missing {missing}" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
