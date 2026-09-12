"""Replace a full frames sidecar with its derived row-strided copy, after verifying it (2026-09-12).

Runs after tools/derive_strided_frames.py: loads both captures, checks the
manifest content hashes, checks that ``--samples`` random stored rows are
byte-identical between the derived sidecar and the source, then (and only
then) deletes the source ``images.npy``, leaves ``FRAMES_DELETED.txt`` beside
it naming the derived manifest, and repoints ``--pointer`` (e.g. the ladder
directory's ``train_manifest.path``) at the derived manifest, keeping the old
pointer as ``<name>.full.bak``. Any check failure leaves everything in place.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so_arm101_v2.learning.vision import load_vision_frames  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--derived-pointer", type=Path, required=True, help="file holding the derived manifest path (written by derive_strided_frames --pointer)")
    p.add_argument("--pointer", type=Path, required=True, help="pointer file to repoint at the derived manifest")
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    derived_manifest = Path(args.derived_pointer.read_text().strip())
    src_manifest, src_frames, src_arrays = load_vision_frames(args.source_manifest)      # validates content hashes
    drv_manifest, drv_frames, drv_arrays = load_vision_frames(derived_manifest)
    if drv_manifest.get("derived_from", {}).get("manifest_content_sha256") != src_manifest["content_sha256"]:
        raise SystemExit("derived manifest does not name this source")
    if any(not np.array_equal(src_arrays[k], drv_arrays[k]) for k in src_arrays):
        raise SystemExit("arrays differ")
    stored = np.flatnonzero(drv_frames.stored_mask)
    picks = np.sort(np.random.default_rng(args.seed).choice(stored, min(args.samples, stored.shape[0]), replace=False))
    started = time.time()
    for start in range(0, picks.shape[0], 200):
        rows = picks[start:start + 200]
        if not np.array_equal(np.asarray(src_frames[rows]), drv_frames[rows]):
            raise SystemExit(f"frame bytes differ at rows {rows[:5]}")
    print(f"verified {picks.shape[0]} stored rows byte-identical in {time.time() - started:.0f}s; deleting the full sidecar", flush=True)
    full = args.source_manifest.parent / src_manifest["frames"]["path"]
    size = full.stat().st_size
    del src_frames
    full.unlink()
    (args.source_manifest.parent / "FRAMES_DELETED.txt").write_text(
        f"full frames sidecar ({size / 1e9:.0f} GB) deleted {time.strftime('%Y-%m-%d %H:%M')} after byte-verified derivation of a row-strided copy: "
        f"{derived_manifest}\n")
    backup = args.pointer.with_name(args.pointer.name + ".full.bak")
    if args.pointer.exists() and not backup.exists():
        shutil.copy2(args.pointer, backup)
    args.pointer.write_text(str(derived_manifest.resolve()) + "\n")
    print(f"freed {size / 1e9:.0f} GB; {args.pointer} -> {derived_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
