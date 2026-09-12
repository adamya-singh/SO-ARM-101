"""Derive a row-strided frames sidecar from a full capture, so the full file can be deleted (2026-09-12).

Every training since 2026-09-10 reads every 18th (or 30th) row of each
episode; the other rows of a 480-frame sidecar are never read, and the
2400-placement ladder capture is 211 GB. This writes a new capture directory
next to the source with ``images.npy`` holding only rows at
``action_index % stride == 0`` (episode-major, ``ceil(rows / stride)`` per
episode, the layout ``capture_oracle_demonstrations(frame_row_stride=...)``
produces), the arrays copied byte-for-byte, and a manifest equal to the
source's except for the ``frames`` block (``rows``, ``sha256``,
``row_stride``, ``rows_per_episode``), ``frame_store.row_stride`` and a
``derived_from`` record; the collection digest is recomputed over the new
frames digest exactly as the capture does when the source manifest carries
the identity fields, otherwise the source digest is kept and marked. Models
trained on the derived capture get a new run identity (different
``manifest_content_sha256`` / ``frames_sha256``); their inputs are the same
bytes as the source rows they use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so_arm101_v2.data._serialization import content_sha256, write_immutable_json  # noqa: E402
from so_arm101_v2.learning.vision import load_vision_frames  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True, help="source capture manifest.json (full sidecar)")
    p.add_argument("--stride", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True, help="capture root; writes <output-dir>/oracle/<suite_id>/<digest16>/")
    p.add_argument("--pointer", type=Path, default=None, help="write the new manifest path here (e.g. <run dir>/train_manifest.path)")
    args = p.parse_args(argv)
    if args.stride < 2:
        raise SystemExit("--stride must be >= 2")
    manifest, frames, arrays = load_vision_frames(args.manifest)
    if "row_stride" in manifest["frames"]:
        raise SystemExit("source is already row-strided")
    episode_lengths = [int(e["rows"]) for e in manifest["episodes"]]
    per_episode = [-(-n // args.stride) for n in episode_lengths]
    total = sum(per_episode)
    suite_id = manifest["suite_id"]
    stage = args.output_dir / "oracle" / suite_id / ".deriving"
    stage.mkdir(parents=True, exist_ok=True)
    out_frames = stage / "images.npy"
    started = time.time()
    store = np.lib.format.open_memmap(out_frames, mode="w+", dtype=np.uint8, shape=(total, 256, 256, 3))
    row = base = 0
    for i, (length, count) in enumerate(zip(episode_lengths, per_episode)):
        store[base:base + count] = frames[row:row + length:args.stride]   # sequential read of the source
        row += length; base += count
        if i % 200 == 0:
            print(f"episode {i}/{len(episode_lengths)} ({time.time() - started:.0f}s)", flush=True)
    store.flush(); del store
    digest = hashlib.sha256()
    with open(out_frames, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    frames_sha256 = digest.hexdigest()
    body = dict(manifest); body.pop("content_sha256")
    source_digest = body["collection_digest"]
    body["frames"] = dict(body["frames"], rows=int(total), sha256=frames_sha256, row_stride=int(args.stride), rows_per_episode=int(per_episode[0]))
    body["frame_store"] = dict(body.get("frame_store", {"format": "npy_memmap_uint8_v1", "frame_shape": [256, 256, 3]}), row_stride=int(args.stride))
    body["derived_from"] = dict(manifest_content_sha256=manifest["content_sha256"], collection_digest=source_digest,
                                frames_sha256=manifest["frames"]["sha256"], stride=int(args.stride), tool="tools/derive_strided_frames.py")
    body["collection_digest"] = content_sha256(dict(source_collection_digest=source_digest, frames_sha256=frames_sha256, row_stride=int(args.stride)))
    body["content_sha256"] = content_sha256(body)
    destination = args.output_dir / "oracle" / suite_id / body["collection_digest"][:16]
    if destination.exists():
        raise SystemExit(f"destination exists: {destination}")
    shutil.copy2(args.manifest.parent / manifest["arrays"]["path"], stage / manifest["arrays"]["path"])
    write_immutable_json(stage / "manifest.json", body)
    stage.rename(destination)
    print(f"derived {total} rows ({total * 196608 / 1e9:.1f} GB) in {time.time() - started:.0f}s -> {destination / 'manifest.json'}", flush=True)
    if args.pointer is not None:
        args.pointer.write_text(str((destination / "manifest.json").resolve()) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
