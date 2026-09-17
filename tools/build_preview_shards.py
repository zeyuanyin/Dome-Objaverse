"""Pack per-object WEBP previews into range-readable tar shards.

The viewer keeps one small index and fetches only the byte range for the selected object.
Uncompressed tar is intentional: it preserves random HTTP Range access and makes each shard
useful for users who want to download previews in batches.

Usage:
  python tools/build_preview_shards.py --objects-per-shard 512
"""

import argparse
import io
import json
import math
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "preview"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects-per-shard", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="pack only the first N objects (smoke test)")
    ap.add_argument("--out", type=Path, default=ROOT / "preview_shards")
    args = ap.parse_args()
    if args.objects_per_shard < 1:
        ap.error("--objects-per-shard must be positive")

    files = sorted(SOURCE.glob("*/*.webp"), key=lambda p: p.stem)
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no per-object previews found under {SOURCE}")
    args.out.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    entries = {}
    total = math.ceil(len(files) / args.objects_per_shard)
    for shard_no in range(total):
        batch = files[shard_no * args.objects_per_shard : (shard_no + 1) * args.objects_per_shard]
        name = f"preview-{shard_no:04d}.tar"
        path = shard_dir / name
        position = 0
        with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
            for source in batch:
                data = source.read_bytes()
                member_name = f"{source.parent.name}/{source.name}"
                info = tarfile.TarInfo(member_name)
                info.size = len(data)
                info.mtime = 0
                tar.addfile(info, io.BytesIO(data))
                entries[member_name[:-5]] = {
                    "shard": f"shards/{name}",
                    "offset": position + 512,
                    "length": len(data),
                }
                position += 512 + math.ceil(len(data) / 512) * 512
        print(f"{name}: {len(batch)} objects, {path.stat().st_size / 1e6:.1f} MB")

    manifest = {
        "version": 1,
        "format": "tar-range",
        "objects": len(entries),
        "objects_per_shard": args.objects_per_shard,
        "entries": entries,
    }
    tmp = args.out / "index.json.tmp"
    tmp.write_text(json.dumps(manifest, separators=(",", ":")))
    os.replace(tmp, args.out / "index.json")
    print(f"index.json: {len(entries)} objects, {total} shards")


if __name__ == "__main__":
    main()
