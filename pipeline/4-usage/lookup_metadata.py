"""Looks up an object's Objaverse UID, source license path, caption and camera distance.

  python pipeline/4-usage/lookup_metadata.py 0/10228
  python pipeline/4-usage/lookup_metadata.py --uid 00130eea9e884209be92f68378a817e8   # reverse lookup
  python pipeline/4-usage/lookup_metadata.py --caption-contains "rubber duck"

All of this comes from metadata/objects.parquet, which ships in the repo (5.8MB, 83,296 rows) --
no need to download the upstream 46MB UID index or 48MB caption file separately.
"""

import argparse
import os

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OBJECTS_PARQUET = os.path.join(ROOT, "metadata", "objects.parquet")


def load():
    return pq.read_table(OBJECTS_PARQUET).to_pylist()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("object_id", nargs="?", help='e.g. "0/10228"')
    g.add_argument("--uid", help="Objaverse UID, e.g. 00130eea9e884209be92f68378a817e8")
    g.add_argument("--caption-contains", help="substring search over all 83,296 captions")
    args = ap.parse_args()

    rows = load()

    if args.object_id:
        matches = [r for r in rows if r["object_id"] == args.object_id]
    elif args.uid:
        matches = [r for r in rows if r["objaverse_uid"] == args.uid]
    else:
        q = args.caption_contains.lower()
        matches = [r for r in rows if q in r["caption"].lower()][:20]

    if not matches:
        print("no match")
        return
    for r in matches:
        print(f"object_id       {r['object_id']}")
        print(f"objaverse_uid   {r['objaverse_uid']}")
        print(f"glb_path        {r['glb_path']}   (shard within allenai/objaverse; carries the "
              f"source mesh's own license)")
        print(f"caption         {r['caption']}")
        print(f"camera_distance {r['camera_distance']:.4f}  (needed to unproject depth -- see "
              f"reconstruct_pointcloud.py; NOT a global constant, see README)")
        print()


if __name__ == "__main__":
    main()
