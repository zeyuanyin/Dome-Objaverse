"""Builds the metadata spine described in DESIGN.md.

Outputs:
  metadata/objects.parquet -- 83,296 rows, the join table (also uploaded to the Hub)
  web/objects.json         -- ids + captions, loaded up-front by the viewer
  web/uids.json            -- Objaverse UIDs in the same order, prefetched during idle time

The viewer index is columnar (parallel arrays rather than a row per object) and keeps UIDs in
their own file, because those two choices cut what every visitor downloads from 3.37 MB to
1.32 MB gzipped.

Inputs are the files staged under `metadata/` (see DESIGN.md for where they came from).
"""

import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJECT_LIST = os.path.join(ROOT, "metadata", "cobj_done_list.json")
INDEX_MAP = os.path.join(ROOT, "metadata", "gobjaverse_index_to_objaverse.json")
CAPTIONS = os.path.join(ROOT, "metadata", "text_captions_cap3d.json")
CAM_DISTS = os.path.join(ROOT, "metadata", "camera_distances.json")
OUT_PARQUET = os.path.join(ROOT, "metadata", "objects.parquet")
OUT_INDEX = os.path.join(ROOT, "web", "objects.json")
OUT_UIDS = os.path.join(ROOT, "web", "uids.json")

THUMBS_PER_SHEET = 100  # 10x10 sprite, see DESIGN.md


def main():
    objects = json.load(open(OBJECT_LIST))
    index_map = json.load(open(INDEX_MAP))
    captions = json.load(open(CAPTIONS))
    cam_dists = json.load(open(CAM_DISTS))

    # Sort so sprite-sheet assignment is stable across rebuilds.
    objects.sort(key=lambda o: (int(o.split("/")[0]), int(o.split("/")[1])))

    rows, ids, caps, uids, dists = [], [], [], [], []
    for i, obj in enumerate(objects):
        group_id, index_id = obj.split("/")
        glb_path = index_map.get(obj, "")
        uid = os.path.basename(glb_path).removesuffix(".glb")
        caption = captions.get(obj, "")
        # camera distance varies per object; it comes from that object's own GObjaverse
        # camera JSON, so a single global constant does not reconstruct correctly
        dist = cam_dists.get(obj)
        rows.append(
            {
                "object_id": obj,
                "group_id": int(group_id),
                "index_id": int(index_id),
                "objaverse_uid": uid,
                "glb_path": glb_path,
                "caption": caption,
                "camera_distance": dist,
                "sheet_id": i // THUMBS_PER_SHEET,
                "sheet_pos": i % THUMBS_PER_SHEET,
            }
        )
        ids.append(obj)
        caps.append(caption)
        uids.append(uid)
        dists.append(round(dist, 4) if dist else None)

    pq.write_table(pa.Table.from_pylist(rows), OUT_PARQUET, compression="zstd")

    def dump(path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)

    dump(OUT_INDEX, {"n": len(ids), "thumbs_per_sheet": THUMBS_PER_SHEET,
                     "ids": ids, "captions": caps, "dists": dists})
    dump(OUT_UIDS, uids)

    missing_uid = sum(1 for u in uids if not u)
    missing_cap = sum(1 for c in caps if not c)
    missing_dist = sum(1 for d in dists if d is None)
    print(f"objects: {len(rows)}  missing uid: {missing_uid}  missing caption: {missing_cap}"
          f"  missing camera distance: {missing_dist}")
    print(f"sheets: {rows[-1]['sheet_id'] + 1}")
    for p in (OUT_PARQUET, OUT_INDEX, OUT_UIDS):
        print(f"  {os.path.relpath(p, ROOT)}: {os.path.getsize(p) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
