"""Packs rendered top-down views into the `complete-objaverse-top` split's parquet files.

Reference copy (see repo README for the dataset itself; not maintained as a runnable pipeline
-- the source renders in `DATA_DIR` are not part of this release).

Schema per object: `view_id` (int64) / `image_png` (binary) / `nd_png` (binary), 12 rows
(elevation ~90 deg, 12-step azimuth sweep). Same rows are merged into
`dome_objaverse` as view_id 36-47 by `complete_objaverse_parquet_gen.py`.
"""

import multiprocessing
import os

import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

DATA_DIR = os.environ.get("COBJ_TOP_DIR", "./complete-objaverse-top")
SAVE_DIR = os.environ.get("SAVE_DIR", "./complete_objaverse_top_parquet")
ITEMS_FILE = "cobj_done_list.json"  # bundled manifest: 83,296 "<group_id>/<object_id>" entries

error_list = []


def main(item):
    dir_id, object_id = item.split("/")
    save_path = os.path.join(SAVE_DIR, f"{dir_id}/{object_id}.parquet")
    if os.path.exists(save_path):
        return

    object_dir = os.path.join(DATA_DIR, dir_id, object_id, "campos_512_v4")
    rows_data = []
    for i in range(12):
        view_dir = os.path.join(object_dir, f"{i:04}")
        image_path = os.path.join(view_dir, f"rgba_{i:04}.png")
        nd_path = os.path.join(view_dir, f"nd_{i:04}.png")
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            with open(nd_path, "rb") as f:
                nd_bytes = f.read()
        except (OSError, IOError) as e:
            print(e)
            error_list.append(item)
            continue
        rows_data.append({"view_id": i, "image_png": image_bytes, "nd_png": nd_bytes})

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows_data), save_path)


if __name__ == "__main__":
    import json

    with open(ITEMS_FILE, "r") as f:
        items = json.load(f)

    num_workers = 4
    with multiprocessing.Pool(processes=num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap(main, items), total=len(items)):
            pass

    print(f"errors: {len(error_list)}")
    with open("error_list.txt", "w") as f:
        for item in error_list:
            f.write(item + "\n")
