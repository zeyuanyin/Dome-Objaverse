"""Packs rendered views into the `dome_objaverse` split's parquet files.

Reference copy (see repo README for the dataset itself; not maintained as a runnable pipeline
-- the source renders in `DATA_DIR` are not part of this release).

Schema per object: `view_id` (int64) / `image_png` (binary) / `nd_png` (binary), 48 rows:
36 base views (view_id 0-35, elevation {0,30,60} x 12 azimuths) merged with 12 top views
(view_id 36-47, elevation ~90 x 12 azimuths) from the `complete-objaverse-top` render pass.
"""

import multiprocessing
import os

import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

DATA_DIR = os.environ.get("COBJ_DIR", "./complete-objaverse")
TOP_DATA_DIR = os.environ.get("COBJ_TOP_DIR", "./complete-objaverse-top")
SAVE_DIR = os.environ.get("SAVE_DIR", "./dome_objaverse")
ITEMS_FILE = "cobj_done_list.json"  # bundled manifest: 83,296 "<group_id>/<object_id>" entries

error_list = []


def main(item):
    dir_id, object_id = item.split("/")
    save_path = os.path.join(SAVE_DIR, f"{dir_id}/{object_id}.parquet")
    if os.path.exists(save_path):
        return

    rows_data = []

    base_dir = os.path.join(DATA_DIR, dir_id, object_id, "campos_512_v4")
    for i in range(36):
        view_dir = os.path.join(base_dir, f"{i:04}")
        rows_data.append(_load_view_row(view_dir, file_idx=i, view_id=i, item=item))

    # Top views: file names restart at 0000 under `complete-objaverse-top`, but get
    # `view_id` 36-47 in the merged schema.
    top_dir = os.path.join(TOP_DATA_DIR, dir_id, object_id, "campos_512_v4")
    for i in range(12):
        view_dir = os.path.join(top_dir, f"{i:04}")
        rows_data.append(_load_view_row(view_dir, file_idx=i, view_id=i + 36, item=item))

    rows_data = [r for r in rows_data if r is not None]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows_data), save_path)


def _load_view_row(view_dir, file_idx, view_id, item):
    image_path = os.path.join(view_dir, f"rgba_{file_idx:04}.png")
    nd_path = os.path.join(view_dir, f"nd_{file_idx:04}.png")
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        with open(nd_path, "rb") as f:
            nd_bytes = f.read()
    except (OSError, IOError) as e:
        print(e)
        error_list.append(item)
        return None
    return {"view_id": view_id, "image_png": image_bytes, "nd_png": nd_bytes}


if __name__ == "__main__":
    import json

    with open(ITEMS_FILE, "r") as f:
        items = json.load(f)

    num_workers = 8
    with multiprocessing.Pool(processes=num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap(main, items), total=len(items)):
            pass

    print(f"errors: {len(error_list)}")
    with open("error_list.txt", "w") as f:
        for item in error_list:
            f.write(item + "\n")
