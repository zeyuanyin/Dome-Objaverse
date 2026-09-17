"""Packs rendered GObjaverse views into the `gobjaverse_parquet` split's parquet files.

Reference copy (see repo README for the dataset itself; this is not maintained as a runnable
pipeline -- the source renders it reads from `DATA_DIR` are not part of this release).

Schema per object: `keys` (str) / `values` (binary) key-value blobs, 200 rows = 40 views x
5 assets/view (`<id>.png`, `<id>_albedo.png`, `<id>_mr.png`, `<id>_nd.png`, `<id>.json`).

Fixed a real bug in the original script while archiving it: `save_parquet` was called *inside*
the per-view loop, so it rewrote each object's parquet file 40 times (once per accumulated
view) instead of once after collecting all views -- moved below the loop.
"""

import multiprocessing
import os

import cv2
import imageio.v2 as imageio
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # PNG/EXR encoding only, no GPU needed

DATA_DIR = os.environ.get("GOBJ_UNZIP_DIR", "./gobjaverse_unzip")
SAVE_DIR = os.environ.get("SAVE_DIR", "./gobjaverse_parquet")
ITEMS_FILE = "cobj_done_list.json"  # bundled manifest: 83,296 "<group_id>/<object_id>" entries


def main(item):
    outputs = {}

    dir_id, object_id = item.split("/")
    object_dir = os.path.join(DATA_DIR, dir_id, object_id, "campos_512_v4")
    save_path = os.path.join(SAVE_DIR, f"{dir_id}/{object_id}.parquet")

    for i in range(40):  # hard-coded 40 views, matching the original GObjaverse camera set
        view_dir = os.path.join(object_dir, f"{i:05}")
        image_path = os.path.join(view_dir, f"{i:05}.png")
        albedo_path = os.path.join(view_dir, f"{i:05}_albedo.png")
        mr_path = os.path.join(view_dir, f"{i:05}_mr.png")
        nd_path = os.path.join(view_dir, f"{i:05}_nd.exr")
        transform_path = os.path.join(view_dir, f"{i:05}.json")

        try:
            outputs[f"{i:05}.png"] = tf.io.encode_png(tf.convert_to_tensor(imageio.imread(image_path), tf.uint8)).numpy()
            outputs[f"{i:05}_albedo.png"] = tf.io.encode_png(tf.convert_to_tensor(imageio.imread(albedo_path)[:, :, :3], tf.uint8)).numpy()
            outputs[f"{i:05}_mr.png"] = tf.io.encode_png(tf.convert_to_tensor(imageio.imread(mr_path)[:, :, :3], tf.uint8)).numpy()

            # normal/depth EXR -> 16-bit PNG: normal [-1,1] -> [0,65535], depth scaled 1/5 then [0,1] -> [0,65535]
            nd = cv2.imread(nd_path, cv2.IMREAD_UNCHANGED)
            nd[:, :, :3] = nd[:, :, :3][..., ::-1]  # BGR -> RGB
            nd[:, :, :3] = (nd[:, :, :3] * 0.5 + 0.5) * 65535
            nd[:, :, 3] = nd[:, :, 3] / 5.0 * 65535
            outputs[f"{i:05}_nd.png"] = tf.io.encode_png(tf.convert_to_tensor(nd, tf.uint16)).numpy()

            with open(transform_path, "r") as f:
                outputs[f"{i:05}.json"] = f.read().encode("utf-8")
        except Exception:
            continue  # skip broken/missing view

    save_parquet(outputs, save_path)


def save_parquet(outputs, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    keys = pa.array(list(outputs.keys()), type=pa.string())
    values = pa.array(list(outputs.values()), type=pa.binary())
    table = pa.Table.from_arrays([keys, values], names=["keys", "values"])
    pq.write_table(table, save_path)


def load_parquet_to_dict(path: str) -> dict:
    table = pq.read_table(path)
    return dict(zip(table.column("keys").to_pylist(), table.column("values").to_pylist()))


if __name__ == "__main__":
    import json

    with open(ITEMS_FILE, "r") as f:
        items = json.load(f)

    num_workers = 8
    with multiprocessing.Pool(processes=num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap(main, items), total=len(items)):
            pass
