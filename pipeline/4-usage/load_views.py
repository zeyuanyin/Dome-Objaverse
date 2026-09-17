"""Loads one object's views (RGB, normal, depth) from a Dome-Objaverse parquet file.

  python pipeline/4-usage/load_views.py 0/10228
  python pipeline/4-usage/load_views.py 0/10228 --split gobjaverse
  python pipeline/4-usage/load_views.py 0/10228 --out-dir /tmp/10228

The released splits use two parquet layouts documented in the README:
  - dome_objaverse: row-per-view,
    columns `view_id`, `image_png`, `nd_png`
  - gobjaverse_parquet: key-value blobs, columns `keys`, `values`

`nd_png` decoding matters and is easy to get backwards -- see decode_normal_depth().
"""

import argparse
import io
import os

import cv2
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "zeyuanyin/Dome-Objaverse"
SPLIT_DIRS = {
    "dome_objaverse": "dome_objaverse",
    "complete": "dome_objaverse",
    "gobjaverse": "gobjaverse_parquet",
}


def decode_normal_depth(nd_png_bytes, depth_scale=5.0):
    """Decodes an `nd_png` blob into (normal_xyz, depth), both float32.

    Must use cv2, not PIL/imageio: both silently downcast this 16-bit-per-channel PNG to 8-bit
    (still reporting a normal-looking "RGBA" array, no error, no warning -- verified directly,
    dtype comes back uint8 with max 255). Depth values then land ~257x too small, e.g. 0.006-0.02
    for an object whose camera distance is ~2. Only `cv2.imdecode(..., cv2.IMREAD_UNCHANGED)`
    preserves the full 16-bit range here.

    Channel order: none needed beyond cv2's own convention. The packer wrote this array through
    cv2.imencode, which treats its input as BGR; cv2.imdecode reverses that same way on read, so
    the two conventions cancel and channels 0/1/2 come back as normal x/y/z directly. Verified on
    a top-down view (camera looking straight down): this order gives a mean surface normal within
    a few degrees of straight up (dot product -0.957 against the view direction); the reversed
    order gives 0.000 (no correlation). A reader that does NOT go through cv2's BGR convention
    (raw file/RGBA order, e.g. a JS PNG decoder) needs the opposite: x/y/z from channels 2/1/0.
    """
    arr = cv2.imdecode(np.frombuffer(nd_png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr.dtype != np.uint16:
        raise RuntimeError(f"expected 16-bit nd_png, got {arr.dtype} -- decoder silently downcast it")
    arr = arr.astype(np.float64)
    normal = (arr[..., :3] / 65535.0 * 2.0 - 1.0).astype(np.float32)
    depth = (arr[..., 3] / 65535.0 * depth_scale).astype(np.float32)
    return normal, depth


def load_complete(object_id, split="complete"):
    """Returns a dict {view_id: {"image": PIL.Image, "normal": HxWx3 float32, "depth": HxW float32}}."""
    path = hf_hub_download(REPO, f"{SPLIT_DIRS[split]}/{object_id}.parquet", repo_type="dataset")
    table = pq.read_table(path)
    out = {}
    for row in table.to_pylist():
        normal, depth = decode_normal_depth(row["nd_png"])
        out[row["view_id"]] = {
            "image": Image.open(io.BytesIO(row["image_png"])).convert("RGBA"),
            "normal": normal,
            "depth": depth,
        }
    return out


def load_gobjaverse(object_id, num_views=40):
    """Returns the same shape as load_complete(), decoded from the key-value schema."""
    path = hf_hub_download(REPO, f"{SPLIT_DIRS['gobjaverse']}/{object_id}.parquet", repo_type="dataset")
    table = pq.read_table(path)
    blobs = dict(zip(table.column("keys").to_pylist(), table.column("values").to_pylist()))
    out = {}
    for i in range(num_views):
        k = f"{i:05d}"
        if f"{k}.png" not in blobs:
            continue
        normal, depth = decode_normal_depth(blobs[f"{k}_nd.png"])
        out[i] = {
            "image": Image.open(io.BytesIO(blobs[f"{k}.png"])).convert("RGBA"),
            "albedo": Image.open(io.BytesIO(blobs[f"{k}_albedo.png"])),
            "metal_roughness": Image.open(io.BytesIO(blobs[f"{k}_mr.png"])),
            "normal": normal,
            "depth": depth,
            "camera_json": blobs[f"{k}.json"].decode("utf-8"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("object_id", help='e.g. "0/10228"')
    ap.add_argument("--split", choices=["dome_objaverse", "gobjaverse", "complete"], default="dome_objaverse")
    ap.add_argument("--out-dir", default=None, help="if set, saves RGB + a depth preview per view")
    args = ap.parse_args()

    views = load_gobjaverse(args.object_id) if args.split == "gobjaverse" else load_complete(args.object_id, args.split)
    print(f"{args.object_id} ({args.split}): {len(views)} views")
    for vid in sorted(views)[:3]:
        v = views[vid]
        print(f"  view {vid}: image {v['image'].size}, depth range "
              f"[{v['depth'].min():.3f}, {v['depth'].max():.3f}]")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for vid, v in views.items():
            v["image"].save(os.path.join(args.out_dir, f"{vid:05d}_rgb.png"))
            depth_vis = np.clip(v["depth"] / max(v["depth"][v["depth"] < 4.9].max(), 1e-6), 0, 1)
            Image.fromarray((depth_vis * 255).astype(np.uint8)).save(
                os.path.join(args.out_dir, f"{vid:05d}_depth.png"))
        print(f"saved {len(views)} views to {args.out_dir}")


if __name__ == "__main__":
    main()
