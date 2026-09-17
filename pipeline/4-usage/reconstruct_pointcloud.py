"""Fuses an object's depth maps into a colored point cloud (.ply).

  python pipeline/4-usage/reconstruct_pointcloud.py 0/10228 --out /tmp/10228.ply

A Python port of the viewer's reconstruction (index.html), kept in sync with it. Encodes every
camera-model fix found while building this dataset -- get any one of them wrong and you get a
misregistered or exploded cloud:

  1. Camera distance is PER OBJECT (metadata/objects.parquet `camera_distance`, range 1.50-2.00),
     not a fixed constant. A constant fitted against one object reconstructs that object and
     misaligns most others -- this was caught on real data, see code-agent.md.
  2. Camera roll comes from azimuth directly, `right = (-sin(az), cos(az), 0)`. Deriving it from
     a cross product with world-up degenerates at elevation 90 and gives every top-down view the
     same roll despite differing azimuths -- smears any asymmetric feature into a ring.
  3. nd_png must be read with cv2 (IMREAD_UNCHANGED), not PIL/imageio, both of which silently
     downcast this 16-bit PNG to 8-bit -- see decode_normal_depth() in load_views.py.
  4. Depth alpha near its max (5.0) is background, not foreground -- it is not zero.
  5. Points are deduped onto a voxel grid; 48 views otherwise each sample the same surface on
     their own pixel lattice, so naive fusion looks like 48 overlapping layers.
  6. Flying pixels (sharp depth discontinuity against a neighbour) and grazing-angle samples
     (surface seen edge-on) are dropped; both are unreliable geometry.
"""

import argparse
import io
import math
import os
import struct

import cv2
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = "zeyuanyin/Dome-Objaverse"
FOV = 0.691150367
DEPTH_SCALE = 5.0
VOXEL = 0.006
FLYING_EPS = 0.02
GRAZING_MIN = 0.3


def camera_distance(object_id):
    row = next(r for r in pq.read_table(os.path.join(ROOT, "metadata", "objects.parquet")).to_pylist()
               if r["object_id"] == object_id)
    return row["camera_distance"]


def angles(view_id):
    """dome_objaverse's view layout: 36 base views (3 elevations x 12 azimuths),
    then 12 top views. See README's Azimuth/elevation detail section."""
    if view_id < 36:
        return {"el": [0, 30, 60][view_id // 12], "az": (view_id % 12) * 30}
    return {"el": 90, "az": (view_id - 36) * 30}


def camera_basis(cam_r, az_deg, el_deg):
    a, e = math.radians(az_deg), math.radians(el_deg)
    pos = np.array([cam_r * math.cos(e) * math.cos(a),
                    cam_r * math.cos(e) * math.sin(a),
                    cam_r * math.sin(e)])
    forward = -pos / cam_r
    right = np.array([-math.sin(a), math.cos(a), 0.0])
    up = np.cross(right, forward)
    return pos, forward, right, up


def decode_nd(nd_png_bytes):
    arr = cv2.imdecode(np.frombuffer(nd_png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr.dtype != np.uint16:
        raise RuntimeError(f"expected 16-bit nd_png, got {arr.dtype}")
    arr = arr.astype(np.float64)
    normal = arr[..., :3] / 65535.0 * 2.0 - 1.0   # cv2 order; see load_views.decode_normal_depth
    depth = arr[..., 3] / 65535.0 * DEPTH_SCALE
    return normal, depth


def unproject_view(image_png, nd_png, cam_r, az, el, step=3):
    normal, depth = decode_nd(nd_png)
    rgb = cv2.imdecode(np.frombuffer(image_png, np.uint8), cv2.IMREAD_UNCHANGED)  # BGR(A), uint8
    H, W = depth.shape
    pos, f, right, up = camera_basis(cam_r, az, el)
    t = math.tan(FOV / 2)

    ys, xs = np.mgrid[step:H - step:step, step:W - step:step]
    d = depth[ys, xs]
    fg = d < DEPTH_SCALE * 0.998

    # flying pixels: depth jumps sharply against a 4-neighbour
    jump = np.zeros_like(d)
    for dy, dx in [(0, step), (0, -step), (step, 0), (-step, 0)]:
        nd_ = depth[np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)]
        valid = nd_ < DEPTH_SCALE * 0.998
        jump = np.maximum(jump, np.where(valid, np.abs(nd_ - d), 0))
    fg &= jump <= FLYING_EPS

    n = normal[ys, xs]
    grazing = np.abs(n @ f) / (np.linalg.norm(n, axis=-1) + 1e-9)
    fg &= grazing >= GRAZING_MIN

    ndx = (xs + 0.5) / W * 2 - 1
    ndy = 1 - (ys + 0.5) / H * 2
    rays = f + right * (ndx * t)[..., None] + up * (ndy * t)[..., None]
    points = pos + rays * d[..., None]

    color = rgb[ys, xs][..., :3][..., ::-1] / 255.0  # BGR -> RGB
    if rgb.shape[-1] == 4:
        fg &= rgb[ys, xs][..., 3] >= 128

    return points[fg], color[fg]


def reconstruct(object_id, split="dome_objaverse", step=3):
    cam_r = camera_distance(object_id)
    path = hf_hub_download(REPO, f"{split}/{object_id}.parquet", repo_type="dataset")
    rows = pq.read_table(path).to_pylist()

    seen = {}
    all_pos, all_col = [], []
    for row in rows:
        a = angles(row["view_id"])
        pts, col = unproject_view(row["image_png"], row["nd_png"], cam_r, a["az"], a["el"], step)
        for p, c in zip(pts, col):
            key = tuple(np.round(p / VOXEL).astype(int))
            if key in seen:
                continue
            seen[key] = True
            all_pos.append(p)
            all_col.append(c)

    print(f"{object_id}: camera_distance={cam_r:.4f}, {len(all_pos)} points "
          f"from {len(rows)} views (deduped to one per {VOXEL*1000:.0f}mm)")
    return np.array(all_pos), np.array(all_col)


def write_ply(path, points, colors):
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n".encode())
        rgb = (colors * 255).astype(np.uint8)
        for p, c in zip(points.astype(np.float32), rgb):
            f.write(struct.pack("<fffBBB", p[0], p[1], p[2], c[0], c[1], c[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("object_id")
    ap.add_argument("--split", default="dome_objaverse")
    ap.add_argument("--step", type=int, default=3, help="pixel stride; lower = denser, slower")
    ap.add_argument("--out", default=None, help="write a .ply; omit to just print the point count")
    args = ap.parse_args()

    points, colors = reconstruct(args.object_id, args.split, args.step)
    if args.out:
        write_ply(args.out, points, colors)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
