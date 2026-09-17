"""Builds the thumbnail and preview tiers described in DESIGN.md.

For each object it fetches that object's parquet once from the Hub and emits:
  web/thumbnails/sheet_NNNN.webp -- 10x10 sprite of 128px thumbs (100 objects per request)
  preview/<group>/<id>.webp    -- every view of one object at 256px in a single sprite

Both are viewer-facing tiers; the full-resolution parquet is never touched by the browser.

Usage:
  python tools/build_previews.py --limit 500 --workers 8
  python tools/build_previews.py --all --workers 32
"""

import argparse
import io
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq
import requests
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "https://huggingface.co/datasets/zeyuanyin/Dome-Objaverse/resolve/main"
SPLIT_DIR = "dome_objaverse"

THUMB_PX = 128
THUMBS_PER_SHEET = 100
SHEET_COLS = 10
PREVIEW_PX = 256
PREVIEW_COLS = 8
# Flatten RGBA onto a neutral background so thumbnails read well in both light and dark themes.
MATTE = (245, 245, 247)


def fetch_views(object_id):
    """Returns the object's views as PIL images, ordered by view_id."""
    r = requests.get(f"{REPO}/{SPLIT_DIR}/{object_id}.parquet", timeout=300)
    r.raise_for_status()
    table = pq.read_table(io.BytesIO(r.content), columns=["view_id", "image_png"])
    rows = sorted(table.to_pylist(), key=lambda x: x["view_id"])
    return [Image.open(io.BytesIO(x["image_png"])).convert("RGBA") for x in rows]


def flatten(img, size):
    img = img.resize((size, size), Image.LANCZOS)
    bg = Image.new("RGB", (size, size), MATTE)
    bg.paste(img, (0, 0), img)
    return bg


def build_preview(views, out_path):
    cols = PREVIEW_COLS
    rows = math.ceil(len(views) / cols)
    sheet = Image.new("RGB", (cols * PREVIEW_PX, rows * PREVIEW_PX), MATTE)
    for i, v in enumerate(views):
        sheet.paste(flatten(v, PREVIEW_PX), ((i % cols) * PREVIEW_PX, (i // cols) * PREVIEW_PX))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path, "WEBP", quality=82, method=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-previews", action="store_true", help="only build thumbnail sheets")
    args = ap.parse_args()

    index = json.load(open(os.path.join(ROOT, "web", "objects.json")))
    objects = [o[0] for o in index["objects"]]
    if not args.all:
        objects = objects[: args.limit]

    thumb_dir = os.path.join(ROOT, "web", "thumbnails")
    prev_dir = os.path.join(ROOT, "preview")
    os.makedirs(thumb_dir, exist_ok=True)

    n_sheets = math.ceil(len(objects) / THUMBS_PER_SHEET)
    print(f"{len(objects)} objects -> {n_sheets} sheet(s), {args.workers} workers")

    for s in range(n_sheets):
        batch = objects[s * THUMBS_PER_SHEET : (s + 1) * THUMBS_PER_SHEET]
        sheet_path = os.path.join(thumb_dir, f"sheet_{s:04d}.webp")
        if os.path.exists(sheet_path):
            continue

        sheet = Image.new("RGB", (SHEET_COLS * THUMB_PX, SHEET_COLS * THUMB_PX), MATTE)
        ok = 0

        def work(pos_obj):
            pos, obj = pos_obj
            views = fetch_views(obj)
            if not args.skip_previews:
                build_preview(views, os.path.join(prev_dir, f"{obj}.webp"))
            return pos, obj, flatten(views[0], THUMB_PX)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(work, p) for p in enumerate(batch)]
            for fut in as_completed(futures):
                try:
                    pos, obj, thumb = fut.result()
                except Exception as e:
                    print(f"  skip: {e}")
                    continue
                sheet.paste(thumb, ((pos % SHEET_COLS) * THUMB_PX, (pos // SHEET_COLS) * THUMB_PX))
                ok += 1

        sheet.save(sheet_path, "WEBP", quality=80, method=4)
        print(f"sheet {s:04d}: {ok}/{len(batch)} objects, {os.path.getsize(sheet_path)/1e3:.0f} KB")


if __name__ == "__main__":
    main()
