"""Deletes the redundant `complete_objaverse_top_parquet` split from the Hugging Face dataset.

That split is byte-identical to `view_id` 36-47 of `dome_objaverse`; removing it
reclaims ~852.7 GB and leaves users one fewer "which split do I want?" decision.

**This is irreversible.** Re-uploading would mean regenerating 852.7 GB. The script therefore
re-verifies redundancy on a random sample before deleting anything, and refuses to run without
an explicit --yes-delete-852gb flag.

  export HF_TOKEN=hf_...
  python tools/delete_top_split.py                      # verify only, deletes nothing
  python tools/delete_top_split.py --yes-delete-852gb   # actually delete

Before running, make sure the viewer no longer points at this split (index.html reads top-down
views out of the complete split) and the docs no longer list it.
"""

import argparse
import io
import os
import random

import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi

REPO = "zeyuanyin/Dome-Objaverse"
TOP = "complete_objaverse_top_parquet"
COMPLETE = "dome_objaverse"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def fetch(split, object_id):
    r = requests.get(f"{BASE}/{split}/{object_id}.parquet", timeout=300)
    r.raise_for_status()
    return {row["view_id"]: (row["image_png"], row["nd_png"])
            for row in pq.read_table(io.BytesIO(r.content)).to_pylist()}


def verify(api, sample_size):
    """Confirms every top view is byte-identical to complete's view_id + 36."""
    files = api.list_repo_files(REPO, repo_type="dataset")
    ids = sorted({f[len(TOP) + 1:-len(".parquet")]
                  for f in files if f.startswith(TOP + "/") and f.endswith(".parquet")})
    print(f"{TOP}: {len(ids)} objects on the Hub")

    picks = random.sample(ids, min(sample_size, len(ids)))
    identical = differing = 0
    for oid in picks:
        top, comp = fetch(TOP, oid), fetch(COMPLETE, oid)
        for vid, (img, nd) in top.items():
            other = comp.get(vid + 36)
            if other and other[0] == img and other[1] == nd:
                identical += 1
            else:
                differing += 1
                print(f"  MISMATCH {oid} view {vid}")
    print(f"verified {len(picks)} objects: {identical} views identical, {differing} differing")
    return differing == 0, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes-delete-852gb", action="store_true",
                    help="actually delete; without it the script only verifies")
    ap.add_argument("--sample", type=int, default=15, help="objects to re-verify before deleting")
    args = ap.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    print(f"authenticated as {api.whoami()['name']}")

    ok, ids = verify(api, args.sample)
    if not ok:
        print("\nREFUSING TO DELETE: the split is not fully redundant.")
        return
    if not args.yes_delete_852gb:
        print("\nVerified redundant. Re-run with --yes-delete-852gb to delete. Nothing changed.")
        return

    print(f"\ndeleting {TOP}/ ({len(ids)} objects, ~852.7 GB)...")
    api.delete_folder(path_in_repo=TOP, repo_id=REPO, repo_type="dataset",
                      commit_message="Remove redundant top-down split (identical to complete view_id 36-47)")
    print("deleted. Update the dataset card so it no longer lists three splits.")


if __name__ == "__main__":
    main()
