"""Copy the 48-view split to its release path using Hub server-side copies.

Run after logging in with ``hf auth login``::

    python tools/migrate_hf_folder.py

This is resumable: each run checks remote destination files and copies only missing objects.
It does not remove the old directory. Verify the destination before deleting the source.
"""

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import CommitOperationCopy, HfApi

REPO = "zeyuanyin/Dome-Objaverse"
SOURCE = "complete-objaverse-parquet"
DEST = "dome_objaverse"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 25000:
        parser.error("--batch-size must be 1–25000")

    api = HfApi()
    ids = json.loads((ROOT / "metadata/cobj_done_list.json").read_text())
    remote = api.list_repo_files(REPO, repo_type="dataset")
    source = {p[len(SOURCE) + 1:-8] for p in remote if p.startswith(SOURCE + "/") and p.endswith(".parquet")}
    dest = {p[len(DEST) + 1:-8] for p in remote if p.startswith(DEST + "/") and p.endswith(".parquet")}
    expected = set(ids)
    if source != expected:
        raise SystemExit(f"source differs from manifest: {len(source)} source, {len(expected)} expected, {len(expected-source)} missing, {len(source-expected)} extra")
    missing = sorted(expected - dest)
    print(f"source={len(source)} destination={len(dest)} missing={len(missing)}", flush=True)
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start:start + args.batch_size]
        operations = [CommitOperationCopy(src_path_in_repo=f"{SOURCE}/{id}.parquet",
                                          path_in_repo=f"{DEST}/{id}.parquet") for id in batch]
        attempt = 0
        while True:
            try:
                commit = api.create_commit(repo_id=REPO, repo_type="dataset",
                                           operations=operations,
                                           commit_message=f"Migrate dome split files {start + 1}–{start + len(batch)}")
                print(f"{start + len(batch)}/{len(missing)} {commit.oid}", flush=True)
                break
            except Exception as exc:
                if getattr(getattr(exc, "response", None), "status_code", None) == 429:
                    print("Hub commit limit reached; retrying this batch in 60s", flush=True)
                    time.sleep(60)
                    continue
                if attempt == 4:
                    raise
                pause = 2 ** attempt * 5
                print(f"retry {attempt + 1} after {type(exc).__name__}: {exc}; sleeping {pause}s", flush=True)
                time.sleep(pause)
                attempt += 1


if __name__ == "__main__":
    main()
