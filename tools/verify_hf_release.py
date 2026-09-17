"""Check the published Hub layout, object coverage, and sampled copy identity.

Use ``--allow-old`` before removing the old complete split; omit it for the final check.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi

REPO = "zeyuanyin/Dome-Objaverse"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-old", action="store_true")
    args = parser.parse_args()
    expected = set(json.loads((ROOT / "metadata/cobj_done_list.json").read_text()))
    api = HfApi()
    paths = api.list_repo_files(REPO, repo_type="dataset")
    counts = Counter(p.split("/")[0] for p in paths)
    print("remote counts:", dict(counts), flush=True)

    for folder in ("dome_objaverse", "gobjaverse_parquet"):
        ids = {p[len(folder) + 1:-8] for p in paths
               if p.startswith(folder + "/") and p.endswith(".parquet")}
        assert ids == expected, (folder, len(ids), len(expected - ids), len(ids - expected))
    preview = {p for p in paths if p.startswith("preview/")}
    shard_names = {f"preview/shards/preview-{i:04d}.tar" for i in range(163)}
    assert preview == {"preview/index.json"} | shard_names, (len(preview), preview - shard_names)
    assert not any(p.startswith("preview_v2/") for p in paths)

    old = {p[len("complete-objaverse-parquet/"):-8] for p in paths
           if p.startswith("complete-objaverse-parquet/") and p.endswith(".parquet")}
    if args.allow_old:
        assert old == expected, ("old", len(old))
        sample = [sorted(expected)[0], sorted(expected)[-1]]
        sample += random.Random(20260916).sample(sorted(expected), 8)
        for object_id in sample:
            filenames = [f"{folder}/{object_id}.parquet"
                         for folder in ("complete-objaverse-parquet", "dome_objaverse")]
            infos = api.get_paths_info(REPO, filenames, repo_type="dataset", expand=True)
            assert len(infos) == 2 and infos[0].size == infos[1].size
            assert infos[0].lfs.sha256 == infos[1].lfs.sha256, object_id
        print("sampled 10 source/destination files: identical size and SHA256", flush=True)
    else:
        assert not old, f"old complete split still has {len(old)} files"
    print("release layout verified", flush=True)


if __name__ == "__main__":
    main()
