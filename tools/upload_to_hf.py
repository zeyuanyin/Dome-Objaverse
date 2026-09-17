"""Publishes the metadata and preview tiers to the Hugging Face dataset.

Reads the token from the environment (HF_TOKEN) or an existing `huggingface-cli login`.
Never hardcode a token in this repo.

  export HF_TOKEN=hf_...
  python tools/upload_to_hf.py --what card
  python tools/upload_to_hf.py --what metadata
  python tools/upload_to_hf.py --what previews

What goes where (see DESIGN.md):
  metadata/objects.parquet                 join table: object_id, uid, glb_path, caption
  metadata/gobjaverse_index_to_objaverse.json
  preview/index.json + preview/shards/*.tar range-readable preview bundles
  README.md                                the dataset card
"""

import argparse
import os

from huggingface_hub import HfApi

REPO = "zeyuanyin/Dome-Objaverse"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["card", "metadata", "previews", "all"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    who = api.whoami()
    print(f"authenticated as {who['name']}")

    def upload_file(local, path_in_repo):
        print(f"  {local} -> {path_in_repo}")
        if args.dry_run:
            return
        api.upload_file(
            path_or_fileobj=local, path_in_repo=path_in_repo,
            repo_id=REPO, repo_type="dataset",
        )

    def upload_folder(local, path_in_repo):
        n = sum(len(f) for _, _, f in os.walk(local))
        print(f"  {local} ({n} files) -> {path_in_repo}")
        if args.dry_run:
            return
        api.upload_folder(
            folder_path=local, path_in_repo=path_in_repo,
            repo_id=REPO, repo_type="dataset",
        )

    if args.what in ("card", "all"):
        print("dataset card:")
        upload_file(os.path.join(ROOT, "docs", "dataset-card.md"), "README.md")

    if args.what in ("metadata", "all"):
        print("metadata:")
        upload_file(os.path.join(ROOT, "metadata", "objects.parquet"), "metadata/objects.parquet")
        upload_file(os.path.join(ROOT, "metadata", "gobjaverse_index_to_objaverse.json"),
                    "metadata/gobjaverse_index_to_objaverse.json")
        upload_file(os.path.join(ROOT, "metadata", "uncomplete_folders.txt"),
                    "metadata/uncomplete_folders.txt")

    if args.what in ("previews", "all"):
        print("previews (large; resumable — rerun if interrupted):")
        upload_folder(os.path.join(ROOT, "preview_shards"), "preview")

    if args.what in ("previews", "all") and not args.dry_run:
        print("\nNow set PREVIEWS_PUBLISHED = true in index.html so the viewer uses them.")

    print("done")


if __name__ == "__main__":
    main()
