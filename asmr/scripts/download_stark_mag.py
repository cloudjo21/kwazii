"""Download STaRK-MAG data from HuggingFace to data/stark_mag/."""

import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "snap-stanford/stark"
REPO_TYPE = "dataset"

QA_FILES = [
    "qa/mag/split/train.index",
    "qa/mag/split/test.index",
    "qa/mag/split/val.index",
    "qa/mag/stark_qa/stark_qa.csv",
]
SKB_ZIP = "skb/mag/processed.zip"


def main() -> None:
    data_root = Path(os.getenv("ASMR_DATA_ROOT", "data/stark_mag"))
    print(f"Downloading STaRK-MAG → {data_root.resolve()}")

    for remote_path in QA_FILES:
        local_path = data_root / remote_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            print(f"  skip (exists): {local_path}")
            continue
        print(f"  download: {remote_path}")
        tmp = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=remote_path,
        )
        import shutil

        shutil.copy(tmp, local_path)

    corpus_path = data_root / "skb" / "mag" / "corpus.json"
    if corpus_path.exists():
        print(f"  skip (exists): {corpus_path}")
    else:
        print(f"  download: {SKB_ZIP}")
        zip_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=SKB_ZIP,
        )
        extract_dir = data_root / "skb" / "mag"
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"  extracting {zip_path} → {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            print("  zip contents:", zf.namelist()[:10])
            zf.extractall(extract_dir)

    print("\nData layout:")
    for p in sorted(data_root.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / 1_048_576
            print(f"  {p.relative_to(data_root)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
