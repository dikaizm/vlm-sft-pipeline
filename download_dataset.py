"""
Download the UCA (UCF-Crime Annotation) dataset from Kaggle.

The Kaggle dataset contains the annotation JSON files (Train/Val/Test splits)
with dense temporal descriptions. Videos (~98 GB) must be provided separately
— see README for the UCF-Crime video download link.

Usage:
    python vlm_sft_pipeline/download_dataset.py
    python vlm_sft_pipeline/download_dataset.py --dest /Volumes/T7/research-vlm/data
"""

import argparse
import json
import os
import shutil
from pathlib import Path


KAGGLE_DATASET = "vigneshwar472/ucaucf-crime-annotation-dataset"
DEFAULT_DEST   = "/Volumes/T7/research-vlm/data"

EXPECTED_FILES = [
    "UCFCrime_Train.json",
    "UCFCrime_Val.json",
    "UCFCrime_Test.json",
]


def download(dest: str) -> None:
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("kagglehub not installed. Run: pip install kagglehub")

    print(f"Downloading dataset: {KAGGLE_DATASET}")
    print("(Requires KAGGLE_USERNAME and KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json)\n")

    cache_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Downloaded to cache: {cache_path}\n")

    # Inspect what was downloaded
    downloaded = []
    for root, _, files in os.walk(cache_path):
        for f in files:
            full = os.path.join(root, f)
            size_kb = os.path.getsize(full) / 1024
            downloaded.append((os.path.relpath(full, cache_path), size_kb))

    print("Contents:")
    for rel_path, size_kb in sorted(downloaded):
        print(f"  {rel_path:<45} {size_kb:>8.1f} KB")

    # Copy JSON annotation files to dest
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    copied = []
    for root, _, files in os.walk(cache_path):
        for f in files:
            if f.endswith(".json") or f.endswith(".txt"):
                src = os.path.join(root, f)
                dst = dest_path / f
                shutil.copy2(src, dst)
                copied.append(f)

    if copied:
        print(f"\nCopied {len(copied)} annotation file(s) to {dest_path}:")
        for f in copied:
            print(f"  {f}")
    else:
        print(f"\n[WARN] No annotation files found in downloaded dataset.")
        print(f"       Raw cache is at: {cache_path}")

    _verify(dest_path)


def _verify(dest_path: Path) -> None:
    print("\n--- Verification ---")
    all_ok = True
    for fname in EXPECTED_FILES:
        fpath = dest_path / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            print(f"  [OK] {fname}  ({len(data)} entries)")
        else:
            print(f"  [MISSING] {fname}")
            all_ok = False

    if all_ok:
        print("\nAll annotation files present. Dataset ready for training.")
        print(f"Data directory: {dest_path}")
        print(f"\nNOTE: UCF-Crime videos (~98 GB) are NOT included in this Kaggle dataset.")
        print(f"      Videos should be placed at:")
        print(f"      {dest_path}/UCF_Crimes/UCF_Crimes/Videos/{{Category}}/{{video}}.mp4")
    else:
        print("\nSome files are missing. Check the Kaggle dataset contents.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UCA annotations from Kaggle")
    parser.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help=f"Destination directory for annotation files (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args()
    download(args.dest)


if __name__ == "__main__":
    main()
