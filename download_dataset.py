"""
Download the UCA (UCF-Crime Annotation) dataset from Kaggle.

The dataset contains annotation JSON files (Train/Val/Test splits) with dense
temporal descriptions. Videos (~98 GB) must be provided separately — see README
for the UCF-Crime video download link.

Usage:
    python vlm-sft-pipeline/download_dataset.py
    python vlm-sft-pipeline/download_dataset.py --dest /Volumes/T7/research-vlm/data

Requires KAGGLE_USERNAME and KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

KAGGLE_DATASET = "vigneshwar472/ucaucf-crime-annotation-dataset"
DEFAULT_DEST   = os.environ.get("DATA_ROOT", "./data")

EXPECTED_FILES = [
    "UCFCrime_Train.json",
    "UCFCrime_Val.json",
    "UCFCrime_Test.json",
]


def download(dest: str) -> None:
    import shutil
    import subprocess

    if not shutil.which("kaggle"):
        raise SystemExit("kaggle not installed. Run: pip install kaggle")

    print(f"Downloading dataset: {KAGGLE_DATASET}")
    print("(Requires KAGGLE_USERNAME and KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json)\n")

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(dest_path), "--unzip"],
        check=True,
    )

    print(f"\nDownloaded to: {dest_path}\n")

    for root, _, files in os.walk(dest_path):
        for f in sorted(files):
            full    = os.path.join(root, f)
            size_kb = os.path.getsize(full) / 1024
            rel     = os.path.relpath(full, dest_path)
            print(f"  {rel:<45} {size_kb:>8.1f} KB")

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
        print(f"\nNOTE: UCF-Crime videos (~98 GB) are NOT included in this dataset.")
        print(f"      Videos should be placed at:")
        print(f"      {dest_path}/UCF_Crimes/UCF_Crimes/Videos/{{Category}}/{{video}}.mp4")
    else:
        print("\nSome files are missing. Check the Kaggle dataset contents.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UCA annotations from Kaggle")
    parser.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args()
    download(args.dest)


if __name__ == "__main__":
    main()
