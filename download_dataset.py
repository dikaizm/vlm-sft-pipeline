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

# Ground-truth counts from the UCA dataset
EXPECTED_COUNTS = {
    "UCFCrime_Train.json": {"videos": 1165, "annotations": 15677},
    "UCFCrime_Val.json":   {"videos": 379,  "annotations": 3534},
    "UCFCrime_Test.json":  {"videos": 310,  "annotations": 4331},
}


def download(dest: str) -> None:
    import shutil

    try:
        import kagglehub
    except ImportError:
        raise SystemExit("kagglehub not installed. Run: pip install kagglehub")

    print(f"Downloading dataset: {KAGGLE_DATASET}")
    print("(Requires KAGGLE_USERNAME and KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json)\n")

    cache_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Downloaded to cache: {cache_path}\n")

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Copy annotation files from cache to dest
    copied = []
    for root, _, files in os.walk(cache_path):
        for f in files:
            if f.endswith(".json") or f.endswith(".txt"):
                src = os.path.join(root, f)
                dst = dest_path / f
                shutil.copy2(src, dst)
                copied.append(f)

    if copied:
        print(f"Copied {len(copied)} file(s) to {dest_path}:")
        for f in copied:
            print(f"  {f}")
    else:
        print(f"[WARN] No annotation files found in cache: {cache_path}")

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
        if not fpath.exists():
            print(f"  [MISSING] {fname}")
            all_ok = False
            continue

        with open(fpath) as f:
            data = json.load(f)

        exp = EXPECTED_COUNTS[fname]
        n_videos = len(data)
        n_annotations = 0
        issues = []

        for vid, ann in data.items():
            if not all(k in ann for k in ("duration", "timestamps", "sentences")):
                issues.append(f"{vid}: missing required fields")
                continue
            ts, sents = ann["timestamps"], ann["sentences"]
            if len(ts) != len(sents):
                issues.append(f"{vid}: timestamps/sentences length mismatch")
            n_annotations += len(ts)
            for s, e in ts:
                if float(s) >= float(e) or float(s) < 0:
                    issues.append(f"{vid}: invalid timestamp [{s}, {e}]")

        video_ok = n_videos == exp["videos"]
        ann_ok   = n_annotations == exp["annotations"]
        status   = "[OK]" if (video_ok and ann_ok and not issues) else "[WARN]"

        print(f"  {status} {fname}")
        print(f"         videos: {n_videos} (expected {exp['videos']}) {'✓' if video_ok else '✗'}")
        print(f"         annotations: {n_annotations} (expected {exp['annotations']}) {'✓' if ann_ok else '✗'}")
        if issues:
            print(f"         issues: {len(issues)}")
            for msg in issues[:5]:
                print(f"           - {msg}")
            if len(issues) > 5:
                print(f"           ... and {len(issues) - 5} more")
            all_ok = False
        if not (video_ok and ann_ok):
            all_ok = False

    if all_ok:
        print("\nDataset complete and valid. Ready for training.")
        print(f"Data directory: {dest_path}")
        print(f"\nNOTE: UCF-Crime videos (~98 GB) are NOT included in this dataset.")
        print(f"      Videos should be placed at:")
        print(f"      {dest_path}/UCF_Crimes/UCF_Crimes/Videos/{{Category}}/{{video}}.mp4")
    else:
        print("\nVerification failed. Re-download or check the dataset.")


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
