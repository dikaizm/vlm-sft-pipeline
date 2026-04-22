"""
Download the UCA (UCF-Crime Annotation) dataset from Kaggle.

The dataset contains annotation JSON files (Train/Val/Test splits) and the
UCF-Crime videos (~98 GB). Full dataset is ~105 GB — dest needs ~210 GB free
during extraction (zip + extracted files). The zip is deleted after extraction.

Usage:
    python vlm-sft-pipeline/download_dataset.py
    python vlm-sft-pipeline/download_dataset.py --dest /data

Requires KAGGLE_USERNAME and KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json.
"""

import argparse
import json
import os
import shutil
import subprocess
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

EXPECTED_COUNTS = {
    "UCFCrime_Train.json": {"videos": 1165, "annotations": 15677},
    "UCFCrime_Val.json":   {"videos": 379,  "annotations": 3534},
    "UCFCrime_Test.json":  {"videos": 310,  "annotations": 4331},
}


def download(dest: str) -> None:
    if not shutil.which("kaggle"):
        raise SystemExit("kaggle not installed. Run: pip install kaggle")

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {KAGGLE_DATASET}")
    print(f"Destination: {dest_path}")
    print("(Requires KAGGLE_USERNAME and KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json)")
    print("NOTE: Full dataset is ~105 GB. Dest needs ~210 GB free during extraction.\n")

    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(dest_path),
            "--unzip",
        ],
        check=True,
    )

    # Delete the zip left behind by kaggle CLI
    zip_file = dest_path / "ucaucf-crime-annotation-dataset.zip"
    if zip_file.exists():
        print(f"\nRemoving zip: {zip_file} ({zip_file.stat().st_size / 1e9:.1f} GB)...")
        zip_file.unlink()
        print("Zip removed.")

    print(f"\nExtracted to: {dest_path}")
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
        n_videos      = len(data)
        n_annotations = 0
        issues        = []

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

    # Check videos directory
    video_dir = dest_path / "UCF_Crimes" / "UCF_Crimes" / "Videos"
    if video_dir.exists():
        categories = [d for d in video_dir.iterdir() if d.is_dir()]
        n_videos   = sum(len(list(c.glob("*.mp4"))) for c in categories)
        print(f"\n  Videos: {n_videos} mp4 files across {len(categories)} categories")
        print(f"  Path: {video_dir}")
    else:
        print(f"\n  [MISSING] Videos directory: {video_dir}")
        all_ok = False

    if all_ok:
        print("\nDataset complete and valid. Ready for training.")
    else:
        print("\nVerification failed. Re-download or check the dataset.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UCA dataset from Kaggle")
    parser.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args()
    download(args.dest)


if __name__ == "__main__":
    main()
