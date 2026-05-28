"""
Download trained model checkpoint(s) from RunPod S3 volume.

Skips files already present with matching size (resume-safe).
Shows per-file tqdm progress.

Usage:
    python vlm-sft-pipeline/fetch/download_model_s3.py
    python vlm-sft-pipeline/fetch/download_model_s3.py \
        --prefix vlm-sft-pipeline/output/smolvlm2-500m-full-sft/checkpoint-1072 \
        --dest /Volumes/T7/research-vlm/output/smolvlm2-500m-full-sft/checkpoint-1072
    python vlm-sft-pipeline/fetch/download_model_s3.py --dry-run
"""

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BUCKET       = "z3oqdvyui7"
REGION       = "eu-ro-1"
ENDPOINT_URL = "https://s3api-eu-ro-1.runpod.io"

_HERE        = Path(__file__).parent.parent   # vlm-sft-pipeline/
DEFAULT_PREFIX = "vlm-sft-pipeline/output"
DEFAULT_DEST   = str(_HERE / "output")

MAX_WORKERS = 4


# ---------------------------------------------------------------------------
# S3 client
# ---------------------------------------------------------------------------

def make_client():
    return boto3.client(
        "s3",
        region_name=REGION,
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


# ---------------------------------------------------------------------------
# List all objects under prefix
# ---------------------------------------------------------------------------

def list_objects(client, prefix: str) -> list[dict]:
    """Single-page list — no pagination, avoids RunPod S3 token bug."""
    resp = client.list_objects(Bucket=BUCKET, Prefix=prefix, MaxKeys=1000)
    return [{"key": o["Key"], "size": o["Size"]} for o in resp.get("Contents", [])]


# ---------------------------------------------------------------------------
# Download single file with tqdm byte progress
# ---------------------------------------------------------------------------

def download_file(key: str, local_path: Path, size: int,
                  dry_run: bool, pbar_files: tqdm) -> str:
    if local_path.exists() and local_path.stat().st_size == size:
        pbar_files.update(1)
        return "skipped"

    if dry_run:
        pbar_files.update(1)
        return "dry-run"

    local_path.parent.mkdir(parents=True, exist_ok=True)
    client = make_client()

    try:
        with tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=local_path.name,
            leave=False,
        ) as pbar_bytes:
            client.download_file(
                BUCKET, key, str(local_path),
                Callback=lambda n: pbar_bytes.update(n),
            )
        pbar_files.update(1)
        return "downloaded"
    except Exception as e:
        pbar_files.update(1)
        return f"error:{e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download model from RunPod S3")
    parser.add_argument("--prefix",  default=DEFAULT_PREFIX,
                        help=f"S3 key prefix (default: {DEFAULT_PREFIX})")
    parser.add_argument("--dest",    default=DEFAULT_DEST,
                        help=f"Local destination dir (default: <pipeline>/output)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Parallel download workers (default: {MAX_WORKERS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without downloading")
    args = parser.parse_args()

    dest = Path(args.dest)
    prefix = args.prefix.rstrip("/")

    print(f"Bucket   : s3://{BUCKET}/{prefix}")
    print(f"Dest     : {dest}")
    print(f"Workers  : {args.workers}")
    print(f"Dry run  : {args.dry_run}")
    print()

    client = make_client()

    print("Listing objects...")
    objects = list_objects(client, prefix)
    if not objects:
        sys.exit(f"No objects found under s3://{BUCKET}/{prefix}")
    print(f"  {len(objects)} objects found\n")

    # Map S3 key → local path (strip prefix, keep subdirs)
    tasks = []
    for obj in objects:
        rel = obj["key"][len(prefix):].lstrip("/")
        local_path = dest / rel
        tasks.append((obj["key"], local_path, obj["size"]))

    pending  = [(k, p, s) for k, p, s in tasks if not (p.exists() and p.stat().st_size == s)]
    skipped  = len(tasks) - len(pending)
    total_bytes = sum(s for _, _, s in pending)

    print(f"{len(tasks)} files total | {len(pending)} to download | {skipped} already present")
    print(f"Download size: {total_bytes / 1e6:.1f} MB\n")

    if not pending:
        print("Nothing to download.")
        return

    if args.dry_run:
        for key, local_path, size in pending:
            print(f"  [dry-run] {key}  ({size / 1e6:.1f} MB)")
        print(f"\nDry run: {len(pending)} files would be downloaded.")
        return

    stats = {"downloaded": 0, "skipped": 0, "errors": 0, "bytes": 0}

    with tqdm(total=len(pending), unit="file", desc="Overall", dynamic_ncols=True) as pbar_files:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(download_file, key, local_path, size, False, pbar_files): (key, size)
                for key, local_path, size in pending
            }
            for future in as_completed(future_map):
                key, size = future_map[future]
                status = future.result()
                if status == "downloaded":
                    stats["downloaded"] += 1
                    stats["bytes"] += size
                elif status == "skipped":
                    stats["skipped"] += 1
                elif status.startswith("error"):
                    stats["errors"] += 1
                    tqdm.write(f"  [ERROR] {key}: {status}")
                pbar_files.set_postfix({
                    "↓": stats["downloaded"],
                    "err": stats["errors"],
                    "MB": f"{stats['bytes'] / 1e6:.0f}",
                })

    print()
    print("=" * 60)
    print(f"Downloaded : {stats['downloaded']} files ({stats['bytes'] / 1e6:.1f} MB)")
    print(f"Skipped    : {stats['skipped'] + skipped} files")
    print(f"Errors     : {stats['errors']} files")
    print(f"Dest       : {dest}")


if __name__ == "__main__":
    main()
