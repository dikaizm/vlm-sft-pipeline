"""
Download dataset from Google Drive using OAuth token pickle (parallel).

Source  : https://drive.google.com/open?id=1kTUAV-rR3Rio81djO8WzS8mSG-lCHzJN
Token   : vlm-sft-pipeline/ssh/gdrive_token.pickle
Workers : 5 parallel downloads (override with --workers N)

Stages:
  1. Walk Drive folder tree, create local dirs, build flat task list
  2. Filter tasks: skip files already present with matching size
  3. Download remaining files in parallel via ThreadPoolExecutor

Skips files already present with matching size (resume-safe).
Downloads with 32 MB chunks. Retries on transient errors.

Usage:
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/download_gdrive.py
    python vlm-sft-pipeline/download_gdrive.py --out /path/to/data
    python vlm-sft-pipeline/download_gdrive.py --out /path/to/data --dry-run
    python vlm-sft-pipeline/download_gdrive.py --out /path/to/data --workers 3
"""

import argparse
import io
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FOLDER_ID       = "1kTUAV-rR3Rio81djO8WzS8mSG-lCHzJN"
MIME_FOLDER     = "application/vnd.google-apps.folder"
CHUNK_SIZE      = 32 * 1024 * 1024   # 32 MB
MAX_RETRIES     = 5
DEFAULT_WORKERS = 5

_HERE       = Path(__file__).parent.parent   # pipeline root
TOKEN_PATH  = str(_HERE / "ssh" / "gdrive_token.pickle")

from config import DATA_ROOT
DEFAULT_OUT = DATA_ROOT


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def load_credentials(token_path: str):
    if not os.path.isfile(token_path):
        sys.exit(
            f"ERROR: token not found: {token_path}\n"
            "Place gdrive_token.pickle at vlm-sft-pipeline/ssh/gdrive_token.pickle"
        )
    with open(token_path, "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        print("Refreshing expired token...")
        creds.refresh(Request())
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)
    if not creds.valid:
        sys.exit("ERROR: token invalid and cannot be refreshed. Re-authenticate.")
    return creds


# ---------------------------------------------------------------------------
# Per-thread Drive service
# ---------------------------------------------------------------------------

_thread_local = threading.local()

def get_service(credentials):
    if not hasattr(_thread_local, "service"):
        _thread_local.service = build("drive", "v3", credentials=credentials,
                                      cache_discovery=False)
    return _thread_local.service


# ---------------------------------------------------------------------------
# Drive folder listing
# ---------------------------------------------------------------------------

def list_folder(service, folder_id: str) -> list[dict]:
    items      = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageToken=page_token,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


# ---------------------------------------------------------------------------
# Build flat task list — walk Drive tree, create local dirs
# ---------------------------------------------------------------------------

def build_download_tasks(service, folder_id: str, local_dir: Path) -> list[dict]:
    """
    Recursively walk Drive folder. Create local dirs as needed.
    Returns flat list of dicts: {file_id, name, size, dest}.
    """
    tasks = []
    _walk_drive(service, folder_id, local_dir, tasks)
    return tasks


def _walk_drive(service, folder_id: str, local_dir: Path, tasks: list) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    items   = list_folder(service, folder_id)
    folders = sorted([i for i in items if i["mimeType"] == MIME_FOLDER], key=lambda x: x["name"])
    files   = sorted([i for i in items if i["mimeType"] != MIME_FOLDER],  key=lambda x: x["name"])

    for folder in folders:
        _walk_drive(service, folder["id"], local_dir / folder["name"], tasks)

    for f in files:
        tasks.append({
            "file_id": f["id"],
            "name":    f["name"],
            "size":    int(f.get("size", 0)),
            "dest":    local_dir / f["name"],
        })


# ---------------------------------------------------------------------------
# Download single file
# ---------------------------------------------------------------------------

def download_file(credentials, task: dict, dry_run: bool) -> str:
    """Returns: 'downloaded' | 'skipped' | 'dry-run' | 'error:<msg>'"""
    dest        = task["dest"]
    remote_size = task["size"]

    if dest.exists() and remote_size and dest.stat().st_size == remote_size:
        return "skipped"

    if dry_run:
        return "dry-run"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp     = dest.with_suffix(dest.suffix + ".part")
    service = get_service(credentials)

    for attempt in range(1, MAX_RETRIES + 1):
        fh = None
        try:
            request    = service.files().get_media(fileId=task["file_id"],
                                                   supportsAllDrives=True)
            fh         = io.FileIO(str(tmp), mode="wb")
            downloader = MediaIoBaseDownload(fh, request, chunksize=CHUNK_SIZE)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            fh.close()
            tmp.rename(dest)
            return "downloaded"

        except HttpError as e:
            if fh:
                fh.close()
            tmp.unlink(missing_ok=True)
            if attempt == MAX_RETRIES or e.resp.status not in (429, 500, 502, 503, 504):
                return f"error:HTTP {e.resp.status}"
            time.sleep(2 ** attempt)
        except Exception as e:
            if fh:
                fh.close()
            tmp.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                return f"error:{e}"
            time.sleep(2 ** attempt)

    return "error:max_retries"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download dataset from Google Drive in parallel")
    parser.add_argument("--out",       default=DEFAULT_OUT,       help="Local output directory")
    parser.add_argument("--token",     default=TOKEN_PATH,        help="Path to token.pickle")
    parser.add_argument("--folder-id", default=FOLDER_ID,         help="Drive folder ID")
    parser.add_argument("--workers",   type=int, default=DEFAULT_WORKERS, help="Parallel download workers (default: 5)")
    parser.add_argument("--dry-run",   action="store_true",       help="List files without downloading")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output    : {out_dir.resolve()}")
    print(f"Folder ID : {args.folder_id}")
    print(f"Token     : {args.token}")
    print(f"Workers   : {args.workers}")
    print(f"Dry run   : {args.dry_run}")
    print()

    creds   = load_credentials(args.token)
    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    try:
        meta = service.files().get(
            fileId=args.folder_id, fields="id, name", supportsAllDrives=True,
        ).execute()
        print(f"Connected to folder: {meta['name']}\n")
    except HttpError as e:
        sys.exit(f"ERROR: cannot access folder {args.folder_id}: {e}")

    # Stage 1: walk Drive tree, build task list
    print("Walking Drive folder tree and building task list...")
    with tqdm(desc="  Scanning", unit="folder", dynamic_ncols=True) as _:
        tasks = build_download_tasks(service, args.folder_id, out_dir)
    print(f"  {len(tasks)} files found on Drive\n")

    # Stage 2: filter already-present files
    pending = [t for t in tasks
               if not (t["dest"].exists() and t["size"] and
                       t["dest"].stat().st_size == t["size"])]
    skipped_upfront = len(tasks) - len(pending)
    total_bytes     = sum(t["size"] for t in pending)

    print(f"  To download : {len(pending)} files ({total_bytes / 1e9:.2f} GB)")
    print(f"  Already present : {skipped_upfront} files\n")

    if not pending:
        print("Nothing to download.")
        return

    if args.dry_run:
        for t in pending:
            print(f"  [dry-run] {t['dest'].relative_to(out_dir)} ({t['size'] / 1e6:.1f} MB)")
        print(f"\nDry run: {len(pending)} files would be downloaded.")
        return

    # Stage 3: parallel download
    stats      = {"downloaded": 0, "skipped": 0, "errors": 0, "bytes": 0}
    stats_lock = threading.Lock()

    print(f"Downloading {len(pending)} files with {args.workers} parallel workers...\n")

    with tqdm(total=len(pending), unit="file", desc="Downloading", dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(download_file, creds, task, False): task
                for task in pending
            }

            for future in as_completed(future_map):
                task   = future_map[future]
                status = future.result()

                with stats_lock:
                    if status == "downloaded":
                        stats["downloaded"] += 1
                        stats["bytes"]      += task["size"]
                    elif status == "skipped":
                        stats["skipped"] += 1
                    elif status.startswith("error"):
                        stats["errors"] += 1
                        tqdm.write(f"  [ERROR] {task['name']}: {status}")

                pbar.set_postfix({
                    "↓": stats["downloaded"],
                    "skip": stats["skipped"],
                    "err": stats["errors"],
                    "GB": f"{stats['bytes'] / 1e9:.2f}",
                })
                pbar.update(1)

    print()
    print("=" * 50)
    print(f"Downloaded : {stats['downloaded']} files ({stats['bytes'] / 1e9:.2f} GB)")
    print(f"Skipped    : {stats['skipped'] + skipped_upfront} files (already present)")
    print(f"Errors     : {stats['errors']} files")
    print(f"Output     : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
