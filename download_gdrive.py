"""
Download dataset from Google Drive using OAuth token pickle.

Source : https://drive.google.com/open?id=1kTUAV-rR3Rio81djO8WzS8mSG-lCHzJN
Token  : vlm-sft-pipeline/ssh/gdrive_token.pickle

Skips files already present with matching size (resume-safe).
Downloads with 32 MB chunks. Retries on transient errors.

Usage:
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/download_gdrive.py
    python vlm-sft-pipeline/download_gdrive.py --out /path/to/data
    python vlm-sft-pipeline/download_gdrive.py --out /path/to/data --dry-run
"""

import argparse
import io
import os
import pickle
import sys
import time
from pathlib import Path

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

FOLDER_ID   = "1kTUAV-rR3Rio81djO8WzS8mSG-lCHzJN"
MIME_FOLDER = "application/vnd.google-apps.folder"
CHUNK_SIZE  = 32 * 1024 * 1024   # 32 MB
MAX_RETRIES = 5

_HERE       = Path(__file__).parent
TOKEN_PATH  = str(_HERE / "ssh" / "gdrive_token.pickle")
DEFAULT_OUT = os.environ.get("DATA_ROOT", "./data")


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
# Drive helpers
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


def download_file(service, file_id: str, dest: Path,
                  remote_size: int, dry_run: bool) -> str:
    """Returns: 'downloaded' | 'skipped' | 'dry-run' | 'error:<msg>'"""
    if dest.exists():
        if remote_size and dest.stat().st_size == remote_size:
            return "skipped"

    if dry_run:
        return "dry-run"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request    = service.files().get_media(fileId=file_id,
                                                   supportsAllDrives=True)
            fh         = io.FileIO(str(tmp), mode="wb")
            downloader = MediaIoBaseDownload(fh, request, chunksize=CHUNK_SIZE)

            with tqdm(
                total=remote_size or 0,
                unit="B", unit_scale=True,
                desc=f"  {dest.name}",
                leave=False, dynamic_ncols=True,
            ) as pbar:
                done = False
                prev = 0
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        cur = int(status.resumable_progress)
                        pbar.update(cur - prev)
                        prev = cur

            fh.close()
            tmp.rename(dest)
            return "downloaded"

        except HttpError as e:
            fh.close()
            tmp.unlink(missing_ok=True)
            if attempt == MAX_RETRIES or e.resp.status not in (429, 500, 502, 503, 504):
                return f"error:HTTP {e.resp.status}"
            time.sleep(2 ** attempt)
        except Exception as e:
            fh.close()
            tmp.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                return f"error:{e}"
            time.sleep(2 ** attempt)

    return "error:max_retries"


# ---------------------------------------------------------------------------
# Recursive download
# ---------------------------------------------------------------------------

def download_folder(service, folder_id: str, local_dir: Path,
                    dry_run: bool, stats: dict, pbar: tqdm, depth: int = 0) -> None:
    items   = list_folder(service, folder_id)
    folders = sorted([i for i in items if i["mimeType"] == MIME_FOLDER], key=lambda x: x["name"])
    files   = sorted([i for i in items if i["mimeType"] != MIME_FOLDER],  key=lambda x: x["name"])

    for folder in folders:
        local_dir.mkdir(parents=True, exist_ok=True)
        download_folder(service, folder["id"], local_dir / folder["name"],
                        dry_run, stats, pbar, depth + 1)

    for f in files:
        remote_size = int(f.get("size", 0))
        dest        = local_dir / f["name"]

        status = download_file(service, f["id"], dest, remote_size, dry_run)

        if status == "downloaded":
            stats["downloaded"] += 1
            stats["bytes"]      += remote_size
        elif status == "skipped":
            stats["skipped"] += 1
        elif status == "dry-run":
            stats["dry_run"] += 1
            mb = remote_size / 1e6
            tqdm.write(f"  [dry-run] {dest}  ({mb:.1f} MB)")
        else:
            stats["errors"] += 1
            tqdm.write(f"  [ERROR] {f['name']}: {status}")

        pbar.set_postfix({"↓": stats["downloaded"], "skip": stats["skipped"]})
        pbar.update(1)


# ---------------------------------------------------------------------------
# Count remote files (for progress bar total)
# ---------------------------------------------------------------------------

def count_remote_files(service, folder_id: str) -> int:
    total = 0
    queue = [folder_id]
    with tqdm(desc="  Counting remote files", unit="folder", dynamic_ncols=True) as pbar:
        while queue:
            fid   = queue.pop()
            items = list_folder(service, fid)
            for i in items:
                if i["mimeType"] == MIME_FOLDER:
                    queue.append(i["id"])
                else:
                    total += 1
            pbar.update(1)
            pbar.set_postfix({"files": total})
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download dataset from Google Drive")
    parser.add_argument("--out",       default=DEFAULT_OUT,  help="Local output directory")
    parser.add_argument("--token",     default=TOKEN_PATH,   help="Path to token.pickle")
    parser.add_argument("--folder-id", default=FOLDER_ID,    help="Drive folder ID")
    parser.add_argument("--dry-run",   action="store_true",  help="List files without downloading")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output    : {out_dir.resolve()}")
    print(f"Folder ID : {args.folder_id}")
    print(f"Token     : {args.token}")
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

    print("Counting remote files...")
    total = count_remote_files(service, args.folder_id)
    print(f"  {total} files in Drive folder\n")

    stats = {"downloaded": 0, "skipped": 0, "errors": 0, "dry_run": 0, "bytes": 0}

    with tqdm(total=total, unit="file", desc="Downloading", dynamic_ncols=True) as pbar:
        download_folder(service, args.folder_id, out_dir, args.dry_run, stats, pbar)

    print()
    print("=" * 50)
    if args.dry_run:
        print(f"Dry run  : {stats['dry_run']} files would be downloaded")
    else:
        print(f"Downloaded : {stats['downloaded']} files ({stats['bytes'] / 1e9:.2f} GB)")
        print(f"Skipped    : {stats['skipped']} files (already present)")
        print(f"Errors     : {stats['errors']} files")
    print(f"Output     : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
