"""
Upload a trained model checkpoint directory to Google Drive.

Skips files already present with matching size (resume-safe).
Skips .log, .part, __pycache__ and hidden files.
Optionally logs the Drive folder link to an MLflow run.

Token : vlm-sft-pipeline/ssh/gdrive_token.pickle

Usage:
    # Upload default output dir, create/reuse folder named after checkpoint
    python vlm-sft-pipeline/fetch/upload_model.py \
        --source ./output/smolvlm2-500m-full-sft

    # Specify target Drive parent folder and MLflow run ID
    python vlm-sft-pipeline/fetch/upload_model.py \
        --source ./output/smolvlm2-500m-full-sft \
        --parent-id <drive-folder-id> \
        --mlflow-run-id <run-id>

    # Dry run
    python vlm-sft-pipeline/fetch/upload_model.py \
        --source ./output/smolvlm2-500m-full-sft --dry-run
"""

import argparse
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_HERE        = Path(__file__).parent.parent             # vlm-sft-pipeline/
TOKEN_PATH   = str(_HERE / "ssh" / "gdrive_token.pickle")

# Parent folder on Drive where model subfolders are created.
# Default: same research project folder used for dataset.
DEFAULT_PARENT_ID = "1kTUAV-rR3Rio81djO8WzS8mSG-lCHzJN"

CHUNK_SIZE   = 32 * 1024 * 1024   # 32 MB
MIME_FOLDER  = "application/vnd.google-apps.folder"
MAX_RETRIES  = 5
MAX_WORKERS  = 4

MLFLOW_URI        = os.environ.get("MLFLOW_URI",        "https://mlflow-geoai.stelarea.com/")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "smolvlm2-surveillance-sft")

# Files to skip (not needed for inference)
SKIP_SUFFIXES = {".log", ".part", ".tmp", ".pyc"}
SKIP_NAMES    = {"__pycache__", ".git", ".DS_Store"}


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
# Drive helpers
# ---------------------------------------------------------------------------

def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def get_or_create_folder(service, parent_id: str, name: str) -> str:
    resp = service.files().list(
        q=(f"'{parent_id}' in parents and name='{_escape(name)}' "
           f"and mimeType='{MIME_FOLDER}' and trashed=false"),
        fields="files(id)",
        pageSize=1,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    items = resp.get("files", [])
    if items:
        return items[0]["id"]
    meta = {"name": name, "mimeType": MIME_FOLDER, "parents": [parent_id]}
    return service.files().create(
        body=meta, fields="id", supportsAllDrives=True,
    ).execute()["id"]


def build_remote_index(service, folder_id: str) -> dict:
    """(parent_id, filename) → remote_size for all files under folder_id."""
    index = {}
    queue = [folder_id]
    with tqdm(desc="  Scanning Drive", unit="folder", dynamic_ncols=True) as pbar:
        while queue:
            fid = queue.pop()
            pbar.update(1)
            page_token = None
            while True:
                resp = service.files().list(
                    q=f"'{fid}' in parents and trashed=false",
                    fields="nextPageToken, files(id, name, mimeType, size, parents)",
                    pageToken=page_token,
                    pageSize=1000,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                ).execute()
                for f in resp.get("files", []):
                    if f["mimeType"] == MIME_FOLDER:
                        queue.append(f["id"])
                    else:
                        for p in f.get("parents", []):
                            index[(p, f["name"])] = int(f.get("size", 0))
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            pbar.set_postfix({"files": len(index)})
    return index


# ---------------------------------------------------------------------------
# Walk local dir — build (local_path, drive_parent_id) task list
# ---------------------------------------------------------------------------

class FolderCache:
    def __init__(self, service):
        self._svc   = service
        self._cache = {}
        self._lock  = threading.Lock()

    def get_or_create(self, parent_id: str, name: str) -> str:
        key = (parent_id, name)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        fid = get_or_create_folder(self._svc, parent_id, name)
        with self._lock:
            self._cache[key] = fid
        return fid


def _should_skip(path: Path) -> bool:
    if path.name.startswith(".") or path.name.startswith("._"):
        return True
    if path.name in SKIP_NAMES:
        return True
    if path.suffix in SKIP_SUFFIXES:
        return True
    return False


def walk_local(local_dir: Path, drive_parent_id: str,
               folder_cache: FolderCache, remote_index: dict) -> list[tuple]:
    tasks = []
    _walk(local_dir, drive_parent_id, folder_cache, remote_index, tasks)
    return tasks


def _walk(local_dir: Path, parent_id: str, folder_cache: FolderCache,
          remote_index: dict, tasks: list) -> None:
    try:
        entries = sorted(local_dir.iterdir())
    except PermissionError:
        return
    for entry in entries:
        if _should_skip(entry):
            continue
        if entry.is_dir():
            child_id = folder_cache.get_or_create(parent_id, entry.name)
            _walk(entry, child_id, folder_cache, remote_index, tasks)
        elif entry.is_file():
            tasks.append((entry, parent_id))


# ---------------------------------------------------------------------------
# Upload single file
# ---------------------------------------------------------------------------

def upload_file(credentials, local_path: Path, parent_id: str,
                remote_index: dict, index_lock: threading.Lock,
                dry_run: bool) -> str:
    name       = local_path.name
    local_size = local_path.stat().st_size
    key        = (parent_id, name)

    with index_lock:
        if remote_index.get(key) == local_size:
            return "skipped"
    if dry_run:
        return "dry-run"

    suffix   = local_path.suffix.lower()
    mime_map = {
        ".safetensors": "application/octet-stream",
        ".bin":         "application/octet-stream",
        ".json":        "application/json",
        ".model":       "application/octet-stream",
        ".txt":         "text/plain",
    }
    mime    = mime_map.get(suffix, "application/octet-stream")
    service = get_service(credentials)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            media   = MediaFileUpload(str(local_path), mimetype=mime,
                                      chunksize=CHUNK_SIZE, resumable=True)
            meta    = {"name": name, "parents": [parent_id]}
            request = service.files().create(
                body=meta, media_body=media, fields="id",
                supportsAllDrives=True,
            )
            response = None
            while response is None:
                _, response = request.next_chunk()
            with index_lock:
                remote_index[key] = local_size
            return "uploaded"

        except HttpError as e:
            if attempt == MAX_RETRIES or e.resp.status not in (429, 500, 502, 503, 504):
                return f"error:HTTP {e.resp.status} {name}"
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == MAX_RETRIES:
                return f"error:{e}"
            time.sleep(2 ** attempt)

    return "error:max_retries"


# ---------------------------------------------------------------------------
# Drive share link helper
# ---------------------------------------------------------------------------

def get_folder_link(folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}"


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def log_to_mlflow(run_id: str, folder_id: str, folder_name: str,
                  n_uploaded: int, total_bytes: int):
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_params({
                "gdrive_model_folder_id":   folder_id,
                "gdrive_model_folder_name": folder_name,
                "gdrive_model_link":        get_folder_link(folder_id),
            })
            mlflow.log_metrics({
                "upload_files":      n_uploaded,
                "upload_bytes":      total_bytes,
            })
        print(f"MLflow: Drive link logged to run {run_id}")
    except Exception as e:
        print(f"[WARN] MLflow logging failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Upload model checkpoint to Google Drive")
    parser.add_argument("--source",         required=True,
                        help="Local model directory (e.g. ./output/smolvlm2-500m-full-sft)")
    parser.add_argument("--parent-id",      default=DEFAULT_PARENT_ID,
                        help="Drive parent folder ID (default: project root folder)")
    parser.add_argument("--folder-name",    default=None,
                        help="Subfolder name on Drive (default: source dir name)")
    parser.add_argument("--token",          default=TOKEN_PATH,
                        help="Path to gdrive_token.pickle")
    parser.add_argument("--workers",        type=int, default=2,
                        help="Parallel upload workers (default: 2)")
    parser.add_argument("--mlflow-run-id",  default=None,
                        help="MLflow run ID to log Drive link to")
    parser.add_argument("--dry-run",        action="store_true",
                        help="List files without uploading")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        sys.exit(f"ERROR: source not found: {source}")

    folder_name = args.folder_name or source.name
    token_path  = args.token if os.path.isabs(args.token) else \
                  str(Path(__file__).parent / args.token)

    print(f"Source      : {source.resolve()}")
    print(f"Drive folder: {folder_name}  (under parent {args.parent_id})")
    print(f"Token       : {token_path}")
    print(f"Workers     : {args.workers}")
    print(f"Dry run     : {args.dry_run}")
    print()

    creds   = load_credentials(token_path)
    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    # Create/find the model subfolder on Drive
    print(f"Resolving Drive folder '{folder_name}'...")
    model_folder_id = get_or_create_folder(service, args.parent_id, folder_name)
    link = get_folder_link(model_folder_id)
    print(f"  Folder ID : {model_folder_id}")
    print(f"  Link      : {link}\n")

    # Build remote index
    print("Building remote file index...")
    remote_index      = build_remote_index(service, model_folder_id)
    remote_index_lock = threading.Lock()
    print(f"  {len(remote_index)} files already on Drive\n")

    # Walk local, create subfolders
    print("Scanning local checkpoint directory...")
    folder_cache = FolderCache(service)
    all_tasks    = walk_local(source, model_folder_id, folder_cache, remote_index)

    pending = [(p, pid) for p, pid in all_tasks
               if remote_index.get((pid, p.name)) != p.stat().st_size]
    skipped = len(all_tasks) - len(pending)
    total_bytes = sum(p.stat().st_size for p, _ in pending)

    print(f"  {len(all_tasks)} files total | {len(pending)} to upload | {skipped} already present")
    print(f"  Upload size: {total_bytes / 1e6:.1f} MB\n")

    if not pending:
        print("Nothing to upload.")
        print(f"Drive link: {link}")
        if args.mlflow_run_id:
            log_to_mlflow(args.mlflow_run_id, model_folder_id, folder_name, 0, 0)
        return

    if args.dry_run:
        for path, _ in pending:
            print(f"  [dry-run] {path.relative_to(source)}  ({path.stat().st_size / 1e6:.1f} MB)")
        print(f"\nDry run: {len(pending)} files would be uploaded.")
        print(f"Drive link: {link}")
        return

    # Parallel upload
    stats = {"uploaded": 0, "skipped": 0, "errors": 0, "bytes": 0}
    print(f"Uploading {len(pending)} files with {args.workers} workers...\n")

    with tqdm(total=len(pending), unit="file", dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    upload_file, creds, path, parent_id,
                    remote_index, remote_index_lock, False,
                ): path
                for path, parent_id in pending
            }
            for future in as_completed(future_map):
                path   = future_map[future]
                status = future.result()
                if status == "uploaded":
                    stats["uploaded"] += 1
                    stats["bytes"]    += path.stat().st_size
                elif status == "skipped":
                    stats["skipped"] += 1
                elif status.startswith("error"):
                    stats["errors"] += 1
                    tqdm.write(f"  [ERROR] {path.name}: {status}")
                pbar.set_postfix({
                    "↑": stats["uploaded"],
                    "skip": stats["skipped"],
                    "err": stats["errors"],
                    "MB": f"{stats['bytes'] / 1e6:.0f}",
                })
                pbar.update(1)

    print()
    print("=" * 60)
    print(f"Uploaded  : {stats['uploaded']} files ({stats['bytes'] / 1e6:.1f} MB)")
    print(f"Skipped   : {stats['skipped'] + skipped} files")
    print(f"Errors    : {stats['errors']} files")
    print(f"Drive link: {link}")

    if args.mlflow_run_id:
        log_to_mlflow(args.mlflow_run_id, model_folder_id, folder_name,
                      stats["uploaded"], stats["bytes"])


if __name__ == "__main__":
    main()
