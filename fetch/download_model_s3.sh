#!/usr/bin/env bash
# Download trained model checkpoint(s) from RunPod S3 volume.
#
# Usage:
#   ./fetch/download_model_s3.sh
#   ./fetch/download_model_s3.sh --prefix output/smolvlm2-500m-full-sft/checkpoint-1072
#   ./fetch/download_model_s3.sh --prefix output/ --dest /Volumes/T7/research-vlm/output
#   ./fetch/download_model_s3.sh --dry-run

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BUCKET="s3://z3oqdvyui7"
REGION="eu-ro-1"
ENDPOINT="https://s3api-eu-ro-1.runpod.io"

# Default: download all output/ into local output/ inside pipeline root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"

S3_PREFIX="vlm-sft-pipeline/output"
LOCAL_DEST="$PIPELINE_ROOT/output"
DRY_RUN=false

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)   S3_PREFIX="$2"; shift 2 ;;
        --dest)     LOCAL_DEST="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--prefix S3_PREFIX] [--dest LOCAL_DIR] [--dry-run]"
            echo "  --prefix   S3 key prefix to download (default: output)"
            echo "  --dest     Local destination directory (default: <pipeline>/output)"
            echo "  --dry-run  List files without downloading"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

AWS_ARGS=(
    --region "$REGION"
    --endpoint-url "$ENDPOINT"
)

echo "Bucket   : $BUCKET/$S3_PREFIX"
echo "Dest     : $LOCAL_DEST"
echo "Dry run  : $DRY_RUN"
echo ""

# ---------------------------------------------------------------------------
# List first so user can see what's there
# ---------------------------------------------------------------------------
echo "=== Listing $BUCKET/$S3_PREFIX ==="
# --recursive breaks RunPod S3 pagination; list top-level prefixes only
aws s3 ls "${AWS_ARGS[@]}" "$BUCKET/$S3_PREFIX/"
echo ""

if $DRY_RUN; then
    echo "Dry run — no files downloaded."
    exit 0
fi

# ---------------------------------------------------------------------------
# Sync (resume-safe: skips files already present with matching size/etag)
# ---------------------------------------------------------------------------
mkdir -p "$LOCAL_DEST"

echo "=== Syncing to $LOCAL_DEST ==="
aws s3 sync "${AWS_ARGS[@]}" \
    "$BUCKET/$S3_PREFIX" \
    "$LOCAL_DEST" \
    --exclude "*.log" \
    --exclude "*.part" \
    --exclude "*.tmp" \
    --exclude "__pycache__/*" \
    --no-progress

echo ""
echo "Done. Files saved to: $LOCAL_DEST"
