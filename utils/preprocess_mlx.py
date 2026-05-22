"""
Preprocess UCF-Crime videos into HuggingFace dataset for mlx-vlm fine-tuning.

Extracts 16 frames per dense window (120s), formats as multi-image chat messages
with temporal ground-truth annotations.

Output: dataset pushed to HF Hub or saved locally as Parquet.

Usage:
    DATA_ROOT=/Volumes/T7/research-vlm/data python vlm-sft-pipeline/preprocess_mlx.py
    DATA_ROOT=/Volumes/T7/research-vlm/data python vlm-sft-pipeline/preprocess_mlx.py --max-videos 200
"""

import argparse
import json
import os
import random
import re
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset, DatasetDict
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT       = os.environ.get("DATA_ROOT", "/Volumes/T7/research-vlm/data")
VIDEO_ROOT      = f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos"
TRAIN_JSON      = f"{DATA_ROOT}/UCFCrime_Train.json"
VAL_JSON        = f"{DATA_ROOT}/UCFCrime_Val.json"
TEST_JSON       = f"{DATA_ROOT}/UCFCrime_Test.json"

NUM_FRAMES      = 16
MAX_DURATION    = 120.0          # seconds per window
MAX_ANNOTATIONS = 12             # cap annotations per video
MAX_TRAIN       = 200            # videos (not clips)
MAX_VAL         = 50
SEED            = 42

DENSE_PROMPT = (
    "Describe ALL activities in this surveillance video. "
    "For each activity, provide a description and its start and end timestamps in seconds. "
    "List them in chronological order."
)

OUTPUT_REPO = os.environ.get("HF_DATASET_REPO", "smolvlm-surveillance-dense")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _category_from_id(video_id: str) -> str:
    return re.sub(r"\d+_x264$", "", video_id)


def extract_frames(video_path: str, start: float, end: float, n_frames: int) -> list:
    """Extract n_frames uniformly from video segment [start, end]."""
    try:
        import av
        container = av.open(video_path)
        stream    = container.streams.video[0]
        duration  = float(stream.duration * stream.time_base) if stream.duration else end

        t_start = max(0.0, min(start, duration))
        t_end   = max(t_start + 0.1, min(end, duration))

        collected = OrderedDict()
        container.seek(int(t_start * 1_000_000), any_frame=False, backward=True)

        for frame in container.decode(video=0):
            t = float(frame.pts * stream.time_base)
            if t > t_end + 1.0:
                break
            slot = int((t - t_start) / (t_end - t_start + 1e-9) * n_frames)
            slot = max(0, min(slot, n_frames - 1))
            if slot not in collected:
                collected[slot] = frame.to_image()
            if len(collected) >= n_frames:
                break
        container.close()

        if collected:
            for i in range(n_frames):
                if i not in collected:
                    collected[i] = collected[min(collected.keys(), key=lambda k: abs(k - i))]
            return [collected[i] for i in range(n_frames)]
    except Exception:
        pass

    return [Image.new("RGB", (224, 224), color=0)] * n_frames


def load_video_samples(json_path: str, max_videos: int) -> list[dict]:
    """Load videos with all annotations grouped."""
    with open(json_path) as f:
        data = json.load(f)

    samples = []
    for video_id, ann in data.items():
        category   = _category_from_id(video_id)
        video_path = os.path.join(VIDEO_ROOT, category, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            continue

        duration      = float(ann.get("duration", MAX_DURATION))
        effective_end = min(duration, MAX_DURATION)

        pairs = []
        for (start, end), sentence in zip(ann["timestamps"], ann["sentences"]):
            start, end = float(start), float(end)
            if end <= start or start > effective_end:
                continue
            end = min(end, effective_end)
            pairs.append((start, end, sentence.strip()))

        if not pairs:
            continue

        pairs.sort(key=lambda x: (x[0], x[1]))
        pairs = pairs[:MAX_ANNOTATIONS]

        samples.append({
            "video_id":      video_id,
            "video_path":    video_path,
            "effective_end": effective_end,
            "timestamps":    [[s, e] for s, e, _ in pairs],
            "sentences":     [sent for _, _, sent in pairs],
        })

    random.seed(SEED)
    random.shuffle(samples)
    return samples if max_videos == -1 else samples[:max_videos]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_mlx_dataset(json_path: str, max_videos: int) -> list[dict]:
    """Convert UCF-Crime videos into mlx-vlm LoRA format:
    {
        "images": [PIL.Image, ...],   # 16 frames
        "messages": [
            {"role": "user",     "content": [{"type":"image"}*16 + {"type":"text", "text": prompt}]},
            {"role": "assistant","content": [{"type":"text", "text": response}]},
        ]
    }
    """
    samples = load_video_samples(json_path, max_videos)
    dataset_rows = []

    print(f"Processing {len(samples)} videos...")
    skipped = 0

    for idx, s in enumerate(samples):
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(samples)}")

        # Extract frames
        frames = extract_frames(s["video_path"], 0.0, s["effective_end"], NUM_FRAMES)

        # Skip if all black (extraction failure)
        if all(f.getextrema() == (0, 0) for f in frames):
            skipped += 1
            continue

        # Build numbered-list response
        lines = []
        for i, (ts, sent) in enumerate(zip(s["timestamps"], s["sentences"]), 1):
            lines.append(f"{i}. [{ts[0]:.1f}, {ts[1]:.1f}] {sent}")
        response = "\n".join(lines)

        # Resize all frames to 512x512 (SmolVLM native resolution)
        resized_frames = [f.resize((512, 512), Image.BICUBIC) for f in frames]

        # Build messages: one {"type": "image"} per frame + text prompt
        user_content = []
        for _ in range(NUM_FRAMES):
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": DENSE_PROMPT})

        messages = [
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]

        dataset_rows.append({
            "images":   resized_frames,     # list of 16 PIL Images @ 512x512
            "messages": messages,
        })

    if skipped:
        print(f"  Skipped {skipped} videos (frame extraction failed)")

    return dataset_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=MAX_TRAIN)
    parser.add_argument("--max-val",   type=int, default=MAX_VAL)
    parser.add_argument("--output",    default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo",  default=OUTPUT_REPO)
    args = parser.parse_args()

    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"TRAIN_JSON: {TRAIN_JSON}")
    print(f"VAL_JSON: {VAL_JSON}")

    # --- Train ---
    print("\n=== Building train dataset ===")
    train_rows = build_mlx_dataset(TRAIN_JSON, args.max_train)
    print(f"Train: {len(train_rows)} videos")

    # --- Val ---
    print("\n=== Building val dataset ===")
    val_rows = build_mlx_dataset(VAL_JSON, args.max_val)
    print(f"Val:   {len(val_rows)} videos")

    # --- Create HuggingFace datasets ---
    # We store image paths instead of PIL objects when saving; use Dataset.from_list
    # for PIL-image datasets.
    train_ds = Dataset.from_list(train_rows)
    val_ds   = Dataset.from_list(val_rows)

    dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    # --- Save locally as Parquet (compatible with load_dataset) ---
    output_path = args.output or f"./output/surveillance-dense-dataset"
    os.makedirs(output_path, exist_ok=True)

    # Save each split as a Parquet directory
    for split_name, split_ds in dataset_dict.items():
        split_dir = os.path.join(output_path, split_name)
        os.makedirs(split_dir, exist_ok=True)
        split_ds.to_parquet(os.path.join(split_dir, "data.parquet"))

    print(f"\nDataset saved to: {output_path} (Parquet format)")

    # --- Push to hub ---
    if args.push_to_hub:
        dataset_dict.push_to_hub(args.hub_repo)
        print(f"Dataset pushed to: {args.hub_repo}")

    print("\nDone.")


if __name__ == "__main__":
    main()
