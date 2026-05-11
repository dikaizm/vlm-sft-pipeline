"""
MLX Inference with fine-tuned SmolVLM for surveillance dense captioning.

Loads a base model + adapter (or fine-tuned model) and runs inference on
surveillance videos — single-pass or sliding window.

Usage:
    # Single video
    DATA_ROOT=/Volumes/T7/research-vlm/data \
    python vlm-sft-pipeline/infer_mlx.py --video /path/to/video.mp4

    # With adapter
    python vlm-sft-pipeline/infer_mlx.py --adapter ./output/adapters.safetensors --video video.mp4

    # Batch on test set
    python vlm-sft-pipeline/infer_mlx.py --test-json $DATA_ROOT/UCFCrime_Test.json --n 5
"""

import argparse
import json
import os
import random
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT     = os.environ.get("DATA_ROOT", "/Volumes/T7/research-vlm/data")
VIDEO_ROOT    = f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos"
TEST_JSON     = f"{DATA_ROOT}/UCFCrime_Test.json"

MODEL_PATH    = "mlx-community/SmolVLM-256M-Instruct-bf16"
OUTPUT_DIR    = "./output/smolvlm-256m-mlx-sft"

NUM_FRAMES      = 16
MAX_DURATION    = 120.0
WINDOW_SIZE     = 120.0
WINDOW_STRIDE   = 60.0
MAX_NEW_TOKENS  = 512
SEED            = 99

DENSE_PROMPT = (
    "Describe ALL activities in this surveillance video. "
    "For each activity, provide a description and its start and end timestamps in seconds. "
    "List them in chronological order."
)

WINDOW_PROMPT_TEMPLATE = (
    "Describe ALL activities visible in this video segment. "
    "Timestamps are relative to {offset:.0f}s of the full video. "
    "For each activity, provide a description and its start and end timestamps in seconds. "
    "List them in chronological order."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _category_from_id(video_id: str) -> str:
    return re.sub(r"\d+_x264$", "", video_id)


def get_video_duration(video_path: str) -> float:
    try:
        import av
        container = av.open(video_path)
        stream    = container.streams.video[0]
        duration  = float(stream.duration * stream.time_base) if stream.duration else 0.0
        container.close()
        return duration
    except Exception:
        return 0.0


def extract_frames(video_path: str, start: float, end: float, n_frames: int) -> list:
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
    except Exception as e:
        print(f"  [WARN] frame extraction failed: {e}")

    return [Image.new("RGB", (224, 224), color=0)] * n_frames


def parse_dense_output(text: str) -> list[dict]:
    pattern = r"\d+\.\s*\[(\d+\.?\d*),\s*(\d+\.?\d*)\]\s*(.+)"
    activities = []
    for m in re.finditer(pattern, text):
        activities.append({
            "start":       float(m.group(1)),
            "end":         float(m.group(2)),
            "description": m.group(3).strip(),
        })
    return activities


def tiou(pred_start, pred_end, gt_start, gt_end) -> float:
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = (pred_end - pred_start) + (gt_end - gt_start) - inter
    return inter / union if union > 0 else 0.0


def deduplicate_activities(activities: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    if not activities:
        return []
    activities = sorted(activities, key=lambda x: x["start"])
    kept = []
    for act in activities:
        duplicate = False
        for existing in kept:
            if tiou(act["start"], act["end"], existing["start"], existing["end"]) > iou_threshold:
                if len(act["description"]) > len(existing["description"]):
                    existing.update(act)
                duplicate = True
                break
        if not duplicate:
            kept.append(dict(act))
    return kept


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, processor, config, frames: list, prompt: str) -> str:
    """Run multi-image inference with MLX-VLM."""
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm import generate as mlx_generate

    # Build messages: one {"type": "image"} per frame
    user_content = []
    for _ in range(len(frames)):
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "user", "content": user_content}
    ]

    formatted = apply_chat_template(
        processor, config, prompt, num_images=len(frames)
    )

    # mlx-vlm generate expects images as a list
    output = mlx_generate(
        model, processor, formatted, frames,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        verbose=False,
    )

    return output.strip()


def infer_single_pass(model, processor, config, video_path: str) -> tuple[list[dict], str]:
    duration      = get_video_duration(video_path)
    effective_end = min(duration, MAX_DURATION) if duration > 0 else MAX_DURATION
    frames        = extract_frames(video_path, 0.0, effective_end, NUM_FRAMES)
    raw_text      = run_inference(model, processor, config, frames, DENSE_PROMPT)
    return parse_dense_output(raw_text), raw_text


def infer_sliding_window(model, processor, config, video_path: str) -> tuple[list[dict], list[dict]]:
    duration = get_video_duration(video_path)
    if duration <= 0:
        activities, _ = infer_single_pass(model, processor, config, video_path)
        return activities, []

    all_activities = []
    window_records = []
    t = 0.0

    while t < duration:
        w_start = t
        w_end   = min(t + WINDOW_SIZE, duration)
        frames  = extract_frames(video_path, w_start, w_end, NUM_FRAMES)
        prompt  = WINDOW_PROMPT_TEMPLATE.format(offset=w_start)
        raw     = run_inference(model, processor, config, frames, prompt)

        acts = parse_dense_output(raw)
        for a in acts:
            a["start"] += w_start
            a["end"]   += w_start

        window_records.append({
            "window_start": w_start,
            "window_end":   w_end,
            "raw_output":   raw,
            "activities":   acts,
        })
        all_activities.extend(acts)

        t += WINDOW_STRIDE
        if w_end >= duration:
            break

    deduped = deduplicate_activities(all_activities)
    return deduped, window_records


# ---------------------------------------------------------------------------
# Test set loader
# ---------------------------------------------------------------------------

def load_test_samples(test_json: str, n: int) -> list[dict]:
    with open(test_json) as f:
        data = json.load(f)

    items = []
    for video_id, ann in data.items():
        category   = _category_from_id(video_id)
        video_path = os.path.join(VIDEO_ROOT, category, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            continue

        pairs = []
        for (start, end), sentence in zip(ann["timestamps"], ann["sentences"]):
            start, end = float(start), float(end)
            if end > start:
                pairs.append([start, end, sentence.strip()])
        if not pairs:
            continue

        items.append({
            "video_id":   video_id,
            "video_path": video_path,
            "duration":   float(ann.get("duration", MAX_DURATION)),
            "gt":         pairs,
        })

    random.seed(SEED)
    random.shuffle(items)
    return items[:n] if n > 0 else items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--video",     help="Single video file path")
    mode.add_argument("--test-json", help="Path to UCFCrime_Test.json for batch eval")

    parser.add_argument("--model",     default=MODEL_PATH)
    parser.add_argument("--adapter",   default=None)
    parser.add_argument("--sliding-window", action="store_true")
    parser.add_argument("--n",         type=int, default=5)
    parser.add_argument("--output",    default=None)
    args = parser.parse_args()

    print(f"Model: {args.model}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")
    print()

    # --- Load model ---
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model, processor = load(args.model, adapter_path=args.adapter)
    config = load_config(args.model)
    print(f"Model loaded. Config type: {type(config).__name__}")

    sep = "=" * 72
    video_results = []

    # --- Mode A: single video ---
    if args.video:
        video_path = args.video
        duration = get_video_duration(video_path)
        print(f"Video: {video_path}")
        print(f"Duration: {duration:.1f}s\n")

        if args.sliding_window:
            activities, windows = infer_sliding_window(model, processor, config, video_path)
        else:
            activities, raw = infer_single_pass(model, processor, config, video_path)
            windows = []

        print(sep)
        print(f"Detected {len(activities)} activities:")
        for a in activities:
            print(f"  [{a['start']:.1f}, {a['end']:.1f}] {a['description']}")
        print(sep)

        video_results.append({
            "video_path": video_path,
            "duration":   duration,
            "mode":       "sliding_window" if args.sliding_window else "single_pass",
            "activities": activities,
            "windows":    windows if args.sliding_window else [],
        })

    # --- Mode B: batch on test set ---
    else:
        samples = load_test_samples(args.test_json, args.n)
        if not samples:
            sys.exit("No test samples found")
        print(f"Loaded {len(samples)} test videos\n")

        for i, s in enumerate(samples, 1):
            print(sep)
            print(f"[{i}/{len(samples)}] {s['video_id']}")

            if args.sliding_window:
                activities, windows = infer_sliding_window(model, processor, config, s["video_path"])
                raw_output = None
            else:
                activities, raw_output = infer_single_pass(model, processor, config, s["video_path"])
                windows = []

            print(f"  Predicted: {len(activities)} activities")
            for a in activities:
                print(f"    [{a['start']:.1f}, {a['end']:.1f}] {a['description']}")
            if raw_output is not None:
                print(f"  Raw: {raw_output[:200]!r}")
            print()

            video_results.append({
                "video_id":   s["video_id"],
                "duration":   s["duration"],
                "gt":         s["gt"],
                "predicted":  activities,
                "raw_output": raw_output,
                "mode":       "sliding_window" if args.sliding_window else "single_pass",
            })

    print(sep)
    print("Done.")

    # --- Save results ---
    run_name = f"infer-mlx-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_path = Path(args.output) if args.output else \
               Path(OUTPUT_DIR) / "results" / f"{run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "run_name":    run_name,
        "model":       args.model,
        "adapter":     args.adapter,
        "mode":        "sliding_window" if args.sliding_window else "single_pass",
        "n_videos":    len(video_results),
        "num_frames":  NUM_FRAMES,
        "videos":      video_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
