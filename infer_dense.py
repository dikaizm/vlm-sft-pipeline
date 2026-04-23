"""
Dense video captioning inference — find ALL activities in a raw surveillance video.

Unlike infer.py (which requires pre-segmented clips), this script runs the model
on a full video and outputs a numbered list of all detected activities with timestamps:

    1. [0.0, 5.3] A woman walks across the parking lot.
    2. [7.0, 8.5] A man in white pushes another person.
    ...

Two modes:
  - Single-pass (default): extract 16 frames from full video, run one inference
  - Sliding window (--sliding-window): chunk long videos, offset timestamps, deduplicate

Usage:
    # Single video
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/infer_dense.py --video /path/to/video.mp4

    # Sliding window for videos > 120s
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/infer_dense.py \\
        --video /path/to/long_video.mp4 --sliding-window

    # Batch mode on test set
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/infer_dense.py \\
        --test-json $DATA_ROOT/UCFCrime_Test.json --n 10
"""

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.video_utils import VideoMetadata

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT     = os.environ.get("DATA_ROOT", "/Volumes/T7/research-vlm/data")
VIDEO_ROOT    = f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos"

MODEL_ID      = os.environ.get("MODEL_ID",     "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
FINETUNED_DIR = os.environ.get("FINETUNED_DIR", "./output/smolvlm2-500m-dense-sft")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR",    "./output/smolvlm2-500m-dense-sft")

MLFLOW_URI        = os.environ.get("MLFLOW_URI",        "https://mlflow-geoai.stelarea.com/")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "smolvlm2-surveillance-dense")

NUM_FRAMES      = 16
MAX_DURATION    = 120.0   # seconds per window
WINDOW_SIZE     = 120.0   # sliding window size
WINDOW_STRIDE   = 60.0    # 50% overlap
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

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _category_from_id(video_id: str) -> str:
    return re.sub(r"\d+_x264$", "", video_id)


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds via PyAV."""
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

        collected = {}
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


def _make_video_metadata(start: float, end: float, n_frames: int) -> VideoMetadata:
    frame_timestamps = [
        start + i * (end - start) / max(n_frames - 1, 1)
        for i in range(n_frames)
    ]
    return VideoMetadata(
        total_num_frames=max(int(end), n_frames),
        fps=1.0,
        frames_indices=[int(t) for t in frame_timestamps],
        duration=float(end),
    )


def parse_dense_output(text: str) -> list[dict]:
    """Parse numbered activity list from model output.

    Matches: "1. [0.0, 5.3] Description text here"
    Returns: [{"start": 0.0, "end": 5.3, "description": "Description text here"}, ...]
    """
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
    """Remove near-duplicate activities from sliding window overlap.
    When two predictions overlap heavily (tIoU > threshold), keep the one with
    the longer description (more informative).
    """
    if not activities:
        return []

    activities = sorted(activities, key=lambda x: x["start"])
    kept = []

    for act in activities:
        duplicate = False
        for existing in kept:
            if tiou(act["start"], act["end"], existing["start"], existing["end"]) > iou_threshold:
                # Replace if current description is longer
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

def run_inference(model, processor, device, frames: list,
                  start: float, end: float, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    metadata = _make_video_metadata(start, end, len(frames))
    inputs = processor(
        text=[text],
        videos=[[frames]],
        video_metadata=[metadata],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    new_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()


def infer_single_pass(model, processor, device, video_path: str) -> tuple[list[dict], str]:
    """Single-pass inference on full video (or first MAX_DURATION seconds)."""
    duration      = get_video_duration(video_path)
    effective_end = min(duration, MAX_DURATION) if duration > 0 else MAX_DURATION

    frames   = extract_frames(video_path, 0.0, effective_end, NUM_FRAMES)
    raw_text = run_inference(model, processor, device, frames, 0.0, effective_end, DENSE_PROMPT)
    return parse_dense_output(raw_text), raw_text


def infer_sliding_window(model, processor, device, video_path: str) -> tuple[list[dict], list[dict]]:
    """Sliding window inference over full video. Returns deduplicated activities + per-window records."""
    duration = get_video_duration(video_path)
    if duration <= 0:
        print("  [WARN] Could not determine video duration; falling back to single-pass.")
        activities, _ = infer_single_pass(model, processor, device, video_path)
        return activities, []

    all_activities = []
    window_records = []

    t = 0.0
    while t < duration:
        w_start = t
        w_end   = min(t + WINDOW_SIZE, duration)

        frames = extract_frames(video_path, w_start, w_end, NUM_FRAMES)
        prompt = WINDOW_PROMPT_TEMPLATE.format(offset=w_start)
        raw    = run_inference(model, processor, device, frames, w_start, w_end, prompt)

        # Timestamps from model are relative to window start
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
# Test set batch mode
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

        duration = float(ann.get("duration", MAX_DURATION))
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
            "duration":   duration,
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

    parser.add_argument("--finetuned",       default=FINETUNED_DIR, help="Fine-tuned model dir")
    parser.add_argument("--sliding-window",  action="store_true",   help="Use sliding window (for long videos)")
    parser.add_argument("--n",               type=int, default=10,  help="Number of test clips (batch mode)")
    parser.add_argument("--output",          default=None,          help="Output JSON path")
    parser.add_argument("--no-mlflow",       action="store_true",   help="Disable MLflow logging")
    args = parser.parse_args()

    device = get_device()
    dtype  = torch.float32 if device.type == "mps" else torch.bfloat16
    print(f"Device: {device}  dtype: {dtype}\n")

    run_name = f"infer-dense-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    out_path = Path(args.output) if args.output else \
               Path(OUTPUT_DIR) / "results" / f"{run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- MLflow ---
    mlflow_run = None
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(run_name=run_name, tags={"type": "inference_dense"})
            mlflow.log_params({
                "finetuned_dir":  args.finetuned,
                "sliding_window": args.sliding_window,
                "num_frames":     NUM_FRAMES,
                "device":         str(device),
            })
            print(f"MLflow run: {mlflow_run.info.run_id}")
        except Exception as e:
            print(f"[WARN] MLflow init failed: {e}")
            mlflow_run = None

    # --- Load model ---
    finetuned_available = args.finetuned and os.path.isdir(args.finetuned)
    model_source = args.finetuned if finetuned_available else MODEL_ID
    print(f"Loading model from {model_source} ...")
    if not finetuned_available:
        print(f"  [INFO] Fine-tuned dir not found ({args.finetuned}), using base model (zero-shot)")
    processor = AutoProcessor.from_pretrained(model_source)
    model     = AutoModelForImageTextToText.from_pretrained(
        model_source, torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M\n")

    sep = "=" * 72
    video_results = []

    # ------------------------------------------------------------------ #
    # Mode A: single video                                                #
    # ------------------------------------------------------------------ #
    if args.video:
        video_path = args.video
        print(f"Video: {video_path}")
        duration = get_video_duration(video_path)
        print(f"Duration: {duration:.1f}s\n")

        if args.sliding_window:
            activities, windows = infer_sliding_window(model, processor, device, video_path)
        else:
            activities, raw = infer_single_pass(model, processor, device, video_path)
            windows = []

        print(sep)
        print(f"Detected {len(activities)} activities:")
        for a in activities:
            print(f"  [{a['start']:.1f}, {a['end']:.1f}] {a['description']}")
        print(sep)

        video_results.append({
            "video_path":  video_path,
            "duration":    duration,
            "mode":        "sliding_window" if args.sliding_window else "single_pass",
            "activities":  activities,
            "windows":     windows if args.sliding_window else [],
        })

    # ------------------------------------------------------------------ #
    # Mode B: batch on test set                                           #
    # ------------------------------------------------------------------ #
    else:
        samples = load_test_samples(args.test_json, args.n)
        if not samples:
            sys.exit("No test samples found — check test-json path and VIDEO_ROOT.")
        print(f"Loaded {len(samples)} test videos\n")

        for i, s in enumerate(samples, 1):
            print(sep)
            print(f"[{i}/{len(samples)}] {s['video_id']}  duration={s['duration']:.1f}s")
            print(f"  GT activities: {len(s['gt'])}")

            if args.sliding_window:
                activities, windows = infer_sliding_window(model, processor, device, s["video_path"])
                raw_output = None
            else:
                activities, raw_output = infer_single_pass(model, processor, device, s["video_path"])
                windows = []

            print(f"  Predicted:     {len(activities)} activities")
            for a in activities:
                print(f"    [{a['start']:.1f}, {a['end']:.1f}] {a['description']}")
            if raw_output is not None:
                print(f"  Raw output:    {raw_output[:200]!r}{'...' if len(raw_output) > 200 else ''}")
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

    # --- Save JSON ---
    output = {
        "run_name":     run_name,
        "finetuned_dir": args.finetuned,
        "mode":         "sliding_window" if args.sliding_window else "single_pass",
        "n_videos":     len(video_results),
        "num_frames":   NUM_FRAMES,
        "device":       str(device),
        "videos":       video_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # --- MLflow artifact ---
    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_artifact(str(out_path), artifact_path="results")
            mlflow.log_metric("n_videos_inferred", len(video_results))
            mlflow.end_run()
            print("MLflow artifact logged.")
        except Exception as e:
            print(f"[WARN] MLflow artifact log failed: {e}")
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    main()
