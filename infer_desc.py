"""
Activity description inference — describe ALL activities in a raw surveillance video.

No timestamp output. Model generates a numbered list of activities observed in the video.

Usage:
    # Single video
    python vlm-sft-pipeline/infer_desc.py --video /path/to/video.mp4

    # Batch mode on test set
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/infer_desc.py \\
        --test-json $DATA_ROOT/UCFCrime_Test.json --n 10

    # Use base model (zero-shot, no fine-tune)
    python vlm-sft-pipeline/infer_desc.py --video /path/to/video.mp4 --base-model
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

DATA_ROOT     = os.environ.get("DATA_ROOT", "./data")
VIDEO_ROOT    = f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos"

MODEL_ID      = os.environ.get("MODEL_ID",      "HuggingFaceTB/SmolVLM2-2.2B-Video-Instruct")
FINETUNED_DIR = os.environ.get("FINETUNED_DIR", "./output/smolvlm2-desc-sft")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR",    "./output/smolvlm2-desc-sft")

MLFLOW_URI        = os.environ.get("MLFLOW_URI",        "https://mlflow-geoai.stelarea.com/")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "smolvlm2-surveillance-desc")

NUM_FRAMES     = 32
MAX_DURATION   = 90.0
MAX_NEW_TOKENS = 512
SEED           = 99

DESC_PROMPT = (
    "Describe all activities in this surveillance video. "
    "List each activity on a new line, numbered from 1."
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
        total_num_frames=max(int(end * 10), n_frames),
        fps=10.0,
        frames_indices=[round(t * 10) for t in frame_timestamps],
        duration=float(end),
    )


def parse_desc_output(text: str) -> list[str]:
    """Parse numbered list output into a list of activity strings."""
    sentences = []
    for line in text.strip().split("\n"):
        line = line.strip()
        m = re.match(r"^\d+\.\s*(.+)", line)
        if m:
            sentences.append(m.group(1).strip())
        elif line and not re.match(r"^\d+\.?\s*$", line):
            sentences.append(line)
    return sentences


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, processor, device, frames: list, start: float, end: float) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": DESC_PROMPT},
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
        max_length=4096,
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    new_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens[0], skip_special_tokens=True).strip()


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
            for root_dir, _, files in os.walk(VIDEO_ROOT):
                if f"{video_id}.mp4" in files:
                    video_path = os.path.join(root_dir, f"{video_id}.mp4")
                    break
        if not os.path.isfile(video_path):
            continue

        duration = float(ann.get("duration", MAX_DURATION))
        gt_sents = []
        for (start, end), sentence in zip(ann["timestamps"], ann["sentences"]):
            start, end = float(start), float(end)
            if end > start:
                gt_sents.append(sentence.strip())

        if not gt_sents:
            continue

        items.append({
            "video_id":    video_id,
            "video_path":  video_path,
            "duration":    duration,
            "gt_sentences": gt_sents,
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

    parser.add_argument("--finetuned",   default=FINETUNED_DIR, help="Fine-tuned model dir")
    parser.add_argument("--base-model",  action="store_true",   help="Force use of base model (zero-shot)")
    parser.add_argument("--n",           type=int, default=10,  help="Number of test clips (batch mode)")
    parser.add_argument("--output",      default=None,          help="Output JSON path")
    parser.add_argument("--no-mlflow",   action="store_true",   help="Disable MLflow logging")
    args = parser.parse_args()

    device = get_device()
    dtype  = torch.float32 if device.type == "mps" else torch.bfloat16
    print(f"Device: {device}  dtype: {dtype}\n")

    run_name = f"infer-desc-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
            mlflow_run = mlflow.start_run(run_name=run_name, tags={"type": "inference_desc"})
            mlflow.log_params({
                "finetuned_dir": args.finetuned,
                "base_model":    args.base_model,
                "num_frames":    NUM_FRAMES,
                "device":        str(device),
            })
            print(f"MLflow run: {mlflow_run.info.run_id}")
        except Exception as e:
            print(f"[WARN] MLflow init failed: {e}")
            mlflow_run = None

    # --- Load model ---
    finetuned_available = not args.base_model and args.finetuned and os.path.isdir(args.finetuned)
    model_source = args.finetuned if finetuned_available else MODEL_ID
    print(f"Loading model from {model_source} ...")
    if not finetuned_available:
        print(f"  [INFO] Using base model (zero-shot)")
    processor = AutoProcessor.from_pretrained(model_source)
    model     = AutoModelForImageTextToText.from_pretrained(
        model_source, torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M\n")

    sep = "=" * 72
    video_results = []

    # ------------------------------------------------------------------ #
    # Mode A: single video                                                #
    # ------------------------------------------------------------------ #
    if args.video:
        video_path = args.video
        print(f"Video: {video_path}")
        duration    = get_video_duration(video_path)
        eff_end     = min(duration, MAX_DURATION) if duration > 0 else MAX_DURATION
        print(f"Duration: {duration:.1f}s\n")

        frames    = extract_frames(video_path, 0.0, eff_end, NUM_FRAMES)
        raw_text  = run_inference(model, processor, device, frames, 0.0, eff_end)
        sentences = parse_desc_output(raw_text)

        print(sep)
        print(f"Detected {len(sentences)} activities:")
        for j, s in enumerate(sentences, 1):
            print(f"  {j}. {s}")
        print(sep)

        video_results.append({
            "video_path": video_path,
            "duration":   duration,
            "raw_output": raw_text,
            "activities": sentences,
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
            duration = s["duration"]
            eff_end  = min(duration, MAX_DURATION)
            print(f"[{i}/{len(samples)}] {s['video_id']}  duration={duration:.1f}s")
            print(f"  GT activities: {len(s['gt_sentences'])}")

            frames    = extract_frames(s["video_path"], 0.0, eff_end, NUM_FRAMES)
            raw_text  = run_inference(model, processor, device, frames, 0.0, eff_end)
            sentences = parse_desc_output(raw_text)

            print(f"  Predicted: {len(sentences)} activities")
            for j, sent in enumerate(sentences, 1):
                print(f"    {j}. {sent}")
            print(f"  Raw: {raw_text[:200]!r}{'...' if len(raw_text) > 200 else ''}")
            print()

            video_results.append({
                "video_id":      s["video_id"],
                "duration":      duration,
                "gt_sentences":  s["gt_sentences"],
                "predicted":     sentences,
                "raw_output":    raw_text,
            })

    print(sep)
    print("Done.")

    output = {
        "run_name":      run_name,
        "model_source":  model_source,
        "n_videos":      len(video_results),
        "num_frames":    NUM_FRAMES,
        "device":        str(device),
        "videos":        video_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

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
