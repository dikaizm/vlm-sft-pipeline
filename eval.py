"""
Formal evaluation of SmolVLM2 on the UCF-Crime test set.

Metrics computed per sample:
  - tIoU      : Temporal Intersection over Union
  - ROUGE-L   : Longest common subsequence recall/precision/F1
  - BLEU-4    : 4-gram BLEU via sacrebleu
  - BERTScore : Semantic similarity F1 via roberta-large (batched after inference)

Aggregated results are written to a JSON file and printed as a summary table.

Usage:
    python vlm_sft_pipeline/eval.py
    python vlm_sft_pipeline/eval.py --n 50 --model ./output/smolvlm2-500m-small-sft
    python vlm_sft_pipeline/eval.py --n -1   # full test set
"""

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.video_utils import VideoMetadata

from rouge_score import rouge_scorer as rouge_lib
import sacrebleu
from bert_score import score as bert_score_fn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT  = "/Volumes/T7/research-vlm/data"
VIDEO_ROOT = f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos"
TEST_JSON  = f"{DATA_ROOT}/UCFCrime_Test.json"

DEFAULT_MODEL  = "./output/smolvlm2-500m-small-sft"
NUM_FRAMES     = 4
SEED           = 99
MAX_NEW_TOKENS = 128

PROMPT = (
    "Describe the activity in this surveillance video clip "
    "and provide the start and end timestamps in seconds."
)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _category_from_id(video_id: str) -> str:
    return re.sub(r"\d+_x264$", "", video_id)


def load_test_samples(n: int) -> list[dict]:
    with open(TEST_JSON) as f:
        data = json.load(f)

    items = []
    for video_id, ann in data.items():
        category   = _category_from_id(video_id)
        video_path = os.path.join(VIDEO_ROOT, category, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            continue
        for (start, end), sentence in zip(ann["timestamps"], ann["sentences"]):
            if end <= start:
                continue
            items.append({
                "video_id":   video_id,
                "video_path": video_path,
                "start":      float(start),
                "end":        float(end),
                "gt":         sentence.strip(),
            })

    random.seed(SEED)
    random.shuffle(items)
    return items if n == -1 else items[:n]

# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

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
        print(f"  [WARN] frame extraction failed for {video_path}: {e}")

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
        max_length=1024,
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    new_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def tiou(pred_start: float, pred_end: float,
         gt_start: float, gt_end: float) -> float:
    """Temporal Intersection over Union."""
    inter_start = max(pred_start, gt_start)
    inter_end   = min(pred_end,   gt_end)
    inter = max(0.0, inter_end - inter_start)
    union = (pred_end - pred_start) + (gt_end - gt_start) - inter
    return inter / union if union > 0 else 0.0


def parse_timestamps(text: str) -> tuple[float, float] | None:
    """Extract [start, end] from model output. Returns None if not found."""
    # Match patterns like [19.8, 26.7] or Timestamps: [2.6, 12.6]
    m = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def extract_description(text: str) -> str:
    """Strip timestamp portion from the response for text metric computation."""
    # Remove everything from 'Timestamps:' onward
    cleaned = re.sub(r"[Tt]imestamps?:?\s*\[.*?\]", "", text).strip()
    return cleaned if cleaned else text


_rouge = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(pred: str, ref: str) -> float:
    scores = _rouge.score(ref, pred)
    return scores["rougeL"].fmeasure


def bleu4(pred: str, ref: str) -> float:
    result = sacrebleu.corpus_bleu([pred], [[ref]], smooth_method="exp")
    return result.score / 100.0   # normalise to [0, 1]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=50,
                        help="Number of test clips (-1 = full test set)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Path to fine-tuned model directory")
    parser.add_argument("--out",   default=None,
                        help="Output JSON path (default: eval_results_<timestamp>.json)")
    args = parser.parse_args()

    device = get_device()
    dtype  = torch.float32 if device.type == "mps" else torch.bfloat16
    print(f"Device : {device}  dtype: {dtype}")
    print(f"Model  : {args.model}")
    n_label = "all" if args.n == -1 else str(args.n)
    print(f"Samples: {n_label}\n")

    # Load model
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(args.model)
    model     = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M\n")

    # Load samples
    samples = load_test_samples(args.n)
    if not samples:
        sys.exit("No test samples found — check TEST_JSON / VIDEO_ROOT.")
    print(f"Loaded {len(samples)} test samples\n")

    results = []
    tiou_scores, rouge_scores, bleu_scores = [], [], []
    pred_descs, ref_descs = [], []
    timestamp_found = 0

    for i, s in enumerate(samples, 1):
        print(f"[{i:>4}/{len(samples)}] {s['video_id']}  "
              f"GT=[{s['start']:.1f}, {s['end']:.1f}]", end="  ", flush=True)

        frames = extract_frames(s["video_path"], s["start"], s["end"], NUM_FRAMES)
        pred   = run_inference(model, processor, device, frames,
                               s["start"], s["end"], PROMPT)

        # --- tIoU ---
        ts = parse_timestamps(pred)
        if ts is not None:
            ts_iou = tiou(ts[0], ts[1], s["start"], s["end"])
            timestamp_found += 1
        else:
            ts_iou = 0.0
        tiou_scores.append(ts_iou)

        # --- ROUGE-L & BLEU-4 on description text ---
        pred_desc = extract_description(pred)
        rl = rouge_l(pred_desc, s["gt"])
        b4 = bleu4(pred_desc, s["gt"])
        rouge_scores.append(rl)
        bleu_scores.append(b4)
        pred_descs.append(pred_desc)
        ref_descs.append(s["gt"])

        print(f"tIoU={ts_iou:.3f}  ROUGE-L={rl:.3f}  BLEU-4={b4:.3f}")

        results.append({
            "video_id":   s["video_id"],
            "gt_start":   s["start"],
            "gt_end":     s["end"],
            "gt_text":    s["gt"],
            "pred":       pred,
            "pred_start": ts[0] if ts else None,
            "pred_end":   ts[1] if ts else None,
            "tiou":       ts_iou,
            "rouge_l":    rl,
            "bleu4":      b4,
        })

    # --- BERTScore (batched for efficiency) ---
    print("\nComputing BERTScore (batched)...")
    _, _, bert_f1 = bert_score_fn(
        pred_descs, ref_descs,
        lang="en", model_type="roberta-large", verbose=False
    )
    bert_scores = bert_f1.tolist()
    for r, bs in zip(results, bert_scores):
        r["bertscore_f1"] = bs

    # --- Aggregate ---
    n = len(samples)
    mean_tiou   = sum(tiou_scores) / n
    mean_rouge  = sum(rouge_scores) / n
    mean_bleu   = sum(bleu_scores) / n
    mean_bert   = sum(bert_scores) / n
    ts_rate     = timestamp_found / n

    # tIoU recall at thresholds
    tiou_03 = sum(1 for v in tiou_scores if v >= 0.3) / n
    tiou_05 = sum(1 for v in tiou_scores if v >= 0.5) / n
    tiou_07 = sum(1 for v in tiou_scores if v >= 0.7) / n

    print(f"\n{'='*72}")
    print(f"  Samples evaluated : {n}")
    print(f"  Timestamp found   : {timestamp_found}/{n}  ({ts_rate*100:.1f}%)")
    print(f"  Mean tIoU         : {mean_tiou:.4f}")
    print(f"  tIoU@0.3          : {tiou_03:.4f}")
    print(f"  tIoU@0.5          : {tiou_05:.4f}")
    print(f"  tIoU@0.7          : {tiou_07:.4f}")
    print(f"  Mean ROUGE-L      : {mean_rouge:.4f}")
    print(f"  Mean BLEU-4       : {mean_bleu:.4f}")
    print(f"  Mean BERTScore F1 : {mean_bert:.4f}")
    print(f"{'='*72}\n")

    summary = {
        "model":              args.model,
        "n_samples":          n,
        "timestamp_rate":     ts_rate,
        "mean_tiou":          mean_tiou,
        "tiou_at_0.3":        tiou_03,
        "tiou_at_0.5":        tiou_05,
        "tiou_at_0.7":        tiou_07,
        "mean_rouge_l":       mean_rouge,
        "mean_bleu4":         mean_bleu,
        "mean_bertscore_f1":  mean_bert,
        "per_sample":         results,
    }

    out_path = args.out or f"eval_results_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
