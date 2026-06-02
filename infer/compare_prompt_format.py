"""
Compare class-first vs description-first prompt formats on zero-shot VLM.

Runs the SAME test clips through ONE zero-shot model with TWO prompt formats:
  - class_first  : "[ClassName] description"
  - desc_first   : "description [ClassName]"

Reports metrics side-by-side to inform training format choice.

Usage:
    DATA_ROOT=/path/to/data python infer/compare_prompt_format.py --n 50
    DATA_ROOT=/path/to/data python infer/compare_prompt_format.py --model Qwen/Qwen3-VL-2B-Instruct --n 100
"""

import argparse
import json
import os
import random
import re
import sys
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
TEST_JSON     = os.environ.get("TEST_JSON", f"{DATA_ROOT}/classified/UCFCrime_Test_deepseek_v4_pro.json")
ANOMALY_TEST_SPLIT = os.environ.get(
    "ANOMALY_TEST_SPLIT",
    f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt",
)

NUM_FRAMES     = 4
MAX_NEW_TOKENS = 128

UCF_CLASSES = frozenset([
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
])

CLASS_LIST_STR = (
    "Normal, Abuse, Arrest, Arson, Assault, Burglary, "
    "Explosion, Fighting, RoadAccidents, Robbery, Shooting, "
    "Shoplifting, Stealing, Vandalism"
)

PROMPT_CLASS_FIRST = (
    "Watch this surveillance video clip carefully. "
    f"Identify the activity class from: {CLASS_LIST_STR}. "
    "Respond with [ClassName] followed by one sentence describing the activity."
)

PROMPT_DESC_FIRST = (
    "Watch this surveillance video clip carefully. "
    "Describe the activity in one sentence, then classify it. "
    f"Pick the class from: {CLASS_LIST_STR}. "
    "Respond with the description first, then [ClassName] at the end."
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_class(text: str) -> str:
    """Extract first valid [ClassName] from VLM output."""
    for m in re.findall(r'\[(\w+)\]', text):
        if m in UCF_CLASSES:
            return m
    # Fallback: bare class name anywhere in text
    for cls in sorted(UCF_CLASSES, key=len, reverse=True):
        if re.search(rf'\b{cls}\b', text):
            return cls
    return "Unknown"


# ---------------------------------------------------------------------------
# Helpers (mirror infer.py)
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _category_from_id(video_id):
    return re.sub(r"\d+_x264$", "", video_id)


def _load_anomaly_test_ids():
    if not os.path.isfile(ANOMALY_TEST_SPLIT):
        return set()
    with open(ANOMALY_TEST_SPLIT) as f:
        return {line.strip().split("/")[-1].replace(".mp4", "") for line in f if line.strip()}


def load_test_samples(n: int, crime_ratio: float = 0.8):
    with open(TEST_JSON) as f:
        data = json.load(f)
    allowed = _load_anomaly_test_ids()

    items = []
    for video_id, ann in data.items():
        if allowed and video_id not in allowed:
            continue
        category   = _category_from_id(video_id)
        video_path = os.path.join(VIDEO_ROOT, category, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            continue
        for (start, end), sent_entry in zip(ann["timestamps"], ann["sentences"]):
            if end <= start:
                continue
            if isinstance(sent_entry, dict):
                gt_text  = sent_entry["text"].strip()
                gt_class = sent_entry.get("class", "Unknown")
            else:
                gt_text, gt_class = sent_entry.strip(), "Unknown"
            items.append({"video_id": video_id, "video_path": video_path,
                          "start": float(start), "end": float(end),
                          "gt": gt_text, "gt_class": gt_class})

    crime  = [x for x in items if x["gt_class"] not in ("Normal", "Unknown")]
    normal = [x for x in items if x["gt_class"] == "Normal"]
    random.shuffle(crime)
    random.shuffle(normal)

    n_crime  = min(int(n * crime_ratio), len(crime))
    n_normal = min(n - n_crime, len(normal))
    selected = crime[:n_crime] + normal[:n_normal]
    random.shuffle(selected)
    print(f"  Sampled {n_crime} crime + {n_normal} normal")
    return selected


def extract_frames(video_path, start, end, n_frames):
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


def _make_video_metadata(start, end, n_frames):
    return VideoMetadata(
        total_num_frames=n_frames,
        fps=1.0,
        frames_indices=list(range(n_frames)),
        duration=float(end - start),
    )


def _is_image_mode_processor(processor) -> bool:
    """Detect processors that want frames as individual images (e.g. LFM2-VL)."""
    return type(processor).__name__ == "Lfm2VlProcessor"


def run_inference(model, processor, device, frames, start, end, prompt,
                  rep_penalty=1.3):
    if _is_image_mode_processor(processor):
        # LFM2-VL: frames as individual {"type":"image"} items, apply_chat_template returns tensors
        content = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_tensors="pt", return_dict=True,
        ).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                repetition_penalty=rep_penalty,
            )
        new_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
        return processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

    # Default: video-mode (SmolVLM2, Qwen3-VL, etc.)
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    metadata = _make_video_metadata(start, end, len(frames))
    inputs = processor(
        text=[text], videos=[[frames]], video_metadata=[metadata],
        return_tensors="pt", padding=True, truncation=True, max_length=1024,
    ).to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 repetition_penalty=rep_penalty)
    new_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds_classes, gt_classes, pred_texts, gt_texts):
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except ImportError:
        return {}

    m = {}
    m["unknown_rate"] = round(sum(p == "Unknown" for p in preds_classes) / max(len(preds_classes), 1), 4)

    pairs = [(p, g) for p, g in zip(preds_classes, gt_classes) if g != "Unknown"]
    if not pairs:
        return m
    preds, gts = zip(*pairs)
    all_labels = sorted(UCF_CLASSES)

    m["cls_accuracy"]    = round(accuracy_score(gts, preds), 4)
    m["f1_macro"]        = round(f1_score(gts, preds, average="macro",    zero_division=0, labels=all_labels), 4)
    m["f1_weighted"]     = round(f1_score(gts, preds, average="weighted", zero_division=0, labels=all_labels), 4)

    b_gts   = [0 if g == "Normal" else 1 for g in gts]
    b_preds = [0 if p in ("Normal", "Unknown") else 1 for p in preds]
    m["binary_accuracy"]  = round(accuracy_score(b_gts, b_preds), 4)
    m["binary_f1"]        = round(f1_score(b_gts, b_preds, average="binary", zero_division=0), 4)
    m["binary_precision"] = round(precision_score(b_gts, b_preds, average="binary", zero_division=0), 4)
    m["binary_recall"]    = round(recall_score(b_gts, b_preds, average="binary", zero_division=0), 4)

    # Caption metrics
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, pred)["rougeL"].fmeasure
                  for pred, ref in zip(pred_texts, gt_texts)]
        m["rougeL"] = round(sum(scores) / len(scores), 4)
    except Exception:
        pass

    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                        help="HF model ID for zero-shot eval")
    parser.add_argument("--n",           type=int,   default=50)
    parser.add_argument("--crime-ratio", type=float, default=0.8)
    parser.add_argument("--context-pad", type=float, default=5.0)
    parser.add_argument("--rep-penalty",        type=float, default=1.3)
    parser.add_argument("--output",              default=None)
    parser.add_argument("--trust-remote-code",   action="store_true",
                        help="Pass trust_remote_code=True (needed for InternVL3 etc.)")
    args = parser.parse_args()

    device = get_device()
    dtype  = torch.float32 if device.type == "mps" else torch.bfloat16
    print(f"Device: {device}  dtype: {dtype}")
    print(f"Model:  {args.model}\n")

    trc = args.trust_remote_code
    print(f"Loading model ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=trc)
    model     = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=trc).to(device)
    model.eval()

    print(f"\nLoading {args.n} test samples ...")
    samples = load_test_samples(args.n, args.crime_ratio)
    if not samples:
        sys.exit("No samples found")

    sep = "=" * 72
    results = []

    cf_classes, df_classes = [], []
    cf_texts,   df_texts   = [], []
    gt_classes, gt_texts   = [], []

    for i, s in enumerate(samples, 1):
        pad_s = max(0.0, s["start"] - args.context_pad)
        pad_e = s["end"] + args.context_pad

        print(sep)
        print(f"[{i}/{len(samples)}] {s['video_id']}  {s['start']:.1f}-{s['end']:.1f}s")
        print(f"  GT class: {s['gt_class']}  |  GT: {s['gt']}")

        frames = extract_frames(s["video_path"], pad_s, pad_e, NUM_FRAMES)

        # Class-first
        cf_out = run_inference(model, processor, device, frames, pad_s, pad_e,
                                PROMPT_CLASS_FIRST, args.rep_penalty)
        cf_cls = parse_class(cf_out)
        print(f"  CF [{cf_cls}]: {cf_out}")

        # Description-first
        df_out = run_inference(model, processor, device, frames, pad_s, pad_e,
                                PROMPT_DESC_FIRST, args.rep_penalty)
        df_cls = parse_class(df_out)
        print(f"  DF [{df_cls}]: {df_out}")

        cf_classes.append(cf_cls);  df_classes.append(df_cls)
        cf_texts.append(cf_out);    df_texts.append(df_out)
        gt_classes.append(s["gt_class"]); gt_texts.append(s["gt"])

        results.append({
            "video_id": s["video_id"], "start": s["start"], "end": s["end"],
            "gt": s["gt"], "gt_class": s["gt_class"],
            "class_first": cf_out, "class_first_class": cf_cls,
            "desc_first":  df_out, "desc_first_class":  df_cls,
        })

    print(sep)

    # Metrics
    cf_metrics = compute_metrics(cf_classes, gt_classes, cf_texts, gt_texts)
    df_metrics = compute_metrics(df_classes, gt_classes, df_texts, gt_texts)

    print("\n=== Class-first prompt ===")
    for k, v in cf_metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Description-first prompt ===")
    for k, v in df_metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Delta (desc_first - class_first) ===")
    for k in sorted(set(cf_metrics) | set(df_metrics)):
        if isinstance(cf_metrics.get(k), (int, float)) and isinstance(df_metrics.get(k), (int, float)):
            print(f"  {k}: {df_metrics[k] - cf_metrics[k]:+.4f}")

    # Save
    out_path = args.output or f"./results/prompt_compare_{args.model.replace('/', '_')}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model":        args.model,
            "n_clips":      len(results),
            "crime_ratio":  args.crime_ratio,
            "context_pad":  args.context_pad,
            "metrics": {"class_first": cf_metrics, "desc_first": df_metrics},
            "clips":        results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
