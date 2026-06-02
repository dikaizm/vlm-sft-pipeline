"""
Video-level inference with aggregation across sub-clips.

For each test video, runs inference on up to --clips-per-video sub-clips,
then aggregates predictions:
  - Binary  : anomaly if ANY clip predicts a crime class
  - 14-class: majority vote across all clip predictions

Saves per-video results + per-clip details to JSON.

Usage:
    DATA_ROOT=/path/to/data python infer/infer_video.py --finetuned ./output/.../checkpoint-400 --n 20
    DATA_ROOT=/path/to/data python infer/infer_video.py --finetuned ./output/.../checkpoint-400 --n 20 --no-zeroshot
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
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
TEST_JSON     = os.environ.get("TEST_JSON", f"{DATA_ROOT}/classified/UCFCrime_Test_deepseek_v4_pro.json")

MODEL_ID      = os.environ.get("MODEL_ID",     "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
FINETUNED_DIR = os.environ.get("FINETUNED_DIR", "./output/smolvlm2-2b-lora-sft")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR",    "./output/smolvlm2-2b-lora-sft")

ANOMALY_TEST_SPLIT = os.environ.get(
    "ANOMALY_TEST_SPLIT",
    f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt",
)

MLFLOW_URI        = os.environ.get("MLFLOW_URI",        "https://mlflow-geoai.stelarea.com/")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "smolvlm2-surveillance-sft")

NUM_FRAMES     = 4
MAX_NEW_TOKENS = 128

UCF_CLASSES = frozenset([
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
])

PROMPT = (
    "Watch this surveillance video clip carefully. "
    "Identify the activity class from: Normal, Abuse, Arrest, Arson, Assault, Burglary, "
    "Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism. "
    "Respond with [ClassName] followed by one sentence describing the activity."
)


# ---------------------------------------------------------------------------
# Shared helpers (mirrors infer.py)
# ---------------------------------------------------------------------------

class ClassFirstLogitsProcessor:
    def __init__(self, valid_class_seqs: list[list[int]], prompt_length: int):
        self.seqs = valid_class_seqs
        self.prompt_len = prompt_length
        self.max_prefix_len = max(len(s) for s in valid_class_seqs)

    def __call__(self, input_ids, scores):
        gen_pos = input_ids.shape[1] - self.prompt_len
        if gen_pos >= self.max_prefix_len:
            return scores
        valid_ids: set[int] = set()
        for seq in self.seqs:
            if gen_pos >= len(seq):
                continue
            if gen_pos == 0 or all(
                input_ids[0, self.prompt_len + i].item() == seq[i]
                for i in range(gen_pos)
            ):
                valid_ids.add(seq[gen_pos])
        if valid_ids:
            mask = torch.full_like(scores, float("-inf"))
            for vid in valid_ids:
                mask[:, vid] = 0.0
            scores = scores + mask
        return scores


def _build_class_logits_processor(processor, prompt_length: int):
    seqs = [processor.tokenizer.encode(f"[{cls}]", add_special_tokens=False) for cls in sorted(UCF_CLASSES)]
    return ClassFirstLogitsProcessor(seqs, prompt_length)


def parse_class(text: str) -> str:
    for m in re.findall(r'\[(\w+)\]', text):
        if m in UCF_CLASSES:
            return m
    return "Unknown"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _category_from_id(video_id: str) -> str:
    return re.sub(r"\d+_x264$", "", video_id)


def _load_anomaly_test_ids() -> set[str]:
    if not os.path.isfile(ANOMALY_TEST_SPLIT):
        print(f"[WARN] Anomaly_Test.txt not found — no leakage filter applied")
        return set()
    with open(ANOMALY_TEST_SPLIT) as f:
        ids = {line.strip().split("/")[-1].replace(".mp4", "") for line in f if line.strip()}
    print(f"  Anomaly_Test.txt: {len(ids)} official test videos loaded")
    return ids


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
    return VideoMetadata(
        total_num_frames=n_frames,
        fps=1.0,
        frames_indices=list(range(n_frames)),
        duration=float(end - start),
    )


def run_inference(model, processor, device, frames, start, end, prompt,
                  do_sample=False, temperature=0.7, top_p=0.9,
                  repetition_penalty=1.0, constrained=False) -> str:
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    metadata = _make_video_metadata(start, end, len(frames))
    inputs = processor(
        text=[text], videos=[[frames]], video_metadata=[metadata],
        return_tensors="pt", padding=True, truncation=True, max_length=1024,
    ).to(device)

    gen_kwargs: dict = {"max_new_tokens": MAX_NEW_TOKENS}
    if do_sample:
        gen_kwargs.update({"do_sample": True, "temperature": temperature,
                           "top_p": top_p, "repetition_penalty": repetition_penalty})
    if constrained:
        prompt_length = inputs["input_ids"].shape[1]
        gen_kwargs["logits_processor"] = [_build_class_logits_processor(processor, prompt_length)]

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Video-level data loading
# ---------------------------------------------------------------------------

def load_test_videos(n_videos: int, clips_per_video: int, crime_ratio: float = 0.8) -> list[dict]:
    """
    Returns list of video-level dicts, each with:
      video_id, video_path, gt_class (video-level), clips: [{start, end, gt, gt_class}]
    """
    with open(TEST_JSON) as f:
        data = json.load(f)

    allowed_ids = _load_anomaly_test_ids()

    videos = []
    skipped = 0
    for video_id, ann in data.items():
        if allowed_ids and video_id not in allowed_ids:
            skipped += 1
            continue
        category   = _category_from_id(video_id)
        video_path = os.path.join(VIDEO_ROOT, category, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            continue

        clips = []
        for (start, end), sent_entry in zip(ann["timestamps"], ann["sentences"]):
            if end <= start:
                continue
            if isinstance(sent_entry, dict):
                gt_text  = sent_entry["text"].strip()
                gt_class = sent_entry.get("class", "Unknown")
            else:
                gt_text  = sent_entry.strip()
                gt_class = "Unknown"
            clips.append({"start": float(start), "end": float(end),
                           "gt": gt_text, "gt_class": gt_class})

        if not clips:
            continue

        # Video GT: crime if any clip has crime label
        crime_clips = [c for c in clips if c["gt_class"] not in ("Normal", "Unknown")]
        video_gt = crime_clips[0]["gt_class"] if crime_clips else "Normal"

        videos.append({
            "video_id":   video_id,
            "video_path": video_path,
            "gt_class":   video_gt,
            "clips":      clips,
        })

    if skipped:
        print(f"  Filtered {skipped} train-split videos")

    # Stratified by video GT class
    crime_vids  = [v for v in videos if v["gt_class"] != "Normal"]
    normal_vids = [v for v in videos if v["gt_class"] == "Normal"]

    random.shuffle(crime_vids)
    random.shuffle(normal_vids)

    n_crime  = min(int(n_videos * crime_ratio), len(crime_vids))
    n_normal = min(n_videos - n_crime, len(normal_vids))
    selected = crime_vids[:n_crime] + normal_vids[:n_normal]
    random.shuffle(selected)

    print(f"  Sampled {n_crime} crime + {n_normal} normal videos (crime_ratio={crime_ratio})")

    # Cap clips per video
    for v in selected:
        clips = v["clips"]
        if len(clips) > clips_per_video:
            # Prefer crime clips, fill rest with normal
            crime_c  = [c for c in clips if c["gt_class"] not in ("Normal", "Unknown")]
            normal_c = [c for c in clips if c["gt_class"] == "Normal"]
            random.shuffle(crime_c)
            random.shuffle(normal_c)
            n_c = min(len(crime_c), clips_per_video)
            n_n = min(clips_per_video - n_c, len(normal_c))
            v["clips"] = crime_c[:n_c] + normal_c[:n_n]

    return selected


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_predictions(clip_preds: list[str]) -> dict:
    """
    clip_preds: list of predicted class strings (may include 'Unknown')

    Returns:
      binary_pred  : 'Anomaly' if any pred is a crime class, else 'Normal'
      multiclass_pred: majority vote (most common non-Unknown class; fallback Normal)
      vote_counts  : Counter of predictions
    """
    known = [p for p in clip_preds if p != "Unknown"]
    crime = [p for p in known if p != "Normal"]

    binary_pred = "Anomaly" if crime else "Normal"

    if known:
        counter = Counter(known)
        multiclass_pred = counter.most_common(1)[0][0]
    else:
        multiclass_pred = "Normal"

    return {
        "binary_pred":      binary_pred,
        "multiclass_pred":  multiclass_pred,
        "vote_counts":      dict(Counter(clip_preds)),
        "n_crime_votes":    len(crime),
        "n_total_votes":    len(clip_preds),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_video_metrics(video_results: list[dict], model_key: str) -> dict:
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except ImportError:
        print("[WARN] scikit-learn not installed")
        return {}

    gt_binary   = [1 if v["gt_class"] != "Normal" else 0 for v in video_results]
    pred_binary = [1 if v[model_key]["binary_pred"] == "Anomaly" else 0 for v in video_results]
    pred_multi  = [v[model_key]["multiclass_pred"] for v in video_results]
    gt_multi    = [v["gt_class"] for v in video_results]

    all_labels = sorted(UCF_CLASSES)

    return {
        "binary_accuracy":  round(accuracy_score(gt_binary, pred_binary), 4),
        "binary_f1":        round(f1_score(gt_binary, pred_binary, average="binary", zero_division=0), 4),
        "binary_precision": round(precision_score(gt_binary, pred_binary, average="binary", zero_division=0), 4),
        "binary_recall":    round(recall_score(gt_binary, pred_binary, average="binary", zero_division=0), 4),
        "cls_accuracy":     round(accuracy_score(gt_multi, pred_multi), 4),
        "f1_macro":         round(f1_score(gt_multi, pred_multi, average="macro", zero_division=0, labels=all_labels), 4),
        "f1_weighted":      round(f1_score(gt_multi, pred_multi, average="weighted", zero_division=0, labels=all_labels), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",               type=int,   default=20,          help="Number of test videos")
    parser.add_argument("--clips-per-video", type=int,   default=5,           help="Max sub-clips to infer per video (default: 5)")
    parser.add_argument("--context-pad",     type=float, default=5.0,         help="Seconds to pad before/after each clip (default: 5.0)")
    parser.add_argument("--crime-ratio",     type=float, default=0.8,         help="Fraction of videos from crime classes (default: 0.8)")
    parser.add_argument("--finetuned",       default=FINETUNED_DIR,           help="Fine-tuned model dir or checkpoint")
    parser.add_argument("--no-zeroshot",     action="store_true",             help="Skip zero-shot model")
    parser.add_argument("--output",          default=None,                    help="Path to save JSON results")
    parser.add_argument("--no-mlflow",       action="store_true",             help="Disable MLflow logging")
    parser.add_argument("--sample",          action="store_true",             help="Sampling decoding")
    parser.add_argument("--temperature",     type=float, default=0.7,         help="Sampling temperature")
    parser.add_argument("--top-p",           type=float, default=0.9,         help="Top-p sampling")
    parser.add_argument("--rep-penalty",     type=float, default=1.3,         help="Repetition penalty")
    parser.add_argument("--constrained",     action="store_true",             help="Constrain first tokens to [ClassName]")
    args = parser.parse_args()

    device = get_device()
    dtype  = torch.float32 if device.type == "mps" else torch.bfloat16
    print(f"Device: {device}  dtype: {dtype}\n")

    ft_path   = Path(args.finetuned).resolve()
    ckpt_part = ft_path.name
    base_part = ft_path.parent.name
    run_name  = f"infer-video-{base_part}-{ckpt_part}" if ckpt_part.startswith("checkpoint-") else f"infer-video-{base_part}"

    out_path = Path(args.output) if args.output else Path(OUTPUT_DIR) / "results" / f"{run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- MLflow ---
    mlflow_run = None
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(run_name=run_name, tags={"type": "inference-video"})
            mlflow.log_params({
                "model_id": MODEL_ID, "finetuned_dir": args.finetuned,
                "n_videos": args.n, "clips_per_video": args.clips_per_video,
                "context_pad": args.context_pad, "crime_ratio": args.crime_ratio,
            })
        except Exception as e:
            print(f"[WARN] MLflow init failed: {e}")
            mlflow_run = None

    # --- Load fine-tuned model ---
    print(f"Loading fine-tuned model from {args.finetuned} ...")
    adapter_cfg_path = Path(args.finetuned) / "adapter_config.json"
    is_lora_ckpt = adapter_cfg_path.exists()

    if is_lora_ckpt:
        with open(adapter_cfg_path) as f:
            _cfg = json.load(f)
        base_id  = _cfg.get("base_model_name_or_path", MODEL_ID)
        proc_src = args.finetuned if (Path(args.finetuned) / "tokenizer.json").exists() else base_id
        ft_processor = AutoProcessor.from_pretrained(proc_src)
        from peft import PeftModel
        _base    = AutoModelForImageTextToText.from_pretrained(base_id, torch_dtype=dtype).to(device)
        ft_model = PeftModel.from_pretrained(_base, args.finetuned).to(device)
    else:
        proc_src = args.finetuned if (Path(args.finetuned) / "tokenizer.json").exists() else MODEL_ID
        ft_processor = AutoProcessor.from_pretrained(proc_src)
        ft_model     = AutoModelForImageTextToText.from_pretrained(args.finetuned, torch_dtype=dtype).to(device)
    ft_model.eval()

    zs_model = zs_processor = None
    if not args.no_zeroshot:
        print(f"Loading zero-shot model ({MODEL_ID}) ...")
        zs_processor = AutoProcessor.from_pretrained(MODEL_ID)
        zs_model     = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
        zs_model.eval()

    # --- Load videos ---
    print(f"\nLoading {args.n} test videos (max {args.clips_per_video} clips each) ...")
    videos = load_test_videos(args.n, args.clips_per_video, crime_ratio=args.crime_ratio)
    if not videos:
        sys.exit("No test videos found.")
    total_clips = sum(len(v["clips"]) for v in videos)
    print(f"  {len(videos)} videos, {total_clips} total clips\n")

    infer_kwargs = dict(do_sample=args.sample, temperature=args.temperature,
                        top_p=args.top_p, repetition_penalty=args.rep_penalty,
                        constrained=args.constrained)

    # --- Run inference ---
    sep = "=" * 72
    video_results = []

    for vi, v in enumerate(videos, 1):
        print(sep)
        print(f"[{vi}/{len(videos)}] {v['video_id']}  GT: {v['gt_class']}  ({len(v['clips'])} clips)")

        ft_clip_preds  = []
        zs_clip_preds  = []
        clip_details   = []

        for ci, clip in enumerate(v["clips"], 1):
            pad_start = max(0.0, clip["start"] - args.context_pad)
            pad_end   = clip["end"] + args.context_pad
            print(f"  Clip {ci}: {clip['start']:.1f}–{clip['end']:.1f}s  pad→{pad_start:.1f}–{pad_end:.1f}s  GT_cls={clip['gt_class']}")

            frames = extract_frames(v["video_path"], pad_start, pad_end, NUM_FRAMES)

            detail = {"start": clip["start"], "end": clip["end"],
                      "pad_start": pad_start, "pad_end": pad_end,
                      "gt": clip["gt"], "gt_class": clip["gt_class"]}

            if zs_model is not None:
                zs_out = run_inference(zs_model, zs_processor, device, frames, pad_start, pad_end, PROMPT, **infer_kwargs)
                zs_cls = parse_class(zs_out)
                zs_clip_preds.append(zs_cls)
                detail["zeroshot"] = zs_out
                detail["zeroshot_class"] = zs_cls
                print(f"    ZS : [{zs_cls}] {zs_out}")

            ft_out = run_inference(ft_model, ft_processor, device, frames, pad_start, pad_end, PROMPT, **infer_kwargs)
            ft_cls = parse_class(ft_out)
            ft_clip_preds.append(ft_cls)
            detail["finetuned"] = ft_out
            detail["finetuned_class"] = ft_cls
            print(f"    FT : [{ft_cls}] {ft_out}")

            clip_details.append(detail)

        ft_agg = aggregate_predictions(ft_clip_preds)
        zs_agg = aggregate_predictions(zs_clip_preds) if zs_clip_preds else None

        print(f"  FT  agg → binary={ft_agg['binary_pred']}  multi={ft_agg['multiclass_pred']}  votes={ft_agg['vote_counts']}")
        if zs_agg:
            print(f"  ZS  agg → binary={zs_agg['binary_pred']}  multi={zs_agg['multiclass_pred']}  votes={zs_agg['vote_counts']}")

        video_results.append({
            "video_id":  v["video_id"],
            "gt_class":  v["gt_class"],
            "finetuned": ft_agg,
            "zeroshot":  zs_agg,
            "clips":     clip_details,
        })

    print(sep)

    # --- Video-level metrics ---
    ft_metrics = _compute_video_metrics(video_results, "finetuned")
    print("\nVideo-level metrics — Fine-tuned:")
    for k, val in ft_metrics.items():
        print(f"  {k}: {val}")

    zs_metrics = {}
    if zs_model is not None:
        zs_metrics = _compute_video_metrics(video_results, "zeroshot")
        print("Video-level metrics — Zero-shot:")
        for k, val in zs_metrics.items():
            print(f"  {k}: {val}")

    # --- Save ---
    output = {
        "run_name":      run_name,
        "finetuned_dir": args.finetuned,
        "n_videos":      len(video_results),
        "clips_per_video": args.clips_per_video,
        "context_pad":   args.context_pad,
        "metrics": {"finetuned": ft_metrics, "zeroshot": zs_metrics},
        "videos":        video_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    if mlflow_run is not None:
        try:
            import mlflow as _mlflow
            _mlflow.log_metrics(ft_metrics)
            _mlflow.log_artifact(str(out_path), artifact_path="results")
            _mlflow.end_run()
        except Exception as e:
            print(f"[WARN] MLflow log failed: {e}")


if __name__ == "__main__":
    main()
