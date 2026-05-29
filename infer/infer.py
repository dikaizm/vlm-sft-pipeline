"""
Inference comparison: zero-shot vs fine-tuned SmolVLM2 on UCF-Crime clips.

Samples N clips from the test set, runs both models, prints results side-by-side,
and saves per-clip predictions to a JSON file under OUTPUT_DIR/results/.
Metrics are optionally logged to MLflow.

Usage:
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/infer.py
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/infer.py --n 10 --finetuned ./output/smolvlm2-500m-small-sft
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/infer.py --no-zeroshot --output results/my_run.json
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
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute BLEU-4, ROUGE-L, and BERTScore (F1) for a list of predictions."""
    metrics = {}

    # ROUGE-L
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, pred)["rougeL"].fmeasure
                  for pred, ref in zip(predictions, references)]
        metrics["rougeL"] = round(sum(scores) / len(scores), 4)
    except ImportError:
        print("[WARN] rouge_score not installed — skipping ROUGE-L. pip install rouge-score")

    # BLEU-4
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs_tok  = [[ref.lower().split()] for ref in references]
        hyps_tok  = [pred.lower().split() for pred in predictions]
        smoothie  = SmoothingFunction().method1
        metrics["bleu4"] = round(corpus_bleu(refs_tok, hyps_tok,
                                             smoothing_function=smoothie), 4)
    except ImportError:
        print("[WARN] nltk not installed — skipping BLEU-4. pip install nltk")

    # BERTScore (optional — skipped if not installed)
    try:
        import bert_score
        P, R, F = bert_score.score(predictions, references,
                                   lang="en", verbose=False,
                                   device="cpu")
        metrics["bertscore_f1"] = round(F.mean().item(), 4)
    except ImportError:
        pass  # BERTScore is optional — no warning spam

    return metrics

# ---------------------------------------------------------------------------
# Config  (all overridable via .env or CLI flags)
# ---------------------------------------------------------------------------

DATA_ROOT     = os.environ.get("DATA_ROOT", "/Volumes/T7/research-vlm/data")
VIDEO_ROOT    = f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos"
TEST_JSON     = os.environ.get("TEST_JSON", f"{DATA_ROOT}/UCFCrime_Test.json")

MODEL_ID      = os.environ.get("MODEL_ID",     "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
FINETUNED_DIR = os.environ.get("FINETUNED_DIR", "./output/smolvlm2-500m-small-sft")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR",    "./output/smolvlm2-500m-small-sft")

MLFLOW_URI        = os.environ.get("MLFLOW_URI",        "https://mlflow-geoai.stelarea.com/")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "smolvlm2-surveillance-sft")

NUM_FRAMES     = 4
SEED           = 99   # different from training seed
MAX_NEW_TOKENS = 128

PROMPT = "Describe the activity shown in this surveillance video clip."


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
    return items[:n]


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


def run_inference(model, processor, device, frames: list, start: float, end: float, prompt: str) -> str:
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int, default=5,            help="Number of test clips")
    parser.add_argument("--finetuned",   default=FINETUNED_DIR,          help="Fine-tuned model dir")
    parser.add_argument("--no-zeroshot", action="store_true",            help="Skip zero-shot model")
    parser.add_argument("--output",      default=None,                   help="Path to save JSON results (default: OUTPUT_DIR/results/<run>.json)")
    parser.add_argument("--no-mlflow",   action="store_true",            help="Disable MLflow logging")
    args = parser.parse_args()

    device = get_device()
    dtype  = torch.float32 if device.type == "mps" else torch.bfloat16
    print(f"Device: {device}  dtype: {dtype}\n")

    # Build run name from finetuned path: <parent-dir>-<checkpoint-dir>
    ft_path   = Path(args.finetuned).resolve()
    ckpt_part = ft_path.name          # e.g. checkpoint-1330
    base_part = ft_path.parent.name   # e.g. smolvlm2-500m-full-unfrz-sft-20260528-232935
    if ckpt_part.startswith("checkpoint-"):
        run_name = f"{base_part}-{ckpt_part}"
    else:
        run_name = base_part  # finetuned dir IS the model dir (no checkpoint subdir)

    # Resolve output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(OUTPUT_DIR) / "results" / f"{run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- MLflow ---
    mlflow_run = None
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(run_name=run_name, tags={"type": "inference"})
            mlflow.log_params({
                "model_id":      MODEL_ID,
                "finetuned_dir": args.finetuned,
                "n_clips":       args.n,
                "num_frames":    NUM_FRAMES,
                "zero_shot":     not args.no_zeroshot,
                "device":        str(device),
            })
            print(f"MLflow run: {mlflow_run.info.run_id}")
        except Exception as e:
            print(f"[WARN] MLflow init failed: {e}")
            mlflow_run = None

    # --- Load fine-tuned model ---
    # Processor may not be in checkpoint dir — fall back to base model
    print(f"Loading fine-tuned model from {args.finetuned} ...")
    proc_src = args.finetuned if (Path(args.finetuned) / "tokenizer.json").exists() else MODEL_ID
    ft_processor = AutoProcessor.from_pretrained(proc_src)
    ft_model     = AutoModelForImageTextToText.from_pretrained(
        args.finetuned, torch_dtype=dtype
    ).to(device)
    ft_model.eval()
    print(f"  Params: {sum(p.numel() for p in ft_model.parameters())/1e6:.0f}M")

    # --- Optionally load zero-shot model ---
    zs_model = zs_processor = None
    if not args.no_zeroshot:
        print(f"\nLoading zero-shot model ({MODEL_ID}) ...")
        zs_processor = AutoProcessor.from_pretrained(MODEL_ID)
        zs_model     = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=dtype
        ).to(device)
        zs_model.eval()

    # --- Load test samples ---
    print(f"\nLoading {args.n} test samples ...")
    samples = load_test_samples(args.n)
    if not samples:
        sys.exit("No test samples found — check TEST_JSON path and VIDEO_ROOT.")
    print(f"  Loaded {len(samples)} samples\n")

    # --- Run inference ---
    sep = "=" * 72
    clip_results = []

    for i, s in enumerate(samples, 1):
        print(sep)
        print(f"[{i}/{len(samples)}] {s['video_id']}")
        print(f"  Clip  : {s['start']:.1f}s – {s['end']:.1f}s")
        print(f"  GT    : {s['gt']}")

        frames = extract_frames(s["video_path"], s["start"], s["end"], NUM_FRAMES)

        record = {
            "video_id":  s["video_id"],
            "start":     s["start"],
            "end":       s["end"],
            "gt":        s["gt"],
        }

        if zs_model is not None:
            zs_out = run_inference(zs_model, zs_processor, device, frames, s["start"], s["end"], PROMPT)
            print(f"  ZeroShot : {zs_out}")
            record["zeroshot"] = zs_out

        ft_out = run_inference(ft_model, ft_processor, device, frames, s["start"], s["end"], PROMPT)
        print(f"  FineTuned: {ft_out}")
        record["finetuned"] = ft_out

        clip_results.append(record)
        print()

    print(sep)
    print("Done.")

    # --- Compute metrics ---
    gts = [r["gt"] for r in clip_results]

    ft_metrics = zs_metrics = {}
    if clip_results:
        print("\nComputing metrics...")
        ft_preds   = [r["finetuned"] for r in clip_results]
        ft_metrics = _compute_metrics(ft_preds, gts)
        print("  Fine-tuned  :", "  ".join(f"{k}={v:.4f}" for k, v in ft_metrics.items()))

        if "zeroshot" in clip_results[0]:
            zs_preds   = [r["zeroshot"] for r in clip_results]
            zs_metrics = _compute_metrics(zs_preds, gts)
            print("  Zero-shot   :", "  ".join(f"{k}={v:.4f}" for k, v in zs_metrics.items()))

    # --- Save results to JSON ---
    output = {
        "run_name":        run_name,
        "finetuned_dir":   args.finetuned,
        "model_id":        MODEL_ID,
        "n_clips":         len(clip_results),
        "num_frames":      NUM_FRAMES,
        "device":          str(device),
        "metrics": {
            "finetuned": ft_metrics,
            "zeroshot":  zs_metrics,
        },
        "clips":           clip_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # --- Log to MLflow ---
    if mlflow_run is not None:
        try:
            mlflow.log_metric("n_clips_inferred", len(clip_results))
            for k, v in ft_metrics.items():
                mlflow.log_metric(f"ft/{k}", v)
            for k, v in zs_metrics.items():
                mlflow.log_metric(f"zs/{k}", v)
            mlflow.log_artifact(str(out_path), artifact_path="results")
            mlflow.end_run()
            print("MLflow metrics and artifact logged.")
        except Exception as e:
            print(f"[WARN] MLflow log failed: {e}")
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    main()
