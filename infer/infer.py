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

class ClassFirstLogitsProcessor:
    """Constrain first generated tokens to a valid [ClassName] sequence."""

    def __init__(self, valid_class_seqs: list[list[int]], prompt_length: int):
        self.seqs = valid_class_seqs
        self.prompt_len = prompt_length
        self.max_prefix_len = max(len(s) for s in valid_class_seqs)

    def __call__(self, input_ids: "torch.LongTensor", scores: "torch.FloatTensor") -> "torch.FloatTensor":
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
    """Pre-tokenize all [ClassName] strings and return a ClassFirstLogitsProcessor."""
    seqs = []
    for cls in sorted(UCF_CLASSES):
        ids = processor.tokenizer.encode(f"[{cls}]", add_special_tokens=False)
        seqs.append(ids)
    return ClassFirstLogitsProcessor(seqs, prompt_length)


def _compute_caption_metrics(predictions: list[str], references: list[str]) -> dict:
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


def _compute_cls_metrics(pred_classes: list[str], gt_classes: list[str]) -> dict:
    """Classification metrics: accuracy, F1 macro/weighted, precision, recall (14-class + binary)."""
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except ImportError:
        print("[WARN] scikit-learn not installed — skipping classification metrics. pip install scikit-learn")
        return {}

    metrics = {}

    # Unknown rate (model failed to output a valid class tag)
    metrics["unknown_rate"] = round(
        sum(p == "Unknown" for p in pred_classes) / max(len(pred_classes), 1), 4
    )

    # Filter to samples where GT class is known
    pairs = [(p, g) for p, g in zip(pred_classes, gt_classes) if g != "Unknown"]
    if not pairs:
        return metrics

    preds, gts = zip(*pairs)
    all_labels = sorted(UCF_CLASSES)

    # 14-class multiclass (Unknown pred counts as wrong)
    metrics["cls_accuracy"]    = round(accuracy_score(gts, preds), 4)
    metrics["f1_macro"]        = round(f1_score(gts, preds, average="macro",    zero_division=0, labels=all_labels), 4)
    metrics["f1_weighted"]     = round(f1_score(gts, preds, average="weighted", zero_division=0, labels=all_labels), 4)
    metrics["precision_macro"] = round(precision_score(gts, preds, average="macro", zero_division=0, labels=all_labels), 4)
    metrics["recall_macro"]    = round(recall_score(gts, preds, average="macro",    zero_division=0, labels=all_labels), 4)

    # Binary: Normal=0, Anomaly=1 (Unknown pred → 0, conservative: didn't flag anomaly)
    b_gts   = [0 if g == "Normal" else 1 for g in gts]
    b_preds = [0 if p in ("Normal", "Unknown") else 1 for p in preds]

    metrics["binary_accuracy"]  = round(accuracy_score(b_gts, b_preds), 4)
    metrics["binary_f1"]        = round(f1_score(b_gts, b_preds, average="binary", zero_division=0), 4)
    metrics["binary_precision"] = round(precision_score(b_gts, b_preds, average="binary", zero_division=0), 4)
    metrics["binary_recall"]    = round(recall_score(b_gts, b_preds, average="binary", zero_division=0), 4)

    return metrics

# ---------------------------------------------------------------------------
# Config  (all overridable via .env or CLI flags)
# ---------------------------------------------------------------------------

DATA_ROOT     = os.environ.get("DATA_ROOT", "/Volumes/T7/research-vlm/data")
VIDEO_ROOT    = f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos"
TEST_JSON     = os.environ.get("TEST_JSON", f"{DATA_ROOT}/classified/UCFCrime_Test_deepseek_v4_pro.json")

MODEL_ID      = os.environ.get("MODEL_ID",     "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
FINETUNED_DIR = os.environ.get("FINETUNED_DIR", "./output/smolvlm2-500m-small-sft")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR",    "./output/smolvlm2-500m-small-sft")

# Official UCF-Crime anomaly detection split — used to prevent leakage from train videos
ANOMALY_TEST_SPLIT = os.environ.get(
    "ANOMALY_TEST_SPLIT",
    f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt",
)

MLFLOW_URI        = os.environ.get("MLFLOW_URI",        "https://mlflow-geoai.stelarea.com/")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "smolvlm2-surveillance-sft")

NUM_FRAMES     = 4
SEED           = 99   # different from training seed
MAX_NEW_TOKENS = 128

UCF_CLASSES = frozenset([
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
])

PROMPT = (
    "Watch this surveillance video clip carefully. "
    "Respond with exactly: [ClassName] followed by one sentence describing the activity. "
    "Use square brackets exactly as shown. "
    "ClassName must be one of: Normal, Abuse, Arrest, Arson, Assault, Burglary, Explosion, "
    "Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism."
)


def parse_class(text: str) -> str:
    """Extract first valid [ClassName] from VLM output. Returns 'Unknown' if not found."""
    for m in re.findall(r'\[(\w+)\]', text):
        if m in UCF_CLASSES:
            return m
    return "Unknown"


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


def _load_anomaly_test_ids() -> set[str]:
    """Load official Anomaly_Test.txt split — returns set of video_ids without extension."""
    if not os.path.isfile(ANOMALY_TEST_SPLIT):
        print(f"[WARN] Anomaly_Test.txt not found at {ANOMALY_TEST_SPLIT} — no leakage filter applied")
        return set()
    with open(ANOMALY_TEST_SPLIT) as f:
        ids = set()
        for line in f:
            line = line.strip()
            if line:
                # Format: Category/VideoName.mp4
                ids.add(line.split("/")[-1].replace(".mp4", ""))
    print(f"  Anomaly_Test.txt: {len(ids)} official test videos loaded")
    return ids


def load_test_samples(n: int) -> list[dict]:
    with open(TEST_JSON) as f:
        data = json.load(f)

    allowed_ids = _load_anomaly_test_ids()

    items = []
    skipped_leakage = 0
    for video_id, ann in data.items():
        if allowed_ids and video_id not in allowed_ids:
            skipped_leakage += 1
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
                gt_text  = sent_entry.strip()
                gt_class = "Unknown"
            items.append({
                "video_id":   video_id,
                "video_path": video_path,
                "start":      float(start),
                "end":        float(end),
                "gt":         gt_text,
                "gt_class":   gt_class,
            })

    if skipped_leakage:
        print(f"  Filtered {skipped_leakage} videos present in training split (leakage prevention)")

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


def run_inference(model, processor, device, frames: list, start: float, end: float, prompt: str,
                  do_sample: bool = False, temperature: float = 0.7,
                  top_p: float = 0.9, repetition_penalty: float = 1.0,
                  constrained: bool = False) -> str:
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int, default=5,            help="Number of test clips")
    parser.add_argument("--finetuned",   default=FINETUNED_DIR,          help="Fine-tuned model dir")
    parser.add_argument("--no-zeroshot", action="store_true",            help="Skip zero-shot model")
    parser.add_argument("--output",      default=None,                   help="Path to save JSON results (default: OUTPUT_DIR/results/<run>.json)")
    parser.add_argument("--no-mlflow",   action="store_true",            help="Disable MLflow logging")
    parser.add_argument("--sample",      action="store_true",            help="Use sampling instead of greedy decoding (more diverse but noisier metrics)")
    parser.add_argument("--temperature", type=float, default=0.7,        help="Sampling temperature (default: 0.7, only used with --sample)")
    parser.add_argument("--top-p",       type=float, default=0.9,        help="Top-p nucleus sampling (default: 0.9, only used with --sample)")
    parser.add_argument("--rep-penalty", type=float, default=1.3,        help="Repetition penalty (default: 1.3; set 1.0 to disable)")
    parser.add_argument("--constrained", action="store_true",            help="Constrain first tokens to valid [ClassName] (class-first model only)")
    args = parser.parse_args()

    device = get_device()
    dtype  = torch.float32 if device.type == "mps" else torch.bfloat16
    print(f"Device: {device}  dtype: {dtype}\n")

    # Build run name from finetuned path: <parent-dir>-<checkpoint-dir>
    ft_path   = Path(args.finetuned).resolve()
    ckpt_part = ft_path.name          # e.g. checkpoint-1330
    base_part = ft_path.parent.name   # e.g. smolvlm2-500m-full-unfrz-sft-20260528-232935
    if ckpt_part.startswith("checkpoint-"):
        run_name = f"infer-{base_part}-{ckpt_part}"
    else:
        run_name = f"infer-{base_part}"  # finetuned dir IS the model dir (no checkpoint subdir)

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
    # Detect LoRA adapter checkpoint vs full-FT checkpoint
    print(f"Loading fine-tuned model from {args.finetuned} ...")
    adapter_cfg_path = Path(args.finetuned) / "adapter_config.json"
    is_lora_ckpt = adapter_cfg_path.exists()

    if is_lora_ckpt:
        import json as _json
        with open(adapter_cfg_path) as _f:
            _adapter_cfg = _json.load(_f)
        base_id = _adapter_cfg.get("base_model_name_or_path", MODEL_ID)
        print(f"  LoRA adapter detected. Base model: {base_id}")
        proc_src = args.finetuned if (Path(args.finetuned) / "tokenizer.json").exists() else base_id
        ft_processor = AutoProcessor.from_pretrained(proc_src)
        from peft import PeftModel
        _base = AutoModelForImageTextToText.from_pretrained(base_id, torch_dtype=dtype).to(device)
        ft_model = PeftModel.from_pretrained(_base, args.finetuned).to(device)
    else:
        proc_src = args.finetuned if (Path(args.finetuned) / "tokenizer.json").exists() else MODEL_ID
        ft_processor = AutoProcessor.from_pretrained(proc_src)
        ft_model = AutoModelForImageTextToText.from_pretrained(
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
            "gt_class":  s.get("gt_class", "Unknown"),
        }
        print(f"  GT class: {record['gt_class']}")

        infer_kwargs = dict(do_sample=args.sample, temperature=args.temperature,
                            top_p=args.top_p, repetition_penalty=args.rep_penalty,
                            constrained=args.constrained)

        if zs_model is not None:
            zs_out = run_inference(zs_model, zs_processor, device, frames, s["start"], s["end"], PROMPT,
                                   **infer_kwargs)
            zs_cls = parse_class(zs_out)
            print(f"  ZeroShot : {zs_out}")
            record["zeroshot"]       = zs_out
            record["zeroshot_class"] = zs_cls

        ft_out = run_inference(ft_model, ft_processor, device, frames, s["start"], s["end"], PROMPT,
                               **infer_kwargs)
        ft_cls = parse_class(ft_out)
        print(f"  FineTuned: {ft_out}")
        record["finetuned"]       = ft_out
        record["finetuned_class"] = ft_cls

        clip_results.append(record)
        print()

    print(sep)
    print("Done.")

    # --- Compute metrics ---
    gts = [r["gt"] for r in clip_results]

    ft_metrics = zs_metrics = {}
    if clip_results:
        print("\nComputing metrics...")
        gt_classes = [r["gt_class"] for r in clip_results]

        ft_preds   = [r["finetuned"] for r in clip_results]
        ft_metrics = _compute_caption_metrics(ft_preds, gts)
        ft_metrics.update(_compute_cls_metrics([r["finetuned_class"] for r in clip_results], gt_classes))
        print("  Fine-tuned  :")
        print("    Caption :", "  ".join(f"{k}={v:.4f}" for k, v in ft_metrics.items() if k in ("bleu4", "rougeL", "bertscore_f1")))
        print("    Cls 14  :", "  ".join(f"{k}={v:.4f}" for k, v in ft_metrics.items() if k in ("cls_accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro")))
        print("    Cls Bin :", "  ".join(f"{k}={v:.4f}" for k, v in ft_metrics.items() if k.startswith("binary")))
        print(f"    Unknown rate: {ft_metrics.get('unknown_rate', 'N/A')}")

        if "zeroshot" in clip_results[0]:
            zs_preds   = [r["zeroshot"] for r in clip_results]
            zs_metrics = _compute_caption_metrics(zs_preds, gts)
            zs_metrics.update(_compute_cls_metrics([r["zeroshot_class"] for r in clip_results], gt_classes))
            print("  Zero-shot   :")
            print("    Caption :", "  ".join(f"{k}={v:.4f}" for k, v in zs_metrics.items() if k in ("bleu4", "rougeL", "bertscore_f1")))
            print("    Cls 14  :", "  ".join(f"{k}={v:.4f}" for k, v in zs_metrics.items() if k in ("cls_accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro")))
            print("    Cls Bin :", "  ".join(f"{k}={v:.4f}" for k, v in zs_metrics.items() if k.startswith("binary")))
            print(f"    Unknown rate: {zs_metrics.get('unknown_rate', 'N/A')}")

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

    # --- Log to MLflow (parent + nested child runs) ---
    if mlflow_run is not None:
        try:
            import mlflow as _mlflow
            # Parent: summary params + artifact
            _mlflow.log_metric("n_clips_inferred", len(clip_results))
            _mlflow.log_artifact(str(out_path), artifact_path="results")

            # Child run: fine-tuned metrics
            with _mlflow.start_run(run_name="finetuned", nested=True,
                                   tags={"type": "finetuned"}):
                _mlflow.log_params({"model": args.finetuned, "n_clips": len(clip_results)})
                _mlflow.log_metrics(ft_metrics)

            # Child run: zero-shot metrics
            if zs_metrics:
                with _mlflow.start_run(run_name="zero-shot", nested=True,
                                       tags={"type": "zero-shot"}):
                    _mlflow.log_params({"model": MODEL_ID, "n_clips": len(clip_results)})
                    _mlflow.log_metrics(zs_metrics)

            _mlflow.end_run()
            print("MLflow nested runs logged.")
        except Exception as e:
            print(f"[WARN] MLflow log failed: {e}")
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    main()
