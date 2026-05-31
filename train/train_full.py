"""
Full-dataset SFT pipeline for SmolVLM2 (500M or 2.2B) on UCF-Crime + UCA dataset.

Supports full fine-tune and LoRA (PEFT). Logs to MLflow.

Usage:
    # Full fine-tune, 500M, full dataset
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/train/train_full.py
        --model HuggingFaceTB/SmolVLM2-500M-Video-Instruct

    # LoRA, 2.2B
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/train/train_full.py \
        --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --lora

    # Pilot run (200 samples) to validate pipeline
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/train/train_full.py \
        --max-train 200 --max-val 50
"""

import argparse
import functools
import hashlib
import json
import logging
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Reduce CUDA allocator fragmentation — must be set before any CUDA allocation
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import mlflow
import torch
from PIL import Image
from datasets import Dataset
import torch.nn as nn
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.video_utils import VideoMetadata

# ---------------------------------------------------------------------------
# Defaults (all overridable via CLI or env)
# ---------------------------------------------------------------------------

_PIPELINE_ROOT = Path(__file__).parent.parent   # vlm-sft-pipeline/

_DATA_ROOT    = os.environ.get("DATA_ROOT", str(_PIPELINE_ROOT / "data"))
_VIDEO_ROOT   = f"{_DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos"
_TRAIN_JSON   = f"{_DATA_ROOT}/classified/UCFCrime_Train_deepseek_v4_pro.json"
_VAL_JSON     = f"{_DATA_ROOT}/classified/UCFCrime_Val_deepseek_v4_pro.json"
_OUTPUT_DIR   = os.environ.get("OUTPUT_DIR", str(_PIPELINE_ROOT / "output" / "smolvlm2-full-sft"))
_MODEL_ID     = os.environ.get("MODEL_ID",   "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

MLFLOW_URI        = os.environ.get("MLFLOW_URI",        "https://mlflow-geoai.stelarea.com/")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "smolvlm2-surveillance-sft")

UCF_CLASSES = frozenset([
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
])

PROMPT = (
    "Classify and describe this surveillance video clip. "
    "Start with [ClassName], then describe the activity. "
    "Classes: Normal, Abuse, Arrest, Arson, Assault, Burglary, Explosion, "
    "Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism."
)

FRAMES_PER_SEC = 4      # 4 frames per second of clip duration
MAX_FRAMES     = 48     # frames per sub-clip (48 × 64 tokens = 3072 visual tokens; leaves 1024 for text in 4096 ctx)
MIN_FRAMES     = 2      # floor for very short clips
MAX_LENGTH     = 4096   # 3072 visual + ~512 text + padding headroom
SEG_DURATION   = MAX_FRAMES / FRAMES_PER_SEC  # 12s — matches dataset median annotation duration (10.9s)
SEG_STRIDE     = SEG_DURATION * 0.75          # 9s stride = 25% overlap between sub-clips
SEED           = 42

# Frame cache: extracted JPEG frames stored on disk, keyed by clip hash.
# Eliminates PyAV video seek/decode on repeated epochs (~4 GB for full dataset).
# Set to None to disable.
FRAME_CACHE_DIR = os.environ.get("FRAME_CACHE_DIR", str(_PIPELINE_ROOT / "frame_cache"))

# Crime weight: multiplier applied to loss on crime-class samples (set from --crime-weight arg)
CRIME_WEIGHT = 3.0

# Per-token weight on [ClassName] bracket tokens — boosts gradient signal for classification.
# Counteracts gradient dilution: class is ~4 tokens vs ~50 description tokens in a typical GT.
CLASS_TOKEN_WEIGHT = 5.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("train_full")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# MLflow callback
# ---------------------------------------------------------------------------

class MLflowMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        step = state.global_step
        # Prefix train vs eval metrics for clarity in MLflow UI
        prefixed = {}
        for k, v in logs.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith("eval_"):
                prefixed[k] = v          # already has eval_ prefix
            else:
                prefixed[f"train/{k}"] = v
        logging.getLogger("train_full").info(
            "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in prefixed.items())
        )
        try:
            mlflow.log_metrics(prefixed, step=step, synchronous=False)
        except Exception as e:
            logging.getLogger("train_full").warning(f"MLflow log_metrics failed (step {step}): {e}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or not state.is_world_process_zero:
            return
        step = state.global_step
        try:
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                step=step,
                synchronous=False,
            )
        except Exception as e:
            logging.getLogger("train_full").warning(f"MLflow eval metrics failed (step {step}): {e}")

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        try:
            mlflow.log_metrics({
                "train/best_eval_loss": state.best_metric,
                "train/total_steps":    state.global_step,
                "train/total_epochs":   state.epoch,
            }, step=state.global_step)
        except Exception as e:
            logging.getLogger("train_full").warning(f"MLflow train_end metrics failed: {e}")


# ---------------------------------------------------------------------------
# Custom trainer: per-sample weighted loss for crime/normal imbalance
# ---------------------------------------------------------------------------

class SurveillanceTrainer(Trainer):
    """Trainer with per-sample crime weighting + per-token class-bracket weighting.

    Batch must contain:
      - 'sample_weight' [B]: crime clips weighted higher than Normal
      - 'token_weight'  [B, T]: [ClassName] bracket tokens weighted higher (counteracts
        gradient dilution where class is ~4 tokens vs ~50 description tokens)

    Loss formulation (matches Idefics3/SmolVLM2 internal shift):
      - Left-shift by 1 for next-token prediction
      - Ignore positions where label == -100 (system/user tokens, padding)
      - Per-token CE × token_weight × valid_mask
      - Per-sample loss = weighted mean over valid tokens
      - Batch loss     = sample_weight-weighted mean over samples
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights      = inputs.pop("sample_weight", None)
        token_weight = inputs.pop("token_weight",  None)
        outputs = model(**inputs)

        # Skip custom weighted loss in eval: gradients don't matter, and the
        # reduction="none" path materializes [B, T, V] tensor — OOMs on large eval batches.
        if weights is None or not model.training:
            loss = outputs.loss
        else:
            logits = outputs.logits          # [B, T, V]
            labels = inputs["labels"]        # [B, T]

            # Causal LM shift: predict token t+1 from logits at t
            shift_logits = logits[..., :-1, :].contiguous()   # [B, T-1, V]
            shift_labels = labels[..., 1:].contiguous()       # [B, T-1]
            # Free original logits immediately — CE upcast to fp32 would otherwise
            # hold both bf16 [B,T,V] and fp32 [B,T,V] simultaneously → OOM
            del logits
            outputs.logits = None

            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            per_token = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_logits.size(0), -1)                  # [B, T-1]

            mask = (shift_labels != -100).float()

            # Per-token weight: class bracket tokens get CLASS_TOKEN_WEIGHT, others 1.0
            if token_weight is not None:
                shift_tw = token_weight[..., 1:].to(per_token.device)
                mask = mask * shift_tw

            denom = mask.sum(-1).clamp(min=1)
            per_sample = (per_token * mask).sum(-1) / denom   # [B]

            loss = (per_sample * weights.to(per_sample.device)).mean()

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _category_from_id(video_id: str) -> str:
    return re.sub(r"\d+_x264$", "", video_id)


def _segment_clip(start: float, end: float, sentence: str,
                  video_path: str) -> list[dict]:
    """Split annotation clip into SEG_DURATION sub-clips, all sharing the GT sentence.
    Ensures every frame of every clip goes into training regardless of clip length.
    """
    duration = end - start
    if duration < 0.5:
        return []

    segments = []
    seg_start = start
    while seg_start < end:
        seg_end = min(seg_start + SEG_DURATION, end)
        if seg_end - seg_start >= 0.5:
            segments.append({
                "video_path": video_path,
                "start":      float(seg_start),
                "end":        float(seg_end),
                "sentence":   sentence,
            })
        seg_start += SEG_STRIDE
    return segments


def _load_samples(json_path: str, video_root: str, max_samples: int) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)

    items = []
    skipped = 0
    n_clips = 0
    for video_id, ann in data.items():
        category   = _category_from_id(video_id)
        video_path = os.path.join(video_root, category, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            skipped += 1
            continue
        for (start, end), sent_entry in zip(ann["timestamps"], ann["sentences"]):
            if end <= start:
                continue
            # Support both old format (str) and new classified format ({"text": ..., "class": ...})
            if isinstance(sent_entry, dict):
                sentence_text  = sent_entry["text"].strip()
                sentence_class = sent_entry.get("class", "Normal")
            else:
                sentence_text  = sent_entry.strip()
                sentence_class = "Normal"
            segs = _segment_clip(float(start), float(end), sentence_text, video_path)
            for seg in segs:
                seg["class"] = sentence_class
            n_clips += 1
            items.extend(segs)

    random.seed(SEED)
    random.shuffle(items)
    logging.getLogger("train_full").info(
        f"  {n_clips} annotations → {len(items)} sub-clips, {skipped} videos not found"
    )
    return items if max_samples == -1 else items[:max_samples]


def build_dataset(json_path: str, video_root: str, max_samples: int, logger) -> Dataset:
    samples = _load_samples(json_path, video_root, max_samples)
    logger.info(f"Dataset: {len(samples)} samples from {Path(json_path).name}")
    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def adaptive_n_frames(start: float, end: float) -> int:
    """1 frame/sec of clip duration, clamped to [MIN_FRAMES, MAX_FRAMES]."""
    n = int(round((end - start) * FRAMES_PER_SEC))
    return max(MIN_FRAMES, min(n, MAX_FRAMES))


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
        logging.getLogger("train_full").warning(f"Frame extraction failed for {video_path}: {e}")

    return [Image.new("RGB", (224, 224), color=0)] * n_frames


def _clip_cache_dir(video_path: str, start: float, end: float,
                    n_frames: int, cache_root: str) -> Path:
    key = f"{video_path}|{start:.3f}|{end:.3f}|{n_frames}|{FRAMES_PER_SEC}"
    h   = hashlib.sha256(key.encode()).hexdigest()[:16]
    return Path(cache_root) / h


def load_or_extract_frames(video_path: str, start: float, end: float,
                            n_frames: int) -> list:
    """Return PIL frames from JPEG cache if available, else extract and cache."""
    if not FRAME_CACHE_DIR:
        return extract_frames(video_path, start, end, n_frames)

    clip_dir = _clip_cache_dir(video_path, start, end, n_frames, FRAME_CACHE_DIR)

    # Cache hit: all N JPEG files present
    jpegs = sorted(clip_dir.glob("frame_*.jpg"))
    if len(jpegs) == n_frames:
        return [Image.open(p).convert("RGB") for p in jpegs]

    # Cache miss: extract then save
    frames = extract_frames(video_path, start, end, n_frames)
    clip_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(frames):
        img.save(clip_dir / f"frame_{i:03d}.jpg", quality=90)
    return frames


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _make_video_metadata(start: float, end: float, n_frames: int) -> VideoMetadata:
    frame_timestamps = [
        start + i * (end - start) / max(n_frames - 1, 1)
        for i in range(n_frames)
    ]
    return VideoMetadata(
        total_num_frames=n_frames,
        fps=1.0,
        frames_indices=list(range(n_frames)),
        duration=float(end - start),
    )


def collate_fn(batch: list[dict], processor, model) -> dict:
    texts       = []
    frame_lists = []
    metadatas   = []

    for sample in batch:
        n_frames = adaptive_n_frames(sample["start"], sample["end"])
        frames = load_or_extract_frames(
            sample["video_path"], sample["start"], sample["end"], n_frames
        )
        frame_lists.append(frames)
        metadatas.append(_make_video_metadata(sample["start"], sample["end"], n_frames))

        cls = sample.get("class", "Normal")
        gt  = f"[{cls}] {sample['sentence']}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": gt}],
            },
        ]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        texts.append(text)

    # Pad all frame lists to same length within batch (duplicate last frame)
    max_n = max(len(f) for f in frame_lists)
    frame_lists_padded = [
        frames + [frames[-1]] * (max_n - len(frames))
        for frames in frame_lists
    ]
    metadatas_padded = [
        _make_video_metadata(s["start"], s["end"], max_n)
        for s in batch
    ]

    encoded = processor(
        text=texts,
        videos=[[frames] for frames in frame_lists_padded],
        video_metadata=metadatas_padded,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    labels = encoded["input_ids"].clone()
    split_positions: list[int | None] = []

    assistant_token = processor.tokenizer.encode("Assistant:", add_special_tokens=False)
    for i, ids in enumerate(labels):
        ids_list  = ids.tolist()
        split_pos = None
        for j in range(len(ids_list) - len(assistant_token), -1, -1):
            if ids_list[j : j + len(assistant_token)] == assistant_token:
                split_pos = j + len(assistant_token)
                break
        split_positions.append(split_pos)
        if split_pos is not None:
            labels[i, :split_pos] = -100
        else:
            labels[i] = torch.full_like(ids, -100)

    labels[labels == processor.tokenizer.pad_token_id] = -100
    encoded["labels"] = labels

    # Per-token weight: upweight [ClassName] bracket tokens (first tokens of GT in class-first format)
    token_weight = torch.ones_like(labels, dtype=torch.float32)
    for i, sample in enumerate(batch):
        sp = split_positions[i]
        if sp is None:
            continue
        cls = sample.get("class", "Normal")
        # Leading space matches context tokenization: chat template renders
        # "Assistant: [Class]" where ' [' is a single combined token
        class_ids = processor.tokenizer.encode(f" [{cls}]", add_special_tokens=False)
        ids_list  = encoded["input_ids"][i].tolist()
        match_pos = None
        for offset in range(4):
            pos = sp + offset
            if pos + len(class_ids) <= len(ids_list) and ids_list[pos : pos + len(class_ids)] == class_ids:
                match_pos = pos
                break
        if match_pos is None:
            match_pos = sp
        end_pos = min(match_pos + len(class_ids), labels.shape[1])
        token_weight[i, match_pos:end_pos] = float(CLASS_TOKEN_WEIGHT)
    encoded["token_weight"] = token_weight

    # Per-sample weight: crime clips weighted higher than Normal
    weights = []
    for sample in batch:
        cls = sample.get("class", "Normal")
        weights.append(1.0 if cls == "Normal" else float(CRIME_WEIGHT))
    encoded["sample_weight"] = torch.tensor(weights, dtype=torch.float32)

    return dict(encoded)


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------

def apply_lora(model, rank: int, logger):
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        logger.error("peft not installed. Run: pip install peft")
        sys.exit(1)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["down_proj", "o_proj", "k_proj", "q_proj",
                        "gate_proj", "up_proj", "v_proj"],
        lora_dropout=0.05,
        use_dora=True,
        init_lora_weights="gaussian",
        bias="none",
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA rank={rank}  trainable={trainable/1e6:.1f}M / {total/1e6:.0f}M "
                f"({100*trainable/total:.1f}%)")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full SFT for SmolVLM2 on UCF-Crime")
    parser.add_argument("--model",      default=_MODEL_ID,
                        help="Model ID or local path")
    parser.add_argument("--output",     default=_OUTPUT_DIR,
                        help="Output directory for checkpoints")
    parser.add_argument("--lora",       action="store_true",
                        help="Use LoRA (PEFT) instead of full fine-tune")
    parser.add_argument("--lora-rank",  type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--max-train",  type=int, default=-1,
                        help="Max training samples (-1 = full dataset)")
    parser.add_argument("--max-val",    type=int, default=-1,
                        help="Max validation samples (-1 = full dataset)")
    parser.add_argument("--epochs",     type=int, default=3,
                        help="Training epochs (default: 3)")
    parser.add_argument("--lr",         type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch",      type=int, default=8,
                        help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--data-root",     default=_DATA_ROOT,
                        help="Root directory of dataset")
    parser.add_argument("--freeze-vision", action="store_true",
                        help="Freeze vision encoder (default: unfrozen — trains full model)")
    parser.add_argument("--frame-cache", default=None,
                        help="Directory to cache extracted JPEG frames (~4 GB for full dataset). "
                             "Skips PyAV decode on epoch 2+. Also settable via FRAME_CACHE_DIR env.")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint dir (e.g. ./output/.../checkpoint-1072). "
                             "Optimizer state loaded if present; otherwise resumes from model weights only.")
    parser.add_argument("--crime-weight", type=float, default=3.0,
                        help="Loss multiplier for crime-class samples vs Normal (default: 3.0). "
                             "Set to 1.0 to disable weighting.")
    parser.add_argument("--class-token-weight", type=float, default=5.0,
                        help="Per-token loss multiplier for [ClassName] bracket tokens (default: 5.0). "
                             "Set to 1.0 to disable.")
    parser.add_argument("--eval-steps", type=int, default=None,
                        help="Override eval/save interval (steps). Default: max(50, steps_per_epoch//5). "
                             "Use a small value (e.g. 5) for local smoke tests.")
    parser.add_argument("--eval-during-training", action="store_true",
                        help="Enable validation loss eval during training (eval_strategy=steps). "
                             "Disabled by default — eval at large batch adds ~25 GB peak VRAM. "
                             "Use with --eval-steps 10 to test eval OOM before full run.")
    parser.add_argument("--no-grad-checkpoint", action="store_true",
                        help="Disable gradient checkpointing. Only use on GPUs with extreme VRAM (>400 GB).")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU training (disables MPS). Slow but avoids MPS OOM on local Mac.")
    parser.add_argument("--train-json", default=None,
                        help="Path to training JSON (default: <data-root>/classified/UCFCrime_Train_deepseek_v4_pro.json)")
    parser.add_argument("--val-json", default=None,
                        help="Path to validation JSON (default: <data-root>/classified/UCFCrime_Val_deepseek_v4_pro.json)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        if args.force_cpu:
            dev = "cpu (forced)"
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        print(f"[warn] CUDA not found — running on {dev}. "
              "For full training use a CUDA GPU.")

    # Apply frame cache config (CLI overrides env)
    global FRAME_CACHE_DIR, CRIME_WEIGHT, CLASS_TOKEN_WEIGHT
    if args.frame_cache:
        FRAME_CACHE_DIR = args.frame_cache
    CRIME_WEIGHT       = args.crime_weight
    CLASS_TOKEN_WEIGHT = args.class_token_weight

    video_root = f"{args.data_root}/UCF_Crimes/UCF_Crimes/Videos"
    train_json = args.train_json or f"{args.data_root}/classified/UCFCrime_Train_deepseek_v4_pro.json"
    val_json   = args.val_json   or f"{args.data_root}/classified/UCFCrime_Val_deepseek_v4_pro.json"

    if args.lora:
        mode_tag = "lora"
    elif args.freeze_vision:
        mode_tag = "full-sft"
    else:
        mode_tag = "full-unfrz-sft"
    model_tag = "500m" if "500M" in args.model or "500m" in args.model else "2b"
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name  = f"smolvlm2-{model_tag}-{mode_tag}-{ts}"

    output_dir = args.output if args.output != _OUTPUT_DIR else f"./output/smolvlm2-{model_tag}-{mode_tag}-{ts}"
    log_file   = os.path.join(output_dir, "logs", f"{run_name}.log")
    logger     = setup_logging(log_file)

    logger.info(f"=== SmolVLM2 Full SFT ===")
    logger.info(f"Model      : {args.model}")
    logger.info(f"Mode       : {'LoRA rank=' + str(args.lora_rank) if args.lora else 'full fine-tune'}")
    logger.info(f"Train      : {'all' if args.max_train == -1 else args.max_train} samples")
    logger.info(f"Val        : {'all' if args.max_val == -1 else args.max_val} samples")
    logger.info(f"Epochs     : {args.epochs}  LR: {args.lr}  Batch: {args.batch}  GradAccum: {args.grad_accum}")
    logger.info(f"Frames     : {FRAMES_PER_SEC}fps  max={MAX_FRAMES}  min={MIN_FRAMES}  seg={SEG_DURATION:.1f}s  stride={SEG_STRIDE:.1f}s  MaxLen: {MAX_LENGTH}")
    logger.info(f"Frame cache: {FRAME_CACHE_DIR or 'disabled'}")
    logger.info(f"Crime weight: {args.crime_weight}x (Normal=1.0)")
    logger.info(f"Class-token weight: {args.class_token_weight}x ([ClassName] bracket tokens)")
    logger.info(f"Output     : {output_dir}")
    if torch.cuda.is_available():
        logger.info(f"GPU        : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        logger.info("Device     : Apple MPS")
    else:
        logger.info("Device     : CPU")

    # --- MLflow ---
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    eff_batch = args.batch * args.grad_accum
    hparams = {
        "model_id":                    args.model,
        "mode":                        mode_tag,
        "lora_rank":                   args.lora_rank if args.lora else None,
        "frames_per_sec":              FRAMES_PER_SEC,
        "max_frames":                  MAX_FRAMES,
        "min_frames":                  MIN_FRAMES,
        "seg_duration_s":              SEG_DURATION,
        "seg_stride_s":                SEG_STRIDE,
        "max_train_samples":           args.max_train,
        "num_epochs":                  args.epochs,
        "learning_rate":               args.lr,
        "batch_size":                  args.batch,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size":        eff_batch,
        "max_length":                  MAX_LENGTH,
        "lr_scheduler":                "cosine",
        "warmup_ratio":                0.05,
        "optimizer":                   "adamw_bnb_8bit" if torch.cuda.is_available() else "adamw_torch",
        "precision":                   "bf16" if torch.cuda.is_available() else ("bf16-mps" if torch.backends.mps.is_available() else "fp32"),
        "device":                      (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")),
        "vram_gb":                     (round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None),
        "task":                        "activity_description",
        "freeze_vision":               args.freeze_vision,
        "dataloader_workers":          4,
        "frame_cache":                 bool(FRAME_CACHE_DIR),
        "seed":                        SEED,
        "crime_weight":                args.crime_weight,
        "class_token_weight":          args.class_token_weight,
    }

    run = mlflow.start_run(run_name=run_name)
    try:
        logger.info(f"MLflow run: {run.info.run_id}")
        try:
            mlflow.log_params({k: v for k, v in hparams.items() if v is not None})
        except Exception as e:
            logger.warning(f"MLflow log_params failed: {e}")

        # Suppress noisy transformers warnings (processor kwargs deprecation spam)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        # --- Model & processor ---
        logger.info("Loading model and processor...")
        processor = AutoProcessor.from_pretrained(args.model)

        # flash_attention_2 > sdpa > eager (fallback chain)
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError as e:
            attn_impl = "sdpa" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "eager"
            logger.warning(f"flash-attn not available ({e}), falling back to {attn_impl}")
        logger.info(f"Attention impl: {attn_impl}")
        try:
            mlflow.log_param("attn_impl", attn_impl)
        except Exception:
            pass

        _use_accel = torch.cuda.is_available() or (torch.backends.mps.is_available() and not args.force_cpu)
        _dtype = torch.bfloat16 if _use_accel else torch.float32
        _device_map = "auto" if torch.cuda.is_available() else None
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=_dtype,
            _attn_implementation=attn_impl,
            device_map=_device_map,
        )
        if args.force_cpu:
            model = model.to("cpu")

        if args.lora:
            model = apply_lora(model, args.lora_rank, logger)
        else:
            if args.freeze_vision:
                for param in model.model.vision_model.parameters():
                    param.requires_grad = False
                logger.info("Vision encoder: FROZEN")
            else:
                logger.info("Vision encoder: UNFROZEN (full model trains)")
            if torch.cuda.is_available() and not args.no_grad_checkpoint:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                logger.info("Gradient checkpointing: ENABLED")
            elif torch.cuda.is_available():
                logger.info("Gradient checkpointing: DISABLED (--no-grad-checkpoint)")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Params: {total_params/1e6:.0f}M total, {trainable_params/1e6:.1f}M trainable")
        try:
            mlflow.log_params({
                "total_params_m":     round(total_params / 1e6, 1),
                "trainable_params_m": round(trainable_params / 1e6, 1),
                "trainable_pct":      round(100 * trainable_params / total_params, 1),
            })
        except Exception:
            pass

        # --- Dataset ---
        logger.info("Building datasets...")
        train_ds = build_dataset(train_json, video_root, args.max_train, logger)
        val_ds   = build_dataset(val_json,   video_root, args.max_val,   logger)

        # Eval every ~20% of training steps, save best checkpoint
        steps_per_epoch = max(1, len(train_ds) // (args.batch * args.grad_accum))
        eval_steps = args.eval_steps if args.eval_steps else min(steps_per_epoch, 100)
        save_steps = eval_steps

        logger.info(f"Steps/epoch: {steps_per_epoch}  Eval every: {eval_steps} steps")
        try:
            mlflow.log_params({
                "train_samples": len(train_ds),
                "val_samples":   len(val_ds),
                "steps_per_epoch": steps_per_epoch,
                "eval_steps":    eval_steps,
            })
        except Exception:
            pass

        os.makedirs(output_dir, exist_ok=True)

        _use_cuda = torch.cuda.is_available()
        _use_mps  = (not _use_cuda) and torch.backends.mps.is_available() and not args.force_cpu
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            use_cpu=args.force_cpu,
            optim="adamw_bnb_8bit" if _use_cuda else "adamw_torch",
            bf16=_use_cuda,
            max_grad_norm=1.0,
            logging_steps=10,
            save_steps=save_steps,
            eval_strategy="steps" if args.eval_during_training else "no",
            eval_steps=eval_steps if args.eval_during_training else None,
            load_best_model_at_end=False,
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_num_workers=8,
            dataloader_prefetch_factor=2,
            report_to="none",
            disable_tqdm=False,
            seed=SEED,
        )

        collator = functools.partial(collate_fn, processor=processor, model=model)

        trainer = SurveillanceTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            callbacks=[MLflowMetricsCallback()],
        )

        # --- Train ---
        resume = args.resume or True  # True = auto-detect last checkpoint in output_dir
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=resume if args.resume else None)

        try:
            mlflow.log_metrics({
                "train_loss":               train_result.training_loss,
                "train_runtime_seconds":    train_result.metrics["train_runtime"],
                "train_samples_per_second": train_result.metrics["train_samples_per_second"],
            })
        except Exception as e:
            logger.warning(f"MLflow final metrics failed: {e}")

        logger.info(f"Final train_loss : {train_result.training_loss:.4f}")
        logger.info(f"Runtime          : {train_result.metrics['train_runtime']:.0f}s")

        # --- Save ---
        logger.info("Saving model...")
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)

        # Save LoRA config alongside for easy loading
        if args.lora:
            import json as _json
            lora_meta = {"base_model": args.model, "lora_rank": args.lora_rank}
            with open(os.path.join(output_dir, "lora_meta.json"), "w") as f:
                _json.dump(lora_meta, f, indent=2)

        try:
            mlflow.log_artifact(log_file, artifact_path="logs")
        except Exception as e:
            logger.warning(f"MLflow artifact log failed: {e}")

        # Log trainer_state.json and training_args as artifacts
        for artifact_name in ("trainer_state.json", "training_args.bin"):
            artifact_path = os.path.join(output_dir, artifact_name)
            if os.path.isfile(artifact_path):
                try:
                    mlflow.log_artifact(artifact_path, artifact_path="checkpoints")
                except Exception as e:
                    logger.warning(f"MLflow artifact {artifact_name} failed: {e}")

        # Save training_args as JSON for readability
        try:
            training_args_json = os.path.join(output_dir, "training_args.json")
            with open(training_args_json, "w") as f:
                json.dump(training_args.to_dict(), f, indent=2, default=str)
            mlflow.log_artifact(training_args_json, artifact_path="checkpoints")
        except Exception as e:
            logger.warning(f"MLflow training_args.json failed: {e}")

        logger.info(f"Done. Checkpoint: {output_dir}")

    finally:
        try:
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"MLflow end_run failed: {e}")


if __name__ == "__main__":
    main()
