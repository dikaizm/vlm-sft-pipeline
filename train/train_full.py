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

import mlflow
import torch
from PIL import Image
from datasets import Dataset
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
_TRAIN_JSON   = f"{_DATA_ROOT}/UCFCrime_Train.json"
_VAL_JSON     = f"{_DATA_ROOT}/UCFCrime_Val.json"
_OUTPUT_DIR   = os.environ.get("OUTPUT_DIR", str(_PIPELINE_ROOT / "output" / "smolvlm2-full-sft"))
_MODEL_ID     = os.environ.get("MODEL_ID",   "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

MLFLOW_URI        = os.environ.get("MLFLOW_URI",        "https://mlflow-geoai.stelarea.com/")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "smolvlm2-surveillance-sft")

FRAMES_PER_SEC = 4      # 4 frames per second of clip duration
MAX_FRAMES     = 32     # cap — 32 frames × 64 tokens = 2048 visual tokens, fits in MAX_LENGTH=4096
MIN_FRAMES     = 2      # floor for very short clips
MAX_LENGTH     = 4096   # 1024 visual + ~512 text + padding headroom
SEED        = 42

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
        if not logs:
            return
        step = state.global_step
        metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        logging.getLogger("train_full").info(
            "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in metrics.items())
        )
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logging.getLogger("train_full").warning(f"MLflow log_metrics failed (step {step}): {e}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _category_from_id(video_id: str) -> str:
    return re.sub(r"\d+_x264$", "", video_id)


def _load_samples(json_path: str, video_root: str, max_samples: int) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)

    items = []
    skipped = 0
    for video_id, ann in data.items():
        category   = _category_from_id(video_id)
        video_path = os.path.join(video_root, category, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            skipped += 1
            continue
        for (start, end), sentence in zip(ann["timestamps"], ann["sentences"]):
            if end <= start:
                continue
            # Skip clips shorter than 0.5s (sparse keyframe risk)
            if (end - start) < 0.5:
                continue
            items.append({
                "video_path": video_path,
                "start":      float(start),
                "end":        float(end),
                "sentence":   sentence.strip(),
            })

    random.seed(SEED)
    random.shuffle(items)
    logging.getLogger("train_full").info(
        f"  {len(items)} clips loaded, {skipped} videos not found"
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
        frames = extract_frames(
            sample["video_path"], sample["start"], sample["end"], n_frames
        )
        frame_lists.append(frames)
        metadatas.append(_make_video_metadata(sample["start"], sample["end"], n_frames))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": "Describe the activity shown in this surveillance video clip."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["sentence"]}],
            },
        ]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        texts.append(text)

    encoded = processor(
        text=texts,
        videos=[[frames] for frames in frame_lists],
        video_metadata=metadatas,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    labels = encoded["input_ids"].clone()

    assistant_token = processor.tokenizer.encode("Assistant:", add_special_tokens=False)
    for i, ids in enumerate(labels):
        ids_list  = ids.tolist()
        split_pos = None
        for j in range(len(ids_list) - len(assistant_token), -1, -1):
            if ids_list[j : j + len(assistant_token)] == assistant_token:
                split_pos = j + len(assistant_token) + 1
                break
        if split_pos is not None:
            labels[i, :split_pos] = -100
        else:
            labels[i] = torch.full_like(ids, -100)

    labels[labels == processor.tokenizer.pad_token_id] = -100
    encoded["labels"] = labels

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
                        help="Max validation samples (-1 = full)")
    parser.add_argument("--epochs",     type=int, default=3,
                        help="Training epochs (default: 3)")
    parser.add_argument("--lr",         type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch",      type=int, default=1,
                        help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--data-root",  default=_DATA_ROOT,
                        help="Root directory of dataset")
    args = parser.parse_args()

    assert torch.cuda.is_available(), (
        "CUDA not found. train_full.py requires a CUDA GPU.\n"
        "Check: nvidia-smi"
    )

    video_root = f"{args.data_root}/UCF_Crimes/UCF_Crimes/Videos"
    train_json = f"{args.data_root}/UCFCrime_Train.json"
    val_json   = f"{args.data_root}/UCFCrime_Val.json"

    mode_tag = "lora" if args.lora else "full"
    model_tag = "500m" if "500M" in args.model or "500m" in args.model else "2b"
    run_name  = f"smolvlm2-{model_tag}-{mode_tag}-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    output_dir = args.output if args.output != _OUTPUT_DIR else f"./output/smolvlm2-{model_tag}-{mode_tag}-sft"
    log_file   = os.path.join(output_dir, "logs", f"{run_name}.log")
    logger     = setup_logging(log_file)

    logger.info(f"=== SmolVLM2 Full SFT ===")
    logger.info(f"Model      : {args.model}")
    logger.info(f"Mode       : {'LoRA rank=' + str(args.lora_rank) if args.lora else 'full fine-tune'}")
    logger.info(f"Train      : {'all' if args.max_train == -1 else args.max_train} samples")
    logger.info(f"Val        : {'all' if args.max_val == -1 else args.max_val} samples")
    logger.info(f"Epochs     : {args.epochs}  LR: {args.lr}  Batch: {args.batch}  GradAccum: {args.grad_accum}")
    logger.info(f"Frames     : {FRAMES_PER_SEC}fps  max={MAX_FRAMES}  min={MIN_FRAMES}  MaxLen: {MAX_LENGTH}")
    logger.info(f"Output     : {output_dir}")
    logger.info(f"GPU        : {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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
        "max_train_samples":           args.max_train,
        "num_epochs":                  args.epochs,
        "learning_rate":               args.lr,
        "batch_size":                  args.batch,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size":        eff_batch,
        "max_length":                  MAX_LENGTH,
        "lr_scheduler":                "cosine",
        "warmup_ratio":                0.03,
        "optimizer":                   "adamw_bnb_8bit",
        "precision":                   "bf16",
        "device":                      torch.cuda.get_device_name(0),
        "task":                        "activity_description",
        "seed":                        SEED,
    }

    run = mlflow.start_run(run_name=run_name)
    try:
        logger.info(f"MLflow run: {run.info.run_id}")
        try:
            mlflow.log_params({k: v for k, v in hparams.items() if v is not None})
        except Exception as e:
            logger.warning(f"MLflow log_params failed: {e}")

        # --- Model & processor ---
        logger.info("Loading model and processor...")
        processor = AutoProcessor.from_pretrained(args.model)

        # flash_attention_2 > sdpa > eager (fallback chain)
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError as e:
            attn_impl = "sdpa" if torch.cuda.is_bf16_supported() else "eager"
            logger.warning(f"flash-attn not available ({e}), falling back to {attn_impl}")
        logger.info(f"Attention impl: {attn_impl}")

        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            _attn_implementation=attn_impl,
            device_map="auto",
        )

        if args.lora:
            model = apply_lora(model, args.lora_rank, logger)
        else:
            # Freeze vision encoder — only fine-tune LLM layers
            for param in model.model.vision_model.parameters():
                param.requires_grad = False
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Params: {total_params/1e6:.0f}M total, {trainable_params/1e6:.1f}M trainable")

        # --- Dataset ---
        logger.info("Building datasets...")
        train_ds = build_dataset(train_json, video_root, args.max_train, logger)
        val_ds   = build_dataset(val_json,   video_root, args.max_val,   logger)

        # Eval every ~10% of training steps, save best checkpoint
        steps_per_epoch = max(1, len(train_ds) // (args.batch * args.grad_accum))
        eval_steps = max(50, steps_per_epoch // 10)
        save_steps = eval_steps

        logger.info(f"Steps/epoch: {steps_per_epoch}  Eval every: {eval_steps} steps")

        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            optim="adamw_bnb_8bit",
            bf16=True,
            max_grad_norm=1.0,
            logging_steps=10,
            save_steps=save_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            report_to="none",
            seed=SEED,
        )

        collator = functools.partial(collate_fn, processor=processor, model=model)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            callbacks=[MLflowMetricsCallback()],
        )

        # --- Train ---
        logger.info("Starting training...")
        train_result = trainer.train()

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

        logger.info(f"Done. Checkpoint: {output_dir}")

    finally:
        try:
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"MLflow end_run failed: {e}")


if __name__ == "__main__":
    main()
