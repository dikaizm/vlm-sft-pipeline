"""
Small SFT pipeline for SmolVLM2-500M on UCF-Crime + UCA dataset.
Trains on a 200-sample subset to validate the full pipeline end-to-end.

Logs metrics, hyperparameters, and a training log file to MLflow.

Usage:
    python vlm_sft_pipeline/train_small.py
"""

import json
import logging
import re
import random
import functools
import os
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import numpy as np
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
# Config
# ---------------------------------------------------------------------------

DATA_ROOT    = "/Volumes/T7/research-vlm/data"
VIDEO_ROOT   = f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos"
TRAIN_JSON   = f"{DATA_ROOT}/UCFCrime_Train.json"
VAL_JSON     = f"{DATA_ROOT}/UCFCrime_Val.json"
OUTPUT_DIR   = "./output/smolvlm2-500m-small-sft"

MODEL_ID     = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
NUM_FRAMES   = 4       # 8 → 4 to halve visual token count (MPS memory limit)
MAX_TRAIN    = 200
MAX_VAL      = 50
SEED         = 42

MLFLOW_URI        = "https://mlflow-geoai.stelarea.com/"
MLFLOW_EXPERIMENT = "smolvlm2-surveillance-sft"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    """Write logs to both stdout and a file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    # file handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    # stdout handler
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# MLflow callback
# ---------------------------------------------------------------------------

class MLflowMetricsCallback(TrainerCallback):
    """Logs per-step and per-epoch metrics to the active MLflow run.
    Network errors are swallowed so a transient outage never kills training."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        metrics = {
            k: v for k, v in logs.items()
            if isinstance(v, (int, float))
        }
        # Print to our logger so progress is visible in the log file
        logging.getLogger("train").info(
            "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in metrics.items())
        )
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logging.getLogger("train").warning(f"MLflow log_metrics failed (step {step}): {e}")


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    # MPS does not support AMP gradient scaling — train in float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _category_from_id(video_id: str) -> str:
    return re.sub(r"\d+_x264$", "", video_id)


def _load_samples(json_path: str, max_samples: int) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)

    items = []
    for video_id, ann in data.items():
        category  = _category_from_id(video_id)
        video_path = os.path.join(VIDEO_ROOT, category, f"{video_id}.mp4")

        if not os.path.isfile(video_path):
            continue

        for (start, end), sentence in zip(ann["timestamps"], ann["sentences"]):
            if end <= start:
                continue
            items.append({
                "video_path": video_path,
                "start":      float(start),
                "end":        float(end),
                "sentence":   sentence.strip(),
            })

    random.seed(SEED)
    random.shuffle(items)
    return items[:max_samples]


def build_dataset(json_path: str, max_samples: int, logger) -> Dataset:
    samples = _load_samples(json_path, max_samples)
    logger.info(f"Loaded {len(samples)} samples from {Path(json_path).name}")
    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, start: float, end: float, n_frames: int) -> list:
    """Return n_frames PIL Images sampled uniformly from [start, end] seconds."""
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
        logging.getLogger("train").warning(f"Frame extraction failed for {video_path}: {e}")

    return [Image.new("RGB", (224, 224), color=0)] * n_frames


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def _make_video_metadata(start: float, end: float, n_frames: int) -> VideoMetadata:
    """Build VideoMetadata with absolute frame timestamps (fps=1, integer seconds)."""
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


def collate_fn(batch: list[dict], processor, model) -> dict:
    texts       = []
    frame_lists = []
    metadatas   = []

    for sample in batch:
        frames = extract_frames(
            sample["video_path"], sample["start"], sample["end"], NUM_FRAMES
        )
        frame_lists.append(frames)
        metadatas.append(_make_video_metadata(sample["start"], sample["end"], NUM_FRAMES))

        # Response includes description + timestamps
        response = (
            f"{sample['sentence']} "
            f"Timestamps: [{sample['start']:.1f}, {sample['end']:.1f}]"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": (
                        "Describe the activity in this surveillance video clip "
                        "and provide the start and end timestamps in seconds."
                    )},
                ],
            },
            {
                "role": "assistant",
                # Must be a list — plain string silently drops the text in SmolVLM2's template
                "content": [{"type": "text", "text": response}],
            },
        ]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        texts.append(text)

    # videos must be wrapped as [[frames], [frames], ...] — one list-of-frames per video
    # video_metadata: one VideoMetadata per video in the batch
    encoded = processor(
        text=texts,
        videos=[[frames] for frames in frame_lists],
        video_metadata=metadatas,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )

    labels = encoded["input_ids"].clone()

    # Mask prompt tokens: only compute loss on assistant response.
    # SmolVLM2 template: "Assistant: {response}<end_of_utterance>"
    assistant_token = processor.tokenizer.encode("Assistant:", add_special_tokens=False)
    for i, ids in enumerate(labels):
        ids_list  = ids.tolist()
        split_pos = None
        for j in range(len(ids_list) - len(assistant_token), -1, -1):
            if ids_list[j : j + len(assistant_token)] == assistant_token:
                split_pos = j + len(assistant_token) + 1  # +1 to skip the space
                break
        if split_pos is not None:
            labels[i, :split_pos] = -100
        else:
            labels[i] = torch.full_like(ids, -100)

    labels[labels == processor.tokenizer.pad_token_id] = -100
    encoded["labels"] = labels

    return {k: v.to(model.device) for k, v in encoded.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    run_name = f"smolvlm2-500m-small-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_file = os.path.join(OUTPUT_DIR, "logs", f"{run_name}.log")
    logger   = setup_logging(log_file)

    logger.info("=== SmolVLM2-500M Small SFT ===")
    logger.info(f"Model  : {MODEL_ID}")
    logger.info(f"Train  : {MAX_TRAIN} samples | Val: {MAX_VAL} samples")
    logger.info(f"Output : {OUTPUT_DIR}")
    logger.info(f"Log    : {log_file}")

    # --- Device ---
    device = get_device()
    dtype  = get_dtype(device)
    logger.info(f"Device : {device} | dtype: {dtype}")

    # --- MLflow ---
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    hparams = {
        "model_id":           MODEL_ID,
        "num_frames":         NUM_FRAMES,
        "max_train_samples":  MAX_TRAIN,
        "num_epochs":         10,
        "learning_rate":      2e-5,
        "batch_size":         1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 4,
        "max_length":         1024,
        "lr_scheduler":       "cosine",
        "warmup_steps":       20,
        "device":             str(device),
        "dtype":              str(dtype),
        "task":               "captioning+temporal_grounding",
        "seed":               SEED,
    }

    run = mlflow.start_run(run_name=run_name)
    try:
        logger.info(f"MLflow run: {run.info.run_id}  ({MLFLOW_URI})")

        try:
            mlflow.log_params(hparams)
        except Exception as e:
            logger.warning(f"MLflow log_params failed: {e}")

        # --- Model & processor ---
        logger.info("Loading model and processor...")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model     = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype)
        model     = model.to(device)
        model.gradient_checkpointing_enable()
        logger.info(f"Params : {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

        # --- Dataset ---
        logger.info("Building datasets...")
        train_ds = build_dataset(TRAIN_JSON, MAX_TRAIN, logger)
        val_ds   = build_dataset(VAL_JSON,   MAX_VAL,   logger)

        # --- Training args ---
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=10,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            warmup_steps=20,
            bf16=(device.type == "cuda" and torch.cuda.is_bf16_supported()),
            fp16=(device.type == "cuda" and not torch.cuda.is_bf16_supported()),
            logging_steps=1,
            save_steps=50,
            eval_strategy="no",       # eval OOMs on MPS without grad checkpointing
            remove_unused_columns=False,
            dataloader_num_workers=0,
            report_to="none",         # metrics logged via MLflowMetricsCallback instead
        )

        collator = functools.partial(collate_fn, processor=processor, model=model)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=collator,
            callbacks=[MLflowMetricsCallback()],
        )

        # --- Train ---
        logger.info("Starting training...")
        train_result = trainer.train()

        # Log final summary metrics
        try:
            mlflow.log_metrics({
                "train_loss":               train_result.training_loss,
                "train_runtime_seconds":    train_result.metrics["train_runtime"],
                "train_samples_per_second": train_result.metrics["train_samples_per_second"],
            })
        except Exception as e:
            logger.warning(f"MLflow final log_metrics failed: {e}")

        logger.info(f"Final train_loss : {train_result.training_loss:.4f}")
        logger.info(f"Runtime          : {train_result.metrics['train_runtime']:.0f}s")

        # --- Save model ---
        logger.info("Saving final model...")
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)

        # --- Upload log file as artifact ---
        try:
            mlflow.log_artifact(log_file, artifact_path="logs")
        except Exception as e:
            logger.warning(f"MLflow log_artifact failed: {e}")
        logger.info(f"Done. Checkpoint: {OUTPUT_DIR}")
        logger.info(f"MLflow run URL: {MLFLOW_URI}#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
    finally:
        try:
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"MLflow end_run failed (run already complete): {e}")


if __name__ == "__main__":
    main()
