"""
Dense Video Captioning SFT — SmolVLM2-500M on UCF-Crime / UCA dataset.

Unlike train_small.py (which trains on single pre-segmented clips), this script
trains the model to detect ALL activities in a full video and output them as a
numbered list with timestamps:

    1. [0.0, 5.3] A woman walks across the parking lot.
    2. [7.0, 8.5] A man in white pushes another person.
    ...

Each training sample = one full video with ALL its annotations grouped together.
This teaches the model to find crimes in raw footage — not just describe given clips.

Usage:
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/train_dense.py
"""

import json
import logging
import re
import random
import functools
import os
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
# Config
# ---------------------------------------------------------------------------

DATA_ROOT  = os.environ["DATA_ROOT"]
VIDEO_ROOT = f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos"
TRAIN_JSON = f"{DATA_ROOT}/UCFCrime_Train.json"
VAL_JSON   = f"{DATA_ROOT}/UCFCrime_Val.json"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR",   "./output/smolvlm2-500m-dense-sft")
MODEL_ID   = os.environ.get("MODEL_ID",     "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

MLFLOW_URI        = os.environ.get("MLFLOW_URI",        "https://mlflow-geoai.stelarea.com/")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "smolvlm2-surveillance-dense")

NUM_FRAMES      = 16
MAX_LENGTH      = 4096
MAX_TRAIN       = 200      # videos (not clips); each has ~13 annotations
MAX_VAL         = 50
MAX_DURATION    = 120.0    # seconds — cap video length for VRAM
MAX_ANNOTATIONS = 12       # cap annotations per video
SEED            = 42

DENSE_PROMPT = (
    "Describe ALL activities in this surveillance video. "
    "For each activity, provide a description and its start and end timestamps in seconds. "
    "List them in chronological order."
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("train_dense")
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
        logging.getLogger("train_dense").info(
            "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in metrics.items())
        )
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logging.getLogger("train_dense").warning(f"MLflow log_metrics failed (step {step}): {e}")


# ---------------------------------------------------------------------------
# Dataset — group by video (dense captioning)
# ---------------------------------------------------------------------------

def _category_from_id(video_id: str) -> str:
    return re.sub(r"\d+_x264$", "", video_id)


def _load_video_samples(json_path: str, max_videos: int) -> list[dict]:
    """Each sample = one full video with ALL its annotations.

    Returns one dict per video:
      {
        "video_id":      str,
        "video_path":    str,
        "duration":      float,
        "effective_end": float,          # min(duration, MAX_DURATION)
        "timestamps":    [[s, e], ...],  # sorted by start
        "sentences":     [str, ...],
      }
    """
    with open(json_path) as f:
        data = json.load(f)

    samples = []
    for video_id, ann in data.items():
        category   = _category_from_id(video_id)
        video_path = os.path.join(VIDEO_ROOT, category, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            continue

        duration      = float(ann.get("duration", MAX_DURATION))
        effective_end = min(duration, MAX_DURATION)

        # Collect valid annotations within effective_end
        pairs = []
        for (start, end), sentence in zip(ann["timestamps"], ann["sentences"]):
            start, end = float(start), float(end)
            if end <= start:
                continue
            if start > effective_end:
                continue
            end = min(end, effective_end)
            pairs.append((start, end, sentence.strip()))

        if not pairs:
            continue

        # Sort chronologically, cap at MAX_ANNOTATIONS
        pairs.sort(key=lambda x: (x[0], x[1]))
        pairs = pairs[:MAX_ANNOTATIONS]

        samples.append({
            "video_id":      video_id,
            "video_path":    video_path,
            "duration":      duration,
            "effective_end": effective_end,
            "timestamps":    [[s, e] for s, e, _ in pairs],
            "sentences":     [sent for _, _, sent in pairs],
        })

    random.seed(SEED)
    random.shuffle(samples)
    return samples if max_videos == -1 else samples[:max_videos]


def build_dataset(json_path: str, max_videos: int, logger) -> Dataset:
    samples = _load_video_samples(json_path, max_videos)
    total_annotations = sum(len(s["timestamps"]) for s in samples)
    logger.info(
        f"Loaded {len(samples)} videos ({total_annotations} annotations) "
        f"from {Path(json_path).name}"
    )
    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Frame extraction (verbatim from train_small.py)
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
        logging.getLogger("train_dense").warning(f"Frame extraction failed for {video_path}: {e}")

    return [Image.new("RGB", (224, 224), color=0)] * n_frames


# ---------------------------------------------------------------------------
# VideoMetadata (verbatim from train_small.py)
# ---------------------------------------------------------------------------

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
# Collate function — dense multi-activity response
# ---------------------------------------------------------------------------

def collate_fn_dense(batch: list[dict], processor, model) -> dict:
    texts       = []
    frame_lists = []
    metadatas   = []

    for sample in batch:
        effective_end = sample["effective_end"]

        # Extract frames from full video span [0, effective_end]
        frames = extract_frames(sample["video_path"], 0.0, effective_end, NUM_FRAMES)
        frame_lists.append(frames)
        metadatas.append(_make_video_metadata(0.0, effective_end, NUM_FRAMES))

        # Build numbered-list response: "1. [s, e] description\n2. ..."
        lines = []
        for idx, (ts, sent) in enumerate(
            zip(sample["timestamps"], sample["sentences"]), 1
        ):
            start, end = ts[0], ts[1]
            lines.append(f"{idx}. [{start:.1f}, {end:.1f}] {sent}")
        response = "\n".join(lines)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": DENSE_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
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

    # Label masking — only train on assistant response (verbatim from train_small.py)
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

    return {k: v.to(model.device) for k, v in encoded.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    assert torch.cuda.is_available(), (
        "CUDA not found. train_dense.py requires a CUDA GPU (A40 recommended).\n"
        "Check your environment: nvidia-smi"
    )

    run_name = f"smolvlm2-500m-dense-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_file = os.path.join(OUTPUT_DIR, "logs", f"{run_name}.log")
    logger   = setup_logging(log_file)

    logger.info("=== SmolVLM2-500M Dense Video Captioning SFT ===")
    logger.info(f"Model         : {MODEL_ID}")
    logger.info(f"Train         : {MAX_TRAIN} videos | Val: {MAX_VAL} videos")
    logger.info(f"Max duration  : {MAX_DURATION}s | Max annotations: {MAX_ANNOTATIONS}")
    logger.info(f"Frames        : {NUM_FRAMES} | Max length: {MAX_LENGTH}")
    logger.info(f"Output        : {OUTPUT_DIR}")
    logger.info(f"GPU           : {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- MLflow ---
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    hparams = {
        "model_id":                    MODEL_ID,
        "num_frames":                  NUM_FRAMES,
        "max_train_videos":            MAX_TRAIN,
        "max_duration_seconds":        MAX_DURATION,
        "max_annotations_per_video":   MAX_ANNOTATIONS,
        "num_epochs":                  5,
        "learning_rate":               2e-5,
        "batch_size":                  4,
        "gradient_accumulation_steps": 2,
        "effective_batch_size":        8,
        "max_length":                  MAX_LENGTH,
        "lr_scheduler":                "cosine",
        "warmup_steps":                30,
        "optimizer":                   "adamw_bnb_8bit",
        "precision":                   "bf16",
        "device":                      torch.cuda.get_device_name(0),
        "task":                        "dense_video_captioning",
        "seed":                        SEED,
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
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

        # --- Dataset ---
        logger.info("Building datasets...")
        train_ds = build_dataset(TRAIN_JSON, MAX_TRAIN, logger)
        val_ds   = build_dataset(VAL_JSON,   MAX_VAL,   logger)

        # --- Training args ---
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            num_train_epochs=5,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            warmup_steps=30,
            optim="adamw_bnb_8bit",
            bf16=True,
            max_grad_norm=1.0,
            logging_steps=5,
            save_steps=100,
            eval_strategy="steps",
            eval_steps=100,
            remove_unused_columns=False,
            dataloader_num_workers=2,
            report_to="none",
        )

        collator = functools.partial(collate_fn_dense, processor=processor, model=model)

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
            logger.warning(f"MLflow final log_metrics failed: {e}")

        logger.info(f"Final train_loss : {train_result.training_loss:.4f}")
        logger.info(f"Runtime          : {train_result.metrics['train_runtime']:.0f}s")

        # --- Save model ---
        logger.info("Saving final model...")
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)

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
            logger.warning(f"MLflow end_run failed: {e}")


if __name__ == "__main__":
    main()
