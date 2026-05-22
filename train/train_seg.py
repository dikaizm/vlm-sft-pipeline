"""
Per-segment Activity Description SFT — SmolVLM2 on UCA dataset.

Each training sample = one annotation segment [t_start, t_end].
Model sees only that clip and outputs:
    <t_start><t_end> [Category] description

Key improvements over train_desc.py:
- Per-segment samples: forces temporal grounding (model must localize within clip)
- Category-balanced sampling: fixes 13-class imbalance via WeightedRandomSampler
- Full fine-tune including vision encoder: critical for crime category visual features
- PyAV frame extraction per segment: no MAX_DURATION truncation needed

Usage:
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/train_seg.py
    MODEL_ID=HuggingFaceTB/SmolVLM2-500M-Video-Instruct DATA_ROOT=/path/to/data python vlm-sft-pipeline/train_seg.py
"""

import json
import logging
import re
import random
import functools
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import av
import mlflow
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler
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

from config import DATA_ROOT, VIDEO_ROOT, TRAIN_JSON, VAL_JSON

OUTPUT_DIR = os.environ.get("OUTPUT_DIR",   "./output/smolvlm2-seg-sft")
MODEL_ID   = os.environ.get("MODEL_ID",     "HuggingFaceTB/SmolVLM2-2.2B-Video-Instruct")

MLFLOW_URI        = os.environ.get("MLFLOW_URI",        "https://mlflow-geoai.stelarea.com/")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "smolvlm2-surveillance-seg")

SEED       = 42
NUM_FRAMES = 16    # frames per segment (shorter clips → 16 sufficient)
MAX_LENGTH = 2048
MAX_TRAIN  = -1    # max unique videos to draw segments from (-1 = all)
MAX_VAL    = -1


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("train_seg")
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
        trainer = kwargs.get("trainer")
        if trainer is not None and hasattr(trainer.optimizer, "param_groups"):
            metrics["learning_rate"] = trainer.optimizer.param_groups[0]["lr"]
        logger = logging.getLogger("train_seg")
        logger.info(
            "  ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in sorted(metrics.items()))
        )
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"MLflow log_metrics failed (step {step}): {e}")


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, t_start: float, t_end: float, n_frames: int) -> list:
    try:
        container = av.open(video_path)
        stream    = container.streams.video[0]
        duration  = float(stream.duration * stream.time_base) if stream.duration else t_end

        t_start = max(0.0, min(t_start, duration))
        t_end   = max(t_start + 0.1, min(t_end, duration))

        collected = {}
        container.seek(int(t_start * 1_000_000), any_frame=False, backward=True)

        for frame in container.decode(video=0):
            t = float(frame.pts * stream.time_base)
            if t > t_end + 1.0:
                break
            span = t_end - t_start + 1e-9
            slot = int((t - t_start) / span * n_frames)
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
        logging.getLogger("train_seg").warning(f"Frame extraction failed ({video_path}): {e}")

    return [Image.new("RGB", (224, 224), color=0)] * n_frames


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(t_start: float, t_end: float, n_frames: int) -> VideoMetadata:
    step = (t_end - t_start) / max(n_frames - 1, 1)
    frame_ts = [t_start + i * step for i in range(n_frames)]
    return VideoMetadata(
        total_num_frames=max(int(t_end * 30), n_frames),
        fps=30.0,
        frames_indices=[round(t * 30) for t in frame_ts],
        duration=float(t_end),
    )


def _make_seg_prompt(t_start: float, t_end: float, n_frames: int) -> str:
    """Prompt with absolute frame timestamps so model can output grounded timestamps."""
    step = (t_end - t_start) / max(n_frames - 1, 1)
    frame_ts = [round(t_start + i * step) for i in range(n_frames)]
    frame_ctx = (
        f"Clip: {int(t_start)}s to {int(t_end)}s. "
        f"Frames at: {', '.join(str(t) + 's' for t in frame_ts)}."
    )
    return (
        f"{frame_ctx}\n"
        "Describe the activity in this surveillance clip. "
        "Format: '<t_start><t_end> [Category] description' using the timestamps above. "
        "If none, write 'None detected.'"
    )


def _category_from_id(video_id: str) -> str:
    return re.sub(r"\d+_x264$", "", video_id)


# ---------------------------------------------------------------------------
# Dataset — per-segment samples
# ---------------------------------------------------------------------------

def _load_seg_samples(json_path: str, max_videos: int) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)

    mp4_map = {}
    for root_dir, _, files in os.walk(VIDEO_ROOT):
        for fname in files:
            if fname.endswith(".mp4"):
                mp4_map[fname] = os.path.join(root_dir, fname)

    # Collect per-video first, then expand to per-segment
    videos = []
    for video_id, ann in data.items():
        category   = _category_from_id(video_id)
        video_path = os.path.join(VIDEO_ROOT, category, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            fallback = mp4_map.get(f"{video_id}.mp4")
            if fallback:
                video_path = fallback
            else:
                continue
        videos.append((video_id, category, video_path, ann))

    random.seed(SEED)
    random.shuffle(videos)
    if max_videos != -1:
        videos = videos[:max_videos]

    samples = []
    for video_id, category, video_path, ann in videos:
        duration = float(ann.get("duration", 0.0))
        for (start, end), sentence in zip(ann["timestamps"], ann["sentences"]):
            start, end = float(start), float(end)
            if end <= start:
                continue
            if duration > 0:
                end = min(end, duration)
            if end <= start:
                continue
            samples.append({
                "video_id":   video_id,
                "category":   category,
                "video_path": video_path,
                "t_start":    start,
                "t_end":      end,
                "sentence":   sentence.strip(),
            })

    return samples


def build_dataset(json_path: str, max_videos: int, logger) -> Dataset:
    samples = _load_seg_samples(json_path, max_videos)
    counts  = Counter(s["category"] for s in samples)
    logger.info(
        f"Loaded {len(samples)} segments from {len(set(s['video_id'] for s in samples))} videos "
        f"({Path(json_path).name})"
    )
    logger.info(f"Category distribution: {dict(sorted(counts.items()))}")
    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn_seg(batch: list[dict], processor) -> dict:
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]

    instances = []
    for sample in batch:
        frames   = extract_frames(sample["video_path"], sample["t_start"], sample["t_end"], NUM_FRAMES)
        prompt   = _make_seg_prompt(sample["t_start"], sample["t_end"], len(frames))
        response = f"<{int(sample['t_start'])}><{int(sample['t_end'])}> [{sample['category']}] {sample['sentence']}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]

        text     = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        metadata = _make_metadata(sample["t_start"], sample["t_end"], len(frames))

        instance = processor(
            text=[text],
            videos=[[frames]],
            video_metadata=[metadata],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        instances.append(instance)

    input_ids = pad_sequence(
        [inst["input_ids"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id,
    )
    attention_mask = pad_sequence(
        [inst["attention_mask"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [inst["input_ids"].squeeze(0).clone() for inst in instances],
        batch_first=True,
        padding_value=-100,
    )

    # Mask image tokens and padding
    labels[labels == image_token_id]                   = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask prompt tokens up to and including "Assistant:"
    assistant_token = processor.tokenizer.encode("Assistant:", add_special_tokens=False)
    for i, ids in enumerate(input_ids):
        ids_list  = ids.tolist()
        split_pos = None
        for j in range(len(ids_list) - len(assistant_token), -1, -1):
            if ids_list[j : j + len(assistant_token)] == assistant_token:
                split_pos = j + len(assistant_token) + 1
                break
        if split_pos is not None:
            labels[i, :split_pos] = -100

    out = {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }

    # Pad pixel_values across batch
    pvs = [inst["pixel_values"].squeeze(0) for inst in instances if "pixel_values" in inst]
    if pvs:
        max_frames = max(pv.shape[0] for pv in pvs)
        max_h      = max(pv.shape[-2] for pv in pvs)
        max_w      = max(pv.shape[-1] for pv in pvs)

        padded_pvs = []
        for inst in instances:
            pv = inst.get("pixel_values")
            if pv is None:
                padded_pvs.append(torch.zeros((max_frames, 3, max_h, max_w), dtype=torch.float32))
            else:
                pv = pv.squeeze(0)
                f, c, h, w = pv.shape
                padded = torch.zeros((max_frames, c, max_h, max_w), dtype=pv.dtype, device=pv.device)
                padded[:f, :, :h, :w] = pv
                padded_pvs.append(padded)
        out["pixel_values"] = torch.stack(padded_pvs, dim=0)

    return out


# ---------------------------------------------------------------------------
# Balanced trainer — WeightedRandomSampler for category balance
# ---------------------------------------------------------------------------

class BalancedTrainer(Trainer):
    """Overrides train DataLoader to use category-balanced sampling."""

    def get_train_dataloader(self) -> DataLoader:
        ds         = self.train_dataset
        categories = [s["category"] for s in ds]
        counts     = Counter(categories)
        weights    = torch.tensor([1.0 / counts[c] for c in categories], dtype=torch.float)
        sampler    = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        return DataLoader(
            ds,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    assert torch.cuda.is_available(), (
        "CUDA not found. train_seg.py requires a CUDA GPU.\n"
        "Check: nvidia-smi"
    )

    run_name = f"smolvlm2-seg-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_file = os.path.join(OUTPUT_DIR, "logs", f"{run_name}.log")
    logger   = setup_logging(log_file)

    logger.info("=== SmolVLM2 Per-Segment Activity Description SFT ===")
    logger.info(f"Model     : {MODEL_ID}")
    logger.info(f"Frames    : {NUM_FRAMES}/segment | Max length: {MAX_LENGTH}")
    logger.info(f"Output    : {OUTPUT_DIR}")
    logger.info(f"GPU       : {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    logger.info("Building datasets...")
    train_ds = build_dataset(TRAIN_JSON, MAX_TRAIN, logger)
    val_ds   = build_dataset(VAL_JSON,   MAX_VAL,   logger)

    hparams = {
        "model_id":          MODEL_ID,
        "num_frames":        NUM_FRAMES,
        "train_segments":    len(train_ds),
        "val_segments":      len(val_ds),
        "num_epochs":        5,
        "learning_rate":     1e-4,
        "batch_size":        1,
        "grad_accum_steps":  8,
        "lr_scheduler":      "cosine",
        "warmup_steps":      50,
        "optimizer":         "adamw_hf",
        "precision":         "bf16",
        "sampling":          "category_balanced",
        "task":              "per_segment_temporal_category",
        "seed":              SEED,
        "device":            torch.cuda.get_device_name(0),
    }

    run = mlflow.start_run(run_name=run_name)
    try:
        logger.info(f"MLflow run: {run.info.run_id}  ({MLFLOW_URI})")
        try:
            mlflow.log_params(hparams)
        except Exception as e:
            logger.warning(f"MLflow log_params failed: {e}")

        logger.info("Loading model and processor...")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model     = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            _attn_implementations="flash_attention_2",
            device_map="auto",
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params    : {total_params / 1e6:.0f}M")
        logger.info(f"Trainable params: {trainable_params / 1e6:.0f}M  (full fine-tune)")
        peak_mem = torch.cuda.max_memory_allocated()
        logger.info(f"GPU RAM after load: {peak_mem / 1e9:.2f} GB")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=5,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            optim="adamw_hf",
            bf16=True,
            max_grad_norm=1.0,
            weight_decay=0.01,
            logging_steps=25,
            save_strategy="steps",
            save_steps=250,
            save_total_limit=1,
            eval_strategy="steps",
            eval_steps=250,
            eval_accumulation_steps=4,
            remove_unused_columns=False,
            dataloader_num_workers=2,
            dataloader_pin_memory=False,
            report_to="tensorboard",
            seed=SEED,
        )

        collator = functools.partial(collate_fn_seg, processor=processor)

        trainer = BalancedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            callbacks=[MLflowMetricsCallback()],
        )

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

        logger.info("Saving final model...")
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)

        try:
            mlflow.log_artifact(log_file, artifact_path="logs")
        except Exception as e:
            logger.warning(f"MLflow log_artifact failed: {e}")

        logger.info(f"Done. Checkpoint: {OUTPUT_DIR}")
        logger.info(
            f"MLflow URL: {MLFLOW_URI}#/experiments/"
            f"{run.info.experiment_id}/runs/{run.info.run_id}"
        )

    finally:
        try:
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"MLflow end_run failed: {e}")


if __name__ == "__main__":
    main()
