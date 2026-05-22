"""
Activity Description SFT — SmolVLM2 on UCF-Crime / UCA dataset.

Model learns to describe surveillance video activities with approximate timestamps
and crime category labels (Option A: video-level label applied to all annotations).

Output format:
    <0><25> [Assault] Two men engaged in a physical altercation near the entrance.
    <25><60> [Assault] Bystanders attempt to intervene and separate the fighters.

Category is derived from the video filename prefix (e.g. Assault031 → Assault).
All annotations in a video share the same video-level category label.

Works with both SmolVLM2-500M and SmolVLM2-2.2B (set MODEL_ID env var).

Usage:
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/train_desc.py
    MODEL_ID=HuggingFaceTB/SmolVLM2-500M-Video-Instruct DATA_ROOT=/path/to/data python vlm-sft-pipeline/train_desc.py
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
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

from config import DATA_ROOT, VIDEO_ROOT, TRAIN_JSON, VAL_JSON

OUTPUT_DIR = os.environ.get("OUTPUT_DIR",   "./output/smolvlm2-desc-sft")
MODEL_ID   = os.environ.get("MODEL_ID",     "HuggingFaceTB/SmolVLM2-2.2B-Video-Instruct")

MLFLOW_URI        = os.environ.get("MLFLOW_URI",        "https://mlflow-geoai.stelarea.com/")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "smolvlm2-surveillance-desc")

SEED            = 42
NUM_FRAMES      = 32
MAX_LENGTH      = 4096
MAX_TRAIN       = -1
MAX_VAL         = 379
MAX_DURATION    = 90.0
MAX_ANNOTATIONS = 12

DESC_PROMPT = (
    "List each activity observed in this surveillance video. "
    "Format each line as: '<t_start><t_end> [Category] description' "
    "using approximate timestamps in seconds. "
    "If none, write 'None detected.'"
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("train_desc")
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
        logger = logging.getLogger("train_desc")
        logger.info(
            "  ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in sorted(metrics.items()))
        )
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"MLflow log_metrics failed (step {step}): {e}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _category_from_id(video_id: str) -> str:
    return re.sub(r"\d+_x264$", "", video_id)


def _load_video_samples(json_path: str, max_videos: int) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)

    mp4_map = {}
    for root_dir, _, files in os.walk(VIDEO_ROOT):
        for fname in files:
            if fname.endswith(".mp4"):
                mp4_map[fname] = os.path.join(root_dir, fname)

    samples = []
    for video_id, ann in data.items():
        category   = _category_from_id(video_id)
        video_path = os.path.join(VIDEO_ROOT, category, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            fallback = mp4_map.get(f"{video_id}.mp4")
            if fallback:
                video_path = fallback
            else:
                continue

        duration      = float(ann.get("duration", MAX_DURATION))
        effective_end = min(duration, MAX_DURATION)

        pairs = []
        for (start, end), sentence in zip(ann["timestamps"], ann["sentences"]):
            start, end = float(start), float(end)
            if end <= start or start > effective_end:
                continue
            pairs.append((start, min(end, effective_end), sentence.strip()))

        if not pairs:
            continue

        pairs.sort(key=lambda x: x[0])
        pairs = pairs[:MAX_ANNOTATIONS]

        samples.append({
            "video_id":      video_id,
            "category":      category,
            "video_path":    video_path,
            "annotations": [
                {"t_start": int(t0), "t_end": int(t1), "sentence": s}
                for t0, t1, s in pairs
            ],
        })

    random.seed(SEED)
    random.shuffle(samples)
    return samples if max_videos == -1 else samples[:max_videos]


def build_dataset(json_path: str, max_videos: int, logger) -> Dataset:
    samples = _load_video_samples(json_path, max_videos)
    total_ann = sum(len(s["annotations"]) for s in samples)
    logger.info(
        f"Loaded {len(samples)} videos ({total_ann} annotations) "
        f"from {Path(json_path).name}"
    )
    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn_desc(batch: list[dict], processor) -> dict:
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]

    instances = []
    for sample in batch:
        response = "\n".join(
            f"<{ann['t_start']}><{ann['t_end']}> [{sample['category']}] {ann['sentence']}"
            for ann in sample["annotations"]
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": sample["video_path"]},
                    {"type": "text", "text": DESC_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]

        instance = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
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

    # Mask image tokens and padding in labels
    labels[labels == image_token_id] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask prompt tokens (everything up to and including "Assistant:")
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

    # Pad pixel_values across batch (variable frames/resolution)
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
# Main
# ---------------------------------------------------------------------------

def main():
    assert torch.cuda.is_available(), (
        "CUDA not found. train_desc.py requires a CUDA GPU.\n"
        "Check: nvidia-smi"
    )

    run_name = f"smolvlm2-desc-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_file = os.path.join(OUTPUT_DIR, "logs", f"{run_name}.log")
    logger   = setup_logging(log_file)

    logger.info("=== SmolVLM2 Activity Description SFT (timestamp + category) ===")
    logger.info(f"Model         : {MODEL_ID}")
    logger.info(f"Train         : {MAX_TRAIN} videos | Val: {MAX_VAL} videos")
    logger.info(f"Max duration  : {MAX_DURATION}s | Max annotations: {MAX_ANNOTATIONS}")
    logger.info(f"Frames        : {NUM_FRAMES} | Max length: {MAX_LENGTH}")
    logger.info(f"Output        : {OUTPUT_DIR}")
    logger.info(f"GPU           : {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    hparams = {
        "model_id":                    MODEL_ID,
        "num_frames":                  NUM_FRAMES,
        "max_train_videos":            train_count if MAX_TRAIN == -1 else MAX_TRAIN,
        "max_duration_seconds":        MAX_DURATION,
        "max_annotations_per_video":   MAX_ANNOTATIONS,
        "num_epochs":                  5,
        "learning_rate":               2e-5,
        "batch_size":                  4,
        "gradient_accumulation_steps": 8,
        "lr_scheduler":                "cosine",
        "warmup_steps":                30,
        "optimizer":                   "adamw_bnb_8bit",
        "precision":                   "bf16",
        "device":                      torch.cuda.get_device_name(0),
        "task":                        "activity_description_timestamp_category_optionA",
        "seed":                        SEED,
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
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            _attn_implementations="flash_attention_2",
            device_map="auto",
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
        peak_mem = torch.cuda.max_memory_allocated()
        print(f"The model as is is holding: {peak_mem / 1024**3:.2f} of GPU RAM")

        logger.info("Building datasets...")
        train_ds = build_dataset(TRAIN_JSON, MAX_TRAIN, logger)
        val_ds   = build_dataset(VAL_JSON,   MAX_VAL,   logger)

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
            logging_steps=25,
            save_steps=250,
            save_total_limit=1,
            eval_strategy="steps",
            eval_accumulation_steps=4,
            eval_steps=250,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            dataloader_pin_memory=False,
            report_to="tensorboard",
        )

        collator = functools.partial(collate_fn_desc, processor=processor)

        trainer = Trainer(
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
        logger.info(f"MLflow run URL: {MLFLOW_URI}#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

    finally:
        try:
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"MLflow end_run failed: {e}")


if __name__ == "__main__":
    main()
