"""
LoRA fine-tune: HuggingFaceTB/SmolVLM2-2.2B-Instruct on UCF-Crime + UCA dataset.
All config hardcoded below. Tuned for RTX PRO 6000 Blackwell (96GB VRAM).

Usage:
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/train/smolvlm2_2b_lora.py
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Hardcoded config
# ---------------------------------------------------------------------------

MODEL_ID      = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DATA_ROOT     = os.environ.get("DATA_ROOT", str(Path(__file__).parent.parent / "data"))
OUTPUT_DIR    = str(Path(__file__).parent.parent / "output" / "smolvlm2-2b-lora")

MLFLOW_URI        = "https://mlflow-geoai.stelarea.com/"
MLFLOW_EXPERIMENT = "vlm-surveillance"

# 96GB VRAM — big batch, more frames, higher LoRA rank, no grad checkpointing
LORA_RANK     = 32   # 16→32: more capacity, VRAM is not a constraint
EPOCHS        = 3
LR            = 1e-4
VISION_LR     = 5e-5
BATCH         = 16   # 4→16: fills VRAM, reduces wall time
GRAD_ACCUM    = 2    # effective batch = 32
MAX_FRAMES    = 32   # 16→32: better temporal coverage per clip
MAX_NORMAL    = 1500
SAMPLER       = "sqrt"
CLASS_TOKEN_W = 5.0
KL_COEF       = 0.0  # >0 enables KL-to-base retention (try 0.5-1.0 to fight rare-class forgetting)
FRAME_JITTER  = 1.5
EVAL_STEPS    = 50
SAVE_STEPS    = 50
NO_GRAD_CKPT  = False # keep grad checkpointing ON — guards against OOM on batch=16 x 32-frame x rank-32; ~20-30% slower but safe

# Must be set before train_full is imported (module-level constants read at import time)
os.environ["MLFLOW_URI"]        = MLFLOW_URI
os.environ["MLFLOW_EXPERIMENT"] = MLFLOW_EXPERIMENT

# ---------------------------------------------------------------------------
# Inject config as argv so train_full.main() picks it up
# ---------------------------------------------------------------------------

sys.argv = [
    "train_full.py",
    "--model",              MODEL_ID,
    "--lora",
    "--lora-rank",          str(LORA_RANK),
    "--max-train",          "-1",
    "--max-val",            "-1",
    "--epochs",             str(EPOCHS),
    "--lr",                 str(LR),
    "--vision-lr",          str(VISION_LR),
    "--batch",              str(BATCH),
    "--grad-accum",         str(GRAD_ACCUM),
    "--max-frames",         str(MAX_FRAMES),
    "--max-normal",         str(MAX_NORMAL),
    "--sampler",            SAMPLER,
    "--class-token-weight", str(CLASS_TOKEN_W),
    "--kl-coef",            str(KL_COEF),
    "--frame-jitter",       str(FRAME_JITTER),
    "--save-steps",         str(SAVE_STEPS),
    "--data-root",          DATA_ROOT,
    "--output",             OUTPUT_DIR,
    *(["--no-grad-checkpoint"] if NO_GRAD_CKPT else []),
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from train_full import main

if __name__ == "__main__":
    main()
