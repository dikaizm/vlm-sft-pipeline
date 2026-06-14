"""
LoRA fine-tune: Qwen/Qwen3-VL-2B-Instruct on UCF-Crime + UCA dataset.
All config hardcoded below. Tuned for RTX PRO 6000 Blackwell (96GB VRAM).

Usage:
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/train/qwen3_vl_2b_lora.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Hardcoded config
# ---------------------------------------------------------------------------

MODEL_ID      = "Qwen/Qwen3-VL-2B-Instruct"
DATA_ROOT     = os.environ.get("DATA_ROOT", str(Path(__file__).parent.parent / "data"))
RUN_TAG       = datetime.now().strftime("%Y%m%d-%H%M%S")
_EXTRA = sys.argv[1:]  # passthrough CLI args (e.g. --resume <ckpt>)
OUTPUT_DIR    = str(Path(__file__).parent.parent / "output" / f"qwen3-vl-2b-lora-{RUN_TAG}")

# Resume: if --resume <ckpt> passed, continue in that checkpoint's run folder
if "--resume" in _EXTRA:
    _ri = _EXTRA.index("--resume")
    if _ri + 1 < len(_EXTRA):
        OUTPUT_DIR = str(Path(_EXTRA[_ri + 1]).resolve().parent)

MLFLOW_URI        = "https://mlflow-geoai.stelarea.com/"
MLFLOW_EXPERIMENT = "vlm-surveillance"

# 96GB VRAM — big batch, more frames, higher LoRA rank, no grad checkpointing
LORA_RANK     = 16   # tiny dataset -> small adapter regularizes + retains base (alpha=2r=32)
EPOCHS        = 2
LR            = 9e-5   # sqrt-scaled for eff batch 24 (was 1e-4 @ eff 32)
VISION_LR     = 4.5e-5
BATCH         = 12   # H200 141GB
GRAD_ACCUM    = 2    # eff batch = 24
MAX_FRAMES    = 32
MAX_NORMAL    = 1500  # caps pure-normal only (transitional all kept); Normal ~28% eff w/ sqrt
SAMPLER       = "sqrt"
CLASS_TOKEN_W = 5.0
KL_COEF       = 0.5
FRAME_JITTER  = 1.5
EVAL_STEPS    = 50
SAVE_STEPS    = 50
NO_GRAD_CKPT  = False

# Must be set before train_full is imported (module-level constants read at import time)
os.environ["MLFLOW_URI"]        = MLFLOW_URI
os.environ["MLFLOW_EXPERIMENT"] = MLFLOW_EXPERIMENT
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Inject config as argv so train_full.main() picks it up
# ---------------------------------------------------------------------------

sys.argv = [
    "train_full.py",
    "--model",              MODEL_ID,
    "--trust-remote-code",
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
    "--fold-val",
    "--sampler",            SAMPLER,
    "--class-token-weight", str(CLASS_TOKEN_W),
    "--kl-coef",            str(KL_COEF),
    "--frame-jitter",       str(FRAME_JITTER),
    "--save-steps",         str(SAVE_STEPS),
    "--data-root",          DATA_ROOT,
    "--output",             OUTPUT_DIR,
    "--frame-cache",        str(Path(__file__).parent.parent / "frame_cache_24"),
    *(["--no-grad-checkpoint"] if NO_GRAD_CKPT else []),
]
sys.argv += _EXTRA  # forward passthrough CLI args (--resume, etc.)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from train_full import main

if __name__ == "__main__":
    main()
