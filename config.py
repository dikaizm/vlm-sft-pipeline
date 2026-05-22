"""
Central configuration — data paths and shared constants.

All scripts import from here instead of redefining paths locally.
Override DATA_ROOT via env var:
    DATA_ROOT=/path/to/data python vlm-sft-pipeline/train_desc.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

DATA_ROOT  = os.environ.get("DATA_ROOT", "./data")
VIDEO_ROOT = f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos"
TRAIN_JSON = f"{DATA_ROOT}/UCFCrime_Train.json"
VAL_JSON   = f"{DATA_ROOT}/UCFCrime_Val.json"
TEST_JSON  = f"{DATA_ROOT}/UCFCrime_Test.json"
