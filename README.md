# VLM SFT Pipeline — Class-First Weighted Loss

Branch: `class-first-weighted-loss`

Full supervised fine-tuning of SmolVLM2-500M on UCF-Crime + UCA dataset for simultaneous **video captioning** and **anomaly classification** from surveillance clips.

---

## Methodological Approach

### Two-Axis Evaluation

This branch frames the task as producing two outputs simultaneously from a single inference pass:

1. **Captioning** — describe the activity in the surveillance clip (BLEU-4, ROUGE-L, BERTScore)
2. **Classification** — identify the activity class from 14 UCF-Crime categories (F1-macro, binary F1, precision/recall)

The model generates a single text sequence; classification is extracted by parsing the first `[ClassName]` token.

### Class-First Output Format

The ground-truth response is formatted as:

```
[ClassName] <activity description sentence>
```

Example: `[Shoplifting] A man in a yellow shirt conceals merchandise under his jacket.`

**Why class-first (not description-first):**
- Placing the class token at the start of generation gives the strongest gradient signal — autoregressive loss on early tokens has zero contamination from the description that follows
- Description-first causes the class token gradient to compete with ~50 description tokens it has already attended to, diluting signal
- Class-first also prevents truncation loss: if generation hits `max_new_tokens`, the class is already emitted

### Weighted Token Loss

Standard CE loss treats all output tokens equally. For a typical GT of `[Normal] A woman walks through the store`, the class bracket is ~4 tokens out of ~50 total — only 8% of the loss. The model is weakly incentivized to get the class right.

This branch applies a **per-token weight multiplier** to `[ClassName]` bracket tokens:

```
CLASS_TOKEN_WEIGHT = 5.0
```

Implementation in `SurveillanceTrainer.compute_loss`:

```python
# token_weight tensor: 5.0 at [ClassName] positions, 1.0 elsewhere
shift_tw = token_weight[..., 1:].to(per_token.device)
mask = mask * shift_tw
denom = mask.sum(-1).clamp(min=1)
per_sample = (per_token * mask).sum(-1) / denom
```

This raises the effective class-token contribution from ~8% to ~30% of the total loss without discarding description supervision.

### Sample-Level Crime Weighting

UCF-Crime is heavily imbalanced: ~90% Normal, ~10% crime. Without correction, the model learns to predict Normal for everything.

```
CRIME_WEIGHT = 3.0   # loss multiplier on all non-Normal samples
```

Applied as a per-sample scalar on top of the token-level loss.

### Constrained Decoding (Optional)

At inference, `--constrained` activates `ClassFirstLogitsProcessor`, which forces the first generated tokens to match a valid `[ClassName]` sequence. This eliminates the Unknown rate entirely but requires a class-first-trained checkpoint.

```python
class ClassFirstLogitsProcessor:
    """Force first tokens to a valid [ClassName] by masking invalid vocab positions."""
```

Use only with checkpoints trained on this branch. Applying constrained decoding to a model trained without class-first format will produce degenerate output.

---

## Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Model | SmolVLM2-500M-Video-Instruct | Full fine-tune, vision encoder unfrozen |
| Epochs | 3 | Full UCF-Crime train split |
| LR | 2e-5 | Cosine schedule, 5% warmup |
| Batch | 8 | Effective 32 with grad_accum=4 |
| Grad accum | 4 | |
| Frames/sec | 4 | Up to 48 frames per sub-clip |
| Segment | 12s | 75% overlap (9s stride) |
| Max length | 4096 | 3072 visual + ~1024 text |
| CRIME_WEIGHT | 3.0 | Non-Normal sample loss multiplier |
| CLASS_TOKEN_WEIGHT | 5.0 | `[ClassName]` bracket token multiplier |
| Optimizer | adamw_bnb_8bit | 8-bit Adam on CUDA |
| Precision | bf16 | Flash Attention 2 |
| Gradient checkpointing | ON | Required for full fine-tune at seq len 4096 |

---

## Key Files

| File | Purpose |
|---|---|
| `train/train_full.py` | Full SFT pipeline — data loading, collator, `SurveillanceTrainer`, training loop |
| `infer/infer.py` | Batch inference — zero-shot vs fine-tuned comparison, captioning + classification metrics |
| `infer/infer_realtime.py` | Real-time inference on live video stream |
| `eval/eval.py` | Standalone evaluation against saved predictions |
| `config.py` | Centralized data path configuration |

---

## Usage

### Training

```bash
DATA_ROOT=/path/to/data \
FRAME_CACHE_DIR=/path/to/cache \
python train/train_full.py \
  --model HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --epochs 3 \
  --crime-weight 3.0 \
  --class-token-weight 5.0
```

### Inference

```bash
DATA_ROOT=/path/to/data \
python infer/infer.py \
  --finetuned /path/to/checkpoint \
  --n 50 \
  --rep-penalty 1.3

# With constrained decoding (class-first checkpoints only):
python infer/infer.py \
  --finetuned /path/to/checkpoint \
  --constrained \
  --rep-penalty 1.3
```

---

## Metrics

**Captioning** (against GT sentence):
- BLEU-4
- ROUGE-L
- BERTScore F1 (roberta-large)

**Classification** (parsed from first `[ClassName]` in output):
- 14-class: accuracy, F1-macro, F1-weighted, precision/recall macro
- Binary (Normal vs Anomaly): accuracy, F1, precision, recall
- Unknown rate: fraction of outputs with no valid `[ClassName]`

Unknown predictions are mapped to Normal (conservative: did not flag anomaly).

---

## Expected Training Trajectory

| Epoch | Expected behavior |
|---|---|
| 0–1 | Caption style learned, description quality improves. `[ClassName]` may not yet appear. |
| 1–2 | `[ClassName]` begins appearing in outputs. Unknown rate drops. |
| 2–3 | Class format stable. Crime-specific vocabulary emerges. Classification metrics non-zero. |
