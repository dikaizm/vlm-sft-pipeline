# VLM SFT Pipeline — 2.2B LoRA

Branch: `exp/2b-lora`

LoRA fine-tuning of SmolVLM2-2.2B-Video-Instruct on UCF-Crime + UCA dataset for simultaneous **video captioning** and **anomaly classification**. Inherits class-first weighted loss + sqrt-balanced sampling from upstream 500M branch.

**Rationale for 2.2B + LoRA:**
- 500M proved insufficient on 14-class minority discrimination (mode collapsed with full balancing, marginal gains with sqrt). Per-class sample counts (~80 each for crime classes) require more representational capacity than 500M provides.
- 2.2B has ~4× the parameters → better rare-class learning from limited data.
- LoRA keeps trainable params at ~0.5% (~10M) so optimizer + gradient memory is tiny. Main VRAM cost is activations (similar per-sample as 500M, but more layers).
- B200 192 GB fits batch=16 comfortably with LoRA.

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

### Class-Aware Sampling

UCF-Crime is heavily imbalanced: ~90% Normal, ~10% crime spread across 13 crime classes (~80 samples each). Scalar loss weighting (`crime-weight=3`) leaves minority crime tokens with ~38× less exposure than `[Normal]`.

`SurveillanceTrainer._get_train_sampler` supports three modes via `--sampler`:

| Mode | Weight | Normal prob / batch | Each crime class | Notes |
|---|---|---|---|---|
| `raw` | uniform | ~90% | ~0.75% | Natural distribution |
| `sqrt` (default) | `1/sqrt(count)` | ~46% | ~4% | Boosts minority without collapsing |
| `balanced` | `1/count` | ~7% | ~7% | Uniform across 14 classes — caused mode collapse to literal `[ClassName]` on 500M |

**Why `sqrt` (not full balance)**: Full balancing (`1/count`) showed all 14 class tokens per batch with equal frequency. On a 500M model, gradient pulled toward 14 different class-token destinations simultaneously, causing collapse — model defaulted to emitting the literal `[ClassName]` placeholder from the prompt template (Unknown rate 0.95 at step 400 vs 0.65 at step 300 with raw distribution).

Sqrt scaling keeps Normal as the majority anchor (~46%) while giving each crime class ~5× the exposure of raw distribution. The model retains a clear default but receives enough minority-class signal to differentiate.

`CRIME_WEIGHT` is preserved as a CLI option for additional emphasis on top of the sampler, but defaults to `1.0` (disabled).

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
| Model | SmolVLM2-2.2B-Video-Instruct | LoRA on LLM (q/k/v/o/gate/up/down proj), vision encoder frozen |
| LoRA rank | 16 | DoRA enabled, alpha = 2× rank |
| Trainable params | ~10M / 2.2B (~0.5%) | LoRA adapters only |
| Epochs | 3 | Full UCF-Crime train split |
| LR | 1e-4 | LoRA typically uses 5–10× higher LR than full FT |
| Batch | 16 | Effective 32 with grad_accum=2 |
| Grad accum | 2 | |
| Frames/sec | 4 | Up to 48 frames per sub-clip |
| Segment | 12s | 75% overlap (9s stride) |
| Max length | 4096 | 3072 visual + ~1024 text |
| CRIME_WEIGHT | 1.0 | Disabled by default — class-balanced sampler handles imbalance |
| Sampler | sqrt | WeightedRandomSampler, 1/sqrt(class_count) — minority boost without collapse |
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

### Training (2.2B LoRA)

```bash
DATA_ROOT=/path/to/data \
FRAME_CACHE_DIR=/path/to/cache \
python train/train_full.py \
  --model HuggingFaceTB/SmolVLM2-2.2B-Video-Instruct \
  --lora \
  --lora-rank 16 \
  --batch 16 \
  --grad-accum 2 \
  --lr 1e-4 \
  --epochs 3
```

LoRA on LLM layers only (vision encoder stays frozen). Sampler defaults to `sqrt`. If OOM at start → drop to `--batch 8 --grad-accum 4`.

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
