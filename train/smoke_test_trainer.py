"""
Smoke test: SurveillanceTrainer gradient flow on 2 synthetic samples.

Verifies:
  1. sample_weight tensor passes through correctly
  2. Loss is finite and nonzero
  3. Gradients exist on model parameters
  4. Crime-weighted loss > Normal-weighted loss for identical logits
  5. No shape errors in causal LM shift

Usage:
    python vlm-sft-pipeline/train/smoke_test_trainer.py
"""

import sys
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from pathlib import Path

# ---------------------------------------------------------------------------
# Inline SurveillanceTrainer (mirrors train_full.py exactly)
# ---------------------------------------------------------------------------

class SurveillanceTrainer:
    """Standalone copy for testing — no HF Trainer dependency."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("sample_weight", None)
        outputs = model(**inputs)

        if weights is None:
            loss = outputs.loss
        else:
            logits = outputs.logits
            labels = inputs["labels"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            per_token = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_logits.size(0), -1)

            mask = (shift_labels != -100).float()
            denom = mask.sum(-1).clamp(min=1)
            per_sample = (per_token * mask).sum(-1) / denom

            loss = (per_sample * weights.to(per_sample.device)).mean()

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Tiny fake model for testing
# ---------------------------------------------------------------------------

class FakeLMOutput:
    def __init__(self, logits):
        self.logits = logits
        self.loss = None


class TinyLM(nn.Module):
    """2-layer linear model. Accepts input_ids shape [B,T] and returns logits [B,T,V]."""

    def __init__(self, vocab_size=100, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj  = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        h = self.embed(input_ids)          # [B, T, hidden]
        logits = self.proj(h)              # [B, T, V]
        return FakeLMOutput(logits)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_batch(B=2, T=16, V=100, crime_weight=3.0):
    """Synthetic batch with 1 Normal + 1 Crime sample."""
    input_ids = torch.randint(0, V, (B, T))
    labels    = input_ids.clone()
    labels[:, :4] = -100           # mask first 4 tokens (system prompt)
    attention_mask = torch.ones(B, T, dtype=torch.long)
    weights = torch.tensor([1.0, crime_weight])   # [Normal, Crime]

    return {
        "input_ids":      input_ids,
        "labels":         labels,
        "attention_mask": attention_mask,
        "sample_weight":  weights,
    }


def run_tests():
    V = 100
    model = TinyLM(vocab_size=V)
    trainer = SurveillanceTrainer()

    print("=" * 60)
    print("SurveillanceTrainer smoke test")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Test 1: Loss is finite, nonzero, scalar
    # -----------------------------------------------------------------------
    batch = make_batch(B=2, T=16, V=V)
    loss = trainer.compute_loss(model, batch)

    assert loss.ndim == 0,          f"FAIL T1: loss not scalar, shape={loss.shape}"
    assert loss.item() > 0,         f"FAIL T1: loss={loss.item()} not positive"
    assert torch.isfinite(loss),    f"FAIL T1: loss not finite"
    print(f"[PASS] T1 - loss finite scalar: {loss.item():.4f}")

    # -----------------------------------------------------------------------
    # Test 2: Gradients flow to model params
    # -----------------------------------------------------------------------
    model.zero_grad()
    batch = make_batch(B=2, T=16, V=V)
    loss = trainer.compute_loss(model, batch)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0,                    "FAIL T2: no gradients"
    assert all(torch.isfinite(g).all() for g in grads), "FAIL T2: NaN/Inf gradient"
    print(f"[PASS] T2 - gradients flow, {len(grads)} param tensors with grad")

    # -----------------------------------------------------------------------
    # Test 3: Crime weight amplifies loss
    # -----------------------------------------------------------------------
    # Uniform weights → baseline loss
    batch1 = make_batch(B=2, T=16, V=V, crime_weight=1.0)
    batch2 = make_batch(B=2, T=16, V=V, crime_weight=3.0)

    # Same model + same data, only weights differ
    torch.manual_seed(42)
    input_ids = torch.randint(0, V, (2, 16))
    labels    = input_ids.clone(); labels[:, :4] = -100
    attn      = torch.ones(2, 16, dtype=torch.long)

    b1 = {"input_ids": input_ids.clone(), "labels": labels.clone(),
          "attention_mask": attn.clone(), "sample_weight": torch.tensor([1.0, 1.0])}
    b2 = {"input_ids": input_ids.clone(), "labels": labels.clone(),
          "attention_mask": attn.clone(), "sample_weight": torch.tensor([1.0, 3.0])}

    loss1 = trainer.compute_loss(model, b1).item()
    loss2 = trainer.compute_loss(model, b2).item()

    assert loss2 > loss1, f"FAIL T3: crime-weighted loss ({loss2:.4f}) not > uniform ({loss1:.4f})"
    print(f"[PASS] T3 - crime weight amplifies loss: uniform={loss1:.4f}, crime3x={loss2:.4f}")

    # -----------------------------------------------------------------------
    # Test 4: Fallback to model.loss when no sample_weight
    # -----------------------------------------------------------------------
    class FakeLMOutputWithLoss:
        def __init__(self, logits):
            self.logits = logits
            self.loss = torch.tensor(1.234)

    class TinyLMWithLoss(nn.Module):
        def forward(self, input_ids, **kwargs):
            B, T = input_ids.shape
            return FakeLMOutputWithLoss(torch.randn(B, T, V))

    model2 = TinyLMWithLoss()
    b3 = {"input_ids": torch.randint(0, V, (2, 16)),
          "labels":     torch.randint(0, V, (2, 16))}
    loss3 = trainer.compute_loss(model2, b3)
    assert abs(loss3.item() - 1.234) < 1e-4, f"FAIL T4: fallback loss wrong: {loss3.item()}"
    print(f"[PASS] T4 - fallback to model.loss when no sample_weight: {loss3.item():.4f}")

    # -----------------------------------------------------------------------
    # Test 5: All-masked labels give finite loss (denom clamp=1 guard)
    # -----------------------------------------------------------------------
    model.zero_grad()
    b4 = {"input_ids": torch.randint(0, V, (2, 16)),
          "labels": torch.full((2, 16), -100, dtype=torch.long),
          "attention_mask": torch.ones(2, 16, dtype=torch.long),
          "sample_weight": torch.tensor([1.0, 3.0])}
    loss4 = trainer.compute_loss(model, b4)
    assert torch.isfinite(loss4), f"FAIL T5: loss not finite for all-masked labels"
    print(f"[PASS] T5 - all-masked labels safe (clamp guard): loss={loss4.item():.4f}")

    print()
    print("All tests passed.")
    return True


if __name__ == "__main__":
    run_tests()
