"""
Smoke test: SurveillanceTrainer gradient flow on 2 synthetic samples.

Verifies:
  1. token_weight boosts loss on [ClassName] bracket tokens
  2. Loss is finite and nonzero
  3. Gradients exist on model parameters
  4. Fallback to model.loss when no token_weight
  5. All-masked labels give finite loss (denom clamp guard)

Usage:
    python vlm-sft-pipeline/train/smoke_test_trainer.py
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Inline SurveillanceTrainer (mirrors train_full.py exactly)
# ---------------------------------------------------------------------------

class SurveillanceTrainer:
    """Standalone copy for testing — no HF Trainer dependency."""

    training = True

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        token_weight = inputs.pop("token_weight", None)
        outputs = model(**inputs)

        if token_weight is None or not self.training:
            loss = outputs.loss
        else:
            logits = outputs.logits
            labels = inputs["labels"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            del logits
            outputs.logits = None

            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            per_token = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_logits.size(0), -1)

            mask = (shift_labels != -100).float()

            shift_tw = token_weight[..., 1:].to(per_token.device)
            mask = mask * shift_tw

            denom = mask.sum(-1).clamp(min=1)
            per_sample = (per_token * mask).sum(-1) / denom

            loss = per_sample.mean()

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Tiny fake model for testing
# ---------------------------------------------------------------------------

class FakeLMOutput:
    def __init__(self, logits):
        self.logits = logits
        self.loss = None


class TinyLM(nn.Module):
    def __init__(self, vocab_size=100, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj  = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        h = self.embed(input_ids)
        logits = self.proj(h)
        return FakeLMOutput(logits)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def make_batch(B=2, T=16, V=100, class_token_weight=5.0):
    """Synthetic batch — token_weight=5.0 at positions 4-7, 1.0 elsewhere."""
    input_ids = torch.randint(0, V, (B, T))
    labels = input_ids.clone()
    labels[:, :4] = -100
    attention_mask = torch.ones(B, T, dtype=torch.long)
    token_weight = torch.ones(B, T)
    token_weight[:, 4:8] = class_token_weight

    return {
        "input_ids":     input_ids,
        "labels":        labels,
        "attention_mask": attention_mask,
        "token_weight":  token_weight,
    }


def run_tests():
    V = 100
    model = TinyLM(vocab_size=V)
    trainer = SurveillanceTrainer()

    print("=" * 60)
    print("SurveillanceTrainer smoke test")
    print("=" * 60)

    # T1: Loss is finite, nonzero, scalar
    batch = make_batch(B=2, T=16, V=V)
    loss = trainer.compute_loss(model, batch)

    assert loss.ndim == 0,       f"FAIL T1: loss not scalar, shape={loss.shape}"
    assert loss.item() > 0,      f"FAIL T1: loss={loss.item()} not positive"
    assert torch.isfinite(loss), f"FAIL T1: loss not finite"
    print(f"[PASS] T1 - loss finite scalar: {loss.item():.4f}")

    # T2: Gradients flow to model params
    model.zero_grad()
    batch = make_batch(B=2, T=16, V=V)
    loss = trainer.compute_loss(model, batch)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0,                              "FAIL T2: no gradients"
    assert all(torch.isfinite(g).all() for g in grads), "FAIL T2: NaN/Inf gradient"
    print(f"[PASS] T2 - gradients flow, {len(grads)} param tensors with grad")

    # T3: CLASS_TOKEN_WEIGHT amplifies loss vs uniform weights
    torch.manual_seed(42)
    input_ids = torch.randint(0, V, (2, 16))
    labels    = input_ids.clone(); labels[:, :4] = -100
    attn      = torch.ones(2, 16, dtype=torch.long)

    tw_uniform = torch.ones(2, 16)
    tw_boosted = torch.ones(2, 16); tw_boosted[:, 4:8] = 5.0

    b1 = {"input_ids": input_ids.clone(), "labels": labels.clone(),
          "attention_mask": attn.clone(), "token_weight": tw_uniform}
    b2 = {"input_ids": input_ids.clone(), "labels": labels.clone(),
          "attention_mask": attn.clone(), "token_weight": tw_boosted}

    loss1 = trainer.compute_loss(model, b1).item()
    loss2 = trainer.compute_loss(model, b2).item()

    assert loss2 != loss1, f"FAIL T3: boosted loss ({loss2:.4f}) == uniform ({loss1:.4f})"
    print(f"[PASS] T3 - token_weight changes loss: uniform={loss1:.4f}, boosted={loss2:.4f}")

    # T4: Fallback to model.loss when no token_weight
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
          "labels":    torch.randint(0, V, (2, 16))}
    loss3 = trainer.compute_loss(model2, b3)
    assert abs(loss3.item() - 1.234) < 1e-4, f"FAIL T4: fallback loss wrong: {loss3.item()}"
    print(f"[PASS] T4 - fallback to model.loss when no token_weight: {loss3.item():.4f}")

    # T5: All-masked labels give finite loss (denom clamp guard)
    model.zero_grad()
    b4 = {"input_ids":      torch.randint(0, V, (2, 16)),
          "labels":          torch.full((2, 16), -100, dtype=torch.long),
          "attention_mask":  torch.ones(2, 16, dtype=torch.long),
          "token_weight":    torch.ones(2, 16) * 5.0}
    loss4 = trainer.compute_loss(model, b4)
    assert torch.isfinite(loss4), f"FAIL T5: loss not finite for all-masked labels"
    print(f"[PASS] T5 - all-masked labels safe (clamp guard): loss={loss4.item():.4f}")

    print()
    print("All tests passed.")
    return True


if __name__ == "__main__":
    run_tests()
