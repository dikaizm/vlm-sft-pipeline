"""
WiSE-FT adapter-scale sweep: interpolate base and LoRA-merged weights to
recover pretrained (zero-shot) knowledge lost to fine-tuning.

    theta(lambda) = (1 - lambda) * theta_base + lambda * theta_finetuned

  lambda=0.0 -> pure zero-shot   (knows rare classes, poor format)
  lambda=1.0 -> full fine-tune   (great format, forgot rare classes)
  0<lambda<1 -> robust blend

True weight-space interpolation (not adapter scaling) so it is correct for
DoRA adapters as well. Runs the SAME test clips at each lambda and reports
classification / caption metrics + a rare-class hit breakdown.

Usage:
    DATA_ROOT=/path/to/data \
    MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
    FINETUNED_DIR=/Volumes/T7/research-vlm/output/qwen3-vl-2b-lora \
    python vlm-sft-pipeline/infer/sweep_lora_scale.py --n 25 \
        --lambdas 0.0 0.3 0.5 0.7 0.85 1.0
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# Reuse the exact inference + metric helpers used by infer.py for consistency
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import infer as I  # noqa: E402


def _snapshot_cpu(model) -> dict:
    return {k: v.detach().to("cpu", torch.float32).clone()
            for k, v in model.state_dict().items()}


def _rare_breakdown(preds, gts, classes=("Explosion", "Arson", "Shooting")):
    """Per-class recall for the train-starved classes that regressed."""
    out = {}
    for c in classes:
        idx = [i for i, g in enumerate(gts) if g == c]
        if not idx:
            continue
        hit = sum(1 for i in idx if preds[i] == c)
        out[c] = f"{hit}/{len(idx)}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--crime-ratio", type=float, default=0.8)
    ap.add_argument("--context-pad", type=float, default=5.0)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.3, 0.5, 0.7, 0.85, 1.0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/lora_scale_sweep.json")
    args = ap.parse_args()

    base_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
    ft_dir  = os.environ["FINETUNED_DIR"]
    device  = I.get_device()
    dtype   = torch.float32 if device.type == "mps" else torch.bfloat16
    print(f"Device: {device}  base: {base_id}  adapter: {ft_dir}")

    # --- Working model (we mutate its weights per lambda) + theta_base ---
    print("Loading base model (theta_base) ...")
    model = AutoModelForImageTextToText.from_pretrained(
        base_id, dtype=dtype, trust_remote_code=True).to(device)
    model.eval()
    base_sd = _snapshot_cpu(model)

    # --- theta_finetuned = base + merged LoRA delta ---
    print("Loading + merging adapter (theta_finetuned) ...")
    tmp = AutoModelForImageTextToText.from_pretrained(
        base_id, dtype=dtype, trust_remote_code=True)
    tmp = PeftModel.from_pretrained(tmp, ft_dir).merge_and_unload()
    ft_sd = _snapshot_cpu(tmp)
    del tmp
    if device.type == "cuda":
        torch.cuda.empty_cache()

    keys = [k for k in base_sd if k in ft_sd and base_sd[k].shape == ft_sd[k].shape]
    print(f"  interpolating {len(keys)} weight tensors")

    processor = AutoProcessor.from_pretrained(ft_dir, trust_remote_code=True)

    # --- Fixed clip set (same across all lambda) ---
    random.seed(args.seed)
    samples = I.load_test_samples(args.n, args.crime_ratio)
    print(f"Loaded {len(samples)} clips\n")

    # Pre-extract frames once (identical inputs across lambda)
    frame_cache = []
    for s in samples:
        ps = max(0.0, s["start"] - args.context_pad)
        pe = s["end"] + args.context_pad
        frame_cache.append((I.extract_frames(s["video_path"], ps, pe, I.NUM_FRAMES), ps, pe))

    results = {}
    for lam in args.lambdas:
        # theta(lambda) in-place into the working model
        new_sd = dict(model.state_dict())
        for k in keys:
            new_sd[k] = ((1.0 - lam) * base_sd[k] + lam * ft_sd[k]).to(dtype)
        model.load_state_dict(new_sd, strict=False)

        preds, gts, pred_txt, gt_txt = [], [], [], []
        for s, (frames, ps, pe) in zip(samples, frame_cache):
            out = I.run_inference(model, processor, device, frames, ps, pe, I.PROMPT)
            preds.append(I.parse_class(out)); gts.append(s["gt_class"])
            pred_txt.append(out);             gt_txt.append(s["gt"])

        cls = I._compute_cls_metrics(preds, gts)
        cap = I._compute_caption_metrics(pred_txt, gt_txt)
        rare = _rare_breakdown(preds, gts)
        results[f"{lam:.2f}"] = {"cls": cls, "cap": cap, "rare_recall": rare}

        print(f"lambda={lam:.2f}  "
              f"binF1={cls.get('binary_f1','-')}  "
              f"acc14={cls.get('cls_accuracy','-')}  "
              f"f1mac={cls.get('f1_macro','-')}  "
              f"unk={cls.get('unknown_rate','-')}  "
              f"R-L={cap.get('rougeL','-')}  rare={rare}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"base": base_id, "adapter": ft_dir, "n": len(samples),
                   "lambdas": results}, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
