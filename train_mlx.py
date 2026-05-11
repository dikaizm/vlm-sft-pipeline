"""
MLX SFT for SmolVLM-256M — Surveillance Dense Video Captioning.
Runs full fine-tuning (or LoRA) on Apple Silicon.
"""

import argparse, os, sys
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

from dotenv import load_dotenv; load_dotenv()

MODEL_PATH  = "mlx-community/SmolVLM-256M-Instruct-bf16"
MODEL_4BIT  = "mlx-community/SmolVLM-256M-Instruct-4bit"
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", "./output/smolvlm-256m-mlx-sft")
DATASET     = os.environ.get("DATASET_PATH", "./output/surveillance-dense-dataset-test")


class ModelWrapper:
    """Fixes attention_mask → cache arg mismatch in sft_trainer."""
    def __init__(self, base): self._model = base
    def __getattr__(self, name): return getattr(self._model, name)
    def __call__(self, input_ids, pixel_values, attention_mask=None, **kw):
        return self._model(input_ids, pixel_values, cache=None, **kw)
    def __getitem__(self, key): return self._model[key]
    def __setitem__(self, key, val): self._model[key] = val
    def __contains__(self, key): return key in self._model
    def items(self): return self._model.items()
    def keys(self): return self._model.keys()
    def values(self): return self._model.values()
    def update(self, d): return self._model.update(d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default=MODEL_PATH)
    parser.add_argument("--dataset",   default=DATASET)
    parser.add_argument("--output",    default=None)
    parser.add_argument("--epochs",    type=int, default=5)
    parser.add_argument("--batch-size",type=int, default=4)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--max-seq-len",type=int, default=3072)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha",type=int, default=16)
    parser.add_argument("--lora",      action="store_true")
    parser.add_argument("--qlora",     action="store_true")
    parser.add_argument("--image-resize-shape", type=int, nargs=2, default=None)
    args = parser.parse_args()

    # --- Config ---
    if args.qlora:
        model_path, lr, full = MODEL_4BIT, args.lr, False
        print("QLoRA mode (4-bit + LoRA)")
    elif args.lora:
        model_path, lr, full = args.model, args.lr, False
        print("LoRA mode")
    else:
        model_path, lr, full = args.model, args.lr / 2, True
        print("Full fine-tune mode")

    output_path = args.output or os.path.join(OUTPUT_DIR, "adapters.safetensors")
    os.makedirs(os.path.dirname(output_path) or OUTPUT_DIR, exist_ok=True)

    print(f"Model: {model_path}\nDataset: {args.dataset}\nEpochs: {args.epochs}")
    print(f"Batch: {args.batch_size}  LR: {lr}  Seq len: {args.max_seq_len}\n")

    # --- Load model ---
    print("Loading model...")
    from mlx_vlm import load as mlx_load
    model, processor = mlx_load(model_path, processor_config={"trust_remote_code": True})
    config = model.config.__dict__

    # Disable Idefics3 sub-image splitting
    processor.image_processor.size = {"longest_edge": 512}
    processor.image_processor.max_image_size = {"longest_edge": 512}
    print(f"  Processor patched: size={processor.image_processor.size}")

    # Fix attention_mask → cache arg mismatch
    model = ModelWrapper(model)

    # --- Load dataset ---
    print(f"\nLoading dataset: {args.dataset}")
    from datasets import load_dataset as hf_load
    dataset = hf_load(args.dataset, split="train")
    print(f"  {len(dataset)} samples, columns: {dataset.column_names}")
    s = dataset[0]
    print(f"  Sample: {len(s['images'])} images @ {s['images'][0].size}")

    # --- VisionDataset ---
    from mlx_vlm.trainer.datasets import VisionDataset
    train_dataset = VisionDataset(dataset, config, processor,
                                   image_resize_shape=args.image_resize_shape)
    iters = (len(train_dataset) // args.batch_size) * args.epochs
    print(f"  {iters} iterations ({args.epochs}e × {len(train_dataset)}s / {args.batch_size}b)")

    # --- Setup training ---
    from mlx_vlm.trainer.utils import unfreeze_modules, find_all_linear_names, \
        get_peft_model, print_trainable_parameters

    if full:
        unfreeze_modules(model, ["language_model"])
    else:
        modules = find_all_linear_names(model.language_model)
        model = get_peft_model(model, modules, rank=args.lora_rank,
                                alpha=args.lora_alpha, dropout=0.05, verbose=False)

    print_trainable_parameters(model)
    optimizer = optim.Adam(learning_rate=lr)

    # --- Train ---
    from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
    t_args = TrainingArgs(
        batch_size=args.batch_size, iters=iters,
        steps_per_report=5, steps_per_eval=100, steps_per_save=100,
        val_batches=4, max_seq_length=args.max_seq_len,
        adapter_file=output_path, grad_checkpoint=True,
        learning_rate=lr, grad_clip=1.0,
        gradient_accumulation_steps=1, full_finetune=full,
    )

    print(f"\n{'='*60}\nStarting training...\n{'='*60}\n")
    train(model=model, optimizer=optimizer, train_dataset=train_dataset,
          val_dataset=None, args=t_args, train_on_completions=True, assistant_id=77091)

    print(f"\nDone. Saved to {output_path}")


if __name__ == "__main__":
    main()
