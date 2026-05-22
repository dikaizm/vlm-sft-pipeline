"""
Path 1 test — base SmolVLM2 with frame timestamps injected into the text prompt.

Runs 5 random test samples and compares:
  A) Prompt WITHOUT frame timestamps (baseline)
  B) Prompt WITH frame timestamps (Path 1)

Usage:
    python vlm-sft-pipeline/test_path1.py
    MODEL_ID=HuggingFaceTB/SmolVLM2-2.2B-Video-Instruct python vlm-sft-pipeline/test_path1.py
"""

import json
import os
import random
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.video_utils import VideoMetadata

DATA_ROOT  = os.environ.get("DATA_ROOT", "/Volumes/T7/research-vlm/data")
VIDEO_ROOT = f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos"
TEST_JSON  = f"{DATA_ROOT}/UCFCrime_Test.json"
MODEL_ID   = os.environ.get("MODEL_ID", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

NUM_FRAMES     = 16
MAX_DURATION   = 90.0
MAX_NEW_TOKENS = 256
N_SAMPLES      = 5
SEED           = 42

PROMPT_BASELINE = (
    "List every activity in this surveillance video. "
    "Format: N. [start, end] description."
)


def prompt_with_timestamps(frame_times: list[float], duration: float) -> str:
    ts_str = ", ".join(f"{t:.1f}" for t in frame_times)
    return (
        f"This surveillance video spans 0–{duration:.1f}s. "
        f"Sampled frames at: {ts_str}s.\n"
        "List every activity. Format: N. [start, end] description."
    )


def device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def extract_frames(video_path: str, start: float, end: float, n: int) -> list:
    try:
        import av
        container = av.open(video_path)
        stream    = container.streams.video[0]
        duration  = float(stream.duration * stream.time_base) if stream.duration else end
        t_start   = max(0.0, min(start, duration))
        t_end     = max(t_start + 0.1, min(end, duration))
        collected = {}
        container.seek(int(t_start * 1_000_000), any_frame=False, backward=True)
        for frame in container.decode(video=0):
            t    = float(frame.pts * stream.time_base)
            if t > t_end + 1.0:
                break
            slot = int((t - t_start) / (t_end - t_start + 1e-9) * n)
            slot = max(0, min(slot, n - 1))
            if slot not in collected:
                collected[slot] = frame.to_image()
            if len(collected) >= n:
                break
        container.close()
        if collected:
            for i in range(n):
                if i not in collected:
                    collected[i] = collected[min(collected.keys(), key=lambda k: abs(k - i))]
            return [collected[i] for i in range(n)]
    except Exception as e:
        print(f"  [WARN] frame extraction failed: {e}")
    return [Image.new("RGB", (224, 224), color=0)] * n


def make_metadata(start: float, end: float, n: int) -> VideoMetadata:
    frame_times = [start + i * (end - start) / max(n - 1, 1) for i in range(n)]
    return VideoMetadata(
        total_num_frames=max(int(end * 10), n),
        fps=10.0,
        frames_indices=[round(t * 10) for t in frame_times],
        duration=float(end),
    )


def run(model, processor, device, dtype, frames, start, end, prompt):
    meta = make_metadata(start, end, len(frames))
    messages = [{
        "role": "user",
        "content": [{"type": "video"}, {"type": "text", "text": prompt}],
    }]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[text], videos=[[frames]], video_metadata=[meta],
        return_tensors="pt", padding=True, truncation=True, max_length=2048,
    ).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()


def load_samples(n: int) -> list[dict]:
    mp4_map = {}
    for root, _, files in os.walk(VIDEO_ROOT):
        for f in files:
            if f.endswith(".mp4"):
                mp4_map[f] = os.path.join(root, f)

    with open(TEST_JSON) as f:
        data = json.load(f)

    items = []
    for vid_id, ann in data.items():
        cat   = re.sub(r"\d+_x264$", "", vid_id)
        path  = mp4_map.get(f"{vid_id}.mp4") or os.path.join(VIDEO_ROOT, cat, f"{vid_id}.mp4")
        if not os.path.isfile(path):
            continue
        duration = float(ann.get("duration", MAX_DURATION))
        eff_end  = min(duration, MAX_DURATION)
        pairs = [
            (float(s), min(float(e), eff_end), sent.strip())
            for (s, e), sent in zip(ann["timestamps"], ann["sentences"])
            if float(e) > float(s) and float(s) <= eff_end
        ]
        if pairs:
            items.append({"video_id": vid_id, "path": path,
                          "duration": duration, "eff_end": eff_end, "pairs": pairs})

    random.seed(SEED)
    random.shuffle(items)
    return items[:n]


def main():
    device, dtype = device_and_dtype()
    print(f"Model  : {MODEL_ID}")
    print(f"Device : {device}  dtype: {dtype}\n")

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"Params : {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M\n")

    samples = load_samples(N_SAMPLES)
    if not samples:
        sys.exit("No test samples found — check VIDEO_ROOT / TEST_JSON paths.")

    for i, s in enumerate(samples, 1):
        print(f"{'='*70}")
        print(f"[{i}/{N_SAMPLES}] {s['video_id']}  duration={s['eff_end']:.1f}s")

        # Ground truth
        print("GT:")
        for j, (ts, te, sent) in enumerate(s["pairs"][:5], 1):
            print(f"  {j}. [{ts:.1f}, {te:.1f}] {sent}")

        frames     = extract_frames(s["path"], 0.0, s["eff_end"], NUM_FRAMES)
        frame_times = [i * s["eff_end"] / max(NUM_FRAMES - 1, 1) for i in range(NUM_FRAMES)]

        # A: baseline prompt
        print("\nA) Baseline (no timestamps in prompt):")
        out_a = run(model, processor, device, dtype, frames, 0.0, s["eff_end"], PROMPT_BASELINE)
        print(f"  {out_a[:500]}")

        # B: Path 1 — prompt includes frame timestamps
        print("\nB) Path 1 (frame timestamps in prompt):")
        p1 = prompt_with_timestamps(frame_times, s["eff_end"])
        out_b = run(model, processor, device, dtype, frames, 0.0, s["eff_end"], p1)
        print(f"  {out_b[:500]}")
        print()


if __name__ == "__main__":
    main()
