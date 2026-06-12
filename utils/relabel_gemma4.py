"""
Relabel UCF-Crime sentences using Gemma 4 VLM on actual video frames.
Follows the same structure as classify_sentences.py.

For each sentence in a crime video, Gemma 4 sees the clip frames and decides:
  Normal  — ordinary activity visible, label Normal
  Anomaly — crime activity visible, label = parent video class

Normal parent videos are always labelled Normal (no inference needed).

Usage:
    python utils/relabel_gemma4.py --split all
    python utils/relabel_gemma4.py --split Train
    python utils/relabel_gemma4.py --split Val
    python utils/relabel_gemma4.py --split Test

Env vars:
    DATA_ROOT      path to data dir  (default: /Volumes/T7/research-vlm/data)
    GEMMA4_MODEL   MLX model id      (default: mlx-community/gemma-4-12B-it-8bit)
"""

import argparse
import json
import os
import re
import time
from datetime import timedelta
from pathlib import Path

from PIL import Image

DATA_DIR   = os.environ.get("DATA_ROOT", "/Volumes/T7/research-vlm/data")
OUTPUT_DIR = os.path.join(DATA_DIR, "classified")
VIDEO_ROOT = os.path.join(DATA_DIR, "UCF_Crimes/UCF_Crimes/Videos")
MODEL_ID   = os.environ.get("GEMMA4_MODEL", "mlx-community/gemma-4-12B-it-8bit")

NUM_FRAMES = 8

ALL_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting",
    "Shoplifting", "Stealing", "Vandalism",
]

PROMPT_TEMPLATE = """\
You are a surveillance video analyst. Watch these frames from a CCTV clip.

Parent video category: {parent_class}
Annotation sentence: "{sentence}"

Based on what you actually see in the frames, answer ONE question:
Is the activity in this clip Normal (ordinary activity), or Anomaly ({parent_class} is happening)?

Rules:
- Answer Anomaly only if the criminal/anomalous activity is VISIBLE in the frames.
- If frames show ordinary activity (walking, standing, waiting) answer Normal, even if the parent video is a crime video.
- Reply with ONLY: Normal  OR  Anomaly

Answer:"""


def extract_label(video_key: str) -> str:
    if "Normal" in video_key or "normal" in video_key:
        return "Normal"
    for cat in ALL_CATEGORIES:
        if video_key.startswith(cat):
            return cat
    match = re.match(r"^([A-Z][a-z]+)", video_key)
    if not match:
        return "Normal"
    label = match.group(1)
    return "RoadAccidents" if label == "Road" else (label if label in set(ALL_CATEGORIES) else "Normal")


def fmt_dur(s: float) -> str:
    return str(timedelta(seconds=int(s)))


def find_video(video_id: str) -> str | None:
    for category in os.listdir(VIDEO_ROOT):
        p = Path(VIDEO_ROOT) / category / f"{video_id}.mp4"
        if p.exists():
            return str(p)
    return None


def load_model():
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    print(f"Loading {MODEL_ID} ...")
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)
    print("Model loaded (MLX)\n")
    return model, processor, config


def extract_all_frames(video_path: str, timestamps: list) -> dict[int, list[Image.Image]]:
    """Extract frames for all sentence indices at once — one video open."""
    result: dict[int, list[Image.Image]] = {}
    n = NUM_FRAMES
    try:
        import av
        container = av.open(video_path)
        stream    = container.streams.video[0]
        dur       = float(stream.duration * stream.time_base) if stream.duration else 1e9

        for idx, ts in enumerate(timestamps):
            t_start = max(0.0, min(float(ts[0]), dur))
            t_end   = max(t_start + 0.1, min(float(ts[1]), dur))
            collected: dict[int, Image.Image] = {}

            container.seek(int(t_start * 1_000_000), any_frame=False, backward=True)
            for frame in container.decode(video=0):
                t = float(frame.pts * stream.time_base)
                if t > t_end + 1.0:
                    break
                slot = int((t - t_start) / (t_end - t_start + 1e-9) * n)
                slot = max(0, min(slot, n - 1))
                if slot not in collected:
                    collected[slot] = frame.to_image()
                if len(collected) >= n:
                    break

            if collected:
                for i in range(n):
                    if i not in collected:
                        collected[i] = collected[min(collected.keys(), key=lambda k: abs(k - i))]
                result[idx] = [collected[i] for i in range(n)]
            else:
                result[idx] = [Image.new("RGB", (224, 224), 0)] * n

        container.close()
    except Exception as e:
        print(f"  [WARN] frame extraction: {e}")
        for idx in range(len(timestamps)):
            result[idx] = [Image.new("RGB", (224, 224), 0)] * NUM_FRAMES
    return result


def classify_clip(model, processor, config, frames: list[Image.Image],
                  sentence: str, parent_class: str) -> str:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt    = PROMPT_TEMPLATE.format(parent_class=parent_class, sentence=sentence)
    formatted = apply_chat_template(processor, config, prompt, num_images=len(frames))
    result    = generate(model, processor, formatted, image=frames, max_tokens=10, verbose=False)
    raw       = (result.text if hasattr(result, "text") else str(result)).strip().lower()

    if "anomaly" in raw:
        return parent_class
    return "Normal"


def process_split(split_name: str, model, processor, config) -> str:
    from tqdm import tqdm

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_slug = MODEL_ID.split("/")[-1].replace("-", "_").lower()
    inp = os.path.join(DATA_DIR, f"UCFCrime_{split_name}.json")
    out = os.path.join(OUTPUT_DIR, f"UCFCrime_{split_name}_{model_slug}.json")

    # resume
    existing: dict = {}
    if os.path.exists(out):
        with open(out) as f:
            existing = json.load(f)
        print(f"[{split_name}] Resuming: {len(existing)} videos done")

    with open(inp) as f:
        data = json.load(f)

    video_keys = list(data.keys())
    pending    = [k for k in video_keys if k not in existing or not all(
        isinstance(s, dict) and "class" in s
        for s in existing[k].get("sentences", [])
    )]

    total_sent = sum(len(data[k]["sentences"]) for k in pending)
    print(f"[{split_name}] {len(pending)} videos, {total_sent} sentences pending\n")

    if not pending:
        print(f"[{split_name}] All done!")
        return out

    start_t  = time.time()
    done_sent = 0

    pbar = tqdm(total=total_sent, unit="sent", desc=split_name, dynamic_ncols=True)

    for vk in pending:
        label      = extract_label(vk)
        sentences  = data[vk]["sentences"]
        video_path = find_video(vk)
        timestamps = data[vk].get("timestamps", [])
        texts      = [s if isinstance(s, str) else s.get("text", "") for s in sentences]

        # Normal parent videos: all Normal, no VLM. Crime videos: VLM every sentence.
        need_vlm = []
        if label != "Normal" and video_path is not None:
            need_vlm = [i for i in range(len(texts)) if i < len(timestamps)]

        # one video open: extract frames for all sentences needing VLM
        frames_cache: dict[int, list] = {}
        if need_vlm:
            vlm_timestamps = [timestamps[i] for i in need_vlm]
            extracted = extract_all_frames(video_path, vlm_timestamps)
            frames_cache = {need_vlm[j]: extracted[j] for j in range(len(need_vlm))}

        new_sents = []
        for i, text in enumerate(texts):
            if i in frames_cache:
                cls = classify_clip(model, processor, config, frames_cache[i], text, label)
            else:
                cls = "Normal"   # Normal parent, missing video, or missing timestamp

            new_sents.append({"text": text, "class": cls})
            done_sent += 1
            pbar.update(1)
            pbar.set_postfix(video=vk[:22], cls=cls)

        entry = dict(data[vk])
        entry["sentences"] = new_sents
        existing[vk] = entry

        # incremental atomic save after each video (temp + rename so a kill
        # mid-write can't corrupt the resume file)
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(existing, f, indent=4, ensure_ascii=False)
        os.replace(tmp, out)

    pbar.close()
    elapsed = time.time() - start_t
    print(f"\n[{split_name}] Done: {len(existing)} videos → {out} ({fmt_dur(elapsed)})")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["Train", "Val", "Test", "all"], default="all")
    args = parser.parse_args()

    model, processor, config = load_model()

    splits = ["Train", "Val", "Test"] if args.split == "all" else [args.split]
    for s in splits:
        process_split(s, model, processor, config)


if __name__ == "__main__":
    main()
