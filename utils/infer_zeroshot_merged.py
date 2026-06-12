"""
Zero-shot clip-level inference with merged class taxonomy (Zou et al. WACV 2026).

Merged classes based on semantic clusters:
  - PhysicalRestraint : Abuse + Arrest
  - FireHazard        : Arson + Explosion
  - Theft             : Robbery + Stealing + Shoplifting

Remaining standalone: Assault, Burglary, Fighting, RoadAccidents, Shooting, Vandalism, Normal

Usage:
    python utils/infer_zeroshot_merged.py --video /path/to/Abuse006_x264.mp4
"""

import argparse
import json
import os
import re
from pathlib import Path

from PIL import Image

DATA_DIR   = os.environ.get("DATA_ROOT", "/Volumes/T7/research-vlm/data")
MODEL_ID   = os.environ.get("VLM_MODEL", "mlx-community/Qwen3-VL-2B-Instruct-8bit")
NUM_FRAMES = 8

# original → merged mapping
MERGE_MAP = {
    "Abuse":       "PhysicalRestraint",
    "Arrest":      "PhysicalRestraint",
    "Arson":       "FireHazard",
    "Explosion":   "FireHazard",
    "Robbery":     "Theft",
    "Stealing":    "Theft",
    "Shoplifting": "Theft",
    # standalone
    "Assault":      "Assault",
    "Burglary":     "Burglary",
    "Fighting":     "Fighting",
    "RoadAccidents":"RoadAccidents",
    "Shooting":     "Shooting",
    "Vandalism":    "Vandalism",
    "Normal":       "Normal",
}

MERGED_CLASSES = [
    "PhysicalRestraint",  # Abuse + Arrest
    "FireHazard",         # Arson + Explosion
    "Theft",              # Robbery + Stealing + Shoplifting
    "Assault",
    "Burglary",
    "Fighting",
    "RoadAccidents",
    "Shooting",
    "Vandalism",
    "Normal",
]

CLASSES_STR = ", ".join(MERGED_CLASSES)

PROMPT_TEMPLATE = """\
You are a surveillance video analyst. Watch these frames from a CCTV clip.

Classify the activity into ONE of these categories:
{classes}

Category descriptions:
- PhysicalRestraint: person being physically held down, restrained, or subjected to excessive force (abuse, police restraint)
- FireHazard: fire, arson, explosion, or fire-related destruction
- Theft: robbery, stealing, shoplifting, or any theft of property
- Assault: physical attack or assault on a person
- Burglary: breaking into a building or property
- Fighting: physical fight or brawl between multiple people
- RoadAccidents: vehicle collision or road accident
- Shooting: gunshot or shooting incident
- Vandalism: property damage or vandalism
- Normal: no criminal activity visible

Rules:
- Choose the MOST specific matching category.
- Choose Normal if no criminal activity is clearly visible.
- Reply with ONLY the category name, nothing else.

Category:"""

ALL_ORIGINAL = list(MERGE_MAP.keys())


def to_merged(label: str) -> str:
    return MERGE_MAP.get(label, label)


def extract_label(video_key: str) -> str:
    if "Normal" in video_key or "normal" in video_key:
        return "Normal"
    for cat in ALL_ORIGINAL:
        if video_key.startswith(cat):
            return cat
    m = re.match(r"^([A-Z][a-z]+)", video_key)
    if not m:
        return "Normal"
    label = m.group(1)
    return "RoadAccidents" if label == "Road" else (label if label in set(ALL_ORIGINAL) else "Normal")


def extract_frames(video_path: str, t_start: float, t_end: float, n: int = NUM_FRAMES) -> list[Image.Image]:
    import av
    collected: dict[int, Image.Image] = {}
    try:
        container = av.open(video_path)
        stream    = container.streams.video[0]
        dur       = float(stream.duration * stream.time_base) if stream.duration else 1e9
        t_start   = max(0.0, min(t_start, dur))
        t_end     = max(t_start + 0.1, min(t_end, dur))

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
        container.close()
    except Exception as e:
        print(f"  [WARN] frame extraction: {e}")

    if not collected:
        return [Image.new("RGB", (224, 224), 0)] * n

    for i in range(n):
        if i not in collected:
            collected[i] = collected[min(collected.keys(), key=lambda k: abs(k - i))]
    return [collected[i] for i in range(n)]


def load_model():
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    print(f"Loading {MODEL_ID} ...")
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)
    print("Model ready\n")
    return model, processor, config


def classify_clip(model, processor, config, frames: list[Image.Image]) -> str:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt    = PROMPT_TEMPLATE.format(classes=CLASSES_STR)
    formatted = apply_chat_template(processor, config, prompt, num_images=len(frames))
    result    = generate(model, processor, formatted, image=frames, max_tokens=20, verbose=False)
    raw       = (result.text if hasattr(result, "text") else str(result)).strip()

    raw_lower = raw.lower()
    for cat in MERGED_CLASSES:
        if cat.lower() in raw_lower:
            return cat
    if "normal" in raw_lower:
        return "Normal"
    return raw


def find_uca_entry(video_id: str, split: str) -> tuple[dict, str] | tuple[None, None]:
    splits = [split] if split != "auto" else ["Train", "Val", "Test"]
    for s in splits:
        for suffix in [
            f"UCFCrime_{s}_gemma_4_12b_it_8bit.json",
            f"UCFCrime_{s}.json",
        ]:
            path = os.path.join(DATA_DIR, "classified", suffix)
            if not os.path.exists(path):
                path = os.path.join(DATA_DIR, suffix)
            if not os.path.exists(path):
                continue
            data = json.load(open(path))
            if video_id in data:
                return data[video_id], s
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True)
    parser.add_argument("--split",      default="auto", choices=["auto", "Train", "Val", "Test"])
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--out",        default=None)
    args = parser.parse_args()

    video_path = args.video
    video_id   = Path(video_path).stem
    parent_orig   = extract_label(video_id)
    parent_merged = to_merged(parent_orig)

    print(f"Video          : {video_id}")
    print(f"Parent (orig)  : {parent_orig}")
    print(f"Parent (merged): {parent_merged}")

    entry, found_split = find_uca_entry(video_id, args.split)
    if entry is None:
        print(f"[WARN] {video_id} not found in UCA JSON")
        return

    sentences  = entry.get("sentences", [])
    timestamps = entry.get("timestamps", [])
    print(f"Split          : {found_split}  |  Sentences: {len(sentences)}\n")

    model, processor, config = load_model()

    results = []
    for i, sent in enumerate(sentences):
        if i >= len(timestamps):
            break
        text      = sent.get("text", "") if isinstance(sent, dict) else str(sent)
        gt_orig   = sent.get("class", parent_orig) if isinstance(sent, dict) else parent_orig
        gt_merged = to_merged(gt_orig)
        number    = sent.get("number", i + 1) if isinstance(sent, dict) else i + 1
        t_start, t_end = float(timestamps[i][0]), float(timestamps[i][1])

        frames = extract_frames(video_path, t_start, t_end, args.num_frames)
        pred   = classify_clip(model, processor, config, frames)

        match = "✓" if pred == gt_merged else "✗"
        print(f"#{number:3d} [{t_start:.1f}s-{t_end:.1f}s]  GT={gt_merged:<20} PRED={pred:<20} {match}")
        print(f"       (orig GT: {gt_orig})  {text[:80]}")
        print()

        results.append({
            "number":    number,
            "start":     t_start,
            "end":       t_end,
            "gt_orig":   gt_orig,
            "gt_merged": gt_merged,
            "pred":      pred,
            "match":     pred == gt_merged,
            "text":      text,
        })

    total   = len(results)
    correct = sum(r["match"] for r in results)
    print(f"Accuracy (merged): {correct}/{total} ({100*correct/max(total,1):.1f}%)")

    if args.out:
        out = {"video_id": video_id, "parent_orig": parent_orig,
               "parent_merged": parent_merged, "clips": results}
        json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
