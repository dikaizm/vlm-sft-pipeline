"""
Render clips from inference results with caption overlay (PyAV + PIL).

Usage:
    python utils/render_clips.py --results results/fulltest_qwen3_ckpt350.json \
        --out-dir /tmp/rendered_clips --n 20

    # wrong predictions only
    python utils/render_clips.py --results results/fulltest_qwen3_ckpt350.json \
        --out-dir /tmp/rendered_wrong --only-wrong
"""

import argparse
import json
import os
import textwrap
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA_ROOT  = os.environ.get("DATA_ROOT", "/Volumes/T7/research-vlm/data")
VIDEO_ROOT = f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos"

FONT_PCT    = 0.030  # font size as fraction of frame height
LINE_WRAP   = 72
BOX_ALPHA   = 140  # 0-255


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def wrap_text(text: str, width: int = LINE_WRAP) -> str:
    return "\n".join(textwrap.wrap(text, width))


def draw_overlay(frame_np: np.ndarray, gt_class: str, gt_text: str,
                 ft_class: str, ft_text: str,
                 top_label: str = "GT", bot_label: str = "FT") -> np.ndarray:
    img = Image.fromarray(frame_np)
    w, h = img.size

    font_size   = max(8, int(h * FONT_PCT))
    line_h      = font_size + 2
    box_height  = line_h * 8 + 10  # 2 headers + ~3 body lines each + padding

    overlay = Image.new("RGBA", (w, box_height), (0, 0, 0, BOX_ALPHA))
    img_rgba = img.convert("RGBA")
    img_rgba.paste(overlay, (0, h - box_height), overlay)
    img = img_rgba.convert("RGB")

    draw = ImageDraw.Draw(img)
    font_header = get_font(font_size)
    font_body   = get_font(max(7, font_size - 1))

    y = h - box_height + 6

    # GT row
    draw.text((8, y), f"{top_label} [{gt_class}]", font=font_header, fill=(144, 238, 144))
    y += line_h + 2
    for line in wrap_text(gt_text).split("\n"):
        draw.text((8, y), line, font=font_body, fill=(220, 220, 220))
        y += line_h

    y += 4

    draw.text((8, y), f"{bot_label} [{ft_class}]", font=font_header, fill=(255, 215, 0))
    y += line_h + 2
    for line in wrap_text(ft_text).split("\n"):
        draw.text((8, y), line, font=font_body, fill=(220, 220, 220))
        y += line_h

    return np.array(img)


def find_video(video_id: str) -> str | None:
    for category in os.listdir(VIDEO_ROOT):
        p = Path(VIDEO_ROOT) / category / f"{video_id}.mp4"
        if p.exists():
            return str(p)
    return None


def render_clip(clip: dict, out_dir: Path, idx: int,
                top_label: str = "GT", bot_label: str = "FT") -> None:
    video_id = clip["video_id"]
    start    = float(clip["start"])
    end      = float(clip["end"])

    gt_class = clip.get("gt_class", "?")
    gt_text  = clip.get("gt", "")
    ft_class = clip.get("finetuned_class", "?")
    ft_text  = clip.get("finetuned", "")

    video_path = find_video(video_id)
    if video_path is None:
        print(f"[SKIP] {video_id} — not found")
        return

    out_path = out_dir / f"{idx:04d}_{video_id}_{int(start)}s.mp4"
    if out_path.exists():
        print(f"[SKIP] {out_path.name}")
        return

    try:
        in_container = av.open(video_path)
        video_stream = in_container.streams.video[0]
        fps = float(video_stream.average_rate)

        out_container = av.open(str(out_path), mode="w")
        out_stream = out_container.add_stream("libx264", rate=int(fps))
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"crf": "23", "preset": "fast"}

        seek_pts = int(start / video_stream.time_base)
        in_container.seek(seek_pts, stream=video_stream)

        width = height = None
        for packet in in_container.demux(video_stream):
            for frame in packet.decode():
                pts_sec = float(frame.pts * video_stream.time_base)
                if pts_sec < start:
                    continue
                if pts_sec > end:
                    break

                frame_np = frame.to_ndarray(format="rgb24")

                if width is None:
                    height, width = frame_np.shape[:2]
                    out_stream.width  = width
                    out_stream.height = height

                frame_np = draw_overlay(frame_np, gt_class, gt_text, ft_class, ft_text,
                                        top_label=top_label, bot_label=bot_label)

                out_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                for pkt in out_stream.encode(out_frame):
                    out_container.mux(pkt)
            else:
                if width and float(packet.pts * video_stream.time_base if packet.pts else 0) > end:
                    break
                continue
            break

        for pkt in out_stream.encode():
            out_container.mux(pkt)

        out_container.close()
        in_container.close()
        print(f"[OK] {out_path.name}")

    except Exception as e:
        print(f"[ERROR] {video_id}: {e}")
        if out_path.exists():
            out_path.unlink()


import re as _re

ALL_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting",
    "Shoplifting", "Stealing", "Vandalism",
]


def _extract_parent(video_key: str) -> str:
    if "Normal" in video_key or "normal" in video_key:
        return "Normal"
    for cat in ALL_CATEGORIES:
        if video_key.startswith(cat):
            return cat
    m = _re.match(r"^([A-Z][a-z]+)", video_key)
    if not m:
        return "Normal"
    label = m.group(1)
    return "RoadAccidents" if label == "Road" else (label if label in set(ALL_CATEGORIES) else "Normal")


def draw_single_overlay(frame_np: np.ndarray, cls: str, text: str, number: int = 0) -> np.ndarray:
    """Two-row overlay: GT caption + Gemma4 label."""
    img    = Image.fromarray(frame_np)
    w, h   = img.size
    font_size  = max(8, int(h * FONT_PCT))
    line_h     = font_size + 2
    box_height = line_h * 8 + 10

    overlay  = Image.new("RGBA", (w, box_height), (0, 0, 0, BOX_ALPHA))
    img_rgba = img.convert("RGBA")
    img_rgba.paste(overlay, (0, h - box_height), overlay)
    img = img_rgba.convert("RGB")

    draw        = ImageDraw.Draw(img)
    font_header = get_font(font_size)
    font_body   = get_font(max(7, font_size - 1))

    y = h - box_height + 6
    num_str = f"#{number} " if number else ""
    draw.text((8, y), f"{num_str}GT [{cls}]", font=font_header, fill=(144, 238, 144))
    y += line_h + 2
    for line in wrap_text(text).split("\n"):
        draw.text((8, y), line, font=font_body, fill=(220, 220, 220))
        y += line_h

    return np.array(img)


def _build_intervals(data: dict) -> dict[str, list]:
    """video_id → sorted list of (start, end, cls, text)."""
    result: dict[str, list] = {}
    for vk, vdata in data.items():
        sentences  = vdata.get("sentences", [])
        timestamps = vdata.get("timestamps", [])
        ivs = []
        for i, sent in enumerate(sentences):
            if i >= len(timestamps):
                continue
            text   = sent.get("text", "") if isinstance(sent, dict) else str(sent)
            cls    = sent.get("class", _extract_parent(vk)) if isinstance(sent, dict) else _extract_parent(vk)
            number = sent.get("number", i + 1) if isinstance(sent, dict) else i + 1
            ivs.append((float(timestamps[i][0]), float(timestamps[i][1]), cls, text, number))
        ivs.sort(key=lambda x: x[0])
        result[vk] = ivs
    return result


def render_video_uca(video_id: str, intervals: list, out_dir: Path, idx: int) -> None:
    """Render full video with dynamic per-sentence Gemma4 label overlay."""
    video_path = find_video(video_id)
    if video_path is None:
        print(f"[SKIP] {video_id} — not found")
        return

    out_path = out_dir / f"{idx:04d}_{video_id}.mp4"
    if out_path.exists():
        print(f"[SKIP] {out_path.name}")
        return

    try:
        in_container  = av.open(video_path)
        video_stream  = in_container.streams.video[0]
        fps           = float(video_stream.average_rate)

        out_container = av.open(str(out_path), mode="w")
        out_stream    = out_container.add_stream("libx264", rate=int(fps))
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"crf": "23", "preset": "fast"}

        width = height = None
        for packet in in_container.demux(video_stream):
            for frame in packet.decode():
                pts_sec  = float(frame.pts * video_stream.time_base)
                frame_np = frame.to_ndarray(format="rgb24")

                if width is None:
                    height, width  = frame_np.shape[:2]
                    out_stream.width  = width
                    out_stream.height = height

                active = next((iv for iv in intervals if iv[0] <= pts_sec <= iv[1]), None)
                if active:
                    frame_np = draw_single_overlay(frame_np, active[2], active[3],
                                                   number=active[4] if len(active) > 4 else 0)

                out_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                for pkt in out_stream.encode(out_frame):
                    out_container.mux(pkt)

        for pkt in out_stream.encode():
            out_container.mux(pkt)

        out_container.close()
        in_container.close()
        print(f"[OK] {out_path.name}")

    except Exception as e:
        print(f"[ERROR] {video_id}: {e}")
        if out_path.exists():
            out_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",    required=True)
    parser.add_argument("--out-dir",    default="rendered_clips")
    parser.add_argument("--n",          type=int, default=None)
    parser.add_argument("--only-wrong", action="store_true",
                        help="Only clips where Gemma4 label != parent class (UCA) or ft!=gt (infer)")
    parser.add_argument("--mode",       choices=["infer", "uca"], default="infer",
                        help="infer: inference results JSON; uca: classified UCA JSON")
    parser.add_argument("--class",      dest="filter_class", default=None,
                        help="Filter to one parent class, e.g. Abuse")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "uca":
        intervals_map = _build_intervals(data)

        if args.filter_class:
            intervals_map = {
                vk: ivs for vk, ivs in intervals_map.items()
                if _extract_parent(vk) == args.filter_class
            }
            print(f"Videos with parent={args.filter_class}: {len(intervals_map)}")

        if args.only_wrong:
            # keep videos with at least one sentence where gemma4 != parent
            intervals_map = {
                vk: ivs for vk, ivs in intervals_map.items()
                if any(iv[2] != _extract_parent(vk) for iv in ivs)
            }
            print(f"Videos with Gemma4≠parent: {len(intervals_map)}")

        video_ids = list(intervals_map.keys())
        if args.n:
            video_ids = video_ids[:args.n]

        for i, vk in enumerate(video_ids):
            render_video_uca(vk, intervals_map[vk], out_dir, i)

    else:
        clips = data["clips"]
        if args.only_wrong:
            clips = [c for c in clips if c.get("finetuned_class") != c.get("gt_class")]
            print(f"Wrong-prediction clips: {len(clips)}")
        if args.n:
            clips = clips[:args.n]
        for i, clip in enumerate(clips):
            render_clip(clip, out_dir, i)

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
