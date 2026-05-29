"""
Realtime surveillance pipeline — webcam input, fine-tuned SmolVLM2 inference.

Captures a rolling window of frames from the webcam, samples clips at a fixed
interval, runs the VLM, and overlays the activity description + anomaly alert
on the live video display.

Usage:
    python vlm-sft-pipeline/infer/infer_realtime.py --model ./output/smolvlm2-500m-full-sft
    python vlm-sft-pipeline/infer/infer_realtime.py --model ./output/smolvlm2-2b-lora-sft --camera 0
    python vlm-sft-pipeline/infer/infer_realtime.py --model HuggingFaceTB/SmolVLM2-500M-Video-Instruct  # zero-shot

Controls:
    q / ESC  — quit
    s        — save current frame + prediction to ./output/realtime/
    p        — pause/resume inference
"""

import argparse
import os
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.video_utils import VideoMetadata

# ---------------------------------------------------------------------------
# Crime category keywords for anomaly detection
# ---------------------------------------------------------------------------

CRIME_KEYWORDS = {
    "Abuse":         ["abuse", "hit", "beat", "assault", "attack", "hurt", "harm", "force", "violenc"],
    "Arrest":        ["arrest", "handcuff", "detain", "custody", "police", "officer", "apprehend"],
    "Arson":         ["fire", "burn", "flame", "arson", "ignit", "torch"],
    "Assault":       ["assault", "punch", "kick", "strik", "attack", "fight", "aggress"],
    "Burglary":      ["burglar", "break", "enter", "intrude", "trespass", "unlock", "pry"],
    "Explosion":     ["explo", "blast", "bomb", "detonate", "burst"],
    "Fighting":      ["fight", "brawl", "wrestl", "struggle", "altercation", "scuffle"],
    "RoadAccidents": ["crash", "collision", "accident", "impact", "veer", "vehicle"],
    "Robbery":       ["rob", "steal", "grab", "snatch", "threaten", "weapon", "gun", "knife"],
    "Shooting":      ["shoot", "gun", "firearm", "shot", "bullet", "weapon"],
    "Shoplifting":   ["shoplift", "steal", "conceal", "merchandise", "pocket", "take"],
    "Stealing":      ["steal", "theft", "take", "grab", "snatch", "pick"],
    "Vandalism":     ["vandal", "spray", "graffiti", "damage", "smash", "break", "destr"],
}

ANOMALY_COLOR  = (0, 0, 220)    # BGR red for alert
NORMAL_COLOR   = (0, 200, 0)    # BGR green for normal
TEXT_COLOR     = (255, 255, 255)
SHADOW_COLOR   = (0, 0, 0)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# VLM inference helpers
# ---------------------------------------------------------------------------

def _make_video_metadata(n_frames: int, clip_duration: float) -> VideoMetadata:
    frame_timestamps = [i * clip_duration / max(n_frames - 1, 1) for i in range(n_frames)]
    return VideoMetadata(
        total_num_frames=n_frames,
        fps=1.0,
        frames_indices=list(range(n_frames)),
        duration=clip_duration,
    )


def run_inference(model, processor, device, frames: list[Image.Image],
                  clip_duration: float, max_new_tokens: int = 80) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "Describe the activity shown in this surveillance video clip."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    metadata = _make_video_metadata(len(frames), clip_duration)
    dtype = torch.float32 if device.type in ("mps", "cpu") else torch.bfloat16

    inputs = processor(
        text=[text],
        videos=[[frames]],
        video_metadata=[metadata],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomaly(description: str) -> tuple[bool, str]:
    """Return (is_anomaly, matched_category). Case-insensitive keyword scan."""
    desc_lower = description.lower()
    for category, keywords in CRIME_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return True, category
    return False, ""


# ---------------------------------------------------------------------------
# Frame buffer
# ---------------------------------------------------------------------------

class FrameBuffer:
    """Thread-safe rolling buffer of (PIL.Image, timestamp) tuples."""

    def __init__(self, maxlen: int):
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, frame: Image.Image, ts: float):
        with self._lock:
            self._buf.append((frame, ts))

    def sample(self, n: int) -> list[Image.Image]:
        with self._lock:
            buf = list(self._buf)
        if not buf:
            return []
        if len(buf) <= n:
            return [f for f, _ in buf]
        step = len(buf) / n
        indices = [int(i * step) for i in range(n)]
        return [buf[i][0] for i in indices]

    def __len__(self):
        with self._lock:
            return len(self._buf)


# ---------------------------------------------------------------------------
# Inference worker (runs in background thread)
# ---------------------------------------------------------------------------

class InferenceWorker(threading.Thread):
    def __init__(self, model, processor, device, frame_buffer: FrameBuffer,
                 n_frames: int, clip_duration: float, result_queue: queue.Queue):
        super().__init__(daemon=True)
        self.model        = model
        self.processor    = processor
        self.device       = device
        self.frame_buffer = frame_buffer
        self.n_frames     = n_frames
        self.clip_duration = clip_duration
        self.result_queue = result_queue
        self._stop_event  = threading.Event()
        self._pause_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def toggle_pause(self):
        if self._pause_event.is_set():
            self._pause_event.clear()
        else:
            self._pause_event.set()

    @property
    def paused(self):
        return self._pause_event.is_set()

    def run(self):
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.1)
                continue

            frames = self.frame_buffer.sample(self.n_frames)
            if len(frames) < self.n_frames:
                time.sleep(0.2)
                continue

            t0 = time.time()
            try:
                desc = run_inference(
                    self.model, self.processor, self.device,
                    frames, self.clip_duration
                )
            except Exception as e:
                desc = f"[inference error: {e}]"

            latency = time.time() - t0
            is_anomaly, category = detect_anomaly(desc)

            # Drain old results, keep queue fresh
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break

            self.result_queue.put({
                "description": desc,
                "is_anomaly":  is_anomaly,
                "category":    category,
                "latency":     latency,
                "timestamp":   datetime.now().strftime("%H:%M:%S"),
            })

            # Wait before next inference cycle
            time.sleep(self.clip_duration)


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def _put_text_with_shadow(img, text: str, pos: tuple, font_scale: float,
                           color: tuple, thickness: int = 1):
    cv2.putText(img, text, (pos[0]+1, pos[1]+1), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, SHADOW_COLOR, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            if cur:
                lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def render_overlay(frame_bgr, result: dict | None, paused: bool,
                   buffer_fill: int, n_frames_needed: int):
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    # Status bar at bottom
    bar_h = 30
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (30, 30, 30), -1)

    status = "PAUSED" if paused else ("BUFFERING..." if buffer_fill < n_frames_needed else "LIVE")
    _put_text_with_shadow(overlay, status, (8, h - 9), 0.55, TEXT_COLOR)

    if result:
        ts = result.get("timestamp", "")
        lat = result.get("latency", 0)
        _put_text_with_shadow(overlay, f"{ts}  {lat:.1f}s", (w - 160, h - 9), 0.45, TEXT_COLOR)

    # Anomaly border
    if result and result["is_anomaly"]:
        border = 6
        cv2.rectangle(overlay, (0, 0), (w, h), ANOMALY_COLOR, border)

        # Alert banner
        cv2.rectangle(overlay, (0, 0), (w, 48), (0, 0, 180), -1)
        cat = result["category"].upper()
        _put_text_with_shadow(overlay, f"ANOMALY DETECTED — {cat}", (10, 32),
                               0.75, (255, 255, 0), thickness=2)

    # Description box
    if result and result["description"] and not result["description"].startswith("["):
        desc_lines = wrap_text(result["description"], max_chars=max(30, w // 14))
        box_h = len(desc_lines) * 22 + 12
        box_y = h - bar_h - box_h - 4
        cv2.rectangle(overlay, (0, box_y), (w, h - bar_h), (20, 20, 20), -1)
        for idx, line in enumerate(desc_lines):
            color = ANOMALY_COLOR if result["is_anomaly"] else NORMAL_COLOR
            _put_text_with_shadow(overlay, line, (8, box_y + 18 + idx * 22),
                                   0.50, color)

    cv2.addWeighted(overlay, 0.85, frame_bgr, 0.15, 0, frame_bgr)
    return frame_bgr


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Realtime surveillance with SmolVLM2")
    parser.add_argument("--model",        required=True,
                        help="Fine-tuned model dir or HuggingFace model ID")
    parser.add_argument("--base-model",   default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                        help="Base model ID for processor if checkpoint has none")
    parser.add_argument("--camera",       type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--video",        default=None,
                        help="Path to video file (overrides --camera)")
    parser.add_argument("--n-frames",     type=int, default=4,
                        help="Frames per inference clip (default: 4)")
    parser.add_argument("--clip-seconds", type=float, default=2.0,
                        help="Duration of rolling clip window in seconds (default: 2.0)")
    parser.add_argument("--fps",          type=float, default=10.0,
                        help="Target capture FPS (default: 10)")
    parser.add_argument("--save-dir",     default="./output/realtime",
                        help="Directory for saved frames")
    parser.add_argument("--width",        type=int, default=640)
    parser.add_argument("--height",       type=int, default=480)
    args = parser.parse_args()

    device = get_device()
    dtype  = torch.float32 if device.type in ("mps", "cpu") else torch.bfloat16
    print(f"Device : {device}  dtype: {dtype}")
    print(f"Model  : {args.model}")

    # Load model
    print("Loading model...")
    proc_src  = args.model if (Path(args.model) / "preprocessor_config.json").exists() else args.base_model
    processor = AutoProcessor.from_pretrained(proc_src)
    model     = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M\n")

    # Camera or video file
    src = args.video if args.video else args.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open {'video file: ' + args.video if args.video else 'camera ' + str(args.camera)}")
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    else:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
        args.fps = native_fps
        print(f"Video file: {args.video}  FPS: {native_fps:.1f}")

    # Buffer holds enough frames for the rolling clip window
    buf_maxlen = max(args.n_frames * 4, int(args.fps * args.clip_seconds * 2))
    frame_buf  = FrameBuffer(maxlen=buf_maxlen)
    result_q   = queue.Queue(maxsize=1)

    # Inference worker
    worker = InferenceWorker(
        model=model,
        processor=processor,
        device=device,
        frame_buffer=frame_buf,
        n_frames=args.n_frames,
        clip_duration=args.clip_seconds,
        result_queue=result_q,
    )
    worker.start()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("Realtime surveillance running.")
    print("  q / ESC : quit")
    print("  s       : save frame")
    print("  p       : pause/resume inference\n")

    last_result = None
    frame_interval = 1.0 / args.fps
    last_cap_time  = 0.0

    try:
        while True:
            now = time.time()
            if now - last_cap_time < frame_interval:
                time.sleep(0.001)
                continue
            last_cap_time = now

            ret, bgr = cap.read()
            if not ret:
                if args.video:
                    print("End of video.")
                    break
                print("[WARN] Frame capture failed, retrying...")
                time.sleep(0.1)
                continue

            # Push PIL frame into buffer
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            frame_buf.push(pil, now)

            # Check for new inference result
            try:
                last_result = result_q.get_nowait()
                if last_result["is_anomaly"]:
                    print(f"[ALERT] {last_result['timestamp']}  "
                          f"{last_result['category'].upper()}  —  "
                          f"{last_result['description'][:80]}")
                else:
                    print(f"[INFO]  {last_result['timestamp']}  "
                          f"{last_result['description'][:80]}")
            except queue.Empty:
                pass

            # Render
            display = render_overlay(
                bgr.copy(), last_result,
                paused=worker.paused,
                buffer_fill=len(frame_buf),
                n_frames_needed=args.n_frames,
            )
            cv2.imshow("SmolVLM2 Surveillance", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):   # q or ESC
                break
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(args.save_dir, f"frame_{ts}.jpg")
                cv2.imwrite(save_path, display)
                desc = last_result["description"] if last_result else ""
                print(f"Saved: {save_path}  ({desc[:60]})")
            elif key == ord('p'):
                worker.toggle_pause()
                state = "PAUSED" if worker.paused else "RESUMED"
                print(f"Inference {state}")

    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
