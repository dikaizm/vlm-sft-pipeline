"""
Evaluation of activity description model on UCF-Crime test set (no timestamps).

Parses numbered-list output and compares against GT sentences.
Metrics: ROUGE-L, BLEU-4, BERTScore.
No tIoU (no temporal localization in this approach).

Usage:
    python vlm-sft-pipeline/eval_desc.py --n 50
    python vlm-sft-pipeline/eval_desc.py --model ./output/smolvlm2-desc-sft --n 100
"""

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, PreTrainedTokenizerBase
from transformers.video_utils import VideoMetadata
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bert_score_fn

# bert_score 0.3.13 calls build_inputs_with_special_tokens removed in transformers ≥5.0
if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
    def _build(self, token_ids_0, token_ids_1=None):
        cls = [self.cls_token_id] if getattr(self, "cls_token_id", None) is not None else []
        sep = [self.sep_token_id] if getattr(self, "sep_token_id", None) is not None else []
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep
    PreTrainedTokenizerBase.build_inputs_with_special_tokens = _build

DATA_ROOT      = os.environ.get("DATA_ROOT", "./data")
VIDEO_ROOT     = f"{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos"
TEST_JSON      = f"{DATA_ROOT}/UCFCrime_Test.json"
DEFAULT_MODEL  = "./output/smolvlm2-500m-desc-sft/checkpoint-700"
NUM_FRAMES     = 32
MAX_LENGTH     = 4096
MAX_DURATION   = 90.0
MAX_NEW_TOKENS = 512
SEED           = 99

DESC_PROMPT = (
    "Describe all activities in this surveillance video. "
    "List each activity on a new line, numbered from 1."
)


def extract_frames(video_path, start, end, n_frames):
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
            t = float(frame.pts * stream.time_base)
            if t > t_end + 1.0:
                break
            slot = int((t - t_start) / (t_end - t_start + 1e-9) * n_frames)
            slot = max(0, min(slot, n_frames - 1))
            if slot not in collected:
                collected[slot] = frame.to_image()
            if len(collected) >= n_frames:
                break
        container.close()
        if collected:
            for i in range(n_frames):
                if i not in collected:
                    collected[i] = collected[min(collected.keys(), key=lambda k: abs(k - i))]
            return [collected[i] for i in range(n_frames)]
    except Exception as e:
        print(f"  [WARN] frame extraction failed: {e}")
    return [Image.new("RGB", (224, 224))] * n_frames


def make_metadata(start, end, n_frames):
    frame_ts = [start + i * (end - start) / max(n_frames - 1, 1) for i in range(n_frames)]
    return VideoMetadata(
        total_num_frames=max(int(end * 10), n_frames),
        fps=10.0,
        frames_indices=[round(t * 10) for t in frame_ts],
        duration=float(end),
    )


def run_inference(model, processor, device, frames, start, end):
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": DESC_PROMPT}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    md = make_metadata(start, end, len(frames))
    inputs = processor(
        text=[text], videos=[[frames]], video_metadata=[md],
        return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
    ).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new = out[:, inputs["input_ids"].shape[1]:]
    return processor.decode(new[0], skip_special_tokens=True).strip()


def parse_desc_output(text: str) -> list[str]:
    """Parse numbered list: '1. sentence\\n2. sentence\\n...'"""
    sentences = []
    for line in text.strip().split("\n"):
        line = line.strip()
        # strip leading number+dot or bullet
        m = re.match(r"^\d+\.\s*(.+)", line)
        if m:
            sentences.append(m.group(1).strip())
        elif line and not re.match(r"^\d+\.?\s*$", line):
            sentences.append(line)
    return sentences


_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(pred, ref):
    return _rouge.score(ref, pred)["rougeL"].fmeasure


def bleu4(pred, ref):
    return sacrebleu.corpus_bleu([pred], [[ref]], smooth_method="exp").score / 100.0


def load_test_videos(n):
    with open(TEST_JSON) as f:
        data = json.load(f)
    items = []
    for vid, ann in data.items():
        cat  = re.sub(r"\d+_x264$", "", vid)
        path = os.path.join(VIDEO_ROOT, cat, f"{vid}.mp4")
        if not os.path.isfile(path):
            for root_dir, _, files in os.walk(VIDEO_ROOT):
                if f"{vid}.mp4" in files:
                    path = os.path.join(root_dir, f"{vid}.mp4")
                    break
        if not os.path.isfile(path):
            continue
        duration = float(ann.get("duration", MAX_DURATION))
        eff_end  = min(duration, MAX_DURATION)
        gts = []
        for (s, e), sent in zip(ann["timestamps"], ann["sentences"]):
            s, e = float(s), float(e)
            if e <= s or s > eff_end:
                continue
            gts.append(sent.strip())
        if gts:
            items.append({
                "video_id":      vid,
                "video_path":    path,
                "effective_end": eff_end,
                "gts":           gts,
            })
    random.seed(SEED)
    random.shuffle(items)
    return items if n == -1 else items[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=50)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out",   default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device} | Model: {args.model} | Samples: {args.n}")

    processor = AutoProcessor.from_pretrained(args.model)
    model     = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M\n")

    videos    = load_test_videos(args.n)
    total_gt  = sum(len(v["gts"]) for v in videos)
    print(f"Loaded {len(videos)} videos ({total_gt} GT annotations)\n")

    all_rouge, all_bleu = [], []
    all_pred_descs, all_ref_descs = [], []
    results = []

    for i, v in enumerate(videos, 1):
        eff_end   = v["effective_end"]
        frames    = extract_frames(v["video_path"], 0.0, eff_end, NUM_FRAMES)
        pred_text = run_inference(model, processor, device, frames, 0.0, eff_end)
        pred_sents = parse_desc_output(pred_text)
        gts        = v["gts"]

        print(f"[{i:>3}/{len(videos)}] {v['video_id']} ({len(gts)} GT, {len(pred_sents)} pred)")

        # Greedy match: pair predictions to GT by position
        n_pairs = min(len(pred_sents), len(gts))
        for j in range(n_pairs):
            rl = rouge_l(pred_sents[j], gts[j])
            b4 = bleu4(pred_sents[j], gts[j])
            all_rouge.append(rl)
            all_bleu.append(b4)
            all_pred_descs.append(pred_sents[j])
            all_ref_descs.append(gts[j])

        # Unmatched GTs → zero
        for j in range(n_pairs, len(gts)):
            all_rouge.append(0.0)
            all_bleu.append(0.0)
            all_pred_descs.append("")
            all_ref_descs.append(gts[j])

        results.append({
            "video_id":   v["video_id"],
            "pred_text":  pred_text,
            "pred_sents": pred_sents,
            "gts":        gts,
        })

    n_items = len(all_rouge)
    if n_items == 0:
        print("No results")
        return

    print("\nComputing BERTScore...")
    _, _, bert_f1 = bert_score_fn(
        all_pred_descs, all_ref_descs,
        lang="en", model_type="roberta-large", verbose=False,
    )
    all_bert = bert_f1.tolist()

    mean_rouge = sum(all_rouge) / n_items
    mean_bleu  = sum(all_bleu)  / n_items
    mean_bert  = sum(all_bert)  / n_items

    print(f"\n{'='*60}")
    print(f"  Samples (GTs)    : {n_items}")
    print(f"  Mean ROUGE-L     : {mean_rouge:.4f}")
    print(f"  Mean BLEU-4      : {mean_bleu:.4f}")
    print(f"  Mean BERTScore F1: {mean_bert:.4f}")
    print(f"{'='*60}")

    out_path = args.out or f"eval_desc_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    summary = {
        "model":              args.model,
        "n_videos":           len(videos),
        "n_gts":              n_items,
        "mean_rouge_l":       mean_rouge,
        "mean_bleu4":         mean_bleu,
        "mean_bertscore_f1":  mean_bert,
        "per_sample":         results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
