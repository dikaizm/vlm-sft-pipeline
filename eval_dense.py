"""
Evaluation of dense SFT model on UCF-Crime test set.
Uses the dense prompt format and parses numbered-list output.
Metrics: tIoU, ROUGE-L, BLEU-4, BERTScore.
"""
import json, os, re, random, sys, torch, argparse
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.video_utils import VideoMetadata
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bert_score_fn

DATA_ROOT = './data'
VIDEO_ROOT = f'{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos'
TEST_JSON = f'{DATA_ROOT}/UCFCrime_Test.json'
DEFAULT_MODEL = './output/smolvlm2-500m-dense-sft'
NUM_FRAMES = 16
MAX_DURATION = 90.0
MAX_NEW_TOKENS = 256
SEED = 99

DENSE_PROMPT = (
    'Describe ALL activities in this surveillance video. '
    'For each activity, provide a description and its start and end timestamps in seconds. '
    'List them in chronological order.'
)

def extract_frames(video_path, start, end, n_frames):
    try:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else end
        t_start = max(0.0, min(start, duration))
        t_end = max(t_start + 0.1, min(end, duration))
        collected = {}
        container.seek(int(t_start * 1_000_000), any_frame=False, backward=True)
        for frame in container.decode(video=0):
            t = float(frame.pts * stream.time_base)
            if t > t_end + 1.0: break
            slot = int((t - t_start) / (t_end - t_start + 1e-9) * n_frames)
            slot = max(0, min(slot, n_frames - 1))
            if slot not in collected: collected[slot] = frame.to_image()
            if len(collected) >= n_frames: break
        container.close()
        if collected:
            for i in range(n_frames):
                if i not in collected:
                    collected[i] = collected[min(collected.keys(), key=lambda k: abs(k-i))]
            return [collected[i] for i in range(n_frames)]
    except Exception as e:
        print(f'  [WARN] frame extraction failed: {e}')
    return [Image.new('RGB', (224, 224))] * n_frames

def make_metadata(start, end, n_frames):
    return VideoMetadata(
        total_num_frames=int(max(end, n_frames)), fps=1.0,
        frames_indices=[int(start + i*(end-start)/max(n_frames-1,1)) for i in range(n_frames)],
        duration=float(end))

def run_inference(model, processor, device, frames, start, end):
    msgs = [{'role':'user','content':[{'type':'video'},{'type':'text','text':DENSE_PROMPT}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    md = make_metadata(start, end, len(frames))
    inputs = processor(text=[text], videos=[[frames]], video_metadata=[md],
                       return_tensors='pt', truncation=False, max_length=None).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new = out[:, inputs['input_ids'].shape[1]:]
    return processor.tokenizer.decode(new[0], skip_special_tokens=True).strip()

def parse_dense_output(text):
    """Parse numbered list: '1. [s, e] desc\\n2. [s, e] desc'"""
    entries = []
    for line in text.strip().split('\n'):
        m = re.match(r'\s*\d+\.?\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]\s*(.+)', line)
        if m:
            entries.append({'start': float(m.group(1)), 'end': float(m.group(2)), 'desc': m.group(3).strip()})
    return entries

def tiou(pred_s, pred_e, gt_s, gt_e):
    inter = max(0.0, min(pred_e, gt_e) - max(pred_s, gt_s))
    union = (pred_e - pred_s) + (gt_e - gt_s) - inter
    return inter / union if union > 0 else 0.0

_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def rouge_l(pred, ref):
    return _rouge.score(ref, pred)['rougeL'].fmeasure

def bleu4(pred, ref):
    return sacrebleu.corpus_bleu([pred], [[ref]], smooth_method='exp').score / 100.0

def load_test_videos(n):
    with open(TEST_JSON) as f:
        data = json.load(f)
    items = []
    for vid, ann in data.items():
        cat = re.sub(r'\d+_x264$', '', vid)
        path = os.path.join(VIDEO_ROOT, cat, f'{vid}.mp4')
        if not os.path.isfile(path):
            # fallback search
            for root_dir, _, files in os.walk(VIDEO_ROOT):
                if f'{vid}.mp4' in files:
                    path = os.path.join(root_dir, f'{vid}.mp4')
                    break
        if not os.path.isfile(path):
            continue
        duration = float(ann.get('duration', MAX_DURATION))
        eff_end = min(duration, MAX_DURATION)
        gts = []
        for (s, e), sent in zip(ann['timestamps'], ann['sentences']):
            s, e = float(s), float(e)
            if e <= s or s > eff_end: continue
            gts.append({'start': s, 'end': min(e, eff_end), 'desc': sent.strip()})
        if gts:
            items.append({'video_id': vid, 'video_path': path, 'duration': duration, 'effective_end': eff_end, 'gts': gts})
    random.seed(SEED)
    random.shuffle(items)
    return items if n == -1 else items[:n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    print(f'Device: {device} | Model: {args.model} | Samples: {args.n}')

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M\n')

    videos = load_test_videos(args.n)
    total_gt = sum(len(v['gts']) for v in videos)
    print(f'Loaded {len(videos)} videos ({total_gt} GT annotations)\n')

    all_tiou, all_rouge, all_bleu = [], [], []
    all_pred_descs, all_ref_descs = [], []
    results = []
    ts_found = 0

    for i, v in enumerate(videos, 1):
        eff_end = v['effective_end']
        frames = extract_frames(v['video_path'], 0.0, eff_end, NUM_FRAMES)
        pred_text = run_inference(model, processor, device, frames, 0.0, eff_end)
        entries = parse_dense_output(pred_text)
        gts = v['gts']

        print(f'[{i:>3}/{len(videos)}] {v["video_id"]} ({len(gts)} GT, {len(entries)} pred)')

        # Match predictions to ground truth by temporal proximity
        matched_gt = set()
        for e_idx, entry in enumerate(entries):
            best_tiou_val, best_gt_idx = 0.0, -1
            for g_idx, gt in enumerate(gts):
                if g_idx in matched_gt: continue
                val = tiou(entry['start'], entry['end'], gt['start'], gt['end'])
                if val > best_tiou_val:
                    best_tiou_val, best_gt_idx = val, g_idx
            if best_gt_idx >= 0 and best_tiou_val > 0:
                ts_found += 1
                matched_gt.add(best_gt_idx)
                gt = gts[best_gt_idx]
                rl = rouge_l(entry['desc'], gt['desc'])
                b4 = bleu4(entry['desc'], gt['desc'])
                all_tiou.append(best_tiou_val)
                all_rouge.append(rl)
                all_bleu.append(b4)
                all_pred_descs.append(entry['desc'])
                all_ref_descs.append(gt['desc'])
                results.append({**entry, 'tiou': best_tiou_val, 'rouge_l': rl, 'bleu4': b4,
                                'gt_start': gt['start'], 'gt_end': gt['end'], 'gt_desc': gt['desc'],
                                'video_id': v['video_id']})

        # Unmatched GTs get zero scores
        for g_idx, gt in enumerate(gts):
            if g_idx not in matched_gt:
                all_tiou.append(0.0)
                all_rouge.append(0.0)
                all_bleu.append(0.0)
                all_pred_descs.append('')
                all_ref_descs.append(gt['desc'])

    n_items = len(all_tiou)
    if n_items == 0:
        print('No results'); return

    print('\nComputing BERTScore...')
    _, _, bert_f1 = bert_score_fn(all_pred_descs, all_ref_descs, lang='en', model_type='roberta-large', verbose=False)
    all_bert = bert_f1.tolist()

    mean_tiou = sum(all_tiou) / n_items
    mean_rouge = sum(all_rouge) / n_items
    mean_bleu = sum(all_bleu) / n_items
    mean_bert = sum(all_bert) / n_items

    print(f'\n{"="*60}')
    print(f'  Samples (GTs)       : {n_items}')
    print(f'  Timestamp matched   : {ts_found}')
    print(f'  Mean tIoU           : {mean_tiou:.4f}')
    print(f'  tIoU@0.3            : {sum(1 for v in all_tiou if v>=0.3)/n_items:.4f}')
    print(f'  tIoU@0.5            : {sum(1 for v in all_tiou if v>=0.5)/n_items:.4f}')
    print(f'  Mean ROUGE-L        : {mean_rouge:.4f}')
    print(f'  Mean BLEU-4         : {mean_bleu:.4f}')
    print(f'  Mean BERTScore F1   : {mean_bert:.4f}')
    print(f'{"="*60}')

    out_path = args.out or f'eval_dense_{datetime.now().strftime("%Y%m%d-%H%M%S")}.json'
    summary = {'model': args.model, 'n_videos': len(videos), 'n_gts': n_items,
               'timestamp_matched': ts_found, 'mean_tiou': mean_tiou,
               'tiou_at_0.3': sum(1 for v in all_tiou if v>=0.3)/n_items,
               'tiou_at_0.5': sum(1 for v in all_tiou if v>=0.5)/n_items,
               'mean_rouge_l': mean_rouge, 'mean_bleu4': mean_bleu,
               'mean_bertscore_f1': mean_bert, 'per_sample': results}
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved: {out_path}')

if __name__ == '__main__':
    main()
