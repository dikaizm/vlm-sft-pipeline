import json, os, re, torch, argparse, pickle
from PIL import Image
from tqdm import tqdm
import av

DATA_ROOT = './data'
VIDEO_ROOT = f'{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos'
NUM_FRAMES = 16
MAX_DURATION = 90.0
FRAME_SIZE = 384

def extract_frames(video_path, eff_end):
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else eff_end
        t_end = max(0.1, min(eff_end, duration))
        collected = {}
        container.seek(0, any_frame=False, backward=True)
        for frame in container.decode(video=0):
            t = float(frame.pts * stream.time_base)
            if t > t_end + 1.0: break
            slot = int(t / (t_end + 1e-9) * NUM_FRAMES)
            slot = max(0, min(slot, NUM_FRAMES - 1))
            if slot not in collected:
                collected[slot] = frame.to_image().resize((FRAME_SIZE, FRAME_SIZE))
            if len(collected) >= NUM_FRAMES: break
        container.close()
        if collected:
            for i in range(NUM_FRAMES):
                if i not in collected:
                    collected[i] = collected[min(collected.keys(), key=lambda k: abs(k-i))]
            return [collected[i] for i in range(NUM_FRAMES)]
    except:
        pass
    return [Image.new('RGB', (FRAME_SIZE, FRAME_SIZE))] * NUM_FRAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=-1)
    parser.add_argument('--split', choices=['train','val','test'], default='train')
    parser.add_argument('--out', default='./output/features_train.pkl')
    args = parser.parse_args()

    json_map = {'train': './data/UCFCrime_Train.json', 'val': './data/UCFCrime_Val.json', 'test': './data/UCFCrime_Test.json'}
    json_path = json_map[args.split]

    with open(json_path) as f:
        data = json.load(f)

    # Build file map for fallback
    mp4_map = {}
    for root_dir, _, files in os.walk(VIDEO_ROOT):
        for fname in files:
            if fname.endswith('.mp4'):
                mp4_map[fname] = os.path.join(root_dir, fname)

    results = {}
    items = list(data.items())
    if args.max > 0:
        items = items[:args.max]

    for video_id, ann in tqdm(items, desc=f'Extracting {args.split}'):
        cat = re.sub(r'\d+_x264$', '', video_id)
        path = os.path.join(VIDEO_ROOT, cat, f'{video_id}.mp4')
        if not os.path.isfile(path):
            path = mp4_map.get(f'{video_id}.mp4', None)
        if not path or not os.path.isfile(path):
            continue

        duration = float(ann.get('duration', MAX_DURATION))
        eff_end = min(duration, MAX_DURATION)

        frames = extract_frames(path, eff_end)

        # Store frames and annotations
        gts = []
        for (s, e), sent in zip(ann['timestamps'], ann['sentences']):
            s, e = float(s), float(e)
            if e <= s or s > eff_end: continue
            gts.append({'start': s, 'end': min(e, eff_end), 'desc': sent.strip()})

        if gts:
            results[video_id] = {
                'frames': frames,
                'duration': duration,
                'effective_end': eff_end,
                'annotations': gts,
                'path': path,
            }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved {len(results)} videos to {args.out}')

if __name__ == '__main__':
    main()
