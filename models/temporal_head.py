"""
Temporal action detection head for SmolVLM2 dense captioning.
Trains a lightweight head on frozen vision encoder features to predict [start, end].
Keeps existing 500M model for activity descriptions.
"""
import json, os, re, random, torch, argparse, numpy as np
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.video_utils import VideoMetadata

DATA_ROOT = './data'
VIDEO_ROOT = f'{DATA_ROOT}/UCF_Crimes/UCF_Crimes/Videos'
TRAIN_JSON = f'{DATA_ROOT}/UCFCrime_Train.json'
VAL_JSON = f'{DATA_ROOT}/UCFCrime_Val.json'
TEST_JSON = f'{DATA_ROOT}/UCFCrime_Test.json'

ENCODER_ID = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
NUM_FRAMES = 16
MAX_DURATION = 90.0
FRAME_SIZE = 384
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
IOU_WEIGHT = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ActionDetectionHead(nn.Module):
    """Per-frame actionness + temporal boundary regression."""
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.actionness = nn.Conv1d(hidden_dim, 1, 1)  # per-frame action probability
        self.boundary = nn.Conv1d(hidden_dim, 2, 1)     # [start_offset, end_offset]

    def forward(self, features):
        # features: [B, T, D] → [B, D, T]
        x = features.permute(0, 2, 1)
        x = self.conv(x)
        actionness = self.actionness(x).squeeze(1)       # [B, T]
        boundary = self.boundary(x).permute(0, 2, 1)      # [B, T, 2]
        return torch.sigmoid(actionness), boundary


def extract_frames(video_path, n_frames, eff_end):
    try:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else eff_end
        t_end = max(0.1, min(eff_end, duration))
        collected = {}
        container.seek(0, any_frame=False, backward=True)
        for frame in container.decode(video=0):
            t = float(frame.pts * stream.time_base)
            if t > t_end + 1.0: break
            slot = int(t / (t_end + 1e-9) * n_frames)
            slot = max(0, min(slot, n_frames - 1))
            if slot not in collected:
                collected[slot] = frame.to_image().resize((FRAME_SIZE, FRAME_SIZE))
            if len(collected) >= n_frames: break
        container.close()
        if collected:
            for i in range(n_frames):
                if i not in collected:
                    collected[i] = collected[min(collected.keys(), key=lambda k: abs(k-i))]
            return [collected[i] for i in range(n_frames)]
    except:
        pass
    return [Image.new('RGB', (FRAME_SIZE, FRAME_SIZE))] * n_frames


def load_video_data(json_path, max_videos=-1):
    """Load videos with GT annotations (same as train_dense.py)."""
    with open(json_path) as f:
        data = json.load(f)

    mp4_map = {}
    for root_dir, _, files in os.walk(VIDEO_ROOT):
        for fname in files:
            if fname.endswith('.mp4'):
                mp4_map[fname] = os.path.join(root_dir, fname)

    samples = []
    for video_id, ann in data.items():
        category = re.sub(r'\d+_x264$', '', video_id)
        video_path = os.path.join(VIDEO_ROOT, category, f'{video_id}.mp4')
        if not os.path.isfile(video_path):
            fallback = mp4_map.get(f'{video_id}.mp4')
            if fallback: video_path = fallback
            else: continue

        duration = float(ann.get('duration', MAX_DURATION))
        eff_end = min(duration, MAX_DURATION)

        pairs = []
        for (start, end), sentence in zip(ann['timestamps'], ann['sentences']):
            start, end = float(start), float(end)
            if end <= start or start > eff_end: continue
            pairs.append((start, min(end, eff_end), sentence.strip()))

        if pairs:
            samples.append({
                'video_id': video_id, 'video_path': video_path,
                'duration': duration, 'effective_end': eff_end,
                'annotations': [(s, e, d) for s, e, d in pairs],
            })

    random.seed(42)
    random.shuffle(samples)
    if max_videos > 0: samples = samples[:max_videos]
    return samples


class TemporalDataset(Dataset):
    """Each sample = one video with its frame features + GT segments."""
    def __init__(self, samples, feature_extractor):
        self.samples = samples
        self.extractor = feature_extractor

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        eff_end = s['effective_end']
        frames = extract_frames(s['video_path'], NUM_FRAMES, eff_end)

        with torch.no_grad():
            features = self.extractor(frames)  # [NUM_FRAMES, D]

        # Create actionness labels (1 where action, 0 otherwise)
        actionness = torch.zeros(NUM_FRAMES)
        for start, end, _ in s['annotations']:
            start_bin = int(start / eff_end * NUM_FRAMES)
            end_bin = int(end / eff_end * NUM_FRAMES)
            start_bin = max(0, min(start_bin, NUM_FRAMES - 1))
            end_bin = max(0, min(end_bin, NUM_FRAMES - 1))
            actionness[start_bin:end_bin+1] = 1.0

        return {
            'features': features.cpu(),
            'actionness': actionness,
            'effective_end': eff_end,
            'annotations': s['annotations'],
            'video_id': s['video_id'],
        }


class FeatureExtractor:
    """Extract frame features using frozen SmolVLM2 vision encoder."""
    def __init__(self, encoder_id, device):
        self.device = device
        # Load just the vision model
        import transformers
        config = transformers.AutoConfig.from_pretrained(encoder_id)
        self.model = AutoModel.from_pretrained(
            encoder_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.processor = AutoProcessor.from_pretrained(encoder_id)

    def __call__(self, frames):
        # Process frames through vision encoder
        # Simpler approach: use the processor to encode images, get vision features
        # We use the vision_model directly
        try:
            # Convert frames to tensors via processor
            pixel_values = self.processor.image_processor(
                frames, return_tensors='pt'
            ).pixel_values.to(self.device, dtype=torch.bfloat16)

            with torch.no_grad():
                outputs = self.model.vision_model(pixel_values)
            # outputs.last_hidden_state: [1, N_patches, D]
            # Mean pool over patch dimension to get per-frame features
            features = outputs.last_hidden_state.mean(dim=1)  # [1, D]
            return features.squeeze(0).float()
        except Exception as e:
            print(f'Feature extraction error: {e}')
            return torch.zeros(1152).float()  # fallback


def iou_loss(pred_starts, pred_ends, gt_starts, gt_ends):
    """1 - tIoU as loss."""
    inter = torch.clamp(torch.min(pred_ends, gt_ends) - torch.max(pred_starts, gt_starts), min=0)
    union = (pred_ends - pred_starts) + (gt_ends - gt_starts) - inter + 1e-8
    return 1.0 - (inter / union).mean()


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc='train'):
        features = batch['features'].to(DEVICE)       # [B, T, D]
        actionness_gt = batch['actionness'].to(DEVICE) # [B, T]

        actionness_pred, _ = model(features)

        # BCE loss for actionness
        bce = criterion(actionness_pred, actionness_gt)

        optimizer.zero_grad()
        bce.backward()
        optimizer.step()
        total_loss += bce.item()

    return total_loss / len(dataloader)


def train():
    print(f'Loading feature extractor from {ENCODER_ID}...')
    extractor = FeatureExtractor(ENCODER_ID, DEVICE)

    print('Loading datasets...')
    train_samples = load_video_data(TRAIN_JSON)
    val_samples = load_video_data(VAL_JSON)
    print(f'Train: {len(train_samples)} videos, Val: {len(val_samples)} videos')

    train_ds = TemporalDataset(train_samples, extractor)
    val_ds = TemporalDataset(val_samples, extractor)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # Get feature dimension from first sample
    sample = train_ds[0]
    feature_dim = sample['features'].shape[-1]
    print(f'Feature dim: {feature_dim}')

    model = ActionDetectionHead(feature_dim).to(DEVICE)
    print(f'Temporal head params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_dl, optimizer, criterion)
        val_loss = validate(model, val_dl, criterion)
        scheduler.step()

        print(f'Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './output/temporal_head_best.pt')

    print(f'Best val loss: {best_val_loss:.4f}')
    print('Model saved to ./output/temporal_head_best.pt')


def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(DEVICE)
            actionness_gt = batch['actionness'].to(DEVICE)
            actionness_pred, _ = model(features)
            loss = criterion(actionness_pred, actionness_gt)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def collate_batch(batches):
    features = torch.stack([b['features'] for b in batches])
    actionness = torch.stack([b['actionness'] for b in batches])
    return {
        'features': features,
        'actionness': actionness,
        'effective_end': [b['effective_end'] for b in batches],
        'annotations': [b['annotations'] for b in batches],
        'video_id': [b['video_id'] for b in batches],
    }


if __name__ == '__main__':
    train()
