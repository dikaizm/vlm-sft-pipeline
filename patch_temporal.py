"""
Patches SmolVLM2 model to insert temporal transformer between vision encoder and LLM.
"""
import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, hidden_dim, num_frames=16, num_layers=2, num_heads=8):
        super().__init__()
        self.num_frames = num_frames
        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, patch_features):
        B, total, D = patch_features.shape
        f = self.num_frames

        if total % f == 0:
            p = total // f
            x = patch_features.view(B, f, p, D).mean(dim=2)
        else:
            x = patch_features[:, :f * (total//f)]
            x = x.view(B, f, -1, D).mean(dim=2)

        x = x + self.temporal_pos[:, :f, :]
        x = self.transformer(x)
        return x


def patch_model_with_temporal(model, num_frames=16):
    vm = model.model.vision_model
    hidden_dim = vm.config.hidden_size if hasattr(vm.config, 'hidden_size') else 1152

    temp = TemporalTransformer(hidden_dim, num_frames=num_frames)
    orig_forward = vm.forward

    def temporal_vision_forward(pixel_values, **kwargs):
        outputs = orig_forward(pixel_values, **kwargs)
        features = outputs.last_hidden_state

        # Per-frame temporal processing
        temporal_out = temp(features)  # [B, num_frames, D]

        # Repeat back to per-patch granularity
        B, f, D = temporal_out.shape
        patches_per_frame = features.shape[1] // f if features.shape[1] % f == 0 else 1

        if patches_per_frame > 0 and features.shape[1] % f == 0:
            expanded = temporal_out.unsqueeze(2).expand(B, f, patches_per_frame, D)
            expanded = expanded.reshape(B, f * patches_per_frame, D)
            features = expanded
        else:
            # Just use per-frame features as-is and let downstream handle mismatch
            features = temporal_out

        outputs.last_hidden_state = features
        return outputs

    vm.forward = temporal_vision_forward
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    temp = temp.to(device).to(dtype)
    model.add_module("temporal_transformer", temp)

    n_params = sum(p.numel() for p in model.temporal_transformer.parameters())
    print(f'TemporalTransformer: {n_params:,} params')
    return model
