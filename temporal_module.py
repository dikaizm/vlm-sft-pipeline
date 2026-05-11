import torch
from torch import nn
import math

class TemporalTransformer(nn.Module):
    """Cross-frame temporal attention + learnable positional embeddings.
    Takes per-frame patch tokens from SigLIP and fuses them with temporal awareness.
    """
    def __init__(self, hidden_dim=1152, num_layers=2, num_heads=8, max_frames=16):
        super().__init__()
        self.max_frames = max_frames
        self.hidden_dim = hidden_dim
        
        # Learnable temporal position embeddings (one per frame slot)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, max_frames, 1, hidden_dim) * 0.02)
        
        # Cross-frame attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Learnable frame importance (for event boundary detection)
        self.frame_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Learnable [EVENT_START] and [EVENT_END] tokens
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.end_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
    def forward(self, patch_features, num_frames_per_video=None):
        """
        Args:
            patch_features: [B, total_patches, D] from SigLIP vision encoder
                            Each frame produces N patches, total_patches = num_frames * N
        Returns:
            frame_features: [B, num_frames, D] temporally-aware per-frame features
            start_logits: [B, num_frames] event start probabilities
            end_logits: [B, num_frames] event end probabilities
        """
        B = patch_features.shape[0]
        
        # Mean-pool patches per frame → [B, num_frames, D]
        patches_per_frame = patch_features.shape[1] // self.max_frames
        if patches_per_frame > 1 and patch_features.shape[1] % self.max_frames == 0:
            frame_features = patch_features.view(B, self.max_frames, patches_per_frame, -1).mean(dim=2)
        else:
            # Fallback: reshape best effort
            frame_features = patch_features[:, :self.max_frames*patches_per_frame]
            frame_features = frame_features.view(B, self.max_frames, patches_per_frame, -1).mean(dim=2)
        
        # Add temporal position embeddings
        frame_features = frame_features + self.temporal_pos_embed[:, :frame_features.shape[1]]
        
        # Cross-frame transformer attention
        frame_features = self.transformer(frame_features)  # [B, num_frames, D]
        
        # Frame importance scores (for boundary detection)
        frame_scores = self.frame_importance(frame_features).squeeze(-1)  # [B, num_frames]
        
        return frame_features, frame_scores


class SmolVLM2WithTemporal(nn.Module):
    """Wraps SmolVLM2 with temporal module inserted between vision encoder and LLM."""
    def __init__(self, base_model, hidden_dim=1152, num_frames=16):
        super().__init__()
        self.base = base_model
        self.num_frames = num_frames
        
        # Find vision encoder hidden dim
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'vision_model'):
            vm = base_model.model.vision_model
        elif hasattr(base_model, 'vision_model'):
            vm = base_model.vision_model
        else:
            vm = None
        
        if vm is not None and hasattr(vm.config, 'hidden_size'):
            hidden_dim = vm.config.hidden_size
        
        self.temporal = TemporalTransformer(hidden_dim=hidden_dim, max_frames=num_frames)
        self.hidden_dim = hidden_dim
        
    def forward(self, pixel_values, input_ids, attention_mask, labels=None, **kwargs):
        """Replaces the standard forward to insert temporal features."""
        # We hook into the base model's vision processing
        # The base model internally handles vision encoding + projection
        # We modify the flow by adding temporal features before the LLM
        
        # For simplicity, we use a two-step approach:
        # 1. Get vision features from the encoder
        # 2. Pass through temporal module
        # 3. Feed into LLM with text
        
        # This requires accessing internal model components
        # For now, return base model's forward and we'll hook properly
        return self.base(pixel_values=pixel_values, input_ids=input_ids, 
                        attention_mask=attention_mask, labels=labels, **kwargs)
