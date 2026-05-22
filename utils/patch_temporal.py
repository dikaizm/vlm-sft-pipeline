"""
Post-connector temporal bridge for SmolVLM2.

Inserts a sliding window Q-Former + BiLSTM AFTER the pixel-shuffle connector
and BEFORE the LLM. Enriches visual tokens with cross-frame temporal context.
Timestamps are handled by Vid2Seq-style special time tokens in the LLM vocabulary
— no separate regression head needed.

Architecture:
    connector output  [B, F*T, D]
          ↓
    PostConnectorTemporalBridge
      sliding window Q-Former (window=16 frames, stride=4)
      BiLSTM over windows → frame-level context
      residual add + LayerNorm
          ↓
    [B, F*T, D]  enriched visual tokens → LLM
"""

import torch
import torch.nn as nn


class PostConnectorTemporalBridge(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_frames: int = 32,
        n_queries: int = 8,
        n_heads: int = 8,
        window_size: int = 16,
        stride: int = 4,
        lstm_hidden: int = 256,
    ):
        super().__init__()
        self.num_frames  = num_frames
        self.window_size = window_size
        self.stride      = stride

        self.queries      = nn.Parameter(torch.randn(1, n_queries, hidden_dim) * 0.02)
        self.qformer_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=0.0, batch_first=True)
        self.qformer_norm = nn.LayerNorm(hidden_dim)

        self.bilstm    = nn.LSTM(hidden_dim, lstm_hidden, num_layers=1,
                                 batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(lstm_hidden * 2, hidden_dim)
        self.out_norm  = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, total, D = x.shape
        F = self.num_frames

        if total % F != 0:
            return x
        T = total // F

        frames = x.view(B, F, T, D)

        # Sliding window Q-Former
        window_starts = list(range(0, F, self.stride))
        window_reprs  = []
        for ws in window_starts:
            we          = min(ws + self.window_size, F)
            win_tokens  = frames[:, ws:we].reshape(B, (we - ws) * T, D)
            q           = self.queries.expand(B, -1, -1)
            attn_out, _ = self.qformer_attn(q, win_tokens, win_tokens)
            q           = self.qformer_norm(q + attn_out)
            window_reprs.append(q.mean(dim=1))       # [B, D]

        # BiLSTM over windows
        window_seq  = torch.stack(window_reprs, dim=1)   # [B, n_windows, D]
        lstm_out, _ = self.bilstm(window_seq)
        lstm_out    = self.lstm_proj(lstm_out)            # [B, n_windows, D]

        # Map windows → frames (average overlapping windows)
        frame_context = torch.zeros(B, F, D, device=x.device, dtype=x.dtype)
        overlap       = torch.zeros(F,       device=x.device, dtype=x.dtype)
        for i, ws in enumerate(window_starts):
            we = min(ws + self.window_size, F)
            frame_context[:, ws:we] += lstm_out[:, i:i+1].expand(B, we - ws, D)
            overlap[ws:we] += 1.0
        frame_context = frame_context / overlap.clamp(min=1.0).view(1, F, 1)

        context = frame_context.unsqueeze(2).expand(B, F, T, D).reshape(B, total, D)
        return self.out_norm(x + context)


class TemporalConnectorWrapper(nn.Module):
    def __init__(self, connector: nn.Module, bridge: PostConnectorTemporalBridge):
        super().__init__()
        self.connector = connector
        self.bridge    = bridge

    def forward(self, x, **kwargs):
        return self.bridge(self.connector(x, **kwargs))


def patch_model_with_temporal(model, num_frames: int = 32) -> tuple:
    """
    Wraps model.model.connector with PostConnectorTemporalBridge.
    Returns (model, bridge).
    """
    lm = getattr(model.model, "text_model", None) or getattr(model.model, "language_model", None)
    hidden_dim = getattr(lm.config, "hidden_size", 2048) if lm is not None else 2048

    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    bridge = PostConnectorTemporalBridge(hidden_dim=hidden_dim, num_frames=num_frames)
    bridge = bridge.to(device).to(dtype)

    model.model.connector = TemporalConnectorWrapper(model.model.connector, bridge)

    n_params = sum(p.numel() for p in bridge.parameters())
    print(f"PostConnectorTemporalBridge: {n_params:,} params  "
          f"[dim={hidden_dim}, frames={num_frames}, window=16/stride=4, BiLSTM-256]")

    return model, bridge
