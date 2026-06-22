"""Baseline models for comparison and ablation.

The ConcatTransformer implementation intentionally avoids PyTorch's fused
scaled-dot-product attention kernels. On some older NVIDIA GPUs, the fused FMHA
kernel selected by ``nn.TransformerEncoderLayer`` can raise errors such as:
``kernel fmha_cutlassF_f16_aligned_64x64_rf_sm80 is for sm80-sm100``.
The safe transformer below uses explicit Q/K/V projections + torch.matmul, so it
runs on CPU and older CUDA devices without depending on FlashAttention/FMHA.
"""
from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn

from .encoders import MultiScaleTemporalEncoder


def _no_autocast_if_cuda(x: torch.Tensor):
    """Disable autocast inside attention when running on CUDA.

    This keeps the transformer attention path in float32 and prevents PyTorch from
    dispatching half-precision fused FMHA kernels that are not available on older
    GPUs. On CPU the context is a no-op.
    """
    if x.is_cuda:
        return torch.amp.autocast("cuda", enabled=False)
    return nullcontext()


class SingleModalityCNN(nn.Module):
    def __init__(
        self,
        modality_index: int = 0,
        num_classes: int = 5,
        embed_dim: int = 128,
        encoder_hidden: int = 48,
        kernel_sizes=(3, 7, 15),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.modality_index = modality_index
        self.encoder = MultiScaleTemporalEncoder(1, encoder_hidden, embed_dim, kernel_sizes, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x, return_features: bool = False):
        z = self.encoder(x[:, self.modality_index:self.modality_index + 1, :])
        logits = self.classifier(z)
        out = {"logits": logits, "fused": z, "attention": None}
        return out


class SimpleConcatCNN(nn.Module):
    """Simple baseline that treats all channels as ordinary CNN input."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 5,
        base_channels: int = 64,
        embed_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(base_channels * 2, embed_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x, return_features: bool = False):
        z = self.net(x).squeeze(-1)
        logits = self.classifier(z)
        out = {"logits": logits, "fused": z, "attention": None}
        return out


class SafeTransformerLayer(nn.Module):
    """Transformer encoder layer implemented without fused SDPA/FMHA kernels."""

    def __init__(self, embed_dim: int = 128, nhead: int = 4, dropout: float = 0.25):
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(f"embed_dim must be divisible by nhead, got {embed_dim=} and {nhead=}")
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.last_attention = None

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        b, m, d = t.shape
        return t.reshape(b, m, self.nhead, self.head_dim).transpose(1, 2)  # [B,H,M,Dh]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, M, D], where M is the number of modality tokens.
        with _no_autocast_if_cuda(x):
            residual = x.float()
            q = self._split_heads(self.q_proj(residual))
            k = self._split_heads(self.k_proj(residual))
            v = self._split_heads(self.v_proj(residual))
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().reshape(residual.size(0), residual.size(1), self.embed_dim)
            out = self.out_proj(out)
            x = self.norm1(residual + self.dropout(out))
            x = self.norm2(x + self.dropout(self.ffn(x)))
            self.last_attention = attn.mean(dim=1).detach()
            return x


class SafeTransformerEncoder(nn.Module):
    """Stack of SafeTransformerLayer objects."""

    def __init__(self, embed_dim: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.layers = nn.ModuleList([
            SafeTransformerLayer(embed_dim=embed_dim, nhead=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.last_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        if self.layers:
            self.last_attention = self.layers[-1].last_attention
        return x


class ConcatTransformer(nn.Module):
    """Encode each modality, then use a safe Transformer over modality tokens.

    This baseline keeps the original modeling idea but replaces
    ``nn.TransformerEncoder`` with a manual attention implementation to avoid
    FlashAttention/FMHA kernel dispatch problems on older GPUs.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 5,
        embed_dim: int = 128,
        encoder_hidden: int = 48,
        kernel_sizes=(3, 7, 15),
        dropout: float = 0.25,
        nhead: int = 4,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.encoders = nn.ModuleList([
            MultiScaleTemporalEncoder(1, encoder_hidden, embed_dim, kernel_sizes, dropout)
            for _ in range(input_channels)
        ])
        self.transformer = SafeTransformerEncoder(embed_dim=embed_dim, nhead=nhead, num_layers=num_layers, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x, return_features: bool = False):
        raw_nodes = []
        for i in range(self.input_channels):
            raw_nodes.append(self.encoders[i](x[:, i:i + 1, :]))
        raw_nodes = torch.stack(raw_nodes, dim=1)
        nodes = self.transformer(raw_nodes)
        z = nodes.mean(dim=1)
        logits = self.classifier(z)
        out = {"logits": logits, "fused": z, "attention": self.transformer.last_attention}
        return out
