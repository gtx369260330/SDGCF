"""Modality-specific temporal encoders."""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleTemporalEncoder(nn.Module):
    """Encode one modality signal [B, 1, T] into embedding [B, D]."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 48,
        embed_dim: int = 128,
        kernel_sizes=(3, 7, 15),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                ConvBlock1D(in_channels, hidden_dim, k, dropout=dropout),
                nn.MaxPool1d(kernel_size=4, stride=4),
                ConvBlock1D(hidden_dim, hidden_dim, k, dropout=dropout),
                nn.AdaptiveAvgPool1d(1),
            )
            for k in kernel_sizes
        ])
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * len(kernel_sizes), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for branch in self.branches:
            z = branch(x).squeeze(-1)  # [B, hidden]
            feats.append(z)
        z = torch.cat(feats, dim=-1)
        return self.proj(z)
