"""Dynamic graph attention over modality nodes."""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class DynamicGraphAttention(nn.Module):
    """Residual content-based self-attention over modality nodes."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25,
        init_graph_alpha: float = 0.06,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

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

        # Small initialization keeps the module close to the no-graph path.
        self.graph_alpha_logit = nn.Parameter(torch.tensor(_logit(init_graph_alpha), dtype=torch.float32))
        self.last_attention = None
        self.last_graph_alpha = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, m, d = x.shape
        return x.view(b, m, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,M,hd]

    def forward(self, nodes: torch.Tensor):
        q = self._split_heads(self.q_proj(nodes))
        k = self._split_heads(self.k_proj(nodes))
        v = self._split_heads(self.v_proj(nodes))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,M,M]

        attn_raw = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn_raw)

        context = torch.matmul(attn, v)  # [B,H,M,hd]
        context = context.transpose(1, 2).contiguous().view(nodes.size(0), nodes.size(1), self.embed_dim)
        update = self.out_proj(context)

        alpha = torch.sigmoid(self.graph_alpha_logit).to(dtype=nodes.dtype, device=nodes.device)
        h = self.norm1(nodes + alpha * self.dropout(update))
        ffn_update = self.ffn(h)
        h = self.norm2(h + alpha * self.dropout(ffn_update))

        attn_mean = attn_raw.mean(dim=1)  # [B,M,M], not dropout-distorted
        self.last_attention = attn_mean.detach()
        self.last_graph_alpha = alpha.detach()
        return h, attn_mean


class FixedGraphConvolution(nn.Module):
    """Residual graph convolution over modality nodes with a fixed adjacency matrix."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_nodes: int = 3,
        dropout: float = 0.25,
        init_graph_alpha: float = 0.06,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes

        adjacency = torch.ones(num_nodes, num_nodes, dtype=torch.float32)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("adjacency", adjacency)

        self.node_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.graph_alpha_logit = nn.Parameter(torch.tensor(_logit(init_graph_alpha), dtype=torch.float32))
        self.last_attention = None
        self.last_graph_alpha = None

    def forward(self, nodes: torch.Tensor):
        adjacency = self.adjacency.to(dtype=nodes.dtype, device=nodes.device)
        context = torch.matmul(adjacency.unsqueeze(0), nodes)
        update = self.node_proj(context)

        alpha = torch.sigmoid(self.graph_alpha_logit).to(dtype=nodes.dtype, device=nodes.device)
        h = self.norm1(nodes + alpha * self.dropout(update))
        ffn_update = self.ffn(h)
        h = self.norm2(h + alpha * self.dropout(ffn_update))

        attention = adjacency.unsqueeze(0).expand(nodes.size(0), -1, -1)
        self.last_attention = attention.detach()
        self.last_graph_alpha = alpha.detach()
        return h, attention
