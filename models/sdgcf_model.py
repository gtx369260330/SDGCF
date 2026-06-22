"""SDGCF model for three-channel EEG/EOG sleep staging.

SDGCF means Simple Dynamic Graph Concatenation Fusion. Each EEG/EOG channel
becomes one modality node. Content-based graph attention updates the three
nodes, which are concatenated directly for classification.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .encoders import MultiScaleTemporalEncoder
from .graph_fusion import DynamicGraphAttention, FixedGraphConvolution


class SDGCFNet(nn.Module):
    """Three-node EEG/EOG dynamic graph model with direct concat fusion."""

    NUM_MODALITY_NODES = 3

    def __init__(
        self,
        input_channels: int = NUM_MODALITY_NODES,
        num_classes: int = 5,
        embed_dim: int = 128,
        encoder_hidden: int = 48,
        kernel_sizes=(3, 7, 15),
        dropout: float = 0.25,
        graph_heads: int = 2,
        use_graph: bool = True,
        graph_type: str = "dynamic",
        graph_alpha_init: float = 0.08,
        use_auxiliary_heads: bool = True,
    ) -> None:
        super().__init__()
        if input_channels != self.NUM_MODALITY_NODES:
            raise ValueError(
                f"SDGCF expects exactly {self.NUM_MODALITY_NODES} EEG/EOG channels, "
                f"got {input_channels}"
            )

        self.input_channels = input_channels
        self.use_graph = bool(use_graph)
        self.graph_type = str(graph_type).lower()
        self.modality_encoders = nn.ModuleList(
            [
                MultiScaleTemporalEncoder(
                    in_channels=1,
                    hidden_dim=encoder_hidden,
                    embed_dim=embed_dim,
                    kernel_sizes=kernel_sizes,
                    dropout=dropout,
                )
                for _ in range(input_channels)
            ]
        )
        if self.use_graph and self.graph_type == "dynamic":
            self.graph_attention = DynamicGraphAttention(
                embed_dim=embed_dim,
                num_heads=graph_heads,
                dropout=dropout,
                init_graph_alpha=graph_alpha_init,
            )
        elif self.use_graph and self.graph_type == "fixed":
            self.graph_attention = FixedGraphConvolution(
                embed_dim=embed_dim,
                num_nodes=input_channels,
                dropout=dropout,
                init_graph_alpha=graph_alpha_init,
            )
        elif self.use_graph:
            raise ValueError(f"Unknown graph_type: {graph_type}")
        else:
            self.graph_attention = None
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * input_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        self.auxiliary_heads = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(embed_dim, max(32, embed_dim // 2)),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(max(32, embed_dim // 2), num_classes),
                    )
                    for _ in range(input_channels)
                ]
            )
            if use_auxiliary_heads
            else None
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, Any]:
        if x.dim() != 3:
            raise ValueError(f"Expected x [B,C,T], got {x.shape}")
        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {x.size(1)}")

        modality_nodes = torch.stack(
            [
                encoder(x[:, index : index + 1, :])
                for index, encoder in enumerate(self.modality_encoders)
            ],
            dim=1,
        )
        if self.graph_attention is not None:
            graph_nodes, attention = self.graph_attention(modality_nodes)
        else:
            graph_nodes = modality_nodes
            attention = None

        fused = graph_nodes.reshape(x.size(0), -1)
        out: Dict[str, Any] = {
            "logits": self.classifier(fused),
            "attention": attention,
            "fused": fused,
            "modality_nodes": modality_nodes,
            "graph_nodes": graph_nodes,
            "regularize_graph": self.graph_attention is not None,
        }
        if self.graph_attention is not None:
            out["graph_alpha"] = self.graph_attention.last_graph_alpha
        if self.auxiliary_heads is not None:
            out["aux_logits"] = torch.stack(
                [
                    head(modality_nodes[:, index, :])
                    for index, head in enumerate(self.auxiliary_heads)
                ],
                dim=1,
            )
        return out
