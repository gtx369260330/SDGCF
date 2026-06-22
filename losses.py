"""Loss functions for sleep-stage classification and SDGCF regularization."""
from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSleepStageLoss(nn.Module):
    """Blend weighted label-smoothed CE with weighted focal loss."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.03,
        focal_gamma: float = 1.5,
        focal_blend: float = 0.25,
    ) -> None:
        super().__init__()
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.detach().clone())
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.focal_blend = float(focal_blend)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()
        target_log_probability = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probability = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.class_weights is None:
            sample_weights = torch.ones_like(target_probability)
        else:
            sample_weights = self.class_weights[targets]
        focal = (
            sample_weights
            * (1.0 - target_probability).pow(self.focal_gamma)
            * -target_log_probability
        ).sum() / sample_weights.sum().clamp_min(1e-6)
        return (1.0 - self.focal_blend) * ce + self.focal_blend * focal


def _attention_entropy_floor_loss(attention: torch.Tensor, floor_ratio: float) -> torch.Tensor:
    entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(dim=-1)
    floor = float(floor_ratio) * math.log(max(2, attention.size(-1)))
    return F.relu(floor - entropy).square().mean()


def _node_oversmoothing_loss(graph_nodes: torch.Tensor, margin: float) -> torch.Tensor:
    normalized = F.normalize(graph_nodes, p=2, dim=-1)
    similarity = torch.matmul(normalized, normalized.transpose(1, 2))
    modality_count = similarity.size(1)
    off_diagonal = 1.0 - torch.eye(
        modality_count,
        device=similarity.device,
        dtype=similarity.dtype,
    ).unsqueeze(0)
    penalty = F.relu(similarity - float(margin)).square() * off_diagonal
    return penalty.sum() / off_diagonal.sum().clamp_min(1.0) / max(1, similarity.size(0))


def compute_total_loss(
    out: Dict[str, Any],
    targets: torch.Tensor,
    criterion: nn.Module,
    *,
    auxiliary_weight: float = 0.0,
    attention_entropy_weight: float = 0.0,
    attention_entropy_floor: float = 0.0,
    node_diversity_weight: float = 0.0,
    node_diversity_margin: float = 0.90,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return classification loss plus lightweight graph regularizers."""
    parts: Dict[str, torch.Tensor] = {"main": criterion(out["logits"], targets)}
    total = parts["main"]

    aux_logits = out.get("aux_logits")
    if aux_logits is not None and auxiliary_weight > 0:
        auxiliary = torch.stack(
            [criterion(aux_logits[:, index, :], targets) for index in range(aux_logits.size(1))]
        ).mean()
        parts["auxiliary"] = auxiliary
        total = total + float(auxiliary_weight) * auxiliary

    regularize_graph = bool(out.get("regularize_graph", False))
    attention = out.get("attention")
    if regularize_graph and attention is not None and attention_entropy_weight > 0:
        entropy_floor = _attention_entropy_floor_loss(attention, attention_entropy_floor)
        parts["attention_entropy_floor"] = entropy_floor
        total = total + float(attention_entropy_weight) * entropy_floor

    graph_nodes = out.get("graph_nodes")
    if regularize_graph and graph_nodes is not None and node_diversity_weight > 0:
        node_oversmoothing = _node_oversmoothing_loss(graph_nodes, node_diversity_margin)
        parts["node_oversmoothing"] = node_oversmoothing
        total = total + float(node_diversity_weight) * node_oversmoothing

    parts["total"] = total
    return total, parts
