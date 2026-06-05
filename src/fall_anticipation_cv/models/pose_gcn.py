"""Graph Convolutional Network over COCO-17 skeleton keypoints.

Architecture:
  [B, T, J*C]  →  reshape  →  [B*T, J, C]
              →  GCNLayer × 2
              →  mean-pool joints  →  [B*T, gcn_out]
              →  reshape  →  [B, T, gcn_out]
              →  TemporalTransformerClassifier  →  [B, num_classes]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# COCO-17 skeleton edges (0-indexed)
COCO17_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                    # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]
NUM_JOINTS = 17


def build_normalized_adjacency(num_joints: int = NUM_JOINTS) -> torch.Tensor:
    """D^{-1/2} (A + I) D^{-1/2} symmetric normalization."""
    A = torch.eye(num_joints)
    for i, j in COCO17_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    deg = A.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[deg == 0] = 0.0
    D = torch.diag(d_inv_sqrt)
    return D @ A @ D


class GCNLayer(nn.Module):
    """Single graph convolutional layer with learnable edge weights."""

    def __init__(self, in_features: int, out_features: int, num_joints: int = NUM_JOINTS):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.edge_weight = nn.Parameter(torch.ones(num_joints, num_joints))
        self.bn = nn.BatchNorm1d(out_features)
        A_norm = build_normalized_adjacency(num_joints)
        self.register_buffer("A_norm", A_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, J, in_features]
        A = self.A_norm * torch.sigmoid(self.edge_weight)   # soft masking
        x = torch.bmm(A.unsqueeze(0).expand(x.size(0), -1, -1), x)
        x = self.linear(x)                                   # [N, J, out_features]
        N, J, C = x.shape
        x = self.bn(x.reshape(N * J, C)).reshape(N, J, C)
        return F.relu(x)


class PoseGCNTransformer(nn.Module):
    """Spatial GCN + temporal Transformer for pose-based fall anticipation."""

    def __init__(
        self,
        input_dim: int,
        num_joints: int = NUM_JOINTS,
        gcn_hidden: int = 64,
        gcn_out: int = 128,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.joint_features = input_dim // num_joints  # channels per joint

        self.gcn1 = GCNLayer(self.joint_features, gcn_hidden, num_joints)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_out, num_joints)

        from fall_anticipation_cv.models.temporal_transformer import TemporalTransformerClassifier

        self.temporal_classifier = TemporalTransformerClassifier(
            input_dim=gcn_out,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, J*C]
        B, T, _ = x.shape
        x = x.reshape(B * T, self.num_joints, self.joint_features)
        x = self.gcn1(x)
        x = self.gcn2(x)
        x = x.mean(dim=1)              # [B*T, gcn_out]
        x = x.reshape(B, T, -1)        # [B, T, gcn_out]
        return self.temporal_classifier(x, lengths)
