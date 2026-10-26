"""Transformer on shape/average/position features from the new mesh only."""

import torch
import torch.nn.functional as F
from torch import nn

from config import _COORD_DIM
from src.utils.fuse_mesh_component import NUM_JOINTS
from src.models.gcn import ModelOutput
from src.models.positional_encoding import PositionalEncoding
from src.training.train import masked_mean_pool


class ISLR_Transformer_Mesh(nn.Module):
    """Input (B, T, 3 * num_nodes * 2), ordered shape/average/position.

    Nodes contain retained pose and two hands of [64 mesh, 21 MediaPipe].
    There is no separate old landmark input or GCN.
    """

    def __init__(self, d_model=256,
                 num_encoder_layers=3, nhead=8, dim_feedforward=2048,
                 dropout=0.1, max_seq_len=5000, num_classes=1000,
                 num_nodes=NUM_JOINTS):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = num_nodes * _COORD_DIM
        self.max_seq_len = max_seq_len
        def make_projection(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, d_model), nn.GELU(),
                nn.Dropout(dropout), nn.LayerNorm(d_model),
            )

        self.shape_projection = make_projection(self.feature_dim)
        self.average_projection = make_projection(self.feature_dim)
        self.position_projection = make_projection(self.feature_dim)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes),
        )

    def encode(self, features, video_mask=None):
        if features.ndim != 3 or features.shape[-1] != 3 * self.feature_dim:
            raise ValueError(f"features must have shape (B, T, {3 * self.feature_dim})")
        batch, frames = features.shape[:2]
        if not 0 < frames <= self.max_seq_len:
            raise ValueError("Frame count must be positive and within max_seq_len")
        if video_mask is None:
            video_mask = torch.ones(batch, frames, device=features.device, dtype=torch.bool)
        else:
            video_mask = video_mask.to(device=features.device, dtype=torch.bool)
        if video_mask.shape != (batch, frames) or not video_mask.any(dim=1).all():
            raise ValueError("video_mask needs shape (B, T) and valid frames in every sample")
        valid = video_mask.unsqueeze(-1)
        features = features.masked_fill(~valid, 0)
        shape, average, position = features.split(self.feature_dim, dim=-1)
        fused = (self.shape_projection(shape) + self.average_projection(average)
                 + self.position_projection(position)).masked_fill(~valid, 0)

        encoded = self.encoder(self.pos_encoder(fused), src_key_padding_mask=~video_mask)

        return self.encoder_norm(encoded).masked_fill(~valid, 0), video_mask

    def forward(self, features, labels=None, video_mask=None):
        encoded, video_mask = self.encode(features, video_mask)
        pooled = masked_mean_pool(encoded, video_mask)
        logits = self.classifier(pooled)
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return ModelOutput(logits=logits, loss=loss, encoded=encoded)
