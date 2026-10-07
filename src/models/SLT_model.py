import torch
import torch.nn.functional as F
from torch import nn

from config import _COORD_DIM, _N_HAND, _N_POSE, _NUM_NODE
from src.models.positional_encoding import PositionalEncoding
from training.train import masked_mean_pool


class ISLR_Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_encoder_layers=3,
        nhead=8,
        dim_feedforward=128 * 8,
        dropout=0.1,
        max_seq_len=5000,
        num_classes=1000,
    ):
        super().__init__()

        d_model = hidden_dim
        self.num_nodes = _NUM_NODE

        self.pose_projection = nn.Sequential(
            nn.Linear(self.num_nodes * _COORD_DIM, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        self.pos_encoder = PositionalEncoding(
            d_model=d_model, max_len=max_seq_len, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.encoder_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def encode(self, features, video_mask):
        if video_mask is None:
            raise ValueError("video_mask is required")


        video_mask = video_mask.bool()

        x = self.pose_projection(features)

        x = self.pos_encoder(x)
        x = self.encoder(
            x,
            src_key_padding_mask=~video_mask,  # True = ignore (padding)
        )
        x = self.encoder_norm(x)

        return x

    def forward(self, features, labels=None, video_mask=None):

        x = self.encode(features, video_mask)
        pooled = masked_mean_pool(x, video_mask.bool())
        logits = self.classifier(pooled)  # (B, num_classes)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss

class ISLR_Transformer_Selector(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_encoder_layers=6,
        nhead=8,
        dim_feedforward=256 * 8,
        dropout=0.2,
        max_seq_len=5000,
        num_classes=1000,
    ):
        super().__init__()

        d_model = hidden_dim
        self.num_nodes = _NUM_NODE

        self.pose_projection = nn.Sequential(
            nn.Linear(_N_POSE * _COORD_DIM, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        self.hand_projection = nn.Sequential(
            nn.Linear(_N_HAND * _COORD_DIM * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        self.pos_encoder = PositionalEncoding(
            d_model=d_model, max_len=max_seq_len, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(d_model, num_classes)

        self.frame_importance_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def encode(self, features, video_mask):
        if video_mask is None:
            raise ValueError("video_mask is required")

        video_mask = video_mask.bool()

        pose_features = features[:, :, : _N_POSE * _COORD_DIM]
        hand_features = features[:, :, _N_POSE * _COORD_DIM :]

        x_pose = self.pose_projection(pose_features)
        x_hand = self.hand_projection(hand_features)
        x = x_pose + x_hand

        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=~video_mask)
        x = self.encoder_norm(x)

        return x

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")

        video_mask = video_mask.bool()

        x = self.encode(features, video_mask)

        frame_scores = self.frame_importance_head(x).squeeze(-1)
        frame_scores = frame_scores.masked_fill(~video_mask, -1e9)

        frame_importance = torch.softmax(frame_scores, dim=1)

        weights = frame_importance.unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)


        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
            "frame_importance": frame_importance,
            "encoded": x,
        }