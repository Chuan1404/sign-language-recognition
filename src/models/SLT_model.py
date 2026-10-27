import torch
import torch.nn.functional as F
from torch import nn

from config import _COORD_DIM, _N_HAND, _N_POSE, _NUM_NODE
from src.models.positional_encoding import PositionalEncoding
from training.train import masked_mean_pool


class ISLR_Transformer(nn.Module):
    """Sum shape/average/position/velocity projections before one shared temporal encoder.

    Input: (B, T, 3 * num_nodes * _COORD_DIM + 2 * num_nodes).
    Ordered shape/average/position/velocity; velocity has signed normalized XY components per node per frame.
    video_mask: (B, T), True for valid frames. Returns (logits, loss).
    """

    def __init__(self, hidden_dim=256, num_encoder_layers=3, nhead=8,
                 dim_feedforward=512 * 4, dropout=0.1, max_seq_len=200,
                 num_classes=1000, num_nodes=_NUM_NODE):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = num_nodes * _COORD_DIM

        def make_projection(input_dim=self.feature_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(),
                nn.Dropout(dropout), nn.LayerNorm(hidden_dim),
            )

        self.shape_projection = make_projection()
        self.average_projection = make_projection()
        self.position_projection = make_projection()
        self.velocity_projection = make_projection(num_nodes * 2)

        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, features, video_mask):
        video_mask = video_mask.to(device=features.device, dtype=torch.bool)
        features = features.masked_fill(~video_mask.unsqueeze(-1), 0)
        shape, average, position, velocity = features.split(
            [self.feature_dim] * 3 + [self.num_nodes * 2], dim=-1)
        x = (self.shape_projection(shape)
             + self.average_projection(average)
             + self.position_projection(position)
             + self.velocity_projection(velocity)
             )

        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=~video_mask)
        return self.encoder_norm(x).masked_fill(~video_mask.unsqueeze(-1), 0)

    def forward(self, features, labels=None, video_mask=None):
        video_mask = video_mask.to(device=features.device, dtype=torch.bool)
        encoded = self.encode(features, video_mask)
        pooled = masked_mean_pool(encoded, video_mask)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return logits, loss


