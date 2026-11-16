import torch
import torch.nn.functional as F
from torch import nn

from config import _COORD_DIM, _NUM_NODE
from src.models.positional_encoding import PositionalEncoding
from src.training.train import masked_mean_pool


class ISLR_Transformer(nn.Module):
    def __init__(self, hidden_dim=512, num_encoder_layers=3, nhead=8, dim_feedforward=512 * 4, dropout=0.1,
                 max_seq_len=300, num_classes=1000, num_nodes=_NUM_NODE):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = num_nodes * _COORD_DIM

        def make_projection(input_dim=self.feature_dim):
            return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim), )

        self.shape_projection = make_projection()
        self.average_projection = make_projection()
        self.position_projection = make_projection()

        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)

        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True, )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, features, video_mask):
        video_mask = video_mask.to(device=features.device, dtype=torch.bool)
        features = features.masked_fill(~video_mask.unsqueeze(-1), 0)

        shape, average, position = features.split(self.feature_dim, dim=-1)
        x = (self.shape_projection(shape) + self.average_projection(average) + self.position_projection(position))

        x = self.pos_encoder(x)
        x = self._encode_temporal(x, video_mask)
        return self.encoder_norm(x)

    def _encode_temporal(self, x, video_mask):
        return self.encoder(x, src_key_padding_mask=~video_mask)

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            video_mask = torch.ones(features.shape[:2], device=features.device, dtype=torch.bool)
        else:
            video_mask = video_mask.to(device=features.device, dtype=torch.bool)

        encoded = self.encode(features, video_mask)
        pooled = masked_mean_pool(encoded, video_mask)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return logits, loss
