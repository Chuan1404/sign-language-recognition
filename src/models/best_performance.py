import torch
import torch.nn as nn
from config import _NUM_NODE, _COORD_DIM
from models import build_direction_adjacency_from_edges, HAND_EDGES, GCNBlock, PositionalEncoding, ModelOutput
import torch.nn.functional as F

class ISLR_Transformer_GCN(nn.Module):
    def __init__(self, gcn_channels=(64, 64, 128, 128), d_model=128, num_encoder_layers=3, nhead=8,
                 dim_feedforward=128 * 8, dropout=0.1, max_seq_len=5000, num_classes=1000, ):
        super().__init__()
        self.num_nodes = _NUM_NODE

        self.A_in, self.A_out = build_direction_adjacency_from_edges(HAND_EDGES, num_nodes=self.num_nodes, )
        self.register_buffer("adjacency_matrix", self.A_in)
        channels = [_COORD_DIM, *gcn_channels]

        self.feature_dim = self.num_nodes * _COORD_DIM

        def make_projection():
            return nn.Sequential(nn.Linear(self.feature_dim, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.LayerNorm(d_model), )

        self.shape_projection = make_projection()
        self.average_projection = make_projection()
        self.position_projection = make_projection()

        self.gcn_block = nn.ModuleList(
            [GCNBlock(channels[i], channels[i + 1], self.num_nodes, self.adjacency_matrix) for i in
             range(len(channels) - 1)])

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True, norm_first=True, )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes), )

        self.frame_importance_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def encode_gcn(self, features, video_mask):

        for block in self.gcn_block:
            features = block(features, video_mask)

        features = features.mean(dim=-2)  # (B, T, gcn_out)
        # return self.gcn_proj(x)  # (B, T, d_model)
        return features

    def encode_transformer(self, features, video_mask, x_rgb=None):
        video_mask = video_mask.to(
            device=features.device,
            dtype=torch.bool,
        )

        B, T, D = features.shape

        shape_feature, average_feature, position_feature = torch.split(features, self.feature_dim, dim=-1, )

        x_shape = self.shape_projection(shape_feature)
        x_average = self.average_projection(average_feature)
        x_position = self.position_projection(position_feature)

        # Giữ nhánh GCN hiện tại, đầu vào là shape feature.
        x_gcn = self.encode_gcn(shape_feature.reshape(B, T, self.num_nodes, _COORD_DIM), video_mask, )

        x = x_shape + x_average + x_position + x_gcn

        # x = x_average                                     # best=73.00% loss=1.2133 75.00%
        # x = x_average + x_gcn                             # best=66.00% loss=1.4254 95.00%

        # x = x_position                                    # best=72.00% loss=1.2574 75.00%
        # x = x_position + x_gcn                            # best=70.00% loss=1.3490 71.00%

        # x = x_shape                                       # best=66.00% loss=1.2873 67.00%
        # x = x_shape + x_gcn                               # best=59.00% loss=1.3645 67.00%

        # x = x_shape + x_average                           # best=73.00% loss=1.1507 75.00%
        # x = x_shape + x_average + x_gcn                   # best=71.00% loss=1.1686 74.00

        # x = x_shape + x_average + x_position  # best=75.00% loss=1.0840 77.00%
        # x = x_shape + x_average + x_position + x_gcn      # best=75.00% loss=1.1256 78.00%
        # x = x_gcn                                           # best=62.00% loss=1.4346 66.00%

        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=~video_mask, )

        return self.encoder_norm(x)

    def forward(self, features, labels=None, video_mask=None, x_rgb=None):

        if video_mask is None:
            video_mask = torch.ones(
                features.shape[:2], dtype=torch.bool, device=features.device
            )
        else:
            video_mask = video_mask.to(device=features.device, dtype=torch.bool)
        if not video_mask.any(dim=1).all():
            raise ValueError("Each video must have at least one valid frame")

        t = self.encode_transformer(features, video_mask, x_rgb=x_rgb)
        important_score = self.frame_importance_head(t).squeeze(-1)
        important_score = important_score.masked_fill(~video_mask, float("-inf"))
        frame_importance = torch.softmax(important_score, dim=1)

        # Weighted pooling lets classification loss train the importance head.
        # pooled = (t * frame_importance.unsqueeze(-1)).sum(dim=1)
        t = t.sum(dim=1)
        logits = self.classifier(t)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ModelOutput(
            logits=logits, loss=loss,
            important_score=important_score,
            frame_importance=frame_importance,
            encoded=t,
        )
