"""Frozen snapshot of ISLR_Transformer_VCN and its dependencies (2026-11-08).

Only PyTorch and the Python standard library are required. Configuration,
graph edges and helper implementations are copied here so later edits to
config.py, gcn.py or spatial_graph.py do not alter this saved model.
Current behavior: random finger masking in train, node mean pooling,
decouple_p=1, use_tcn=False. No trained weights are stored in this file.
"""


import math

import torch
import torch.nn.functional as F
from torch import nn


_REMOVE_POSE_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
_N_POSE = 3
_N_HAND = 21
_NUM_NODE = 45
_COORD_DIM = 2


FULL_BODY_EDGES = [(1, 2),
 (0, 1),
 (0, 5),
 (0, 9),
 (0, 13),
 (0, 17),
 (1, 2),
 (2, 3),
 (3, 4),
 (5, 6),
 (6, 7),
 (7, 8),
 (9, 10),
 (10, 11),
 (11, 12),
 (13, 14),
 (14, 15),
 (15, 16),
 (17, 18),
 (18, 19),
 (19, 20),
 (5, 9),
 (9, 13),
 (13, 17),
 (21, 22),
 (21, 26),
 (21, 30),
 (21, 34),
 (21, 38),
 (22, 23),
 (23, 24),
 (24, 25),
 (26, 27),
 (27, 28),
 (28, 29),
 (30, 31),
 (31, 32),
 (32, 33),
 (34, 35),
 (35, 36),
 (36, 37),
 (38, 39),
 (39, 40),
 (40, 41),
 (26, 30),
 (30, 34),
 (34, 38)]


# Training defaults from main.py; model constructor defaults are kept below.
TRAINING_DEFAULTS = {'model_name': 'landmarks_100_finger_delta.pt',
 'lr': 0.0001,
 'batch_size': 8,
 'epochs': 100,
 'top_k': 5,
 'patience': 10}


def build_adjacency_from_edges(edges, num_nodes):
    A = torch.eye(num_nodes)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def masked_mean_pool(x, video_mask):
    mask = video_mask.unsqueeze(-1).float()  # (B, T, 1)
    summed = (x * mask).sum(dim=1)  # (B, D)
    counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1) — avoid /0
    return summed / counts


class ModelOutput(dict):

    def __init__(self, logits=None, loss=None, **kwargs):
        super().__init__(logits=logits, loss=loss, **kwargs)
        self.__dict__ = self

    def __iter__(self):
        yield self["logits"]
        yield self["loss"]

    def __getitem__(self, key):
        return super().__getitem__(key)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, T, D)

        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        x = x + self.pe[:, :T, :].to(x.device)
        return self.dropout(x)


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=9, stride=1, dropout=0.1):
        super().__init__()

        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(out_channels=channels, in_channels=channels, kernel_size=(kernel_size, 1),
                              # (out_channels, in_channels, kernel_H, kernel_W)
                              padding=(pad, 0), stride=(stride, 1), bias=False, )
        self.bn = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, N, C) → (B, C, T, N)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.drop(x)
        x = x.permute(0, 2, 3, 1)  # (B, T, N, C)
        return x


class RandomFingerMask(nn.Module):
    """Mask one finger per hand and sample, consistently across all frames."""

    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes

    def forward(self, features):
        if not self.training:
            return features
        batch = features.shape[0]
        # Layout: retained pose joints, left hand, right hand. Wrist index is 0.
        hand_offsets = torch.tensor(
            [self.num_nodes - 2 * _N_HAND, self.num_nodes - _N_HAND],
            device=features.device,
        )
        fingers = torch.randint(5, (batch, 2), device=features.device)
        joints = (hand_offsets[None, :, None] + 1 + 4 * fingers[:, :, None]
                  + torch.arange(4, device=features.device))
        keep = torch.ones(batch, self.num_nodes, dtype=torch.bool, device=features.device)
        keep.scatter_(1, joints.reshape(batch, 8), False)
        return features.masked_fill(~keep[:, None, :, None], 0)


class V_GCN(nn.Module):
    """Independent graph convolution with input-dependent joint attention."""

    def __init__(self, in_channels, out_channels, num_nodes, base_adjacency, decouple_p=1):
        super().__init__()

        self.V = num_nodes
        self.p = decouple_p
        self.phi = nn.Linear(in_channels, out_channels)

        base_adjacency = base_adjacency.float()
        self.register_buffer("I", torch.eye(self.V))
        self.A_in = nn.Parameter(base_adjacency.unsqueeze(0).repeat(self.p, 1, 1) * 1e-3)

    def _normalized_A(self, A_raw):
        deg = A_raw.sum(-1).clamp(min=1e-6)
        d_inv_sqrt = deg.pow(-0.5)
        D_inv_sqrt = torch.diag_embed(d_inv_sqrt)

        return D_inv_sqrt @ A_raw @ D_inv_sqrt

    def forward(self, feature):
        feat = self.phi(feature)

        scores = torch.matmul(feat, feat.transpose(-1, -2)) * feat.size(-1) ** -0.5
        A_shape = (scores.unsqueeze(2) + self.A_in).softmax(dim=-1)
        A_raw = self.I + A_shape

        # A_raw = self.I.unsqueeze(0) + self.A_in
        A_norm = self._normalized_A(A_raw)
        out = torch.einsum("btpvw,btwc->btpvc", A_norm, feat).mean(dim=2)
        return out, A_raw


class V_GCN_Block(nn.Module):
    """Shape-only counterpart of SelfPacingDroppingBlock."""

    def __init__(self, in_ch, out_ch, num_nodes, base_adjacency,
                 decouple_p=4, drop=True, use_tcn=True,
                 temporal_kernel_size=9, dropout=0.1):
        super().__init__()
        self.gcn = V_GCN(in_ch, out_ch, num_nodes, base_adjacency,
                         decouple_p)
        self.use_tcn = use_tcn
        self.tcn = TemporalConv(out_ch, kernel_size=temporal_kernel_size if use_tcn else 1,
                                dropout=dropout)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, shape_feature, mask=None):
        valid = (torch.ones(shape_feature.shape[:2], device=shape_feature.device, dtype=torch.bool)
                 if mask is None else mask.to(device=shape_feature.device, dtype=torch.bool))
        valid = valid[:, :, None, None]

        shape_feature = shape_feature.masked_fill(~valid, 0)

        feat, _ = self.gcn(shape_feature)
        feat = feat.masked_fill(~valid, 0)

        if self.use_tcn:
            feat = self.act(self.tcn(feat))

        if not self.use_tcn:
            feat = self.act(feat)

        return self.dropout(feat).masked_fill(~valid, 0)


class ISLR_Transformer_VCN(nn.Module):
    def __init__(self, gcn_channels=(128, 128, 256, 256), d_model=256, num_encoder_layers=3, nhead=8,
                 dim_feedforward=256 * 8, dropout=0.1, max_seq_len=300, num_classes=1000,
                 num_nodes=_NUM_NODE):

        super().__init__()
        self.num_nodes = num_nodes
        self.finger_mask = RandomFingerMask(num_nodes)

        self.A_in = build_adjacency_from_edges(FULL_BODY_EDGES, num_nodes=self.num_nodes, )
        self.register_buffer("adjacency_matrix", self.A_in)
        channels = [_COORD_DIM, *gcn_channels]

        self.feature_dim = num_nodes * _COORD_DIM

        def make_projection(input_dim=self.feature_dim):
            return nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.LayerNorm(d_model), )

        self.shape_projection = make_projection()
        self.average_projection = make_projection()
        self.position_projection = make_projection()

        self.gcn_block = nn.ModuleList(
            [V_GCN_Block(channels[i], channels[i + 1], self.num_nodes, self.adjacency_matrix, decouple_p=1, use_tcn=False, dropout=dropout) for i in
             range(len(channels) - 1)])

        self.node_attention = nn.Linear(channels[-1], 1)
        self.gcn_output_projection = nn.Linear(channels[-1], d_model)

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True, norm_first=True, )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        fusion_dim = d_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Dropout(dropout), nn.Linear(fusion_dim, num_classes),
        )

        # self.frame_importance_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
        #     nn.Linear(d_model // 2, 1), )

    def encode_gcn(self, features, video_mask):
        features = self.finger_mask(features)

        for block in self.gcn_block:
            features = block(features, video_mask)

        # Learn node importance per frame; normalize over the node dimension.
        # weights = self.node_attention(features).softmax(dim=-2)
        # features = (features * weights).sum(dim=-2)
        features = features.mean(dim=-2)
        return features

    def encode_transformer(self, features, video_mask, x_rgb=None):
        video_mask = video_mask.to(device=features.device, dtype=torch.bool, )

        features = features.masked_fill(~video_mask.unsqueeze(-1), 0)
        B, T, D = features.shape

        shape_feature, average_feature, position_feature = features.split(self.feature_dim, dim=-1)
        x_shape = self.shape_projection(shape_feature)
        x_average = self.average_projection(average_feature)
        x_position = self.position_projection(position_feature)

        # x = x_shape + x_average + x_position

        # The GCN branch takes the position features.
        x_gcn = self.encode_gcn(position_feature.reshape(B, T, self.num_nodes, _COORD_DIM), video_mask, )
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

        video_mask = video_mask.to(device=features.device, dtype=torch.bool)
        features = features.masked_fill(~video_mask.unsqueeze(-1), 0)
        B, T, D = features.shape

        shape_feature, average_feature, position_feature = features.split(self.feature_dim, dim=-1)

        t = self.encode_transformer(features, video_mask, x_rgb=x_rgb)
        # x_gcn = self.encode_gcn(
        #     shape_feature.reshape(B, T, -1, _COORD_DIM),
        #     video_mask,
        # )
        # x_gcn = masked_mean_pool(x_gcn, video_mask)


        t = masked_mean_pool(t, video_mask)
        # t = t + x_gcn
        # t = torch.cat([t, x_gcn], dim=-1)
        logits = self.classifier(t)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ModelOutput(logits=logits,
                           loss=loss,
                            encoded=t,
                           )


# Compatibility name for callers that explicitly select the saved model.
ISLR_Transformer_VCN_BEST = ISLR_Transformer_VCN
