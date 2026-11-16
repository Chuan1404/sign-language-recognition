from src.models.spatial_graph import V_GCN_Block_V2
from src.models.spatial_graph import LOCAL_BODY_EDGES
from src.models.best_performance import RandomFingerMask
import torch
import torch.nn.functional as F
from torch import nn

from config import _COORD_DIM, _NUM_NODE, _N_POSE, _N_HAND
from src.models.positional_encoding import PositionalEncoding
from src.models.spatial_graph import GCNBlock, V_GCN_Block, build_direction_adjacency_from_edges, FULL_BODY_EDGES, \
    HAND_EDGES, build_adjacency_from_edges
from src.training.train import masked_mean_pool


class ModelOutput(dict):

    def __init__(self, logits=None, loss=None, **kwargs):
        super().__init__(logits=logits, loss=loss, **kwargs)
        self.__dict__ = self

    def __iter__(self):
        yield self["logits"]
        yield self["loss"]

    def __getitem__(self, key):
        return super().__getitem__(key)

class ISLR_Transformer_GCN(nn.Module):
    def __init__(self, gcn_channels=(128, 128, 256, 256), d_model=512, num_encoder_layers=3, nhead=8,
                 dim_feedforward=512 * 4, dropout=0.1, max_seq_len=300, num_classes=1000, num_nodes=_NUM_NODE):

        super().__init__()
        self.num_nodes = num_nodes
        self.num_hand_nodes = 2 * _N_HAND

        # self.register_buffer("adjacency_matrix", build_adjacency_from_edges(FULL_BODY_EDGES, self.num_nodes))
        # self.register_buffer("adjacency_matrix", build_adjacency_from_edges(LOCAL_BODY_EDGES, self.num_nodes))
        self.register_buffer("adjacency_matrix", build_adjacency_from_edges(HAND_EDGES, self.num_hand_nodes))
        channels = [_COORD_DIM, *gcn_channels]

        self.feature_dim = num_nodes * _COORD_DIM

        def make_projection(input_dim=self.feature_dim):
            return nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model), )

        self.shape_projection = make_projection()
        self.average_projection = make_projection()
        self.position_projection = make_projection()

        self.node_projection = make_projection(self.num_hand_nodes * channels[-1])


        self.gcn_block = nn.ModuleList([
            V_GCN_Block_V2(in_ch, out_ch, self.num_hand_nodes, self.adjacency_matrix, decouple_p=1,
                        use_tcn=True, dropout=dropout) for in_ch, out_ch in zip(channels, channels[1:])])

        # self.node_attention = nn.Linear(channels[-1], 1)

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)

        def make_encoder():
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=num_encoder_layers)

        self.encoder = make_encoder()
        self.encoder_norm = nn.LayerNorm(d_model)

        fusion_dim = d_model
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def encode_gcn(self, features, video_mask):
        for block in self.gcn_block:
            features = block(features, video_mask)

        features = self.node_projection(features.flatten(start_dim=-2))

        return features

    def encode_transformer(self, features, video_mask):
        video_mask = video_mask.to(device=features.device, dtype=torch.bool, )

        features = features.masked_fill(~video_mask.unsqueeze(-1), 0)
        B, T, D = features.shape

        shape_feature, average_feature, position_feature = features.split(self.feature_dim, dim=-1)
        x_shape = self.shape_projection(shape_feature)
        x_average = self.average_projection(average_feature)
        x_position = self.position_projection(position_feature)

        positions = position_feature.reshape(B, T, self.num_nodes, _COORD_DIM)
        hand_positions = positions[:, :, -self.num_hand_nodes:, :]

        x_gcn = self.encode_gcn(hand_positions, video_mask)
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
            video_mask = torch.ones(features.shape[:2], device=features.device, dtype=torch.bool)
        else:
            video_mask = video_mask.to(device=features.device, dtype=torch.bool)

        frames = self.encode_transformer(features, video_mask)
        out = masked_mean_pool(frames, video_mask)
        logits = self.classifier(out)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ModelOutput(logits=logits, loss=loss, )
