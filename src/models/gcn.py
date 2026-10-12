import torch
import torch.nn.functional as F
from torch import nn

from config import _COORD_DIM, _NUM_NODE, _N_POSE, _N_HAND
from src.models.positional_encoding import PositionalEncoding
from src.models.spatial_graph import GCNBlock, SelfPacingDroppingBlock, build_direction_adjacency_from_edges, \
    FULL_BODY_EDGES, HAND_EDGES, GCN_Transformer, SPDStack
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


class SpatialGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_nodes, adjacency, p=4, dropout=0.1):
        super().__init__()
        self.p = p
        self.phi = nn.Linear(in_ch, out_ch)
        self.A = nn.Parameter(adjacency.clone().unsqueeze(0).repeat(p, 1, 1))  # (p, N, N)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def _normalize(self, A):
        deg = A.sum(dim=-1).clamp(min=1e-6)
        D = torch.diag_embed(deg.pow(-0.5))
        return D @ A @ D

    def forward(self, x):
        # x: (B, T, N, C_in) -> (B, T, N, C_out)
        B, T, N, _ = x.shape
        feat = self.phi(x)
        out = 0.0
        for k in range(self.p):
            A_norm = self._normalize(self.A[k])
            out = out + torch.einsum("nm,btmc->btnc", A_norm, feat)
        out = out / self.p
        out = self.bn(out.reshape(B * T * N, -1)).reshape(B, T, N, -1)
        return self.drop(self.act(out))


class ISLR_GCN(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), num_classes=2000):
        super().__init__()

        self.num_nodes = _NUM_NODE
        self.A_in, self.A_out = build_direction_adjacency_from_edges(HAND_EDGES, num_nodes=self.num_nodes, )
        self.register_buffer("adjacency_matrix", self.A_in)

        channels = [_COORD_DIM, *channels]
        gcn_out_dim = channels[-1]

        self.gcn_block = nn.ModuleList([GCNBlock(in_ch=channels[i], out_ch=channels[i + 1], num_nodes=self.num_nodes,
                                                 base_adjacency=self.adjacency_matrix, ) for i in
                                        range(len(channels) - 1)])

        self.classifier = nn.Linear(gcn_out_dim, num_classes)

    def forward(self, features, labels=None, video_mask=None, important_indices=None):
        if important_indices is not None:
            if isinstance(important_indices, (list, tuple)):
                important_indices = torch.as_tensor(important_indices, dtype=torch.long, device=features.device)
            if torch.is_tensor(important_indices) and important_indices.dim() == 1:
                features = features[:, important_indices]
                if video_mask is not None:
                    video_mask = video_mask[:, important_indices]

        if video_mask is None:
            video_mask = torch.ones((features.shape[0], features.shape[1]), dtype=torch.bool, device=features.device)
        else:
            video_mask = video_mask.bool()

        if features.dim() == 3:
            B, T, D = features.shape
            if D == self.num_nodes * _COORD_DIM:
                features = features.reshape(B, T, self.num_nodes, _COORD_DIM)
            elif D >= self.num_nodes * _COORD_DIM:
                features = features[:, :, : self.num_nodes * _COORD_DIM].reshape(B, T, self.num_nodes, _COORD_DIM)
            else:
                raise ValueError(
                    f"Feature dimension {D} is invalid for num_nodes={self.num_nodes}, coord_dim={_COORD_DIM}")
        elif features.dim() == 4:
            B, T, N, C = features.shape
            if N != self.num_nodes or C != _COORD_DIM:
                if N * C == self.num_nodes * _COORD_DIM:
                    features = features.reshape(B, T, self.num_nodes, _COORD_DIM)
                else:
                    features = features[:, :, : self.num_nodes, :_COORD_DIM]

        for block in self.gcn_block:
            features = block(features, video_mask)

        features = features.mean(dim=-2)

        features = masked_mean_pool(features, video_mask)

        logits = self.classifier(features)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ModelOutput(logits=logits, loss=loss)


class ISLR_Transformer_GCN(nn.Module):
    def __init__(self, gcn_channels=(128, 128, 256, 256), d_model=256, num_encoder_layers=6, nhead=8,
                 dim_feedforward=512 * 4, dropout=0.1, max_seq_len=5000, num_classes=1000, ):
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


class ISLR_Remove_Node_By_GCN(nn.Module):
    def __init__(self, channels=(64, 64, 128, 128), num_classes=2000):
        super().__init__()

        self.num_nodes = _NUM_NODE
        self.num_classes = num_classes
        self.A_in, self.A_out = build_direction_adjacency_from_edges(FULL_BODY_EDGES, num_nodes=self.num_nodes)

        channels = [_COORD_DIM, *channels]
        gcn_out_dim = channels[-1]

        self.gcn_block = nn.ModuleList(
            [GCNBlock(in_ch=channels[i], out_ch=channels[i + 1], num_nodes=self.num_nodes, base_adjacency=self.A_in, )
             for i in range(len(channels) - 1)])

        self.classifier = nn.Linear(gcn_out_dim, num_classes)

    def forward(self, features, labels=None, video_mask=None):
        B, T, _ = features.shape

        x = features.clone().reshape(B, T, self.num_nodes, -1)
        for block in self.gcn_block:
            x = block(x, video_mask)

        x = x.mean(dim=-2)

        x = masked_mean_pool(x, video_mask)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ModelOutput(logits=logits, loss=loss)

class ISLR_GCN_Transformer_V2(nn.Module):
    def __init__(self, gcn_channels=(64, 128), d_model=128,
                 num_encoder_layers=3, nhead=8, dim_feedforward=128 * 4,
                 dropout=0.1, max_seq_len=500, num_classes=1000):
        super().__init__()
        self.num_nodes = _NUM_NODE
        self.feature_dim = self.num_nodes * _COORD_DIM
        adjacency, _ = build_direction_adjacency_from_edges(
            HAND_EDGES, num_nodes=self.num_nodes
        )
        self.model = GCN_Transformer(
            input_dim=_COORD_DIM,
            hidden_dim=d_model,
            num_encoder_layers=num_encoder_layers,
            n_head=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            num_classes=num_classes,
            channels=gcn_channels,
            base_adjacency=adjacency,
            num_nodes=self.num_nodes,
        )

    def forward(self, features, labels=None, video_mask=None, x_rgb=None):
        if features.ndim == 3:
            B, T, D = features.shape
            if D not in (self.feature_dim, 3 * self.feature_dim):
                raise ValueError("Expected shape features or concatenated shape/average/position features")
            # Input layout: [x_shape, x_average, x_position].
            x_shape = features[:, :, :self.feature_dim].reshape(
                B, T, self.num_nodes, _COORD_DIM
            )
        elif features.ndim == 4 and tuple(features.shape[2:]) == (
            self.num_nodes, _COORD_DIM
        ):
            x_shape = features
        else:
            raise ValueError("Invalid feature shape")
        output = self.model(x_shape, video_mask=video_mask, labels=labels)
        return ModelOutput(logits=output["logits"], loss=output["loss"])


class ISLR_Transformer_GCN_LateFusion(nn.Module):
    """Parallel temporal Transformer and SPDStack GCN-TCN using only x_shape.

    Late fusion: alpha * transformer_logits + (1-alpha) * gcn_logits.
    """
    def __init__(self, gcn_channels=(64, 64, 128, 128), d_model=128,
                 num_encoder_layers=3, nhead=8, dim_feedforward=128 * 8,
                 dropout=0.1, max_seq_len=5000, num_classes=1000,
                 fusion_alpha=0.5, decouple_p=4, temporal_kernel_size=9):
        super().__init__()
        if d_model <= 0 or nhead <= 0 or d_model % nhead or d_model % 2:
            raise ValueError("d_model must be positive, even and divisible by nhead")
        if num_encoder_layers <= 0 or max_seq_len <= 0:
            raise ValueError("Layer count and max_seq_len must be positive")
        if not 0 <= fusion_alpha <= 1:
            raise ValueError("fusion_alpha must be in [0, 1]")
        self.num_nodes = _NUM_NODE
        self.feature_dim = self.num_nodes * _COORD_DIM
        self.max_seq_len = max_seq_len
        self.fusion_alpha = fusion_alpha
        self.shape_projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.LayerNorm(d_model),
        )
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_encoder_layers, enable_nested_tensor=False
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        adjacency, _ = build_direction_adjacency_from_edges(
            HAND_EDGES, num_nodes=self.num_nodes
        )
        self.gcn_branch = SPDStack(
            channels=(_COORD_DIM, *gcn_channels), num_nodes=self.num_nodes,
            base_adjacency=adjacency, decouple_p=decouple_p,
            use_tcn=True, temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
        )
        def classifier(channels):
            return nn.Sequential(nn.LayerNorm(channels), nn.Dropout(dropout),
                                 nn.Linear(channels, num_classes))
        self.transformer_classifier = classifier(d_model)
        self.gcn_classifier = classifier(gcn_channels[-1])

    def forward(self, features, labels=None, video_mask=None, x_rgb=None):
        if features.ndim == 3:
            B, T, D = features.shape
            if D not in (self.feature_dim, self.feature_dim * 3):
                raise ValueError("Expected shape-only or shape/average/position features")
            shape = features[:, :, :self.feature_dim].reshape(
                B, T, self.num_nodes, _COORD_DIM
            )
        elif features.ndim == 4 and tuple(features.shape[2:]) == (self.num_nodes, _COORD_DIM):
            shape = features
            B, T = features.shape[:2]
        else:
            raise ValueError("Invalid feature shape")
        if not 1 <= T <= self.max_seq_len:
            raise ValueError("Sequence length must be between 1 and max_seq_len")
        if video_mask is None:
            video_mask = torch.ones(B, T, device=features.device, dtype=torch.bool)
        else:
            video_mask = video_mask.to(device=features.device, dtype=torch.bool)
            if tuple(video_mask.shape) != (B, T):
                raise ValueError("video_mask must have shape (B, T)")
        if not video_mask.any(dim=1).all():
            raise ValueError("Each video must have at least one valid frame")
        shape = shape.masked_fill(~video_mask[:, :, None, None], 0)
        transformer = self.shape_projection(shape.reshape(B, T, self.feature_dim))
        transformer = self.pos_encoder(transformer)
        transformer = transformer.masked_fill(~video_mask[:, :, None], 0)
        transformer = self.encoder(transformer, src_key_padding_mask=~video_mask)
        transformer = self.encoder_norm(transformer).masked_fill(~video_mask[:, :, None], 0)
        gcn = self.gcn_branch(shape, video_mask).mean(dim=2)
        weights = video_mask[:, :, None].to(dtype=transformer.dtype)
        count = weights.sum(dim=1).clamp(min=1)
        transformer_pooled = (transformer * weights).sum(dim=1) / count
        gcn_pooled = (gcn * weights.to(dtype=gcn.dtype)).sum(dim=1) / count.to(dtype=gcn.dtype)
        transformer_logits = self.transformer_classifier(transformer_pooled)
        gcn_logits = self.gcn_classifier(gcn_pooled)
        logits = self.fusion_alpha * transformer_logits + (1 - self.fusion_alpha) * gcn_logits
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return ModelOutput(logits=logits, loss=loss,
                           transformer_logits=transformer_logits, gcn_logits=gcn_logits)
