import torch
import torch.nn.functional as F
from torch import nn

from config import _COORD_DIM, _N_HAND, _N_POSE, _NUM_NODE
from src.models.positional_encoding import PositionalEncoding
from src.models.spatial_graph import GCNBlock, SelfPacingDroppingBlock, build_adjacency
try:
    from src.training.train import masked_mean_pool
except ImportError:
    try:
        from training.train import masked_mean_pool
    except ImportError:
        def masked_mean_pool(x, video_mask=None):
            if video_mask is None:
                return x.mean(dim=1)
            mask = video_mask.unsqueeze(-1).float()  # (B, T, 1)
            summed = (x * mask).sum(dim=1)           # (B, D)
            counts = mask.sum(dim=1).clamp(min=1.0)   # (B, 1)
            return summed / counts


class ModelOutput(dict):
    """Container allowing tuple unpacking: (logits, loss) = output,
    attribute access: output.logits, output.loss,
    and dict access: output['logits'], output['loss'].
    """
    def __init__(self, logits=None, loss=None, **kwargs):
        super().__init__(logits=logits, loss=loss, **kwargs)
        self.__dict__ = self

    def __iter__(self):
        yield self["logits"]
        yield self["loss"]

    def __getitem__(self, key):
        return super().__getitem__(key)


class SpatialGCNBlock(nn.Module):
    """
    CHỈ GCN theo KHÔNG GIAN (giữa các node trong CÙNG 1 frame), áp dụng ĐỘC
    LẬP cho từng frame — KHÔNG có TCN trộn thông tin qua các frame khác nhau
    (khác SelfPacingDroppingBlock trước đây luôn kèm TCN ngay sau GCN, theo
    yêu cầu #1: học thời gian được tách hẳn ra 2 nhánh riêng ở tầng sau).
    """

    def __init__(self, in_ch, out_ch, num_nodes, adjacency, p=4, dropout=0.1):
        super().__init__()
        self.p = p
        self.phi = nn.Linear(in_ch, out_ch)
        self.A = nn.Parameter(
            adjacency.clone().unsqueeze(0).repeat(p, 1, 1)
        )  # (p, N, N)
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
        self.register_buffer("adjacency_matrix", build_adjacency(self.num_nodes))

        channels = [_COORD_DIM, *channels]
        gcn_out_dim = channels[-1]

        self.gcn_block = nn.ModuleList(
            [
                GCNBlock(
                    in_ch=channels[i],
                    out_ch=channels[i + 1],
                    num_nodes=self.num_nodes,
                    base_adjacency=self.adjacency_matrix,
                )
                for i in range(len(channels) - 1)
            ]
        )

        self.classifier = nn.Linear(gcn_out_dim, num_classes)

    def forward(self, features, labels=None, video_mask=None, important_indices=None):
        """
        features: (B, T, 88) or (B, T, num_nodes, coord_dim)
        important_indices: Optional frame indices (1D tensor/list) to filter features to important frames only.
        labels: Ground truth labels (optional).
        video_mask: Boolean/float mask of valid frames (optional).
        """
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
                raise ValueError(f"Feature dimension {D} is invalid for num_nodes={self.num_nodes}, coord_dim={_COORD_DIM}")
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
    def __init__(
        self,
        gcn_channels=(128, 128, 256, 256),
        d_model=256,
        num_encoder_layers=6,
        nhead=8,
        dim_feedforward=256 * 8,
        dropout=0.2,
        max_seq_len=5000,
        num_classes=1000,
    ):
        super().__init__()
        self.num_nodes = _NUM_NODE

        self.register_buffer("adjacency_matrix", build_adjacency())
        channels = [_COORD_DIM, *gcn_channels]

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
        self.gcn_block = nn.ModuleList(
            [
                SelfPacingDroppingBlock(
                    channels[i], channels[i + 1], _N_HAND * 2, self.adjacency_matrix
                )
                for i in range(len(channels) - 1)
            ]
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

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def encode_gcn(self, features, video_mask):

        for block in self.gcn_block:
            features = block(features, video_mask)

        features = features.mean(dim=-2)  # (B, T, gcn_out)
        # return self.gcn_proj(x)  # (B, T, d_model)
        return features

    def encode_transformer(self, features, video_mask):
        B, T, _ = features.shape

        pose_features = features[:, :, : _N_POSE * _COORD_DIM]
        hand_features = features[:, :, _N_POSE * _COORD_DIM :]

        x_gcn = self.encode_gcn(
            features[:, :, _N_POSE * _COORD_DIM :]
            .clone()
            .reshape(B, T, _N_HAND * 2, _COORD_DIM),
            video_mask,
        )
        x_pose = self.pose_projection(pose_features)
        x_hand = self.hand_projection(hand_features)

        # x = x_average                                     # best=73.00% loss=1.2133 75.00%
        # x = x_average + x_gcn                             # best=66.00% loss=1.4254 95.00%

        # x = x_position                                    # best=72.00% loss=1.2574 75.00%
        # x = x_position + x_gcn                            # best=70.00% loss=1.3490 71.00%

        # x = x_shape                                       # best=66.00% loss=1.2873 67.00%
        # x = x_shape + x_gcn                               # best=59.00% loss=1.3645 67.00%

        # x = x_shape + x_average                           # best=73.00% loss=1.1507 75.00%
        # x = x_shape + x_average + x_gcn                   # best=71.00% loss=1.1686 74.00

        # x = x_shape + x_average + x_position  # best=75.00% loss=1.0840 77.00%
        # x = x_shape + x_average + x_position + x_gcn      # best=71.00% loss=1.1394 78.00%
        # x = x_gcn                                           # best=62.00% loss=1.4346 66.00%

        x = x_pose + x_hand + x_gcn
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=~video_mask)
        return self.encoder_norm(x)  # (B, T, d_model)

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

        if features.shape[-1] >= self.num_nodes * 4:
            features = features[:, :, self.num_nodes * 2 : self.num_nodes * 4]

        t = self.encode_transformer(features, video_mask)

        t = masked_mean_pool(t, video_mask)  # (B, d_model)

        logits = self.classifier(t)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ModelOutput(logits=logits, loss=loss)