import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.positional_encoding import PositionalEncoding

from config import _N_POSE, _NUM_NODE, _COORD_DIM, _N_HAND
from src.models.spatial_graph import GrapConvBlock, build_adjacency, GCN_Block

class ClassificationOutput:

    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits


def masked_mean_pool(x, video_mask):
    mask = video_mask.unsqueeze(-1).float()  # (B, T, 1)
    summed = (x * mask).sum(dim=1)  # (B, D)
    counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1) — avoid /0
    return summed / counts

class SignLanguageTranslatorV1(nn.Module):
    def __init__(
            self,
            input_dim=138,
            hidden_dim=512,
            num_encoder_layers=6,
            nhead=8,
            dim_feedforward=512 * 4,
            dropout=0.1,
            max_seq_len=5000,
            num_classes=2000
    ):
        super().__init__()

        d_model = hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.encoder_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def encode(self, features, video_mask):
        if video_mask is None:
            raise ValueError("video_mask is required")

        video_mask = video_mask.bool()

        x = self.input_projection(features)  # (B, T, d_model)
        x = self.pos_encoder(x)  # (B, T, d_model)
        x = self.encoder(
            x,
            src_key_padding_mask=~video_mask  # True = ignore (padding)
        )  # (B, T, d_model)
        x = self.encoder_norm(x)  # (B, T, d_model)

        return x

    def forward(self, features, labels=None, video_mask=None):
        x = self.encode(features, video_mask)
        pooled = masked_mean_pool(x, video_mask.bool())
        logits = self.classifier(pooled)  # (B, num_classes)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss

    @torch.no_grad()
    def predict(self, features, video_mask=None, top_k=1):
        logits = self.forward(features, video_mask=video_mask).logits
        if top_k == 1:
            return logits.argmax(dim=-1)


class ISLR_V1(nn.Module):
    def __init__(
            self,
            input_dim=_NUM_NODE * _COORD_DIM,
            hidden_dim=512,
            num_encoder_layers=6,
            nhead=8,
            dim_feedforward=512 * 4,
            dropout=0.2,
            max_seq_len=5000,
            num_classes=2000
    ):
        super().__init__()

        d_model = hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.encoder_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def encode(self, features, video_mask):
        if video_mask is None:
            raise ValueError("video_mask is required")

        video_mask = video_mask.bool()

        x = self.input_projection(features)
        x = self.pos_encoder(x)  # (B, T, d_model)
        x = self.encoder(
            x,
            src_key_padding_mask=~video_mask  # True = ignore (padding)
        )  # (B, T, d_model)
        x = self.encoder_norm(x)  # (B, T, d_model)

        return x

    def forward(self, features, labels=None, video_mask=None):
        B, T, _ = features.shape

        x = self.encode(features, video_mask)
        pooled = masked_mean_pool(x, video_mask.bool())
        logits = self.classifier(pooled)  # (B, num_classes)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss

    @torch.no_grad()
    def predict(self, features, video_mask=None, top_k=1):
        logits = self.forward(features, video_mask=video_mask).logits
        if top_k == 1:
            return logits.argmax(dim=-1)


class SimpleTCN(nn.Module):
    def __init__(self, channels, kernel_size=9):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(channels, channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0))
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, T, V, C) -> Conv2d cần (B, C, T, V)
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.bn(self.conv(x)))
        return x.permute(0, 2, 3, 1)


class DecoupledGCN(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes,
                 base_adjacency, decouple_p=4):
        super().__init__()

        self.V = num_nodes
        self.p = decouple_p
        self.phi = nn.Linear(in_channels, out_channels)

        base_adjacency = base_adjacency.float()
        self.register_buffer("I", torch.eye(self.V))  # (1, N, N)

        self.A_in = nn.Parameter(
            base_adjacency.unsqueeze(0).repeat(self.p, 1, 1) * 1e-3
        )  # (p, N, N)

        self.A_out = nn.Parameter(
            base_adjacency.t().unsqueeze(0).repeat(self.p, 1, 1) * 1e-3
        )  # (p, N, N)

    def _raw_A(self):
        return self.I.unsqueeze(0) + self.A_in + self.A_out

    def _normalized_A(self, A_raw):
        deg = A_raw.sum(-1).clamp(min=1e-6)  # (p, V)
        d_inv_sqrt = deg.pow(-0.5)
        D_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # (p, V, V)

        return D_inv_sqrt @ A_raw @ D_inv_sqrt  # (p, V, V)

    def forward(self, x):
        feat = self.phi(x)  # (B, T, V, C_out)
        A_raw = self._raw_A()  # (p, V, V)
        A_norm = self._normalized_A(A_raw)

        out = 0
        for k in range(self.p):
            out = out + torch.einsum('vw,btwc->btvc', A_norm[k], feat)
        out = out / self.p
        return out, A_raw


class SelfPacingDroppingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_nodes, base_adjacency,
                 groups, decouple_p=4, drop=True):
        super().__init__()

        self.gcn = DecoupledGCN(in_ch, out_ch, num_nodes, base_adjacency, decouple_p)
        self.tcn = SimpleTCN(out_ch)

    def forward(self, x):
        feat, A_raw = self.gcn(x)
        feat = self.tcn(feat)
        return feat, None


class SPDStack(nn.Module):

    def __init__(self, channels, num_nodes, base_adjacency, groups, num_drop_per_group=1, decouple_p=4):
        super().__init__()

        num_blocks = len(channels) - 1
        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            is_last = (i == num_blocks - 1)

            block = SelfPacingDroppingBlock(
                in_ch=channels[i],
                out_ch=channels[i + 1],
                num_nodes=num_nodes,
                base_adjacency=base_adjacency,
                groups=[],
            )

            self.blocks.append(block)

            if is_last:
                break

    def forward(self, x):
        feat = x
        for block in self.blocks:
            feat, _ = block(feat)

        return feat


class ISLR_V2(nn.Module):
    def __init__(
            self,
            hidden_dim=256,
            channels=(64, 64, 128, 128),
            dropout=0.1,
            num_classes=2000
    ):
        super().__init__()

        d_model = hidden_dim
        out = channels[-1]

        self.num_nodes = _NUM_NODE

        adjacency_edges_matrix = build_adjacency()
        # groups = build_arm_groups(num_pose_points, num_hand_points)

        self.spd = SPDStack(
            channels=[_COORD_DIM, *channels],
            num_nodes=self.num_nodes,
            base_adjacency=adjacency_edges_matrix,
            groups=[]
        )

        self.classifier = nn.Linear(out, num_classes)

    def forward(self, features, labels=None, video_mask=None):
        B, T, _ = features.shape

        features = features.reshape(B, T, self.num_nodes, _COORD_DIM)

        spd_feat = self.spd(features)

        spd_feat = spd_feat.mean(dim=2)  # (B, T, C_out)

        pooled = masked_mean_pool(spd_feat, video_mask.bool())
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss


class ISLR_V3(nn.Module):
    def __init__(self,
                 input_dim=_NUM_NODE * _COORD_DIM,
                 channels=(64, 64, 128, 128, 256, 256),
                 num_classes=2000,
                 hidden_dim=512,
                 num_encoder_layers=6,
                 nhead=8,
                 dim_feedforward=512 * 4,
                 dropout=0.1,
                 max_seq_len=1000):

        super().__init__()

        channels = [_COORD_DIM, *channels]
        out = channels[-1]
        d_model = hidden_dim

        self.num_nodes = _NUM_NODE

        self.register_buffer('adjacency_matrix', build_adjacency())

        self.gcn_stack = nn.ModuleList([
            GrapConvBlock(channels[i], channels[i + 1], self.num_nodes, base_adjacency=self.adjacency_matrix) for i in
            range(len(channels) - 1)
        ])

        self.input_projection = nn.Sequential(
            nn.Linear(out * self.num_nodes, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.encoder_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(out * 2, num_classes)

        self.gcn_output = nn.Sequential(
            nn.Linear(out * self.num_nodes, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        self.fused = nn.Linear(2 * d_model, d_model)

    def encode_gcn(self, features, video_mask):
        B, T, _ = features.shape
        feat = features.clone()

        feat = feat.reshape(B, T, self.num_nodes, _COORD_DIM)

        for block in self.gcn_stack:
            feat = block(feat, video_mask)

        # feat_pooled = feat.mean(dim=2)  # (B, T, C_out)
        feat = feat.reshape(B, T, -1)
        # fused_feat = self.gcn_output(hands_pooled)

        return feat

    def encode_trans(self, gcn_feat, video_mask):
        if video_mask is None:
            raise ValueError("video_mask is required")

        video_mask = video_mask.bool()

        x = self.input_projection(gcn_feat)
        x = self.pos_encoder(x)  # (B, T, d_model)
        x = self.encoder(
            x,
            src_key_padding_mask=~video_mask  # True = ignore (padding)
        )  # (B, T, d_model)
        x = self.encoder_norm(x)  # (B, T, d_model)

        return x

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")

        B, T, _ = features.shape

        gcn_output = self.encode_gcn(features, video_mask)
        trans_output = self.encode_trans(gcn_output, video_mask)

        fused = masked_mean_pool(trans_output, video_mask.bool())
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss


class FrameAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.d_out = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.Q = nn.Linear(input_dim, hidden_dim)
        self.K = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(input_dim, hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, features, video_mask):
        B, T, N, D = features.shape

        x = features.permute(0, 2, 1, 3)

        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        attn_score = Q @ K.transpose(-2, -1)

        mask = video_mask[:, None, None, :].bool()
        attn_score = attn_score.masked_fill(~mask, float("-inf"))

        attn_weight = F.softmax(attn_score / self.d_out ** 0.5, dim=-1)
        attn_weight = self.dropout(attn_weight)

        context = attn_weight @ V
        context = context.permute(0, 2, 1, 3)

        context = self.ffn(context)

        return context


class ISLR_V4(nn.Module):
    def __init__(self, channels=(64, 64, 128, 128, 256, 256), num_classes=2000):
        super().__init__()

        self.num_nodes = _NUM_NODE
        self.register_buffer('adjacency_matrix', build_adjacency())

        channels = [_COORD_DIM, *channels]
        gcn_out_dim = channels[-1]

        self.gcn_block = nn.ModuleList([
            # GrapConvBlock(channels[i], channels[i + 1], self.num_nodes, self.adjacency_matrix) for i in range(len(channels) - 1)
            GCN_Block(channels[i], channels[i + 1], self.num_nodes, self.adjacency_matrix) for i in
            range(len(channels) - 1)
        ])

        self.attn_block = nn.ModuleList([
            FrameAttention(input_dim=gcn_out_dim, hidden_dim=256, dropout=0.1) for i in range(6)
        ])

        self.frame_attention = FrameAttention(input_dim=gcn_out_dim, hidden_dim=256, dropout=0.1)

        self.classifier = nn.Linear(gcn_out_dim, num_classes)

    def forward(self, features, labels=None, video_mask=None):
        B, T, _ = features.shape
        features = features.reshape(B, T, self.num_nodes, _COORD_DIM)

        for block in self.gcn_block:
            features = block(features, video_mask)

        # features = self.frame_attention(features, video_mask)
        for block in self.attn_block:
            features = block(features, video_mask)

        # B T N C -> B N T C
        # features = features.transpose(1, 2)
        # split_features = features.reshape(B, self.num_nodes, T, 2 ,-1)

        # first split feature will be applied Self Attention
        # second split feature will be applied TCN

        # Merge feature
        # print(split_features.shape)

        features = features.mean(dim=-2)

        features = masked_mean_pool(features, video_mask)

        logits = self.classifier(features)
        loss = F.cross_entropy(logits, labels)

        return logits, loss
