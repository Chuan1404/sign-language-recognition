import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.positional_encoding import PositionalEncoding

from config import _N_POSE, _NUM_NODE, _COORD_DIM, _N_HAND
from src.models.spatial_graph import build_adjacency, GCNBlock


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
    def __init__(self, input_dim=138, hidden_dim=512, num_encoder_layers=6, nhead=8, dim_feedforward=512 * 4,
                 dropout=0.1, max_seq_len=5000, num_classes=2000):
        super().__init__()

        d_model = hidden_dim
        self.input_projection = nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.Dropout(dropout),
                                              nn.LayerNorm(d_model))

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.encoder_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def encode(self, features, video_mask):
        if video_mask is None:
            raise ValueError("video_mask is required")

        video_mask = video_mask.bool()

        x = self.input_projection(features)  # (B, T, d_model)
        x = self.pos_encoder(x)  # (B, T, d_model)
        x = self.encoder(x, src_key_padding_mask=~video_mask  # True = ignore (padding)
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


class SimpleTCN(nn.Module):
    def __init__(self, channels, kernel_size=9):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0))
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, T, V, C) -> Conv2d cần (B, C, T, V)
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.bn(self.conv(x)))
        return x.permute(0, 2, 3, 1)


class DecoupledGCN(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes, base_adjacency, decouple_p=4):
        super().__init__()

        self.V = num_nodes
        self.p = decouple_p
        self.phi = nn.Linear(in_channels, out_channels)

        base_adjacency = base_adjacency.float()
        self.register_buffer("I", torch.eye(self.V))  # (1, N, N)

        self.A_in = nn.Parameter(base_adjacency.unsqueeze(0).repeat(self.p, 1, 1) * 1e-3)  # (p, N, N)

        self.A_out = nn.Parameter(base_adjacency.t().unsqueeze(0).repeat(self.p, 1, 1) * 1e-3)  # (p, N, N)

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
    def __init__(self, in_ch, out_ch, num_nodes, base_adjacency, decouple_p=4, drop=True):
        super().__init__()

        self.gcn = DecoupledGCN(in_ch, out_ch, num_nodes, base_adjacency, decouple_p)
        self.tcn = SimpleTCN(out_ch, kernel_size=1)

    def forward(self, x, mask):
        m = None if mask is None else mask[:, :, None, None].to(x.dtype)

        feat, A_raw = self.gcn(x)
        if m is not None:
            feat = feat * m
        feat = self.tcn(feat)
        if m is not None:
            feat = feat * m

        return feat


class SPDStack(nn.Module):

    def __init__(self, channels, num_nodes, base_adjacency, groups, num_drop_per_group=1, decouple_p=4):
        super().__init__()

        num_blocks = len(channels) - 1
        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            is_last = (i == num_blocks - 1)

            block = SelfPacingDroppingBlock(in_ch=channels[i], out_ch=channels[i + 1], num_nodes=num_nodes,
                                            base_adjacency=base_adjacency, groups=[], )

            self.blocks.append(block)

            if is_last:
                break

    def forward(self, x):
        feat = x
        for block in self.blocks:
            feat, _ = block(feat)

        return feat


class ISLR_V2(nn.Module):
    def __init__(self, hidden_dim=256, channels=(64, 64, 128, 128), dropout=0.1, num_classes=2000):
        super().__init__()

        d_model = hidden_dim
        out = channels[-1]

        self.num_nodes = _NUM_NODE

        adjacency_edges_matrix = build_adjacency()
        # groups = build_arm_groups(num_pose_points, num_hand_points)

        self.spd = SPDStack(channels=[_COORD_DIM, *channels], num_nodes=self.num_nodes,
                            base_adjacency=adjacency_edges_matrix, groups=[])

        self.classifier = nn.Linear(out, num_classes)

    def forward(self, features, labels=None, video_mask=None):
        B, T, _ = features.shape

        features = features[:, :, :self.num_nodes * 2]
        features = features.reshape(B, T, self.num_nodes, _COORD_DIM)

        spd_feat = self.spd(features)

        spd_feat = spd_feat.mean(dim=2)  # (B, T, C_out)

        pooled = masked_mean_pool(spd_feat, video_mask.bool())
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss


class FrameAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim phải chia hết cho num_heads"

        self.num_heads = num_heads
        self.d_out = hidden_dim
        self.d_head = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.Q = nn.Linear(input_dim, hidden_dim)
        self.K = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)  # gộp các head lại

        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.LayerNorm(hidden_dim), )

    def forward(self, features, video_mask):
        B, T, D = features.shape

        x = features
        Q = self.Q(x)  # (B, T, hidden_dim)
        K = self.K(x)
        V = self.V(x)

        # Tách thành nhiều head: (B, T, hidden_dim) -> (B, num_heads, T, d_head)
        Q = Q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        attn_score = Q @ K.transpose(-2, -1)  # (B, num_heads, T, T)

        mask = video_mask[:, None, None, :].bool()  # (B, 1, 1, T) — khớp (B, num_heads, T, T)
        attn_score = attn_score.masked_fill(~mask, float("-inf"))

        attn_weight = F.softmax(attn_score / self.d_head ** 0.5, dim=-1)  # chia theo d_head, không phải d_out
        attn_weight = self.dropout(attn_weight)

        context = attn_weight @ V  # (B, num_heads, T, d_head)

        # Gộp các head lại: (B, num_heads, T, d_head) -> (B, T, hidden_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_out)
        context = self.out_proj(context)

        context = self.ffn(context)  # (B, T, hidden_dim)

        return context

class FusionStem(nn.Module):
    def __init__(self, in_ch, out_ch, num_nodes, dropout=0.1):
        super().__init__()
        self.N = num_nodes
        def branch():
            return nn.Sequential(nn.Linear(in_ch, out_ch), nn.GELU(),
                                 nn.Dropout(dropout), nn.LayerNorm(out_ch))
        self.pos, self.shp, self.avg = branch(), branch(), branch()

    def forward(self, x):                      # (B, T, 3N, C_in)
        N = self.N
        # return (self.pos(x[:, :, :N])
        #         + self.shp(x[:, :, N:2*N])
        #         + self.avg(x[:, :, 2*N:3*N]))

        return x[:, :, :N] + x[:, :, N:2 * N] + x[:, :, 2 * N:3 * N]

class ISLR_V3(nn.Module):
    def __init__(self, channels=(64, 64, 128, 128), num_classes=2000):
        super().__init__()

        self.num_nodes = _NUM_NODE
        self.register_buffer('adjacency_matrix', build_adjacency())

        channels = [_COORD_DIM, *channels]
        gcn_out_dim = channels[-1]

        self.stem = FusionStem(channels[0], channels[1], self.num_nodes)

        self.gcn_block = nn.ModuleList([
            SelfPacingDroppingBlock(in_ch=channels[i], out_ch=channels[i + 1], num_nodes=self.num_nodes, base_adjacency=self.adjacency_matrix)
            for i in range(len(channels) - 1)
        ])

        # self.gcn_block = nn.ModuleList([
        #     GCNBlock(in_ch=channels[i], out_ch=channels[i + 1], num_nodes=self.num_nodes, base_adjacency=self.adjacency_matrix)
        #     for i in range(1, len(channels) - 1)
        # ])

        # self.frame_attention_block = nn.ModuleList([
        #     FrameAttention(gcn_out_dim, gcn_out_dim) for _ in range(3)
        # ])

        self.classifier = nn.Linear(gcn_out_dim, num_classes)

    def forward(self, features, labels=None, video_mask=None):
        B, T, _ = features.shape

        video_mask = video_mask.bool()

        features = features.reshape(B, T, self.num_nodes * 3, _COORD_DIM)
        features = self.stem(features)

        for block in self.gcn_block:
            features = block(features, video_mask)
        # features = self.gcn_block(features)

        features = features.mean(dim=-2)

        # for block in self.frame_attention_block:
        #     features = block(features, video_mask)

        features = masked_mean_pool(features, video_mask)

        logits = self.classifier(features)
        loss = F.cross_entropy(logits, labels)

        return logits, loss


class ISLR_V4(nn.Module):
    def __init__(self, input_dim=_NUM_NODE * _COORD_DIM, hidden_dim=256, num_encoder_layers=6, nhead=8,
                 dim_feedforward=256 * 8, dropout=0.2, max_seq_len=5000, num_classes=1000):
        super().__init__()

        d_model = hidden_dim
        self.num_nodes = _NUM_NODE

        self.shape_projection = nn.Sequential(nn.Linear(self.num_nodes * 2, d_model), nn.GELU(), nn.Dropout(dropout),
                                              nn.LayerNorm(d_model))

        self.position_projection = nn.Sequential(nn.Linear(self.num_nodes * 2, d_model), nn.GELU(), nn.Dropout(dropout),
                                                 nn.LayerNorm(d_model))

        self.average_projection = nn.Sequential(nn.Linear(self.num_nodes * 2, d_model), nn.GELU(), nn.Dropout(dropout),
                                            nn.LayerNorm(d_model))

        self.velocity_projection = nn.Sequential(nn.Linear(self.num_nodes * 2, d_model), nn.GELU(), nn.Dropout(dropout),
                                                nn.LayerNorm(d_model))

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True, norm_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.encoder_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def encode(self, features, video_mask):
        if video_mask is None:
            raise ValueError("video_mask is required")

        B, T, _ = features.shape

        video_mask = video_mask.bool()
        position_features = features[:, :, :self.num_nodes * 2]
        shape_features = features[:, :, self.num_nodes * 2:self.num_nodes * 4]
        average_features = features[:, :, self.num_nodes * 4:self.num_nodes * 6]

        # position_features = (vel_features * position_features).reshape(B, T, self.num_nodes * 2)
        # shape_features = (vel_features * shape_features).reshape(B, T, self.num_nodes * 2)

        x_position = self.position_projection(position_features)
        x_shape = self.shape_projection(shape_features)
        x_average = self.average_projection(average_features)
        # x_velocity = self.velocity_projection(velocity_features)

        x = (x_position + x_shape + x_average)

        x = self.pos_encoder(x)  # (B, T, d_model)
        x = self.encoder(x, src_key_padding_mask=~video_mask  # True = ignore (padding)
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


class CrossAttentionLayer(nn.Module):
    """Query từ stream A, Key/Value từ stream B (pre-norm + residual + FFN)."""

    def __init__(self, d_model, nhead=8, dim_feedforward=None, dropout=0.2):
        super().__init__()
        dim_feedforward = dim_feedforward or d_model * 4

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout), )

    def forward(self, q, kv, kv_pad_mask):
        # kv_pad_mask: True = padding (bị bỏ qua)
        out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), key_padding_mask=kv_pad_mask,
                           need_weights=False)
        q = q + self.drop(out)
        q = q + self.ffn(self.norm_ffn(q))
        return q


class ISLR_V5(nn.Module):
    def __init__(self, gcn_channels=(128, 128, 256, 256), d_model=256, num_encoder_layers=6, nhead=8,
                 dim_feedforward=256 * 8, dropout=0.2, max_seq_len=5000, num_classes=1000):
        super().__init__()
        self.num_nodes = _NUM_NODE

        # ---------------- Stream 1: GCN (V3) ----------------
        self.register_buffer('adjacency_matrix', build_adjacency())
        channels = [_COORD_DIM, *gcn_channels]

        self.stem = FusionStem(channels[0], channels[1], self.num_nodes)
        self.gcn_block = nn.ModuleList([
            SelfPacingDroppingBlock(channels[i], channels[i + 1], self.num_nodes, self.adjacency_matrix)
            for i in range(len(channels) - 1)])
        # self.gcn_proj = nn.Sequential(nn.Linear(channels[-1], d_model), nn.LayerNorm(d_model))

        # ---------------- Stream 2: Transformer (V4) ----------------
        self.shape_projection = nn.Sequential(nn.Linear(self.num_nodes * 2, d_model), nn.GELU(), nn.Dropout(dropout),
                                              nn.LayerNorm(d_model))
        self.position_projection = nn.Sequential(nn.Linear(self.num_nodes * 2, d_model), nn.GELU(), nn.Dropout(dropout),
                                                 nn.LayerNorm(d_model))
        self.average_projection = nn.Sequential(nn.Linear(self.num_nodes * 2, d_model), nn.GELU(), nn.Dropout(dropout),
                                                 nn.LayerNorm(d_model))
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # ---------------- Cross-attention fusion ----------------
        # gcn <- transformer  và  transformer <- gcn
        self.norm_g = nn.LayerNorm(d_model)

        # ---------------- Classifier ----------------
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes), )

    # ---- stream 1 ----
    def encode_gcn(self, features, video_mask):
        features = self.stem(features)

        for block in self.gcn_block:
            features = block(features, video_mask)

        features = features.mean(dim=-2)  # (B, T, gcn_out)
        # return self.gcn_proj(x)  # (B, T, d_model)
        return self.norm_g(features)

    # ---- stream 2 ----
    def encode_transformer(self, features, video_mask):
        B, T, _ = features.shape

        x_position = self.position_projection(features[:, :, :self.num_nodes * 2])
        x_shape = self.shape_projection(features[:, :, self.num_nodes * 2:self.num_nodes * 4])
        x_average = self.average_projection(features[:, :, self.num_nodes * 4:self.num_nodes * 6])
        x_gcn = self.encode_gcn(features.clone().reshape(B, T, self.num_nodes * 3, _COORD_DIM), video_mask)

        # x = x_average                                     # best=73.00% loss=1.2133 75.00%
        # x = x_average + x_gcn                             # best=66.00% loss=1.4254 95.00%

        # x = x_position                                    # best=72.00% loss=1.2574 75.00%
        # x = x_position + x_gcn                            # best=70.00% loss=1.3490 71.00%

        # x = x_shape                                       # best=66.00% loss=1.2873 67.00%
        # x = x_shape + x_gcn                               # best=59.00% loss=1.3645 67.00%

        # x = x_shape + x_average                           # best=73.00% loss=1.1507 75.00%
        # x = x_shape + x_average + x_gcn                   # best=71.00% loss=1.1686 74.00

        x = x_shape + x_average + x_position              # best=75.00% loss=1.0840 77.00%
        # x = x_shape + x_average + x_position + x_gcn      # best=71.00% loss=1.1394 78.00%
        # x = x_gcn                                           # best=62.00% loss=1.4346 66.00%
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=~video_mask)
        return self.encoder_norm(x)  # (B, T, d_model)

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")
        video_mask = video_mask.bool()

        B, T, _ = features.shape

        t = self.encode_transformer(features, video_mask)

        t = masked_mean_pool(t, video_mask)  # (B, d_model)

        logits = self.classifier(t)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss

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
        self.A = nn.Parameter(adjacency.clone().unsqueeze(0).repeat(p, 1, 1))   # (p, N, N)
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


class ShortTermTCN(nn.Module):
    """Nửa kênh đầu — học chuyển động NGẮN HẠN bằng Conv1d theo thời gian."""

    def __init__(self, channels, kernel_size=9, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C) -> (B, T, C)
        x = x.transpose(1, 2)                        # (B, C, T)
        x = self.drop(self.act(self.bn(self.conv(x))))
        return x.transpose(1, 2)


class LongTermTransformer(nn.Module):
    """Nửa kênh còn lại — học chuyển động DÀI HẠN bằng self-attention toàn chuỗi."""

    def __init__(self, d_model, nhead=8, dim_feedforward=1024, num_layers=4, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, video_mask):
        # x: (B, T, C), video_mask: (B, T) bool
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=~video_mask)
        return self.norm(x)


class ISLR_V6(nn.Module):
    """
    1) Mỗi frame áp dụng GCN theo KHÔNG GIAN (SpatialGCNBlock) — không TCN
       trong bước này, xử lý độc lập từng frame.
    2) Sau khi pool theo node -> (B,T,C), CHIA ĐÔI kênh:
           nửa 1 -> TCN          (ngắn hạn)
           nửa 2 -> Transformer  (dài hạn)
    3) Ghép lại 2 nửa đã học -> pool theo T -> classifier.
    4) KHÔNG có cross-attention / 2-stream phức tạp như V5 (bỏ tạm phần đó).
    """

    def __init__(
        self,
        gcn_channels=(128, 128, 256, 256),
        gcn_decouple_p=4,
        tcn_kernel_size=9,
        num_transformer_layers=4,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.2,
        max_seq_len=5000,
        num_classes=1000,
    ):
        super().__init__()

        assert gcn_channels[-1] % 2 == 0, "kênh GCN cuối phải chia hết cho 2 để tách nửa TCN / nửa Transformer"

        self.num_nodes = _NUM_NODE
        self.register_buffer('adjacency_matrix', build_adjacency())

        # out_ch truyền vào đây hiện KHÔNG có tác dụng thật (xem ghi chú đầu
        # câu trả lời) — output thật của FusionStem vẫn là _COORD_DIM kênh.
        self.stem = FusionStem(_COORD_DIM, gcn_channels[0], self.num_nodes)

        ch = [_COORD_DIM, *gcn_channels]   # input đầu tiên của GCN = _COORD_DIM, khớp output thật của stem hiện tại
        self.gcn_blocks = nn.ModuleList([
            SpatialGCNBlock(ch[i], ch[i + 1], self.num_nodes, self.adjacency_matrix,
                             p=gcn_decouple_p, dropout=dropout)
            for i in range(len(ch) - 1)
        ])

        gcn_out = gcn_channels[-1]
        self.half = gcn_out // 2

        self.short_term = ShortTermTCN(self.half, kernel_size=tcn_kernel_size, dropout=dropout)
        self.long_term = LongTermTransformer(
            self.half, nhead=nhead, dim_feedforward=dim_feedforward,
            num_layers=num_transformer_layers, dropout=dropout, max_seq_len=max_seq_len
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(gcn_out),
            nn.Dropout(dropout),
            nn.Linear(gcn_out, num_classes),
        )

    def encode_gcn(self, features):
        # features: (B, T, 3N, C_in) -> stem gộp 3 nhóm (position/shape/average) -> (B,T,N,_COORD_DIM)
        x = self.stem(features)

        for block in self.gcn_blocks:
            x = block(x)                      # (B, T, N, C_out) — chỉ GCN, không TCN

        return x.mean(dim=-2)                 # (B, T, C_out) — pool theo node

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")
        video_mask = video_mask.bool()

        B, T, _ = features.shape
        x = self.encode_gcn(features.clone().reshape(B, T, self.num_nodes * 3, _COORD_DIM))  # (B,T,gcn_out)

        x_short, x_long = x[..., :self.half], x[..., self.half:]

        x_short = self.short_term(x_short)                 # (B,T,half) — ngắn hạn
        x_long = self.long_term(x_long, video_mask)          # (B,T,half) — dài hạn

        fused = torch.cat([x_short, x_long], dim=-1)         # (B,T,gcn_out)

        pooled = masked_mean_pool(fused, video_mask)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss