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
        self.tcn = SimpleTCN(out_ch)

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
    def __init__(self, in_ch, out_ch, num_nodes, dropout=0.0):
        super().__init__()
        self.N = num_nodes
        def branch():
            return nn.Sequential(nn.Linear(in_ch, out_ch), nn.GELU(),
                                 nn.Dropout(dropout), nn.LayerNorm(out_ch))
        self.pos, self.shp, self.avg = branch(), branch(), branch()

    def forward(self, x):                      # (B, T, 3N, C_in)
        N = self.N
        return (self.pos(x[:, :, :N])
                + self.shp(x[:, :, N:2*N])
                + self.avg(x[:, :, 2*N:3*N]))

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
            for i in range(1, len(channels) - 1)
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
                 dim_feedforward=256 * 4, dropout=0.2, max_seq_len=5000, num_classes=1000):
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
        velocity_features = features[:, :, self.num_nodes * 6:]

        # position_features = (vel_features * position_features).reshape(B, T, self.num_nodes * 2)
        # shape_features = (vel_features * shape_features).reshape(B, T, self.num_nodes * 2)

        x_position = self.position_projection(position_features)
        x_shape = self.shape_projection(shape_features)
        x_average = self.average_projection(average_features)
        x_velocity = self.velocity_projection(velocity_features)

        x = (x_position + x_shape + x_average + x_velocity)

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
    def __init__(self, gcn_channels=(64, 64, 128, 128), d_model=128, num_encoder_layers=6, nhead=8,
                 dim_feedforward=256 * 4, dropout=0.2, max_seq_len=5000, num_cross_layers=2, num_classes=1000):
        super().__init__()
        self.num_nodes = _NUM_NODE

        # ---------------- Stream 1: GCN (V3) ----------------
        self.register_buffer('adjacency_matrix', build_adjacency())
        channels = [_COORD_DIM, *gcn_channels]

        self.stem = FusionStem(channels[0], channels[1], self.num_nodes)
        self.gcn_block = nn.ModuleList([
            SelfPacingDroppingBlock(channels[i], channels[i + 1], self.num_nodes, self.adjacency_matrix)
            for i in range(1, len(channels) - 1)])
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
        self.cross_g2t = nn.ModuleList(
            [CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_cross_layers)])
        # self.cross_t2g = nn.ModuleList(
        #     [CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_cross_layers)])
        # self.norm_g = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)

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
        return features

    # ---- stream 2 ----
    def encode_transformer(self, features, video_mask):
        B, T, _ = features.shape

        x_position = self.position_projection(features[:, :, :self.num_nodes * 2])
        x_shape = self.shape_projection(features[:, :, self.num_nodes * 2:self.num_nodes * 4])
        x_average = self.average_projection(features[:, :, self.num_nodes * 4:self.num_nodes * 6])

        x = x_position + x_shape + x_average
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=~video_mask)
        return self.encoder_norm(x)  # (B, T, d_model)

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")
        video_mask = video_mask.bool()
        pad_mask = ~video_mask

        B, T, _ = features.shape

        g = self.encode_gcn(features.clone().reshape(B, T, self.num_nodes * 3, _COORD_DIM), video_mask)
        t = self.encode_transformer(features, video_mask)

        for t2g in self.cross_g2t:
            t_new = t2g(t, g, pad_mask)  # Transformer hỏi GCN
            t = t_new

        # g = masked_mean_pool(self.norm_g(g), video_mask)  # (B, d_model)
        t = masked_mean_pool(self.norm_t(t), video_mask)  # (B, d_model)

        # pooled = torch.cat([g, t], dim=-1)  # (B, 2*d_model)
        logits = self.classifier(t)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss
