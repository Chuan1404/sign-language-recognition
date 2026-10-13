import math
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn

from config import _N_HAND, _REMOVE_POSE_IDX, _NUM_NODE, _COORD_DIM

_HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
               (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13),
               (13, 17), ]

_POSE_BONES = [  # face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),  # torso / arms
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),  # hands (pose-model fingers)
    (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20), ]

CUSTOM_EDGES = []

remove_set = set(_REMOVE_POSE_IDX)

remaining_nodes = [n for n in range(32) if n not in remove_set]
old_to_new = {old: new for new, old in enumerate(remaining_nodes)}
n_pose_kept = len(remaining_nodes)


def _remap(edges):
    return [(old_to_new[a], old_to_new[b]) for a, b in edges if a in old_to_new and b in old_to_new]


POSE_EDGES = _remap(_POSE_BONES)

_LEFT_WRIST = 0
_RIGHT_WRIST = _LEFT_WRIST + _N_HAND

_LEFT_HAND_EDGES = [(_LEFT_WRIST + i, _LEFT_WRIST + j) for i, j in _HAND_BONES]
_RIGHT_HAND_EDGES = [(_RIGHT_WRIST + i, _RIGHT_WRIST + j) for i, j in _HAND_BONES]

FULL_BODY_EDGES = POSE_EDGES + _LEFT_HAND_EDGES + _RIGHT_HAND_EDGES
HAND_EDGES = _LEFT_HAND_EDGES + _RIGHT_HAND_EDGES
# FULL_BODY_EDGES = POSE_EDGES

# assert all(0 <= a < _NUM_NODE and 0 <= b < _NUM_NODE for a, b in HAND_EDGES)
assert len({tuple(sorted(e)) for e in HAND_EDGES}) == len(HAND_EDGES), ("duplicate edges")


def build_adjacency_from_edges(edges, num_nodes):
    A = torch.eye(num_nodes)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def build_direction_adjacency_from_edges(edges, num_nodes):
    A_in = torch.eye(num_nodes)
    A_out = torch.eye(num_nodes)
    for i, j in edges:
        A_in[j, i] = 1.0
        A_out[i, j] = 1.0

    return A_in, A_out


def _normalize_adjacency(A):
    deg = A.sum(dim=1).clamp(min=1e-6)
    d_inv_sqrt = torch.diag(deg.pow(-0.5))
    return d_inv_sqrt @ A @ d_inv_sqrt


def _build_hand_group_edges(offset):
    edges = [(offset + 0, offset + 1), (offset + 1, offset + 2), ]
    edges += [(offset + 2 + i, offset + 2 + j) for i, j in _HAND_BONES]
    return edges


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


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False)

        self.conv9 = nn.Conv2d(channels, channels, kernel_size=(9, 1), padding=(4, 0), bias=False)

        self.conv15 = nn.Conv2d(channels, channels, kernel_size=(15, 1), padding=(7, 0), bias=False)

        self.bn = nn.BatchNorm2d(channels * 3)

        self.fusion = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # B, T, N, C
        x = x.permute(0, 3, 1, 2)
        # B, C, T, N

        x3 = self.conv3(x)
        x9 = self.conv9(x)
        x15 = self.conv15(x)

        x = torch.cat([x3, x9, x15], dim=1)

        x = self.bn(x)
        x = self.fusion(x)
        x = self.drop(x)

        return x.permute(0, 2, 3, 1)


class GCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_nodes, base_adjacency, number_matrix=4, dropout=0.1):
        super().__init__()

        self.number_matrix = number_matrix
        self.register_buffer("A_raw", torch.maximum(base_adjacency.float(), torch.eye(num_nodes)))
        self.A_learnable = nn.Parameter(self.A_raw.unsqueeze(0).repeat(self.number_matrix, 1, 1))

        self.linear = nn.Sequential(nn.Linear(in_ch, out_ch), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(out_ch), )

        # self.tcn = TemporalConv(out_ch, kernel_size=9)
        self.tcn = TemporalConv(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Identity() if in_ch == out_ch else nn.Linear(in_ch, out_ch)

    def _normalized_A(self):
        # A = self.A_learnable * self.A_raw.unsqueeze(0)
        A = self.A_learnable

        d = A.sum(-1).clamp(min=1e-6).pow(-0.5)

        A_norm = d[:, :, None] * A * d[:, None, :]

        return A_norm

    def forward(self, features, mask=None):
        m = None if mask is None else mask[:, :, None, None].to(features.dtype)
        res = self.residual(features)
        A_norm = self._normalized_A()

        x = self.linear(features)

        out = 0
        for k in range(self.number_matrix):
            out = out + torch.einsum("vw,btwc->btvc", A_norm[k], x)
        out = out / self.number_matrix

        x = self.act(out)

        if m is not None:
            x = x * m

        # x = self.tcn(x)
        x = self.drop(x)

        # x = x + res

        # if m is not None:
            # x = x * m
        return x


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
            out = out + torch.einsum("vw,btwc->btvc", A_norm[k], feat)
        out = out / self.p

        return out, A_raw


class SelfPacingDroppingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_nodes, base_adjacency,
                 decouple_p=4, drop=True, use_tcn=True,
                 temporal_kernel_size=9, dropout=0.1):
        super().__init__()
        if temporal_kernel_size <= 0 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd number")
        self.gcn = DecoupledGCN(in_ch, out_ch, num_nodes, base_adjacency, decouple_p)
        self.use_tcn = use_tcn
        self.tcn = TemporalConv(out_ch, kernel_size=temporal_kernel_size if use_tcn else 1,
                                dropout=dropout)
        self.act = nn.GELU()

    def forward(self, x, mask=None):
        valid = None if mask is None else mask.to(device=x.device, dtype=torch.bool)[:, :, None, None]
        if valid is not None:
            x = x.masked_fill(~valid, 0)
        feat, _ = self.gcn(x)
        if valid is not None:
            feat = feat.masked_fill(~valid, 0)
        if self.use_tcn:
            feat = self.act(self.tcn(feat))
        if valid is not None:
            feat = feat.masked_fill(~valid, 0)
        return feat


class SPDStack(nn.Module):
    """Stack of decoupled GCN blocks, optionally with temporal convolution.

    Group/drop arguments are retained for compatibility; no node dropping is implemented.
    channels includes the input channel count.
    """
    def __init__(self, channels, num_nodes, base_adjacency, groups=None,
                 num_drop_per_group=1, decouple_p=4, use_tcn=True,
                 temporal_kernel_size=9, dropout=0.1):
        super().__init__()
        if len(channels) < 2 or any(c <= 0 for c in channels) or decouple_p <= 0:
            raise ValueError("Need positive input/output channels and decouple_p")
        self.blocks = nn.ModuleList([
            SelfPacingDroppingBlock(
                channels[i], channels[i + 1], num_nodes, base_adjacency,
                decouple_p=decouple_p, use_tcn=use_tcn,
                temporal_kernel_size=temporal_kernel_size, dropout=dropout,
            ) for i in range(len(channels) - 1)
        ])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x


class _GCNTemporalEncoderLayer(nn.Module):

    def __init__(self, hidden_dim, num_nodes, adjacency, nhead, dim_feedforward, dropout):
        super().__init__()
        self.nhead = nhead
        self.head_dim = num_nodes * hidden_dim // nhead
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.gcn_q = GCNBlock(hidden_dim, hidden_dim, num_nodes, adjacency, dropout=dropout)
        self.gcn_k = GCNBlock(hidden_dim, hidden_dim, num_nodes, adjacency, dropout=dropout)
        self.gcn_v = GCNBlock(hidden_dim, hidden_dim, num_nodes, adjacency, dropout=dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attention_drop = nn.Dropout(dropout)
        self.residual_drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, hidden_dim), )

    def forward(self, x, video_mask):
        batch, frames, nodes, channels = x.shape
        valid = video_mask[:, :, None, None]
        normalized = self.norm1(x).masked_fill(~valid, 0)

        def project(block):
            # Flatten only after GCN: each frame remains one attention token.
            projected = block(normalized, video_mask)
            return projected.reshape(batch, frames, self.nhead, self.head_dim).transpose(1, 2)

        q, k, v = project(self.gcn_q), project(self.gcn_k), project(self.gcn_v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        key_mask = video_mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        # Explicit zeroing also handles samples with no valid frames.
        attention = scores.softmax(dim=-1).masked_fill(~key_mask, 0)
        attended = torch.matmul(self.attention_drop(attention), v)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, frames, nodes, channels)
        x = (x + self.residual_drop(self.out_proj(attended))).masked_fill(~valid, 0)
        x = (x + self.residual_drop(self.ffn(self.norm2(x)))).masked_fill(~valid, 0)
        return x

class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.1,
        channels=(64, 64, 128, 128),
        num_nodes=_NUM_NODE,
        base_adjacency=None,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.attention = GCN_MultiHeadAttention(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_head=n_head,
            dropout=dropout,
            channels=channels,
            num_nodes=num_nodes,
            base_adjacency=base_adjacency,
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, video_mask=None):
        # x: (B, T, N, hidden_dim)
        if video_mask is None:
            video_mask = torch.ones(
                x.shape[:2], dtype=torch.bool, device=x.device
            )
        else:
            video_mask = video_mask.to(
                device=x.device, dtype=torch.bool
            )
            if tuple(video_mask.shape) != tuple(x.shape[:2]):
                raise ValueError("video_mask phải có shape (B, T)")

        valid = video_mask[:, :, None, None]
        x = x.masked_fill(~valid, 0)

        # Pre-norm attention.
        context = self.attention(self.norm1(x), video_mask)
        # context: (B, T, hidden_dim)
        # Cộng ngữ cảnh frame vào từng khớp, giữ residual từng khớp.
        x = x + self.dropout1(context).unsqueeze(2)
        x = x.masked_fill(~valid, 0)

        # FFN trên chiều đặc trưng của từng khớp.
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x.masked_fill(~valid, 0)

class GCN_Transformer(nn.Module):
    def __init__(
        self,
        input_dim=_COORD_DIM,
        hidden_dim=128,
        num_encoder_layers=3,
        n_head=8,
        dim_feedforward=128 * 8,
        dropout=0.1,
        max_seq_len=500,
        num_classes=1000,
        channels=(64, 64, 128, 128),
        base_adjacency=None,
        num_nodes=_NUM_NODE,
    ):
        super().__init__()

        if num_encoder_layers <= 0 or max_seq_len <= 0:
            raise ValueError("Số layer và max_seq_len phải lớn hơn 0")

        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.max_seq_len = max_seq_len

        # Chỉ chiếu channel của mỗi khớp, không flatten các khớp.
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(
            max_seq_len, dtype=torch.float32
        ).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )

        pe = torch.zeros(max_seq_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(
            position * frequency[:pe[:, 1::2].shape[1]]
        )
        self.register_buffer(
            "position_encoding", pe[None, :, None, :]
        )  # (1, max_seq_len, 1, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                channels=channels,
                num_nodes=num_nodes,
                base_adjacency=base_adjacency,
            )
            for _ in range(num_encoder_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, video_mask=None, labels=None):
        # Hỗ trợ (B, T, N*C) và (B, T, N, C).
        if x.ndim == 3:
            if x.shape[-1] != self.num_nodes * self.input_dim:
                raise ValueError("Sai chiều đặc trưng flatten")
            x = x.reshape(
                *x.shape[:2], self.num_nodes, self.input_dim
            )

        if x.ndim != 4 or tuple(x.shape[2:]) != (
            self.num_nodes, self.input_dim
        ):
            raise ValueError("x phải có shape (B, T, num_nodes, input_dim)")

        B, T = x.shape[:2]
        if not 1 <= T <= self.max_seq_len:
            raise ValueError("T phải nằm trong [1, max_seq_len]")

        if video_mask is None:
            video_mask = torch.ones(
                B, T, dtype=torch.bool, device=x.device
            )
        else:
            video_mask = video_mask.to(
                device=x.device, dtype=torch.bool
            )
            if tuple(video_mask.shape) != (B, T):
                raise ValueError("video_mask phải có shape (B, T)")

        valid = video_mask[:, :, None, None]
        x = x.masked_fill(~valid, 0)
        x = self.input_projection(x)
        x = self.dropout(
            x + self.position_encoding[:, :T].to(dtype=x.dtype)
        )
        x = x.masked_fill(~valid, 0)

        for block in self.transformer_blocks:
            x = block(x, video_mask)

        x = self.final_norm(x).masked_fill(~valid, 0)
        x = x.mean(dim=2)  # (B, T, hidden_dim)

        # Mean pooling chỉ trên các frame hợp lệ.
        count = video_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = x.sum(dim=1) / count.to(dtype=x.dtype)

        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"logits": logits, "loss": loss}

class GCN_MultiHeadAttention(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, n_head=8, dropout=0.1,
                 channels=(64, 64, 128, 128), num_nodes=_NUM_NODE,
                 base_adjacency=None):
        super().__init__()
        if hidden_dim <= 0 or n_head <= 0 or hidden_dim % n_head:
            raise ValueError("hidden_dim must be positive and divisible by n_head")
        if not channels or channels[-1] != hidden_dim:
            raise ValueError("channels[-1] must equal hidden_dim")
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.head_dim = hidden_dim // n_head
        if base_adjacency is None:
            base_adjacency = build_adjacency_from_edges(HAND_EDGES, num_nodes)
        if tuple(base_adjacency.shape) != (num_nodes, num_nodes):
            raise ValueError("Invalid adjacency shape")
        dimensions = (input_dim, *channels)
        def make_projection():
            return nn.ModuleList([
                GCNBlock(dimensions[i], dimensions[i + 1], num_nodes,
                         base_adjacency, dropout=dropout)
                for i in range(len(dimensions) - 1)
            ])
        self.W_query = make_projection()
        self.W_key = make_projection()
        self.W_value = make_projection()
        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, video_mask=None):
        B, T, N, C = x.shape
        if N != self.num_nodes or C != self.input_dim or T == 0:
            raise ValueError("Invalid attention input shape")
        if video_mask is None:
            video_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        else:
            video_mask = video_mask.to(device=x.device, dtype=torch.bool)
            if tuple(video_mask.shape) != (B, T):
                raise ValueError("video_mask must have shape (B, T)")
        x = x.masked_fill(~video_mask[:, :, None, None], 0)
        def project(blocks):
            features = x
            for block in blocks:
                features = block(features, video_mask)
            return features.mean(dim=2).reshape(
                B, T, self.n_head, self.head_dim
            ).transpose(1, 2)
        Q, K, V = project(self.W_query), project(self.W_key), project(self.W_value)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_mask = video_mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1).masked_fill(~key_mask, 0)
        context = self.dropout(weights) @ V
        context = context.transpose(1, 2).contiguous().reshape(B, T, self.hidden_dim)
        return self.out_projection(context).masked_fill(~video_mask[:, :, None], 0)
