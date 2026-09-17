import torch
import torch.nn as nn
import torch.nn.functional as F
from config import _N_POSE, _N_HAND, _NUM_NODE, _REMOVE_POSE_IDX


_HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

_POSE_BONES = [
    (0, 1), (0, 2),
    (1, 0), (1, 2),
    (2, 1), (2, 3),
    (3, 1), (3, 7),
    (4, 0), (4, 5),
    (5, 4), (5, 6),
    (6, 5), (6, 8),
    (7, 3),
    (8, 6),
    (9, 10),
    (10, 9),
    (11, 9), (11, 12), (11, 13),
    (12, 10), (12, 11), (12, 14),
    (13, 11), (13, 15),
    (14, 12), (14, 16),
    (15, 13), (15, 17), (15, 21), (15, 19),
    (16, 14), (16, 18), (15, 22), (15, 20),
    (17, 15), (17, 19),
    (18, 16), (18, 20),
    (19, 15), (19, 17),
    (20, 16), (20, 18),
    (21, 15),
    (22, 16),
    (23, 11), (23, 24),
    (24, 12), (24, 22),
]

_LEFT_WRIST = _N_POSE
_RIGHT_WRIST = _LEFT_WRIST + _N_HAND

_LEFT_HAND_EDGES = [(_LEFT_WRIST + i, _LEFT_WRIST + j) for i, j in _HAND_BONES]
_RIGHT_HAND_EDGES = [(_RIGHT_WRIST + i, _RIGHT_WRIST + j) for i, j in _HAND_BONES]

CUSTOM_EDGES = [
    # (9, 11),
    # (10, 12),
    (0, 11), (0, 12)
]

remove_set = set(_REMOVE_POSE_IDX)

filtered_bones = [
    (a, b) for (a, b) in _POSE_BONES
    if a not in remove_set and b not in remove_set
]

all_nodes = sorted(set(n for bone in _POSE_BONES for n in bone))
remaining_nodes = sorted(n for n in all_nodes if n not in remove_set)

old_to_new = {old: new for new, old in enumerate(remaining_nodes)}

remapped_bones = [
    (old_to_new[a], old_to_new[b])
    for (a, b) in filtered_bones
]

FULL_BODY_EDGES = (
_POSE_BONES + CUSTOM_EDGES
+ _LEFT_HAND_EDGES
+ _RIGHT_HAND_EDGES
)


def build_adjacency_from_edges(edges, num_nodes):
    A = torch.eye(num_nodes)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A

def build_adjacency():
    return build_adjacency_from_edges(FULL_BODY_EDGES, _NUM_NODE)

def _normalize_adjacency(A):
    """D^{-1/2} A D^{-1/2}, dùng chung cho từng thành phần A_in / A_out."""
    deg = A.sum(dim=1).clamp(min=1e-6)
    d_inv_sqrt = torch.diag(deg.pow(-0.5))
    return d_inv_sqrt @ A @ d_inv_sqrt

def _build_hand_group_edges(offset):
    edges = [
        (offset + 0, offset + 1),
        (offset + 1, offset + 2),
    ]
    edges += [(offset + 2 + i, offset + 2 + j) for i, j in _HAND_BONES]
    return edges

class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=9, stride=1, dropout=0.1):
        super().__init__()

        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            out_channels=channels,
            in_channels=channels,
            kernel_size=(kernel_size, 1), # (out_channels, in_channels, kernel_H, kernel_W)
            padding=(pad, 0),
            stride=(stride, 1),
            bias=False
        )
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

class GraphConv(nn.Module):

    def __init__(self, in_ch, out_ch, num_nodes, base_adjacency, decouple_p=8):
        super().__init__()

        # asj = normalize_adjacency(base_adjacency)
        self.V = num_nodes

        # self.register_buffer("adj", base_adjacency)  # (N, N)
        self.linear = nn.Linear(in_ch, out_ch)

        self.register_buffer("I", torch.eye(self.V))
        # self.register_buffer('A', base_adjacency)

        self.learnable_A = nn.Parameter(
            torch.tensor(base_adjacency, dtype=torch.float32)
        )
        self.decouple_p = decouple_p

    def _raw_A(self):
        A = self.I + self.learnable_A
        A = 0.5 * (A + A.transpose(-1, -2))  # ép đối xứng
        A = F.relu(A)  # ép không âm
        return A

    def _normalized_A(self, A_raw):
        deg = A_raw.sum(-1).clamp(min=1e-6)          # (p, V)
        d_inv_sqrt = deg.pow(-0.5)
        D_inv_sqrt = torch.diag_embed(d_inv_sqrt)     # (p, V, V)

        return D_inv_sqrt @ A_raw @ D_inv_sqrt        # (p, V, V)

    def forward(self, x):
        x = self.linear(x)

        A_raw = self._raw_A()  # (p, V, V)
        A_norm = self._normalized_A(A_raw)

        out = torch.einsum('vw,btwc->btvc', A_norm, x)

        return out, A_raw

class GrapConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_nodes, base_adjacency):
        super().__init__()

        self.gcn = GraphConv(
            in_ch,
            out_ch,
            num_nodes,
            base_adjacency
        )

        self.tcn = TemporalConv(
            channels=out_ch,
            kernel_size=9
        )

        self.act = nn.GELU()

        self.residual = nn.Linear(in_ch, out_ch)
        if in_ch != out_ch:
            self.residual = nn.Linear(in_ch, out_ch)
        else:
            self.residual = nn.Identity()

    def forward(self, x, mask = None):
        residual = self.residual(x)

        x, _ = self.gcn(x)
        x = self.act(x)

        x = self.tcn(x)
        x = self.act(x + residual)

        if mask is not None:
            x = x * mask[:, :, None, None].to(x.dtype)

        return x

class GCN_Block(nn.Module):
    def __init__(self, in_ch, out_ch, num_nodes, base_adjacency):
        super().__init__()

        self.num_nodes = num_nodes

        self.register_buffer("adj", base_adjacency)  # (N, N)
        self.linear = nn.Linear(in_ch, out_ch)

        self.register_buffer("I", torch.eye(self.num_nodes))
        self.register_buffer('A', base_adjacency)

        self.learnable_A = nn.Parameter(
            torch.tensor(base_adjacency, dtype=torch.float32)
        )

        self.act = nn.GELU()
        self.tcn = TemporalConv(out_ch)
        self.residual = nn.Identity() if in_ch == out_ch else nn.Linear(in_ch, out_ch)

    def _raw_A(self):
        return  self.I + self.learnable_A

    def _normalized_A(self, A_raw):
        deg = A_raw.sum(-1).clamp(min=1e-6)
        d_inv_sqrt = deg.pow(-0.5)
        D_inv_sqrt = torch.diag_embed(d_inv_sqrt)

        return D_inv_sqrt @ A_raw @ D_inv_sqrt

    def forward(self, x, mask = None):
        res = self.residual(x)
        x = self.linear(x)

        A_raw = self._raw_A()
        A_norm = self._normalized_A(A_raw)

        x = torch.einsum('vw,btwc->btvc', A_norm, x)
        x = self.act(x)

        x = self.tcn(x) + res
        x = self.act(x)

        if mask is not None:
            x = x * mask[:, :, None, None].to(x.dtype)

        return x