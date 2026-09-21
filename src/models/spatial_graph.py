import torch
import torch.nn as nn
import torch.nn.functional as F
from config import _N_POSE, _N_HAND, _NUM_NODE, _REMOVE_POSE_IDX


_N_POSE = 33
_N_HAND = 21
# _REMOVE_POSE_IDX defined elsewhere

_HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Each undirected edge listed once
_POSE_BONES = [
    # face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # torso / arms
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    # hands (pose-model fingers)
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
]

CUSTOM_EDGES = [(0, 11), (0, 12)]   # nose -> shoulders

remove_set = set(_REMOVE_POSE_IDX)

# Keep every pose node that isn't removed (not just those appearing in edges)
remaining_nodes = [n for n in range(_N_POSE) if n not in remove_set]
old_to_new = {old: new for new, old in enumerate(remaining_nodes)}
n_pose_kept = len(remaining_nodes)

def _remap(edges):
    return [(old_to_new[a], old_to_new[b])
            for a, b in edges
            if a in old_to_new and b in old_to_new]

pose_edges   = _remap(_POSE_BONES)
custom_edges = _remap(CUSTOM_EDGES)

_LEFT_WRIST  = n_pose_kept
_RIGHT_WRIST = _LEFT_WRIST + _N_HAND

_LEFT_HAND_EDGES  = [(_LEFT_WRIST + i,  _LEFT_WRIST + j)  for i, j in _HAND_BONES]
_RIGHT_HAND_EDGES = [(_RIGHT_WRIST + i, _RIGHT_WRIST + j) for i, j in _HAND_BONES]

# Optional: link pose wrists to hand-model wrists
wrist_links = []
if 15 in old_to_new: wrist_links.append((old_to_new[15], _LEFT_WRIST))
if 16 in old_to_new: wrist_links.append((old_to_new[16], _RIGHT_WRIST))

FULL_BODY_EDGES = (
    pose_edges + custom_edges
    + _LEFT_HAND_EDGES + _RIGHT_HAND_EDGES
    + wrist_links
)

# Sanity check
n_nodes = n_pose_kept + 2 * _N_HAND
assert all(0 <= a < n_nodes and 0 <= b < n_nodes for a, b in FULL_BODY_EDGES)
assert len({tuple(sorted(e)) for e in FULL_BODY_EDGES}) == len(FULL_BODY_EDGES), "duplicate edges"


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

class GCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_nodes, base_adjacency, dropout=0.0):
        super().__init__()
        self.register_buffer("I", torch.eye(num_nodes))
        self.register_buffer("A", base_adjacency.float())
        # học phần lệch so với A gốc, khởi tạo bằng 0 -> ổn định hơn
        self.A_delta = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        self.linear = nn.Sequential(nn.Linear(in_ch, out_ch), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(out_ch))

        self.tcn = TemporalConv(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Identity() if in_ch == out_ch else nn.Linear(in_ch, out_ch)

    def _normalized_A(self):
        A = F.relu(self.A + self.A_delta * (self.A > 0)) + self.I
        d = A.sum(-1).clamp(min=1e-6).pow(-0.5)
        return d[:, None] * A * d[None, :]

    def forward(self, features, mask=None):
        m = None if mask is None else mask[:, :, None, None].to(features.dtype)

        res = self.residual(features)
        x = self.linear(features)

        x = torch.einsum('vw,btwc->btvc', self._normalized_A(), x)
        x = self.act(x)

        if m is not None:
            x = x * m
        x = self.tcn(x)
        x = self.drop(x)

        x = x + res
        if m is not None:
            x = x * m
        return x