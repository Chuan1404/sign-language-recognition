import torch
import torch.nn as nn
from config import _N_POSE, _N_HAND, _NUM_NODE, _REMOVE_POSE_IDX

_HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
               (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13),
               (13, 17), ]

# Each undirected edge listed once
_POSE_BONES = [  # face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),  # torso / arms
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),  # hands (pose-model fingers)
    (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20), ]

CUSTOM_EDGES = [(0, 11), (0, 12)]  # nose -> shoulders

remove_set = set(_REMOVE_POSE_IDX)

# Keep every pose node that isn't removed (not just those appearing in edges)
remaining_nodes = [n for n in range(_N_POSE) if n not in remove_set]
old_to_new = {old: new for new, old in enumerate(remaining_nodes)}
n_pose_kept = len(remaining_nodes)


def _remap(edges):
    return [(old_to_new[a], old_to_new[b]) for a, b in edges if a in old_to_new and b in old_to_new]


pose_edges = _remap(_POSE_BONES)
custom_edges = _remap(CUSTOM_EDGES)

_LEFT_WRIST = 0
_RIGHT_WRIST = _LEFT_WRIST + _N_HAND

_LEFT_HAND_EDGES = [(_LEFT_WRIST + i, _LEFT_WRIST + j) for i, j in _HAND_BONES]
_RIGHT_HAND_EDGES = [(_RIGHT_WRIST + i, _RIGHT_WRIST + j) for i, j in _HAND_BONES]

# Optional: link pose wrists to hand-model wrists
wrist_links = []
if 15 in old_to_new: wrist_links.append((old_to_new[15], _LEFT_WRIST))
if 16 in old_to_new: wrist_links.append((old_to_new[16], _RIGHT_WRIST))

FULL_BODY_EDGES = (_LEFT_HAND_EDGES + _RIGHT_HAND_EDGES + wrist_links)

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
    return build_adjacency_from_edges(FULL_BODY_EDGES, _N_HAND * 2)


def _normalize_adjacency(A):
    """D^{-1/2} A D^{-1/2}, dùng chung cho từng thành phần A_in / A_out."""
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
                              padding=(pad, 0), stride=(stride, 1), bias=False)
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
    def __init__(self, in_ch, out_ch, num_nodes, base_adjacency, dropout=0.0):
        super().__init__()

        self.register_buffer("I", torch.eye(num_nodes))
        self.register_buffer("A", base_adjacency.float())
        self.A_delta = self.A_delta = nn.ParameterList(
            [nn.Parameter(torch.zeros(num_nodes, num_nodes)) for _ in range(1)])

        self.linear = nn.Sequential(nn.Linear(in_ch, out_ch), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(out_ch))

        # self.tcn = TemporalConv(out_ch, kernel_size=9)
        self.tcn = MultiScaleTemporalConv(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Identity() if in_ch == out_ch else nn.Linear(in_ch, out_ch)

    def _normalized_A(self):
        delta_mean = torch.stack(list(self.A_delta), dim=0).mean(dim=0)

        A = self.A + delta_mean * (self.A > 0) + self.I
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
        self.tcn = TemporalConv(out_ch, kernel_size=1)

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

# class SimpleTCN(nn.Module):
#     def __init__(self, channels, kernel_size=9):
#         super().__init__()
#         pad = (kernel_size - 1) // 2
#         self.conv = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0))
#         self.bn = nn.BatchNorm2d(channels)
#         self.act = nn.GELU()
#
#     def forward(self, x):
#         # x: (B, T, V, C) -> Conv2d cần (B, C, T, V)
#         x = x.permute(0, 3, 1, 2)
#         x = self.act(self.bn(self.conv(x)))
#         return x.permute(0, 2, 3, 1)