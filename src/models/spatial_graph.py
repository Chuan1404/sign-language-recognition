import math
import os

import torch
from torch import nn
from config import _N_HAND, _REMOVE_POSE_IDX, _NUM_NODE, _COORD_DIM, _N_POSE

_HAND_BONES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10),
               (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
               (5, 9), (9, 13), (13, 17)]

_POSE_BONES = [
    # (0, 9), (0, 10), (9, 10),  # Nose and mouth corners
    # (0, 11), (0, 12),
    (11, 12), (11, 13), (12, 14),
]

CUSTOM_EDGES = []

remove_set = set(_REMOVE_POSE_IDX)

remaining_nodes = [n for n in range(33) if n not in remove_set]
old_to_new = {old: new for new, old in enumerate(remaining_nodes)}
n_pose_kept = len(remaining_nodes)


def _remap(edges):
    return [(old_to_new[a], old_to_new[b]) for a, b in edges if a in old_to_new and b in old_to_new]


POSE_EDGES = _remap(_POSE_BONES)

_LEFT_WRIST = 0
_RIGHT_WRIST = _LEFT_WRIST + _N_HAND

_LEFT_HAND_EDGES = [(_LEFT_WRIST + i, _LEFT_WRIST + j) for i, j in _HAND_BONES]
_RIGHT_HAND_EDGES = [(_RIGHT_WRIST + i, _RIGHT_WRIST + j) for i, j in _HAND_BONES]

HAND_EDGES = _LEFT_HAND_EDGES + _RIGHT_HAND_EDGES

_ELBOW_WRIST_EDGES = [
    (old_to_new[elbow], n_pose_kept + wrist)
    for elbow, wrist in ((13, _LEFT_WRIST), (14, _RIGHT_WRIST))
    if elbow in old_to_new
]
LOCAL_BODY_EDGES = POSE_EDGES + [(n_pose_kept + i, n_pose_kept + j) for i, j in HAND_EDGES]
FULL_BODY_EDGES = LOCAL_BODY_EDGES

assert all(0 <= a < _NUM_NODE and 0 <= b < _NUM_NODE for a, b in FULL_BODY_EDGES)
assert len({tuple(sorted(e)) for e in HAND_EDGES}) == len(HAND_EDGES), ("duplicate edges")


def build_adjacency_from_edges(edges, num_nodes):
    A = torch.eye(num_nodes)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    return A


def build_global_graph():
    from .vsnet_backbone.graph import EDGES

    if _N_HAND != 21:
        raise ValueError("VSNet grouping needs two 21-point hands")

    left = n_pose_kept
    right = left + _N_HAND
    paired_pose = [(a, b) for a, b in ((11, 12), (13, 14)) if a in old_to_new and b in old_to_new]
    groups = (
        (*[old_to_new[a] for a, _ in paired_pose], left, left + 2, left + 3, left + 4),
        (*[old_to_new[b] for _, b in paired_pose], right, right + 2, right + 3, right + 4),
        tuple(range(left + 5, left + 13)),
        tuple(range(left + 13, left + 21)),
        tuple(range(right + 5, right + 13)),
        tuple(range(right + 13, right + 21)),
    )
    adjacency = build_adjacency_from_edges(EDGES["wlasl_44"], 44)
    # Slots 0/6 are shoulders and 1/7 are elbows in the original 44-node ordering.
    excluded = {slot + offset for slot, pair in enumerate(((11, 12), (13, 14)))
                if pair not in paired_pose for offset in (0, 6)}
    kept = [slot for slot in range(44) if slot not in excluded]
    return groups, adjacency[kept][:, kept]


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
        x = self.conv(x.permute(0, 3, 1, 2))
        return self.drop(self.bn(x)).permute(0, 2, 3, 1)


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


class Node_Attention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.scale = channels ** -0.5

    def forward(self, features, video_mask=None, return_attention=False):
        if video_mask is not None:
            video_mask = video_mask.to(device=features.device, dtype=torch.bool)
            features = features.masked_fill(~video_mask[:, :, None, None], 0)
        q, k, v = self.query(features), self.key(features), self.value(features)
        weights = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        pooled = (weights @ v).mean(dim=-2)
        if video_mask is not None:
            pooled = pooled.masked_fill(~video_mask[:, :, None], 0)
            weights = weights.masked_fill(~video_mask[:, :, None, None], 0)
        if return_attention:
            return pooled, weights
        return pooled

class V_GCN(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes, base_adjacency, decouple_p=1, learnable_adjacency=False):
        super().__init__()

        self.V = num_nodes
        self.p = decouple_p
        self.phi = nn.Linear(in_channels, out_channels)

        base_adjacency = base_adjacency.float()
        self.register_buffer("I", torch.eye(self.V))

        adjacency = base_adjacency.unsqueeze(0).repeat(self.p, 1, 1).clone()
        self.register_buffer("A_in", adjacency)

        # if learnable_adjacency:
        #     self.A_in = nn.Parameter(adjacency * 1e-3)
        # else:
        #     self.register_buffer("A_in", adjacency)

    def _normalized_A(self, A_raw):
        deg = A_raw.sum(-1).clamp(min=1e-6)
        d_inv_sqrt = deg.pow(-0.5)
        D_inv_sqrt = torch.diag_embed(d_inv_sqrt)

        return D_inv_sqrt @ A_raw @ D_inv_sqrt

    def forward(self, feature):
        feat = self.phi(feature)

        A_raw = self.I.unsqueeze(0) + self.A_in
        A_norm = self._normalized_A(A_raw)

        out = torch.einsum("pvw,btwc->btpvc", A_norm, feat)  # (B, T, P, V, C)
        out = out.mean(dim=2)
        return out, A_raw

class V_GCN_Block(nn.Module):

    def __init__(self, in_ch, out_ch, num_nodes, base_adjacency, decouple_p=4, drop=True, use_tcn=False,
                 temporal_kernel_size=9, dropout=0.1, residual=False):

        super().__init__()
        self.gcn = V_GCN(in_ch, out_ch, num_nodes, base_adjacency, decouple_p)

        self.use_tcn = use_tcn
        self.tcn = TemporalConv(out_ch, kernel_size=temporal_kernel_size if use_tcn else 1, dropout=dropout)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Identity() if in_ch == out_ch else nn.Linear(in_ch, out_ch, bias=False)) if residual else None

    def forward(self, feature, mask=None):
        valid = mask.to(device=feature.device, dtype=torch.bool)
        valid = valid[:, :, None, None]

        feature = feature.masked_fill(~valid, 0)

        feat, _ = self.gcn(feature)
        feat = feat.masked_fill(~valid, 0)

        if self.use_tcn:
            feat = self.act(self.tcn(feat))

        else:
            feat = self.act(feat)

        feat = self.dropout(feat)

        if self.residual is not None:
            feat = feat + self.residual(feature)

        return feat.masked_fill(~valid, 0)

class V_GCN_Block_V2(nn.Module):

    def __init__(self, in_ch, out_ch, num_nodes, base_adjacency, decouple_p=4, drop=True, use_tcn=False,
                 temporal_kernel_size=9, dropout=0.1, residual=False):

        super().__init__()
        self.gcn = V_GCN_V2(in_ch, out_ch, num_nodes, base_adjacency, decouple_p)

        self.use_tcn = use_tcn
        self.tcn = TemporalConv(out_ch, kernel_size=temporal_kernel_size if use_tcn else 1, dropout=dropout)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Identity() if in_ch == out_ch else nn.Linear(in_ch, out_ch, bias=False)) if residual else None

    def forward(self, feature, mask=None):
        valid = mask.to(device=feature.device, dtype=torch.bool)
        valid = valid[:, :, None, None]

        feature = feature.masked_fill(~valid, 0)

        feat, _ = self.gcn(feature)
        feat = feat.masked_fill(~valid, 0)

        if self.use_tcn:
            feat = self.act(self.tcn(feat))

        else:
            feat = self.act(feat)

        feat = self.dropout(feat)

        if self.residual is not None:
            feat = feat + self.residual(feature)

        return feat.masked_fill(~valid, 0)

class V_GCN_V2(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes, base_adjacency, decouple_p=1, learnable_adjacency=False,
                 mirror_input_x=False, dropout = 0.1):
        super().__init__()

        self.V = num_nodes
        self.p = decouple_p

        self.learnable_adjacency = learnable_adjacency
        self.mirror_input_x = mirror_input_x
        self.phi = nn.Sequential(nn.Linear(in_channels, out_channels), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(out_channels), )

        base_adjacency = base_adjacency.float()
        self.register_buffer("I", torch.eye(self.V))
        adjacency = base_adjacency.unsqueeze(0).repeat(self.p, 1, 1).clone()

        if learnable_adjacency:
            self.A_in = nn.Parameter(adjacency * 1e-3)
        else:
            self.register_buffer("A_in", adjacency)

        # Keep full-layout indices so both branches return (B, T, V, C).
        mirror = torch.zeros_like(base_adjacency)
        if self.V == _NUM_NODE:
            # groups, _ = build_global_graph()
            # # Each source contributes equally within the partner's one-hop neighborhood.
            # neighborhood = base_adjacency.ne(0).to(base_adjacency.dtype)
            # neighborhood.fill_diagonal_(1)
            # neighborhood = neighborhood / neighborhood.sum(dim=1, keepdim=True)
            # for left, right in ((0, 1), (2, 4), (3, 5)):
            #     a, b = list(groups[left]), list(groups[right])
            #     mirror[a] = neighborhood[b]
            #     mirror[b] = neighborhood[a]
            pairs = _remap([(13, 14)])
            pairs.extend((n_pose_kept + i, n_pose_kept + _N_HAND + i) for i in range(_N_HAND))
            for left, right in pairs:
                mirror[left, right] = mirror[right, left] = 1.0

        elif self.V == 2 * _N_HAND:
            left = torch.arange(_N_HAND, device=mirror.device)
            right = left + _N_HAND
            mirror[left, right] = 1.0
            mirror[right, left] = 1.0

        self.register_buffer("mirror_A", mirror, persistent=False)
        self.register_buffer("has_partner", mirror.any(dim=1)[:, None], persistent=False)

    def _raw_A(self):
        return self.I.unsqueeze(0) + self.A_in

    def _normalized_A(self, A_raw):
        deg = A_raw.sum(-1).clamp(min=1e-6)
        d_inv_sqrt = deg.pow(-0.5)
        D_inv_sqrt = torch.diag_embed(d_inv_sqrt)

        return D_inv_sqrt @ A_raw @ D_inv_sqrt

    def forward(self, feature):
        feat = self.phi(feature)

        if not self.learnable_adjacency:
            # base_adjacency already contains self-loops. No learned edges or QK scores.
            A_raw = self.A_in
            A_norm = self._normalized_A(A_raw)
            local = torch.einsum("pvw,btwc->btpvc", A_norm, feat).mean(dim=2)
        else:
            A_raw = self.I + self.A_in
            A_norm = self._normalized_A(A_raw)
            local = torch.einsum("pvw,btwc->btpvc", A_norm, feat)  # (B, T, P, V, C)
            local = local.mean(dim=2)

        mirror_feat = feat

        opposite = torch.einsum("vw,btwc->btvc", self.mirror_A, mirror_feat)
        out = torch.where(self.has_partner, (local + opposite) * 0.5, local)
        return out, A_raw
