"""
VSNet — triển khai lại theo:
"VSNet: Focusing on the Linguistic Characteristics of Sign Language" (CVPR 2025)
Li, Chen, Li, Pu, Jin, Ren.

KIẾN TRÚC (theo Section 3 của paper):
    Input (T, V=44, C) đã chuẩn hoá theo node V0
      -> SPD (Self-Pacing Dropping Block) x3:
             mỗi block = Decoupled Graph Conv (Eq.2) + TCN + Weak Joints
             Dropping (Eq.3) — 2 block đầu vừa học vừa loại khớp yếu, block
             thứ 3 chỉ trích đặc trưng, không loại khớp nữa.
      -> gộp các khớp còn lại trong từng nhóm thành G=6 "VS token" (Visual
         Symbol) bằng attention-pool
      -> VSformer x8: mỗi block chia kênh làm 4 phần bằng nhau, xử lý song
         song bằng SPA (Spatial Pairing Attention, Eq.5), TWA (Temporal
         Window Attention, Eq.6 — cả TWA-s và TWA-l), GCN, TCN, rồi nối lại
      -> Downsample (Conv kernel=7, stride=2 theo T) + BatchNorm
      -> Average Pooling (mean theo CẢ T và G) -> Fully Connected -> logits

============================== GHI CHÚ QUAN TRỌNG ==============================
Paper mô tả layout 44 khớp và ranh giới các nhóm khớp (Fig.4, Fig.5 — skeleton
type-1/2/3) HOÀN TOÀN bằng hình minh hoạ, KHÔNG có bảng chỉ số cụ thể trong
text. Vì chỉ có bản PDF/OCR (không đọc được chi tiết hình vẽ), các phần sau
đây là SUY LUẬN HỢP LÝ dựa trên mô tả text, KHÔNG đảm bảo khớp 100% con số gốc
của paper — bạn cần đối chiếu lại với layout 44-điểm thật của mình:

    - Giả định layout 44 khớp: 2 điểm cánh tay (index 0-1) + 21 khớp tay trái
      (index 2-22, layout chuẩn: 0=cổ tay,1-4=cái,5-8=trỏ,9-12=giữa,13-16=áp
      út,17-20=út) + 21 khớp tay phải (index 23-43, cùng layout).
    - Giả định 6 nhóm khớp ban đầu (khớp với G=6 trong paper):
        group 0: cánh tay trái (1 điểm)
        group 1: cánh tay phải (1 điểm)
        group 2: tay trái - cổ tay+cái+trỏ (9 khớp)
        group 3: tay trái - giữa+áp út+út (12 khớp)
        group 4: tay phải - cổ tay+cái+trỏ (9 khớp)
        group 5: tay phải - giữa+áp út+út (12 khớp)
      (paper chỉ nói "grouping each finger and arm separately tạo quá nhiều
      nhóm" và cuối cùng SPA dùng đúng 3 cặp [arm, left, right] — nên nhóm
      6-phần ở trên được chọn sao cho gộp lại đúng khớp cặp SPA mô tả).
    - Công thức TWA (Eq.6, dùng phi1/phi2/phi3 chia kênh làm 3) được đơn giản
      hoá thành attention chuẩn (Q,K,V riêng) cho TWA-s và TWA-l — cùng mục
      đích (nắm bắt vận động ngắn hạn trong 1 window và dài hạn giữa các
      window), khác cách tham số hoá chi tiết.
    - Vị trí/số lần Downsample trong VSformer×8 không nêu rõ trong text (chỉ
      thấy 1 khối "Down Sample" trong Fig.3) — để `downsample_at` cấu hình
      được, mặc định downsample sau block thứ 3 và thứ 6 (giống các model
      khác trong codebase của bạn).
    - "Multi-Grouping Ensemble" (Section 3.4 — ensemble 3 kiểu group + bone
      data, trọng số theo top-1 accuracy) KHÔNG được implement ở đây vì đó
      là 1 pipeline train 4 model riêng rồi ensemble ở mức inference, không
      phải kiến trúc 1 model — có thể làm thêm nếu bạn cần.
==================================================================================
"""

import torch
import torch.nn as nn


# =============================================================================
# Thành phần dùng chung
# -----------------------------------------------------------------------------

class _TemporalConv(nn.Module):
    """TCN đơn giản nhất — conv 1D theo trục T, áp dụng độc lập cho từng node/kênh."""

    def __init__(self, channels, kernel_size=9, stride=1, dropout=0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, N, C) -> (B, C, T, N)
        x = x.permute(0, 3, 1, 2)
        x = self.drop(self.conv(x))
        x = self.bn(x)
        x = x.permute(0, 2, 3, 1)  # (B, T, N, C)
        return x


def _derive_next_groups(groups, arm_group_ids, drop_per_group):
    """
    Tính lại ranh giới nhóm SAU KHI weak-joint-dropping đã loại bớt khớp yếu
    trong mỗi nhóm — vì Cor (Eq.3) chỉ phụ thuộc ma trận A (tham số học được
    của model, KHÔNG phụ thuộc input x), thứ tự/khớp bị giữ lại trong mỗi
    nhóm là NHẤT QUÁN giữa các sample cùng 1 bước train, nên ranh giới nhóm
    ở tầng tiếp theo có thể tính trước (không cần biết batch cụ thể).
    """
    next_groups = []
    offset = 0
    for g_id, group in enumerate(groups):
        n_drop = 0 if g_id in arm_group_ids else drop_per_group
        remaining = max(len(group) - n_drop, 0)
        next_groups.append(list(range(offset, offset + remaining)))
        offset += remaining
    return next_groups


# =============================================================================
# SPD — Self-Pacing Dropping Block
# -----------------------------------------------------------------------------

class DecoupledGraphConv(nn.Module):
    """
    Eq.2:  F_out = (1/p) * sum_{k=0}^{p-1} D_k^{-1/2} A_k D_k^{-1/2} phi(F_in)

    A_k: p ma trận adjacency HỌC ĐƯỢC ĐỘC LẬP (không softmax/chuẩn hoá trước
    khi dùng để tính Cor — paper nói rõ "these connections are not normalized
    via softmax"), khởi tạo từ base_adjacency (I hoặc I+In+Out nếu bạn có sẵn
    đồ thị tự nhiên của skeleton).
    """

    def __init__(self, in_channels, out_channels, num_nodes, p=3, base_adjacency=None):
        super().__init__()
        self.p = p
        self.phi = nn.Linear(in_channels, out_channels)

        base = base_adjacency.clone() if base_adjacency is not None else torch.eye(num_nodes)
        self.A = nn.Parameter(base.unsqueeze(0).repeat(p, 1, 1))   # (p, N, N)

    def _normalize(self, A):
        deg = A.sum(dim=-1).clamp(min=1e-6)
        deg_inv_sqrt = deg.pow(-0.5)
        D = torch.diag_embed(deg_inv_sqrt)
        return D @ A @ D

    def forward(self, x):
        # x: (B, T, N, C_in) -> (B, T, N, C_out)
        feat = self.phi(x)
        out = 0.0
        for k in range(self.p):
            A_norm = self._normalize(self.A[k])
            out = out + torch.einsum("nm,btmc->btnc", A_norm, feat)
        out = out / self.p
        return out, self.A   # trả cả A (chưa chuẩn hoá) để tính Cor (Eq.3)


def compute_correlation(A):
    """
    Eq.3:  Cor_i = sum_j max(A_0(i,j), A_1(i,j), ..., A_p(i,j))
    A: (p, N, N) -> Cor: (N,). Cor thấp = khớp càng "độc lập" (weak joint).
    """
    max_over_k, _ = A.max(dim=0)     # (N, N)
    return max_over_k.sum(dim=-1)     # (N,)


class WeakJointsDropping(nn.Module):
    """
    Với mỗi nhóm khớp, loại `drop_per_group` khớp có Cor THẤP NHẤT (độc lập
    nhất). Nhóm cánh tay (arm_group_ids) không bị loại (paper: "groups
    involving both arms will have two fewer joints dropped").
    """

    def __init__(self, groups, drop_per_group, arm_group_ids):
        super().__init__()
        self.groups = groups
        self.drop_per_group = drop_per_group
        self.arm_group_ids = set(arm_group_ids)

    def forward(self, x, cor):
        # x: (B, T, N, C) ; cor: (N,)
        keep_idx = []
        for g_id, group in enumerate(self.groups):
            n_drop = 0 if g_id in self.arm_group_ids else self.drop_per_group
            if n_drop <= 0 or len(group) <= n_drop:
                keep_idx.extend(group)
                continue
            group_idx_t = torch.tensor(group, device=cor.device)
            group_cor = cor[group_idx_t]
            order = torch.argsort(group_cor, descending=True)          # Cor cao -> giữ trước
            kept_local = order[: len(group) - n_drop].tolist()
            keep_idx.extend(sorted(group[i] for i in kept_local))

        keep_idx = sorted(keep_idx)
        keep_idx_t = torch.tensor(keep_idx, device=x.device, dtype=torch.long)
        x = x.index_select(dim=2, index=keep_idx_t)
        return x


class SPDBlock(nn.Module):
    """1 block SPD = Keypoint Activation (Decoupled GCN + TCN) + Weak Joints Dropping."""

    def __init__(self, in_ch, out_ch, num_nodes, groups, arm_group_ids,
                 p=3, drop_per_group=2, kernel_size=9, dropout=0.0,
                 base_adjacency=None, is_last=False):
        super().__init__()
        self.gcn = DecoupledGraphConv(in_ch, out_ch, num_nodes, p=p, base_adjacency=base_adjacency)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.tcn = _TemporalConv(out_ch, kernel_size=kernel_size, dropout=dropout)
        self.is_last = is_last

        self.dropper = None if is_last else WeakJointsDropping(groups, drop_per_group, arm_group_ids)

    def forward(self, x):
        # x: (B, T, N, C_in)
        B, T, N, _ = x.shape

        feat, A = self.gcn(x)                                         # (B, T, N, C_out)
        feat = self.bn(feat.reshape(B * T * N, -1)).reshape(B, T, N, -1)
        feat = self.relu(feat)
        feat = self.tcn(feat)                                          # (B, T, N, C_out)

        if self.dropper is not None:
            cor = compute_correlation(A)                               # (N,) — chỉ phụ thuộc A, không phụ thuộc x
            feat = self.dropper(feat, cor)                              # (B, T, N', C_out)

        return feat


class GroupPool(nn.Module):
    """
    Gộp các khớp còn lại trong mỗi nhóm (sau 3 block SPD) thành ĐÚNG 1 token
    mỗi nhóm bằng attention-pool có trọng số học được — cho ra Ft ∈ R^{T×G×C}
    (G=6 mặc định) đúng dạng input của VSformer trong paper.
    """

    def __init__(self, channels, groups):
        super().__init__()
        self.groups = groups
        self.attn = nn.Sequential(
            nn.Linear(channels, max(channels // 2, 8)),
            nn.Tanh(),
            nn.Linear(max(channels // 2, 8), 1),
        )

    def forward(self, x):
        # x: (B, T, N, C) -> (B, T, G, C)
        outs = []
        for group in self.groups:
            idx = torch.tensor(group, device=x.device)
            sub = x.index_select(dim=2, index=idx)                 # (B, T, g, C)
            w = torch.softmax(self.attn(sub), dim=2)                 # (B, T, g, 1)
            outs.append((sub * w).sum(dim=2, keepdim=True))          # (B, T, 1, C)
        return torch.cat(outs, dim=2)                                # (B, T, G, C)


# =============================================================================
# VSformer — SPA + TWA + GCN + TCN song song
# -----------------------------------------------------------------------------

class SpatialPairingAttention(nn.Module):
    """
    Eq.5. Ft ∈ R^{G×C} được chia thành các nhóm con [P_arm, P_left, P_right]
    (mặc định 3 cặp theo paper). Tính self-attention ĐỘC LẬP trong từng cặp
    (không cho các cặp chú ý chéo nhau — vì 2 tay ở xa nhau trong cấu trúc
    khung xương thường không liên quan trực tiếp), rồi ghép lại.
    """

    def __init__(self, channels, spa_groups):
        super().__init__()
        self.spa_groups = spa_groups
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.scale = channels ** -0.5

    def forward(self, x):
        # x: (B, T, G, C)
        q, k, v = self.q(x), self.k(x), self.v(x)
        out = torch.zeros_like(x)
        for group in self.spa_groups:
            idx = torch.tensor(group, device=x.device)
            qi, ki, vi = q.index_select(2, idx), k.index_select(2, idx), v.index_select(2, idx)
            attn = torch.softmax(qi @ ki.transpose(-1, -2) * self.scale, dim=-1)
            out.index_copy_(2, idx, attn @ vi)
        return out


class TemporalWindowAttention(nn.Module):
    """
    Eq.6 (đơn giản hoá, xem ghi chú đầu file).
    TWA-s: attention TRONG từng cửa sổ k frame liên tiếp — nắm chuyển động
           ngắn hạn.
    TWA-l: attention GIỮA các frame CÙNG chỉ số trong mỗi cửa sổ — nắm bắt
           mô-típ chuyển động lặp lại/dài hạn theo chu kỳ cửa sổ.
    """

    def __init__(self, channels, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.scale = channels ** -0.5

    def _attend(self, x):
        # x: (..., L, C)
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        return attn @ v

    def forward(self, x):
        # x: (B, T, G, C)
        B, T, G, C = x.shape
        k = min(self.window_size, T)
        n = T // k
        if n == 0:
            n, k = 1, T
        usable_T = n * k
        x_used = x[:, :usable_T]

        # TWA-s: attention theo chiều k (trong từng window)
        x_s = x_used.reshape(B, n, k, G, C).permute(0, 1, 3, 2, 4)      # (B, n, G, k, C)
        out_s = self._attend(x_s).permute(0, 1, 3, 2, 4).reshape(B, usable_T, G, C)

        # TWA-l: attention theo chiều n (giữa các window, cùng index trong window)
        x_l = x_used.reshape(B, n, k, G, C).permute(0, 2, 3, 1, 4)      # (B, k, G, n, C)
        out_l = self._attend(x_l).permute(0, 3, 1, 2, 4).reshape(B, usable_T, G, C)

        out = out_s + out_l
        if usable_T < T:
            out = torch.cat([out, x[:, usable_T:]], dim=1)   # phần dư (< 1 window) giữ nguyên
        return out


class VSformerBlock(nn.Module):
    """
    1 block VSformer: chia kênh làm 4 phần bằng nhau, xử lý song song bằng
    SPA / TWA / GCN / TCN, ghép lại, cộng residual + FFN (kiểu Transformer
    chuẩn, pre-norm).
    """

    def __init__(self, channels, num_groups, spa_groups, window_size=8, dropout=0.1):
        super().__init__()
        assert channels % 4 == 0, "channels phải chia hết cho 4 (SPA/TWA/GCN/TCN mỗi phần C/4)"
        c4 = channels // 4

        self.norm1 = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, channels)

        self.spa = SpatialPairingAttention(c4, spa_groups)
        self.twa = TemporalWindowAttention(c4, window_size=window_size)
        # Đồ thị nhỏ (G node, thường G=6) — học đầy đủ (không cố định theo bone tự nhiên)
        self.gcn = DecoupledGraphConv(c4, c4, num_groups, p=1, base_adjacency=torch.eye(num_groups))
        self.tcn = _TemporalConv(c4, kernel_size=5, dropout=dropout)

        self.out_proj = nn.Linear(channels, channels)
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x):
        # x: (B, T, G, C)
        residual = x
        h = self.norm1(x)
        h = self.in_proj(h)

        c4 = h.shape[-1] // 4
        h_spa, h_twa, h_gcn, h_tcn = h.split(c4, dim=-1)

        out_spa = self.spa(h_spa)
        out_twa = self.twa(h_twa)
        out_gcn, _ = self.gcn(h_gcn)
        out_tcn = self.tcn(h_tcn)

        out = torch.cat([out_spa, out_twa, out_gcn, out_tcn], dim=-1)
        out = self.drop(self.out_proj(out))
        x = residual + out

        x = x + self.ffn(self.norm2(x))
        return x


class TemporalDownsample(nn.Module):
    """Conv kernel=7, stride=2 theo trục T + BatchNorm — đúng caption Fig.3."""

    def __init__(self, channels, kernel_size=7, stride=2):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                               stride=(stride, 1), padding=(pad, 0))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # x: (B, T, G, C) -> (B, C, T, G)
        x = x.permute(0, 3, 1, 2)
        x = self.bn(self.conv(x))
        x = x.permute(0, 2, 3, 1)
        return x


# =============================================================================
# VSNet — toàn bộ pipeline
# -----------------------------------------------------------------------------

class VSNet(nn.Module):

    def __init__(
        self,
        num_classes=2000,
        n_pose=5,                            # số điểm pose/cánh tay — 5 khớp với feature [B,T,94]=(5+21+21)*2
        n_hand=21,                            # số khớp mỗi tay (layout chuẩn 21 điểm)
        in_channels=2,                        # (x, y) — đổi thành 3 nếu dùng cả z
        spd_channels=(64, 128, 256),
        vsformer_dim=256,
        num_vsformer_blocks=8,
        p_decouple=3,
        drop_per_group=2,
        window_size=8,
        downsample_at=(3, 6),                # sau block VSformer thứ mấy (1-indexed) thì downsample T 2x
        dropout=0.1,
        split_pose_group=False,               # True: tách pose thành 2 nhóm để có G=6 giống paper gốc (2 điểm cánh tay)
    ):
        super().__init__()

        assert vsformer_dim == spd_channels[-1], \
            "vsformer_dim phải bằng kênh output cuối của SPD (không cộng thêm projection ở đây)"
        assert n_hand == 21, \
            "Cách chia nhóm tay (cổ tay+cái+trỏ=9 | giữa+áp út+út=12) giả định layout 21 điểm chuẩn"

        num_nodes = n_pose + 2 * n_hand

        if split_pose_group and n_pose >= 2:
            half = n_pose // 2
            pose_groups = [list(range(0, half)), list(range(half, n_pose))]
            arm_group_ids = {0, 1}
        else:
            pose_groups = [list(range(0, n_pose))]
            arm_group_ids = {0}

        left_start = n_pose
        right_start = n_pose + n_hand
        # tay trái/phải: cổ tay+cái+trỏ (9 điểm) | giữa+áp út+út (12 điểm) — layout 21 điểm chuẩn
        groups_stage1 = pose_groups + [
            list(range(left_start, left_start + 9)), list(range(left_start + 9, left_start + n_hand)),
            list(range(right_start, right_start + 9)), list(range(right_start + 9, right_start + n_hand)),
        ]

        groups_stage2 = _derive_next_groups(groups_stage1, arm_group_ids, drop_per_group)
        groups_stage3 = _derive_next_groups(groups_stage2, arm_group_ids, drop_per_group)  # dùng cho GroupPool, KHÔNG drop thêm

        n1 = num_nodes
        n2 = sum(len(g) for g in groups_stage2)
        n3 = sum(len(g) for g in groups_stage3)

        ch = [in_channels, *spd_channels]
        self.spd_blocks = nn.ModuleList([
            SPDBlock(ch[0], ch[1], num_nodes=n1, groups=groups_stage1, arm_group_ids=arm_group_ids,
                     p=p_decouple, drop_per_group=drop_per_group, dropout=dropout, is_last=False),
            SPDBlock(ch[1], ch[2], num_nodes=n2, groups=groups_stage2, arm_group_ids=arm_group_ids,
                     p=p_decouple, drop_per_group=drop_per_group, dropout=dropout, is_last=False),
            SPDBlock(ch[2], ch[3], num_nodes=n3, groups=groups_stage3, arm_group_ids=arm_group_ids,
                     p=p_decouple, drop_per_group=drop_per_group, dropout=dropout, is_last=True),
        ])

        self.group_pool = GroupPool(spd_channels[-1], groups_stage3)   # (B,T,n3,C) -> (B,T,G,C)
        num_vs_groups = len(groups_stage3)                              # G — 5 (mặc định) hoặc 6 (split_pose_group=True)

        # SPA hoạt động trên G token: [P_arm(1|2 token), P_left(2 token), P_right(2 token)]
        n_pose_groups = len(pose_groups)
        spa_groups = [
            list(range(0, n_pose_groups)),
            list(range(n_pose_groups, n_pose_groups + 2)),
            list(range(n_pose_groups + 2, n_pose_groups + 4)),
        ]

        self.vsformer_blocks = nn.ModuleList([
            VSformerBlock(vsformer_dim, num_vs_groups, spa_groups, window_size=window_size, dropout=dropout)
            for _ in range(num_vsformer_blocks)
        ])
        self.downsample_at = set(downsample_at)
        self.downsample = TemporalDownsample(vsformer_dim)

        self.classifier = nn.Linear(vsformer_dim, num_classes)

    def forward(self, features, labels=None, video_mask=None):
        """
        features: (B, T, num_nodes * in_channels) — toạ độ phẳng, ĐÃ chuẩn hoá
                  theo node V0 (Eq.1) từ bước tiền xử lý.
        video_mask: (B, T) bool hoặc None. Paper gốc dùng độ dài cố định 64
                  frame nên KHÔNG xử lý padding — nếu bạn có video dài khác
                  nhau, `video_mask` ở đây chỉ được dùng ở bước pooling cuối
                  (masked mean), KHÔNG lan truyền vào SPA/TWA/GCN/TCN bên
                  trong — đây là giới hạn cần biết trước khi dùng với dữ liệu
                  độ dài thay đổi (khuyến nghị: cắt/pad về đúng 1 độ dài cố
                  định như paper làm, để tránh vấn đề này hoàn toàn).
        """
        B, T, _ = features.shape
        x = features.reshape(B, T, -1, self.spd_blocks[0].gcn.phi.in_features)  # (B,T,N,C_in)

        for block in self.spd_blocks:
            x = block(x)

        x = self.group_pool(x)                    # (B, T, G, vsformer_dim)

        for i, block in enumerate(self.vsformer_blocks, start=1):
            x = block(x)
            if i in self.downsample_at:
                x = self.downsample(x)
                T = x.shape[1]
                if video_mask is not None:
                    video_mask = video_mask[:, ::2][:, :T]

        if video_mask is not None:
            mask = video_mask[:, :x.shape[1]].bool().unsqueeze(-1).unsqueeze(-1).float()  # (B,T,1,1)
            pooled = (x * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1.0)
        else:
            pooled = x.mean(dim=(1, 2))            # Average Pooling — mean theo CẢ T và G (đúng paper)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels, label_smoothing=0.1)  # paper: CE + label smoothing

        return logits, loss

    @torch.no_grad()
    def predict(self, features, video_mask=None, top_k=1):
        logits, _ = self.forward(features, video_mask=video_mask)
        if top_k == 1:
            return logits.argmax(dim=-1)
        return logits.topk(k=top_k, dim=-1).indices