"""
KeypointMotionTransformer — học chuyển động khung xương bằng bài toán Masked
Keypoint Modeling: che từng đoạn liên tục theo từng luồng (pose/trái/phải)
độc lập, bắt model tái tạo lại đúng toạ độ đã che dựa vào ngữ cảnh 2 chiều
(nhìn được cả frame trước lẫn sau vị trí bị che — khác transformer sinh văn
bản chỉ nhìn về quá khứ).

Input mỗi frame = [coords_đã_che (pose_dim+left_dim+right_dim), mask_flags(3)]
    - coords_đã_che: vùng bị che = 0 (không mang thông tin thật).
    - mask_flags: 1 = "đây là chỗ bị che, đừng tin giá trị 0, hãy đoán lại".
      Đây LÀ tín hiệu duy nhất giúp model phân biệt "0 = giá trị thật" và
      "0 = bị che" — bắt buộc phải có, nếu không model không thể học đúng.
"""


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding chuẩn (Vaswani et al.), tự chứa trong
    file này để không phụ thuộc module ngoài."""

    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class KeypointMotionTransformer(nn.Module):

    def __init__(
        self,
        pose_dim=10,     # 33 * coord_dim (mặc định coord_dim=2)
        left_dim=42,      # 21 * coord_dim
        right_dim=42,     # 21 * coord_dim
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=512,
    ):
        super().__init__()

        self.pose_dim = pose_dim
        self.left_dim = left_dim
        self.right_dim = right_dim

        in_dim = pose_dim + left_dim + right_dim + 3   # +3 mask flags (pose/left/right)

        self.input_projection = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # Lưu ý: KHÔNG dùng causal mask — model cần nhìn được cả 2 chiều
        # thời gian (trước và sau vị trí bị che) để đoán đúng, khác hẳn
        # transformer sinh văn bản (chỉ nhìn quá khứ).
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        self.pose_head = nn.Linear(d_model, pose_dim)
        self.left_head = nn.Linear(d_model, left_dim)
        self.right_head = nn.Linear(d_model, right_dim)

    def forward(self, coords_input, mask_flags, video_mask=None):
        """
        coords_input: (B, T, pose_dim+left_dim+right_dim) — vùng bị che = 0
        mask_flags  : (B, T, 3) — 1 = bị che (pose, left, right)
        video_mask  : (B, T) bool hoặc None — True = frame thật (không phải padding)
        """
        x = torch.cat([coords_input, mask_flags], dim=-1)
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        pad_mask = None
        if video_mask is not None:
            pad_mask = ~video_mask.bool()

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.encoder_norm(x)

        pose_pred = self.pose_head(x)
        left_pred = self.left_head(x)
        right_pred = self.right_head(x)

        return pose_pred, left_pred, right_pred

    @torch.no_grad()
    def restore(self, coords_input, mask_flags, video_mask=None):
        """
        Dùng lúc inference để khôi phục điểm thiếu trên dữ liệu THẬT:
        chỉ ghi đè lại các vị trí có mask_flags=1 bằng giá trị model dự đoán,
        giữ nguyên các vị trí đã có dữ liệu thật (không "sửa" cả những gì
        đã đúng).
        """
        pose_pred, left_pred, right_pred = self.forward(coords_input, mask_flags, video_mask)
        pred = torch.cat([pose_pred, left_pred, right_pred], dim=-1)

        # mask_flags (B,T,3) -> mở rộng ra đúng số chiều toạ độ mỗi luồng
        m_pose = mask_flags[..., 0:1].expand(-1, -1, self.pose_dim)
        m_left = mask_flags[..., 1:2].expand(-1, -1, self.left_dim)
        m_right = mask_flags[..., 2:3].expand(-1, -1, self.right_dim)
        m_full = torch.cat([m_pose, m_left, m_right], dim=-1)      # (B,T,D)

        restored = torch.where(m_full > 0.5, pred, coords_input)
        return restored


class MotionReconstructionLoss(nn.Module):
    """
    Loss = SmoothL1(vị trí) + velocity_weight * SmoothL1(vận tốc — sai phân
    frame-liên-tiếp), tính CHỈ trên các vị trí bị che (mask_flags=1).

    Thành phần vận tốc chính là phần khiến model học "chuyển động" thay vì
    chỉ khớp toạ độ tĩnh từng frame độc lập — nếu chỉ có loss vị trí, model
    có thể đoán đúng từng điểm riêng lẻ nhưng cho ra quỹ đạo giật cục/không
    mượt giữa các frame liên tiếp.
    """

    def __init__(self, velocity_weight=0.5):
        super().__init__()
        self.velocity_weight = velocity_weight

    def _masked_loss(self, pred, target, mask):
        # pred, target: (B,T,D) ; mask: (B,T,1) hoặc (B,T,D)
        diff = F.smooth_l1_loss(pred, target, reduction="none")
        diff = diff * mask
        denom = mask.sum().clamp(min=1.0)
        return diff.sum() / denom

    def forward(self, pred_pose, pred_left, pred_right, target, mask_flags, pose_dim, left_dim, right_dim):
        target_pose = target[..., :pose_dim]
        target_left = target[..., pose_dim:pose_dim + left_dim]
        target_right = target[..., pose_dim + left_dim:pose_dim + left_dim + right_dim]

        m_pose = mask_flags[..., 0:1]
        m_left = mask_flags[..., 1:2]
        m_right = mask_flags[..., 2:3]

        pos_loss = (
            self._masked_loss(pred_pose, target_pose, m_pose.expand_as(target_pose)) +
            self._masked_loss(pred_left, target_left, m_left.expand_as(target_left)) +
            self._masked_loss(pred_right, target_right, m_right.expand_as(target_right))
        )

        # Vận tốc = sai phân giữa 2 frame liên tiếp — 1 vị trí "cần tính vận
        # tốc" là masked nếu 1 trong 2 đầu mút (t hoặc t-1) đang bị che.
        def velocity(x):
            return x[:, 1:] - x[:, :-1]

        vel_pred_pose, vel_target_pose = velocity(pred_pose), velocity(target_pose)
        vel_pred_left, vel_target_left = velocity(pred_left), velocity(target_left)
        vel_pred_right, vel_target_right = velocity(pred_right), velocity(target_right)

        vel_mask_pose = ((m_pose[:, 1:] + m_pose[:, :-1]) > 0).float().expand_as(vel_pred_pose)
        vel_mask_left = ((m_left[:, 1:] + m_left[:, :-1]) > 0).float().expand_as(vel_pred_left)
        vel_mask_right = ((m_right[:, 1:] + m_right[:, :-1]) > 0).float().expand_as(vel_pred_right)

        vel_loss = (
            self._masked_loss(vel_pred_pose, vel_target_pose, vel_mask_pose) +
            self._masked_loss(vel_pred_left, vel_target_left, vel_mask_left) +
            self._masked_loss(vel_pred_right, vel_target_right, vel_mask_right)
        )

        total = pos_loss + self.velocity_weight * vel_loss
        return total, pos_loss, vel_loss