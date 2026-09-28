import torch
import torch.nn as nn
from torchvision.models.video.resnet import BasicBlock

from config import _NUM_NODE, _COORD_DIM, _N_POSE, _N_HAND
from src.models.positional_encoding import PositionalEncoding
from src.models.spatial_graph import build_adjacency, GCNBlock, SelfPacingDroppingBlock
import torchvision.models as models
import torch.nn.functional as F


def masked_mean_pool(x, video_mask):
    mask = video_mask.unsqueeze(-1).float()  # (B, T, 1)
    summed = (x * mask).sum(dim=1)  # (B, D)
    counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1) — avoid /0
    return summed / counts


class ISLR_GCN(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), num_classes=2000):
        super().__init__()

        self.num_nodes = _NUM_NODE
        self.register_buffer('adjacency_matrix', build_adjacency())

        channels = [_COORD_DIM, *channels]
        gcn_out_dim = channels[-1]

        self.gcn_block = nn.ModuleList([GCNBlock(in_ch=channels[i], out_ch=channels[i + 1], num_nodes=self.num_nodes,
                                                 base_adjacency=self.adjacency_matrix) for i in
                                        range(len(channels) - 1)])

        self.classifier = nn.Linear(gcn_out_dim, num_classes)

    def forward(self, features, labels=None, video_mask=None):
        B, T, _ = features.shape

        video_mask = video_mask.bool()

        features = features.reshape(B, T, self.num_nodes * 3, _COORD_DIM)
        features = features[:, :, :self.num_nodes, :]

        # features = self.stem(features)
        for block in self.gcn_block:
            features = block(features, video_mask)

        features = features.mean(dim=-2)

        features = masked_mean_pool(features, video_mask)

        logits = self.classifier(features)
        loss = F.cross_entropy(logits, labels)

        return logits, loss


class ISLR_Transformer(nn.Module):
    def __init__(self, hidden_dim=256, num_encoder_layers=6, nhead=8, dim_feedforward=256 * 8, dropout=0.2,
                 max_seq_len=5000, num_classes=1000):
        super().__init__()

        d_model = hidden_dim
        self.num_nodes = _NUM_NODE

        self.pose_projection = nn.Sequential(nn.Linear(_N_POSE * _COORD_DIM, d_model), nn.GELU(), nn.Dropout(dropout),
                                             nn.LayerNorm(d_model))

        self.hand_projection = nn.Sequential(nn.Linear(_N_HAND * _COORD_DIM * 2, d_model), nn.GELU(),
                                             nn.Dropout(dropout), nn.LayerNorm(d_model))

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

        pose_features = features[:, :, :_N_POSE * _COORD_DIM]
        hand_features = features[:, :, _N_POSE * _COORD_DIM:]

        x_pose = self.pose_projection(pose_features)
        x_hand = self.hand_projection(hand_features)

        x = x_pose + x_hand
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


class ISLR_Transformer_Selector(nn.Module):
    def __init__(self, input_dim=_NUM_NODE * _COORD_DIM, hidden_dim=256, num_encoder_layers=6, nhead=8,
                 dim_feedforward=256 * 8, dropout=0.2, max_seq_len=5000, num_classes=1000):
        super().__init__()

        d_model = hidden_dim
        self.num_nodes = _NUM_NODE

        self.pose_projection = nn.Sequential(nn.Linear(_N_POSE * _COORD_DIM, d_model), nn.GELU(), nn.Dropout(dropout),
                                             nn.LayerNorm(d_model))

        self.hand_projection = nn.Sequential(nn.Linear(_N_HAND * _COORD_DIM * 2, d_model), nn.GELU(),
                                             nn.Dropout(dropout), nn.LayerNorm(d_model))

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True, norm_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(d_model, num_classes)

        self.frame_importance_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
                                                   nn.Linear(d_model // 2, 1))

    def encode(self, features, video_mask):
        if video_mask is None:
            raise ValueError("video_mask is required")

        video_mask = video_mask.bool()

        pose_features = features[:, :, :_N_POSE * _COORD_DIM]
        hand_features = features[:, :, _N_POSE * _COORD_DIM:]

        x_pose = self.pose_projection(pose_features)
        x_hand = self.hand_projection(hand_features)
        x = x_pose + x_hand

        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=~video_mask)
        x = self.encoder_norm(x)

        return x

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")

        video_mask = video_mask.bool()

        x = self.encode(features, video_mask)

        frame_scores = self.frame_importance_head(x).squeeze(-1)
        frame_scores = frame_scores.masked_fill(~video_mask, -1e9)

        frame_importance = torch.softmax(frame_scores, dim=1)

        pooled = torch.sum(x * frame_importance.unsqueeze(-1), dim=1)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"logits": logits, "loss": loss, "frame_importance": frame_importance, "encoded": x}


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


class ISLR_Transformer_GCN(nn.Module):
    def __init__(self, gcn_channels=(128, 128, 256, 256), d_model=256, num_encoder_layers=6, nhead=8,
                 dim_feedforward=256 * 8, dropout=0.2, max_seq_len=5000, num_classes=1000):
        super().__init__()
        self.num_nodes = _NUM_NODE

        self.register_buffer('adjacency_matrix', build_adjacency())
        channels = [_COORD_DIM, *gcn_channels]

        self.pose_projection = nn.Sequential(nn.Linear(_N_POSE * _COORD_DIM, d_model), nn.GELU(), nn.Dropout(dropout),
                                             nn.LayerNorm(d_model))

        self.hand_projection = nn.Sequential(nn.Linear(_N_HAND * _COORD_DIM * 2, d_model), nn.GELU(),
                                             nn.Dropout(dropout), nn.LayerNorm(d_model))
        self.gcn_block = nn.ModuleList(
            [SelfPacingDroppingBlock(channels[i], channels[i + 1], _N_HAND * 2, self.adjacency_matrix) for i in
             range(len(channels) - 1)])

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes), )

    def encode_gcn(self, features, video_mask):

        for block in self.gcn_block:
            features = block(features, video_mask)

        features = features.mean(dim=-2)  # (B, T, gcn_out)
        # return self.gcn_proj(x)  # (B, T, d_model)
        return features

    def encode_transformer(self, features, video_mask):
        B, T, _ = features.shape

        pose_features = features[:, :, :_N_POSE * _COORD_DIM]
        hand_features = features[:, :, _N_POSE * _COORD_DIM:]

        x_gcn = self.encode_gcn(features[:, :, _N_POSE * _COORD_DIM:].clone().reshape(B, T, _N_HAND * 2, _COORD_DIM),
                                video_mask)
        x_pose = self.pose_projection(pose_features)
        x_hand = self.hand_projection(hand_features)

        # x = x_average                                     # best=73.00% loss=1.2133 75.00%
        # x = x_average + x_gcn                             # best=66.00% loss=1.4254 95.00%

        # x = x_position                                    # best=72.00% loss=1.2574 75.00%
        # x = x_position + x_gcn                            # best=70.00% loss=1.3490 71.00%

        # x = x_shape                                       # best=66.00% loss=1.2873 67.00%
        # x = x_shape + x_gcn                               # best=59.00% loss=1.3645 67.00%

        # x = x_shape + x_average                           # best=73.00% loss=1.1507 75.00%
        # x = x_shape + x_average + x_gcn                   # best=71.00% loss=1.1686 74.00

        # x = x_shape + x_average + x_position  # best=75.00% loss=1.0840 77.00%
        # x = x_shape + x_average + x_position + x_gcn      # best=71.00% loss=1.1394 78.00%
        # x = x_gcn                                           # best=62.00% loss=1.4346 66.00%

        x = x_pose + x_hand + x_gcn
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=~video_mask)
        return self.encoder_norm(x)  # (B, T, d_model)

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")
        video_mask = video_mask.bool()

        B, T, _ = features.shape
        features = features[:, :, self.num_nodes * 2:self.num_nodes * 4]

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
        self.A = nn.Parameter(adjacency.clone().unsqueeze(0).repeat(p, 1, 1))  # (p, N, N)
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


class ISLR_EncoderDecoder(nn.Module):
    def __init__(self, input_dim=_NUM_NODE * _COORD_DIM, hidden_dim=256, num_encoder_layers=6, num_decoder_layers=2,
                 nhead=8, dim_feedforward=256 * 8, dropout=0.2, max_seq_len=5000, num_classes=1000, num_queries=1):
        super().__init__()

        d_model = hidden_dim
        self.num_nodes = _NUM_NODE
        self.num_queries = num_queries

        # ----- Encoder: giữ nguyên -----
        self.position_projection = nn.Sequential(nn.Linear(self.num_nodes * 2, d_model), nn.GELU(), nn.Dropout(dropout),
                                                 nn.LayerNorm(d_model))
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # ----- Decoder: thay cho masked_mean_pool -----
        # Learnable query — giống [CLS] token, nhưng đóng vai trò "câu hỏi":
        # "trong chuỗi frame này, đặc trưng gloss là gì?"
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.decoder_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(d_model, num_classes)

    def encode(self, features, video_mask):
        if video_mask is None:
            raise ValueError("video_mask is required")

        video_mask = video_mask.bool()
        position_features = features[:, :, :self.num_nodes * 2]

        x = self.position_projection(position_features)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=~video_mask)
        x = self.encoder_norm(x)
        return x  # (B, T, d_model) — KHÔNG pool, để decoder tự chọn lọc

    def forward(self, features, labels=None, video_mask=None):
        B, T, _ = features.shape
        video_mask = video_mask.bool()

        memory = self.encode(features, video_mask)  # (B, T, d_model)

        # Query không phụ thuộc input, chỉ cần lặp lại theo batch size
        queries = self.query_embed.expand(B, -1, -1)  # (B, num_queries, d_model)

        # Decoder: query cross-attend vào memory (không cần tgt_mask vì không autoregressive)
        decoded = self.decoder(tgt=queries, memory=memory, memory_key_padding_mask=~video_mask,
                               # bỏ qua frame padding khi attend
                               )
        decoded = self.decoder_norm(decoded)  # (B, num_queries, d_model)

        pooled = decoded.mean(dim=1)  # nếu num_queries=1 thì chỉ là squeeze, không mất thông tin
        logits = self.classifier(pooled)  # (B, num_classes)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment, fold_div=8):
        super().__init__()
        self.net, self.n_segment, self.fold_div = net, n_segment, fold_div

    def forward(self, x):  # x: (N*K, C, H, W), các frame của cùng 1 chuỗi nằm liền nhau
        nt, c, h, w = x.shape
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # frame t nhận từ t+1
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]  # frame t nhận từ t-1
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return self.net(out.view(nt, c, h, w))


class RGBEncoder(nn.Module):
    def __init__(self, d_model=256, dropout=0.2, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        self.shift_modules = []
        for layer in (resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4):
            for block in layer:
                if isinstance(block, BasicBlock):
                    block.conv1 = TemporalShift(block.conv1, n_segment=1)  # cập nhật theo T ở forward
                    self.shift_modules.append(block.conv1)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (N,512,1,1)
        self.proj = nn.Sequential(nn.Linear(512, d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model))
        self.time_embed = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, rgb_frames, video_mask):
        B, T, C, H, W = rgb_frames.shape
        for m in self.shift_modules:
            m.n_segment = T

        x = rgb_frames.reshape(B * T, C, H, W)
        x = self.backbone(x).view(B, T, 512)
        x = self.proj(x)  # (B,T,d_model)

        lengths = video_mask.bool().sum(dim=1, keepdim=True).clamp(min=2).float()  # (B,1)
        pos = torch.arange(T, device=x.device).float().unsqueeze(0)  # (1,T)
        t = (pos / (lengths - 1)).clamp(max=1.0).unsqueeze(-1)  # (B,T,1)
        return x + self.time_embed(t)


class PoseRGBFusionModel(nn.Module):
    def __init__(self, num_classes=1000, d_model=256, nhead=8, num_fusion_layers=2, proj_dim=128, dropout=0.2):
        super().__init__()

        # Encoder
        self.pose_encoder = ISLR_Transformer(hidden_dim=d_model, dropout=dropout, num_classes=num_classes)
        self.rgb_encoder = RGBEncoder(d_model=d_model, dropout=dropout)

        # Stage 1: projection head + temperature (chỉ dùng khi pretrain)
        def head():
            return nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, proj_dim))

        self.pose_head, self.rgb_head = head(), head()
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))  # ln(1/0.07) như CLIP

        # Stage 2: cross-attention hai chiều
        self.pose_from_rgb = nn.ModuleList(
            [CrossAttentionLayer(d_model, nhead, dropout=dropout) for _ in range(num_fusion_layers)])
        self.rgb_from_pose = nn.ModuleList(
            [CrossAttentionLayer(d_model, nhead, dropout=dropout) for _ in range(num_fusion_layers)])
        self.fusion_norm = nn.LayerNorm(2 * d_model)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * d_model, num_classes)

    # ---------------- encode ----------------
    def encode(self, pose_feature, rgb_feature, video_mask):
        video_mask = video_mask.bool()
        pose = self.pose_encoder.encode(pose_feature, video_mask)  # (B,T,D)
        rgb = self.rgb_encoder(rgb_feature, video_mask)  # (B,T,D)
        return pose, rgb

    # ---------------- Stage 1 ----------------
    @staticmethod
    def _info_nce(a, b, scale):
        logits = scale * a @ b.t()
        target = torch.arange(a.size(0), device=a.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))

    def forward_pretrain(self, pose_feature, rgb_feature, video_mask):
        """Không dùng label. Cùng video = positive, khác video = negative (InfoNCE đối xứng)."""
        video_mask = video_mask.bool()
        pose, rgb = self.encode(pose_feature, rgb_feature, video_mask)
        scale = self.logit_scale.exp().clamp(max=100)

        # cả hai đều pool có mask (RGB cũng có T frame kèm padding)
        p_g = F.normalize(self.pose_head(masked_mean_pool(pose, video_mask)), dim=-1)
        r_g = F.normalize(self.rgb_head(masked_mean_pool(rgb, video_mask)), dim=-1)
        loss = self._info_nce(p_g, r_g, scale)

        return {"loss": loss, "loss_video": loss.detach(), "p_g": p_g.detach(), "r_g": r_g.detach()}

    # ---------------- Stage 2 ----------------
    def forward(self, pose_feature, rgb_feature, labels=None, video_mask=None):
        video_mask = video_mask.bool()
        pose, rgb = self.encode(pose_feature, rgb_feature, video_mask)
        pad = ~video_mask  # True = padding (pose và RGB dùng chung mask vì cùng T frame)

        for pose_layer, rgb_layer in zip(self.pose_from_rgb, self.rgb_from_pose):
            new_pose = pose_layer(pose, rgb, pad)  # Pose Query -> RGB Key/Value
            new_rgb = rgb_layer(rgb, pose, pad)  # RGB Query  -> Pose Key/Value
            pose, rgb = new_pose, new_rgb

        pooled = torch.cat([masked_mean_pool(pose, video_mask), masked_mean_pool(rgb, video_mask)], dim=-1)
        logits = self.classifier(self.drop(self.fusion_norm(pooled)))

        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}

    # ---------------- tiện ích ----------------
    def freeze_encoders(self, freeze=True):
        """Dùng ở đầu Stage 2 (vài epoch đầu) để fusion không phá embedding đã pretrain."""
        for m in (self.pose_encoder, self.rgb_encoder):
            for p in m.parameters():
                p.requires_grad = not freeze

    def load_pretrained(self, path):
        ckpt = torch.load(path, map_location="cpu")
        print(self.load_state_dict(ckpt.get("model", ckpt), strict=False))
