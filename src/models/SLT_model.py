import torch
import torch.nn as nn

from config import _NUM_NODE, _COORD_DIM
from src.models.positional_encoding import PositionalEncoding
from src.models.spatial_graph import build_adjacency, GCNBlock, SelfPacingDroppingBlock


def masked_mean_pool(x, video_mask):
    mask = video_mask.unsqueeze(-1).float()  # (B, T, 1)
    summed = (x * mask).sum(dim=1)  # (B, D)
    counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1) — avoid /0
    return summed / counts


class FusionStem(nn.Module):
    def __init__(self, in_ch, out_ch, num_nodes, dropout=0.1):
        super().__init__()
        self.N = num_nodes

        def branch():
            return nn.Sequential(nn.Linear(in_ch, out_ch), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(out_ch))

        self.pos, self.shp, self.avg = branch(), branch(), branch()

    def forward(self, x):  # (B, T, 3N, C_in)
        N = self.N
        # return (self.pos(x[:, :, :N])
        #         + self.shp(x[:, :, N:2*N])
        #         + self.avg(x[:, :, 2*N:3*N]))

        return self.pos(x[:, :, N:2 * N])


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

        x = x_position + x_shape + x_average

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


class ISLR_Transformer_GCN(nn.Module):
    def __init__(self, gcn_channels=(64, 64, 64, 64), d_model=256, num_encoder_layers=6, nhead=8,
                 dim_feedforward=256 * 8, dropout=0.2, max_seq_len=5000, num_classes=1000):
        super().__init__()
        self.num_nodes = _NUM_NODE

        # ---------------- Stream 1: GCN (V3) ----------------
        self.register_buffer('adjacency_matrix', build_adjacency())
        channels = [_COORD_DIM, *gcn_channels]

        self.stem = FusionStem(channels[0], channels[1], self.num_nodes)
        self.gcn_block = nn.ModuleList(
            [SelfPacingDroppingBlock(channels[i], channels[i + 1], self.num_nodes, self.adjacency_matrix) for i in
             range(len(channels) - 1)])
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
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes), )

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

        x = x_shape + x_average + x_position  # best=75.00% loss=1.0840 77.00%
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
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.drop(self.act(self.bn(self.conv(x))))
        return x.transpose(1, 2)


class LongTermTransformer(nn.Module):
    """Nửa kênh còn lại — học chuyển động DÀI HẠN bằng self-attention toàn chuỗi."""

    def __init__(self, d_model, nhead=8, dim_feedforward=1024, num_layers=4, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, video_mask):
        # x: (B, T, C), video_mask: (B, T) bool
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=~video_mask)
        return self.norm(x)


class ISLR_6(nn.Module):
    """
    1) Mỗi frame áp dụng GCN theo KHÔNG GIAN (SpatialGCNBlock) — không TCN
       trong bước này, xử lý độc lập từng frame.
    2) Sau khi pool theo node -> (B,T,C), CHIA ĐÔI kênh:
           nửa 1 -> TCN          (ngắn hạn)
           nửa 2 -> Transformer  (dài hạn)
    3) Ghép lại 2 nửa đã học -> pool theo T -> classifier.
    4) KHÔNG có cross-attention / 2-stream phức tạp như V5 (bỏ tạm phần đó).
    """

    def __init__(self, gcn_channels=(128, 128, 256, 256), gcn_decouple_p=4, tcn_kernel_size=9, num_transformer_layers=4,
                 nhead=8, dim_feedforward=1024, dropout=0.2, max_seq_len=5000, num_classes=1000, ):
        super().__init__()

        assert gcn_channels[-1] % 2 == 0, "kênh GCN cuối phải chia hết cho 2 để tách nửa TCN / nửa Transformer"

        self.num_nodes = _NUM_NODE
        self.register_buffer('adjacency_matrix', build_adjacency())

        # out_ch truyền vào đây hiện KHÔNG có tác dụng thật (xem ghi chú đầu
        # câu trả lời) — output thật của FusionStem vẫn là _COORD_DIM kênh.
        self.stem = FusionStem(_COORD_DIM, gcn_channels[0], self.num_nodes)

        ch = [_COORD_DIM, *gcn_channels]  # input đầu tiên của GCN = _COORD_DIM, khớp output thật của stem hiện tại
        self.gcn_blocks = nn.ModuleList(
            [SpatialGCNBlock(ch[i], ch[i + 1], self.num_nodes, self.adjacency_matrix, p=gcn_decouple_p, dropout=dropout)
             for i in range(len(ch) - 1)])

        gcn_out = gcn_channels[-1]
        self.half = gcn_out // 2

        self.short_term = ShortTermTCN(self.half, kernel_size=tcn_kernel_size, dropout=dropout)
        self.long_term = LongTermTransformer(self.half, nhead=nhead, dim_feedforward=dim_feedforward,
                                             num_layers=num_transformer_layers, dropout=dropout,
                                             max_seq_len=max_seq_len)

        self.classifier = nn.Sequential(nn.LayerNorm(gcn_out), nn.Dropout(dropout), nn.Linear(gcn_out, num_classes), )

    def encode_gcn(self, features):
        # features: (B, T, 3N, C_in) -> stem gộp 3 nhóm (position/shape/average) -> (B,T,N,_COORD_DIM)
        x = self.stem(features)

        for block in self.gcn_blocks:
            x = block(x)  # (B, T, N, C_out) — chỉ GCN, không TCN

        return x.mean(dim=-2)  # (B, T, C_out) — pool theo node

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")
        video_mask = video_mask.bool()

        B, T, _ = features.shape
        x = self.encode_gcn(features.clone().reshape(B, T, self.num_nodes * 3, _COORD_DIM))  # (B,T,gcn_out)

        x_short, x_long = x[..., :self.half], x[..., self.half:]

        x_short = self.short_term(x_short)  # (B,T,half) — ngắn hạn
        x_long = self.long_term(x_long, video_mask)  # (B,T,half) — dài hạn

        fused = torch.cat([x_short, x_long], dim=-1)  # (B,T,gcn_out)

        pooled = masked_mean_pool(fused, video_mask)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss


import torch.nn.functional as F


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
