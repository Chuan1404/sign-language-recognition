import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    from src.training.train import masked_mean_pool
except ImportError:
    try:
        from training.train import masked_mean_pool
    except ImportError:
        def masked_mean_pool(x, video_mask):
            if video_mask is None:
                return x.mean(dim=1)
            mask = video_mask.unsqueeze(-1).float()
            summed = (x * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            return summed / counts

BACKBONES = {}


def register_backbone(*names):
    def deco(cls):
        for n in names:
            BACKBONES[n] = cls
        return cls
    return deco


@register_backbone("resnet18", "resnet34", "resnet50")
class ResNetBackbone(nn.Module):
    def __init__(self, name="resnet18", pretrained=True, **_):
        super().__init__()
        m = getattr(models, name)(weights="DEFAULT" if pretrained else None)
        self.out_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.net = m

    def forward(self, x):
        return self.net(x)


@register_backbone("vit_tiny", "vit_small", "vit_base")
class ViTBackbone(nn.Module):
    _NAMES = {
        "vit_tiny": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "vit_base": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    }

    def __init__(self, name="vit_tiny", pretrained=True, img_size=112, drop_path=0.1, **_):
        super().__init__()
        import timm
        self.net = timm.create_model(
            self._NAMES[name], pretrained=pretrained, num_classes=0,
            img_size=img_size, drop_path_rate=drop_path,
        )
        self.out_dim = self.net.num_features

    def forward(self, x):
        return self.net(x)


def build_backbone(name, pretrained=True, img_size=224, drop_path=0.1):
    if name not in BACKBONES:
        raise ValueError(f"Backbone '{name}' không có. Có: {list(BACKBONES)}")
    return BACKBONES[name](name=name, pretrained=pretrained,
                           img_size=img_size, drop_path=drop_path)


class RGBEncoder(nn.Module):

    def __init__(self, d_model=256, dropout=0.2, pretrained=True,
                 variant="resnet18", img_size=112, drop_path=0.1):
        super().__init__()
        self.backbone = build_backbone(variant, pretrained, img_size, drop_path)
        self.feat_dim = self.backbone.out_dim
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, rgb_frames, video_mask):
        B, T = rgb_frames.shape[:2]
        video_mask = video_mask.bool()
        if not video_mask.any():
            return self.proj(rgb_frames.new_zeros(B, T, self.feat_dim))
        feats = self.backbone(rgb_frames[video_mask])            # (N_valid, feat_dim)
        x = rgb_frames.new_zeros(B, T, self.feat_dim, dtype=feats.dtype)
        x[video_mask] = feats
        return self.proj(x)


TEMPORALS = {}


def register_temporal(name):
    def deco(cls):
        TEMPORALS[name] = cls
        return cls
    return deco


@register_temporal("none")
class TemporalIdentity(nn.Module):
    def __init__(self, d_model=256, **_):
        super().__init__()

    def forward(self, x, mask):
        return x


@register_temporal("bilstm")
class TemporalBiLSTM(nn.Module):
    def __init__(self, d_model=256, num_layers=6, dropout=0.2, **_):
        super().__init__()
        self.lstm = nn.LSTM(
            d_model, d_model // 2, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        T = x.size(1)
        lengths = mask.long().sum(1).clamp(min=1).cpu()          # tránh length = 0
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        return self.norm(out)


@register_temporal("transformer")
class TemporalTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2, dropout=0.2, max_len=128, **_):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.pos[:, : x.size(1)]
        x = self.enc(x, src_key_padding_mask=~mask)
        return self.norm(x)


@register_temporal("opticalflow")
class TemporalOpticalFlow(nn.Module):
    """Motion features from optical flow between consecutive frames, fused with the frame
    features and then passed to a base temporal model (default: BiLSTM).

    Flow comes from a frozen RAFT-small. Needs the raw frames, so it is called with frames=.
    """
    needs_frames = True

    def __init__(self, d_model=256, nhead=8, dropout=0.2, base="bilstm", pretrained=True, **_):
        super().__init__()
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        self.d_model = d_model
        self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT if pretrained else None)
        for p in self.raft.parameters():
            p.requires_grad = False
        self.flow_enc = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 32, 7, 2, 3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, d_model),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model),
        )
        self.base = build_temporal(base, d_model=d_model, nhead=nhead, dropout=dropout)

    def train(self, mode=True):
        super().train(mode)
        self.raft.eval()  # RAFT stays frozen and in eval mode
        return self

    def _flow_feat(self, frames, mask):
        # frames: (B, T, 3, H, W), mask: (B, T) bool -> (B, T, D); frame 0 and invalid pairs get zeros
        B, T = frames.shape[:2]
        pair = mask[:, 1:] & mask[:, :-1]                           # (B, T-1)
        if T < 2 or not pair.any():
            return frames.new_zeros(B, T, self.d_model)
        # RAFT expects [-1, 1]; rescale per video so it works with any input normalization
        lo = frames.amin(dim=(1, 2, 3, 4), keepdim=True)
        hi = frames.amax(dim=(1, 2, 3, 4), keepdim=True)
        frames = (frames - lo) / (hi - lo).clamp(min=1e-6) * 2 - 1
        f1, f2 = frames[:, :-1][pair], frames[:, 1:][pair]          # (N_pair, 3, H, W)
        with torch.no_grad(), torch.autocast(device_type=frames.device.type, enabled=False):
            flow = self.raft(f1.float(), f2.float())[-1]            # (N_pair, 2, H, W)
        feat = self.flow_enc(flow)
        out = feat.new_zeros(B, T, self.d_model)
        out[:, 1:][pair] = feat
        return out

    def forward(self, x, mask, frames=None):
        B, T = x.shape[:2]
        if frames.ndim == 6:                                        # two-hand input (B, T, 2, 3, H, W)
            K = frames.shape[2]
            frames = frames.transpose(1, 2).reshape(B * K, T, *frames.shape[3:])
            hand_mask = mask.unsqueeze(1).expand(B, K, T).reshape(B * K, T)
            flow = self._flow_feat(frames, hand_mask).reshape(B, K, T, -1).mean(1)
        else:
            flow = self._flow_feat(frames, mask)
        x = self.fuse(torch.cat([x, flow.to(x.dtype)], dim=-1))
        return self.base(x, mask)


def build_temporal(name, **kw):
    if name not in TEMPORALS:
        raise ValueError(f"Temporal '{name}' không có. Có: {list(TEMPORALS)}")
    return TEMPORALS[name](**kw)


class RGBModel(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        d_model=256,
        nhead=8,
        dropout=0.1,
        rgb_backbone="resnet18",
        temporal="bilstm",
        rgb_pretrained=True,
        img_size=224,
    ):
        super().__init__()
        self.rgb_encoder = RGBEncoder(
            d_model=d_model, dropout=dropout, pretrained=rgb_pretrained,
            variant=rgb_backbone, img_size=img_size,
        )
        self.temporal = build_temporal(temporal, d_model=d_model, nhead=nhead, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, pose_feature=None, rgb_feature=None, labels=None,
                feature_mask=None, rgb_mask=None):
        rgb_mask = rgb_mask.bool()
        x = self.rgb_encoder(rgb_feature, rgb_mask)
        x = self._run_temporal(x, rgb_mask, rgb_feature)
        pooled = masked_mean_pool(x, rgb_mask)
        logits = self.classifier(self.drop(pooled))
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}

    def _run_temporal(self, x, mask, frames):
        if getattr(self.temporal, "needs_frames", False):
            return self.temporal(x, mask, frames=frames)
        return self.temporal(x, mask)

    def freeze_encoders(self, freeze=True):
        for p in self.rgb_encoder.parameters():
            p.requires_grad = not freeze


class TwoHandRGBModel(RGBModel):
    def __init__(
        self,
        num_classes=1000,
        d_model=256,
        nhead=8,
        dropout=0.1,
        rgb_backbone="resnet18",
        temporal="bilstm",
        rgb_pretrained=True,
        img_size=224,
        share_encoder=True,
    ):
        super().__init__(
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            rgb_backbone=rgb_backbone,
            temporal=temporal,
            rgb_pretrained=rgb_pretrained,
            img_size=img_size,
        )
        self.share_encoder = share_encoder
        if not share_encoder:
            self.left_encoder = self.rgb_encoder
            self.right_encoder = RGBEncoder(
                d_model=d_model,
                dropout=dropout,
                pretrained=rgb_pretrained,
                variant=rgb_backbone,
                img_size=img_size,
            )

        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def freeze_encoders(self, freeze=True):
        if self.share_encoder:
            super().freeze_encoders(freeze=freeze)
        else:
            for p in self.left_encoder.parameters():
                p.requires_grad = not freeze
            for p in self.right_encoder.parameters():
                p.requires_grad = not freeze

    def _normalize_input(self, pose_feature=None, rgb_feature=None, left_rgb=None, right_rgb=None):
        # Trường hợp gọi model(rgb) với 1 positional argument thì pose_feature thực chất là rgb_feature
        if rgb_feature is None and pose_feature is not None:
            if isinstance(pose_feature, (tuple, list, dict)) or (torch.is_tensor(pose_feature) and pose_feature.ndim >= 4):
                rgb_feature = pose_feature
                pose_feature = None

        if rgb_feature is None and left_rgb is not None and right_rgb is not None:
            rgb_feature = (left_rgb, right_rgb)

        if isinstance(rgb_feature, (tuple, list)):
            left, right = rgb_feature[0], rgb_feature[1]
            rgb_feature = torch.stack([left, right], dim=2)  # (B, T, 2, 3, H, W)
        elif isinstance(rgb_feature, dict):
            rgb_feature = torch.stack([rgb_feature["left"], rgb_feature["right"]], dim=2)
        elif torch.is_tensor(rgb_feature):
            if rgb_feature.ndim == 6 and rgb_feature.shape[1] == 2 and rgb_feature.shape[2] != 2:
                # (B, 2, T, 3, H, W) -> (B, T, 2, 3, H, W)
                rgb_feature = rgb_feature.transpose(1, 2)
            elif rgb_feature.ndim == 5 and rgb_feature.shape[2] != 2:
                # Fallback crop đơn (B, T, 3, H, W) -> nhân đôi cho cả 2 tay
                rgb_feature = torch.stack([rgb_feature, rgb_feature], dim=2)

        return pose_feature, rgb_feature

    def forward(
        self,
        pose_feature=None,
        rgb_feature=None,
        labels=None,
        feature_mask=None,
        rgb_mask=None,
        left_rgb=None,
        right_rgb=None,
        **kwargs,
    ):
        pose_feature, rgb_feature = self._normalize_input(pose_feature, rgb_feature, left_rgb, right_rgb)
        if rgb_feature is None:
            raise ValueError("TwoHandRGBModel yêu cầu rgb_feature (hoặc left_rgb, right_rgb).")

        B, T = rgb_feature.shape[:2]
        if rgb_mask is None:
            rgb_mask = torch.ones((B, T), dtype=torch.bool, device=rgb_feature.device)
        else:
            rgb_mask = rgb_mask.bool()

        if self.share_encoder:
            # (B, T, 2, 3, H, W) -> (B, 2, T, 3, H, W) -> (2B, T, 3, H, W)
            hands = rgb_feature.transpose(1, 2).reshape(B * 2, T, *rgb_feature.shape[3:])
            hands_mask = rgb_mask.unsqueeze(1).expand(B, 2, T).reshape(B * 2, T)

            x = self.rgb_encoder(hands, hands_mask)                      # (2B, T, D)
            x = x.reshape(B, 2, T, -1).transpose(1, 2).reshape(B, T, -1)  # (B, T, 2D)
        else:
            left_frames = rgb_feature[:, :, 0]
            right_frames = rgb_feature[:, :, 1]
            x_left = self.left_encoder(left_frames, rgb_mask)
            x_right = self.right_encoder(right_frames, rgb_mask)
            x = torch.cat([x_left, x_right], dim=-1)                     # (B, T, 2D)

        x = self.fuse(x)                                                  # (B, T, D)

        x = self._run_temporal(x, rgb_mask, rgb_feature)
        pooled = masked_mean_pool(x, rgb_mask)
        logits = self.classifier(self.drop(pooled))
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}


# Ví dụ:
# RGBModel(rgb_backbone="resnet18", temporal="bilstm")       # thử nghiệm này
# RGBModel(rgb_backbone="resnet18", temporal="transformer")
# RGBModel(rgb_backbone="vit_tiny", temporal="bilstm")
# RGBModel(rgb_backbone="resnet18", temporal="none")         # chỉ mean pool
# TwoHandRGBModel(rgb_backbone="resnet18", temporal="bilstm")  # input (B, T, 2, 3, H, W)
# RGBModel(rgb_backbone="resnet18", temporal="opticalflow")      # RAFT flow + BiLSTM