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


class TwoHandRGBModel(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        d_model=256,
        nhead=8,
        dropout=0.1,
        rgb_backbone="resnet18",
        temporal="bilstm",
        rgb_pretrained=True,
        img_size=112,
        share_encoder=True,
    ):
        super().__init__()

        self.share_encoder = share_encoder

        def make_encoder():
            if rgb_backbone not in ("resnet18", "resnet34", "resnet50"):
                raise ValueError("Model này hiện chỉ hỗ trợ ResNet.")

            backbone = getattr(models, rgb_backbone)(
                weights="DEFAULT" if rgb_pretrained else None
            )

            # Dừng tại layer3, bỏ layer4, avgpool và fc.
            encoder = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
            )

            channels = 1024 if rgb_backbone == "resnet50" else 256
            return encoder, channels

        self.left_encoder, channels = make_encoder()

        if not share_encoder:
            self.right_encoder, _ = make_encoder()

        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        self.fuse = nn.Sequential(
            nn.Linear(channels * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        self.temporal = build_temporal(
            temporal,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        self.feature_channels = channels

    def _encode_frames(self, frames, mask, encoder):
        # frames: (B, T, 3, H, W)
        B, T = frames.shape[:2]

        if not mask.any():
            return frames.new_zeros(B, T, self.feature_channels)

        # Với input 224×224:
        # ResNet18/34 -> (N_valid, 256, 14, 14)
        feature_maps = encoder(frames[mask])

        vectors = self.spatial_pool(feature_maps).flatten(1)

        output = vectors.new_zeros(B, T, self.feature_channels)
        output[mask] = vectors

        return output

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
        # Hỗ trợ model(rgb_feature=...), model(rgb),
        # hoặc cách gọi hiện tại model(landmarks, rgb).
        if rgb_feature is None:
            if left_rgb is not None and right_rgb is not None:
                rgb_feature = (left_rgb, right_rgb)
            elif (
                torch.is_tensor(pose_feature)
                and pose_feature.ndim == 6
            ):
                rgb_feature = pose_feature
            elif isinstance(pose_feature, (tuple, list, dict)):
                rgb_feature = pose_feature

        if isinstance(rgb_feature, dict):
            rgb_feature = torch.stack(
                [rgb_feature["left"], rgb_feature["right"]],
                dim=2,
            )
        elif isinstance(rgb_feature, (tuple, list)):
            rgb_feature = torch.stack(rgb_feature, dim=2)

        if (
            not torch.is_tensor(rgb_feature)
            or rgb_feature.ndim != 6
            or rgb_feature.shape[2:4] != (2, 3)
        ):
            raise ValueError(
                "RGB phải có shape (B, T, 2, 3, H, W)."
            )

        B, T = rgb_feature.shape[:2]

        if rgb_mask is None:
            rgb_mask = torch.ones(
                B, T,
                dtype=torch.bool,
                device=rgb_feature.device,
            )
        else:
            rgb_mask = rgb_mask.to(
                device=rgb_feature.device,
                dtype=torch.bool,
            )

        if self.share_encoder:
            # Encode cả hai tay trong cùng một lần gọi.
            hands = rgb_feature.transpose(1, 2).reshape(
                B * 2, T, *rgb_feature.shape[3:]
            )
            hand_mask = rgb_mask[:, None].expand(
                B, 2, T
            ).reshape(B * 2, T)

            x = self._encode_frames(
                hands, hand_mask, self.left_encoder
            )

            x = x.reshape(
                B, 2, T, self.feature_channels
            ).transpose(1, 2).reshape(
                B, T, self.feature_channels * 2
            )
        else:
            left = self._encode_frames(
                rgb_feature[:, :, 0],
                rgb_mask,
                self.left_encoder,
            )
            right = self._encode_frames(
                rgb_feature[:, :, 1],
                rgb_mask,
                self.right_encoder,
            )
            x = torch.cat([left, right], dim=-1)

        x = self.fuse(x)
        x = x.masked_fill(~rgb_mask[..., None], 0.0)

        if getattr(self.temporal, "needs_frames", False):
            x = self.temporal(x, rgb_mask, frames=rgb_feature)
        else:
            x = self.temporal(x, rgb_mask)

        pooled = masked_mean_pool(x, rgb_mask)
        logits = self.classifier(self.drop(pooled))

        loss = (
            F.cross_entropy(logits, labels)
            if labels is not None
            else None
        )

        return {"logits": logits, "loss": loss}

    def freeze_encoders(self, freeze=True):
        encoders = [self.left_encoder]
        if not self.share_encoder:
            encoders.append(self.right_encoder)

        for encoder in encoders:
            for parameter in encoder.parameters():
                parameter.requires_grad = not freeze


# Ví dụ:
# RGBModel(rgb_backbone="resnet18", temporal="bilstm")       # thử nghiệm này
# RGBModel(rgb_backbone="resnet18", temporal="transformer")
# RGBModel(rgb_backbone="vit_tiny", temporal="bilstm")
# RGBModel(rgb_backbone="resnet18", temporal="none")         # chỉ mean pool
# TwoHandRGBModel(rgb_backbone="resnet18", temporal="bilstm")  # input (B, T, 2, 3, H, W)
# RGBModel(rgb_backbone="resnet18", temporal="opticalflow")      # RAFT flow + BiLSTM

class ISLR_Multimodal(nn.Module):
    """Pose/RGB classification with weighted probability late fusion.

    Pose input: [shape, average, position], (B, T_pose, 3 * N * C).
    RGB input: (B, T_rgb, 2, 3, H, W), or a left/right tuple or dict.
    Both branches must use the same class-index mapping.
    Returned logits are normalized log probabilities of the mixture.
    """

    def __init__(
        self,
        num_classes=1000,
        pose_kwargs=None,
        pose_checkpoint=None,
        rgb_checkpoint=None,
        freeze_pose=False,
        freeze_rgb=False,
        rgb_backbone="resnet18",
        temporal="bilstm",
        rgb_d_model=256,
        rgb_nhead=8,
        rgb_dropout=0.1,
        rgb_pretrained=True,
        img_size=112,
        share_encoder=True,
        pose_weight=0.5,
        pose_temperature=1.0,
        rgb_temperature=1.0,
    ):
        super().__init__()
        import math
        from src.models.gcn import ISLR_Transformer_GCN

        if not math.isfinite(pose_weight) or not 0.0 <= pose_weight <= 1.0:
            raise ValueError("pose_weight must be between 0 and 1")
        if any(not math.isfinite(t) or t <= 0 for t in (pose_temperature, rgb_temperature)):
            raise ValueError("Temperatures must be finite and positive")

        pose_config = dict(pose_kwargs or {})
        if "num_classes" in pose_config and pose_config["num_classes"] != num_classes:
            raise ValueError("Pose and RGB must have the same num_classes")
        pose_config["num_classes"] = num_classes
        self.pose_model = ISLR_Transformer_GCN(**pose_config)
        self.rgb_model = TwoHandRGBModel(
            num_classes=num_classes,
            d_model=rgb_d_model,
            nhead=rgb_nhead,
            dropout=rgb_dropout,
            rgb_backbone=rgb_backbone,
            temporal=temporal,
            rgb_pretrained=rgb_pretrained if rgb_checkpoint is None else False,
            img_size=img_size,
            share_encoder=share_encoder,
        )
        self.register_buffer("pose_weight", torch.tensor(float(pose_weight)))
        self.register_buffer("pose_temperature", torch.tensor(float(pose_temperature)))
        self.register_buffer("rgb_temperature", torch.tensor(float(rgb_temperature)))

        if pose_checkpoint is not None or rgb_checkpoint is not None:
            if pose_checkpoint is None or rgb_checkpoint is None:
                raise ValueError("Supply both pose_checkpoint and rgb_checkpoint")
            self.load_branch_weights(pose_checkpoint, rgb_checkpoint)
        self.set_branch_trainable(not freeze_pose, not freeze_rgb)

    @staticmethod
    def _checkpoint_state(path):
        # Load weights only; optimizer/scheduler state is intentionally ignored.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint: {path}")
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if not isinstance(state, dict):
            raise ValueError(f"Invalid model state: {path}")
        return state

    def load_branch_weights(self, pose_checkpoint, rgb_checkpoint):
        """Strict load: constructor settings and class mapping must match training."""
        pose_state = self._checkpoint_state(pose_checkpoint)
        rgb_state = self._checkpoint_state(rgb_checkpoint)
        # Validate both before modifying either branch.
        for name, branch, state in (
            ("pose", self.pose_model, pose_state),
            ("RGB", self.rgb_model, rgb_state),
        ):
            expected = branch.state_dict()
            if expected.keys() != state.keys():
                missing = sorted(expected.keys() - state.keys())
                unexpected = sorted(state.keys() - expected.keys())
                raise ValueError(f"{name} checkpoint mismatch: missing={missing}, unexpected={unexpected}")
            bad_shapes = [key for key in expected if expected[key].shape != state[key].shape]
            if bad_shapes:
                raise ValueError(f"{name} checkpoint shape mismatch: {bad_shapes}")
        self.pose_model.load_state_dict(pose_state, strict=True)
        self.rgb_model.load_state_dict(rgb_state, strict=True)

    def set_branch_trainable(self, pose=True, rgb=True):
        self.freeze_pose = not pose
        self.freeze_rgb = not rgb
        for parameter in self.pose_model.parameters():
            parameter.requires_grad = pose
        for parameter in self.rgb_model.parameters():
            parameter.requires_grad = rgb
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "freeze_pose", False):
            self.pose_model.eval()
        if getattr(self, "freeze_rgb", False):
            self.rgb_model.eval()
        return self

    @staticmethod
    def _video_mask(mask, batch_size, length, device, name):
        if mask is None:
            mask = torch.ones(batch_size, length, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device=device, dtype=torch.bool)
        if mask.shape != (batch_size, length):
            raise ValueError(f"{name} must have shape {(batch_size, length)}")
        if not mask.any(dim=1).all():
            raise ValueError(f"{name}: each video must have at least one valid frame")
        return mask

    def forward(
        self,
        pose_feature=None,
        rgb_feature=None,
        labels=None,
        feature_mask=None,
        rgb_mask=None,
        left_rgb=None,
        right_rgb=None,
    ):
        if pose_feature is None or pose_feature.ndim != 3:
            raise ValueError("pose_feature must have shape (B, T, 3 * N * C)")
        expected_dim = 3 * self.pose_model.feature_dim
        if pose_feature.shape[-1] != expected_dim:
            raise ValueError(f"Pose feature dimension must be {expected_dim}")

        if rgb_feature is None and left_rgb is not None and right_rgb is not None:
            rgb_feature = (left_rgb, right_rgb)
        if isinstance(rgb_feature, dict):
            rgb_feature = torch.stack([rgb_feature["left"], rgb_feature["right"]], dim=2)
        elif isinstance(rgb_feature, (tuple, list)):
            rgb_feature = torch.stack(rgb_feature, dim=2)
        if not torch.is_tensor(rgb_feature) or rgb_feature.ndim != 6 or rgb_feature.shape[2:4] != (2, 3):
            raise ValueError("rgb_feature must have shape (B, T, 2, 3, H, W)")
        if rgb_feature.shape[0] != pose_feature.shape[0]:
            raise ValueError("Pose and RGB batch sizes must match")
        if rgb_feature.device != pose_feature.device:
            raise ValueError("Pose and RGB must be on the same device")

        B, T_pose = pose_feature.shape[:2]
        feature_mask = self._video_mask(feature_mask, B, T_pose, pose_feature.device, "feature_mask")
        rgb_mask = self._video_mask(rgb_mask, B, rgb_feature.shape[1], rgb_feature.device, "rgb_mask")
        # Packed BiLSTM expects valid frames followed by right padding.
        if (rgb_mask[:, 1:] & ~rgb_mask[:, :-1]).any():
            raise ValueError("rgb_mask must use right padding; no gaps between valid frames")

        pose_output = self.pose_model(pose_feature, video_mask=feature_mask)
        rgb_output = self.rgb_model(rgb_feature=rgb_feature, rgb_mask=rgb_mask)
        pose_logits = pose_output["logits"]
        rgb_logits = rgb_output["logits"]

        # Log-space mixture avoids underflow; fusion happens after classification.
        pose_log_probs = F.log_softmax(pose_logits / self.pose_temperature, dim=-1)
        rgb_log_probs = F.log_softmax(rgb_logits / self.rgb_temperature, dim=-1)
        log_probs = torch.logaddexp(
            pose_log_probs + self.pose_weight.log(),
            rgb_log_probs + (1.0 - self.pose_weight).log(),
        )
        loss = F.nll_loss(log_probs, labels) if labels is not None else None

        return {
            "logits": log_probs,
            "probabilities": log_probs.exp(),
            "loss": loss,
            "pose_logits": pose_logits,
            "rgb_logits": rgb_logits,
            "pose_probabilities": pose_log_probs.exp(),
            "rgb_probabilities": rgb_log_probs.exp(),
            "important_score": pose_output["important_score"],
            "frame_importance": pose_output["frame_importance"],
        }

