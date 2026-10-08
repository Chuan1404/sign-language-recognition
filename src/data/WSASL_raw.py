import json
import os
import re
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import _COORD_DIM, _NUM_NODE
from pipeline_config import IMAGE_SIZE, MAX_FRAMES
from src.utils.rgb import normalize_imagenet
from src.utils import FusionComponent

NUM_JOINTS = _NUM_NODE

RGB_CACHE_FILENAME = "rgb_frames.npy"
INDICES_FILENAME = "indices.npy"


def _read_split(annotation_dir, mode):
    with open(os.path.join(annotation_dir, f"{mode}.json"), "r") as f:
        data = json.load(f)
    return {str(d["video_id"]): d["gloss"] for d in data}

class _LandmarkBase(Dataset):
    _LANDMARK_KEYS = ("left_hand", "right_hand", "pose")

    def __init__(self, feature_dir, annotation_dir, fusion_component, mode="train", cache_landmarks=True):
        self.feature_dir = feature_dir
        self.fusion_component = fusion_component
        self.mode = mode
        self.cache_landmarks = cache_landmarks
        self._feature_cache = {}

        with open(os.path.join(annotation_dir, "gloss2idx.json"), "r") as f:
            self.gloss2idx = json.load(f)

        entries = _read_split(annotation_dir, mode)
        self.samples = []
        self.skipped = Counter()

        for video_name in sorted(entries):
            feat_dir = os.path.join(feature_dir, video_name)
            if not os.path.isdir(feat_dir):
                self.skipped["no_feature_dir"] += 1
                continue

            paths = {f"{k}_path": os.path.join(feat_dir, f"{k}.npy") for k in self._LANDMARK_KEYS}
            if not all(os.path.exists(p) for p in paths.values()):
                self.skipped["missing_landmark_file"] += 1
                continue

            gloss = self._resolve_gloss(video_name, entries[video_name], feat_dir)
            if gloss is None or gloss not in self.gloss2idx:
                self.skipped["unknown_gloss"] += 1
                continue

            extra, reason = self._extra_sample_info(video_name)
            if extra is None:
                self.skipped[reason] += 1
                continue

            self.samples.append(
                {"video_name": video_name, "label": self.gloss2idx[gloss], **paths, **extra}
            )

        self._report()

    def _extra_sample_info(self, video_name):
        return {}, None

    def _resolve_gloss(self, video_name, gloss_ann, feat_dir):
        text_path = os.path.join(feat_dir, "gloss.txt")
        gloss_txt = None
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                gloss_txt = f.read().strip()
        if gloss_ann is not None:
            if gloss_txt is not None and gloss_txt != gloss_ann:
                self.skipped["gloss_mismatch(ann_used)"] += 1
            return gloss_ann
        return gloss_txt

    def _report(self):
        name = type(self).__name__
        print(f"[{name}] mode={self.mode}: {len(self.samples)} mẫu hợp lệ.")
        for reason, n in self.skipped.items():
            print(f"[{name}]   - {reason}: {n}")
        if not self.samples:
            raise RuntimeError(f"[{name}] Không có mẫu nào cho mode='{self.mode}'. Kiểm tra đường dẫn/cache.")

    @property
    def labels(self):
        return [s["label"] for s in self.samples]

    def _load_features(self, item):
        vid = item["video_name"]
        feats = self._feature_cache.get(vid)

        if feats is None:
            left = np.load(item["left_hand_path"])
            right = np.load(item["right_hand_path"])
            pose = np.load(item["pose_path"])

            T = left.shape[0]

            pose = pose.reshape(T, 33, 3)[:, :, :_COORD_DIM]
            left = left.reshape(T, 21, 3)[:, :, :_COORD_DIM]
            right = right.reshape(T, 21, 3)[:, :, :_COORD_DIM]

            shape_feature = self.fusion_component.fuse_follow_shape(
                pose, left, right
            )
            average_feature = self.fusion_component.fuse(
                pose, left, right
            )
            position_feature = self.fusion_component.fuse_follow_position(
                pose, left, right
            )

            # Mỗi block có D = _NUM_NODE * _COORD_DIM.
            feats = np.concatenate(
                [shape_feature, average_feature, position_feature],
                axis=-1,
            ).astype(np.float32)

            if self.cache_landmarks:
                self._feature_cache[vid] = feats

        return feats

    def __len__(self):
        return len(self.samples)

class WLASLLandmarksDataset(_LandmarkBase):
    def __init__(
        self,
        feature_dir,
        annotation_dir,
        fusion_component=None,
        mode="train",
        cache_landmarks=True,
        indices_dir=None,
        importance_json=None,
        only_important_frames=False,
    ):
        self.indices_dir = indices_dir
        self.importance_json = importance_json
        self.only_important_frames = only_important_frames or (indices_dir is not None)
        self.importance_data = None
        if self.importance_json and os.path.exists(self.importance_json):
            with open(self.importance_json, "r") as f:
                self.importance_data = json.load(f)

        if fusion_component is None:
            fusion_component = FusionComponent()


        super().__init__(feature_dir, annotation_dir, fusion_component, mode, cache_landmarks)

    def _extra_sample_info(self, video_name):
        if self.only_important_frames and self.indices_dir:
            idx_path = os.path.join(self.indices_dir, str(video_name), INDICES_FILENAME)
            if os.path.exists(idx_path):
                return {"indices_path": idx_path}, None
            if self.importance_data and str(video_name) in self.importance_data:
                return {"video_id_str": str(video_name)}, None
            return None, "missing_indices"
        return {}, None

    def __getitem__(self, idx):
        item = self.samples[idx]
        feature = self._load_features(item)

        if self.only_important_frames:
            if "indices_path" in item:
                indices = np.load(item["indices_path"])
                if len(indices) > 0 and indices.max() < len(feature):
                    feature = feature[indices]
                elif len(indices) > 0:
                    valid_idx = indices[indices < len(feature)]
                    if len(valid_idx) > 0:
                        feature = feature[valid_idx]
            elif self.importance_data and str(item["video_name"]) in self.importance_data:
                scores = np.asarray(self.importance_data[str(item["video_name"])]["importance"])
                T = len(feature)
                k = int(np.sum(scores > (1.5 / max(T, 1))))
                k = int(np.clip(k, min(8, T), min(100, T)))
                top_indices = np.sort(np.argsort(-scores)[:k])
                feature = feature[top_indices]

        return feature, item["label"], item["video_name"]

class WLASLRelatively(_LandmarkBase):
    def __init__(
        self,
        feature_dir,
        annotation_dir,
        fusion_component=None,
        mode="train",
        cache_landmarks=True,
        indices_dir=None,
        importance_json=None,
        only_important_frames=False,
    ):
        self.indices_dir = indices_dir
        self.importance_json = importance_json
        self.only_important_frames = only_important_frames or (indices_dir is not None)
        self.importance_data = None
        if self.importance_json and os.path.exists(self.importance_json):
            with open(self.importance_json, "r") as f:
                self.importance_data = json.load(f)

        if fusion_component is None:
            fusion_component = FusionComponent()


        super().__init__(feature_dir, annotation_dir, fusion_component, mode, cache_landmarks)

class WLASLImportantLandmarksDataset(WLASLLandmarksDataset):
    """
    Dataset WLASL Landmarks trích xuất CHỈ TRÊN CÁC FRAME IMPORTANT.
    Phục vụ trực tiếp cho mô hình GCN (như ISLR_GCN, ISLR_Transformer_GCN)
    để chỉ học trên các frame chuyển động cử chỉ quan trọng.
    """
    def __init__(
        self,
        feature_dir,
        annotation_dir,
        indices_dir=None,
        fusion_component=None,
        mode="train",
        cache_landmarks=True,
        importance_json=None,
    ):
        if indices_dir is None:
            try:
                from pipeline_config import SELECTED_RGB_DIR
                indices_dir = SELECTED_RGB_DIR
            except Exception:
                indices_dir = os.path.join("outputs", "selected_rgb")
        super().__init__(
            feature_dir=feature_dir,
            annotation_dir=annotation_dir,
            fusion_component=fusion_component,
            mode=mode,
            cache_landmarks=cache_landmarks,
            indices_dir=indices_dir,
            importance_json=importance_json,
            only_important_frames=True,
        )

def _sample_positions(n, max_frames, train):
    if n <= max_frames:
        return None
    if train:
        edges = np.linspace(0, n, max_frames + 1)
        pos = [np.random.randint(int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)) for i in range(max_frames)]
        return np.asarray(pos, dtype=np.int64)
    return np.linspace(0, n - 1, max_frames).round().astype(np.int64)


def _augment_clip(x):
    orig_shape = x.shape
    if x.ndim == 5:
        T, num_hands, C, H, W = orig_shape
        x = x.reshape(T * num_hands, C, H, W)
    elif x.ndim == 4:
        T, C, H, W = orig_shape
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {orig_shape}")

    scale = np.random.uniform(0.8, 1.0)
    ch, cw = int(round(H * scale)), int(round(W * scale))
    top = np.random.randint(0, H - ch + 1)
    left = np.random.randint(0, W - cw + 1)
    if (ch, cw) != (H, W):
        x = x[:, :, top:top + ch, left:left + cw]
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    brightness = np.random.uniform(0.8, 1.2)
    contrast = np.random.uniform(0.8, 1.2)
    x = x * brightness
    mean = x.mean()
    x = (x - mean) * contrast + mean
    x = x.clamp(0.0, 1.0)

    if len(orig_shape) == 5:
        x = x.reshape(orig_shape)
    return x


class WLASLLandmarksRGBDataset(_LandmarkBase):

    def __init__(self, feature_dir, annotation_dir, rgb_dir, fusion_component, mode="train",
                 max_frames=MAX_FRAMES, augment=None, cache_landmarks=True):
        self.rgb_dir = rgb_dir
        self.max_frames = max_frames
        self.augment = (mode == "train") if augment is None else augment
        super().__init__(feature_dir, annotation_dir, fusion_component, mode, cache_landmarks)

    def _extra_sample_info(self, video_name):
        d = os.path.join(self.rgb_dir, video_name)
        rgb_path = os.path.join(d, RGB_CACHE_FILENAME)
        idx_path = os.path.join(d, INDICES_FILENAME)
        if not os.path.exists(rgb_path):
            return None, "missing_rgb_cache(chạy extract_important_frame.py)"
        if not os.path.exists(idx_path):
            return None, "missing_indices(cache cũ, cần extract lại)"
        return {"rgb_cache_path": rgb_path, "indices_path": idx_path}, None

    def __getitem__(self, idx):
        item = self.samples[idx]
        feats = self._load_features(item)                     # (T, D)
        indices = np.load(item["indices_path"])               # (N,)
        rgb = np.load(item["rgb_cache_path"], mmap_mode="r")  # (N, 3, H, W) or (N, 2, 3, H, W)

        if len(indices) != len(rgb):
            raise ValueError(f"{item['video_name']}: indices ({len(indices)}) != rgb ({len(rgb)})")
        if indices.max() >= len(feats):
            raise ValueError(
                f"{item['video_name']}: indices vượt độ dài landmark ({indices.max()} >= {len(feats)}). "
                f"Landmark và cache RGB được tạo từ hai phiên bản dữ liệu khác nhau."
            )

        feats = feats[indices]
        positions = _sample_positions(len(indices), self.max_frames, self.augment)
        if positions is not None:
            feats = feats[positions]
            rgb_arr = rgb[positions]
        else:
            rgb_arr = np.asarray(rgb)

        frames = torch.from_numpy(np.ascontiguousarray(rgb_arr))
        if frames.dtype == torch.uint8:
            x = frames.float() / 255.0
            if self.augment:
                x = _augment_clip(x)
            x = normalize_imagenet(x)
        else:
            x = frames.float()  # cache cũ đã chuẩn hóa sẵn (float32)

        leading_shape = x.shape[:-3]

        x = F.interpolate(
            x.reshape(-1, *x.shape[-3:]),
            size=(112, 112),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        x = x.reshape(*leading_shape, 3, 112, 112)

        return feats, x, item["label"], item["video_name"]

        return feats, x, item["label"], item["video_name"]


class WLASLTwoHandRGBDataset(_LandmarkBase):
    def __init__(
        self,
        feature_dir,
        annotation_dir,
        rgb_dir,
        fusion_component=None,
        mode="train",
        max_frames=MAX_FRAMES,
        augment=None,
        cache_landmarks=True,
        image_size=IMAGE_SIZE,
        return_tuple_hands=False,
    ):
        self.rgb_dir = rgb_dir
        self.max_frames = max_frames
        self.augment = (mode == "train") if augment is None else augment
        self.image_size = image_size
        self.return_tuple_hands = return_tuple_hands

        if fusion_component is None:
            try:
                from src.utils import FusionComponent
                fusion_component = FusionComponent()
            except Exception:
                pass

        super().__init__(feature_dir, annotation_dir, fusion_component, mode, cache_landmarks)

    def _find_image_pairs(self, video_name):
        d = os.path.join(self.rgb_dir, str(video_name))
        if not os.path.isdir(d):
            return None, "missing_video_dir"

        preview_dir = os.path.join(d, "preview")
        search_dirs = [preview_dir, d] if os.path.isdir(preview_dir) else [d]

        idx_path = os.path.join(d, INDICES_FILENAME)
        has_idx_file = os.path.exists(idx_path)
        cached_indices = np.load(idx_path) if has_idx_file else None

        image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        for sdir in search_dirs:
            if not os.path.isdir(sdir):
                continue
            all_files = os.listdir(sdir)
            left_files = {}
            for f in all_files:
                f_lower = f.lower()
                if any(f_lower.endswith(ext) for ext in image_exts):
                    base, ext = os.path.splitext(f)
                    if base.endswith("_left"):
                        prefix = base[:-5]
                        left_files[prefix] = (f, ext)

            if left_files:
                pairs = []
                for prefix, (l_file, ext) in left_files.items():
                    r_file = f"{prefix}_right{ext}"
                    r_path = os.path.join(sdir, r_file) if os.path.exists(os.path.join(sdir, r_file)) else None
                    l_path = os.path.join(sdir, l_file)

                    m = re.search(r"(\d+)$", prefix)
                    frame_idx = int(m.group(1)) if m else None
                    pairs.append((frame_idx, prefix, l_path, r_path))

                if all(p[0] is not None for p in pairs):
                    pairs.sort(key=lambda p: p[0])
                else:
                    pairs.sort(key=lambda p: p[1])

                indices = [p[0] if p[0] is not None else i for i, p in enumerate(pairs)]
                return {
                    "image_pairs": [(p[2], p[3]) for p in pairs],
                    "indices": np.array(indices, dtype=np.int64),
                    "indices_path": idx_path if has_idx_file else None,
                }, None

        # Fallback tới rgb_frames.npy nếu có
        rgb_path = os.path.join(d, RGB_CACHE_FILENAME)
        if os.path.exists(rgb_path) and has_idx_file:
            return {
                "rgb_cache_path": rgb_path,
                "indices_path": idx_path,
                "indices": cached_indices,
            }, None

        return None, "missing_two_hand_images(_left/_right)"

    def _extra_sample_info(self, video_name):
        return self._find_image_pairs(video_name)

    def _load_hand_image(self, path):
        if path is not None and os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                    img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                return img.transpose(2, 0, 1)  # (3, H, W)
        return np.zeros((3, self.image_size, self.image_size), dtype=np.uint8)

    def __getitem__(self, idx):
        item = self.samples[idx]
        feats = self._load_features(item)

        if "image_pairs" in item:
            image_pairs = list(item["image_pairs"])
            indices = np.array(item["indices"], dtype=np.int64)

            if len(feats) > 0 and len(indices) > 0 and indices.max() >= len(feats):
                valid = indices < len(feats)
                indices = indices[valid]
                image_pairs = [p for p, v in zip(image_pairs, valid) if v]

            if len(image_pairs) == 0:
                raise ValueError(f"{item['video_name']}: Không có frame hợp lệ sau khi lọc indices.")

            positions = _sample_positions(len(image_pairs), self.max_frames, self.augment)
            if positions is not None:
                image_pairs = [image_pairs[p] for p in positions]
                indices = indices[positions]

            feats = feats[indices]

            frames = []
            for l_path, r_path in image_pairs:
                l_img = self._load_hand_image(l_path)
                r_img = self._load_hand_image(r_path)
                frames.append(np.stack([l_img, r_img], axis=0))

            rgb_arr = np.stack(frames, axis=0)  # (N_selected, 2, 3, H, W) uint8
            frames_t = torch.from_numpy(np.ascontiguousarray(rgb_arr))
            x = frames_t.float() / 255.0
            if self.augment:
                x = _augment_clip(x)
            x = normalize_imagenet(x)

        else:
            indices = np.load(item["indices_path"])
            rgb = np.load(item["rgb_cache_path"], mmap_mode="r")

            if len(feats) > 0 and len(indices) > 0 and indices.max() >= len(feats):
                valid = indices < len(feats)
                indices = indices[valid]
                rgb = rgb[valid]

            feats = feats[indices]
            positions = _sample_positions(len(indices), self.max_frames, self.augment)
            if positions is not None:
                feats = feats[positions]
                rgb_arr = rgb[positions]
            else:
                rgb_arr = np.asarray(rgb)

            frames_t = torch.from_numpy(np.ascontiguousarray(rgb_arr))
            if frames_t.dtype == torch.uint8:
                x = frames_t.float() / 255.0
                if self.augment:
                    x = _augment_clip(x)
                x = normalize_imagenet(x)
            else:
                x = frames_t.float()

        if self.return_tuple_hands:
            # (T, 3, H, W), (T, 3, H, W)
            return feats, (x[:, 0], x[:, 1]), item["label"], item["video_name"]

        return feats, x, item["label"], item["video_name"]

class MultiFeatureSkeletonAugmentor:
    """Feature: (T, 3 * NUM_JOINTS * _COORD_DIM).
    Thứ tự block: [shape, average, position].
    """

    def __init__(
        self,
        rotation_deg=10.0,
        scale_range=(0.9, 1.1),
        noise_std=0.005,
        noise_prob=0.5,
        speed_perturb_prob=0.5,
        speed_range=(0.8, 1.25),
        frame_dropout_prob=0.0,
        max_frame_dropout_ratio=0.1,
        rng=None,
    ):
        self.rotation_deg = rotation_deg
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.speed_perturb_prob = speed_perturb_prob
        self.speed_range = speed_range
        self.frame_dropout_prob = frame_dropout_prob
        self.max_frame_dropout_ratio = max_frame_dropout_ratio
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, fused):
        fused = np.array(fused, dtype=np.float32, copy=True)

        block_dim = NUM_JOINTS * _COORD_DIM
        expected_dim = 3 * block_dim

        if fused.ndim != 2 or fused.shape[1] != expected_dim:
            raise ValueError(
                f"Expected (T, {expected_dim}), got {fused.shape}"
            )

        if len(fused) == 0:
            return fused

        # Lấy nguyên frame để cả ba block luôn đồng bộ.
        if self.rng.random() < self.speed_perturb_prob:
            fused = self._speed_perturb(fused)

        if self.rng.random() < self.frame_dropout_prob:
            fused = self._drop_frames(fused)

        T = len(fused)
        blocks = fused.reshape(T, 3, NUM_JOINTS, _COORD_DIM)

        # Một góc xoay và một scale chung cho cả clip.
        angle = np.deg2rad(
            self.rng.uniform(-self.rotation_deg, self.rotation_deg)
        )
        cos, sin = np.cos(angle), np.sin(angle)

        if _COORD_DIM == 2:
            rotation = np.array(
                [[cos, -sin], [sin, cos]],
                dtype=np.float32,
            )
        elif _COORD_DIM == 3:
            rotation = np.array(
                [[cos, -sin, 0],
                 [sin, cos, 0],
                 [0, 0, 1]],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unsupported coord dim: {_COORD_DIM}")

        # Giữ các vector 0: có thể là điểm thiếu hoặc điểm gốc.
        nonzero_mask = np.any(blocks != 0, axis=-1)

        blocks = blocks @ rotation.T
        blocks *= self.rng.uniform(*self.scale_range)

        if (
            self.noise_std > 0
            and self.rng.random() < self.noise_prob
        ):
            # Noise chung theo joint giữa ba block.
            noise = self.rng.normal(
                0.0,
                self.noise_std,
                size=(T, 1, NUM_JOINTS, _COORD_DIM),
            ).astype(np.float32)

            blocks += noise * nonzero_mask[..., None]

        blocks = np.where(
            nonzero_mask[..., None], blocks, 0.0
        )

        return blocks.reshape(T, expected_dim).astype(np.float32)

    def _speed_perturb(self, fused):
        T = len(fused)
        if T <= 2:
            return fused

        speed = self.rng.uniform(*self.speed_range)
        new_T = max(2, int(round(T / speed)))

        indices = np.rint(
            np.linspace(0, T - 1, new_T)
        ).astype(np.int64)

        return fused[indices].copy()

    def _drop_frames(self, fused):
        T = len(fused)
        if T <= 2:
            return fused

        max_drop = min(
            int(T * self.max_frame_dropout_ratio),
            T - 2,
        )
        if max_drop < 1:
            return fused

        n_drop = int(self.rng.integers(1, max_drop + 1))
        drop_indices = self.rng.choice(
            T, size=n_drop, replace=False
        )

        keep = np.ones(T, dtype=bool)
        keep[drop_indices] = False

        return fused[keep].copy()

# Alias tiện dụng
WLASLLandmarksTwoHandRGBDataset = WLASLTwoHandRGBDataset