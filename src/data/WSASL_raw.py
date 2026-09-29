import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import _COORD_DIM
from pipeline_config import MAX_FRAMES
from src.utils.rgb import normalize_imagenet

RGB_CACHE_FILENAME = "rgb_frames.npy"   # (N, 3, H, W) uint8
INDICES_FILENAME = "indices.npy"        # (N,) chỉ số frame trong không gian landmark (0..T-1)


def _read_split(annotation_dir, mode):
    with open(os.path.join(annotation_dir, f"{mode}.json"), "r") as f:
        data = json.load(f)
    return {str(d["video_id"]): d.get("gloss") for d in data}


class _LandmarkBase(Dataset):
    """Logic chung: đọc split, kiểm tra file, lấy nhãn, cache đặc trưng landmark."""

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

    # ---- hooks ----
    def _extra_sample_info(self, video_name):
        return {}, None

    # ---- helpers ----
    def _resolve_gloss(self, video_name, gloss_ann, feat_dir):
        """Ưu tiên gloss trong annotation (nguồn sự thật); fallback gloss.txt."""
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
            feats = self.fusion_component.fuse_follow_shape(pose, left, right)  # (T, D)
            if self.cache_landmarks:
                self._feature_cache[vid] = feats
        return feats

    def __len__(self):
        return len(self.samples)


class WLASLLandmarksDataset(_LandmarkBase):
    """Dùng cho selector: giữ nguyên toàn bộ T frame để chấm importance."""

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self._load_features(item), item["label"], item["video_name"]

def _sample_positions(n, max_frames, train):
    """Chọn tối đa max_frames vị trí theo thứ tự thời gian. Train: mỗi đoạn lấy ngẫu nhiên (temporal jitter)."""
    if n <= max_frames:
        return None
    if train:
        edges = np.linspace(0, n, max_frames + 1)
        pos = [np.random.randint(int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)) for i in range(max_frames)]
        return np.asarray(pos, dtype=np.int64)
    return np.linspace(0, n - 1, max_frames).round().astype(np.int64)


def _augment_clip(x):
    """x: float (T, 3, H, W) trong [0,1]. Cùng một phép biến đổi cho cả clip.
    Không lật ngang vì đổi tay thuận thay đổi nghĩa/kết quả của ký hiệu."""
    _, _, H, W = x.shape
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
    return x.clamp(0.0, 1.0)


class WLASLLandmarksRGBDataset(_LandmarkBase):
    """Landmark + RGB đã chọn frame.

    - RGB đọc từ <rgb_dir>/<video>/{rgb_frames.npy, indices.npy}.
    - indices.npy cho biết mỗi frame RGB ứng với frame landmark nào; landmark được lấy đúng các
      chỉ số đó nên pose[i] và rgb[i] LUÔN cùng thời điểm và cùng độ dài.
    - max_frames được áp dụng thật (train: jitter ngẫu nhiên, eval: đều).
    """

    def __init__(self, feature_dir, annotation_dir, rgb_dir, fusion_component, mode="train",
                 max_frames=MAX_FRAMES, augment=None, cache_landmarks=True):
        print(feature_dir)
        print(annotation_dir)
        print(rgb_dir)
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
        rgb = np.load(item["rgb_cache_path"], mmap_mode="r")  # (N, 3, H, W)

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

        return feats, x, item["label"], item["video_name"]