import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import _COORD_DIM


class WLASLLandmarksDataset(Dataset):
    def __init__(self, feature_dir, annotation_dir, fusion_component, mode="train"):
        self.feature_dir = feature_dir
        self.fusion_component = fusion_component

        self.samples = []

        with open(os.path.join(annotation_dir, "gloss2idx.json"), "r") as f:
            self.gloss2idx = json.load(f)

        with open(os.path.join(annotation_dir, f"{mode}.json"), "r") as f:
            data = json.load(f)
            video_ids = [d["video_id"] for d in data]
            all_video_names = sorted(video_ids)

        for index, video_name in enumerate(all_video_names):
            video_dir = os.path.join(feature_dir, video_name)

            if not os.path.isdir(video_dir):
                continue

            left_hand_path = os.path.join(video_dir, "left_hand.npy")
            right_hand_path = os.path.join(video_dir, "right_hand.npy")
            pose_path = os.path.join(video_dir, "pose.npy")

            text_path = os.path.join(video_dir, "gloss.txt")

            if (
                os.path.exists(left_hand_path)
                and os.path.exists(right_hand_path)
                and os.path.exists(pose_path)
                and os.path.exists(text_path)
            ):
                self.samples.append(
                    {
                        "video_name": video_name,
                        "left_hand_path": left_hand_path,
                        "right_hand_path": right_hand_path,
                        "pose_path": pose_path,
                        "text_path": text_path,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        item = self.samples[idx]

        left_features = np.load(item["left_hand_path"])
        right_features = np.load(item["right_hand_path"])
        pose_features = np.load(item["pose_path"])
        video_id = item["video_name"]

        T = left_features.shape[0]

        pose_features = pose_features.reshape(T, 33, 3)[:, :, :_COORD_DIM]
        left_features = left_features.reshape(T, 21, 3)[:, :, :_COORD_DIM]
        right_features = right_features.reshape(T, 21, 3)[:, :, :_COORD_DIM]

        with open(item["text_path"], "r", encoding="utf-8") as f:
            gloss = f.read().strip()
        label_id = self.gloss2idx[gloss]

        # position_features = self.fusion_component.fuse_follow_position(pose_features, left_features, right_features)
        shape_features = self.fusion_component.fuse_follow_shape(
            pose_features, left_features, right_features
        )
        # average_feature = self.fusion_component.fuse(pose_features, left_features, right_features)

        features = np.concatenate([shape_features], axis=-1)
        return features, label_id, video_id


from torchvision import transforms

# Tên file cache RGB được tạo bởi preprocess_rgb.py
_RGB_CACHE_FILENAME = "rgb_frames.npy"


class WLASLLandmarksRGBDataset(Dataset):
    """Dataset kết hợp landmark features và RGB frames.

    Args:
        feature_dir: Thư mục chứa các thư mục feature (left_hand.npy, ...).
        annotation_dir: Thư mục chứa gloss2idx.json và {mode}.json.
        rgb_dir: Thư mục chứa các thư mục ảnh thô (mỗi video một thư mục).
            Chỉ dùng khi ``use_cache=False``.
        fusion_component: Object xử lý fusion landmark.
        mode: ``"train"`` hoặc ``"test"``.
        max_frames: Số frame tối đa (chưa dùng đến, để dành cho future use).
        crop_mode: Chế độ crop (chưa dùng đến).
        use_cache: Nếu ``True`` (mặc định), load RGB từ file ``rgb_frames.npy``
            đã được pre-process bởi ``preprocess_rgb.py``.
            Nếu ``False``, load và transform ảnh trực tiếp trong lúc train
            (chậm hơn đáng kể).
    """

    def __init__(
        self,
        feature_dir,
        annotation_dir,
        rgb_dir,
        fusion_component,
        mode="train",
        max_frames=64,
        crop_mode=None,
        use_cache=True,
    ):
        self.feature_dir = feature_dir
        self.fusion_component = fusion_component
        self.max_frames = max_frames
        self.crop_mode = crop_mode
        self.rgb_dir = rgb_dir
        self.use_cache = use_cache

        # Transform chỉ dùng khi use_cache=False (on-the-fly loading)
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        with open(os.path.join(annotation_dir, "gloss2idx.json"), "r") as f:
            self.gloss2idx = json.load(f)

        with open(os.path.join(annotation_dir, f"{mode}.json"), "r") as f:
            data = json.load(f)
            all_video_names = sorted(d["video_id"] for d in data)

        self.samples = []
        missing_cache = 0

        for video_name in all_video_names:
            feat_dir = os.path.join(feature_dir, video_name)
            if not os.path.isdir(feat_dir):
                continue

            rgb_cache_path = os.path.join(feat_dir, _RGB_CACHE_FILENAME)

            # Khi dùng cache, bỏ qua video chưa được pre-process
            if use_cache and not os.path.exists(rgb_cache_path):
                missing_cache += 1
                continue

            self.samples.append(
                {
                    "video_name": video_name,
                    "feat_dir": feat_dir,
                    "left_hand_path": os.path.join(feat_dir, "left_hand.npy"),
                    "right_hand_path": os.path.join(feat_dir, "right_hand.npy"),
                    "pose_path": os.path.join(feat_dir, "pose.npy"),
                    "text_path": os.path.join(feat_dir, "gloss.txt"),
                    "rgb_cache_path": rgb_cache_path,
                    "rgb_path": os.path.join(rgb_dir, video_name),
                }
            )

        if use_cache and missing_cache > 0:
            print(
                f"[WLASLLandmarksRGBDataset] Cảnh báo: bỏ qua {missing_cache} video "
                f"chưa có {_RGB_CACHE_FILENAME}. "
                f"Hãy chạy preprocess_rgb.py trước."
            )

    def __len__(self):
        return len(self.samples)

    def _load_rgb_cached(self, item) -> torch.Tensor:
        """Load RGB từ file .npy đã pre-process. Trả về tensor (T, C, H, W)."""
        arr = np.load(item["rgb_cache_path"])  # (T, C, H, W), float32
        return torch.from_numpy(arr)

    def _load_rgb_online(self, item) -> torch.Tensor:
        """Load và transform ảnh trực tiếp từ thư mục. Trả về tensor (T, C, H, W)."""
        frames = []
        for filename in sorted(os.listdir(item["rgb_path"])):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(item["rgb_path"], filename)
            image = Image.open(path).convert("RGB")
            frames.append(self.rgb_transform(image))
        return torch.stack(frames, dim=0)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # --- Landmark features ---
        left_features = np.load(item["left_hand_path"])
        right_features = np.load(item["right_hand_path"])
        pose_features = np.load(item["pose_path"])
        video_id = item["video_name"]

        T = left_features.shape[0]
        pose_features = pose_features.reshape(T, 33, 3)[:, :, :_COORD_DIM]
        left_features = left_features.reshape(T, 21, 3)[:, :, :_COORD_DIM]
        right_features = right_features.reshape(T, 21, 3)[:, :, :_COORD_DIM]

        with open(item["text_path"], "r", encoding="utf-8") as f:
            gloss = f.read().strip()

        label_id = self.gloss2idx[gloss]
        shape_features = self.fusion_component.fuse_follow_shape(
            pose_features, left_features, right_features
        )
        features = np.concatenate([shape_features], axis=-1)  # (T, D)

        # --- RGB frames ---
        if self.use_cache:
            rgbs = self._load_rgb_cached(item)
        else:
            rgbs = self._load_rgb_online(item)

        return features, rgbs, label_id, video_id
