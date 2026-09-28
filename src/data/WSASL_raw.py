import json
import os

import numpy as np
import torch
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

            if (os.path.exists(left_hand_path) and os.path.exists(right_hand_path) and os.path.exists(
                    pose_path) and os.path.exists(text_path)):
                self.samples.append(
                    {"video_name": video_name, "left_hand_path": left_hand_path, "right_hand_path": right_hand_path, "pose_path": pose_path,
                     "text_path": text_path, })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        item = self.samples[idx]

        left_features = np.load(item["left_hand_path"])
        right_features = np.load(item["right_hand_path"])
        pose_features = np.load(item["pose_path"])

        T = left_features.shape[0]

        pose_features = pose_features.reshape(T, 33, 3)[:, :, :_COORD_DIM]
        left_features = left_features.reshape(T, 21, 3)[:, :, :_COORD_DIM]
        right_features = right_features.reshape(T, 21, 3)[:, :, :_COORD_DIM]

        with open(item["text_path"], "r", encoding="utf-8") as f:
            gloss = f.read().strip()
        label_id = self.gloss2idx[gloss]

        # position_features = self.fusion_component.fuse_follow_position(pose_features, left_features, right_features)
        shape_features = self.fusion_component.fuse_follow_shape(pose_features, left_features, right_features)
        # average_feature = self.fusion_component.fuse(pose_features, left_features, right_features)

        features = np.concatenate([shape_features], axis=-1)
        return features, label_id


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# crop_mode -> (thư mục con trong rgb_dir, hậu tố tên file). Phải khớp với crop_video.py
_RGB_LAYOUT = {
    None: ("full", ("",)),
    "body": ("body", ("",)),
    "full_hand": ("full_hand", ("_right", "_left")),  # trục 1 của rgb: [right, left]
}


class WLASLLandmarksRGBDataset(Dataset):
    """
    Trả về (features, rgb_frames, label_id, video_id)
        features   : (T, D)  landmarks đã fuse
        rgb_frames : ảnh đã crop sẵn bởi crop_video.py, chuẩn hoá ImageNet, frame i khớp frame i của pose
                       crop_mode None / "body" : (T, 3, H, W)
                       crop_mode "full_hand"   : (T, 2, 3, H, W)   [right, left]
        label_id   : int
        video_id   : str, hoặc ["<video>_right", "<video>_left"] với "full_hand"

    Dataset KHÔNG đọc video, KHÔNG crop. Chạy crop_video.py trước để tạo rgb_dir/<crop_mode>/*.npy.

    max_frames: nếu video dài hơn thì lấy mẫu đều xuống max_frames (áp dụng cho CẢ pose và RGB
                để giữ khớp nhau, đồng thời giới hạn VRAM). None = giữ nguyên.
    """

    def __init__(self, feature_dir, annotation_dir, rgb_dir, fusion_component, mode="train", max_frames=64,
                 crop_mode=None):
        assert crop_mode in _RGB_LAYOUT, f"crop_mode không hợp lệ: {crop_mode}"
        self.feature_dir = feature_dir
        self.fusion_component = fusion_component
        self.max_frames = max_frames
        self.crop_mode = crop_mode

        subdir, self.suffixes = _RGB_LAYOUT[crop_mode]
        self.rgb_dir = os.path.join(rgb_dir, subdir)

        self.samples = []

        with open(os.path.join(annotation_dir, "gloss2idx.json"), "r") as f:
            self.gloss2idx = json.load(f)

        with open(os.path.join(annotation_dir, f"{mode}.json"), "r") as f:
            data = json.load(f)
            all_video_names = sorted(d["video_id"] for d in data)

        skipped = 0
        for video_name in all_video_names:
            feat_dir = os.path.join(feature_dir, video_name)
            if not os.path.isdir(feat_dir):
                continue

            paths = {"left_hand_path": os.path.join(feat_dir, "left_hand.npy"),
                     "right_hand_path": os.path.join(feat_dir, "right_hand.npy"),
                     "pose_path": os.path.join(feat_dir, "pose.npy"),
                     "text_path": os.path.join(feat_dir, "gloss.txt")}
            rgb_paths = [os.path.join(self.rgb_dir, f"{video_name}{s}.npy") for s in self.suffixes]

            if not all(os.path.exists(p) for p in [*paths.values(), *rgb_paths]):
                skipped += 1
                continue

            self.samples.append({"video_name": video_name, "rgb_paths": rgb_paths, **paths})

        print(f"[{mode}] {len(self.samples)} samples (bỏ qua {skipped} video thiếu file landmarks/RGB)")

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

        position_features = self.fusion_component.fuse_follow_position(pose_features, left_features, right_features)
        shape_features = self.fusion_component.fuse_follow_shape(pose_features, left_features, right_features)
        average_feature = self.fusion_component.fuse(pose_features, left_features, right_features)
        features = np.concatenate([position_features, shape_features, average_feature], axis=-1)  # (T, D)

        # chọn frame (lấy mẫu đều nếu T quá dài)
        if self.max_frames is not None and T > self.max_frames:
            sel = np.linspace(0, T - 1, self.max_frames).round().astype(int)
        else:
            sel = np.arange(T)

        # ---- RGB: chỉ load ảnh đã crop sẵn (mmap -> chỉ đọc các frame được chọn) ----
        streams = []
        for path in item["rgb_paths"]:
            arr = np.load(path, mmap_mode="r")  # (T, H, W, 3) uint8
            n = arr.shape[0]
            # crop_video.py lưu đúng T frame nên n == T; giữ ánh xạ này phòng khi file được tạo bằng cách khác
            vid_idx = sel if n == T else np.round(sel * (n - 1) / max(T - 1, 1)).astype(int).clip(0, n - 1)
            streams.append(np.asarray(arr[vid_idx]))
        rgb = streams[0] if len(streams) == 1 else np.stack(streams, axis=1)  # (T,H,W,3) | (T,2,H,W,3)

        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        # channel-last -> channel-first: (T,H,W,3)->(T,3,H,W) | (T,2,H,W,3)->(T,2,3,H,W)
        rgb = torch.from_numpy(np.ascontiguousarray(np.moveaxis(rgb, -1, -3)))

        if self.crop_mode == "full_hand":
            video_id = [f"{video_id}{s}" for s in self.suffixes]

        return torch.from_numpy(features[sel]).float(), rgb, label_id, video_id


def collate_fn_rgb(batch):
    """
    -> (features (B,T,D), rgb (B,T,3,H,W) | (B,T,2,3,H,W), video_mask (B,T) bool [True = frame thật],
        labels (B,), video_ids list)
    """
    feats, rgbs, labels, video_ids = zip(*batch)

    B = len(batch)
    lengths = [f.shape[0] for f in feats]
    T = max(lengths)

    features = torch.zeros(B, T, *feats[0].shape[1:])
    rgb = torch.zeros(B, T, *rgbs[0].shape[1:])
    video_mask = torch.zeros(B, T, dtype=torch.bool)

    for i, (f, r, L) in enumerate(zip(feats, rgbs, lengths)):
        features[i, :L] = f
        rgb[i, :L] = r
        video_mask[i, :L] = True

    return features, rgb, video_mask, torch.tensor(labels, dtype=torch.long), list(video_ids)