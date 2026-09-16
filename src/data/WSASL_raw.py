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
                    {"left_hand_path": left_hand_path, "right_hand_path": right_hand_path, "pose_path": pose_path,
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

        position_features = self.fusion_component.fuse_follow_position(pose_features, left_features, right_features)
        vel_features = self.fusion_component.fuse_follow_velocity(pose_features, left_features, right_features)
        shape_features = self.fusion_component.fuse_follow_shape(pose_features, left_features, right_features)

        features = np.concatenate([position_features, shape_features, vel_features], axis=-1)
        return features, label_id
