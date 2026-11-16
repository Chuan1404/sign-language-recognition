import json
import os
from collections import Counter

import numpy as np
from torch.utils.data import Dataset

from config import _COORD_DIM, _NUM_NODE
from src.utils import FusionComponent
from src.data.video_filter import excluded_video_ids
from src.data.augmentation import rotate_hand_blocks

NUM_JOINTS = _NUM_NODE



def _read_split(annotation_dir, mode):
    with open(os.path.join(annotation_dir, f"{mode}.json"), "r") as f:
        data = json.load(f)
    return {str(d["video_id"]): d["gloss"] for d in data}


class WLASLLandmarksDataset(Dataset):
    _LANDMARK_KEYS = ("left_hand", "right_hand", "pose")

    def __init__(self, feature_dir, annotation_dir, fusion_component=None, mode="train", cache_landmarks=True):
        self.feature_dir = feature_dir
        self.fusion_component = FusionComponent() if fusion_component is None else fusion_component
        self.mode = mode
        self.cache_landmarks = cache_landmarks
        self._feature_cache = {}

        with open(os.path.join(annotation_dir, "gloss2idx.json"), "r") as f:
            self.gloss2idx = json.load(f)

        entries = _read_split(annotation_dir, mode)
        self.samples = []
        self.skipped = Counter()
        excluded = excluded_video_ids()

        for video_name in sorted(entries):
            if video_name in excluded:
                self.skipped["excluded_video_id"] += 1
                continue
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

            self.samples.append(
                {"video_name": video_name, "label": self.gloss2idx[gloss], **paths}
            )

        self._report()

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
        print(f"[{name}] mode={self.mode}: {len(self.samples)} valid samples.")
        for reason, n in self.skipped.items():
            print(f"[{name}]   - {reason}: {n}")
        if not self.samples:
            raise RuntimeError(f"[{name}] No samples found for mode='{self.mode}'. Check the paths and cache.")

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

            # Each block has _NUM_NODE * _COORD_DIM features.
            feature_blocks = [shape_feature, average_feature, position_feature]
            feats = np.concatenate(feature_blocks, axis=-1).astype(np.float32)

            if self.cache_landmarks:
                self._feature_cache[vid] = feats

        return feats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self._load_features(item), item["label"], item["video_name"]
