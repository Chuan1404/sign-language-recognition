"""Load hand contours produced by WLASL_hand_contour_extract.py.

Example::

    dataset = WLASLHandContourDataset(
        feature_dir="datasets/processed/wlasl100_hand_contour_64",
        pose_dir="D:/SignDetection/datasets/processed/wlasl_features_v2",
        annotation_dir="D:/SignDetection/datasets/annotations/WLASL100",
        mode="train",
    )
    features, label, video_id = dataset[0]

Features are float32 (T, FEATURE_DIM), ordered [pose, left hand, right hand].
Pose nodes follow config._REMOVE_POSE_IDX; each hand has 64 x/y nodes.
All nodes use the shoulder midpoint as origin and shoulder distance as scale,
as in fuse_follow_position. Missing pose nodes and invalid hands stay zero.
All source frames are preserved. pose_dir defaults to feature_dir and must
contain <video_id>/pose.npy with the same frame order as the contours.
"""

import json
import os
from collections import Counter

import numpy as np
from torch.utils.data import Dataset

from config import _REMOVE_POSE_IDX
from src.data.video_filter import excluded_video_ids


NODES_PER_HAND = 64
NODES_PER_POSE = 33 - len(_REMOVE_POSE_IDX)
NUM_JOINTS = NODES_PER_POSE + 2 * NODES_PER_HAND
COORD_DIM = 2
FEATURE_DIM = NUM_JOINTS * COORD_DIM


class WLASLHandContourDataset(Dataset):
    """Return (contour_features, label, video_name), as in WSASL_raw."""

    def __init__(
        self,
        feature_dir,
        annotation_dir,
        mode="train",
        cache_landmarks=True,
        pose_dir=None,
    ):
        self.feature_dir = feature_dir
        self.pose_dir = feature_dir if pose_dir is None else pose_dir
        self.mode = mode
        self.cache_landmarks = cache_landmarks
        self._feature_cache = {}

        with open(os.path.join(annotation_dir, "gloss2idx.json"), encoding="utf-8") as handle:
            self.gloss2idx = json.load(handle)
        with open(os.path.join(annotation_dir, f"{mode}.json"), encoding="utf-8") as handle:
            entries = {str(entry["video_id"]): entry["gloss"] for entry in json.load(handle)}

        self.samples = []
        self.skipped = Counter()
        excluded = excluded_video_ids()
        for video_name, gloss in sorted(entries.items()):
            if video_name in excluded:
                self.skipped["excluded_video_id"] += 1
                continue
            directory = os.path.join(feature_dir, video_name)
            if not os.path.isdir(directory):
                self.skipped["no_feature_dir"] += 1
                continue
            paths = {
                f"{key}_path": os.path.join(directory, f"{key}.npy")
                for key in ("left_hand", "right_hand", "valid")
            }
            if not all(os.path.isfile(path) for path in paths.values()):
                self.skipped["missing_contour_file"] += 1
                continue
            paths["pose_path"] = os.path.join(self.pose_dir, video_name, "pose.npy")
            if not os.path.isfile(paths["pose_path"]):
                self.skipped["missing_pose_file"] += 1
                continue
            # Labels come from split annotations, as in WSASL_raw.
            if gloss not in self.gloss2idx:
                self.skipped["unknown_gloss"] += 1
                continue
            self.samples.append({
                "video_name": video_name,
                "label": self.gloss2idx[gloss],
                **paths,
            })

        name = type(self).__name__
        print(f"[{name}] mode={mode}: {len(self.samples)} samples.")
        for reason, count in self.skipped.items():
            print(f"[{name}]   - {reason}: {count}")
        if not self.samples:
            raise RuntimeError(
                f"[{name}] No contour samples for mode={mode!r}. "
                "Check feature_dir, pose_dir, annotation_dir and extracted features."
            )

    @property
    def labels(self):
        return [item["label"] for item in self.samples]

    def __len__(self):
        return len(self.samples)

    def _load_features(self, item):
        video_name = item["video_name"]
        if video_name in self._feature_cache:
            return self._feature_cache[video_name]

        left = np.load(item["left_hand_path"], allow_pickle=False)
        right = np.load(item["right_hand_path"], allow_pickle=False)
        valid = np.load(item["valid_path"], allow_pickle=False)
        pose = np.load(item["pose_path"], allow_pickle=False)

        frames = len(left)
        if len(right) != frames or len(pose) != frames or valid.shape != (frames, 2):
            raise ValueError(
                f"{video_name}: pose, contours and valid must have matching frame counts "
                f"and order; got left={left.shape}, right={right.shape}, "
                f"pose={pose.shape}, valid={valid.shape}"
            )
        if pose.shape[1:] not in ((99,), (66,), (33, 3), (33, 2)):
            raise ValueError(f"{video_name}: expected 33 pose nodes with x/y or x/y/z, got {pose.shape}")
        pose = pose.reshape(frames, 33, -1)[..., :COORD_DIM].astype(np.float32)
        pose_present = np.any(pose != 0, axis=-1)
        root = (pose[:, 11:12] + pose[:, 12:13]) / 2
        scale = np.linalg.norm(pose[:, 11] - pose[:, 12], axis=-1)
        scale = np.where(scale > 1e-6, scale, 1.0)[:, None, None]
        pose = (pose - root) / scale
        pose[~pose_present] = 0.0
        pose = np.delete(pose, _REMOVE_POSE_IDX, axis=1)

        hands = np.stack([left, right], axis=1).astype(np.float32)
        hands = hands.reshape(frames, 2, NODES_PER_HAND, COORD_DIM)
        hands = (hands - root[:, None]) / scale[:, None]
        hands[~valid.astype(bool)] = 0.0

        features = np.concatenate([pose, hands[:, 0], hands[:, 1]], axis=1)
        features = features.reshape(frames, FEATURE_DIM)
        if self.cache_landmarks:
            self._feature_cache[video_name] = features
        return features

    def __getitem__(self, idx):
        item = self.samples[idx]
        # A caller's augmentation must not modify cached source features.
        return self._load_features(item).copy(), item["label"], item["video_name"]
