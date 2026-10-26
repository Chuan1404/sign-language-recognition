"""Load the new 85-point hands and fuse shape/average/position features."""

import os

import numpy as np

from src.data.WSASL_hand_contour_raw import WLASLHandContourDataset
from src.training.train import collate_fn
from src.utils.fuse_mesh_component import FuseMeshComponent, NODES_PER_HAND, FEATURE_DIM


class WLASLMeshDataset(WLASLHandContourDataset):
    """Return (features, label, video_id); pose_dir only supplies pose.npy.

    Each frame has three FEATURE_DIM blocks: shape, average, position.
    No old left_hand/right_hand landmark files are loaded from pose_dir.
    """

    def __init__(self, feature_dir, annotation_dir, mode="train", pose_dir=None,
                 cache_landmarks=True, fusion_component=None):
        super().__init__(feature_dir, annotation_dir, mode=mode,
                         cache_landmarks=cache_landmarks, pose_dir=pose_dir)
        self.fusion_component = fusion_component or FuseMeshComponent()

    def _load_features(self, item):
        video_id = item['video_name']
        if video_id in self._feature_cache:
            return self._feature_cache[video_id]
        left, right, pose = [np.load(item[key], allow_pickle=False)
                             for key in ('left_hand_path', 'right_hand_path', 'pose_path')]
        frames = len(left)
        expected = ((NODES_PER_HAND * 2,), (NODES_PER_HAND, 2))
        if left.shape[1:] not in expected or right.shape[1:] not in expected:
            raise ValueError(f'{video_id}: expected 85 points (64 mesh + 21 MediaPipe) per hand; '
                             'rerun WLASL_hand_contour_extract.py into wlasl100_hand_mesh_85')
        if frames == 0 or len(right) != frames or len(pose) != frames:
            raise ValueError(f'{video_id}: mesh and pose must have matching nonempty source frames')
        if pose.shape[1:] not in ((99,), (66,), (33, 3), (33, 2)):
            raise ValueError(f'{video_id}: expected 33 pose nodes, got {pose.shape}')
        mask_path = os.path.join(self.feature_dir, video_id, 'keypoint_valid.npy')
        if not os.path.isfile(mask_path):
            raise ValueError(f'{video_id}: missing keypoint_valid.npy; re-extract the new mesh format')
        mask = np.load(mask_path, allow_pickle=False)
        if mask.shape != (frames, 2, NODES_PER_HAND):
            raise ValueError(f'{video_id}: invalid per-point validity shape {mask.shape}')
        pose = pose.reshape(frames, 33, -1)[..., :2].astype(np.float32)
        hands = np.stack([left.reshape(frames, NODES_PER_HAND, 2),
                          right.reshape(frames, NODES_PER_HAND, 2)], axis=1).astype(np.float32)
        hands[~mask.astype(bool)] = 0
        if not np.isfinite(hands).all() or not np.isfinite(pose).all():
            raise ValueError(f'{video_id}: nonfinite coordinates in valid mesh/pose points')
        left, right = hands[:, 0], hands[:, 1]
        fusion = self.fusion_component
        features = np.concatenate([
            fusion.fuse_follow_shape(pose, left, right),
            fusion.fuse(pose, left, right),
            fusion.fuse_follow_position(pose, left, right),
        ], axis=-1).astype(np.float32)
        if features.shape != (frames, 3 * FEATURE_DIM):
            raise ValueError(f'{video_id}: unexpected fused feature shape {features.shape}')
        if self.cache_landmarks:
            self._feature_cache[video_id] = features
        return features


collate_mesh = collate_fn
