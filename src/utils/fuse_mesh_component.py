"""The keypoint shape/average/position normalization for 85-node hands."""

import numpy as np

from config import _COORD_DIM, _REMOVE_POSE_IDX

MESH_NODES = 64
MEDIAPIPE_NODES = 21
NODES_PER_HAND = MESH_NODES + MEDIAPIPE_NODES
NODES_PER_POSE = 33 - len(_REMOVE_POSE_IDX)
NUM_JOINTS = NODES_PER_POSE + 2 * NODES_PER_HAND
FEATURE_DIM = NUM_JOINTS * _COORD_DIM


class FuseMeshComponent:
    """Match FusionComponent, using the original MP wrist at hand index 64.

    Input hands are normalized image x/y, ordered [64 contour, 21 MediaPipe].
    Zero denotes missing points. Points below the hips are retained.
    Inputs are never mutated.
    """

    def __init__(self, num_hand_nodes=NODES_PER_HAND, wrist_index=MESH_NODES):
        if not 0 <= wrist_index < num_hand_nodes:
            raise ValueError("wrist_index must belong to the hand")
        self.num_hand_nodes = num_hand_nodes
        self.wrist_index = wrist_index

    def _fuse(self, pose_feature, left_feature, right_feature, mode):
        frames = len(pose_feature)
        pose = np.asarray(pose_feature).reshape(frames, 33, _COORD_DIM).copy()
        left = np.asarray(left_feature).reshape(frames, self.num_hand_nodes, _COORD_DIM).copy()
        right = np.asarray(right_feature).reshape(frames, self.num_hand_nodes, _COORD_DIM).copy()
        present = [~np.all(points == 0, axis=-1) for points in (pose, left, right)]
        scale = np.linalg.norm(pose[:, 11] - pose[:, 12], axis=-1)
        scale = np.where(scale > 1e-6, scale, 1.0)[:, None, None]
        shoulder_root = (pose[:, 11:12] + pose[:, 12:13]) / 2
        if mode == "shape":
            wrist = self.wrist_index
            roots = (shoulder_root, left[:, wrist:wrist + 1].copy(), right[:, wrist:wrist + 1].copy())
        elif mode == "average":
            points = np.concatenate([pose, left, right], axis=1)
            mask = np.concatenate(present, axis=1)
            root = ((points * mask[..., None]).sum(axis=1)
                    / np.maximum(mask.sum(axis=1, keepdims=True), 1))[:, None]
            roots = (root, root, root)
        else:
            roots = (shoulder_root, shoulder_root, shoulder_root)
        normalized = [(points - root) / scale * mask[..., None]
                      for points, root, mask in zip((pose, left, right), roots, present)]
        normalized[0] = np.delete(normalized[0], _REMOVE_POSE_IDX, axis=1)
        return np.concatenate(normalized, axis=1).reshape(len(pose), -1).astype(np.float32)

    def fuse(self, pose_feature, left_feature, right_feature):
        return self._fuse(pose_feature, left_feature, right_feature, "average")

    def fuse_follow_shape(self, pose_feature, left_feature, right_feature):
        return self._fuse(pose_feature, left_feature, right_feature, "shape")

    def fuse_follow_position(self, pose_feature, left_feature, right_feature):
        return self._fuse(pose_feature, left_feature, right_feature, "position")
