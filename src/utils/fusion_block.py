import numpy as np

from config import _COORD_DIM, _REMOVE_POSE_IDX

_NOSE_IDX = 0
_LEFT_SHOULDER_IDX = 11
_RIGHT_SHOULDER_IDX = 12

_EPS = 1e-6


class FusionComponent:
    def __init__(self):
        pass

    def fuse(self, pose_feature, left_feature, right_feature):

        T = pose_feature.shape[0]

        pose = pose_feature.reshape(T, 33, _COORD_DIM).copy()
        left = left_feature.reshape(T, 21, _COORD_DIM).copy()
        right = right_feature.reshape(T, 21, _COORD_DIM).copy()

        all_points = np.concatenate([pose, left, right], axis=1)  # (T, N, 2)

        left_present_mask = ~np.all(left == 0, axis=-1)
        right_present_mask = ~np.all(right == 0, axis=-1)
        pose_present_mask = ~np.all(pose == 0, axis=-1)

        all_present_mask = np.concatenate([pose_present_mask, left_present_mask, right_present_mask], axis=1)  # (T, N)

        valid_points = all_points * all_present_mask[..., None]

        valid_count = all_present_mask.sum(axis=1, keepdims=True)

        average_point = valid_points.sum(axis=1) / np.maximum(valid_count, 1)

        average_point = average_point.reshape(T, 1, _COORD_DIM)

        scale = np.linalg.norm(pose[:, _LEFT_SHOULDER_IDX] - pose[:, _RIGHT_SHOULDER_IDX], axis=-1, keepdims=True, )

        scale = np.where(scale > _EPS, scale, 1.0)

        scale = scale[:, np.newaxis, :]
        root = average_point

        pose = (pose - root) / scale
        left = (left - root) / scale
        right = (right - root) / scale 

        left = left * left_present_mask[..., None]
        right = right * right_present_mask[..., None]
        pose = pose * pose_present_mask[..., None]

        pose = np.delete(pose, _REMOVE_POSE_IDX, axis=1)

        fused_coords = np.concatenate([pose, left, right], axis=1)

        fused_flat = fused_coords.reshape(T, -1)  # (T, D)

        return fused_flat

    def fuse_follow_position(self, pose_feature, left_feature, right_feature):
        T = pose_feature.shape[0]

        pose = pose_feature.reshape(T, 33, _COORD_DIM).copy()
        left = left_feature.reshape(T, 21, _COORD_DIM).copy()
        right = right_feature.reshape(T, 21, _COORD_DIM).copy()

        pose_present_mask = ~np.all(pose == 0, axis=-1)  # [T, 33]
        left_present_mask = ~np.all(left == 0, axis=-1)  # [T, 21]
        right_present_mask = ~np.all(right == 0, axis=-1)  # [T, 21]

        scale = np.linalg.norm(pose[:, _LEFT_SHOULDER_IDX] - pose[:, _RIGHT_SHOULDER_IDX], axis=-1,
                               keepdims=True, )  # [T, 1]

        scale = np.where(scale > _EPS, scale, 1.0)
        scale = scale[:, np.newaxis, :]  # [T, 1, 1]

        root = (pose[:, np.newaxis, _LEFT_SHOULDER_IDX] + pose[:, np.newaxis, _RIGHT_SHOULDER_IDX]) / 2

        pose = (pose - root) / scale
        left = (left - root) / scale
        right = (right - root) / scale

        pose = pose * pose_present_mask[..., None]
        left = left * left_present_mask[..., None]
        right = right * right_present_mask[..., None]

        pose = np.delete(pose, _REMOVE_POSE_IDX, axis=1)

        fused_coords = np.concatenate([pose, left, right], axis=1)

        fused_flat = fused_coords.reshape(T, -1)

        return fused_flat

    def fuse_follow_shape(self, pose_feature, left_feature, right_feature):
        T = pose_feature.shape[0]

        pose = pose_feature.reshape(T, 33, _COORD_DIM).copy()
        left = left_feature.reshape(T, 21, _COORD_DIM).copy()
        right = right_feature.reshape(T, 21, _COORD_DIM).copy()

        left_present_mask = ~np.all(left == 0, axis=-1)
        right_present_mask = ~np.all(right == 0, axis=-1)
        pose_present_mask = ~np.all(pose == 0, axis=-1)

        pose_root = (pose[:, np.newaxis, _LEFT_SHOULDER_IDX] + pose[:, np.newaxis, _RIGHT_SHOULDER_IDX]) / 2
        left_root = left[:, 0:1, :]
        right_root = right[:, 0:1, :]

        scale = np.linalg.norm(pose[:, _LEFT_SHOULDER_IDX] - pose[:, _RIGHT_SHOULDER_IDX], axis=-1, keepdims=True, )

        scale = np.where(scale > _EPS, scale, 1.0)

        scale = scale[:, np.newaxis, :]

        pose = (pose - pose_root) / scale
        left = (left - left_root) / scale
        right = (right - right_root) / scale

        left = left * left_present_mask[..., None]
        right = right * right_present_mask[..., None]
        pose = pose * pose_present_mask[..., None]

        pose = np.delete(pose, _REMOVE_POSE_IDX, axis=1)

        fused_coords = np.concatenate([pose, left, right], axis=1)
        fused_flat = fused_coords.reshape(T, -1)  # (T, D)

        return fused_flat


    def normalize_by_wrist(self, hand_feature, wrist=None):

        if wrist is None:
            wrist = hand_feature[:, 0:1, :]

        present_mask = ~np.all(hand_feature == 0, axis=-1)  # [T, N]

        wrist_present = ~np.all(wrist == 0, axis=-1)  # [T, 1]
        valid_mask = present_mask & wrist_present  # [T, N]

        normalize_hand_feature = hand_feature - wrist

        normalize_hand_feature = normalize_hand_feature * valid_mask[..., None]

        return normalize_hand_feature

    def _shoulder_scale(self, pose_feature):
        T = pose_feature.shape[0]

        pose = np.asarray(pose_feature, dtype=np.float32).reshape(
            T, 33, _COORD_DIM
        )

        left_shoulder = pose[:, _LEFT_SHOULDER_IDX]
        right_shoulder = pose[:, _RIGHT_SHOULDER_IDX]

        shoulders_present = (
                np.any(left_shoulder != 0, axis=-1)
                & np.any(right_shoulder != 0, axis=-1)
        )

        scale = np.linalg.norm(
            left_shoulder - right_shoulder,
            axis=-1,
        )

        # Skip scaling if a shoulder is missing or the shoulders are too close together.
        valid = shoulders_present & np.isfinite(scale) & (scale > _EPS)
        scale = np.where(valid, scale, 1.0)

        return scale[:, None, None]  # (T, 1, 1)
