import numpy as np

from config import _REMOVE_POSE_IDX, _COORD_DIM

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

        valid_points = (all_points * all_present_mask[..., None])

        valid_count = all_present_mask.sum(axis=1, keepdims=True)

        average_point = (valid_points.sum(axis=1) / np.maximum(valid_count, 1))

        average_point = average_point.reshape(T, 1, _COORD_DIM)

        scale = np.linalg.norm(pose[:, _LEFT_SHOULDER_IDX] - pose[:, _RIGHT_SHOULDER_IDX], axis=-1, keepdims=True)

        scale = np.where(scale > _EPS, scale, 1.0)

        scale = scale[:, np.newaxis, :]
        # root = pose[:, np.newaxis, _NOSE_IDX]
        # root = (pose[:, np.newaxis, _LEFT_SHOULDER_IDX] - pose[:, np.newaxis, _RIGHT_SHOULDER_IDX]) / 2
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
            keepdims=True)  # [T, 1]

        scale = np.where(scale > _EPS, scale, 1.0)
        scale = scale[:, np.newaxis, :]  # [T, 1, 1]

        # pose_root = (pose[:, np.newaxis, _LEFT_SHOULDER_IDX] + pose[:, np.newaxis, _RIGHT_SHOULDER_IDX]) / 2  # [T, 1, 3]
        pose_root = (pose[:, np.newaxis, _LEFT_SHOULDER_IDX] + pose[:, np.newaxis, _RIGHT_SHOULDER_IDX]) / 2

        left_root = left[:, 0:1, :]  # [T, 1, 3]
        right_root = right[:, 0:1, :]  # [T, 1, 3]

        pose_root_present = (pose_present_mask[:, _LEFT_SHOULDER_IDX] & pose_present_mask[:, _RIGHT_SHOULDER_IDX])[
            :, None, None]  # [T, 1, 1]

        left_root_present = (left_present_mask[:, 0])[:, None, None]  # [T, 1, 1]

        right_root_present = (right_present_mask[:, 0])[:, None, None]  # [T, 1, 1]

        # --------------------------------------------------
        # 5. Calculate common root
        # --------------------------------------------------
        root_sum = (pose_root * pose_root_present + left_root * left_root_present + right_root * right_root_present)

        root_count = (
                pose_root_present.astype(np.float32) + left_root_present.astype(np.float32) + right_root_present.astype(
            np.float32))

        root = root_sum / np.maximum(root_count, 1.0)

        # --------------------------------------------------
        # 6. Root + scale normalization
        # --------------------------------------------------
        pose = (pose - root) / scale
        left = (left - root) / scale
        right = (right - root) / scale

        # --------------------------------------------------
        # 7. Restore missing landmarks to zero
        # --------------------------------------------------
        pose = pose * pose_present_mask[..., None]
        left = left * left_present_mask[..., None]
        right = right * right_present_mask[..., None]

        # --------------------------------------------------
        # 8. Remove unnecessary pose landmarks
        # --------------------------------------------------
        pose = np.delete(pose, _REMOVE_POSE_IDX, axis=1)

        # --------------------------------------------------
        # 9. Fuse
        # --------------------------------------------------
        fused_coords = np.concatenate([pose, left, right], axis=1)

        fused_flat = fused_coords.reshape(T, -1)

        return fused_flat

    def fuse_follow_velocity(self, pose_feature, left_feature, right_feature):
        T = pose_feature.shape[0]

        pose = pose_feature.reshape(T, 33, _COORD_DIM)
        left = left_feature.reshape(T, 21, _COORD_DIM)
        right = right_feature.reshape(T, 21, _COORD_DIM)

        pose = np.delete(pose, _REMOVE_POSE_IDX, axis=1)
        p = np.concatenate([pose, left, right], axis=1).astype(np.float32)  # (T, N, C)

        # mặt nạ: node có mặt (không phải toàn 0)
        mask = ~np.all(p == 0, axis=-1)  # (T, N)
        m = mask[..., None]  # (T, N, 1)

        # vận tốc: frame trước - frame sau
        v = np.zeros_like(p)
        v[:-1] = (p[:-1] - p[1:]) * (m[:-1] & m[1:])

        # # gia tốc: vận tốc trước - vận tốc sau
        # a = np.zeros_like(p)
        # a[:-1] = (v[:-1] - v[1:]) * (m[:-1] & m[1:])

        # features = np.concatenate([v, a], axis=-1).astype(np.float32)  # (T, N, 3*C)
        v = v.reshape(T, -1)

        return v

    def fuse_follow_shape(self, pose_feature, left_feature, right_feature):

        T = pose_feature.shape[0]

        pose = pose_feature.reshape(T, 33, _COORD_DIM).copy()
        left = left_feature.reshape(T, 21, _COORD_DIM).copy()
        right = right_feature.reshape(T, 21, _COORD_DIM).copy()

        left_present_mask = ~np.all(left == 0, axis=-1)
        right_present_mask = ~np.all(right == 0, axis=-1)
        pose_present_mask = ~np.all(pose == 0, axis=-1)

        scale = np.linalg.norm(pose[:, _LEFT_SHOULDER_IDX] - pose[:, _RIGHT_SHOULDER_IDX], axis=-1, keepdims=True)

        scale = np.where(scale > _EPS, scale, 1.0)

        scale = scale[:, np.newaxis, :]
        pose_root = pose[:, np.newaxis, _NOSE_IDX]
        # pose_root = (pose[:, np.newaxis, _LEFT_SHOULDER_IDX] + pose[:, np.newaxis, _RIGHT_SHOULDER_IDX]) / 2
        left_root = left[:, 0:1, :]
        right_root = right[:, 0:1, :]

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

    def fuse_follow_hand(self, pose_feature, left_feature, right_feature, use_pose=True):
        T = left_feature.shape[0]

        left = left_feature.reshape(T, 21, _COORD_DIM).copy()
        right = right_feature.reshape(T, 21, _COORD_DIM).copy()

        left_wrist = left[:, 0:1, :].copy()
        right_wrist = right[:, 0:1, :].copy()

        scale = np.linalg.norm(left_wrist[:, 0] - right_wrist[:, 0], axis=-1, keepdims=True)
        scale = np.where(scale > _EPS, scale, 1.0)
        scale = scale[:, np.newaxis, :]

        if use_pose:
            pose = pose_feature.reshape(T, 33, _COORD_DIM).copy()
            pose = np.delete(pose, _REMOVE_POSE_IDX, axis=1)

            left = np.concatenate([pose[:, ::2, :], left], axis=1)
            right = np.concatenate([pose[:, 1::2, :], right], axis=1)

        normalize_left_hand = self.average_hand(left)
        normalize_right_hand = self.average_hand(right)

        normalize_left_hand = normalize_left_hand / scale
        normalize_right_hand = normalize_right_hand / scale

        fused_coords = np.concatenate([normalize_left_hand, normalize_right_hand], axis=1)

        fused_flat = fused_coords.reshape(T, -1)  # (T, D)

        return fused_flat

    def average_hand(self, hand_feature):
        T = hand_feature.shape[0]

        present_mask = ~np.all(hand_feature == 0, axis=-1)

        valid_count = present_mask.sum(axis=1, keepdims=True)

        average_point = (hand_feature.sum(axis=1) / np.maximum(valid_count, 1))
        average_point = average_point.reshape(T, 1, _COORD_DIM)

        root = average_point

        normalize_hand_feature = hand_feature - root

        normalize_hand_feature = normalize_hand_feature * present_mask[..., None]

        return normalize_hand_feature

    def normalize_by_wrist(self, hand_feature, wrist=None):
        T = hand_feature.shape[0]

        if wrist is None:
            wrist = hand_feature[:, 0:1, :]

        present_mask = ~np.all(hand_feature == 0, axis=-1)  # [T, N]

        wrist_present = ~np.all(wrist == 0, axis=-1)  # [T, 1]
        valid_mask = present_mask & wrist_present  # [T, N]

        normalize_hand_feature = hand_feature - wrist

        normalize_hand_feature = (normalize_hand_feature * valid_mask[..., None])

        return normalize_hand_feature
