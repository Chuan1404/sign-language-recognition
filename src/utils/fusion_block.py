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

        average_point = hand_feature.sum(axis=1) / np.maximum(valid_count, 1)
        average_point = average_point.reshape(T, 1, _COORD_DIM)

        root = average_point

        normalize_hand_feature = hand_feature - root

        normalize_hand_feature = normalize_hand_feature * present_mask[..., None]

        return normalize_hand_feature

    def normalize_by_wrist(self, hand_feature, wrist=None):

        if wrist is None:
            wrist = hand_feature[:, 0:1, :]

        present_mask = ~np.all(hand_feature == 0, axis=-1)  # [T, N]

        wrist_present = ~np.all(wrist == 0, axis=-1)  # [T, 1]
        valid_mask = present_mask & wrist_present  # [T, N]

        normalize_hand_feature = hand_feature - wrist

        normalize_hand_feature = normalize_hand_feature * valid_mask[..., None]

        return normalize_hand_feature

    def _relative_hand_inputs(self, left_feature, right_feature):
        T = left_feature.shape[0]

        left = left_feature.reshape(T, 21, _COORD_DIM).astype(np.float32, copy=True)
        right = right_feature.reshape(T, 21, _COORD_DIM).astype(np.float32, copy=True)

        left_mask = ~np.all(left == 0, axis=-1)
        right_mask = ~np.all(right == 0, axis=-1)

        flag = (left_mask.any(axis=1) & right_mask.any(axis=1)).astype(np.float32)[:, None]

        hands = np.concatenate([left, right], axis=1)
        mask = np.concatenate([left_mask, right_mask], axis=1)

        lowest = np.where(mask[..., None], hands, np.inf).min(axis=1, keepdims=True)

        highest = np.where(mask[..., None], hands, -np.inf).max(axis=1, keepdims=True)

        has_hand = mask.any(axis=1)[:, None, None]
        lowest = np.where(has_hand, lowest, 0.0)
        highest = np.where(has_hand, highest, 0.0)

        # Với tọa độ 2D: max(width, height), shape (T, 1, 1).
        scale = (highest - lowest).max(axis=-1, keepdims=True)

        return left, right, left_mask, right_mask, flag, lowest, scale

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

        # Thiếu vai hoặc khoảng cách quá nhỏ: không scale.
        valid = shoulders_present & np.isfinite(scale) & (scale > _EPS)
        scale = np.where(valid, scale, 1.0)

        return scale[:, None, None]  # (T, 1, 1)

    def fuse_relatively(self, pose_feature, left_feature, right_feature):
        (
            left, right, left_mask, right_mask,
            flag, lowest, bbox_scale,
        ) = self._relative_hand_inputs(left_feature, right_feature)

        T = left.shape[0]
        scale = self._shoulder_scale(pose_feature)

        left_normalized = self.normalize_by_wrist(left) / scale
        right_normalized = self.normalize_by_wrist(right) / scale

        return np.concatenate(
            [
                flag,
                left_normalized.reshape(T, 21 * _COORD_DIM),
                right_normalized.reshape(T, 21 * _COORD_DIM),
            ],
            axis=1,
        )

    def fuse_position_relatively(
            self, pose_feature, left_feature, right_feature
    ):
        (
            left, right, left_mask, right_mask,
            flag, lowest, bbox_scale,
        ) = self._relative_hand_inputs(left_feature, right_feature)

        T = left.shape[0]
        scale = self._shoulder_scale(pose_feature)

        hands = np.concatenate([left, right], axis=1)
        mask = np.concatenate([left_mask, right_mask], axis=1)

        normalized = (hands - lowest) / scale
        normalized = np.where(mask[..., None], normalized, 0.0)

        return np.concatenate(
            [
                flag,
                normalized[:, :21].reshape(T, 21 * _COORD_DIM),
                normalized[:, 21:].reshape(T, 21 * _COORD_DIM),
            ],
            axis=1,
        )

    def fuse_relatively_origin(self, pose_feature, left_feature, right_feature):
        # Giữ pose_feature trong signature, hiện tại chỉ dùng hai bàn tay.
        T = left_feature.shape[0]

        left = left_feature.reshape(T, 21, _COORD_DIM).astype(
            np.float32, copy=True
        )
        right = right_feature.reshape(T, 21, _COORD_DIM).astype(
            np.float32, copy=True
        )

        # Xác định landmark có mặt TRƯỚC khi normalize.
        left_present_mask = ~np.all(left == 0, axis=-1)  # (T, 21)
        right_present_mask = ~np.all(right == 0, axis=-1)  # (T, 21)

        left_present = left_present_mask.any(axis=1)
        right_present = right_present_mask.any(axis=1)

        is_two_hands_flag = (
                left_present & right_present
        ).astype(np.float32)[:, None]  # (T, 1)

        hands = np.concatenate([left, right], axis=1)  # (T, 42, 2)
        present_mask = np.concatenate(
            [left_present_mask, right_present_mask], axis=1
        )  # (T, 42)

        # Bounding box chung, chỉ tính từ landmark có mặt.
        lowest = np.where(
            present_mask[..., None], hands, np.inf
        ).min(axis=1, keepdims=True)  # (T, 1, 2)

        highest = np.where(
            present_mask[..., None], hands, -np.inf
        ).max(axis=1, keepdims=True)  # (T, 1, 2)

        # Frame không có tay: đặt bounds = 0 để tránh inf/NaN.
        has_any_hand = present_mask.any(axis=1)[:, None, None]
        lowest = np.where(has_any_hand, lowest, 0.0)
        highest = np.where(has_any_hand, highest, 0.0)

        x_lowest = lowest[..., 0:1]
        x_highest = highest[..., 0:1]
        y_lowest = lowest[..., 1:2]
        y_highest = highest[..., 1:2]

        # Nếu một chiều không có độ rộng, chiều đó trả về 0.
        x_range = x_highest - x_lowest
        y_range = y_highest - y_lowest

        normalized = np.zeros_like(hands)

        np.divide(
            hands[..., 0:1] - x_lowest,
            x_range,
            out=normalized[..., 0:1],
            where=x_range > _EPS,
        )
        np.divide(
            hands[..., 1:2] - y_lowest,
            y_range,
            out=normalized[..., 1:2],
            where=y_range > _EPS,
        )

        # Giữ landmark bị thiếu bằng 0.
        normalized = np.where(
            present_mask[..., None], normalized, 0.0
        )
        normalized = np.clip(normalized, 0.0, 1.0)

        left_normalized = normalized[:, :21].copy()
        right_normalized = normalized[:, 21:].copy()

        # Dùng mask gốc: wrist sau normalize có thể hợp lệ tại (0, 0).
        left_valid_mask = (
                left_present_mask & left_present_mask[:, 0:1]
        )
        right_valid_mask = (
                right_present_mask & right_present_mask[:, 0:1]
        )

        left_normalized -= left_normalized[:, 0:1, :].copy()
        right_normalized -= right_normalized[:, 0:1, :].copy()

        # Nếu wrist bị thiếu, feature của cả tay đó bằng 0.
        left_normalized = np.where(
            left_valid_mask[..., None], left_normalized, 0.0
        )
        right_normalized = np.where(
            right_valid_mask[..., None], right_normalized, 0.0
        )

        fused_flat = np.concatenate(
            [
                is_two_hands_flag,
                left_normalized.reshape(T, -1),
                right_normalized.reshape(T, -1),
            ],
            axis=1,
        )

        return fused_flat