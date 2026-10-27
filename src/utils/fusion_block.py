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

    def fuse_follow_velocity(self, pose_feature, left_feature, right_feature):
        T = pose_feature.shape[0]

        pose = np.asarray(pose_feature).reshape(T, 33, _COORD_DIM)
        left = np.asarray(left_feature).reshape(T, 21, _COORD_DIM)
        right = np.asarray(right_feature).reshape(T, 21, _COORD_DIM)

        pose = np.delete(pose, _REMOVE_POSE_IDX, axis=1)
        points = np.concatenate([pose, left, right], axis=1).astype(np.float32)

        present = np.any(points != 0, axis=-1) & np.isfinite(points).all(axis=-1)
        valid_pairs = present[1:] & present[:-1]
        delta = np.zeros_like(points[1:, :, :2])

        np.subtract(points[1:, :, :2], points[:-1, :, :2],
                    out=delta, where=valid_pairs[..., None])

        velocity = np.zeros((T, points.shape[1], 2), dtype=np.float32)
        velocity[:-1] = delta
        speed = np.linalg.norm(velocity, axis=-1)
        max_speed = speed.max(axis=1, keepdims=True)
        velocity = np.divide(
            velocity, max_speed[..., None], out=np.zeros_like(velocity),
            where=max_speed[..., None] > 0,
        )
        normalized_speed = np.divide(
            speed, max_speed, out=np.zeros_like(speed), where=max_speed > 0
        )
        below_threshold = normalized_speed < 0.1
        removed_mask = below_threshold & (speed > 0)
        # removed_count = np.count_nonzero(removed_mask)
        # moving_count = np.count_nonzero(speed > 0)
        velocity[below_threshold] = 0
        # print(f"[Velocity] Filtered {removed_count}/{moving_count} moving frame-node pairs (< 0.1).")
        pose_indices = np.delete(np.arange(33), _REMOVE_POSE_IDX)
        node_names = ([f"pose[{i}]" for i in pose_indices]
                      + [f"left_hand[{i}]" for i in range(21)]
                      + [f"right_hand[{i}]" for i in range(21)])
        # for frame_idx in np.flatnonzero(removed_mask.any(axis=1)):
        #     removed_nodes = np.flatnonzero(removed_mask[frame_idx])
        #     details = ", ".join(
        #         f"node {i} ({node_names[i]}, speed={normalized_speed[frame_idx, i]:.4f})"
        #         for i in removed_nodes
        #     )
        #     print(f"[Velocity] Frame {frame_idx} -> {frame_idx + 1}: "
        #           f"removed {len(removed_nodes)} nodes: {details}")
        hand_removed_mask = removed_mask.copy()
        hand_removed_mask[:, :len(pose_indices)] = False
        for frame_idx in np.flatnonzero(hand_removed_mask.any(axis=1)):
            removed_nodes = np.flatnonzero(hand_removed_mask[frame_idx])
            details = ", ".join(
                f"node {i} ({node_names[i]}, speed={normalized_speed[frame_idx, i]:.4f})"
                for i in removed_nodes
            )
            # print(f"[Velocity] Frame {frame_idx} -> {frame_idx + 1}: "
            #       f"removed {len(removed_nodes)} hand nodes: {details}")
        # Preserve signed XY velocity for every node, flattened for feature concatenation.
        return velocity.reshape(T, points.shape[1] * 2)

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
