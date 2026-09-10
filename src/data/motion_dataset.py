
import os
import glob

import numpy as np
from torch.utils.data import Dataset
from config import _REMOVE_POSE_IDX, _N_POSE


_LEFT_SHOULDER_IDX = 1
_RIGHT_SHOULDER_IDX = 2
_EPS = 1e-6


def _normalize_by_shoulder(pose, left, right):

    root = (pose[:, _LEFT_SHOULDER_IDX] + pose[:, _RIGHT_SHOULDER_IDX]) / 2.0
    scale = np.linalg.norm(
        pose[:, _LEFT_SHOULDER_IDX] - pose[:, _RIGHT_SHOULDER_IDX], axis=-1, keepdims=True
    )
    scale = np.where(scale > _EPS, scale, 1.0)

    root = root[:, np.newaxis, :]
    scale = scale[:, np.newaxis, :]

    pose = (pose - root) / scale
    left = (left - root) / scale
    right = (right - root) / scale
    return pose, left, right


class MotionKeypointDataset(Dataset):

    STREAM_POSE, STREAM_LEFT, STREAM_RIGHT = 0, 1, 2

    def __init__(
        self,
        segment_dir,
        coord_dim=2,
        mask_prob_per_stream=0.5,     # xác suất 1 luồng bị che (tính riêng từng luồng)
        min_span_ratio=0.1,           # đoạn che tối thiểu, tỉ lệ theo T
        max_span_ratio=0.35,          # đoạn che tối đa, tỉ lệ theo T
        max_spans_per_stream=2,       # tối đa bao nhiêu đoạn che rời rạc / luồng / sample
    ):
        self.segment_dirs = sorted(glob.glob(os.path.join(segment_dir, "*")))
        self.coord_dim = coord_dim
        self.mask_prob_per_stream = mask_prob_per_stream
        self.min_span_ratio = min_span_ratio
        self.max_span_ratio = max_span_ratio
        self.max_spans_per_stream = max_spans_per_stream

        if len(self.segment_dirs) == 0:
            raise FileNotFoundError(f"Không tìm thấy segment nào trong {segment_dir}")

        pose0 = np.load(os.path.join(self.segment_dirs[0], "pose.npy"))
        left0 = np.load(os.path.join(self.segment_dirs[0], "left_hand.npy"))
        right0 = np.load(os.path.join(self.segment_dirs[0], "right_hand.npy"))
        self.pose_dim = _N_POSE * coord_dim
        self.left_dim = 21 * coord_dim
        self.right_dim = 21 * coord_dim
        self.window_size = pose0.shape[0]

        assert left0.shape[0] == self.window_size and right0.shape[0] == self.window_size, \
            "pose/left/right phải cùng độ dài T trong 1 segment"

    def __len__(self):
        return len(self.segment_dirs)

    def _sample_spans(self, T):
        """Sinh danh sách (start, length) cho 1 luồng, có thể rỗng."""
        spans = []
        if np.random.rand() >= self.mask_prob_per_stream:
            return spans

        n_spans = np.random.randint(1, self.max_spans_per_stream + 1)
        for _ in range(n_spans):
            min_span = max(1, int(T * self.min_span_ratio))
            max_span = max(min_span, int(T * self.max_span_ratio))
            span_len = np.random.randint(min_span, max_span + 1)
            span_len = min(span_len, T)
            start = np.random.randint(0, T - span_len + 1)
            spans.append((start, span_len))
        return spans

    def __getitem__(self, idx):
        seg_dir = self.segment_dirs[idx]

        pose = np.load(os.path.join(seg_dir, "pose.npy"))[:, :, :self.coord_dim].astype(np.float32)
        left = np.load(os.path.join(seg_dir, "left_hand.npy"))[:, :, :self.coord_dim].astype(np.float32)
        right = np.load(os.path.join(seg_dir, "right_hand.npy"))[:, :, :self.coord_dim].astype(np.float32)

        pose = np.delete(pose, _REMOVE_POSE_IDX, axis=1)

        pose, left, right = _normalize_by_shoulder(pose, left, right)

        T = pose.shape[0]
        pose_flat = pose.reshape(T, -1)
        left_flat = left.reshape(T, -1)
        right_flat = right.reshape(T, -1)

        target = np.concatenate([pose_flat, left_flat, right_flat], axis=-1)  # (T, D)
        coords_input = target.copy()

        mask_flags = np.zeros((T, 3), dtype=np.float32)

        slices = {
            self.STREAM_POSE: slice(0, self.pose_dim),
            self.STREAM_LEFT: slice(self.pose_dim, self.pose_dim + self.left_dim),
            self.STREAM_RIGHT: slice(self.pose_dim + self.left_dim, self.pose_dim + self.left_dim + self.right_dim),
        }

        for stream_idx, sl in slices.items():
            for start, length in self._sample_spans(T):
                coords_input[start:start + length, sl] = 0.0
                mask_flags[start:start + length, stream_idx] = 1.0

        return (
            coords_input.astype(np.float32),
            mask_flags.astype(np.float32),
            target.astype(np.float32),
        )