import numpy as np
import torch
from config import _COORD_DIM, _N_HAND, _N_POSE, _REMOVE_POSE_IDX

NUM_JOINTS = _N_POSE + 2 * _N_HAND

_POSE_LR_PAIRS_RAW = [
    (11, 12),
]


# def _build_swap_index():
#     swap = np.arange(NUM_JOINTS)
#
#     removed = set(_REMOVE_POSE_IDX)
#     kept = [i for i in range(33) if i not in removed]
#     old_to_new = {old: new for new, old in enumerate(kept)}
#
#     for left_old, right_old in _POSE_LR_PAIRS_RAW:
#         l = old_to_new.get(left_old)
#         r = old_to_new.get(right_old)
#         if l is not None and r is not None:
#             swap[l] = r
#             swap[r] = l
#         # if only one side survived the removal, leave it mapped to itself
#
#     n_pose_kept = len(kept)
#     assert n_pose_kept == _N_POSE, (
#         f"_N_POSE ({_N_POSE}) does not match pose points left after "
#         f"_REMOVE_POSE_IDX removal ({n_pose_kept}). Check config."
#     )
#
#     # 2) Hand blocks: swap the whole left-hand block with the whole
#     # right-hand block, index-for-index (finger topology is identical,
#     # only which array/hand it belongs to changes).
#     left_start = n_pose_kept
#     right_start = n_pose_kept + _N_HAND
#     for i in range(_N_HAND):
#         swap[left_start + i] = right_start + i
#         swap[right_start + i] = left_start + i
#
#     return swap


# _SWAP_IDX = _build_swap_index()


class SkeletonAugmentor:
    def __init__(
        self,
        mirror_prob=0.3,
        rotation_deg=13.0,
        scale_range=(0.9, 1.1),
        noise_std=0.01,
        noise_prob=0.5,
        frame_dropout_prob=0.0,
        max_frame_dropout_ratio=0.1,
        speed_perturb_prob=0.8,
        speed_range=(0.8, 1.25),
        enable_mirror=False,
        enable_noise=True,
        rng=None,
    ):
        self.mirror_prob = mirror_prob
        self.rotation_deg = rotation_deg
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.frame_dropout_prob = frame_dropout_prob
        self.max_frame_dropout_ratio = max_frame_dropout_ratio
        self.speed_perturb_prob = speed_perturb_prob
        self.speed_range = speed_range
        self.enable_mirror = enable_mirror
        self.enable_noise = enable_noise
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, fused):
        fused = np.asarray(fused, dtype=np.float32)
        T, flat_dim = fused.shape[0], fused.shape[1]

        flatten = NUM_JOINTS * _COORD_DIM  # 55 * 2 = 110
        num_blocks = flat_dim // flatten  # 1 hoặc 3

        blocks = fused.reshape(T, num_blocks, NUM_JOINTS, _COORD_DIM).copy()

        present_mask = ~np.all(blocks[:, 0] == 0, axis=-1)  # (T, N)

        # if self.speed_perturb_prob > 0 and self.rng.random() < self.speed_perturb_prob:
        #     blocks, present_mask = self._speed_perturb(blocks, present_mask)
        #     T = blocks.shape[0]

        # ── Rotate & Scale (áp dụng đồng nhất cho tất cả blocks) ──
        blocks = self._rotate(blocks)
        blocks = self._scale(blocks)

        # ── Gaussian noise ──
        if (
            self.enable_noise
            and self.noise_std > 0
            and self.rng.random() < self.noise_prob
        ):
            blocks = self._add_noise(blocks, present_mask)

        blocks = blocks * present_mask[:, None, :, None]  # broadcast: (T,1,N,1)

        coords = blocks.reshape(T, flat_dim)

        # if self.frame_dropout_prob > 0 and self.rng.random() < self.frame_dropout_prob:
        #     coords = self._drop_frames(coords)

        return coords.astype(np.float32)

    # def _mirror(self, blocks):
    #     # blocks: (T, num_blocks, N, C) — hoán đổi node trái ↔ phải, lật trục x
    #     blocks = blocks[:, :, _SWAP_IDX, :].copy()
    #     blocks[..., 0] *= -1.0  # flip x axis (left ↔ right)
    #     return blocks

    def _rotate(self, coords):
        deg = self.rng.uniform(-self.rotation_deg, self.rotation_deg)
        rad = np.deg2rad(deg)
        cos, sin = np.cos(rad), np.sin(rad)

        if _COORD_DIM == 2:
            R = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
        elif _COORD_DIM == 3:
            R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported _COORD_DIM: {_COORD_DIM}")
        return coords @ R.T

    # def _speed_perturb(self, blocks, present_mask):
    #     T = blocks.shape[0]
    #     if T <= 2:
    #         return blocks, present_mask
    #
    #     factor = self.rng.uniform(*self.speed_range)
    #     new_T = max(2, round(T / factor))
    #     if new_T == T:
    #         return blocks, present_mask
    #
    #     new_idx = np.linspace(0, T - 1, new_T)
    #
    #     idx_floor = np.floor(new_idx).astype(np.int64)
    #     idx_ceil = np.clip(idx_floor + 1, 0, T - 1)
    #
    #     extra_dims = blocks.ndim - 1
    #     frac = (new_idx - idx_floor).astype(np.float32).reshape(-1, *([1] * extra_dims))
    #
    #     blocks_new = (
    #         blocks[idx_floor] * (1.0 - frac) + blocks[idx_ceil] * frac
    #     ).astype(np.float32)
    #
    #     nearest_idx = np.round(new_idx).astype(np.int64)
    #     mask_new = present_mask[nearest_idx]
    #
    #     return blocks_new, mask_new

    def _scale(self, coords):
        lo, hi = self.scale_range
        factor = self.rng.uniform(lo, hi)
        return coords * factor

    def _add_noise(self, blocks, present_mask):
        noise = self.rng.normal(0.0, self.noise_std, size=blocks.shape).astype(
            np.float32
        )
        mask = present_mask[
            :, None, :, None
        ]  # (T, 1, N, 1) → broadcast với (T, B, N, C)
        return blocks + noise * mask

    def _drop_frames(self, coords_flat):
        T = coords_flat.shape[0]
        if T <= 2:
            return coords_flat
        max_drop = max(1, int(T * self.max_frame_dropout_ratio))
        n_drop = int(self.rng.integers(1, max_drop + 1))
        n_drop = min(n_drop, T - 2)  # keep at least 2 frames
        drop_idx = self.rng.choice(T, size=n_drop, replace=False)
        keep_mask = np.ones(T, dtype=bool)
        keep_mask[drop_idx] = False
        return coords_flat[keep_mask]

class RelativeHandSkeletonAugmentor:
    """Input/output: (T, 85), gồm [flag, left_hand, right_hand]."""

    def __init__(
        self,
        rotation_deg=10.0,
        scale_range=(0.95, 1.05),
        noise_std=0.005,
        noise_prob=0.5,
        mirror_prob=0.0,
        speed_perturb_prob=0.5,
        speed_range=(0.8, 1.25),
        frame_dropout_prob=0.0,
        max_frame_dropout_ratio=0.1,
        rng=None,
    ):
        self.rotation_deg = rotation_deg
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.mirror_prob = mirror_prob
        self.speed_perturb_prob = speed_perturb_prob
        self.speed_range = speed_range
        self.frame_dropout_prob = frame_dropout_prob
        self.max_frame_dropout_ratio = max_frame_dropout_ratio
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, fused):
        fused = np.array(fused, dtype=np.float32, copy=True)

        if fused.ndim != 2 or fused.shape[1] != 85:
            raise ValueError(
                f"Expected feature shape (T, 85), got {fused.shape}"
            )

        if len(fused) == 0:
            return fused

        # Resample nguyên hàng: flag và hai tay luôn đồng bộ.
        if self.rng.random() < self.speed_perturb_prob:
            fused = self._speed_perturb(fused)

        if self.rng.random() < self.frame_dropout_prob:
            fused = self._drop_frames(fused)

        T = len(fused)
        flag = fused[:, :1].copy()
        hands = fused[:, 1:].reshape(T, 2, 21, 2).copy()

        # Sau wrist normalization, wrist hợp lệ cũng bằng 0.
        # Chỉ dùng mask này cho các điểm khác wrist.
        present_mask = np.any(hands != 0, axis=-1)
        present_mask[:, :, 0] = False

        # Mirror cả clip: đổi tay trái/phải và đảo trục x.
        if self.rng.random() < self.mirror_prob:
            hands = hands[:, ::-1].copy()
            present_mask = present_mask[:, ::-1].copy()
            hands[..., 0] *= -1.0

        # Một góc rotation cho tất cả frame và cả hai tay.
        angle = np.deg2rad(
            self.rng.uniform(-self.rotation_deg, self.rotation_deg)
        )
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array(
            [[cos, -sin], [sin, cos]],
            dtype=np.float32,
        )
        hands = hands @ rotation.T

        # Một scale chung để giữ tỷ lệ giữa hai tay.
        scale = self.rng.uniform(*self.scale_range)
        hands *= scale

        if (
            self.noise_std > 0
            and self.rng.random() < self.noise_prob
        ):
            noise = self.rng.normal(
                0.0, self.noise_std, size=hands.shape
            ).astype(np.float32)

            hands += noise * present_mask[..., None]

        # Giữ các điểm thiếu và wrist bằng 0.
        hands = np.where(present_mask[..., None], hands, 0.0)
        hands[:, :, 0, :] = 0.0

        # Không clip: rotation/scale có thể vượt [-1, 1].
        return np.concatenate(
            [flag, hands.reshape(T, 84)],
            axis=1,
        ).astype(np.float32)

    def _speed_perturb(self, fused):
        T = len(fused)
        if T <= 2:
            return fused

        speed = self.rng.uniform(*self.speed_range)
        new_T = max(2, int(round(T / speed)))

        indices = np.rint(
            np.linspace(0, T - 1, new_T)
        ).astype(np.int64)

        return fused[indices].copy()

    def _drop_frames(self, fused):
        T = len(fused)
        if T <= 2:
            return fused

        max_drop = min(
            int(T * self.max_frame_dropout_ratio),
            T - 2,
        )
        if max_drop < 1:
            return fused

        n_drop = int(self.rng.integers(1, max_drop + 1))
        drop_indices = self.rng.choice(
            T, size=n_drop, replace=False
        )

        keep = np.ones(T, dtype=bool)
        keep[drop_indices] = False

        return fused[keep].copy()

class AugmentedSkeletonDataset:
    def __init__(self, base_dataset, augmentor=None, num_augmentations=1, seed=None):
        self.base_dataset = base_dataset
        self.num_augmentations = num_augmentations
        self.base_len = len(base_dataset)
        self._seed = seed
        self._augmentor = augmentor

    def _get_augmentor(self):
        if self._augmentor is None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            seed_seq = np.random.SeedSequence(self._seed).spawn(worker_id + 1)[-1]
            rng = np.random.default_rng(seed_seq)
            self._augmentor = SkeletonAugmentor(rng=rng)
        return self._augmentor

    def __len__(self):
        return self.base_len * (1 + self.num_augmentations)

    def __getitem__(self, idx):
        stride = 1 + self.num_augmentations
        base_idx = idx // stride
        variant = idx % stride

        feature, label, video_id = self.base_dataset[base_idx]
        if variant == 0:
            return np.array(feature, dtype=np.float32, copy=True), label, video_id

        augmentor = self._get_augmentor()
        feature_aug = augmentor(feature)

        return feature_aug, label, video_id
