import numpy as np
import torch
from config import _COORD_DIM, _N_HAND, _N_POSE, _REMOVE_POSE_IDX

NUM_JOINTS = _N_POSE + 2 * _N_HAND

_POSE_LR_PAIRS_RAW = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
                      (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
                      (21, 22), (23, 24), (25, 26), (27, 28), (29, 30),
                      (31, 32)]


def _build_swap_index():
    swap = np.arange(NUM_JOINTS)
    kept = [i for i in range(33) if i not in set(_REMOVE_POSE_IDX)]
    if len(kept) != _N_POSE:
        raise ValueError("Kept pose nodes must match _N_POSE")
    old_to_new = {old: new for new, old in enumerate(kept)}
    for left, right in _POSE_LR_PAIRS_RAW:
        if left in old_to_new and right in old_to_new:
            l, r = old_to_new[left], old_to_new[right]
            swap[l], swap[r] = r, l
    left_start = _N_POSE
    right_start = _N_POSE + _N_HAND
    for i in range(_N_HAND):
        swap[left_start + i] = right_start + i
        swap[right_start + i] = left_start + i
    return swap


_SWAP_IDX = _build_swap_index()


def rotate_hand_blocks(blocks, rotation):
    """Rotate hands about each wrist in (T, B, N, C); preserve pose and zeros."""
    result = blocks.copy()
    for start in (_N_POSE, _N_POSE + _N_HAND):
        hand = blocks[:, :, start:start + _N_HAND, :]
        wrist = hand[:, :, :1, :]
        present = np.any(hand != 0, axis=-1, keepdims=True)
        rotated = (hand - wrist) @ rotation.T + wrist
        result[:, :, start:start + _N_HAND, :] = np.where(present, rotated, 0)
    return result


class SkeletonAugmentor:
    def __init__(
        self,
        rotation_deg=15.0,
        scale_range=(0.8, 1.2),
        noise_std=0.01,
        noise_prob=0.5,
        frame_dropout_prob=0.0,
        max_frame_dropout_ratio=0.1,
        speed_perturb_prob=0.8,
        speed_range=(0.8, 1.25),
        enable_mirror=True,
        enable_noise=True,
        rng=None,
        enable_viewpoint=False,
        shoulder_width_range=(0.70, 0.70),
        hand_shift_range=(0.20, 0.20),
        mirror_prob=0.5,
        enable_rotation=True,
        enable_scale=True,
    ):
        self.enable_rotation = enable_rotation
        self.enable_scale = enable_scale
        self.enable_viewpoint = enable_viewpoint
        kept = [i for i in range(33) if i not in set(_REMOVE_POSE_IDX)]
        self.shoulder_indices = tuple(kept.index(i) for i in (11, 12) if i in kept)
        self.elbow_indices = tuple(kept.index(i) for i in (13, 14) if i in kept)
        if enable_viewpoint and len(self.shoulder_indices) != 2:
            raise ValueError("Viewpoint augmentation requires pose shoulders 11 and 12 to be retained")
        if speed_perturb_prob > 0 and not 0 < speed_range[0] < 1 < speed_range[1]:
            raise ValueError("speed_range must satisfy 0 < slow_factor < 1 < fast_factor")
        self.shoulder_width_range = shoulder_width_range
        self.hand_shift_range = hand_shift_range
        self.rotation_deg = rotation_deg
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.frame_dropout_prob = frame_dropout_prob
        self.max_frame_dropout_ratio = max_frame_dropout_ratio
        self.speed_perturb_prob = speed_perturb_prob
        self.speed_range = speed_range
        self.enable_mirror = enable_mirror
        self.mirror_prob = mirror_prob
        self.enable_noise = enable_noise
        self.rng = rng if rng is not None else np.random.default_rng()

    @property
    def augmentation_names(self):
        """One indexed sample per enabled method; speed contributes two samples.

        Probability options act as on/off switches here (>0 enables a variant).
        An indexed variant always applies its method, without a random skip.
        """
        names = []
        # if self.enable_rotation and self.rotation_deg > 0:
        #     names.append("rotate")
        # if self.enable_scale and tuple(self.scale_range) != (1, 1):
        #     names.append("scale")
        # if self.enable_mirror and self.mirror_prob > 0:
        #     names.append("mirror")
        if self.speed_perturb_prob > 0:
            names.extend(("speed_fast", "speed_slow"))
        # if self.enable_noise and self.noise_std > 0 and self.noise_prob > 0:
        #     names.append("noise")
        # if self.enable_viewpoint:
        #     names.append("viewpoint")
        # if self.frame_dropout_prob > 0 and self.max_frame_dropout_ratio > 0:
        #     names.append("drop_frames")
        return tuple(names)

    def __call__(self, fused, method=None, view_direction=None):
        fused = np.asarray(fused, dtype=np.float32)
        block_dim = NUM_JOINTS * _COORD_DIM
        names = self.augmentation_names
        if method is None:
            if not names:
                return fused.copy()
            method = self.rng.choice(names)
        if method not in names:
            raise ValueError(f"Unknown or disabled augmentation: {method}")

        frames, flat_dim = fused.shape
        blocks = fused.reshape(frames, flat_dim // block_dim, NUM_JOINTS, _COORD_DIM).copy()
        present_mask = np.any(blocks != 0, axis=(1, 3))
        if method == "rotate":
            blocks = self._rotate(blocks)
        elif method == "scale":
            blocks = self._scale(blocks)
        elif method == "mirror":
            blocks, present_mask = self._mirror(blocks, present_mask)
        elif method in ("speed_fast", "speed_slow"):
            blocks, present_mask = self._speed_perturb(
                blocks, present_mask, direction=method.removeprefix("speed_"),
            )
        elif method == "noise":
            blocks = self._add_noise(blocks, present_mask)
        elif method == "viewpoint":
            blocks = self._simulate_viewpoint(blocks, present_mask, view_direction)
        elif method == "drop_frames":
            return self._drop_frames(fused.copy()).astype(np.float32)
        blocks *= present_mask[:, None, :, None]
        return blocks.reshape(len(blocks), flat_dim).astype(np.float32)

    def _mirror(self, blocks, present_mask):
        """Reflect normalized x coordinates and swap anatomical left/right joints."""
        mirrored = blocks[:, :, _SWAP_IDX, :].copy()
        mirrored[..., 0] *= -1
        return mirrored, present_mask[:, _SWAP_IDX].copy()

    def _simulate_viewpoint(self, blocks, present_mask, direction=None):
        """2D view approximation; three-block layout is [shape, average, position].

        Shoulder x-distance shrinks about its midpoint. Hands translate in global
        blocks only: local wrist-relative shape must not acquire a translation.
        One strength and direction are used for the entire clip.
        """
        if direction is None:
            direction = "left" if self.rng.random() < 0.5 else "right"
        if direction not in ("left", "right"):
            raise ValueError("view_direction must be 'left' or 'right'")
        result = blocks.copy()
        left, right = self.shoulder_indices
        reference = blocks[:, -1]
        valid_shoulders = (
            present_mask[:, left] & present_mask[:, right]
            & np.isfinite(reference[:, [left, right]]).all(axis=(1, 2))
        )
        width = np.abs(reference[:, left, 0] - reference[:, right, 0])
        valid_shoulders &= width > 1e-6
        factor = self.rng.uniform(*self.shoulder_width_range)
        strength = self.rng.uniform(*self.hand_shift_range)
        shift = width * strength * (-1 if direction == "left" else 1)
        for index in range(blocks.shape[1]):
            midpoint = (blocks[:, index, left, 0] + blocks[:, index, right, 0]) * 0.5
            for shoulder in (left, right):
                narrowed = midpoint + factor * (blocks[:, index, shoulder, 0] - midpoint)
                result[:, index, shoulder, 0] = np.where(
                    valid_shoulders, narrowed, blocks[:, index, shoulder, 0]
                )
            # Elbows belong to the torso-relative pose block, including shape.
            # Use the same horizontal displacement as the global hand coordinates.
            for elbow in self.elbow_indices:
                elbow_mask = present_mask[:, elbow] & valid_shoulders
                result[:, index, elbow, 0] += shift * elbow_mask
            if blocks.shape[1] == 3 and index == 0:
                continue
            hand_mask = present_mask[:, _N_POSE:] & valid_shoulders[:, None]
            result[:, index, _N_POSE:, 0] += shift[:, None] * hand_mask
        return result

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
        return rotate_hand_blocks(coords, R)

    def _speed_perturb(self, blocks, present_mask, direction=None):
        T = blocks.shape[0]
        if T <= 2:
            return blocks, present_mask

        if direction == "fast":
            factor = self.speed_range[1]
            new_T = min(T - 1, max(2, round(T / factor)))
        elif direction == "slow":
            factor = self.speed_range[0]
            new_T = max(T + 1, round(T / factor))
        elif direction is None:
            factor = self.rng.uniform(*self.speed_range)
            new_T = max(2, round(T / factor))
        else:
            raise ValueError("direction must be 'fast' or 'slow'")
        if new_T == T:
            return blocks, present_mask

        new_idx = np.linspace(0, T - 1, new_T)

        idx_floor = np.floor(new_idx).astype(np.int64)
        idx_ceil = np.clip(idx_floor + 1, 0, T - 1)

        extra_dims = blocks.ndim - 1
        frac = (new_idx - idx_floor).astype(np.float32).reshape(-1, *([1] * extra_dims))

        blocks_new = (
            blocks[idx_floor] * (1.0 - frac) + blocks[idx_ceil] * frac
        ).astype(np.float32)

        nearest_idx = np.round(new_idx).astype(np.int64)
        mask_new = present_mask[nearest_idx]

        return blocks_new, mask_new

    def _scale(self, coords):
        lo, hi = self.scale_range
        factor = self.rng.uniform(lo, hi)
        return coords * factor

    def  _add_noise(self, blocks, present_mask):
        noise = self.rng.normal(0.0, self.noise_std, size=blocks.shape).astype(
            np.float32
        )
        mask = present_mask[
            :, None, :, None
        ]  # Broadcast (T, 1, N, 1) across feature blocks (T, B, N, C).
        return blocks + noise * mask
    #
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

class AugmentedSkeletonDataset(torch.utils.data.Dataset):
    """Original + one independent copy per method, including fast and slow speed.

    Every variant starts from the original sample. Samples are generated lazily,
    preserving labels/video IDs and leaving the base dataset cache unchanged.
    """

    def __init__(self, base_dataset, augmentor=None, seed=None):
        self.base_dataset = base_dataset
        self.base_len = len(base_dataset)
        self._augmentor = augmentor if augmentor is not None else SkeletonAugmentor()
        self.variant_names = ("original", *self._augmentor.augmentation_names)
        self.num_augmentations = len(self.variant_names) - 1
        self._seed = seed
        self._worker_id = None

    def _get_augmentor(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        if self._seed is not None and self._worker_id != worker_id:
            self._augmentor.rng = np.random.default_rng(np.random.SeedSequence([self._seed, worker_id]))
            self._worker_id = worker_id
        return self._augmentor

    def __len__(self):
        return self.base_len * len(self.variant_names)

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        base_idx, variant = divmod(idx, len(self.variant_names))
        feature, label, video_id = self.base_dataset[base_idx]
        if variant == 0:
            return np.array(feature, dtype=np.float32, copy=True), label, video_id
        feature_aug = self._get_augmentor()(feature, method=self.variant_names[variant])
        return feature_aug, label, video_id
