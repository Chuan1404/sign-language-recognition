import numpy as np

from config import _N_POSE, _N_HAND, _COORD_DIM, _REMOVE_POSE_IDX

NUM_JOINTS = _N_POSE + 2 * _N_HAND

_POSE_LR_PAIRS_RAW = [
    (11, 12), (13, 14),  # shoulder, elbow, wrist
]


def _build_swap_index():
    swap = np.arange(NUM_JOINTS)

    removed = set(_REMOVE_POSE_IDX)
    kept = [i for i in range(33) if i not in removed]
    old_to_new = {old: new for new, old in enumerate(kept)}

    for left_old, right_old in _POSE_LR_PAIRS_RAW:
        l = old_to_new.get(left_old)
        r = old_to_new.get(right_old)
        if l is not None and r is not None:
            swap[l] = r
            swap[r] = l
        # if only one side survived the removal, leave it mapped to itself

    n_pose_kept = len(kept)
    assert n_pose_kept == _N_POSE, (
        f"_N_POSE ({_N_POSE}) does not match pose points left after "
        f"_REMOVE_POSE_IDX removal ({n_pose_kept}). Check config."
    )

    # 2) Hand blocks: swap the whole left-hand block with the whole
    # right-hand block, index-for-index (finger topology is identical,
    # only which array/hand it belongs to changes).
    left_start = n_pose_kept
    right_start = n_pose_kept + _N_HAND
    for i in range(_N_HAND):
        swap[left_start + i] = right_start + i
        swap[right_start + i] = left_start + i

    return swap

_SWAP_IDX = _build_swap_index()

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
        speed_perturb_prob=1.5,
        speed_range=(0.8, 1.25),
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
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, fused):
        fused = np.asarray(fused, dtype=np.float32)
        T = fused.shape[0]
        coords = fused.reshape(T, -1, _COORD_DIM).copy()

        present_mask = ~np.all(coords == 0, axis=-1)  # (T, N), True = real point

        if self.speed_perturb_prob > 0 and self.rng.random() < self.speed_perturb_prob:
            coords, present_mask = self._speed_perturb(coords, present_mask)

        # if self.rng.random() < self.mirror_prob:
        #     coords = self._mirror(coords)
        #     present_mask = present_mask[:, _SWAP_IDX]

        coords = self._rotate(coords)
        coords = self._scale(coords)

        # if self.noise_std > 0 and self.rng.random() < self.noise_prob:
        #     coords = self._add_noise(coords, present_mask)
        #
        coords = coords * present_mask[:, :, None]
        coords = coords.reshape(coords.shape[0], -1)
        #
        # if self.frame_dropout_prob > 0 and self.rng.random() < self.frame_dropout_prob:
        #     coords = self._drop_frames(coords)

        return coords.astype(np.float32)

    def _mirror(self, coords):
        coords = coords[:, _SWAP_IDX, :].copy()
        coords[..., 0] *= -1.0  # flip x axis (left <-> right)
        return coords

    def _rotate(self, coords):
        deg = self.rng.uniform(-self.rotation_deg, self.rotation_deg)
        rad = np.deg2rad(deg)
        cos, sin = np.cos(rad), np.sin(rad)

        if _COORD_DIM == 2:
            R = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
        elif _COORD_DIM == 3:
            R = np.array(
                [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32
            )
        else:
            raise ValueError(f"Unsupported _COORD_DIM: {_COORD_DIM}")

        return coords @ R.T

    def _speed_perturb(self, coords, present_mask):
        """Resample the sequence along the time axis to simulate the
        action being performed faster or slower.

        factor > 1.0  -> faster motion -> fewer output frames
        factor < 1.0  -> slower motion -> more output frames

        Coordinates are linearly interpolated between neighboring frames;
        the presence mask is resampled with nearest-neighbor lookup since
        it's boolean (a point is either tracked or not, no in-between).
        """
        T = coords.shape[0]
        if T <= 2:
            return coords, present_mask

        factor = self.rng.uniform(*self.speed_range)
        new_T = max(2, int(round(T / factor)))
        if new_T == T:
            return coords, present_mask

        old_idx = np.linspace(0, T - 1, T)
        new_idx = np.linspace(0, T - 1, new_T)

        idx_floor = np.floor(new_idx).astype(np.int64)
        idx_ceil = np.clip(idx_floor + 1, 0, T - 1)
        frac = (new_idx - idx_floor).astype(np.float32)[:, None, None]

        coords_new = (
            coords[idx_floor] * (1.0 - frac) + coords[idx_ceil] * frac
        ).astype(np.float32)

        nearest_idx = np.round(new_idx).astype(np.int64)
        mask_new = present_mask[nearest_idx]

        return coords_new, mask_new

    def _scale(self, coords):
        lo, hi = self.scale_range
        factor = self.rng.uniform(lo, hi)
        return coords * factor

    def _add_noise(self, coords, present_mask):
        noise = self.rng.normal(0.0, self.noise_std, size=coords.shape).astype(
            np.float32
        )
        return coords + noise * present_mask[..., None]

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


class AugmentedSkeletonDataset:
    def __init__(self, base_dataset, augmentor=None, num_augmentations=1, seed=None):
        self.base_dataset = base_dataset
        self.num_augmentations = num_augmentations
        self.base_len = len(base_dataset)
        self._seed = seed
        self._augmentor = augmentor  # may be None -> built lazily per worker

    def _get_augmentor(self):
        if self._augmentor is None:
            import torch

            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            seed_seq = np.random.SeedSequence(self._seed).spawn(worker_id + 1)[-1]
            rng = np.random.default_rng(seed_seq)
            self._augmentor = SkeletonAugmentor(rng=rng)
        return self._augmentor

    def __len__(self):
        return self.base_len * (1 + self.num_augmentations)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        base_idx = idx % self.base_len
        variant = idx // self.base_len  # 0 = original, >=1 = augmented

        feature, label = self.base_dataset[base_idx]

        if variant == 0:
            return np.array(feature, dtype=np.float32, copy=True), label

        augmentor = self._get_augmentor()
        feature_aug = augmentor(feature)  # augmentor copies internally, doesn't touch original
        return feature_aug, label