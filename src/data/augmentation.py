import numpy as np

from config import _N_POSE, _N_HAND, _COORD_DIM

NUM_JOINTS = _N_POSE + 2 * _N_HAND
NODES_PER_GROUP = _N_POSE // 2 + _N_HAND

_LEFT_GROUP_START = 0
_RIGHT_GROUP_START = NODES_PER_GROUP


def _build_swap_index():
    swap = np.arange(NUM_JOINTS)
    for i in range(NODES_PER_GROUP):
        left_idx = _LEFT_GROUP_START + i
        right_idx = _RIGHT_GROUP_START + i
        swap[left_idx] = right_idx
        swap[right_idx] = left_idx
    return swap


_SWAP_IDX = _build_swap_index()


class SkeletonAugmentor:
    def __init__(self, mirror_prob=0.3, rotation_deg=13.0):
        self.mirror_prob = mirror_prob
        self.rotation_deg = rotation_deg

    def __call__(self, fused):
        fused = np.asarray(fused, dtype=np.float32)
        T = fused.shape[0]
        coords = fused.reshape(T, NUM_JOINTS, _COORD_DIM).copy()

        # if np.random.rand() < self.mirror_prob:
        #     coords = self._mirror(coords)

        coords = self._rotate(coords)

        coords = coords.reshape(coords.shape[0], -1)
        return coords.astype(np.float32)

    def _mirror(self, coords):
        coords = coords[:, _SWAP_IDX, :].copy()
        coords[..., 0] *= -1.0  # lật trục x (trái <-> phải)
        return coords

    def _rotate(self, coords):
        deg = np.random.uniform(-self.rotation_deg, self.rotation_deg)
        rad = np.deg2rad(deg)
        cos, sin = np.cos(rad), np.sin(rad)

        if _COORD_DIM == 2:
            R = np.array([
                [cos, -sin],
                [sin, cos],
            ], dtype=np.float32)
        elif _COORD_DIM == 3:
            R = np.array([
                [cos, -sin, 0],
                [sin, cos, 0],
                [0, 0, 1],
            ], dtype=np.float32)

        return coords @ R.T


class AugmentedSkeletonDataset:
    def __init__(self, base_dataset, augmentor=None, num_augmentations=1):
        self.base_dataset = base_dataset
        self.augmentor = augmentor or SkeletonAugmentor()
        self.num_augmentations = num_augmentations
        self.base_len = len(base_dataset)

    def __len__(self):
        return self.base_len * (1 + self.num_augmentations)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        base_idx = idx % self.base_len
        variant = idx // self.base_len  # 0 = gốc, >=1 = bản augment

        feature, label = self.base_dataset[base_idx]

        if variant == 0:
            return np.array(feature, dtype=np.float32, copy=True), label

        feature_aug = self.augmentor(feature)  # augmentor tự copy bên trong, không đụng `feature` gốc
        return feature_aug, label