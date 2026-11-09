import unittest

import numpy as np

from config import _COORD_DIM, _N_HAND, _N_POSE
from src.data.augmentation import NUM_JOINTS, SkeletonAugmentor, rotate_hand_blocks
from src.data.WSASL_raw import MultiFeatureSkeletonAugmentor


class FixedAngleRNG:
    def uniform(self, low, high):
        return high


class HandRotationTests(unittest.TestCase):
    def test_rotation_keeps_pose_wrists_and_missing_points(self):
        for count in (1, 3, 4):
            with self.subTest(blocks=count):
                blocks = np.random.default_rng(42).normal(
                    size=(5, count, NUM_JOINTS, _COORD_DIM),
                ).astype(np.float32)
                for start in (_N_POSE, _N_POSE + _N_HAND):
                    blocks[:, :, start + 3] = 0
                original = blocks.copy()
                aug = SkeletonAugmentor(rotation_deg=90, rng=FixedAngleRNG())
                result = aug._rotate(blocks)
                np.testing.assert_array_equal(result[:, :, :_N_POSE], original[:, :, :_N_POSE])
                for start in (_N_POSE, _N_POSE + _N_HAND):
                    np.testing.assert_array_equal(result[:, :, start], original[:, :, start])
                    np.testing.assert_array_equal(result[:, :, start + 3], 0)
                    before = original[:, :, start + 1] - original[:, :, start]
                    after = result[:, :, start + 1] - result[:, :, start]
                    np.testing.assert_allclose(after[..., 0], -before[..., 1], atol=1e-6)
                    np.testing.assert_allclose(after[..., 1], before[..., 0], atol=1e-6)
                np.testing.assert_array_equal(blocks, original)

    def test_multifeature_rotation_preserves_pose(self):
        blocks = np.random.default_rng(7).normal(
            size=(4, 3, NUM_JOINTS, _COORD_DIM),
        ).astype(np.float32)
        blocks[:, :, _N_POSE + 2] = 0
        fused = blocks.reshape(4, -1)
        aug = MultiFeatureSkeletonAugmentor(
            rotation_deg=45, scale_range=(1, 1), noise_prob=0,
            speed_perturb_prob=0, frame_dropout_prob=0, rng=np.random.default_rng(1),
        )
        output = aug(fused).reshape(blocks.shape)
        np.testing.assert_array_equal(output[:, :, :_N_POSE], blocks[:, :, :_N_POSE])
        np.testing.assert_array_equal(output[:, :, _N_POSE + 2], 0)
        self.assertFalse(np.allclose(output[:, :, _N_POSE + 1], blocks[:, :, _N_POSE + 1]))
        np.testing.assert_array_equal(fused, blocks.reshape(4, -1))


if __name__ == '__main__':
    unittest.main()
