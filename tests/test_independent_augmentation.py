import unittest
from unittest.mock import patch

import numpy as np
from torch.utils.data import DataLoader

from config import _COORD_DIM, _N_POSE
from src.data.augmentation import NUM_JOINTS, AugmentedSkeletonDataset, SkeletonAugmentor
from src.training.train import collate_fn


class IndependentAugmentationTests(unittest.TestCase):
    def feature(self, blocks=3, frames=20):
        values = np.arange(1, frames + 1, dtype=np.float32)[:, None]
        return np.broadcast_to(values, (frames, blocks * NUM_JOINTS * _COORD_DIM)).copy()

    def test_default_variants_speed_directions_and_metadata(self):
        source = self.feature()
        dataset = AugmentedSkeletonDataset([(source, 7, 'video')], seed=42)
        self.assertEqual(dataset.variant_names, ('original', 'rotate', 'scale', 'mirror',
                                                'speed_fast', 'speed_slow', 'noise'))
        self.assertEqual(len(dataset), 7)
        for index in range(len(dataset)):
            result, label, video_id = dataset[index]
            self.assertEqual((label, video_id), (7, 'video'))
            self.assertEqual(result.dtype, np.float32)
            self.assertFalse(np.shares_memory(result, source))
        fast = dataset[4][0]
        slow = dataset[5][0]
        self.assertEqual(len(fast), 16)
        self.assertEqual(len(slow), 25)
        for result in (fast, slow):
            np.testing.assert_array_equal(result[0], source[0])
            np.testing.assert_array_equal(result[-1], source[-1])
            self.assertTrue(np.all(np.diff(result[:, 0]) > 0))
        np.testing.assert_array_equal(source, self.feature())
        features, labels, mask, ids = next(iter(DataLoader(dataset, batch_size=7, collate_fn=collate_fn)))
        self.assertEqual(features.shape[:2], (7, 25))
        self.assertEqual(mask.sum(1).tolist(), [20, 20, 20, 20, 16, 25, 20])
        self.assertEqual(labels.tolist(), [7] * 7)

    def test_methods_start_from_original_without_composition(self):
        aug = SkeletonAugmentor(enable_mirror=False, enable_noise=False, speed_perturb_prob=0)
        source = self.feature()
        dataset = AugmentedSkeletonDataset([(source, 2, 'a')], aug)
        with patch.object(aug, '_rotate', side_effect=lambda x: x + 10) as rotate, \
                patch.object(aug, '_scale', side_effect=lambda x: x * 3) as scale:
            np.testing.assert_array_equal(dataset[1][0], source + 10)
            scale.assert_not_called()
            np.testing.assert_array_equal(dataset[2][0], source * 3)
            self.assertEqual(rotate.call_count, 1)
            self.assertEqual(scale.call_count, 1)
        np.testing.assert_array_equal(dataset[0][0], source)

    def test_three_blocks_rotation_and_seed(self):
        source = np.random.default_rng(9).normal(size=self.feature(3).shape).astype(np.float32)
        first = AugmentedSkeletonDataset([(source, 0, 'a')], seed=123)
        second = AugmentedSkeletonDataset([(source, 0, 'a')], seed=123)
        for index in range(len(first)):
            np.testing.assert_array_equal(first[index][0], second[index][0])
        original_pose = source.reshape(20, 3, NUM_JOINTS, _COORD_DIM)[:, :, :_N_POSE]
        rotated = first[1][0].reshape(20, 3, NUM_JOINTS, _COORD_DIM)
        np.testing.assert_array_equal(rotated[:, :, :_N_POSE], original_pose)

    def test_all_methods_and_disabled_methods(self):
        # A configuration retaining both shoulders permits viewpoint simulation.
        removed = [i for i in range(33) if i not in (11, 12, 13)]
        with patch('src.data.augmentation._REMOVE_POSE_IDX', removed):
            aug = SkeletonAugmentor(enable_viewpoint=True, frame_dropout_prob=1)
        dataset = AugmentedSkeletonDataset([(self.feature(), 1, 'a')], aug, seed=3)
        self.assertEqual(len(dataset), 9)
        for i in range(len(dataset)):
            result, _, _ = dataset[i]
            self.assertTrue(np.isfinite(result).all())
        disabled = SkeletonAugmentor(enable_rotation=False, enable_scale=False,
                                     enable_mirror=False, enable_noise=False, speed_perturb_prob=0)
        only_original = AugmentedSkeletonDataset([(self.feature(), 1, 'a')], disabled)
        self.assertEqual(only_original.variant_names, ('original',))
        self.assertEqual(len(only_original), 1)
        with self.assertRaises(IndexError):
            only_original[1]


if __name__ == '__main__':
    unittest.main()
