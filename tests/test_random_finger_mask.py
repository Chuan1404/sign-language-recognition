import importlib.util
import unittest
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    'finger_mask_spatial_graph',
    Path(__file__).resolve().parents[1] / 'src/models/spatial_graph.py',
)
spatial_graph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spatial_graph)


class RandomFingerMaskTests(unittest.TestCase):
    def setUp(self):
        self.nodes = 50
        self.mask = spatial_graph.RandomFingerMask(self.nodes)

    def test_one_complete_finger_per_hand_across_frames(self):
        torch.manual_seed(42)
        x = torch.ones(16, 7, self.nodes, 3)
        y = self.mask(x)
        torch.testing.assert_close(x, torch.ones_like(x))
        torch.testing.assert_close(y[:, :, :8], x[:, :, :8])
        for start in (8, 29):
            torch.testing.assert_close(y[:, :, start], x[:, :, start])
            dropped = y[:, 0, start + 1:start + 21, 0].reshape(16, 5, 4) == 0
            self.assertTrue(torch.all(dropped.sum(dim=(1, 2)) == 4))
            self.assertTrue(torch.all(dropped.all(dim=2).sum(dim=1) == 1))
        torch.testing.assert_close(y, y[:, :1].expand_as(y))

    def test_eval_identity_and_no_random_draw(self):
        self.mask.eval()
        x = torch.randn(2, 5, self.nodes, 2)
        state = torch.get_rng_state().clone()
        self.assertIs(self.mask(x), x)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_new_selection_and_gradients(self):
        torch.manual_seed(7)
        x = torch.ones(32, 4, self.nodes, 2, requires_grad=True)
        y = self.mask(x)
        self.assertFalse(torch.equal(y, self.mask(x)))
        y.sum().backward()
        torch.testing.assert_close(x.grad, y.detach())

    def test_no_checkpoint_parameters_added(self):
        self.assertEqual(len(self.mask.state_dict()), 0)


if __name__ == '__main__':
    unittest.main()
