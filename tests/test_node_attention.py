import unittest

import torch

from src.models.spatial_graph import Node_Attention


class NodeAttentionTests(unittest.TestCase):
    def test_node_weights_and_pooling(self):
        module = Node_Attention(8)
        x = torch.randn(2, 4, 5, 8)
        output, attention = module(x, return_attention=True)
        self.assertEqual(output.shape, (2, 4, 8))
        self.assertEqual(attention.shape, (2, 4, 5, 5))
        torch.testing.assert_close(attention.sum(-1), torch.ones(2, 4, 5))
        self.assertTrue(torch.all(attention >= 0))
        # A permutation of nodes changes their order, not the pooled representation.
        torch.testing.assert_close(module(x[:, :, [3, 0, 4, 1, 2]]), output)

    def test_padding_no_temporal_mixing_and_qkv_gradients(self):
        module = Node_Attention(8)
        x = torch.randn(2, 4, 5, 8, requires_grad=True)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        output, weights = module(x, mask, return_attention=True)
        self.assertTrue(torch.all(output[0, 2:] == 0))
        self.assertTrue(torch.all(weights[0, 2:] == 0))
        changed = x.detach().clone()
        changed[:, 3] = 999
        torch.testing.assert_close(module(changed, mask)[:, :3], output[:, :3])
        output.square().sum().backward()
        for projection in (module.query, module.key, module.value):
            self.assertIsNotNone(projection.weight.grad)
            self.assertTrue(torch.isfinite(projection.weight.grad).all())
            self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.all(x.grad[0, 2:] == 0))


if __name__ == '__main__':
    unittest.main()
