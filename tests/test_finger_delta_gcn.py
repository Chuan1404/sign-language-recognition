import unittest

import torch

from config import _COORD_DIM, _NUM_NODE, _N_HAND
from src.models.gcn import ISLR_Transformer_FCN
from src.models.spatial_graph import FingerMaskVariants


class FingerDeltaGCNTests(unittest.TestCase):
    def make_model(self):
        return ISLR_Transformer_FCN(
            gcn_channels=(8, 12), d_model=12, num_encoder_layers=1,
            nhead=2, dim_feedforward=16, dropout=0, max_seq_len=16, num_classes=3,
        )

    def test_exact_pair_masks_in_train_and_eval(self):
        module = FingerMaskVariants(_NUM_NODE)
        x = torch.ones(2, 4, _NUM_NODE, _COORD_DIM)
        original = x.clone()
        for training in (True, False):
            module.train(training)
            state = torch.get_rng_state().clone()
            variants = module(x)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertEqual(variants.shape, (2, 6, 4, _NUM_NODE, _COORD_DIM))
            torch.testing.assert_close(variants[:, 0], x)
            for finger in range(5):
                expected = x.clone()
                for offset in (_NUM_NODE - 2 * _N_HAND, _NUM_NODE - _N_HAND):
                    start = offset + 1 + 4 * finger
                    expected[:, :, start:start + 4] = 0
                torch.testing.assert_close(variants[:, finger + 1], expected)
        torch.testing.assert_close(x, original)

    def test_batched_variants_match_separate_shared_gcn_passes(self):
        model = self.make_model().eval()
        self.assertTrue(all(block.gcn.p == 1 for block in model.gcn_block))
        x = torch.randn(2, 5, _NUM_NODE, _COORD_DIM)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        with torch.no_grad():
            actual = model.encode_gcn(x, mask)
            separate = []
            for variant in model.finger_mask(x).unbind(dim=1):
                for block in model.gcn_block:
                    variant = block(variant, mask)
                separate.append(variant.mean(dim=-2))
            torch.testing.assert_close(actual, torch.stack(separate, dim=1))
        self.assertTrue(torch.all(actual[0, :, 3:] == 0))

    def test_classifier_receives_deltas_and_training_backpropagates(self):
        model = self.make_model().train()
        x = torch.randn(2, 5, 3 * _NUM_NODE * _COORD_DIM, requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        calls = []
        handles = [block.register_forward_hook(
            lambda module, args, output: calls.append(args[0].shape[0]),
        ) for block in model.gcn_block]
        output = model(x, labels=torch.tensor([0, 2]), video_mask=mask.long())
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, [12, 12])
        self.assertEqual(output.logits.shape, (2, 3))
        self.assertEqual(output.finger_embeddings.shape, (2, 6, 12))
        self.assertEqual(output.encoded.shape, (2, 12 + 6 * 12))
        expected = output.finger_embeddings[:, :1] - output.finger_embeddings[:, 1:]
        torch.testing.assert_close(output.finger_deltas, expected)
        torch.testing.assert_close(output.encoded[:, 12:24], output.finger_embeddings[:, 0])
        torch.testing.assert_close(output.encoded[:, 24:], expected.flatten(1))
        output.loss.backward()
        for block in model.gcn_block:
            self.assertIsNotNone(block.gcn.phi.weight.grad)
            self.assertTrue(torch.isfinite(block.gcn.phi.weight.grad).all())
        self.assertGreater(model.classifier[-1].weight.grad[:, 24:].abs().sum().item(), 0)
        self.assertTrue(torch.all(x.grad[0, 3:] == 0))

    def test_eval_padding_invariance_and_checkpoint_roundtrip(self):
        model = self.make_model().eval()
        x = torch.randn(2, 5, 3 * _NUM_NODE * _COORD_DIM)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        changed = x.clone()
        changed[0, 3:] = 10000
        restored = self.make_model().eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            output = model(x, video_mask=mask)
            torch.testing.assert_close(output.logits, model(changed, video_mask=mask).logits)
            torch.testing.assert_close(output.logits, restored(x, video_mask=mask).logits)
            self.assertTrue(torch.isfinite(output.logits).all())
            torch.testing.assert_close(output.logits, model(x, video_mask=mask.long()).logits)
            torch.testing.assert_close(model(x).logits, model(x, video_mask=torch.ones_like(mask)).logits)


if __name__ == '__main__':
    unittest.main()
