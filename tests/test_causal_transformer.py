import unittest

import torch

from src.models import ISLR_Causal_Transformer, ISLR_Transformer


class CausalTransformerTests(unittest.TestCase):
    def make_model(self, cls=ISLR_Causal_Transformer):
        return cls(hidden_dim=8, num_encoder_layers=2, nhead=2,
                   dim_feedforward=16, dropout=0, max_seq_len=12,
                   num_classes=3, num_nodes=5)

    def features(self):
        model = self.make_model()
        return torch.randn(2, 7, 3 * model.feature_dim)

    def test_future_changes_do_not_change_prefix_embeddings(self):
        model = self.make_model().eval()
        x = self.features()
        changed = x.clone()
        changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10
        mask = torch.ones(2, 7, dtype=torch.bool)
        with torch.no_grad():
            full = model.encode(x, mask)
            modified = model.encode(changed, mask)
            prefix = model.encode(x[:, :4], mask[:, :4])
        torch.testing.assert_close(full[:, :4], modified[:, :4])
        torch.testing.assert_close(full[:, :4], prefix)

    def test_padding_and_training_gradients(self):
        model = self.make_model()
        x = self.features().requires_grad_()
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
        logits, loss = model(x, labels=torch.tensor([0, 2]), video_mask=mask)
        self.assertEqual(logits.shape, (2, 3))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.all(x.grad[0, 4:] == 0))
        self.assertGreater(model.encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum().item(), 0)
        model.eval()
        changed = x.detach().clone()
        changed[0, 4:] = 9999
        with torch.no_grad():
            a, _ = model(x.detach(), video_mask=mask)
            b, _ = model(changed, video_mask=mask.bool())
        torch.testing.assert_close(a, b)

    def test_original_weights_and_single_frame_behavior_preserved(self):
        original = self.make_model(ISLR_Transformer).eval()
        causal = self.make_model().eval()
        causal.load_state_dict(original.state_dict(), strict=True)
        x = self.features()[:, :1]
        mask = torch.ones(2, 1, dtype=torch.bool)
        with torch.no_grad():
            a, _ = original(x, video_mask=mask)
            b, _ = causal(x, video_mask=mask)
        torch.testing.assert_close(a, b)


if __name__ == '__main__':
    unittest.main()
