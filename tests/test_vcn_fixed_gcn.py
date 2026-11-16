import unittest

import torch

from src.models.gcn import ISLR_Transformer_VCN
from src.models.best_performance import V_GCN
from src.models.spatial_graph import V_GCN_V2, V_GCN_Block, Node_Attention


class VCNFixedGCNTransformerTests(unittest.TestCase):
    def make_model(self):
        return ISLR_Transformer_VCN(
            gcn_channels=(8, 12), d_model=12, num_encoder_layers=1,
            nhead=2, dim_feedforward=16, dropout=0, max_seq_len=8, num_classes=3,
        )

    def test_gcn_attention_transformer_gradients_and_learned_A(self):
        model = self.make_model().train()
        self.assertTrue(all(type(block.gcn) is V_GCN for block in model.gcn_block))
        self.assertIsInstance(model.node_attention, Node_Attention)
        x = torch.randn(2, 5, 3 * model.feature_dim, requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        before = [block.gcn.A_in.detach().clone() for block in model.gcn_block]
        output = model(x, labels=torch.tensor([0, 2]), video_mask=mask)
        self.assertEqual(output.logits.shape, (2, 3))
        self.assertEqual(output.encoded.shape, (2, 12))
        output.loss.backward()
        for parameter in (model.node_attention.query.weight, model.gcn_norm.weight,
                          model.gcn_block[0].gcn.phi.weight,
                          model.encoder.layers[0].self_attn.in_proj_weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
        for block, original in zip(model.gcn_block, before):
            torch.testing.assert_close(original, model.adjacency_matrix.unsqueeze(0) * 1e-3)
            self.assertGreater(block.gcn.A_in.grad.abs().sum().item(), 0)
            self.assertFalse(torch.equal(block.gcn.A_in, original))
        self.assertTrue(torch.all(x.grad[0, 3:] == 0))

    def test_transformer_receives_only_encoded_gcn(self):
        model = self.make_model().eval()
        x = torch.randn(2, 4, 3 * model.feature_dim)
        changed = x.clone()
        changed[..., :2 * model.feature_dim] = 999
        seen, incoming = [], []
        hook = model.gcn_norm.register_forward_hook(lambda module, args, out: seen.append(out))
        pre = model.pos_encoder.register_forward_pre_hook(lambda module, args: incoming.append(args[0]))
        for name in ('shape_projection', 'average_projection', 'position_projection'):
            self.assertFalse(hasattr(model, name))
        with torch.no_grad():
            a = model(x)
            b = model(changed)
        torch.testing.assert_close(a.logits, b.logits)
        self.assertEqual(len(incoming), 2)
        torch.testing.assert_close(incoming[0], seen[0])
        hook.remove()
        pre.remove()

    def test_padding_and_optional_mask(self):
        model = self.make_model().eval()
        x = torch.randn(2, 5, 3 * model.feature_dim)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        changed = x.clone()
        changed[0, 3:] = 9999
        with torch.no_grad():
            a = model(x, video_mask=mask)
            b = model(changed, video_mask=mask.bool())
            torch.testing.assert_close(a.logits, b.logits)
            torch.testing.assert_close(model(x).logits, model(x, video_mask=torch.ones_like(mask)).logits)

    def test_residual_preserves_input_when_graph_path_is_zero(self):
        for use_tcn in (False, True):
            for out_channels in (2, 4):
                with self.subTest(tcn=use_tcn, channels=out_channels):
                    block = V_GCN_Block(2, out_channels, 5, torch.eye(5),
                                        use_tcn=use_tcn, dropout=0,
                                        residual=True).eval()
                    with torch.no_grad():
                        block.gcn.phi.weight.zero_()
                        block.gcn.phi.bias.zero_()
                    x = torch.randn(2, 4, 5, 2, requires_grad=True)
                    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
                    output = block(x, mask)
                    expected = block.residual(x).masked_fill(~mask[:, :, None, None], 0)
                    torch.testing.assert_close(output, expected)
                    output.square().sum().backward()
                    self.assertGreater(x.grad[mask].abs().sum().item(), 0)
                    self.assertTrue(torch.all(x.grad[~mask] == 0))

    def test_fixed_graph_equation_and_unchanged_learnable_default(self):
        adjacency = torch.eye(5)
        adjacency[0, 1] = adjacency[1, 0] = 1
        fixed = V_GCN_V2(2, 8, 5, adjacency, learnable_adjacency=False)
        x = torch.randn(2, 4, 5, 2)
        a, graph_a = fixed(x)
        _, graph_b = fixed(x * 3)
        torch.testing.assert_close(graph_a, graph_b)
        degree = adjacency.sum(-1).rsqrt()
        normalized = degree[:, None] * adjacency * degree[None, :]
        torch.testing.assert_close(a, normalized @ fixed.phi(x))
        self.assertIsInstance(V_GCN(2, 8, 5, adjacency).A_in, torch.nn.Parameter)


if __name__ == '__main__':
    unittest.main()
