import unittest

import torch

from student_cache import build_model, prefix_attention
from student_pointer import build_model as build_pointer


class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7506)
        self.config = dict(
            vocab=2048, width=16, heads=2, depth=1, context=256,
            temperature=1.1,
            prefix_cache=dict(theta=10., previous_bonus=0., suffix_bonus=2.,
                              gate=dict(slope=8., offset=.65, intercept=-3.3)))

    def test_attention_values_are_strictly_observed(self):
        ids = torch.tensor([[3, 4, 3, 4, 8]])
        features = torch.randn(1, 5, 16)
        attention = prefix_attention(features, ids, 10., 0., suffix_bonus=2.)
        for position in range(5):
            self.assertTrue(torch.equal(
                attention[:, position, position:],
                torch.zeros_like(attention[:, position, position:])))
        self.assertEqual(attention[0, 1, 0].item(), 1.)
        torch.testing.assert_close(attention[:, 1:].sum(-1), torch.ones(1, 4))

    def test_future_inputs_and_other_examples_do_not_change_predictions(self):
        model = build_model(self.config).eval()
        x = torch.randint(0, 2048, (2, 24))
        changed = x.clone()
        changed[:, 12:] = (changed[:, 12:]+31) % 2048
        with torch.no_grad():
            original = model.predict_log_probs(x)
            altered = model.predict_log_probs(changed)
            alone = model.predict_log_probs(x[:1])
            model.predict_log_probs((x+19) % 2048)
            repeated = model.predict_log_probs(x)
        torch.testing.assert_close(original[:, :12], altered[:, :12])
        torch.testing.assert_close(original[:1], alone)
        torch.testing.assert_close(original, repeated, atol=0, rtol=0)
        torch.testing.assert_close(
            torch.logsumexp(original, -1), torch.zeros(2, 24), atol=1e-6, rtol=0)

    def test_one_token_prefix_has_no_copy_mass(self):
        model = build_model(self.config).eval()
        ids = torch.tensor([[9]])
        with torch.no_grad():
            actual = model.predict_log_probs(ids)
            expected = (model(ids)/self.config['temperature']).log_softmax(-1)
        torch.testing.assert_close(actual, expected)

    def test_training_copy_objective_has_finite_backbone_gradients(self):
        self.config['prefix_cache'] = dict(theta=10., previous_bonus=2., weight=.06)
        self.config['copy_objective_weight'] = .75
        model = build_pointer(self.config).train()
        tokens = torch.tensor([[3, 4, 5, 3, 4, 5, 9], [7, 2, 7, 2, 7, 2, 8]])
        loss = model.training_loss(tokens[:, :-1], tokens[:, 1:])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.token.weight.grad).all())
        self.assertGreater(model.token.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
