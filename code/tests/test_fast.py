"""Fast inference must retain causal probabilities, including short n-gram inputs."""
import unittest

import torch

from student_cache import build_model as build_reference
from student_fast import build_model


def fixture():
    config = dict(vocab=2048, width=16, heads=2, depth=2, context=256,
                  temperature=1.15,
                  prefix_cache=dict(theta=10., previous_bonus=0., suffix_bonus=2.,
                                    gate=dict(slope=8., offset=.65, intercept=-3.17)),
                  ngram=dict(neural_weight=.8, tables=[
                      dict(order=order, weight=.05, contexts=1, entries=1,
                           packed_keys=True, prob_dtype='uint8') for order in (2, 3, 4, 5)],
                      singleton_tables=[dict(order=5, contexts=1)]))
    reference = build_reference(config)
    state = reference.state_dict()

    def key(prefix):
        value = 0
        for token in prefix:
            value = value*2048+token
        low = value & 0xffffffff
        return torch.tensor([value >> 32], dtype=torch.int16), torch.tensor(
            [low if low < 2**31 else low-2**32], dtype=torch.int32)

    for order in (2, 3, 4, 5):
        high, low = key(list(range(2, order+1)))
        state[f'ngram_{order}_context_high'] = high
        state[f'ngram_{order}_context_low'] = low
        state[f'ngram_{order}_offsets'] = torch.tensor([0, 1], dtype=torch.int32)
        state[f'ngram_{order}_tokens'] = torch.tensor([order+1], dtype=torch.int16)
        state[f'ngram_{order}_probs'] = torch.tensor([255], dtype=torch.uint8)
    high, low = key([3, 4, 5, 6])
    state['ngram_5_singleton_high'] = high
    state['ngram_5_singleton_low'] = low
    state['ngram_5_singleton_tokens'] = torch.tensor([7], dtype=torch.int16)
    reference.load_state_dict(state)
    fast = build_model(config)
    fast.load_state_dict(state)
    return reference.eval(), fast.eval()


class FastTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7516)
        self.reference, self.model = fixture()
        self.ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 2, 3]])

    def test_probabilities_match_original_with_all_ngram_orders(self):
        with torch.no_grad():
            actual = self.model.predict_log_probs(self.ids)
            expected = self.reference.predict_log_probs(self.ids)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_one_to_three_tokens_match_longer_causal_prefix(self):
        with torch.no_grad():
            full = self.model.predict_log_probs(self.ids)
            for length in (1, 2, 3):
                actual = self.model.predict_log_probs(self.ids[:, :length])
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, full[:, :length], atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(2, length),
                                           atol=1e-6, rtol=0)

    def test_causality_and_independent_calls(self):
        changed = self.ids.clone()
        changed[:, 4:] += 23
        with torch.no_grad():
            first = self.model.predict_log_probs(self.ids)
            other = self.model.predict_log_probs(changed)
            alone = self.model.predict_log_probs(self.ids[:1])
            again = self.model.predict_log_probs(self.ids)
        torch.testing.assert_close(first[:, :4], other[:, :4], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(first[:1], alone, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(first, again, atol=0, rtol=0)

    def test_forward_keeps_training_gradients(self):
        self.model.train()
        loss = torch.nn.functional.cross_entropy(
            self.model(self.ids[:, :-1]).flatten(0, 1), self.ids[:, 1:].flatten())
        loss.backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == '__main__':
    unittest.main()
