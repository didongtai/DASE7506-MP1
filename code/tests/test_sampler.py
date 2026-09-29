"""Tests for the training-window sampling strategies."""
import unittest
import torch
from train import WindowStartSampler


class SamplerTests(unittest.TestCase):
    def test_coverage_pass_uses_non_overlapping_windows(self):
        rng = torch.Generator().manual_seed(17)
        sampler = WindowStartSampler(
            token_count=256 * 100 + 1,
            batch_size=32,
            context=256,
            mode='coverage',
            generator=rng,
        )
        starts = sampler.sample().sort().values
        self.assertTrue(((starts[1:] - starts[:-1]) >= 256).all())
        self.assertTrue((starts >= 0).all())
        self.assertTrue((starts + 256 < sampler.token_count).all())

    def test_random_sampling_is_reproducible(self):
        a = WindowStartSampler(10_000, 8, 256, 'random', torch.Generator().manual_seed(17))
        b = WindowStartSampler(10_000, 8, 256, 'random', torch.Generator().manual_seed(17))
        torch.testing.assert_close(a.sample(), b.sample())


if __name__ == '__main__':
    unittest.main()
