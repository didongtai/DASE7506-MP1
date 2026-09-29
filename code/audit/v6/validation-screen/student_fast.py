"""FP32 inference implementation with reduced SwiGLU temporary allocations.

The inference path reuses intermediate storage; it does not quantize arithmetic
or cache evaluation inputs. Original parameter names and training gradients remain
available. The original v5 modules are left intact for reproducibility.
"""
import torch
from torch import nn
from torch.nn import functional as F

from student_cache import PrefixCacheGPT


class InplaceSwiGLU(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.gate = original.gate
        self.value = original.value
        self.output = original.output

    def forward(self, x):
        if self.training or torch.is_grad_enabled():
            return self.output(F.silu(self.gate(x)) * self.value(x))
        gate = F.silu(self.gate(x), inplace=True)
        gate.mul_(self.value(x))
        return self.output(gate)


class FastGPT(PrefixCacheGPT):
    def __init__(self, config):
        super().__init__(config)
        for block in self.blocks:
            block.mlp = InplaceSwiGLU(block.mlp)

    def _general_ngram_log_probs(self, ids, probabilities):
        # The legacy table mixer omits bookkeeping for orders longer than the
        # input. Pad only this stateless, causal lookup, then discard padding.
        # Valid positions depend on exactly the same observed prefixes.
        length = ids.shape[1]
        if length < 4:
            padded_ids = F.pad(ids, (0, 4-length))
            padded_probabilities = F.pad(
                probabilities, (0, 0, 0, 4-length), value=1/self.config['vocab'])
            return super()._general_ngram_log_probs(
                padded_ids, padded_probabilities)[:, :length]
        return super()._general_ngram_log_probs(ids, probabilities)


def build_model(config):
    return FastGPT(config)
