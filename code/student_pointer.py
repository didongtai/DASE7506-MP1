"""Train the neural backbone with a causal copy mixture objective."""
import torch
from torch.nn import functional as F

from student_cache import PrefixCacheGPT, prefix_attention


class PointerGPT(PrefixCacheGPT):
    def training_loss(self, ids, targets):
        features = self.features(ids)
        logits = self.head(features).float()
        logp = logits.log_softmax(-1)
        neural_target = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        cache = self.config['prefix_cache']
        attention = prefix_attention(
            features, ids, cache['theta'], cache['previous_bonus'],
            cache.get('decay', 0.))
        # Reference targets enter the loss only; attention is computed entirely
        # from the causal features and tokens already observed in ids.
        matched = (targets[:, :, None] == ids[:, None, 1:])
        copy_target = (attention * matched).sum(-1)
        coefficients = logits.new_full(ids.shape, cache['weight'])
        coefficients[:, 0] = 0
        mixture = ((1-coefficients) * neural_target.exp()
                   + coefficients * copy_target)
        objective_weight = float(self.config.get('copy_objective_weight', 1.))
        return (objective_weight * -mixture.log().mean()
                + (1-objective_weight) * -neural_target.mean())


def build_model(config):
    return PointerGPT(config)
