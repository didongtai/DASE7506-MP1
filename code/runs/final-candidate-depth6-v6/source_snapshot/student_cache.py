"""SwiGLU/n-gram predictor with a causal, stateless neural prefix cache."""
import torch
from torch.nn import functional as F

from student import StudentGPT


def prefix_attention(features, ids, theta, previous_bonus, decay=0.,
                     suffix_bonus=0., return_confidence=False):
    """Key j predicts ids[j+1], which is available only at query t > j."""
    length = ids.shape[1]
    hidden = F.normalize(features.float(), dim=-1)
    similarity = hidden @ hidden[:, :-1].transpose(1, 2)
    query_positions = torch.arange(length, device=ids.device)[:, None]
    key_positions = torch.arange(length-1, device=ids.device)[None, :]
    distance = query_positions-key_positions
    scores = theta * similarity
    same_previous = ids[:, :, None] == ids[:, None, :-1]
    scores = scores + previous_bonus * same_previous
    if suffix_bonus:
        same_suffix = torch.zeros_like(same_previous)
        same_suffix[:, 1:, 1:] = (
            same_previous[:, 1:, 1:] & same_previous[:, :-1, :-1])
        scores = scores + suffix_bonus * same_suffix
    scores = scores - decay * distance
    scores = scores.masked_fill(distance <= 0, -1e9)
    attention = scores.softmax(-1)
    # Position zero has no eligible key.  Its mixture weight is zero below.
    attention = attention * (query_positions > 0)
    if return_confidence:
        confidence = similarity.masked_fill(distance <= 0, -1.).amax(-1)
        return attention, confidence
    return attention


class PrefixCacheGPT(StudentGPT):
    def predict_log_probs(self, ids):
        features = self.features(ids)
        temperature = float(self.config.get('temperature', 1.))
        probabilities = F.softmax(self.head(features).float()/temperature, dim=-1)
        if hasattr(self, 'ngram_tables'):
            probabilities = self._general_ngram_log_probs(ids, probabilities).exp()
        cache = self.config.get('prefix_cache')
        if cache and ids.shape[1] > 1:
            attention, confidence = prefix_attention(
                features, ids, cache['theta'], cache['previous_bonus'],
                cache.get('decay', 0.), cache.get('suffix_bonus', 0.),
                return_confidence=True)
            if 'gate' in cache:
                gate = cache['gate']
                coefficients = torch.sigmoid(
                    gate['intercept'] + gate['slope']*(confidence-gate['offset']))
            else:
                coefficients = probabilities.new_full(ids.shape, float(cache['weight']))
            coefficients = coefficients.unsqueeze(-1) * (
                torch.arange(ids.shape[1], device=ids.device)[None, :, None] > 0)
            probabilities = probabilities * (1-coefficients)
            values = ids[:, None, 1:].expand(-1, ids.shape[1], -1)
            probabilities.scatter_add_(2, values, attention * coefficients)
        return probabilities.log()


def build_model(config):
    return PrefixCacheGPT(config)
