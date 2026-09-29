"""FP32 inference implementation with reduced SwiGLU temporary allocations.

The inference path reuses intermediate storage; it does not quantize arithmetic
or cache evaluation inputs. Original parameter names and training gradients remain
available. The original v5 modules are left intact for reproducibility.
"""
import torch
from torch import nn
from torch.nn import functional as F

from student_cache import PrefixCacheGPT, prefix_attention


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
        # Reconstruct one 16 MiB FP32 bigram lookup from the training-only CSR
        # asset. This runtime representation is never serialized or updated by
        # evaluation inputs; higher orders retain their sparse representation.
        self._dense_bigram_from_sparse = 2 in getattr(self, 'ngram_uint8_orders', ())
        if self._dense_bigram_from_sparse:
            vocab = config['vocab']
            self.register_buffer('ngram_2_dense_probs', torch.empty(vocab, vocab), persistent=False)
            self.register_buffer('ngram_2_seen', torch.empty(vocab, dtype=torch.bool), persistent=False)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self._dense_bigram_from_sparse and 'ngram_2_probs' in state_dict:
            keys = self.ngram_2_context_keys_cache.long()
            offsets = self.ngram_2_offsets.long()
            rows = torch.repeat_interleave(keys, offsets[1:]-offsets[:-1])
            self.ngram_2_dense_probs.zero_()
            self.ngram_2_dense_probs[rows, self.ngram_2_tokens.long()] = (
                self.ngram_2_normalized_probs_cache)
            self.ngram_2_seen.copy_(self.ngram_2_direct_index_cache >= 0)
        return result

    def _general_ngram_log_probs(self, ids, probabilities):
        return self._mix_ngram_probabilities(ids, probabilities).log()

    def predict_log_probs(self, ids):
        features = self.features(ids)
        logits = self.head(features).float()
        logits.div_(float(self.config.get('temperature', 1.)))
        probabilities = logits.softmax(-1)
        if hasattr(self, 'ngram_tables'):
            probabilities = self._mix_ngram_probabilities(ids, probabilities)
        cache = self.config.get('prefix_cache')
        if cache and ids.shape[1] > 1:
            attention, confidence = prefix_attention(
                features, ids, cache['theta'], cache['previous_bonus'],
                cache.get('decay', 0.), cache.get('suffix_bonus', 0.), return_confidence=True)
            if 'gate' in cache:
                gate = cache['gate']
                coefficients = torch.sigmoid(
                    gate['intercept'] + gate['slope']*(confidence-gate['offset']))
            else:
                coefficients = probabilities.new_full(ids.shape, float(cache['weight']))
            coefficients = coefficients.unsqueeze(-1) * (
                torch.arange(ids.shape[1], device=ids.device)[None, :, None] > 0)
            probabilities.mul_(1-coefficients)
            values = ids[:, None, 1:].expand(-1, ids.shape[1], -1)
            probabilities.scatter_add_(2, values, attention*coefficients)
        return probabilities.log()

    def _mix_ngram_probabilities(self, ids, probabilities):
        """Mix sparse causal n-gram distributions with neural probabilities."""
        batch, length = ids.shape
        vocabulary = self.config['vocab']
        row_grid = torch.arange(
            batch * length, device=ids.device).view(batch, length)
        row_count = batch * length
        found_by_order, standard_found_by_order, index_by_order = {}, {}, {}
        singleton_found_by_order, singleton_index_by_order = {}, {}

        for order, _ in self.ngram_tables:
            first_position = order - 2
            if first_position >= length:
                found_by_order[order] = torch.zeros(
                    row_count, dtype=torch.bool, device=ids.device)
                index_by_order[order] = torch.zeros(
                    row_count, dtype=torch.long, device=ids.device)
                standard_found_by_order[order] = found_by_order[order]
                if order in self.ngram_singleton_orders:
                    singleton_found_by_order[order] = found_by_order[order]
                    singleton_index_by_order[order] = index_by_order[order]
                continue
            usable = length - first_position
            contexts = ids[:, :usable].long()
            for offset in range(1, order - 1):
                contexts = contexts * vocabulary + ids[:, offset:offset + usable]
            contexts = contexts.reshape(-1)
            rows = row_grid[:, first_position:].reshape(-1)
            dense_name = f'ngram_{order}_dense_probs'
            if dense_name in self._buffers:
                seen = getattr(self, f'ngram_{order}_seen')[contexts]
                found = torch.zeros(row_count, dtype=torch.bool, device=ids.device)
                indices = torch.zeros(row_count, dtype=torch.long, device=ids.device)
                found[rows[seen]] = True
                indices[rows[seen]] = contexts[seen]
                found_by_order[order] = found
                standard_found_by_order[order] = found
                index_by_order[order] = indices
                continue
            if order in self.ngram_delta_orders or order in self.ngram_packed_orders:
                keys = getattr(self, f'ngram_{order}_context_keys_cache')
            else:
                keys = getattr(self, f'ngram_{order}_context_keys')
            if order in self.ngram_singleton_orders:
                keys = getattr(self, f'ngram_{order}_combined_keys_cache')
                candidate = torch.searchsorted(keys, contexts)
                safe = candidate.clamp_max(len(keys) - 1)
                present = (candidate < len(keys)) & (keys[safe] == contexts)
                encoded = getattr(
                    self, f'ngram_{order}_combined_indices_cache')[safe]
                standard_present = present & (encoded >= 0)
                singleton_present = present & (encoded < 0)
                standard_found = torch.zeros(
                    row_count, dtype=torch.bool, device=ids.device)
                standard_indices = torch.zeros(
                    row_count, dtype=torch.long, device=ids.device)
                singleton_found = torch.zeros(
                    row_count, dtype=torch.bool, device=ids.device)
                singleton_indices = torch.zeros(
                    row_count, dtype=torch.long, device=ids.device)
                standard_found[rows[standard_present]] = True
                standard_indices[rows[standard_present]] = encoded[
                    standard_present].long()
                singleton_found[rows[singleton_present]] = True
                singleton_indices[rows[singleton_present]] = (
                    -encoded[singleton_present].long() - 1)
                standard_found_by_order[order] = standard_found
                index_by_order[order] = standard_indices
                singleton_found_by_order[order] = singleton_found
                singleton_index_by_order[order] = singleton_indices
                found_by_order[order] = standard_found | singleton_found
            else:
                if order in self.ngram_direct_orders:
                    candidate = getattr(
                        self, f'ngram_{order}_direct_index_cache')[contexts]
                    present = candidate >= 0
                    safe = candidate.clamp_min(0).long()
                else:
                    candidate = torch.searchsorted(keys, contexts)
                    safe = candidate.clamp_max(len(keys) - 1)
                    present = (candidate < len(keys)) & (keys[safe] == contexts)
                found = torch.zeros(
                    row_count, dtype=torch.bool, device=ids.device)
                indices = torch.zeros(
                    row_count, dtype=torch.long, device=ids.device)
                found[rows[present]] = True
                indices[rows[present]] = safe[present]
                standard_found_by_order[order] = found
                index_by_order[order] = indices
                found_by_order[order] = found

        neural_coefficients = torch.full(
            (row_count,), self.ngram_neural_weight, device=ids.device)
        table_coefficients = {
            order: torch.zeros(row_count, device=ids.device)
            for order, _ in self.ngram_tables}
        orders = [order for order, _ in self.ngram_tables]
        for desired_order, weight in self.ngram_tables:
            remaining = torch.ones(row_count, dtype=torch.bool, device=ids.device)
            for fallback_order in reversed([x for x in orders if x <= desired_order]):
                selected = remaining & found_by_order[fallback_order]
                table_coefficients[fallback_order][selected] += weight
                remaining &= ~selected
            neural_coefficients[remaining] += weight

        flat_probabilities = probabilities.view(row_count, vocabulary)
        flat_probabilities.mul_(neural_coefficients.unsqueeze(1))
        for order, _ in self.ngram_tables:
            coefficients = table_coefficients[order]
            selected_rows = torch.nonzero(
                (coefficients > 0) & standard_found_by_order[order],
                as_tuple=False).squeeze(1)
            if not selected_rows.numel():
                pass
            else:
                dense_name = f'ngram_{order}_dense_probs'
                if dense_name in self._buffers:
                    contexts = ids.reshape(-1)[selected_rows]
                    flat_probabilities[selected_rows] += (
                        getattr(self, dense_name)[contexts].float()
                        * coefficients[selected_rows].unsqueeze(1))
                else:
                    context_indices = index_by_order[order][selected_rows]
                    offsets = getattr(self, f'ngram_{order}_offsets')
                    starts = offsets[context_indices].long()
                    ends = offsets[context_indices + 1].long()
                    counts = ends - starts
                    repeated_rows = torch.repeat_interleave(selected_rows, counts)
                    repeated_starts = torch.repeat_interleave(starts, counts)
                    group_bases = torch.repeat_interleave(
                        counts.cumsum(0) - counts, counts)
                    entries = (torch.arange(counts.sum(), device=ids.device)
                               - group_bases + repeated_starts)
                    tokens = getattr(
                        self, f'ngram_{order}_tokens')[entries].long()
                    stored_probabilities = getattr(
                        self, f'ngram_{order}_probs')
                    if stored_probabilities.dtype == torch.uint8:
                        values = getattr(
                            self, f'ngram_{order}_normalized_probs_cache')[entries]
                    else:
                        values = stored_probabilities[entries].float()
                    values = values * coefficients[repeated_rows]
                    linear_indices = repeated_rows * vocabulary + tokens
                    flat_probabilities.view(-1).scatter_add_(
                        0, linear_indices, values)
            if order in self.ngram_singleton_orders:
                singleton_rows = torch.nonzero(
                    (coefficients > 0) & singleton_found_by_order[order],
                    as_tuple=False).squeeze(1)
                singleton_indices = singleton_index_by_order[order][singleton_rows]
                singleton_tokens = getattr(
                    self, f'ngram_{order}_singleton_tokens')[
                        singleton_indices].long()
                flat_probabilities.view(-1).scatter_add_(
                    0, singleton_rows * vocabulary + singleton_tokens,
                    coefficients[singleton_rows])

        # The uint8 path normalizes every table row once when loading, so the
        # convex mixture is already normalized.  Legacy FP16 tables retain the
        # explicit correction for their small row-sum rounding errors.
        if len(self.ngram_uint8_orders) != len(self.ngram_tables):
            probabilities.div_(probabilities.sum(-1, keepdim=True))
        return probabilities


def build_model(config):
    return FastGPT(config)
