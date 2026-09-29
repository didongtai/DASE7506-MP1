"""Parameter-matched SwiGLU variant for the MP1 language model.

The baseline feed-forward network uses two matrices with hidden width 4d. This
variant uses three matrices and hidden width floor(8d/3), keeping its leading
parameter count and multiply count approximately unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, width)

    def forward(self, x):
        return self.output(F.silu(self.gate(x)) * self.value(x))


class Block(nn.Module):
    def __init__(self, width=128, heads=4, hidden=None, dropout=0.):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = SwiGLU(width, hidden or (8 * width) // 3)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0., is_causal=True)
        x = x + self.residual_dropout(
            self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.residual_dropout(self.mlp(self.norm2(x)))


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        hidden = config.get('ffn_hidden', (8 * width) // 3)
        dropout = config.get('dropout', 0.)
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(width, config['heads'], hidden, dropout) for _ in range(config['depth'])
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        if config.get('scaled_residual_init', False):
            residual_std = .02 / (2 * config['depth']) ** .5
            for block in self.blocks:
                nn.init.normal_(block.proj.weight, std=residual_std)
                nn.init.normal_(block.mlp.output.weight, std=residual_std)
        self.head.weight = self.token.weight
        ngram = config.get('ngram')
        if ngram:
            if 'tables' in ngram:
                self.ngram_neural_weight = float(ngram['neural_weight'])
                self.ngram_tables = []
                self.ngram_delta_orders = set()
                self.ngram_packed_orders = set()
                self.ngram_uint8_orders = set()
                self.ngram_direct_orders = set()
                for table in ngram['tables']:
                    order = int(table['order'])
                    self.ngram_tables.append((order, float(table['weight'])))
                    if table.get('dense', False):
                        self.register_buffer(
                            f'ngram_{order}_dense_probs',
                            torch.empty(config['vocab'], config['vocab'],
                                        dtype=torch.float16))
                        self.register_buffer(
                            f'ngram_{order}_seen',
                            torch.empty(config['vocab'], dtype=torch.bool))
                        continue
                    if table.get('packed_keys', False):
                        self.ngram_packed_orders.add(order)
                        self.register_buffer(
                            f'ngram_{order}_context_high',
                            torch.empty(table['contexts'], dtype=torch.int16))
                        self.register_buffer(
                            f'ngram_{order}_context_low',
                            torch.empty(table['contexts'], dtype=torch.int32))
                        self.register_buffer(
                            f'ngram_{order}_context_keys_cache',
                            torch.empty(table['contexts'], dtype=torch.int64),
                            persistent=False)
                    elif table.get('delta_keys', False):
                        self.ngram_delta_orders.add(order)
                        self.register_buffer(
                            f'ngram_{order}_context_base',
                            torch.empty((), dtype=torch.int64))
                        self.register_buffer(
                            f'ngram_{order}_context_deltas',
                            torch.empty(table['contexts'], dtype=torch.int32))
                        self.register_buffer(
                            f'ngram_{order}_context_keys_cache',
                            torch.empty(table['contexts'], dtype=torch.int64),
                            persistent=False)
                    else:
                        self.register_buffer(
                            f'ngram_{order}_context_keys',
                            torch.empty(table['contexts'], dtype=torch.int64))
                    self.register_buffer(
                        f'ngram_{order}_offsets',
                        torch.empty(table['contexts'] + 1, dtype=torch.int32))
                    self.register_buffer(
                        f'ngram_{order}_tokens',
                        torch.empty(table['entries'], dtype=torch.int16))
                    self.register_buffer(
                        f'ngram_{order}_probs',
                        torch.empty(
                            table['entries'],
                            dtype=(torch.uint8 if table.get('prob_dtype') == 'uint8'
                                   else torch.float16)))
                    if table.get('prob_dtype') == 'uint8':
                        self.ngram_uint8_orders.add(order)
                        self.register_buffer(
                            f'ngram_{order}_normalized_probs_cache',
                            torch.empty(table['entries'], dtype=torch.float32),
                            persistent=False)
                    if order == 2:
                        self.ngram_direct_orders.add(order)
                        self.register_buffer(
                            f'ngram_{order}_direct_index_cache',
                            torch.empty(
                                config['vocab'] ** (order - 1),
                                dtype=torch.int32),
                            persistent=False)
                self.ngram_singleton_orders = set()
                for table in ngram.get('singleton_tables', []):
                    order = int(table['order'])
                    self.ngram_singleton_orders.add(order)
                    self.register_buffer(
                        f'ngram_{order}_singleton_high',
                        torch.empty(table['contexts'], dtype=torch.int16))
                    self.register_buffer(
                        f'ngram_{order}_singleton_low',
                        torch.empty(table['contexts'], dtype=torch.int32))
                    self.register_buffer(
                        f'ngram_{order}_singleton_tokens',
                        torch.empty(table['contexts'], dtype=torch.int16))
                    self.register_buffer(
                        f'ngram_{order}_singleton_keys_cache',
                        torch.empty(table['contexts'], dtype=torch.int64),
                        persistent=False)
                    standard_contexts = next(
                        item['contexts'] for item in ngram['tables']
                        if int(item['order']) == order)
                    combined_contexts = standard_contexts + table['contexts']
                    self.register_buffer(
                        f'ngram_{order}_combined_keys_cache',
                        torch.empty(combined_contexts, dtype=torch.int64),
                        persistent=False)
                    self.register_buffer(
                        f'ngram_{order}_combined_indices_cache',
                        torch.empty(combined_contexts, dtype=torch.int32),
                        persistent=False)
                return
            self.ngram_neural_weight = float(ngram['neural_weight'])
            self.ngram_bigram_weight = float(ngram['bigram_weight'])
            self.ngram_trigram_weight = float(ngram['trigram_weight'])
            self.register_buffer(
                'bigram_probs', torch.empty(config['vocab'], config['vocab']))
            self.register_buffer(
                'ngram_context_keys', torch.empty(ngram['contexts'], dtype=torch.int32))
            self.register_buffer(
                'ngram_offsets', torch.empty(ngram['contexts'] + 1, dtype=torch.int32))
            self.register_buffer(
                'ngram_tokens', torch.empty(ngram['entries'], dtype=torch.int16))
            self.register_buffer(
                'ngram_probs', torch.empty(ngram['entries']))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.embedding_dropout(self.token(ids) + self.pos(positions))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        # Augmentation scripts first load a neural-only checkpoint with
        # strict=False and populate table buffers afterwards.  Cache rebuilding
        # is needed only when the loaded state already contains n-gram tensors.
        if hasattr(self, 'ngram_tables') and not any(
                name.startswith('ngram_') for name in state_dict):
            return result
        # Delta coding reduces the serialized inference asset.  Reconstructing
        # sorted keys once after loading keeps lookup exact and deterministic.
        for order in getattr(self, 'ngram_delta_orders', ()):
            base = getattr(self, f'ngram_{order}_context_base')
            deltas = getattr(self, f'ngram_{order}_context_deltas')
            getattr(self, f'ngram_{order}_context_keys_cache').copy_(
                base + deltas.long().cumsum(0))
        for order in getattr(self, 'ngram_packed_orders', ()):
            high = getattr(self, f'ngram_{order}_context_high').long()
            low = getattr(self, f'ngram_{order}_context_low').long()
            getattr(self, f'ngram_{order}_context_keys_cache').copy_(
                (high << 32) + (low & 0xffffffff))
        for order in getattr(self, 'ngram_direct_orders', ()):
            if order in getattr(self, 'ngram_delta_orders', ()) or order in getattr(
                    self, 'ngram_packed_orders', ()):
                keys = getattr(self, f'ngram_{order}_context_keys_cache')
            else:
                keys = getattr(self, f'ngram_{order}_context_keys')
            direct = getattr(self, f'ngram_{order}_direct_index_cache')
            direct.fill_(-1)
            direct[keys.long()] = torch.arange(
                len(keys), dtype=torch.int32, device=keys.device)
        for order in getattr(self, 'ngram_singleton_orders', ()):
            high = getattr(self, f'ngram_{order}_singleton_high').long()
            low = getattr(self, f'ngram_{order}_singleton_low').long()
            getattr(self, f'ngram_{order}_singleton_keys_cache').copy_(
                (high << 32) + (low & 0xffffffff))
            if order in getattr(self, 'ngram_delta_orders', ()) or order in getattr(
                    self, 'ngram_packed_orders', ()):
                standard_keys = getattr(
                    self, f'ngram_{order}_context_keys_cache')
            else:
                standard_keys = getattr(self, f'ngram_{order}_context_keys')
            singleton_keys = getattr(self, f'ngram_{order}_singleton_keys_cache')
            combined_keys, permutation = torch.sort(torch.cat((
                standard_keys, singleton_keys)))
            encoded_indices = torch.cat((
                torch.arange(len(standard_keys), dtype=torch.int32,
                             device=standard_keys.device),
                -torch.arange(len(singleton_keys), dtype=torch.int32,
                              device=standard_keys.device) - 1))
            getattr(self, f'ngram_{order}_combined_keys_cache').copy_(
                combined_keys)
            getattr(self, f'ngram_{order}_combined_indices_cache').copy_(
                encoded_indices[permutation])
        for order in getattr(self, 'ngram_uint8_orders', ()):
            probabilities = getattr(self, f'ngram_{order}_probs').int()
            offsets = getattr(self, f'ngram_{order}_offsets').long()
            cumulative = torch.cat((
                torch.zeros(1, dtype=torch.int64, device=probabilities.device),
                probabilities.long().cumsum(0)))
            sums = cumulative[offsets[1:]] - cumulative[offsets[:-1]]
            counts = offsets[1:] - offsets[:-1]
            getattr(self, f'ngram_{order}_normalized_probs_cache').copy_(
                probabilities.float()
                / torch.repeat_interleave(sums.float(), counts))
        return result

    def predict_log_probs(self, ids):
        probabilities = F.softmax(self(ids).float(), dim=-1)
        if hasattr(self, 'ngram_tables'):
            return self._general_ngram_log_probs(ids, probabilities)
        if 'bigram_probs' not in self._buffers:
            return probabilities.log()

        batch, length = ids.shape
        vocabulary = self.config['vocab']
        flat_probabilities = probabilities.view(-1, vocabulary)
        flat_ids = ids.reshape(-1)

        # Every position has a bigram distribution.  The trigram component
        # backs off to that distribution at the first position in a window and
        # whenever its two-token context was unseen in the training corpus.
        bigram_coefficients = torch.full(
            (flat_ids.numel(),),
            self.ngram_bigram_weight + self.ngram_trigram_weight,
            device=ids.device)
        if length > 1:
            contexts = (ids[:, :-1] * vocabulary + ids[:, 1:]).reshape(-1)
            context_indices = torch.searchsorted(self.ngram_context_keys, contexts)
            safe_indices = context_indices.clamp_max(len(self.ngram_context_keys) - 1)
            found = ((context_indices < len(self.ngram_context_keys))
                     & (self.ngram_context_keys[safe_indices] == contexts))
            rows = torch.arange(
                batch * length, device=ids.device).view(batch, length)[:, 1:].reshape(-1)
            found_rows = rows[found]
            found_indices = safe_indices[found]
            bigram_coefficients[found_rows] = self.ngram_bigram_weight
        else:
            found_rows = torch.empty(0, dtype=torch.long, device=ids.device)
            found_indices = torch.empty(0, dtype=torch.long, device=ids.device)

        flat_probabilities.mul_(self.ngram_neural_weight)
        flat_probabilities.add_(
            self.bigram_probs[flat_ids] * bigram_coefficients.unsqueeze(1))

        if found_rows.numel():
            starts = self.ngram_offsets[found_indices].long()
            ends = self.ngram_offsets[found_indices + 1].long()
            counts = ends - starts
            repeated_rows = torch.repeat_interleave(found_rows, counts)
            repeated_starts = torch.repeat_interleave(starts, counts)
            group_bases = torch.repeat_interleave(counts.cumsum(0) - counts, counts)
            entries = (torch.arange(counts.sum(), device=ids.device)
                       - group_bases + repeated_starts)
            linear_indices = (repeated_rows * vocabulary
                              + self.ngram_tokens[entries].long())
            flat_probabilities.view(-1).scatter_add_(
                0, linear_indices,
                self.ngram_trigram_weight * self.ngram_probs[entries])
        return probabilities.log()

    def _general_ngram_log_probs(self, ids, probabilities):
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
        return probabilities.log()


def build_model(config):
    return StudentGPT(config)
