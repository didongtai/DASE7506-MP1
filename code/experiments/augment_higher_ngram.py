"""Attach pruned sparse 2- through 5-gram tables to a neural checkpoint."""
import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROTOCOL, load_data, make_model


VOCAB = 2048
DEFAULT_WEIGHTS = {2: 0.01, 3: 0.06, 4: 0.06, 5: 0.12}
MIN_COUNTS = {2: 1, 3: 1, 4: 2, 5: 2}


def four_values(text, cast, option):
    try:
        values = [cast(value) for value in text.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f'{option} must contain four comma-separated values') from error
    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            f'{option} must contain four comma-separated values')
    return dict(zip((2, 3, 4, 5), values))


HASH_MODULUS = 1_000_003


def build_table(tokens, order, min_count, singleton_fraction=0.):
    size = len(tokens) - order + 1
    context_values = tokens[:size].copy()
    for offset in range(1, order - 1):
        context_values = context_values * VOCAB + tokens[offset:offset + size]
    full_values = context_values * VOCAB + tokens[order - 1:order - 1 + size]
    context_keys, context_counts = np.unique(context_values, return_counts=True)
    full_keys, full_counts = np.unique(full_values, return_counts=True)

    retained_contexts = context_counts >= min_count
    if singleton_fraction:
        threshold = int(singleton_fraction * HASH_MODULUS)
        retained_contexts |= ((context_counts == 1)
                              & (context_keys % HASH_MODULUS < threshold))
    entry_contexts = full_keys // VOCAB
    entry_context_indices = np.searchsorted(context_keys, entry_contexts)
    retained_entries = retained_contexts[entry_context_indices]
    context_keys = context_keys[retained_contexts]
    context_counts = context_counts[retained_contexts]
    full_keys = full_keys[retained_entries]
    full_counts = full_counts[retained_entries]
    entry_contexts = full_keys // VOCAB
    offsets = np.searchsorted(entry_contexts, context_keys).astype(np.int32)
    offsets = np.append(offsets, len(full_keys)).astype(np.int32)
    probabilities = (full_counts / np.repeat(
        context_counts, np.diff(offsets))).astype(np.float16)
    return {
        'context_keys': context_keys.astype(np.int64),
        'offsets': offsets,
        'tokens': (full_keys % VOCAB).astype(np.int16),
        'probs': probabilities,
    }


def delta_encode(keys):
    if not len(keys):
        return np.int64(0), np.empty(0, dtype=np.int32)
    deltas = np.diff(keys, prepend=keys[0])
    if deltas.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError('A context-key delta does not fit in int32.')
    return np.int64(keys[0]), deltas.astype(np.int32)


def pack_keys(keys):
    high = (keys >> 32).astype(np.int16)
    low = (keys & np.int64(0xffffffff)).astype(np.uint32).view(np.int32)
    return high, low


def build_singletons(tokens, order, fraction):
    size = len(tokens) - order + 1
    contexts = tokens[:size].copy()
    for offset in range(1, order - 1):
        contexts = contexts * VOCAB + tokens[offset:offset + size]
    context_keys, first_indices, context_counts = np.unique(
        contexts, return_index=True, return_counts=True)
    selected = ((context_counts == 1)
                & (context_keys % HASH_MODULUS
                   < int(fraction * HASH_MODULUS)))
    keys = context_keys[selected]
    next_tokens = tokens[first_indices[selected] + order - 1].astype(np.int16)
    high, low = pack_keys(keys)
    return {'high': high, 'low': low, 'tokens': next_tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--weights', default='0.01,0.06,0.06,0.12',
                        help='comma-separated weights for orders 2,3,4,5')
    parser.add_argument('--min-counts', default='1,1,2,2',
                        help='comma-separated context thresholds for orders 2,3,4,5')
    parser.add_argument('--singleton-fractions', default='0,0,0,0',
                        help='training-only hash fractions of count-one contexts to retain')
    parser.add_argument('--sparse-bigram', action='store_true',
                        help='store bigrams as CSR instead of a dense table')
    parser.add_argument('--fp16-model-storage', action='store_true',
                        help='store neural floating tensors as FP16; loading restores model dtype')
    parser.add_argument('--delta-context-keys', action='store_true',
                        help='losslessly delta-code sorted context keys as int32')
    parser.add_argument('--packed-context-keys', action='store_true',
                        help='losslessly store absolute context keys as high16/low32')
    parser.add_argument('--uint8-probs', action='store_true',
                        help='store sparse probabilities as normalized 8-bit values')
    parser.add_argument('--split-singleton-order', type=int,
                        help='store selected count-one contexts without CSR offsets/probabilities')
    args = parser.parse_args()
    weights = four_values(args.weights, float, '--weights')
    min_counts = four_values(args.min_counts, int, '--min-counts')
    singleton_fractions = four_values(
        args.singleton_fractions, float, '--singleton-fractions')
    if any(value < 0 for value in weights.values()) or sum(weights.values()) >= 1:
        parser.error('Weights must be non-negative and sum to less than one.')
    if any(value < 1 for value in min_counts.values()):
        parser.error('Minimum counts must be positive integers.')
    if any(value < 0 or value > 1 for value in singleton_fractions.values()):
        parser.error('Singleton fractions must lie between zero and one.')
    if args.delta_context_keys and args.packed_context_keys:
        parser.error('Choose only one context-key compression scheme.')
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if source['protocol'] != PROTOCOL:
        raise ValueError('Checkpoint belongs to a different course protocol.')
    train = load_data()['train'][0].numpy().astype(np.int64, copy=False)
    if (args.split_singleton_order is not None
            and args.split_singleton_order not in weights):
        parser.error('--split-singleton-order must be one of 2,3,4,5.')
    tables, singleton_tables = {}, {}
    for order in sorted(weights):
        split_singletons = order == args.split_singleton_order
        tables[order] = build_table(
            train, order, min_counts[order],
            0. if split_singletons else singleton_fractions[order])
        if split_singletons and singleton_fractions[order]:
            singleton_tables[order] = build_singletons(
                train, order, singleton_fractions[order])
    if args.uint8_probs:
        for table in tables.values():
            table['probs'] = np.clip(
                np.rint(table['probs'].astype(np.float32) * 255),
                1, 255).astype(np.uint8)

    config = copy.deepcopy(source['config'])
    config['ngram'] = {
        'neural_weight': 1 - sum(weights.values()),
        'tables': [
            {'order': order, 'weight': weights[order],
             'contexts': int(len(table['context_keys'])),
             'entries': int(len(table['tokens'])),
             'min_count': min_counts[order],
             'singleton_fraction': singleton_fractions[order],
             'delta_keys': args.delta_context_keys,
             'packed_keys': args.packed_context_keys,
             'prob_dtype': 'uint8' if args.uint8_probs else 'float16',
             'dense': order == 2 and not args.sparse_bigram}
            for order, table in tables.items()
        ],
    }
    if singleton_tables:
        config['ngram']['singleton_tables'] = [
            {'order': order, 'contexts': int(len(table['tokens'])),
             'fraction': singleton_fractions[order], 'packed_keys': True}
            for order, table in singleton_tables.items()]
    model, _ = make_model(source['implementation'], config, torch.device('cpu'))
    missing, unexpected = model.load_state_dict(source['model'], strict=False)
    sparse_orders = [order for order in tables if order != 2 or args.sparse_bigram]
    expected_missing = set()
    for order in sparse_orders:
        if args.packed_context_keys:
            key_names = ('context_high', 'context_low')
        elif args.delta_context_keys:
            key_names = ('context_base', 'context_deltas')
        else:
            key_names = ('context_keys',)
        expected_missing.update(
            f'ngram_{order}_{name}'
            for name in (*key_names, 'offsets', 'tokens', 'probs'))
    for order in singleton_tables:
        expected_missing.update({
            f'ngram_{order}_singleton_high',
            f'ngram_{order}_singleton_low',
            f'ngram_{order}_singleton_tokens'})
    if not args.sparse_bigram:
        expected_missing.update({'ngram_2_dense_probs', 'ngram_2_seen'})
    if set(missing) != expected_missing or unexpected:
        raise ValueError(f'Unexpected state mismatch: missing={missing}, unexpected={unexpected}')
    for order, table in tables.items():
        if order == 2 and not args.sparse_bigram:
            dense = np.zeros((VOCAB, VOCAB), dtype=np.float16)
            rows = np.repeat(table['context_keys'], np.diff(table['offsets']))
            dense[rows, table['tokens'].astype(np.int64)] = table['probs']
            model.ngram_2_dense_probs.copy_(torch.from_numpy(dense))
            seen = np.zeros(VOCAB, dtype=np.bool_)
            seen[table['context_keys']] = True
            model.ngram_2_seen.copy_(torch.from_numpy(seen))
            continue
        if args.packed_context_keys:
            high, low = pack_keys(table['context_keys'])
            getattr(model, f'ngram_{order}_context_high').copy_(
                torch.from_numpy(high))
            getattr(model, f'ngram_{order}_context_low').copy_(
                torch.from_numpy(low))
            names = ('offsets', 'tokens', 'probs')
        elif args.delta_context_keys:
            base, deltas = delta_encode(table['context_keys'])
            getattr(model, f'ngram_{order}_context_base').copy_(torch.as_tensor(base))
            getattr(model, f'ngram_{order}_context_deltas').copy_(torch.from_numpy(deltas))
            names = ('offsets', 'tokens', 'probs')
        else:
            names = ('context_keys', 'offsets', 'tokens', 'probs')
        for name in names:
            getattr(model, f'ngram_{order}_{name}').copy_(torch.from_numpy(table[name]))
    for order, table in singleton_tables.items():
        getattr(model, f'ngram_{order}_singleton_high').copy_(
            torch.from_numpy(table['high']))
        getattr(model, f'ngram_{order}_singleton_low').copy_(
            torch.from_numpy(table['low']))
        getattr(model, f'ngram_{order}_singleton_tokens').copy_(
            torch.from_numpy(table['tokens']))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(source)
    result['config'] = config
    state = model.state_dict()
    if args.fp16_model_storage:
        state = {
            name: value.half()
            if value.is_floating_point() and not name.startswith('ngram_') else value
            for name, value in state.items()
        }
    result['model'] = state
    result['model_storage_dtype'] = 'fp16' if args.fp16_model_storage else 'native'
    result['ngram_source'] = (
        'fixed training split only; context minimum counts '
        + ','.join(f'{order}:{min_counts[order]}' for order in sorted(min_counts))
        + '; count-one hash fractions '
        + ','.join(f'{order}:{singleton_fractions[order]}'
                   for order in sorted(singleton_fractions)))
    torch.save(result, args.output)
    print(f'wrote {args.output} ({args.output.stat().st_size / 2**20:.2f} MiB)')
    for order, table in tables.items():
        print(f'order={order} contexts={len(table["context_keys"])} entries={len(table["tokens"])}')
    for order, table in singleton_tables.items():
        print(f'order={order} singleton_contexts={len(table["tokens"])}')


if __name__ == '__main__':
    main()
