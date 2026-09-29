"""Attach compact train-corpus bigram/trigram tables to a neural checkpoint."""
import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROTOCOL, load_data, make_model


VOCAB = 2048


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bigram-weight', type=float, default=0.04)
    parser.add_argument('--trigram-weight', type=float, default=0.40)
    args = parser.parse_args()
    if args.bigram_weight < 0 or args.trigram_weight < 0:
        parser.error('Mixture weights must be nonnegative.')
    neural_weight = 1 - args.bigram_weight - args.trigram_weight
    if neural_weight <= 0:
        parser.error('The n-gram weights must sum to less than one.')

    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if source['protocol'] != PROTOCOL:
        raise ValueError('Checkpoint belongs to a different course protocol.')
    train = load_data()['train'][0].numpy().astype(np.int64, copy=False)

    pair_keys = train[:-1] * VOCAB + train[1:]
    dense_bigram_counts = np.bincount(
        pair_keys, minlength=VOCAB * VOCAB).reshape(VOCAB, VOCAB)
    row_totals = dense_bigram_counts.sum(1, keepdims=True)
    unigram = np.bincount(train, minlength=VOCAB).astype(np.float64)
    unigram /= unigram.sum()
    bigram_probs = np.divide(
        dense_bigram_counts, row_totals,
        out=np.broadcast_to(unigram, dense_bigram_counts.shape).copy(),
        where=row_totals > 0).astype(np.float32)

    context_values = train[:-2] * VOCAB + train[1:-1]
    context_keys, context_counts = np.unique(context_values, return_counts=True)
    triple_values = context_values * VOCAB + train[2:]
    triple_keys, triple_counts = np.unique(triple_values, return_counts=True)
    offsets = np.searchsorted(
        triple_keys, context_keys * VOCAB).astype(np.int32)
    offsets = np.append(offsets, len(triple_keys)).astype(np.int32)
    continuation_tokens = (triple_keys % VOCAB).astype(np.int16)
    continuation_probs = (
        triple_counts / np.repeat(context_counts, np.diff(offsets))).astype(np.float32)

    config = copy.deepcopy(source['config'])
    config['ngram'] = {
        'contexts': int(len(context_keys)),
        'entries': int(len(triple_keys)),
        'neural_weight': neural_weight,
        'bigram_weight': args.bigram_weight,
        'trigram_weight': args.trigram_weight,
    }
    model, _ = make_model(source['implementation'], config, torch.device('cpu'))
    missing, unexpected = model.load_state_dict(source['model'], strict=False)
    expected_missing = {
        'bigram_probs', 'ngram_context_keys', 'ngram_offsets',
        'ngram_tokens', 'ngram_probs'}
    if set(missing) != expected_missing or unexpected:
        raise ValueError(f'Unexpected state mismatch: missing={missing}, unexpected={unexpected}')
    model.bigram_probs.copy_(torch.from_numpy(bigram_probs))
    model.ngram_context_keys.copy_(torch.from_numpy(context_keys.astype(np.int32)))
    model.ngram_offsets.copy_(torch.from_numpy(offsets))
    model.ngram_tokens.copy_(torch.from_numpy(continuation_tokens))
    model.ngram_probs.copy_(torch.from_numpy(continuation_probs))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(source)
    result['config'] = config
    result['model'] = model.state_dict()
    result['ngram_source'] = 'fixed training split only'
    torch.save(result, args.output)
    size_mib = args.output.stat().st_size / 2**20
    print(f'wrote {args.output} ({size_mib:.2f} MiB)')


if __name__ == '__main__':
    main()
