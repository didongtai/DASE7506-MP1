"""Sweep causal train-corpus n-gram mixtures on the validation split.

This development script computes only the probability assigned to each true
validation target.  It is used to decide whether a normalized n-gram component
is worth implementing in the submitted model.
"""
import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_data, make_model, setup, windows


VOCAB = 2048


def counted_keys(values):
    """Return sorted unique int64 keys and their float64 counts."""
    keys, counts = np.unique(values, return_counts=True)
    return keys, counts.astype(np.float64)


def lookup(keys, values, queries):
    indices = np.searchsorted(keys, queries)
    present = indices < len(keys)
    present[present] &= keys[indices[present]] == queries[present]
    result = np.zeros(len(queries), dtype=np.float64)
    result[present] = values[indices[present]]
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    device, _ = setup('cpu', 'fp32', args.threads)
    data = load_data()
    train = data['train'][0].numpy().astype(np.int64, copy=False)
    validation_bytes = data['validation'][1]

    unigram_counts = np.bincount(train, minlength=VOCAB).astype(np.float64)
    unigram_p = (unigram_counts + 0.1) / (len(train) + 0.1 * VOCAB)
    bigram_keys, bigram_counts = counted_keys(train[:-1] * VOCAB + train[1:])
    bigram_context_counts = np.bincount(train[:-1], minlength=VOCAB).astype(np.float64)
    train_contexts = train[:-2] * VOCAB + train[1:-1]
    trigram_context_keys, trigram_context_counts = counted_keys(train_contexts)
    trigram_keys, trigram_counts = counted_keys(train_contexts * VOCAB + train[2:])

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    neural_parts, unigram_parts, bigram_parts, trigram_parts = [], [], [], []
    for x, y in windows(data['validation'][0]):
        valid = y != -100
        neural = model.predict_log_probs(x).gather(
            -1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp()[valid].numpy()
        x_np, y_np = x.numpy(), y.numpy()
        rows, positions = np.nonzero(valid.numpy())
        targets = y_np[rows, positions]
        previous = x_np[rows, positions]
        pair_queries = previous * VOCAB + targets
        bigram = lookup(bigram_keys, bigram_counts, pair_queries)
        bigram /= bigram_context_counts[previous]

        # Back off to the normalized bigram distribution when a second context
        # token is unavailable or the context pair was unseen in training.
        trigram = bigram.copy()
        has_two = positions > 0
        contexts = (x_np[rows[has_two], positions[has_two] - 1] * VOCAB
                    + previous[has_two])
        triple_queries = contexts * VOCAB + targets[has_two]
        numerators = lookup(trigram_keys, trigram_counts, triple_queries)
        denominators = lookup(trigram_context_keys, trigram_context_counts, contexts)
        conditional = np.zeros_like(numerators)
        np.divide(numerators, denominators, out=conditional, where=denominators > 0)
        trigram[has_two] = np.where(
            denominators > 0, conditional, bigram[has_two])

        neural_parts.append(neural.astype(np.float64))
        unigram_parts.append(unigram_p[targets])
        bigram_parts.append(bigram)
        trigram_parts.append(trigram)

    components = [np.concatenate(parts) for parts in
                  (neural_parts, unigram_parts, bigram_parts, trigram_parts)]
    neural, unigram, bigram, trigram = components
    print(f'neural\t{(-np.log(neural).sum() / math.log(2) / validation_bytes):.9f}')

    rows = []
    # The remaining probability is always assigned to the neural model.  Each
    # component is normalized, so every candidate is also normalized.
    for unigram_weight in (0.0, 0.01, 0.02, 0.04):
        for bigram_weight in (0.0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20):
            for trigram_weight in (0.08, 0.12, 0.16, 0.20, 0.24, 0.28,
                                   0.32, 0.40, 0.50):
                total = unigram_weight + bigram_weight + trigram_weight
                if total == 0 or total >= 1:
                    continue
                probability = ((1-total) * neural + unigram_weight * unigram
                               + bigram_weight * bigram + trigram_weight * trigram)
                bpb = -np.log(probability).sum() / math.log(2) / validation_bytes
                rows.append((bpb, unigram_weight, bigram_weight, trigram_weight))
    for bpb, unigram_weight, bigram_weight, trigram_weight in sorted(rows)[:30]:
        print(f'u={unigram_weight:.2f}\tb={bigram_weight:.2f}\t'
              f't={trigram_weight:.2f}\tbpb={bpb:.9f}')


if __name__ == '__main__':
    main()
