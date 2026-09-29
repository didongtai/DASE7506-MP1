"""Sweep normalized bigram through 5-gram mixtures on validation data."""
import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_data, make_model, setup, windows


VOCAB = 2048
HASH_MODULUS = 1_000_003


def table(tokens, order):
    size = len(tokens) - order + 1
    contexts = tokens[:size].copy()
    for offset in range(1, order - 1):
        contexts = contexts * VOCAB + tokens[offset:offset + size]
    full = contexts * VOCAB + tokens[order - 1:order - 1 + size]
    context_keys, context_counts = np.unique(contexts, return_counts=True)
    keys, counts = np.unique(full, return_counts=True)
    entry_context_indices = np.searchsorted(context_keys, keys // VOCAB)
    codes = np.clip(np.rint(
        counts / context_counts[entry_context_indices] * 255), 1, 255)
    offsets = np.searchsorted(keys // VOCAB, context_keys)
    code_sums = np.add.reduceat(codes, offsets)
    return (context_keys, context_counts.astype(np.float64),
            keys, counts.astype(np.float64), codes, code_sums)


def lookup(keys, values, queries):
    indices = np.searchsorted(keys, queries)
    present = indices < len(keys)
    present[present] &= keys[indices[present]] == queries[present]
    result = np.zeros(len(queries), dtype=np.float64)
    result[present] = values[indices[present]]
    return result


def conditional_target(order, x, rows, positions, targets, fallback, tables,
                       min_context_count=1, singleton_fraction=0., quantized=False):
    result = fallback.copy()
    usable = positions >= order - 2
    if not usable.any():
        return result
    selected_rows, selected_positions = rows[usable], positions[usable]
    contexts = x[selected_rows, selected_positions - (order - 2)].copy()
    for back in range(order - 3, -1, -1):
        contexts = contexts * VOCAB + x[selected_rows, selected_positions - back]
    context_keys, context_counts, keys, counts, codes, code_sums = tables[order]
    denominators = lookup(context_keys, context_counts, contexts)
    numerators = lookup(
        keys, codes if quantized else counts,
        contexts * VOCAB + targets[usable])
    probability_denominators = lookup(
        context_keys, code_sums if quantized else context_counts, contexts)
    probabilities = np.zeros_like(numerators)
    retained = denominators >= min_context_count
    if singleton_fraction:
        retained |= ((denominators == 1)
                     & (contexts % HASH_MODULUS
                        < int(singleton_fraction * HASH_MODULUS)))
    np.divide(numerators, probability_denominators,
              out=probabilities, where=retained)
    result[usable] = np.where(retained, probabilities, fallback[usable])
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--uint8-probs', action='store_true',
                        help='simulate per-context normalized uint8 table probabilities')
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', args.threads)
    data = load_data()
    train = data['train'][0].numpy().astype(np.int64, copy=False)
    tables = {order: table(train, order) for order in (2, 3, 4, 5)}

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    parts = [[] for _ in range(14)]
    for x, y in windows(data['validation'][0]):
        valid = y != -100
        neural = model.predict_log_probs(x).gather(
            -1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp()[valid].numpy().astype(np.float64)
        x_np, y_np = x.numpy(), y.numpy()
        rows, positions = np.nonzero(valid.numpy())
        targets = y_np[rows, positions]
        previous = x_np[rows, positions]
        ckeys, ccounts, keys, counts, codes, code_sums = tables[2]
        bigram = lookup(
            keys, codes if args.uint8_probs else counts,
            previous * VOCAB + targets)
        bigram /= lookup(
            ckeys, code_sums if args.uint8_probs else ccounts, previous)
        q = args.uint8_probs
        trigram = conditional_target(3, x_np, rows, positions, targets, bigram, tables, quantized=q)
        fourgram = conditional_target(4, x_np, rows, positions, targets, trigram, tables, quantized=q)
        fivegram = conditional_target(5, x_np, rows, positions, targets, fourgram, tables, quantized=q)
        trigram_direct = conditional_target(3, x_np, rows, positions, targets, neural, tables, quantized=q)
        fourgram_direct = conditional_target(4, x_np, rows, positions, targets, neural, tables, quantized=q)
        fivegram_direct = conditional_target(5, x_np, rows, positions, targets, neural, tables, quantized=q)
        fourgram_m2 = conditional_target(
            4, x_np, rows, positions, targets, trigram, tables, 2, quantized=q)
        fivegram_m2 = conditional_target(
            5, x_np, rows, positions, targets, fourgram_m2, tables, 2, quantized=q)
        fivegram_m2_full4 = conditional_target(
            5, x_np, rows, positions, targets, fourgram, tables, 2, quantized=q)
        fivegram_m2_full4_partial = conditional_target(
            5, x_np, rows, positions, targets, fourgram, tables, 2, .21, quantized=q)
        fourgram_m3 = conditional_target(
            4, x_np, rows, positions, targets, trigram, tables, 3, quantized=q)
        fivegram_m3 = conditional_target(
            5, x_np, rows, positions, targets, fourgram_m3, tables, 3, quantized=q)
        values = (neural, bigram, trigram, fourgram, fivegram,
                  trigram_direct, fourgram_direct, fivegram_direct,
                  fourgram_m2, fivegram_m2, fourgram_m3, fivegram_m3,
                  fivegram_m2_full4, fivegram_m2_full4_partial)
        for destination, values in zip(parts, values):
            destination.append(values)
    (neural, bigram, trigram, fourgram, fivegram,
     trigram_direct, fourgram_direct, fivegram_direct,
     fourgram_m2, fivegram_m2, fourgram_m3,
     fivegram_m3, fivegram_m2_full4,
     fivegram_m2_full4_partial) = [np.concatenate(x) for x in parts]
    byte_count = data['validation'][1]
    print(f'neural\t{-np.log(neural).sum() / math.log(2) / byte_count:.9f}')
    for name, component in (('tri-direct', trigram_direct),
                            ('four-direct', fourgram_direct),
                            ('five-direct', fivegram_direct)):
        scores = []
        for weight in np.arange(0.02, 0.62, 0.02):
            probability = (1-weight)*neural + weight*component
            score = -np.log(probability).sum() / math.log(2) / byte_count
            scores.append((score, weight))
        score, weight = min(scores)
        print(f'{name}\tweight={weight:.2f}\tbpb={score:.9f}')
    def search(four_component, five_component):
        candidates = []
        for b in (0.0, 0.01, 0.02):
            for t in (0.04, 0.06, 0.08, 0.10, 0.12):
                for f in (0.04, 0.06, 0.08, 0.10, 0.12):
                    for q in (0.10, 0.12, 0.14, 0.16, 0.18, 0.20):
                        total = b + t + f + q
                        if total == 0 or total >= 0.9:
                            continue
                        probability = ((1-total)*neural + b*bigram + t*trigram
                                       + f*four_component + q*five_component)
                        score = -np.log(probability).sum() / math.log(2) / byte_count
                        candidates.append((score, b, t, f, q))
        return sorted(candidates)

    def continuous_search(four_component, five_component, initial):
        components = torch.from_numpy(np.stack(
            (neural, bigram, trigram, four_component, five_component))).double()
        logits = torch.tensor(np.log(initial), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [logits], max_iter=100, tolerance_grad=1e-12,
            tolerance_change=1e-14, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            weights = logits.softmax(0)
            probability = (weights[:, None] * components).sum(0)
            loss = -probability.log().sum() / math.log(2) / byte_count
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            weights = logits.softmax(0)
            probability = (weights[:, None] * components).sum(0)
            score = -probability.log().sum() / math.log(2) / byte_count
        return score.item(), weights.tolist()

    candidates = search(fourgram, fivegram)
    for score, b, t, f, q in sorted(candidates)[:30]:
        print(f'b={b:.2f}\tt={t:.2f}\tf={f:.2f}\tq={q:.2f}\tbpb={score:.9f}')
    for threshold, four_component, five_component in (
            (2, fourgram_m2, fivegram_m2), (3, fourgram_m3, fivegram_m3)):
        score, b, t, f, q = search(four_component, five_component)[0]
        print(f'pruned>={threshold}\tb={b:.2f}\tt={t:.2f}\tf={f:.2f}\t'
              f'q={q:.2f}\tbpb={score:.9f}')
    score, b, t, f, q = search(fourgram, fivegram_m2_full4)[0]
    print(f'hybrid-4>=1-5>=2\tb={b:.2f}\tt={t:.2f}\tf={f:.2f}\t'
          f'q={q:.2f}\tbpb={score:.9f}')
    score, weights = continuous_search(
        fourgram, fivegram_m2_full4, (1-b-t-f-q, b, t, f, q))
    n, b, t, f, q = weights
    print(f'hybrid-continuous\tn={n:.9f}\tb={b:.9f}\tt={t:.9f}\t'
          f'f={f:.9f}\tq={q:.9f}\tbpb={score:.9f}')
    score, b, t, f, q = search(fourgram, fivegram_m2_full4_partial)[0]
    print(f'hybrid-partial21\tb={b:.2f}\tt={t:.2f}\tf={f:.2f}\t'
          f'q={q:.2f}\tbpb={score:.9f}')
    score, weights = continuous_search(
        fourgram, fivegram_m2_full4_partial, (1-b-t-f-q, b, t, f, q))
    n, b, t, f, q = weights
    print(f'hybrid-partial21-continuous\tn={n:.9f}\tb={b:.9f}\tt={t:.9f}\t'
          f'f={f:.9f}\tq={q:.9f}\tbpb={score:.9f}')
    score, weights = continuous_search(
        fourgram, fivegram, (n, b, t, f, q))
    n, b, t, f, q = weights
    print(f'hybrid-full5-continuous\tn={n:.9f}\tb={b:.9f}\tt={t:.9f}\t'
          f'f={f:.9f}\tq={q:.9f}\tbpb={score:.9f}')


if __name__ == '__main__':
    main()
