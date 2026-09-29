"""Validation sweep for count-adaptive recursive n-gram interpolation."""
import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_data, make_model, setup, windows


VOCAB = 2048
MIN_COUNTS = {2: 1, 3: 1, 4: 2, 5: 2}


def table(tokens, order):
    size = len(tokens) - order + 1
    contexts = tokens[:size].copy()
    for offset in range(1, order - 1):
        contexts = contexts * VOCAB + tokens[offset:offset + size]
    full = contexts * VOCAB + tokens[order - 1:order - 1 + size]
    context_keys, context_counts = np.unique(contexts, return_counts=True)
    keys, counts = np.unique(full, return_counts=True)
    return context_keys, context_counts.astype(np.float64), keys, counts.astype(np.float64)


def lookup(keys, values, queries):
    indices = np.searchsorted(keys, queries)
    present = indices < len(keys)
    present[present] &= keys[indices[present]] == queries[present]
    result = np.zeros(len(queries), dtype=np.float64)
    result[present] = values[indices[present]]
    return result


def raw_component(order, x, rows, positions, targets, tables):
    result = np.zeros(len(targets), dtype=np.float64)
    context_count = np.zeros(len(targets), dtype=np.float64)
    usable = positions >= order - 2
    if not usable.any():
        return result, context_count
    selected_rows, selected_positions = rows[usable], positions[usable]
    contexts = x[selected_rows, selected_positions - (order - 2)].copy()
    for back in range(order - 3, -1, -1):
        contexts = contexts * VOCAB + x[selected_rows, selected_positions - back]
    context_keys, context_counts, keys, counts = tables[order]
    denominators = lookup(context_keys, context_counts, contexts)
    retained = denominators >= MIN_COUNTS[order]
    numerators = lookup(keys, counts, contexts * VOCAB + targets[usable])
    conditional = np.zeros_like(numerators)
    np.divide(numerators, denominators, out=conditional, where=retained)
    result[usable] = conditional
    context_count[usable] = np.where(retained, denominators, 0.)
    return result, context_count


def score_candidate(neural, probabilities, counts, taus, global_weight, byte_count):
    recursive = neural.copy()
    for order in (2, 3, 4, 5):
        confidence = counts[order] / (counts[order] + taus[order])
        recursive += confidence * (probabilities[order] - recursive)
    mixed = neural + global_weight * (recursive - neural)
    return -np.log(mixed).sum() / math.log(2) / byte_count


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--trials', type=int, default=600)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', args.threads)
    data = load_data()
    train = data['train'][0].numpy().astype(np.int64, copy=False)
    tables = {order: table(train, order) for order in (2, 3, 4, 5)}

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    neural_parts = []
    probability_parts = {order: [] for order in tables}
    count_parts = {order: [] for order in tables}
    for x, y in windows(data['validation'][0]):
        valid = y != -100
        neural = model.predict_log_probs(x).gather(
            -1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp()[valid]
        neural_parts.append(neural.numpy().astype(np.float64))
        x_np, y_np = x.numpy(), y.numpy()
        rows, positions = np.nonzero(valid.numpy())
        targets = y_np[rows, positions]
        for order in tables:
            probability, count = raw_component(
                order, x_np, rows, positions, targets, tables)
            probability_parts[order].append(probability)
            count_parts[order].append(count)

    neural = np.concatenate(neural_parts)
    probabilities = {order: np.concatenate(parts)
                     for order, parts in probability_parts.items()}
    counts = {order: np.concatenate(parts) for order, parts in count_parts.items()}
    byte_count = data['validation'][1]
    neural_bpb = -np.log(neural).sum() / math.log(2) / byte_count
    print(f'neural\t{neural_bpb:.9f}')

    tau_grid = np.asarray((0.25, 0.5, 1., 2., 4., 8., 16., 32., 64.,
                           128., 256., 512.))
    weight_grid = np.asarray((0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0))
    rng = np.random.default_rng(7506)
    candidates = []
    # Include several structured starting points before deterministic random search.
    structured = [
        ({2: 128., 3: 16., 4: 4., 5: 4.}, weight)
        for weight in weight_grid]
    for taus, weight in structured:
        candidates.append((score_candidate(
            neural, probabilities, counts, taus, weight, byte_count), taus, weight))
    for _ in range(args.trials):
        taus = {order: float(rng.choice(tau_grid)) for order in tables}
        weight = float(rng.choice(weight_grid))
        score = score_candidate(
            neural, probabilities, counts, taus, weight, byte_count)
        candidates.append((score, taus, weight))

    best_score, best_taus, best_weight = min(candidates, key=lambda item: item[0])
    # Coordinate refinement over the full grids.
    for _ in range(3):
        for order in tables:
            local = []
            for tau in tau_grid:
                taus = dict(best_taus)
                taus[order] = float(tau)
                local.append((score_candidate(
                    neural, probabilities, counts, taus, best_weight, byte_count), taus))
            best_score, best_taus = min(local, key=lambda item: item[0])
        local = [(score_candidate(
            neural, probabilities, counts, best_taus, float(weight), byte_count),
                  float(weight)) for weight in weight_grid]
        best_score, best_weight = min(local, key=lambda item: item[0])

    print(f'adaptive\tbpb={best_score:.9f}\tglobal={best_weight:.2f}\t'
          + '\t'.join(f'tau{order}={best_taus[order]:g}' for order in tables))
    for score, taus, weight in sorted(candidates, key=lambda item: item[0])[:20]:
        print(f'candidate\tbpb={score:.9f}\tglobal={weight:.2f}\t'
              + '\t'.join(f'tau{order}={taus[order]:g}' for order in tables))

    # Fit a small count-conditioned softmax gate.  The neural logit is fixed to
    # zero; each n-gram order has one bias and one log-count slope.
    target_probabilities = torch.from_numpy(np.column_stack(
        [neural] + [probabilities[order] for order in tables])).float()
    log_counts = torch.from_numpy(np.column_stack(
        [np.zeros_like(neural)] + [np.log1p(counts[order]) for order in tables])).float()
    present = torch.from_numpy(np.column_stack(
        [np.ones_like(neural, dtype=bool)] + [counts[order] > 0 for order in tables]))
    initial_weights = np.asarray((0.75, 0.01, 0.06, 0.06, 0.12))
    biases = torch.nn.Parameter(torch.from_numpy(
        np.log(initial_weights[1:] / initial_weights[0])).float())
    slopes = torch.nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.LBFGS(
        [biases, slopes], lr=.5, max_iter=120,
        tolerance_grad=1e-9, tolerance_change=1e-12,
        line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        logits = torch.cat((
            torch.zeros(len(neural), 1),
            biases.unsqueeze(0) + slopes.unsqueeze(0) * log_counts[:, 1:]), dim=1)
        logits = logits.masked_fill(~present, -1e9)
        weights = logits.softmax(1)
        mixture = (weights * target_probabilities).sum(1)
        loss = -mixture.log().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        logits = torch.cat((
            torch.zeros(len(neural), 1),
            biases.unsqueeze(0) + slopes.unsqueeze(0) * log_counts[:, 1:]), dim=1)
        logits.masked_fill_(~present, -1e9)
        mixture = (logits.softmax(1) * target_probabilities).sum(1)
        gate_bpb = -mixture.log().double().sum().item() / math.log(2) / byte_count
    print(f'count-gate\tbpb={gate_bpb:.9f}\t'
          f'biases={biases.detach().tolist()}\tslopes={slopes.detach().tolist()}')


if __name__ == '__main__':
    main()
