"""Validation-only screen of longer training-derived n-gram components."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_data, make_model, setup, windows
from experiments.prefix_cache_sweep import optimize_weight

HASH_BASE = np.int64(1000003)
HASH_MASK = np.int64(0x7fffffffffffffff)
PAIR_DTYPE = np.dtype([('context', '<i8'), ('target', '<i2')])


def context_hash(tokens, length):
    size = len(tokens)-length
    values = np.zeros(size, dtype=np.int64)
    for offset in range(length):
        values = (values*HASH_BASE + tokens[offset:offset+size]) & HASH_MASK
    return values


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device, _ = setup(args.device, 'fp32', 4)
    data = load_data()
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(source['implementation'], source['config'], device)
    model.load_state_dict(source['model'])
    model.eval()
    base_parts, xs, ys, masks = [], [], [], []
    for x, y in windows(data['validation'][0], batch_size=16):
        valid = y != -100
        base = model.predict_log_probs(x.to(device)).gather(
            -1, y.clamp_min(0).to(device).unsqueeze(-1)).squeeze(-1).exp().cpu()
        base_parts.append(base[valid].double().numpy())
        xs.append(x.numpy())
        ys.append(y.numpy())
        masks.append(valid.numpy())
    x, y, mask = np.concatenate(xs), np.concatenate(ys), np.concatenate(masks)
    base = np.concatenate(base_parts)
    train = data['train'][0].numpy().astype(np.int64)
    rows, positions = np.nonzero(mask)
    targets = y[rows, positions]
    results = []
    for order in (6, 7, 8, 9):
        contexts = context_hash(train, order-1)
        pairs = np.empty(len(contexts), dtype=PAIR_DTYPE)
        pairs['context'] = contexts
        pairs['target'] = train[order-1:]
        full, counts = np.unique(pairs, return_counts=True)
        keys, context_counts = np.unique(contexts, return_counts=True)
        entry_context = np.searchsorted(keys, full['context'])
        codes = np.clip(np.rint(counts/context_counts[entry_context]*255), 1, 255)
        offsets = np.searchsorted(full['context'], keys)
        sums = np.add.reduceat(codes, offsets)
        usable = positions >= order-2
        selected_rows, selected_positions = rows[usable], positions[usable]
        query_context = np.zeros(usable.sum(), dtype=np.int64)
        for back in range(order-2, -1, -1):
            query_context = (query_context*HASH_BASE+x[selected_rows, selected_positions-back]) & HASH_MASK
        query_pairs = np.empty(len(query_context), dtype=PAIR_DTYPE)
        query_pairs['context'], query_pairs['target'] = query_context, targets[usable]
        ci = np.searchsorted(keys, query_context)
        safe_ci = ci.clip(max=len(keys)-1)
        present = (ci < len(keys)) & (keys[safe_ci] == query_context)
        pi = np.searchsorted(full, query_pairs)
        safe_pi = pi.clip(max=len(full)-1)
        target_present = (pi < len(full)) & (full[safe_pi] == query_pairs)
        conditional = np.where(target_present, codes[safe_pi]/sums[safe_ci], 0.)
        for minimum in (1, 2, 3, 4):
            retained = present & (context_counts[safe_ci] >= minimum)
            component = base.copy()
            component[usable] = np.where(retained, conditional, base[usable])
            score, weight = optimize_weight(base, component, data['validation'][1])
            ncontexts = int((context_counts >= minimum).sum())
            nentries = int((context_counts[entry_context] >= minimum).sum())
            result = dict(order=order, min_count=minimum, weight=weight, bpb=score,
                          contexts=ncontexts, entries=nentries,
                          asset_mib=(12*ncontexts+3*nentries+4)/2**20)
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'split': 'validation', 'candidates': sorted(
        results, key=lambda row: row['bpb'])}, indent=2)+'\n')


if __name__ == '__main__':
    main()
