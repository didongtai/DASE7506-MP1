"""Evaluate causal prefix caches on validation only; no inference state is saved."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_data, make_model, setup, windows


def optimize_weight(base, cache, byte_count):
    def objective(weight):
        return float(-np.log(base + weight * (cache - base)).sum()
                     / math.log(2) / byte_count)
    left, right = 0., .5
    for _ in range(32):
        a = left + (right-left)/3
        b = right - (right-left)/3
        if objective(a) < objective(b):
            right = b
        else:
            left = a
    weight = (left+right)/2
    return objective(weight), weight


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    device, _ = setup(args.device, 'fp32', 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    data = load_data()
    params = [(theta, bonus, decay) for theta in (0., 5., 10., 20., 40., 80.)
              for bonus in (0., 2., 4.) for decay in (0., 1/64)]
    parts = [[] for _ in params]
    base_parts = []
    for batch_index, (x, y) in enumerate(windows(data['validation'][0], batch_size=16)):
        x, y = x.to(device), y.to(device)
        features = model.features(x)
        neural = model.head(features).float().softmax(-1)
        if hasattr(model, 'ngram_tables'):
            logp = model._general_ngram_log_probs(x, neural)
        else:
            logp = neural.log()
        base = logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp()
        valid = y != -100
        base_parts.append(base[valid].double().cpu().numpy())
        hidden = F.normalize(features.float(), dim=-1)
        similarity = hidden @ hidden[:, :-1].transpose(1, 2)
        length = x.shape[1]
        queries = torch.arange(length, device=device)[:, None]
        keys = torch.arange(length-1, device=device)[None, :]
        distance = queries-keys
        causal = distance > 0
        same_previous = (x[:, :, None] == x[:, None, :-1]).float()
        target_match = (y[:, :, None] == x[:, None, 1:]).float()
        for destination, (theta, bonus, decay) in zip(parts, params):
            scores = theta * similarity + bonus * same_previous - decay * distance
            scores.masked_fill_(~causal, -1e9)
            attention = scores.softmax(-1)
            target = (attention * target_match).sum(-1)
            target[:, 0] = base[:, 0]
            destination.append(target[valid].double().cpu().numpy())
        if batch_index % 20 == 0:
            print(json.dumps({'validation_batches': batch_index+1}), flush=True)
    base = np.concatenate(base_parts)
    rows = []
    for params_row, part in zip(params, parts):
        cache = np.concatenate(part)
        bpb, weight = optimize_weight(base, cache, data['validation'][1])
        rows.append(dict(theta=params_row[0], previous_bonus=params_row[1],
                         decay=params_row[2], weight=weight, bpb=bpb))
    rows.sort(key=lambda row: row['bpb'])
    result = {'checkpoint': str(args.checkpoint), 'split': 'validation',
              'base_bpb': float(-np.log(base).sum()/math.log(2)/data['validation'][1]),
              'candidates': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
