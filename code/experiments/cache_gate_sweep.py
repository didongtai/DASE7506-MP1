"""Validation selection of small confidence-conditioned cache mixtures."""
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
from student_cache import prefix_attention


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--refine', action='store_true')
    args = p.parse_args()
    device, _ = setup(args.device, 'fp32', 4)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(source['implementation'], source['config'], device)
    model.load_state_dict(source['model'])
    model.eval()
    data = load_data()
    params = [(theta, bonus, suffix) for theta in (8., 12., 16.)
              for bonus in (0., 2.) for suffix in (0., 2., 4.)]
    if args.refine:
        params = [(theta, 0., suffix) for theta in (8., 10., 12., 14., 16.)
                  for suffix in (0., 2.)]
    bases, cache_parts, feature_parts = [], [[] for _ in params], [[] for _ in params]
    for x, y in windows(data['validation'][0], batch_size=16):
        x, y = x.to(device), y.to(device)
        h = model.features(x)
        probabilities = (model.head(h).float()/model.config.get('temperature', 1.)).softmax(-1)
        if hasattr(model, 'ngram_tables'):
            probabilities = model._general_ngram_log_probs(x, probabilities).exp()
        base = probabilities.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        valid = y != -100
        bases.append(base[valid].double().cpu().numpy())
        target_match = (y[:, :, None] == x[:, None, 1:]).float()
        length = x.shape[1]
        queries = torch.arange(length, device=device)[:, None]
        keys = torch.arange(length-1, device=device)[None, :]
        causal = queries > keys
        same = (x[:, :, None] == x[:, None, :-1])
        same2 = torch.zeros_like(same)
        same2[:, 1:, 1:] = same[:, 1:, 1:] & same[:, :-1, :-1]
        hidden = F.normalize(h.float(), dim=-1)
        similarity = hidden @ hidden[:, :-1].transpose(1, 2)
        for params_index, (theta, bonus, suffix) in enumerate(params):
            scores = theta*similarity + bonus*same + suffix*same2
            scores.masked_fill_(~causal, -1e9)
            attention = scores.softmax(-1)
            cache_target = (attention*target_match).sum(-1)
            cache_target[:, 0] = base[:, 0]
            copy_distribution = torch.zeros_like(probabilities).scatter_add_(
                2, x[:, None, 1:].expand(-1, length, -1), attention)
            max_probability = copy_distribution.amax(-1)
            copy_entropy = -(copy_distribution*copy_distribution.clamp_min(1e-30).log()).sum(-1)
            max_similarity = similarity.masked_fill(~causal, -1.).amax(-1)
            features = torch.stack((max_probability, copy_entropy, max_similarity), -1)
            cache_parts[params_index].append(cache_target[valid].double().cpu().numpy())
            feature_parts[params_index].append(features[valid].double().cpu().numpy())
    base = np.concatenate(bases)
    byte_count = data['validation'][1]
    rows = []
    for params_row, caches, features_list in zip(params, cache_parts, feature_parts):
        cache = np.concatenate(caches)
        features = np.concatenate(features_list)
        feature_slopes = ((2, (8., 12., 16., 24.)),) if args.refine else (
            (0, (0., 1., 2., 4.)), (1, (-1., -.5, -.25)), (2, (2., 4., 8.)))
        for feature_index, slopes in feature_slopes:
            for slope in slopes:
                feature = features[:, feature_index]
                offset = (0.4, 2., .65)[feature_index]
                def objective(intercept):
                    weight = 1/(1+np.exp(-(intercept+slope*(feature-offset))))
                    mixed = base + weight*(cache-base)
                    return float(-np.log(mixed).sum()/math.log(2)/byte_count)
                left, right = -8., -1.
                for _ in range(24):
                    a, b = left+(right-left)/3, right-(right-left)/3
                    if objective(a) < objective(b):
                        right = b
                    else:
                        left = a
                intercept = (left+right)/2
                rows.append(dict(theta=params_row[0], previous_bonus=params_row[1],
                                 suffix_bonus=params_row[2], feature=feature_index,
                                 slope=slope, offset=offset, intercept=intercept,
                                 bpb=objective(intercept)))
    rows.sort(key=lambda row: row['bpb'])
    result = {'split': 'validation', 'checkpoint': str(args.checkpoint),
              'base_bpb': float(-np.log(base).sum()/math.log(2)/byte_count),
              'candidates': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({**result, 'candidates': rows[:15]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
