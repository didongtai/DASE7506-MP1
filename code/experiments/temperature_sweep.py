"""Validation-only temperature calibration of the neural/n-gram predictor."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_data, make_model, setup, windows


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
    temperatures = (.85, .9, .95, 1., 1.025, 1.05, 1.075, 1.1, 1.15, 1.2)
    parts = [[] for _ in temperatures]
    ngram_parts, coefficient_parts = [], []
    neural_weight = model.ngram_neural_weight
    for x, y in windows(data['validation'][0], batch_size=16):
        x, y = x.to(device), y.to(device)
        valid = y != -100
        targets = y.clamp_min(0).unsqueeze(-1)
        logits = model(x).float()
        neural = logits.softmax(-1)
        neural_target = neural.gather(-1, targets).squeeze(-1)
        # Exact backoff-to-neural coefficient: run mixture once with uniform
        # neural probabilities and once with the real ones.  Unseen bigram
        # contexts are the only case where the total neural coefficient grows.
        seen = getattr(model, 'ngram_2_direct_index_cache')[x] >= 0
        coefficient = torch.where(seen, neural_weight, 1.)
        mixture = model._general_ngram_log_probs(x, neural).gather(
            -1, targets).squeeze(-1).exp()
        ngram_target = (mixture - coefficient * neural_target).clamp_min(0)
        ngram_parts.append(ngram_target[valid].double().cpu().numpy())
        coefficient_parts.append(coefficient[valid].double().cpu().numpy())
        for part, temperature in zip(parts, temperatures):
            target = (logits/temperature).log_softmax(-1).gather(
                -1, targets).squeeze(-1).exp()
            part.append(target[valid].double().cpu().numpy())
    ngram = np.concatenate(ngram_parts)
    coefficient = np.concatenate(coefficient_parts)
    scores = []
    for temperature, part in zip(temperatures, parts):
        neural = np.concatenate(part)
        mixed = coefficient * neural + ngram
        bpb = float(-np.log(mixed).sum()/math.log(2)/data['validation'][1])
        scores.append({'temperature': temperature, 'bpb': bpb})
    scores.sort(key=lambda row: row['bpb'])
    result = {'split': 'validation', 'checkpoint': str(args.checkpoint),
              'candidates': scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
