"""Validation-only sweep for a causal within-window token cache.

The cache probability for the next token is its empirical frequency among the
most recent input tokens.  This script evaluates mixtures with a checkpoint's
neural probabilities without changing the submitted implementation.
"""
import argparse
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_data, make_model, setup, windows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--split', choices=['validation'], default='validation',
                        help='Development sweeps are restricted to validation.')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    device, _ = setup('cpu', 'fp32', args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    tokens, byte_count = load_data()[args.split]

    cache_lengths = (8, 16, 32, 64, 128, 256)
    lambdas = (0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4)
    nll = {(length, weight): 0.0 for length in cache_lengths for weight in lambdas}
    neural_nll = 0.0

    for x, y in windows(tokens):
        neural_target_p = model.predict_log_probs(x).gather(
            -1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp()
        valid = y != -100
        neural_nll -= neural_target_p[valid].log().double().sum().item()

        for length in cache_lengths:
            counts = torch.zeros(x.shape[0], 2048)
            cache_target_p = torch.zeros_like(neural_target_p)
            for position in range(x.shape[1]):
                counts.scatter_add_(1, x[:, position:position + 1],
                                    torch.ones(x.shape[0], 1))
                if position >= length:
                    counts.scatter_add_(1, x[:, position-length:position-length+1],
                                        -torch.ones(x.shape[0], 1))
                denominator = min(position + 1, length)
                cache_target_p[:, position] = counts.gather(
                    1, y[:, position:position + 1].clamp_min(0)).squeeze(1) / denominator
            for weight in lambdas:
                probability = (1 - weight) * neural_target_p + weight * cache_target_p
                nll[length, weight] -= probability[valid].log().double().sum().item()

    print(f'neural\t{neural_nll / math.log(2) / byte_count:.9f}')
    for (length, weight), total_nll in sorted(nll.items(), key=lambda item: item[1]):
        print(f'cache={length:3d}\tlambda={weight:.2f}\tbpb={total_nll / math.log(2) / byte_count:.9f}')


if __name__ == '__main__':
    main()
