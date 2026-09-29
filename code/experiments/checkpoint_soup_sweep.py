"""Validation sweep along the line between two compatible checkpoints."""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_data, make_model, setup
from evaluate import score


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', required=True, type=Path)
    parser.add_argument('--second', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--alphas', default='0.5,0.7,0.85,1.0,1.1,1.2,1.3,1.4')
    args = parser.parse_args()
    device, _ = setup(args.device, 'fp32', args.threads)
    first = torch.load(args.first, map_location='cpu', weights_only=True)
    second = torch.load(args.second, map_location='cpu', weights_only=True)
    if (first['implementation'] != second['implementation']
            or first['config'] != second['config']):
        raise ValueError('Checkpoints must use the same implementation and config.')
    first_state, second_state = first['model'], second['model']
    model, _ = make_model(first['implementation'], first['config'], device)
    validation = load_data()['validation']
    for alpha in (float(value) for value in args.alphas.split(',')):
        state = {
            key: first_state[key].lerp(second_state[key], alpha)
            if first_state[key].is_floating_point() else second_state[key]
            for key in first_state}
        model.load_state_dict(state)
        result = score(model, *validation, device, 'fp32')
        print(f'alpha={alpha:g}\tbpb={result["bpb"]:.9f}\tseconds={result["seconds"]:.3f}',
              flush=True)


if __name__ == '__main__':
    main()
