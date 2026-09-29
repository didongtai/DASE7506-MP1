"""Create a reproducible interpolation or extrapolation of two checkpoints."""
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', required=True, type=Path)
    parser.add_argument('--second', required=True, type=Path)
    parser.add_argument('--alpha', required=True, type=float,
                        help='0 selects first, 1 selects second; values above 1 extrapolate.')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    first = torch.load(args.first, map_location='cpu', weights_only=True)
    second = torch.load(args.second, map_location='cpu', weights_only=True)
    if (first['protocol'] != second['protocol']
            or first['implementation'] != second['implementation']
            or first['config'] != second['config']):
        raise ValueError('Checkpoints must use the same protocol, implementation and config.')
    state = {
        key: first['model'][key].lerp(second['model'][key], args.alpha)
        if first['model'][key].is_floating_point() else second['model'][key]
        for key in first['model']}
    result = dict(second)
    result['model'] = state
    result['interpolation'] = {
        'first': str(args.first), 'second': str(args.second), 'alpha': args.alpha}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f'wrote {args.output} ({args.output.stat().st_size / 2**20:.2f} MiB)')


if __name__ == '__main__':
    main()
