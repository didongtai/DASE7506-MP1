"""Copy a self-trained checkpoint into the fast implementation, optionally pruning final blocks."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path.')
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if not 1 <= args.depth <= source['config']['depth']:
        parser.error('Depth must be between one and source depth.')
    result = copy.deepcopy(source)
    result['implementation'] = 'student_fast'
    result['config']['depth'] = args.depth
    result['model'] = {name: tensor for name, tensor in result['model'].items()
                       if not name.startswith('blocks.') or int(name.split('.')[1]) < args.depth}
    result['speed_conversion'] = {
        'source': str(args.checkpoint), 'source_sha256': sha(args.checkpoint),
        'source_depth': source['config']['depth'], 'retained_depth': args.depth,
        'method': 'retain first N blocks; reuse FP32 intermediate storage, reconstruct '
                  'dense bigram lookup from training CSR, mix probabilities without log/exp round trip',
        'selection_split': 'validation', 'additional_training_targets': 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(json.dumps({'checkpoint': str(args.output), 'bytes': args.output.stat().st_size,
                      'sha256': sha(args.output), 'depth': args.depth}, indent=2))


if __name__ == '__main__':
    main()
