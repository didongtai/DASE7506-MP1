"""Assemble training-derived neural/table assets and validation-selected settings."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROTOCOL, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tables', type=Path, required=True)
    p.add_argument('--neural', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--row-index', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists; choose a new checkpoint path.')
    tables = torch.load(args.tables, map_location='cpu', weights_only=True)
    neural = torch.load(args.neural, map_location='cpu', weights_only=True)
    if tables['protocol'] != PROTOCOL or neural['protocol'] != PROTOCOL:
        p.error('Source protocol mismatch.')
    calibration = json.loads(args.calibration.read_text())
    if calibration['split'] != 'validation':
        p.error('Calibration must use validation only.')
    selected = calibration['candidates'][args.row_index]
    result = copy.deepcopy(tables)
    for name, tensor in neural['model'].items():
        if name.startswith('ngram_'):
            continue
        if name not in result['model'] or tensor.shape != result['model'][name].shape:
            p.error(f'Incompatible neural tensor: {name}')
        result['model'][name] = tensor.half() if tensor.is_floating_point() else tensor
    result['implementation'] = 'student_cache'
    for field in ('seed', 'train_tokens', 'sampling'):
        if field in neural:
            result[field] = neural[field]
    for field in ('dropout', 'scaled_residual_init'):
        if field in neural['config']:
            result['config'][field] = neural['config'][field]
    result['config']['temperature'] = selected['temperature']
    result['config']['prefix_cache'] = {
        'theta': selected['theta'], 'previous_bonus': selected['previous_bonus'],
        'suffix_bonus': selected['suffix_bonus'],
        'gate': {name: selected[name] for name in ('slope', 'offset', 'intercept')}}
    weights = selected['weights']
    result['config']['ngram']['neural_weight'] = weights[0]
    for table, weight in zip(result['config']['ngram']['tables'], weights[1:]):
        table['weight'] = weight
    result['assembly_sources'] = {
        'neural': str(args.neural), 'neural_sha256': sha(args.neural),
        'tables': str(args.tables), 'tables_sha256': sha(args.tables),
        'calibration': str(args.calibration), 'calibration_sha256': sha(args.calibration),
        'calibration_row': args.row_index,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    asset_size = args.output.stat().st_size
    if asset_size > 64*2**20:
        raise ValueError(f'Checkpoint exceeds 64 MiB: {asset_size} bytes')
    print(json.dumps({'checkpoint': str(args.output), 'bytes': asset_size,
                      'sha256': sha(args.output), 'settings': selected}, indent=2))


if __name__ == '__main__':
    main()
