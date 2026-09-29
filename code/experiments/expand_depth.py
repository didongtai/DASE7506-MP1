"""Expand a student checkpoint with identity-initialized residual blocks."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROTOCOL, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if source['protocol'] != PROTOCOL or source['implementation'] != 'student':
        raise ValueError('Expected a student checkpoint from the current protocol.')
    config = json.loads(args.config.read_text())
    old_depth = int(source['config']['depth'])
    new_depth = int(config['depth'])
    if new_depth <= old_depth:
        raise ValueError('Target depth must be greater than source depth.')
    for name in ('vocab', 'width', 'heads', 'context', 'ffn_hidden'):
        if config.get(name) != source['config'].get(name):
            raise ValueError(f'Expansion requires matching {name}.')

    model, _ = make_model('student', config, torch.device('cpu'))
    target = model.state_dict()
    copied = []
    for name, value in source['model'].items():
        if name in target and target[name].shape == value.shape:
            target[name] = value
            copied.append(name)
    model.load_state_dict(target)

    # A zero output projection makes each new pre-norm residual block an exact
    # identity map while allowing it to begin learning on the first update.
    with torch.no_grad():
        for block in model.blocks[old_depth:]:
            block.proj.weight.zero_()
            block.proj.bias.zero_()
            block.mlp.output.weight.zero_()
            block.mlp.output.bias.zero_()

    result = {
        'protocol': PROTOCOL,
        'implementation': 'student',
        'config': config,
        'model': model.state_dict(),
        'seed': source.get('seed'),
        'train_tokens': int(source.get('train_tokens', 0)),
        'sampling': source.get('sampling', 'random'),
        'expansion': {
            'source': str(args.checkpoint),
            'source_depth': old_depth,
            'target_depth': new_depth,
            'new_blocks': 'zero-output identity initialization',
            'copied_tensors': len(copied),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f'wrote {args.output} ({args.output.stat().st_size / 2**20:.2f} MiB)')


if __name__ == '__main__':
    main()
