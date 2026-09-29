"""Validation-only removal ablations of a frozen checkpoint; no tuning or saving weights."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_data, make_model, setup, sha
from evaluate import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    device, precision = setup('cpu', 'fp32', 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    data = load_data()['validation']
    results = {'checkpoint_sha256': sha(args.checkpoint), 'split': 'validation',
               'description': 'Same frozen neural weights and temperature; removed components '
                              'are not recalibrated. No test scoring or checkpoint selection.',
               'ablations': {}}
    for name in ('without_prefix_cache', 'neural_only'):
        config = copy.deepcopy(checkpoint['config'])
        config.pop('prefix_cache')
        state = checkpoint['model']
        if name == 'neural_only':
            config.pop('ngram')
            state = {key: value for key, value in state.items() if not key.startswith('ngram_')}
        model, _ = make_model(checkpoint['implementation'], config, device)
        model.load_state_dict(state)
        result = score(model, *data, device, precision)
        result.pop('window_nll_nats')
        results['ablations'][name] = result
        del model
        print(json.dumps({'variant': name, **result}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
