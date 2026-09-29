"""Initialize copy-objective continuation from existing training-only weights."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROTOCOL, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config-output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists; choose a new path.')
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if source['protocol'] != PROTOCOL or 'ngram' in source['config']:
        p.error('Expected a pure neural checkpoint for this protocol.')
    source['implementation'] = 'student_pointer'
    source['config']['prefix_cache'] = dict(theta=10., previous_bonus=2., weight=.06)
    source['config']['copy_objective_weight'] = .75
    source['initialization_source'] = str(args.checkpoint)
    source['initialization_source_sha256'] = sha(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(source, args.output)
    args.config_output.write_text(json.dumps(source['config'], indent=2)+'\n')


if __name__ == '__main__':
    main()
