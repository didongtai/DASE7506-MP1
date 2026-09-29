"""Audit a frozen checkpoint without changing it or the supplied scorer.

This is evidence for local technical checks, not a claim that submission links,
the report, or the entire historical development process have been verified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_data, make_model, setup, sha, windows
from experiments.augment_higher_ngram import build_table, build_singletons, pack_keys


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device, precision = setup('cpu', 'fp32', 4)
    result = {'environment': {'python': platform.python_version(),
                             'torch': torch.__version__, 'platform': platform.platform(),
                             'threads': torch.get_num_threads(), 'precision': precision}}
    manifest = json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
    fixed = ['common.py', 'evaluate.py', 'model.py', 'configs/baseline.json',
             'tests/test_contract.py', 'data/manifest.json', 'data/tokenizer.json',
             'data/wikitext_train.txt', 'data/wikitext_validation.txt',
             'data/wikitext_test.txt']
    result['release_hashes'] = {
        name: {'sha256': sha(ROOT/name), 'matches': sha(ROOT/name) == manifest['code/'+name]}
        for name in fixed}
    frozen = json.loads((args.checkpoint.parent/'source_hashes.json').read_text())
    result['frozen_hashes'] = {
        name: {'sha256': sha(args.checkpoint if name == 'checkpoint.pt' else ROOT/name),
               'matches': sha(args.checkpoint if name == 'checkpoint.pt' else ROOT/name) == expected}
        for name, expected in frozen.items()}
    with zipfile.ZipFile(args.checkpoint) as archive:
        entries = archive.infolist()
        result['assets'] = {
            'checkpoint_bytes': args.checkpoint.stat().st_size,
            'zip_entry_uncompressed_bytes': sum(entry.file_size for entry in entries),
            'all_entries_stored_without_zip_compression': all(
                entry.compress_type == zipfile.ZIP_STORED for entry in entries),
            'inference_source_bytes': sum((ROOT/name).stat().st_size for name in (
                'student.py', 'student_cache.py')),
            'limit_bytes': 64*2**20}
    assets = result['assets']
    assets['checkpoint_plus_inference_sources_bytes'] = (
        assets['checkpoint_bytes'] + assets['inference_source_bytes'])
    assets['remaining_bytes_including_sources'] = (
        assets['limit_bytes'] - assets['checkpoint_plus_inference_sources_bytes'])
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    result['checkpoint_metadata'] = {name: value for name, value in checkpoint.items()
                                     if name != 'model'}
    sources = checkpoint.get('assembly_sources', {})
    result['assembly'] = {}
    for source in ('neural', 'tables'):
        path = ROOT/sources[source]
        parent = torch.load(path, map_location='cpu', weights_only=True)
        selected_names = [name for name in checkpoint['model']
                          if name.startswith('ngram_') == (source == 'tables')]
        equal = []
        for name in selected_names:
            expected = parent['model'][name]
            if source == 'neural' and expected.is_floating_point():
                expected = expected.half()
            equal.append(torch.equal(checkpoint['model'][name], expected))
        result['assembly'][source] = {
            'matches_recorded_parent_hash': sha(path) == sources[source+'_sha256'],
            'tensors_checked': len(equal), 'all_tensors_match': all(equal)}
        del parent
    calibration = json.loads((ROOT/sources['calibration']).read_text())
    result['assembly']['calibration'] = {
        'split': calibration['split'],
        'matches_recorded_hash': sha(ROOT/sources['calibration']) == sources['calibration_sha256'],
        'row': sources['calibration_row']}
    data = load_data()
    train = data['train'][0].numpy().astype(np.int64, copy=False)
    result['training_table_rebuild'] = {}
    ngram = checkpoint['config']['ngram']
    for setting in ngram['tables']:
        order = setting['order']
        split_singletons = any(item['order'] == order for item in ngram['singleton_tables'])
        table = build_table(train, order, setting['min_count'],
                            0. if split_singletons else setting['singleton_fraction'])
        high, low = pack_keys(table.pop('context_keys'))
        table.update(context_high=high, context_low=low)
        table['probs'] = np.clip(np.rint(table['probs'].astype(np.float32)*255), 1, 255).astype(np.uint8)
        result['training_table_rebuild'][str(order)] = {
            name: torch.equal(checkpoint['model'][f'ngram_{order}_{name}'], torch.from_numpy(value))
            for name, value in table.items()}
        del table
    for setting in ngram['singleton_tables']:
        order = setting['order']
        table = build_singletons(train, order, setting['fraction'])
        result['training_table_rebuild'][f'{order}_singletons'] = {
            name: torch.equal(checkpoint['model'][f'ngram_{order}_singleton_{name}'],
                              torch.from_numpy(value)) for name, value in table.items()}
        del table
    print('Training-derived tables rebuilt and compared.', flush=True)
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    result['runtime_parameter_dtypes'] = sorted({str(p.dtype) for p in model.parameters()})
    result['runtime_floating_buffer_dtypes'] = sorted({str(p.dtype) for p in model.buffers()
                                                      if p.is_floating_point()})
    torch.manual_seed(1729)
    text_inputs, _ = next(windows(data['train'][0], batch_size=2))
    random_inputs = torch.randint(0, 2048, (2, 256))
    inputs = torch.cat([text_inputs, random_inputs, torch.zeros(1, 256, dtype=torch.long)])
    before = model_digest(model)
    checks = {}
    with torch.no_grad():
        original = model.predict_log_probs(inputs)
        checks['shape'] = list(original.shape)
        checks['output_dtype'] = str(original.dtype)
        checks['all_finite'] = torch.isfinite(original).all().item()
        checks['max_log_normalization_error'] = original.logsumexp(-1).abs().max().item()
        checks['future_change_max_errors'] = {}
        for cut in (1, 2, 3, 7, 128, 255):
            changed = inputs.clone()
            changed[:, cut:] = (changed[:, cut:]+19) % 2048
            prediction = model.predict_log_probs(changed)
            checks['future_change_max_errors'][str(cut)] = (
                prediction[:, :cut]-original[:, :cut]).abs().max().item()
        alone = model.predict_log_probs(inputs[:1])
        checks['batch_independence_max_error'] = (alone-original[:1]).abs().max().item()
        reverse = model.predict_log_probs(inputs.flip(0)).flip(0)
        checks['batch_permutation_max_error'] = (reverse-original).abs().max().item()
        again = model.predict_log_probs(inputs)
        checks['repeat_after_other_windows_max_error'] = (again-original).abs().max().item()
        checks['short_sequence_interface'] = {}
        for length in (1, 2, 3, 4, 12):
            try:
                output = model.predict_log_probs(inputs[:1, :length])
                checks['short_sequence_interface'][str(length)] = {
                    'ok': True, 'max_prefix_error': (output-original[:1, :length]).abs().max().item()}
            except Exception as error:
                checks['short_sequence_interface'][str(length)] = {
                    'ok': False, 'exception': type(error).__name__, 'message': str(error)}
    checks['all_parameters_and_buffers_unchanged'] = before == model_digest(model)
    result['frozen_model_contract'] = checks
    result['matched_budget_evidence'] = {}
    for run in ('baseline-s17', 'swiglu-random-s17'):
        metrics = json.loads((ROOT/'runs'/run/'metrics.json').read_text())
        result['matched_budget_evidence'][run] = {
            field: metrics.get(field) for field in (
                'seed', 'parameters', 'train_tokens', 'train_seconds', 'validation', 'config')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'assets': assets, 'contract': checks}, indent=2))


if __name__ == '__main__':
    main()
