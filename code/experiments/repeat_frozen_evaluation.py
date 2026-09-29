"""Serial CPU FP32 reproducibility runs of one frozen predictor; no model selection."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT/'runs/final-candidate-depth8-v5/checkpoint.pt')
    parser.add_argument('--hashes', type=Path,
                        default=ROOT/'configs/cache_v5_source_hashes.json')
    parser.add_argument('--label', default='v5')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidate = args.checkpoint.resolve()
    baseline = ROOT/'runs/baseline-s17/checkpoint.pt'
    reference_path = candidate.with_name('test_cpu_fp32.json')
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
    baseline_reference = json.loads(baseline.with_name('test_cpu_fp32.json').read_text())
    hashes = json.loads(args.hashes.read_text())
    watched = {candidate if name == 'checkpoint.pt' else ROOT/name: value
               for name, value in hashes.items()}
    watched[baseline] = baseline_reference['checkpoint_sha256']
    watched[ROOT/'model.py'] = baseline_reference['implementation_sha256']
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    watched.update({ROOT/'data'/name: value for name, value in manifest['sha256'].items()})

    def verify():
        mismatches = [str(path) for path, expected in watched.items() if sha(path) != expected]
        if mismatches:
            raise RuntimeError('Frozen file changed: '+', '.join(mismatches))

    verify()
    schedule = [('baseline-before', baseline)] + [
        (f'{args.label}-{index}', candidate) for index in (1, 2, 3)] + [('baseline-after', baseline)]
    record = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'environment': {'python': platform.python_version(),
                              'torch': importlib.metadata.version('torch'),
                              'platform': platform.platform(), 'device': 'cpu',
                              'precision': 'fp32', 'threads': 4},
              'schedule': [name for name, _ in schedule],
              'reference_bpb': reference['bpb'] if reference else None,
              'reference_source': str(reference_path) if reference else 'first frozen run in this series',
              'frozen_hashes': {str(path.relative_to(ROOT)): value
                                for path, value in watched.items()}, 'runs': []}
    summary = output/'summary.json'

    def save():
        summary.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')

    save()
    for name, checkpoint in schedule:
        verify()
        command = [sys.executable, str(ROOT/'experiments/measure_evaluation.py'),
                   '--checkpoint', str(checkpoint), '--split', 'test', '--threads', '4',
                   '--output', str(output/f'{name}.json')]
        print(json.dumps({'starting': name}), flush=True)
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                   encoding='utf-8', errors='replace')
        (output/f'{name}.stdout.txt').write_text(completed.stdout, encoding='utf-8')
        (output/f'{name}.stderr.txt').write_text(completed.stderr, encoding='utf-8')
        if completed.returncode:
            record['failure'] = {'run': name, 'returncode': completed.returncode}
            save()
            raise RuntimeError(f'{name} failed; see captured logs')
        score = json.loads((output/f'{name}.json').read_text())
        memory = json.loads((output/f'{name}.resources.json').read_text())
        if checkpoint == candidate and reference is None:
            reference = score
            reference_path = output/f'{name}.json'
            record['reference_bpb'] = score['bpb']
            record['reference_source'] = str(reference_path)
        expected = reference if checkpoint == candidate else baseline_reference
        compared = ('bpb', 'token_ppl', 'nll_nats', 'targets', 'utf8_bytes',
                    'checkpoint_sha256', 'implementation_sha256', 'evaluator_sha256',
                    'tokenizer_sha256', 'protocol', 'split', 'precision')
        row = {'run': name, 'command': command, 'bpb': score['bpb'],
               'seconds': score['seconds'], 'peak_working_set_gib': memory['peak_working_set_gib'],
               'all_score_and_identity_fields_exactly_match_reference': all(
                   score[field] == expected[field] for field in compared),
               'mismatching_fields': [field for field in compared if score[field] != expected[field]]}
        record['runs'].append(row)
        verify()
        save()
        print(json.dumps({key: value for key, value in row.items() if key != 'command'}), flush=True)

    # Compare every per-window loss, not only the aggregate score. Import after timing.
    import numpy as np
    original_windows = np.load(reference_path.with_suffix('.window-nll.npy'))
    rows = [row for row in record['runs'] if row['run'].startswith(args.label+'-')]
    for row in rows:
        actual = np.load(output/f"{row['run']}.window-nll.npy")
        row['per_window_losses_exactly_match_reference'] = bool(np.array_equal(actual, original_windows))
        row['max_absolute_per_window_nll_difference'] = float(np.max(np.abs(actual-original_windows)))
    times = [row['seconds'] for row in rows]
    baseline_times = [row['seconds'] for row in record['runs'] if row['run'].startswith('baseline-')]
    faster_baseline = min(baseline_times)
    for row in rows:
        row['ratio_to_faster_baseline'] = row['seconds']/faster_baseline
        row['passes_five_times_limit_with_faster_baseline'] = row['seconds'] <= 5*faster_baseline
    record['statistics'] = {
        'candidate_seconds_min': min(times), 'candidate_seconds_max': max(times),
        'candidate_seconds_mean': statistics.mean(times),
        'candidate_seconds_median': statistics.median(times),
        'candidate_seconds_sample_stdev': statistics.stdev(times),
        'baseline_seconds': baseline_times,
        'faster_baseline_five_times_seconds': 5*faster_baseline,
        'candidate_median_to_baseline_mean_ratio': statistics.median(times)/statistics.mean(baseline_times),
        'worst_candidate_to_faster_baseline_ratio': max(times)/faster_baseline,
        'largest_candidate_peak_working_set_gib': max(row['peak_working_set_gib'] for row in rows),
        'all_candidate_runs_within_five_times_faster_baseline': all(t <= 5*faster_baseline for t in times),
        'all_candidate_scores_exactly_reproduced': all(
            row['all_score_and_identity_fields_exactly_match_reference'] and
            row['per_window_losses_exactly_match_reference'] for row in rows)}
    verify()
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    record['all_frozen_hashes_verified_before_and_after_each_run'] = True
    record['limitations'] = 'Three pre-specified candidate runs on one Windows host; '
    record['limitations'] += 'all runs retained. Does not certify other hosts or historical development compliance.'
    save()
    print(json.dumps(record['statistics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
