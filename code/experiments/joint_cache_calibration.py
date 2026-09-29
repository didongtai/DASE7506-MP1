"""Validation calibration of neural temperature, mixture weights and cache gate."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_data, make_model, setup, windows
from student_cache import prefix_attention


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--temperatures', default='1,1.025,1.05,1.075,1.1,1.125')
    args = p.parse_args()
    device, _ = setup(args.device, 'fp32', 4)
    data = load_data()
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(source['implementation'], source['config'], device)
    model.load_state_dict(source['model'])
    model.eval()
    temperatures = tuple(float(value) for value in args.temperatures.split(','))
    parts = {temperature: [] for temperature in temperatures}
    table_parts = [[] for _ in range(4)]
    copies, confidences, absent_parts = [], [], []
    original_weights = list(model.ngram_tables)
    original_neural = model.ngram_neural_weight
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(windows(data['validation'][0], batch_size=16)):
            x, y = x.to(device), y.to(device)
            valid = y != -100
            targets = y.clamp_min(0).unsqueeze(-1)
            features = model.features(x)
            logits = model.head(features).float()
            neural = logits.softmax(-1)
            for temperature in temperatures:
                probability = (logits/temperature).log_softmax(-1).gather(
                    -1, targets).squeeze(-1).exp()
                parts[temperature].append(probability[valid].double().cpu().numpy())
            for index, (desired_order, _) in enumerate(original_weights):
                model.ngram_neural_weight = 0.
                model.ngram_tables = [(order, float(order == desired_order))
                                      for order, _ in original_weights]
                component = model._general_ngram_log_probs(x, neural.clone()).gather(
                    -1, targets).squeeze(-1).exp()
                table_parts[index].append(component[valid].double().cpu().numpy())
            attention, confidence = prefix_attention(
                features, x, 10., 0., suffix_bonus=2., return_confidence=True)
            copy = (attention * (y[:, :, None] == x[:, None, 1:])).sum(-1)
            copy[:, 0] = 0
            confidence[:, 0] = -100
            copies.append(copy[valid].double().cpu().numpy())
            confidences.append(confidence[valid].double().cpu().numpy())
            absent = getattr(model, 'ngram_2_direct_index_cache')[x] < 0
            absent_parts.append(absent[valid].cpu().numpy())
            if batch_index % 30 == 0:
                print(json.dumps({'collected_batches': batch_index+1}), flush=True)
    model.ngram_neural_weight = original_neural
    model.ngram_tables = original_weights
    del model
    table_probabilities = np.stack([np.concatenate(part) for part in table_parts])
    copy = torch.from_numpy(np.concatenate(copies)).double()
    confidence = torch.from_numpy(np.concatenate(confidences)).double()
    absent = np.concatenate(absent_parts)
    denominator = math.log(2)*data['validation'][1]
    results = []
    for temperature in temperatures:
        neural = np.concatenate(parts[temperature])
        components = np.vstack((neural, table_probabilities))
        components[:, absent] = neural[absent]
        probabilities = torch.from_numpy(components).double()
        for slope in (4., 6., 8., 10.):
            logits = torch.tensor(np.log([original_neural]+[w for _, w in original_weights]),
                                  dtype=torch.float64, requires_grad=True)
            intercept = torch.tensor(-3.2, dtype=torch.float64, requires_grad=True)
            optimizer = torch.optim.LBFGS([logits, intercept], max_iter=65,
                tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                weights = logits.softmax(0)
                base = (weights[:, None]*probabilities).sum(0)
                gate = torch.sigmoid(intercept+slope*(confidence-.65))
                mixed = (1-gate)*base + gate*copy
                loss = -mixed.log().sum()/denominator
                loss.backward()
                return loss
            optimizer.step(closure)
            row = dict(temperature=temperature, theta=10., previous_bonus=0.,
                       suffix_bonus=2., slope=slope, offset=.65,
                       intercept=intercept.item(), weights=logits.softmax(0).detach().tolist(),
                       bpb=closure().item())
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'checkpoint': str(args.checkpoint),
        'split': 'validation', 'candidates': sorted(results, key=lambda row:row['bpb'])}, indent=2)+'\n')


if __name__ == '__main__':
    main()
