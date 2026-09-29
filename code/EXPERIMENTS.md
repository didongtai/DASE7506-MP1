# MP1 experiment record

The recorded v5 checkpoint and mixture-selection scripts use the validation split.
The v5 method was frozen before its reported test evaluation. Earlier candidate
versions were also evaluated on the public test split before later development;
those evaluations must be disclosed, and these records cannot certify that earlier
test feedback had no influence on later decisions. All n-gram statistics and table
entries come only from the fixed training split, independently rebuilt and checked
in the [compliance audit](COMPLIANCE.md).

## Final method

The submitted predictor combines:

1. A 6,913,200-parameter, eight-block causal Transformer with width 256, eight heads,
   SwiGLU feed-forward layers, tied token/output embeddings, dropout 0.1 and scaled
   residual initialization.
2. Training-only 2- through 5-gram conditional distributions. The bigram is stored
   as sparse CSR, all trigram and 4-gram contexts are retained, and 5-gram contexts
   seen at least twice are retained in CSR form. Every count-one 5-gram context is
   stored separately as an exact context/next-token pair. Missing contexts back off
   one order.
3. A causal neural prefix cache. Query hidden state h[t] compares with earlier
   states h[j], j<t; the copied value is the already observed input token x[j+1].
   Thus the cache never uses a future token. It is computed afresh for every window.

The frozen mixture weights are 0.8696562237894708 neural, 0.010260656589039394
bigram, 0.01882805235045812 trigram, 0.028403331348295206 4-gram and
0.0728517359227364 5-gram. Neural softmax temperature is 1.15. Neural tensors
use FP16 checkpoint storage; loading restores them to the model's FP32 evaluation
dtype. Sparse table probabilities use per-context normalized uint8 storage, and
context keys use a lossless high-16/low-32 representation. These encodings make all
training singleton 5-grams fit under the asset limit. The predictor never reads
validation or test targets and carries no state between windows.

The cache attention score is 10 times cosine similarity plus 2 when the last two
input tokens match the key's corresponding two-token suffix. Its interpolation
weight is sigmoid(-3.1730052464846037 + 8 * (maximum past cosine similarity - 0.65));
the weight is zero at the first position. All these scalar settings were selected
on validation. Training-derived neural tensors and table entries use training only.
Inference requires both `student_cache.py` and its dependency `student.py`.

## Selected training ancestry

Every training stage uses batch size 32, context 256, random window sampling and a
cosine schedule ending at 10% of the listed peak LR. CPU stages use FP32. GPU stages
use BF16 on an NVIDIA GeForce RTX 5060 Laptop GPU and 50 warmup steps in the table
below; the later copy-objective stages use 100 warmup steps.

| Stage | Parent | Seed | Peak LR | Added targets | Validation BPB |
|---|---|---:|---:|---:|---:|
| CPU stage 1 | random initialization | 17 | 1e-3 | 9,830,400 | 1.869682568 |
| CPU stage 2 | CPU stage 1 | 18 | 3e-4 | 9,830,400 | 1.776468800 |
| CPU stage 3 | CPU stage 2 | 19 | 2e-4 | 9,830,400 | 1.739936057 |
| Extrapolated start | CPU stages 2/3, alpha 1.44 | - | - | 0 | 1.734692199 |
| GPU stage 1 | extrapolated start | 31 | 4e-3 | 9,830,400 | 1.675098205 |
| GPU stage 2 | GPU stage 1 | 32 | 3e-3 | 9,830,400 | 1.617043833 |
| GPU stage 3 | GPU stage 2 | 33 | 1e-3 | 9,830,400 | 1.591281425 |
| GPU stage 4 | GPU stage 3 | 34 | 5e-4 | 9,830,400 | 1.582160635 |
| GPU stage 5 | GPU stage 4 | 35 | 2.5e-4 | 9,830,400 | 1.576585310 |
| GPU stage 6 | GPU stage 5 | 36 | 1.25e-4 | 9,830,400 | 1.574102983 |
| GPU stage 7a | GPU stage 6 | 37 | 6.25e-5 | 9,830,400 | 1.572445538 |
| GPU stage 7b | GPU stage 6 | 37 | 1.25e-4 | 9,830,400 | 1.572422364 |
| GPU stage 7 average | 50% stage 7a + 50% stage 7b | - | - | 0 | 1.572313431 |
| GPU stage 8a | stage 7 average | 38 | 6.25e-5 | 9,830,400 | 1.571680447 |
| GPU stage 8b | stage 7 average | 38 | 1.25e-4 | 9,830,400 | 1.572013377 |
| Six-layer neural | 75% stage 8a + 25% stage 8b | - | - | 0 | 1.571672798 |
| Depth expansion | six to eight blocks; new blocks are identity maps | - | - | 0 | 1.571672839 |
| New-block training | depth expansion; only final two blocks trainable | 42 | 3e-3 | 9,830,400 | 1.568885112 |
| Joint fine-tune 1 | new-block training; all blocks trainable | 43 | 1.25e-4 | 9,830,400 | 1.567002406 |
| Joint fine-tune 2a | joint fine-tune 1; all blocks trainable | 44 | 6.25e-5 | 9,830,400 | 1.565666199 |
| Joint fine-tune 2b | joint fine-tune 1; all blocks trainable | 44 | 1.25e-4 | 9,830,400 | 1.566029694 |
| Joint fine-tune 2 average | 75% 2a + 25% 2b | - | - | 0 | 1.565654342 |
| Joint fine-tune 3a | joint fine-tune 2 average | 45 | 3.125e-5 | 9,830,400 | 1.565059393 |
| Joint fine-tune 3b | joint fine-tune 2 average | 45 | 6.25e-5 | 9,830,400 | 1.565211166 |
| Pre-copy neural | 75% 3a + 25% 3b | - | - | 0 | **1.565044689** |

The pre-copy checkpoint's path counter is 147,456,000 targets. Accounting for all
19 distinct training runs in its ancestry, including both branches at four averaging
stages, requires **186,777,600** processed targets. The path counter does not include
the additional 39,321,600 targets used by the other averaging branches.

Two additional 6,000-step stages train the same backbone using 75% copy-mixture NLL
and 25% neural-only NLL. The training cache uses cosine scale 10, previous-token
match bonus 2 and fixed copy weight 0.06. No n-gram tables enter this training loss.

| Copy stage | Seed | Peak LR | Added targets | Fixed-cache validation BPB |
|---|---:|---:|---:|---:|
| 1, from pre-copy neural | 47 | 1.5e-4 | 49,152,000 | 1.537151166 |
| 2, from copy stage 1 | 48 | 1e-4 | 49,152,000 | 1.537032212 |

The final checkpoint's path counter is 245,760,000 targets. Its complete selected
ancestry requires **285,081,600 processed targets across 21 training runs**, with
8,993.12 summed training seconds across CPU and GPU. A
rejected dropout-0.2 continuation (seed 46, LR 3e-4) processed another 49,152,000
targets and reached neural-only validation BPB 1.570180533. These three new runs
took 859.91 summed training seconds. The complete recorded GPU/depth search now
contains 55 runs, 648,970,240 run targets and 3,163.16 summed training seconds.
Rejected runs and validation-only sweep JSONs remain under `runs/`.

Across all 70 currently retained `metrics.json` files, including CPU trials,
baselines and benchmarks, the recorded total is **719,667,200 processed targets**
and **12,717.42 summed training seconds**. These totals exclude data preparation,
validation/calibration and any unlogged attempts; they are not elapsed wall time.
The audit exports per-run seeds, ancestry and costs in [audit/v5/run-costs.json](audit/v5/run-costs.json).

## Validation experiments and ablations

| Predictor | Parameters / assets | Validation BPB |
|---|---:|---:|
| Supplied GELU baseline | 1,088,256 parameters | 2.071083507 |
| Parameter-matched SwiGLU | 1,088,424 parameters | 2.010851748 |
| Large Transformer after CPU stage 3 | 5,332,484 parameters | 1.739936057 |
| Final six-layer neural | 5,332,484 parameters | 1.571672798 |
| Pre-copy eight-layer neural | 6,913,200 parameters | 1.565044689 |
| Coarse-grid n-gram mixture | fixed weights; 5-gram count >=2 | 1.531214096 |
| Continuous-weight n-gram mixture | 5-gram count >=2 | 1.528877548 |
| FP16 mixture + 21% count-one 5-grams | 63.754 MiB | 1.528706054 |
| v4: uint8 mixture + all count-one 5-grams | 63.927 MiB | 1.527008407 |
| v4 plus temperature 1.1 | scalar calibration | 1.521239582 |
| Temperature plus fixed-weight neural prefix cache | no extra trained parameters | 1.502778394 |
| Temperature plus similarity-conditioned prefix cache | pre-copy backbone | 1.489913458 |
| Copy stage 1 plus calibrated tables/cache | same 6,913,200 parameters | 1.483809426 |
| v5: copy stage 2 plus calibrated tables/cache | 63.928 MiB | **1.482691946** |

The supplied baseline and parameter-matched SwiGLU control each processed
**9,830,400 training targets**, with seed 17 and random sampling. Their validation
BPB is 2.071083507 and 2.010851748. This controls the small-model architectural
comparison; it does not attribute the entire v5 improvement to SwiGLU at equal cost.

The audit additionally removes components from the **same frozen v5 weights**,
retaining temperature 1.15 and without recalibration or further training:

| Frozen-weight validation ablation | BPB |
|---|---:|
| Neural only | 1.545126083 |
| Neural plus training n-grams; prefix cache removed | 1.517147855 |
| Complete frozen v5 | 1.482691946 |

The conditional gain from the prefix cache is 0.034455909 BPB. These are removal
ablations, not separately optimized models or a control for the copy-training loss.
Raw results: [audit/v5/validation-ablations.json](audit/v5/validation-ablations.json).

Additional findings:

- Coverage-balanced non-overlapping sampling changed small-model validation BPB by
  less than 0.0002.
- A causal within-window repetition cache improved the small SwiGLU model by about
  0.004 BPB and was excluded.
- Count-adaptive n-gram interpolation did not beat the fixed mixture once storage and
  implementation complexity were considered.
- Sparse bigram CSR saved enough asset space to retain all 4-gram contexts. This
  improved the final validation mixture by about 0.00293 BPB over pruning both the
  4-gram and 5-gram tables at count two.
- Continuous optimization of the five mixture weights improved validation BPB by
  about 0.00234 over the coarse grid. Filling the remaining asset budget with a
  training-only hash sample of singleton 5-grams improved another 0.00017 BPB.
- Lossless 48-bit context-key packing and per-context normalized uint8 probabilities
  made room for all 2,018,107 count-one 5-gram contexts. Jointly reoptimizing the
  weights improved validation by another 0.00170 BPB over the 21% singleton model.
- A uniform token-frequency cache gave much less benefit than the neural prefix
  cache. Similarity-conditioned interpolation further improved the neural cache.
- A validation screen of 6- through 9-grams found only small gains at practical
  pruning thresholds; the additional 1-5 MiB did not justify displacing current
  inference assets, so these components were excluded.

## Frozen results and resource checks

### Current timing-safe v6 candidate

The earlier v5 eight-block candidate was retired because its repeated CPU timing
exceeded the 5x limit. The current candidate is
`runs/final-candidate-depth6-v6/checkpoint.pt`, using `student_fast` with six
Transformer blocks. Its validation/test BPB is **1.502201289/1.518913624**.
No test result was used to select this candidate; the depth and inference changes
were selected using validation measurements.

Three pre-specified complete test runs reproduced exactly BPB 1.5189136237091485
and every per-window loss. Times were 26.4081, 29.4858 and 34.7009 seconds; the
bracketing baselines were 9.0511 and 8.8254 seconds. The worst ratio to the faster
baseline was 3.93196x. Peak working set was at most 2.20684 GiB, and checkpoint
plus required inference sources was 60.945 MiB. Full evidence:
[audit/v6/frozen-test-repeat/RESULTS.md](audit/v6/frozen-test-repeat/RESULTS.md).

Checkpoint: `runs/final-candidate-depth8-v5/checkpoint.pt`

| Split | BPB | Token perplexity | CPU FP32 scoring time (s) |
|---|---:|---:|---:|
| Validation | **1.482691946** | 22.9396 | 29.85 |
| Test | **1.499412861** | 22.9756 | 43.01 |

The reproduced supplied baseline test result is 2.101260271 BPB in 8.75 seconds. The
frozen predictor improves test BPB by 0.601847410 (28.64%) and takes 4.91x the same
machine baseline time. Compared with v4's 1.551456326 test BPB, this is a 0.052043465
improvement (3.35%). This v5 test was run only after the method and settings froze.

Initial resource measurements for the frozen predictor (later timing repeats failed):

- Checkpoint/inference asset: 67,033,181 bytes (63.928 MiB), 75,683 bytes below 64 MiB.
- Evaluation process lifetime peak working set: 2,369,155,072 bytes (2.206 GiB),
  including loading, data preparation and complete test scoring; below 4 GiB.
- Complete four-thread test CPU scoring: 43.0067 seconds, below 5x the 8.7504-second
  same-machine baseline (43.7521 seconds). Timing margin is small and depends on load.
- Contract, sampler, prefix-causality and copy-gradient tests: 11/11 passed.

**Later serial reproducibility check: score reproduced; CPU budget failed.**
Three pre-specified unchanged v5 runs produced exactly the same BPB and every
per-window NLL as the original. Scoring times were 34.1322, 34.1866 and 34.2711
seconds; baseline runs before/after were 6.0920 and 6.4525 seconds. Every candidate
run exceeds five times even the slower baseline. Median candidate time divided
by mean baseline time is 5.4505x (worst candidate / faster baseline: 5.6256x).
Largest candidate peak working set was 2.207001 GiB. All runs are retained, with
frozen hashes checked before and after each run. The earlier 4.91x/4.94x observations
do not establish reliable CPU-budget compliance. No weights, inference code,
hyperparameters or evaluator were changed. Full details:
[repeated-evaluation-20260929/RESULTS.md](audit/v5/repeated-evaluation-20260929/RESULTS.md).

Frozen hashes:

```text
checkpoint_sha256     191d188ccd07f6bbbd3862fd0297c3b64b6a475604cef5a1b3e364b87c41c4b1
implementation_sha256 214b6711e62db78ef571ff295b2ce46510893339545b3f9f862da67a82f0f5c4
student_base_sha256   5d480d349cee7c913ce2757ba30cd99daebd39c7769b0ef3fb3d46d281796f91
evaluator_sha256      128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d
tokenizer_sha256      020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e
```

## Reproduction commands

Run from `code/` in the activated environment. Every command requires a new output
directory.

```powershell
python train.py --implementation student --config configs/large.json --device cpu --precision fp32 --threads 4 --seed 17 --steps 1200 --batch-size 32 --learning-rate 0.001 --warmup-steps 100 --run-dir runs/large-1200-s17
python train.py --implementation student --config configs/large.json --resume runs/large-1200-s17/checkpoint.pt --device cpu --precision fp32 --threads 8 --seed 18 --steps 1200 --batch-size 32 --learning-rate 0.0003 --warmup-steps 100 --run-dir runs/large-cont1-s18
python train.py --implementation student --config configs/large.json --resume runs/large-cont1-s18/checkpoint.pt --device cpu --precision fp32 --threads 8 --seed 19 --steps 1200 --batch-size 32 --learning-rate 0.0002 --warmup-steps 100 --run-dir runs/large-cont2-s19

python experiments/interpolate_checkpoint.py --first runs/large-cont1-s18/checkpoint.pt --second runs/large-cont2-s19/checkpoint.pt --alpha 1.44 --output runs/large-extrap-a144/checkpoint.pt

python train.py --implementation student --config configs/large.json --resume runs/large-extrap-a144/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 31 --steps 1200 --batch-size 32 --learning-rate 0.004 --warmup-steps 50 --run-dir runs/gpu-extrap-lr4000-s31
python train.py --implementation student --config configs/large.json --resume runs/gpu-extrap-lr4000-s31/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 32 --steps 1200 --batch-size 32 --learning-rate 0.003 --warmup-steps 50 --run-dir runs/gpu-stage2-lr3000-s32
python train.py --implementation student --config configs/large.json --resume runs/gpu-stage2-lr3000-s32/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 33 --steps 1200 --batch-size 32 --learning-rate 0.001 --warmup-steps 50 --run-dir runs/gpu-stage3-lr1000-s33
python train.py --implementation student --config configs/large.json --resume runs/gpu-stage3-lr1000-s33/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 34 --steps 1200 --batch-size 32 --learning-rate 0.0005 --warmup-steps 50 --run-dir runs/gpu-stage4-lr0500-s34
python train.py --implementation student --config configs/large.json --resume runs/gpu-stage4-lr0500-s34/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 35 --steps 1200 --batch-size 32 --learning-rate 0.00025 --warmup-steps 50 --run-dir runs/gpu-stage5-lr0250-s35
python train.py --implementation student --config configs/large.json --resume runs/gpu-stage5-lr0250-s35/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 36 --steps 1200 --batch-size 32 --learning-rate 0.000125 --warmup-steps 50 --run-dir runs/gpu-stage6-lr0125-s36

python train.py --implementation student --config configs/large.json --resume runs/gpu-stage6-lr0125-s36/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 37 --steps 1200 --batch-size 32 --learning-rate 0.0000625 --warmup-steps 50 --run-dir runs/gpu-stage7-lr00625-s37
python train.py --implementation student --config configs/large.json --resume runs/gpu-stage6-lr0125-s36/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 37 --steps 1200 --batch-size 32 --learning-rate 0.000125 --warmup-steps 50 --run-dir runs/gpu-stage7-lr0125-s37
python experiments/interpolate_checkpoint.py --first runs/gpu-stage7-lr00625-s37/checkpoint.pt --second runs/gpu-stage7-lr0125-s37/checkpoint.pt --alpha 0.5 --output runs/gpu-stage7-avg/checkpoint.pt

python train.py --implementation student --config configs/large.json --resume runs/gpu-stage7-avg/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 38 --steps 1200 --batch-size 32 --learning-rate 0.0000625 --warmup-steps 50 --run-dir runs/gpu-stage8-lr00625-s38
python train.py --implementation student --config configs/large.json --resume runs/gpu-stage7-avg/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 38 --steps 1200 --batch-size 32 --learning-rate 0.000125 --warmup-steps 50 --run-dir runs/gpu-stage8-lr0125-s38
python experiments/interpolate_checkpoint.py --first runs/gpu-stage8-lr00625-s38/checkpoint.pt --second runs/gpu-stage8-lr0125-s38/checkpoint.pt --alpha 0.25 --output runs/gpu-final-pure/checkpoint.pt

python experiments/expand_depth.py --checkpoint runs/gpu-final-pure/checkpoint.pt --config configs/depth8.json --output runs/depth8-expanded/checkpoint.pt
python train.py --implementation student --config configs/depth8.json --resume runs/depth8-expanded/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 42 --steps 1200 --batch-size 32 --learning-rate 0.003 --warmup-steps 50 --train-last-blocks 2 --run-dir runs/depth8-last2-lr3000-s42
python train.py --implementation student --config configs/depth8.json --resume runs/depth8-last2-lr3000-s42/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 43 --steps 1200 --batch-size 32 --learning-rate 0.000125 --warmup-steps 50 --run-dir runs/depth8-joint-lr0125-s43
python train.py --implementation student --config configs/depth8.json --resume runs/depth8-joint-lr0125-s43/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 44 --steps 1200 --batch-size 32 --learning-rate 0.0000625 --warmup-steps 50 --run-dir runs/depth8-joint2-lr00625-s44
python train.py --implementation student --config configs/depth8.json --resume runs/depth8-joint-lr0125-s43/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 44 --steps 1200 --batch-size 32 --learning-rate 0.000125 --warmup-steps 50 --run-dir runs/depth8-joint2-lr0125-s44
python experiments/interpolate_checkpoint.py --first runs/depth8-joint2-lr00625-s44/checkpoint.pt --second runs/depth8-joint2-lr0125-s44/checkpoint.pt --alpha 0.25 --output runs/depth8-final-pure-avg/checkpoint.pt

python train.py --implementation student --config configs/depth8.json --resume runs/depth8-final-pure-avg/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 45 --steps 1200 --batch-size 32 --learning-rate 0.00003125 --warmup-steps 50 --run-dir runs/depth8-joint3-lr003125-s45
python train.py --implementation student --config configs/depth8.json --resume runs/depth8-final-pure-avg/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 45 --steps 1200 --batch-size 32 --learning-rate 0.0000625 --warmup-steps 50 --run-dir runs/depth8-joint3-lr00625-s45
python experiments/interpolate_checkpoint.py --first runs/depth8-joint3-lr003125-s45/checkpoint.pt --second runs/depth8-joint3-lr00625-s45/checkpoint.pt --alpha 0.25 --output runs/depth8-final2-pure-avg/checkpoint.pt

python experiments/augment_higher_ngram.py --checkpoint runs/depth8-final2-pure-avg/checkpoint.pt --output runs/final-candidate-depth8-v4/checkpoint.pt --weights 0.025526641,0.024094672,0.029010147,0.065918324 --min-counts 1,1,1,2 --singleton-fractions 0,0,0,1 --sparse-bigram --fp16-model-storage --packed-context-keys --uint8-probs --split-singleton-order 5
python experiments/prepare_pointer_training.py --checkpoint runs/depth8-final2-pure-avg/checkpoint.pt --output runs/break15-pointer-init/checkpoint.pt --config-output configs/pointer8.json
python train.py --implementation student_pointer --config configs/pointer8.json --resume runs/break15-pointer-init/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 47 --steps 6000 --batch-size 32 --learning-rate 0.00015 --warmup-steps 100 --eval-every 1200 --save-best --run-dir runs/break15-pointer-lr00015-s47
python train.py --implementation student_pointer --config configs/pointer8.json --resume runs/break15-pointer-lr00015-s47/checkpoint.pt --device cuda --precision bf16 --threads 4 --seed 48 --steps 6000 --batch-size 32 --learning-rate 0.0001 --warmup-steps 100 --eval-every 1200 --save-best --run-dir runs/break15-pointer2-lr0001-s48
Copy-Item configs/cache_v5_calibration.json runs/break15-pointer2-calibration.json
python experiments/assemble_cached_checkpoint.py --tables runs/final-candidate-depth8-v4/checkpoint.pt --neural runs/break15-pointer2-lr0001-s48/checkpoint.pt --calibration runs/break15-pointer2-calibration.json --output runs/final-candidate-depth8-v5/checkpoint.pt
python evaluate.py --checkpoint runs/final-candidate-depth8-v5/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python experiments/measure_evaluation.py --checkpoint runs/final-candidate-depth8-v5/checkpoint.pt --split test --threads 4 --output runs/final-candidate-depth8-v5/test_cpu_fp32.json
python -m unittest discover -s tests -v
```

Direct scoring without retraining requires the frozen checkpoint, `student_cache.py`,
`student.py`, the fixed evaluator/common code and supplied data. Runtime imports are
listed in `configs/cache_v5_source_hashes.json`; the frozen source snapshot is also
saved beside the checkpoint. `runs/` and `*.pt` are ignored by Git, so publish the
checkpoint as a separate immutable artifact. The calibration JSON in `configs/`
contains scalar settings and validation scores, not token-level answers.

## AI assistance disclosure

OpenAI Codex provided substantive assistance. It reviewed the assignment code and
course slides; proposed and implemented SwiGLU, model scaling, continuation training,
coverage sampling, checkpoint interpolation, identity-initialized depth expansion,
selective block training, training-corpus n-gram interpolation, FP16 checkpoint packing
and sparse table storage; optimized mixture weights continuously; designed the fixed
training-only singleton hash selection, lossless packed keys and normalized uint8
probability storage; wrote experiment scripts and tests; installed and verified the
CUDA PyTorch environment; executed and analyzed the learning-rate and mixture sweeps;
implemented causal neural prefix caching, similarity-conditioned interpolation,
temperature calibration and the copy-mixture training objective; ran the rejected
dropout and long-n-gram experiments; measured resource use; and drafted this
experiment record. The student must review and
understand the implementation, verify the reported results, retain this disclosure in
the submitted repository/report, and write the final analysis in their own words.
