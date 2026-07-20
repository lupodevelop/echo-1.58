# Results

Every number in the paper, with its source. All evaluations use identical seeded validation
windows and masks; ternary and full-precision models are paired (same tokens, same order,
same masking). Run-to-run noise floor: 0.036 masked-CE (single pair), 0.013 seed standard
deviation (3-seed).

## PTQ collapse vs native QAT (27M, 150M tokens)

The headline "26x" collapse is the round-to-nearest per-tensor result on the best-LR FP16
checkpoint (a different seed from the calibrated table below; the collapse is seed-independent):
FP16 398.5 -> RTN ternary 10346.5 = 25.96x; native QAT 411.9 = +3.4%.

### Calibrated PTQ (seed with FP16 392.0)

| Model | masked-CE | perplexity | vs FP16 |
|---|---|---|---|
| FP16 (as trained) | 5.9714 | 392.0 | 1x |
| RTN per-tensor + int8 activations | 8.8129 | 6720.2 | 17.1x |
| RTN per-channel + int8 | 8.3311 | 4150.9 | 10.6x |
| GPTQ per-channel, activation ordering + int8 | 7.5158 | 1836.9 | 4.7x |
| Native ternary QAT | 6.0452 | 422.1 | 1.08x |

## Scaling of the native penalty (27M)

| Tokens | Ternary | FP16 | gap (perplexity) |
|---|---|---|---|
| 150M | | | +8.6% |
| 300M (3-seed) | 5.9175 | 5.7399 | +19.4% (17.7% at best learning rate) |

## Recovery ablation (27M, 300M tokens, 3-seed means for the winning arms)

| Arm | delta masked-CE | gap closed | category |
|---|---|---|---|
| distillation + per-channel | +0.124 | 70% | additive |
| distillation | +0.089 | 50% | additive |
| per-channel scale | +0.057 | 30% | additive |
| two-stage learning rate | +0.029 | 15% | redistributive |
| zero-threshold + TWN | +0.014 | 7% | redistributive |
| self-conditioning | -0.011 | 0% | redistributive |
| + variance reduction | -0.016 | 0% | redistributive |
| all arms except distillation | -0.035 | -18% | redistributive |

## Recovery at 341M (common re-evaluation protocol, sequence length 1024)

| 341M model | masked-CE | perplexity | vs FP16 |
|---|---|---|---|
| FP16 twin | 4.8100 | 122.7 | ceiling |
| Ternary baseline | 4.9852 | 146.2 | +19.2% |
| + continued distillation (post-hoc) | 4.9125 | 136.0 | +10.8% (41% recovered; 48-49% on test and out-of-domain) |
| + from-scratch recipe | 4.9878 | 146.6 | +19.5% (within noise of baseline: recipe from step 0 does not scale) |

## Constant offset (mature regime, roughly 8+ tokens per parameter)

| Setting | unique tokens | offset (nats/token) |
|---|---|---|
| 27M, fresh, 300M tokens (3-seed) | 300M | 0.178 |
| 27M, repeated, 3 epochs | 75M | 0.192 |
| 27M, repeated, 5 epochs | 75M | 0.184 |
| 27M, repeated, 10 epochs | 75M | 0.194 |
| 27M, repeated, 20 epochs | 75M | 0.219 |
| 341M, fresh, 4B tokens | 4B | 0.175 |
| mean +/- std | | 0.19 +/- 0.016 |

## Repetition ladder (27M, fixed 75M-token pool, masked-CE)

| Track | E=1 | E=2 | E=3 | E=5 | E=10 | E=20 |
|---|---|---|---|---|---|---|
| FP16 twin | 6.3716 | 6.0442 | 5.8730 | 5.7517 | 5.6262 | 5.5547 |
| Ternary | 6.3561 | 6.1553 | 6.0654 | 5.9358 | 5.8206 | 5.7739 |
| Ternary + recipe | 6.3036 | 6.0640 | 5.9518 | 5.8315 | 5.7446 | - |
| gap (perplexity) | -1.5% | +11.8% | +21.2% | +20.2% | +21.5% | +24.5% |

Recovery erodes with depth: 82% (E=2), 59% (E=3), 57% (E=5), 39% (E=10).

## Mask-rate decomposition (341M): where the penalty binds

| mask ratio | ternary CE | FP16 CE | delta-CE (nats) | gap (perplexity) |
|---|---|---|---|---|
| 0.1-0.2 | 2.317 | 2.068 | 0.249 | +28% |
| 0.2-0.3 | 2.658 | 2.399 | 0.259 | +30% |
| 0.3-0.4 | 3.148 | 2.881 | 0.267 | +31% |
| 0.4-0.5 | 3.654 | 3.392 | 0.262 | +30% |
| 0.5-0.6 | 4.215 | 3.977 | 0.239 | +27% |
| 0.6-0.7 | 4.841 | 4.645 | 0.196 | +22% |
| 0.7-0.8 | 5.512 | 5.359 | 0.153 | +16% |
| 0.8-0.9 | 6.263 | 6.168 | 0.095 | +10% |
| 0.9-1.0 | 7.128 | 7.079 | 0.049 | +5% |

The absolute delta-CE (not just the percentage) falls from 0.25 nats at light masking to 0.05
at heavy masking, so the penalty genuinely concentrates in context-rich prediction and is not
a denominator artifact of easier bins.

## Infilling accuracy (341M, held-out text)

| mask ratio | ternary top-1 | FP16 top-1 | relative gap |
|---|---|---|---|
| 15% | 53.6% | 57.7% | -7.1% |
| 30% | 44.8% | 48.8% | -8.2% |
| 50% | 31.8% | 35.0% | -9.0% |
| 70% | 19.0% | 20.7% | -8.4% |

The ternary model retains about 92% of FP16 infilling accuracy at every ratio; the perplexity
gap overstates the cost on the task the objective trains.

## Autoregressive vs masked-diffusion penalty (27M, matched optimizer)

23.6% autoregressive versus 21.1% masked-diffusion: the penalty is in the weights, not the
objective.
