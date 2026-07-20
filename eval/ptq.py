"""The missing baseline: what happens if you ternarize the FP16 model AFTER training (PTQ)?

This is what every existing low-bit dLLM paper does. If PTQ collapses at 1.58 bit while
native QAT does not, that difference is the reason this project exists.

Evaluated on the identical validation slice, mask seed, and metric as training, so the three
numbers below are directly comparable.
"""
import sys
import numpy as np
import torch

from model.arch import TernaryDiffusionLM, TernaryDiffusionConfig, strip_compile_prefix
from train.train_diffusion import evaluate

DATA = "/workspace/data/slice.bin"
SEQ, BATCH, VAL_BATCHES = 512, 32, 40
MASK_ID = 32000


def load(path, force_ternary=None):
    ck = torch.load(path, map_location="cpu")
    cfg = TernaryDiffusionConfig(**ck["cfg"])
    if force_ternary is not None:
        cfg.ternary = force_ternary
    m = TernaryDiffusionLM(cfg)
    m.load_state_dict(strip_compile_prefix(ck["model"]))
    return m.cuda().eval(), cfg


data = np.memmap(DATA, dtype=np.uint16, mode="r")
split = int(len(data) * 0.99)
val = data[split:]

rows = []

# 1. FP16 baseline, as trained (upper bound)
m, cfg = load("/workspace/sweep/fp16_2e-3/best.pt")
s = evaluate(m, val, MASK_ID, BATCH, SEQ, "cuda", n_batches=VAL_BATCHES)
rows.append(("FP16 (as trained)", s["val_masked_ce"], s["val_ppl"]))
del m; torch.cuda.empty_cache()

# 2. PTQ: the SAME FP16 weights, ternarized after the fact (cfg.ternary flipped on).
#    No calibration, no gradient — the naive 1.58-bit PTQ that native training is meant to beat.
m, cfg = load("/workspace/sweep/fp16_2e-3/best.pt", force_ternary=True)
s = evaluate(m, val, MASK_ID, BATCH, SEQ, "cuda", n_batches=VAL_BATCHES)
rows.append(("PTQ ternary (FP16 weights, quantized after)", s["val_masked_ce"], s["val_ppl"]))
del m; torch.cuda.empty_cache()

# 3. Native ternary QAT (ours)
m, cfg = load("/workspace/sweep/tern_8e-3/best.pt")
s = evaluate(m, val, MASK_ID, BATCH, SEQ, "cuda", n_batches=VAL_BATCHES)
rows.append(("QAT ternary (native, ours)", s["val_masked_ce"], s["val_ppl"]))

print()
print(f"{'model':<46} {'masked-CE':>10} {'ppl':>10}")
print("-" * 70)
for name, ce, ppl in rows:
    print(f"{name:<46} {ce:>10.4f} {ppl:>10.1f}")
print()
fp16, ptq, qat = rows[0][1], rows[1][1], rows[2][1]
print(f"PTQ vs FP16 : {ptq - fp16:+.4f} CE  ({100*(np.exp(ptq-fp16)-1):+.1f}% ppl)")
print(f"QAT vs FP16 : {qat - fp16:+.4f} CE  ({100*(np.exp(qat-fp16)-1):+.1f}% ppl)")
