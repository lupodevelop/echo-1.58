"""Training loop for the ternary autoregressive LM.

WHAT IS IN HERE, AND WHAT THE SIBLING PROJECT PAID TO FIND OUT

This file is short because most of the QAT literature did not survive contact with our data. What
we learned in ../ECHO-DIFF-MITIGATIONS, and what it cost:

  KEPT — two-stage LR + weight decay -> 0, and NEVER decaying the latent weights.
    The one schedule finding with direct ternary-LM evidence (Spectra/TriLM ablation). A latent
    weight's only job is to pick a bin; decaying it drags it onto the decision boundary.

  KEPT — distillation from the FP16 twin.
    The only lever that INJECTS information the ternary model cannot extract on its own. If the
    ternary penalty is a capacity ceiling (which our data suggests), this is mechanically the only
    training-time technique that can help, because everything else merely rearranges what is
    already there.

  DROPPED — oscillation dampening (Nagel et al., ICML 2022).
    Measured, not assumed: only 0.09% of ternary weights change bin late in training. The technique
    treats a disease we do not have. Kept the INSTRUMENT (--track-osc) so the claim stays checkable
    on this objective too — the oscillation rate may well differ under AR, and if it does, that is
    itself worth knowing.

  DROPPED — EWGS, ReSTE, DSQ, and the whole "better STE" literature.
    arXiv 2405.05171 proves this class of estimator is equivalent to plain STE under an adaptive
    optimizer. We use AdamW. Not implemented, on purpose.

  DROPPED — progressive soft-to-hard quantization.
    It exists to protect a PRETRAINED FP model from the shock of quantization. We train from
    scratch: there is nothing to protect.

  UNTESTED HERE, ON PURPOSE — self-conditioning, stratified t, antithetic masks.
    All three are diffusion-objective techniques. They have no meaning for autoregression, and the
    strongest of them (self-conditioning) bought exactly nothing even where it did apply.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import arch as B
from model.model_ar import TernaryAR
from model.quant import MitigationConfig, QATConfig


def cfg_27m(vocab=32001, mask=32000):
    """Deliberately IDENTICAL to the diffusion project's bake-off config. If the width, depth or
    head count differed, the AR-vs-diffusion comparison would be confounded and the entire reason
    this folder exists would evaporate."""
    return B.TernaryDiffusionConfig(vocab_size=vocab, mask_token_id=mask, d_model=448,
                                    n_layers=8, n_heads=7, d_ff=1216, max_seq_len=512)


def get_batch(data, batch, seq_len, device, gen=None):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, val, batch, seq_len, device, n_batches=40, seed=1234):
    """Fixed val windows, same seed for every arm. Without that, an arm can win by drawing an easier
    validation set and the comparison is worthless."""
    was = model.training
    model.eval()
    g = torch.Generator().manual_seed(seed)
    tot = torch.zeros((), device=device)
    for _ in range(n_batches):
        x = get_batch(val, batch, seq_len, device, g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += model.loss(x).float()
    if was:
        model.train()
    return (tot / n_batches).item()


def lr_at(step, warmup, total, lr, two_stage=False, floor=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    if not two_stage:
        p = (step - warmup) / max(total - warmup, 1)
        return lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))
    half = total // 2
    if step < half:
        p = (step - warmup) / max(half - warmup, 1)
        return lr * (0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * p)))
    p = (step - half) / max(total - half, 1)
    return lr * 0.3 * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def wd_at(step, total, wd, two_stage=False):
    if not two_stage:
        return wd
    return wd if step < (2 * total) // 3 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tokens", type=float, default=300e6)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--no-ternary", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--twn-alpha", action="store_true")
    ap.add_argument("--chan-scale", action="store_true")
    ap.add_argument("--two-stage", action="store_true")
    ap.add_argument("--track-osc", action="store_true")
    ap.add_argument("--teacher", default=None, help="frozen FP16 checkpoint to distil from")
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(res, exist_ok=True)
    tag = args.tag or ("ar_fp16" if args.no_ternary else "ar_ternary")

    cfg = cfg_27m()
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    cfg.ternary = not args.no_ternary
    qat = QATConfig(thresh=args.thresh, twn_alpha=args.twn_alpha, chan_scale=args.chan_scale)
    model = TernaryAR(cfg, MitigationConfig(), qat).to(dev)
    if args.track_osc:
        model.track_oscillation(True)

    n = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] AR | {'FP16' if args.no_ternary else 'ternary'} | {n/1e6:.1f}M params | seed {args.seed}")
    if not args.no_ternary:
        print(f"  ternary {model.ternary_bytes()/2**20:.1f} MB | zero_frac {model.zero_frac():.3f}")

    teacher = None
    if args.teacher:
        ck = torch.load(args.teacher, map_location="cpu")
        tcfg = B.TernaryDiffusionConfig(**ck["cfg"])
        teacher = TernaryAR(tcfg).to(dev).eval()
        teacher.load_state_dict(B.strip_compile_prefix(ck["model"]))
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"  teacher: {args.teacher} (frozen)")

    if args.compile:
        model = torch.compile(model)

    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    latent = {id(p) for p in base.latent_params()}
    decay = [p for p in model.parameters() if p.requires_grad and id(p) not in latent and p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and (id(p) in latent or p.dim() < 2)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    split = int(len(data) * 0.99)
    train, val = data[:split], data[split:]
    tps = args.batch * args.seq_len
    total = int(args.tokens / tps)
    print(f"  tokens/step {tps:,} | steps {total:,}", flush=True)

    hist, best = [], float("inf")
    model.train()
    t0 = wall = time.time()

    for step in range(total):
        lr_now = lr_at(step, 200, total, args.lr, args.two_stage)
        opt.param_groups[0]["lr"] = opt.param_groups[1]["lr"] = lr_now
        opt.param_groups[0]["weight_decay"] = wd_at(step, total, args.wd, args.two_stage)

        x = get_batch(train, args.batch, args.seq_len, dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(x)
            if teacher is not None:
                # Same batch, so the teacher answers exactly the question the student is graded on.
                with torch.no_grad():
                    tl = teacher(x)
                sl = model(x)
                T = args.kd_temp
                kd = F.kl_div(F.log_softmax(sl / T, dim=-1), F.log_softmax(tl / T, dim=-1),
                              reduction="batchmean", log_target=True) * (T * T)
                loss = loss + args.kd_weight * kd

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()

        if step % args.val_every == 0 and step:
            v = evaluate(model, val, args.batch, args.seq_len, dev)
            best = min(best, v)
            rec = {"step": step, "val_ce": v, "train_loss": loss.item(), "lr": lr_now,
                   "tok_s": args.val_every * tps / (time.time() - t0)}
            if not args.no_ternary:
                rec["zero_frac"] = base.zero_frac()
                if args.track_osc:
                    rec["osc_frac"] = base.osc_frac()
            hist.append(rec)
            osc = f" | osc {rec['osc_frac']:.4f}" if "osc_frac" in rec else ""
            print(f"step {step:6d}/{total} | loss {loss.item():.4f} | val {v:.4f} | best {best:.4f}"
                  f"{osc} | {rec['tok_s']/1e3:.1f}k tok/s", flush=True)
            t0 = time.time()

    v = evaluate(model, val, args.batch, args.seq_len, dev)
    best = min(best, v)
    torch.save({"model": base.state_dict(), "cfg": vars(cfg), "tag": tag, "best_val": best},
               os.path.join(args.out, "final.pt"))

    out = {"tag": tag, "objective": "ar", "ternary": not args.no_ternary, "seed": args.seed,
           "params_m": n / 1e6, "tokens": args.tokens, "steps": total,
           "best_val_ce": best, "best_val_ppl": math.exp(best),
           "zero_frac": base.zero_frac() if not args.no_ternary else None,
           "wall_min": (time.time() - wall) / 60, "gpu": torch.cuda.get_device_name(0),
           "args": vars(args), "history": hist}
    json.dump(out, open(os.path.join(res, f"{tag}.json"), "w"), indent=2, default=str)
    with open(os.path.join(res, "runs.jsonl"), "a") as f:
        f.write(json.dumps({k: out[k] for k in ("tag", "objective", "ternary", "seed", "params_m",
                                                "tokens", "best_val_ce", "best_val_ppl",
                                                "zero_frac", "wall_min", "gpu")}, default=str) + "\n")
    print(f"done. [{tag}] best val CE {best:.4f} (ppl {math.exp(best):.1f}) | {out['wall_min']:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
