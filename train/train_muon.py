"""Ternary AR training with Muon + the speedrun recipe, and the diagnostics that make it falsifiable.

THE 2x2 THIS FILE EXISTS TO RUN

    {AdamW, Muon}  x  {pure STE, clipped STE}

Four runs, and they settle four separate open questions at once — none of which has ever been
measured, because nobody has put Muon and ternary QAT in the same room:

  Q1. Is Muon a better optimizer for a BIN SELECTOR than Adam?
      Mechanism says yes, and for a reason that has nothing to do with why it wins in FP: Adam's
      per-coordinate 1/sqrt(v) ERASES the relative gradient magnitudes across coordinates — exactly
      the signal that says which bins deserve to flip. Muon normalizes per matrix and preserves
      them. (See muon.py. The conceptual anchor is "Latent Weights Do Not Exist", arXiv 1906.02107.)

  Q2. Does the clipped-vs-pure STE choice matter again under Muon?
      arXiv 2405.05171 says custom estimators collapse to plain STE — but the proof runs on ADAM's
      invariance to PER-COORDINATE rescaling, and MUON IS NOT PER-COORDINATE. We closed this
      question on Adam's authority. This grid reopens it.

  Q3. Does Newton-Schulz silently delete the clip mask?
      NS mixes rows and columns, so a coordinate whose gradient was masked to zero comes out with a
      full-size update. Prediction: the (Muon, clipped-STE, mask_after_ns=False) cell shows
      dead_frac climbing monotonically — latent weights escaping [-alpha, alpha], saturating, and
      never coming back. We log dead_frac every eval, so this is directly visible.

  Q4. Does Muon without weight decay freeze the bin pattern?
      The bins depend only on W/alpha, so constant-RMS updates inflate ||W|| ~ sqrt(t) and the
      effective LR on the bins decays ~1/sqrt(t). Prediction: wnorm_over_alpha climbs and flip_rate
      collapses by mid-training, while the loss curve says nothing is wrong.

THE TWO NUMBERS TO WATCH, AND THEY ARE NOT THE LOSS

    flip_rate  — fraction of ternary weights that changed bin. This is the model actually LEARNING.
    dead_frac  — fraction of latent weights that escaped the representable range. This is the model
                 DYING.

A ternary run can have a healthy-looking loss curve while its bin pattern froze at step 3000 and a
third of its weights are saturated. Val loss will not tell you. These two will.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import arch as B
from model.model_ar import TernaryAR
from model.muon import build_optimizers
from model.quant import MitigationConfig, QATConfig


def cfg_27m(vocab=32001, mask=32000):
    return B.TernaryDiffusionConfig(vocab_size=vocab, mask_token_id=mask, d_model=448,
                                    n_layers=8, n_heads=7, d_ff=1216, max_seq_len=512)


def get_batch(data, batch, seq_len, device, gen=None):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, val, batch, seq_len, device, n_batches=40, seed=1234):
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


def wsd(step, total, warmup=200, cooldown_frac=0.6, floor=0.1):
    """Warmup-Stable-Decay. Flat, then linear decay to `floor` x peak over the last cooldown_frac.
    Cosine is not used by any current record; WSD also lets you stop anywhere and still have a
    usable checkpoint."""
    if step < warmup:
        return (step + 1) / warmup
    p = step / total
    if p < 1 - cooldown_frac:
        return 1.0
    return floor + (1 - floor) * (1 - p) / cooldown_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokens", type=float, default=300e6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ternary", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--val-every", type=int, default=500)
    # the 2x2
    ap.add_argument("--optim", choices=["adamw", "muon"], default="muon")
    ap.add_argument("--clip-ste", action="store_true", help="mask the gradient outside [-a, a]")
    ap.add_argument("--no-mask-after-ns", action="store_true",
                    help="do NOT restore the clip mask after Newton-Schulz (exposes the hazard)")
    # LRs (speedrun track-3 tuned baseline; the 175x spread is load-bearing, not a typo)
    ap.add_argument("--muon-lr", type=float, default=0.025)
    ap.add_argument("--muon-wd", type=float, default=0.05)
    ap.add_argument("--embed-lr", type=float, default=0.7)
    ap.add_argument("--head-lr", type=float, default=0.004)
    ap.add_argument("--scalar-lr", type=float, default=0.015)
    ap.add_argument("--adamw-lr", type=float, default=4e-3, help="LR for the matrices in the AdamW arm")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(res, exist_ok=True)

    cfg = cfg_27m()
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    cfg.ternary = not args.no_ternary
    qat = QATConfig(clip_ste=args.clip_ste)
    model = TernaryAR(cfg, MitigationConfig(), qat, relu2=True).to(dev)
    model.track_oscillation(True)          # flip_rate is not optional here: it is the experiment

    n = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] AR | {'FP16' if args.no_ternary else 'ternary'} | {n/1e6:.1f}M params")
    print(f"  optim={args.optim} clip_ste={args.clip_ste} "
          f"mask_after_ns={not args.no_mask_after_ns} | d_ff={cfg.d_ff} (ReLU^2, param-matched)")

    opts = build_optimizers(model, muon_lr=args.muon_lr, muon_wd=args.muon_wd,
                            embed_lr=args.embed_lr, head_lr=args.head_lr,
                            scalar_lr=args.scalar_lr,
                            mask_after_ns=not args.no_mask_after_ns,
                            use_muon=(args.optim == "muon"))
    if args.optim == "adamw":
        opts[0].param_groups[-1]["lr"] = args.adamw_lr
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]

    if args.compile:
        model = torch.compile(model)
    core = model._orig_mod if hasattr(model, "_orig_mod") else model

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
        s = wsd(step, total)
        for o, bl in zip(opts, base_lrs):
            for g, lr0 in zip(o.param_groups, bl):
                g["lr"] = lr0 * s

        x = get_batch(train, args.batch, args.seq_len, dev)
        for o in opts:
            o.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(x)
        loss.backward()
        # NO GRADIENT CLIPPING. Measured as pure overhead in nanochat: with zero-init-style projections,
        # QK-norm and Muon the grad norm never reaches 1.0, so the clip never fires and costs ~2%.
        for o in opts:
            o.step()

        if step % args.val_every == 0 and step:
            v = evaluate(model, val, args.batch, args.seq_len, dev)
            best = min(best, v)
            rec = {"step": step, "val_ce": v, "train_loss": loss.item(),
                   "tok_s": args.val_every * tps / (time.time() - t0)}
            if not args.no_ternary:
                bits = core._bits()
                tot_n = sum(m.weight.numel() for m in bits)
                rec["flip_rate"] = core.osc_frac()
                rec["dead_frac"] = sum(m.dead_frac() * m.weight.numel() for m in bits) / tot_n
                rec["wnorm_alpha"] = sum(m.wnorm_over_alpha() for m in bits) / len(bits)
                rec["zero_frac"] = core.zero_frac()
            hist.append(rec)
            extra = (f" | flip {rec['flip_rate']:.4f} | dead {rec['dead_frac']:.3f} "
                     f"| |W|/a {rec['wnorm_alpha']:.0f}") if not args.no_ternary else ""
            print(f"step {step:6d}/{total} | loss {loss.item():.4f} | val {v:.4f} | best {best:.4f}"
                  f"{extra} | {rec['tok_s']/1e3:.1f}k tok/s", flush=True)
            t0 = time.time()

    v = evaluate(model, val, args.batch, args.seq_len, dev)
    best = min(best, v)
    torch.save({"model": core.state_dict(), "cfg": vars(cfg), "tag": args.tag, "best_val": best},
               os.path.join(args.out, "final.pt"))

    out = {"tag": args.tag, "objective": "ar", "ternary": not args.no_ternary,
           "optim": args.optim, "clip_ste": args.clip_ste,
           "mask_after_ns": not args.no_mask_after_ns,
           "seed": args.seed, "params_m": n / 1e6, "tokens": args.tokens, "steps": total,
           "best_val_ce": best, "best_val_ppl": math.exp(best),
           "final_flip_rate": hist[-1].get("flip_rate") if hist else None,
           "final_dead_frac": hist[-1].get("dead_frac") if hist else None,
           "wall_min": (time.time() - wall) / 60, "gpu": torch.cuda.get_device_name(0),
           "args": vars(args), "history": hist}
    json.dump(out, open(os.path.join(res, f"{args.tag}.json"), "w"), indent=2, default=str)
    print(f"done. [{args.tag}] best val CE {best:.4f} (ppl {math.exp(best):.1f}) | "
          f"flip {out['final_flip_rate']} | dead {out['final_dead_frac']} | "
          f"{out['wall_min']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
