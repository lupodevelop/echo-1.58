"""Training loop for the mitigated model.

Deliberately identical to ../ECHO-DIFF-TERNARY/train.py — same optimizer, same masking, same
validation windows and mask seed — EXCEPT for the knobs listed below. If an arm improves, the knob
is the only thing it can be.

WHAT IS HERE, AND WHY EACH ONE SURVIVED THE LITERATURE

  --self-cond P      Loopholing self-conditioning (arXiv 2510.19304). Second forward pass sees its
                     own hidden state from the first. -1.9 test PPL FROM SCRATCH at 78-110M, which
                     is our regime. The strongest single result in the diffusion-LM literature that
                     applies to us. Costs ~+30-40% step time.

  --stratified       Stratified t. --antithetic: complementary mask pairs (MIRROR). Both are
                     unbiased variance reduction. Neither has a published PPL delta — they buy
                     STABILITY, which is worth more to us than to the people who published them,
                     because ternary QAT already floods the gradient with STE noise.

  --two-stage        BitNet b1.58's optimizer schedule: peak LR ~2x the FP16 one, then a hard drop
                     to a second cosine at 50% of tokens; weight decay 0.1 -> 0 at 2/3. This is the
                     one schedule finding with DIRECT ternary-LM evidence (Spectra/TriLM ablation,
                     arXiv 2407.12327 Fig. 6): both interventions together give the lowest final
                     loss, either alone is worse. Latent weights are NEVER weight-decayed — decaying
                     a weight whose only job is to pick a bin just drags it onto the decision
                     boundary, which is the oscillation you are trying to avoid.

  --dampen LAMBDA    Oscillation dampening (Nagel et al., ICML 2022). Ramped 0 -> lambda, because a
                     constant lambda over-regularizes (their Table 4 is explicit). Only worth
                     turning on if --track-osc says the weights are actually oscillating.

  --teacher CKPT     Distil from the frozen FP16 twin we are training anyway for the gate. The
                     teacher is free: we pay for it either way. Now on the SAME masked batch as the
                     main loss (one student forward, not two) — the old version drew a fresh mask
                     for the KD term, which meant a second full student forward AND a teacher that
                     was answering a different question than the one being graded.

WHAT IS DELIBERATELY NOT HERE
  - Progressive soft-to-hard quantization (lambda ramp 0->1). It exists to protect a PRETRAINED FP
    model from the shock of quantization. We train from scratch: there is nothing to protect, and
    the ramp would spend the first half of the budget training a model we do not ship. The one
    from-scratch test of it (HF, SmolLM-135M) found "minimal improvement".
  - EWGS / ReSTE / DSQ. arXiv 2405.05171 proves this class of estimator is equivalent to plain STE
    under an adaptive optimizer. We use AdamW.
  - Sampler distillation. It reduces sampling steps; it does not make the model better.
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
from model.mitigations import MitigatedDiffusionLM
from model.quant import MitigationConfig, QATConfig
from train.diffusion import diffusion_loss, forward_mask, zero_mask_logit


def cfg_27m(vocab=32001, mask=32000):
    """The bake-off scale. Small enough to run several arms on a cheap GPU, big enough that the
    ternary-FP16 gap is already measurable (8.6% at this size in the baseline project)."""
    return B.TernaryDiffusionConfig(vocab_size=vocab, mask_token_id=mask, d_model=448,
                                    n_layers=8, n_heads=7, d_ff=1216, max_seq_len=512)


def get_batch(data, batch, seq_len, device, gen=None):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True)


def causal_bias(T, device, dtype):
    """Additive causal mask. The model is bidirectional by construction; this makes it AR without
    touching a single line of the architecture, which is exactly the point (see ar_loss)."""
    m = torch.full((T, T), float("-inf"), device=device, dtype=dtype).triu(1)
    return m[None, None]                                        # (1, 1, T, T)


def ar_loss(model, x0):
    """Next-token cross-entropy under a causal mask. THE CONTROL EXPERIMENT OF THIS PROJECT.

    Every published 1.58-bit result — BitNet, TriLM, Spectra, FBI-LLM — is AUTOREGRESSIVE. We are
    masked-diffusion. We measure a 21% ternary penalty; they report far less. The obvious question,
    which nobody in either literature has asked, is whether the penalty is a property of TERNARY
    WEIGHTS or a property of TERNARY WEIGHTS *UNDER A MASKED-DIFFUSION OBJECTIVE*.

    This function answers it, and it is a fair test precisely because NOTHING else changes: same
    transformer, same corpus, same tokenizer, same parameter count, same token budget, same
    quantizer. The only difference is a triangular attention mask and a shifted label. So the two
    gaps — gap_AR and gap_MDM — are directly comparable, and their RATIO is the result.

    Mechanism, if the gap turns out to be larger for diffusion: an AR model predicts ONE token from
    a FULLY CLEAN left context. A masked diffusion model predicts MANY tokens in parallel from a
    PERFORATED context, and must model the dependencies among its own simultaneous predictions.
    That is a harder function to represent, and representing functions is exactly what a ternary
    weight has ten times less room to do.
    """
    T = x0.shape[1]
    bias = causal_bias(T, x0.device, torch.float32)
    logits = model(x0, attn_bias=bias)                          # (B, T, V)
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           x0[:, 1:].reshape(-1))


@torch.no_grad()
def evaluate_ar(model, val, batch, seq_len, device, n_batches=40, seed=1234):
    """Same fixed val windows as the diffusion evaluator, so both objectives are scored on exactly
    the same text. NOTE the two CE numbers are NOT comparable to each other (different tasks) — but
    the ternary-vs-FP16 GAP within each objective is, and the gap is what we are measuring."""
    was = model.training
    model.eval()
    ig = torch.Generator().manual_seed(seed)
    tot = torch.zeros((), device=device)
    for _ in range(n_batches):
        x0 = get_batch(val, batch, seq_len, device, ig)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += ar_loss(model, x0).float()
    if was:
        model.train()
    return (tot / n_batches).item()


@torch.no_grad()
def evaluate(model, val, mask_id, batch, seq_len, device, n_batches=40, t_eps=1e-3, seed=1234,
             self_cond=False):
    """IDENTICAL masked val set for every arm: fixed seeds for both the window draw and the mask.
    Without that, an arm can win by drawing an easier val set and the whole bake-off is worthless.

    Note evaluation uses NO stratification and NO antithetic pairs even when training does — those
    are training-time variance reducers, and using them here would change what is being measured.
    """
    was = model.training
    model.eval()
    ig = torch.Generator().manual_seed(seed)
    mg = torch.Generator(device=device).manual_seed(seed)
    ce_tot = torch.zeros((), device=device)
    n_tot = torch.zeros((), device=device)
    for _ in range(n_batches):
        x0 = get_batch(val, batch, seq_len, device, ig)
        xt, m, _ = forward_mask(x0, mask_id, t_eps, generator=mg)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self_cond:
                _, h = model(xt, return_h=True)
                lg = model(xt, h_prev=h)
            else:
                lg = model(xt)
        lg = zero_mask_logit(lg.float(), mask_id)
        bs, t, v = lg.shape
        ce = F.cross_entropy(lg.view(bs * t, v), x0.view(bs * t), reduction="none").view(bs, t)
        ce_tot += (ce * m).sum()
        n_tot += m.sum()
    if was:
        model.train()
    return (ce_tot / n_tot.clamp(min=1)).item()


def lr_at(step, warmup, total, lr, two_stage=False, floor=0.1):
    """One cosine, or BitNet's two-stage schedule: cosine to 50%, then hard-drop the peak and run a
    second cosine. The abrupt drop causes a transient loss bump; it is harmless and expected."""
    if step < warmup:
        return lr * (step + 1) / warmup
    if not two_stage:
        p = (step - warmup) / max(total - warmup, 1)
        return lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))

    half = total // 2
    if step < half:
        p = (step - warmup) / max(half - warmup, 1)
        return lr * (0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * p)))     # stage 1: peak -> 0.3*peak
    p = (step - half) / max(total - half, 1)
    return lr * 0.3 * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))   # stage 2


def wd_at(step, total, wd, two_stage=False):
    """Weight decay -> 0 at 2/3 of training (TriLM). Large WD keeps latent weights small, i.e.
    pinned near the ternary decision boundary, i.e. flipping. Killing it late lets them commit."""
    if not two_stage:
        return wd
    return wd if step < (2 * total) // 3 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None, help="name of this arm in results/")
    ap.add_argument("--tokens", type=float, default=150e6)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--no-ternary", action="store_true")
    ap.add_argument("--ar", action="store_true",
                    help="autoregressive control: causal mask + next-token loss, same everything else")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--val-every", type=int, default=500)
    # inference-capacity mitigations
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--lowrank", type=int, default=0)
    ap.add_argument("--lowrank-bits", type=int, default=8)
    ap.add_argument("--outliers", type=float, default=0.0)
    # QAT knobs
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--twn-alpha", action="store_true")
    ap.add_argument("--absmedian", action="store_true")
    ap.add_argument("--chan-scale", action="store_true")
    ap.add_argument("--dampen", type=float, default=0.0)
    ap.add_argument("--two-stage", action="store_true")
    ap.add_argument("--track-osc", action="store_true")
    # diffusion objective
    ap.add_argument("--self-cond", type=float, default=0.0, help="prob. of the 2nd (conditioned) pass")
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--antithetic", action="store_true")
    # distillation
    ap.add_argument("--teacher", default=None, help="frozen FP16 checkpoint")
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--cfg-from-teacher", action="store_true",
                    help="build the student at the teacher's EXACT architecture (for the 300M run): "
                         "read d_model/n_layers/etc. from the teacher checkpoint's cfg, flip ternary "
                         "on. Guarantees student and teacher are shape-compatible for distillation "
                         "instead of hardcoding a 300M config that might drift from the trained twin.")
    ap.add_argument("--init-from", default=None,
                    help="CONTINUED-DISTILLATION mode: initialise the student from an already-trained "
                         "ternary checkpoint instead of from scratch. Loaded strict=False so a new "
                         "per-channel scale starts at identity (1.0). The reading this enables: 'I "
                         "trained ternary, I have a known degradation, and by continuing with a "
                         "teacher I recover X of it', without paying for a from-scratch run. Use a "
                         "small LR so the continued training does not destroy what the model learned.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(res_dir, exist_ok=True)
    tag = args.tag or ("fp16" if args.no_ternary else "ternary")

    if args.cfg_from_teacher:
        if not args.teacher:
            raise SystemExit("--cfg-from-teacher needs --teacher")
        tck = torch.load(args.teacher, map_location="cpu")
        cfg = B.TernaryDiffusionConfig(**tck["cfg"])
        print(f"  cfg from teacher: d_model={cfg.d_model} n_layers={cfg.n_layers}")
    else:
        cfg = cfg_27m()
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    cfg.ternary = not args.no_ternary
    mit = MitigationConfig(per_channel_scale=args.per_channel, lowrank=args.lowrank,
                           lowrank_bits=args.lowrank_bits, outliers=args.outliers)
    qat = QATConfig(thresh=args.thresh, twn_alpha=args.twn_alpha, absmedian=args.absmedian,
                    chan_scale=args.chan_scale, dampen=args.dampen)
    model = MitigatedDiffusionLM(cfg, mit, qat, self_cond=args.self_cond > 0).to(dev)
    if args.init_from:
        ick = torch.load(args.init_from, map_location="cpu")
        sd = B.strip_compile_prefix(ick["model"])
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # missing should be only the new per-channel scales (start at identity); unexpected should
        # be empty. Anything else means the checkpoint architecture does not match and the continued
        # run would be silently training garbage.
        newp = [k for k in missing if "cscale" in k]
        bad = [k for k in missing if "cscale" not in k]
        print(f"  init-from {args.init_from}: loaded, {len(newp)} new per-channel scales at identity, "
              f"{len(unexpected)} unexpected keys")
        assert not bad, f"init-from mismatch, missing non-cscale keys: {bad[:5]}"
        assert not unexpected, f"init-from mismatch, unexpected keys: {unexpected[:5]}"
    if args.track_osc:
        model.track_oscillation(True)

    n_params = sum(p.numel() for p in model.parameters())
    extra = model.mitigation_bytes() / 2**20
    tern = model.ternary_bytes() / 2**20
    print(f"[{tag}] {'FP16' if args.no_ternary else 'ternary'} | {n_params/1e6:.1f}M params | seed {args.seed}")
    print(f"  qat: {qat}\n  mit: {mit}")
    if not args.no_ternary:
        print(f"  memory: ternary {tern:.1f} MB + extra {extra:.2f} MB "
              f"(+{100*extra/max(tern,1e-9):.1f}%) | zero_frac {model.zero_frac():.3f}")

    teacher = None
    if args.teacher:
        ck = torch.load(args.teacher, map_location="cpu")
        tcfg = B.TernaryDiffusionConfig(**ck["cfg"])
        teacher = B.TernaryDiffusionLM(tcfg)
        teacher.load_state_dict(B.strip_compile_prefix(ck["model"]))
        teacher = teacher.to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"  teacher: {args.teacher} (frozen, ternary={tcfg.ternary})")

    if args.compile:
        model = torch.compile(model)

    # Two param groups. The latent BitLinear weights get NO weight decay, ever: their job is to
    # pick a bin, and decaying them is a constant pull toward the decision boundary.
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    latent_ids = {id(p) for p in base.latent_params()}
    decay = [p for p in model.parameters() if p.requires_grad and id(p) not in latent_ids and p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and (id(p) in latent_ids or p.dim() < 2)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    print(f"  params: {sum(p.numel() for p in decay)/1e6:.1f}M decayed, "
          f"{sum(p.numel() for p in nodecay)/1e6:.1f}M not (latent + 1-D)")

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    split = int(len(data) * 0.99)
    train, val = data[:split], data[split:]

    # Antithetic masking runs 2B examples per step (the two complementary halves), so a step costs
    # DOUBLE the FLOPs and sees double the tokens. Without this correction the antithetic arms
    # silently train on 2x the token budget and "win" for that reason, not for variance reduction —
    # exactly the confound that invalidated the first t_varred run. Halve the steps to hold the
    # token budget fixed, which is the only fair comparison.
    per_step_mult = 2 if args.antithetic else 1
    tps = args.batch * args.seq_len * per_step_mult
    total = int(args.tokens / tps)
    print(f"  tokens/step {tps:,} (antithetic x{per_step_mult}) | steps {total:,}", flush=True)

    hist = []
    best = float("inf")
    model.train()
    t0 = time.time()
    wall = time.time()

    for step in range(total):
        lr_now = lr_at(step, 200, total, args.lr, args.two_stage)
        wd_now = wd_at(step, total, args.wd, args.two_stage)
        opt.param_groups[0]["lr"] = lr_now
        opt.param_groups[1]["lr"] = lr_now
        opt.param_groups[0]["weight_decay"] = wd_now

        x0 = get_batch(train, args.batch, args.seq_len, dev)
        opt.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if args.ar:
                loss = ar_loss(model, x0)
            else:
                loss, st = diffusion_loss(model, x0, cfg.mask_token_id, stratified=args.stratified,
                                          antithetic=args.antithetic, self_cond_p=args.self_cond)

            if teacher is not None:
                # SAME masked batch as the main loss: one extra forward for the teacher, none for
                # the student. The teacher's full distribution over the masked positions is a far
                # richer target than the one-hot label — this is BitDistiller's lever, and the most
                # reliable gap-closer in the QAT literature.
                xt, m, _ = forward_mask(x0, cfg.mask_token_id, stratified=args.stratified)
                with torch.no_grad():
                    tl = teacher(xt)
                sl = model(xt)
                T = args.kd_temp
                kd = F.kl_div(F.log_softmax(sl[m] / T, dim=-1),
                              F.log_softmax(tl[m] / T, dim=-1),
                              reduction="batchmean", log_target=True) * (T * T)
                loss = loss + args.kd_weight * kd

            if args.dampen > 0:
                # ramped 0 -> lambda: a constant lambda over-regularizes (Nagel Table 4)
                lam = args.dampen * (step / max(total - 1, 1))
                loss = loss + lam * base.dampen_loss()

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()

        if step % args.val_every == 0 and step:
            v = (evaluate_ar(model, val, args.batch, args.seq_len, dev) if args.ar else
                 evaluate(model, val, cfg.mask_token_id, args.batch, args.seq_len, dev,
                          self_cond=args.self_cond > 0))
            best = min(best, v)
            rec = {"step": step, "val_masked_ce": v, "train_loss": loss.item(),
                   "lr": lr_now, "wd": wd_now,
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

    v = (evaluate_ar(model, val, args.batch, args.seq_len, dev) if args.ar else
         evaluate(model, val, cfg.mask_token_id, args.batch, args.seq_len, dev,
                  self_cond=args.self_cond > 0))
    best = min(best, v)

    # The FP16 arm is the distillation teacher for a later arm, so every arm saves its weights.
    # Costs one file; not saving it would mean re-running an hour of GPU to get the teacher back.
    torch.save({"model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                "cfg": vars(cfg), "tag": tag, "best_val": best},
               os.path.join(args.out, "final.pt"))

    # Everything a table in the paper needs, and everything a referee needs to rerun it.
    out = {
        "tag": tag, "ternary": not args.no_ternary, "seed": args.seed,
        "objective": "ar" if args.ar else "mdm",
        "params_m": n_params / 1e6, "tokens": args.tokens, "steps": total,
        "best_val_masked_ce": best, "best_val_ppl": math.exp(best),
        "final_val_masked_ce": v,
        "ternary_mb": tern, "extra_mb": extra,
        "zero_frac": base.zero_frac() if not args.no_ternary else None,
        "wall_min": (time.time() - wall) / 60,
        "gpu": torch.cuda.get_device_name(0),
        "cfg": vars(cfg) if hasattr(cfg, "__dict__") else str(cfg),
        "args": vars(args),
        "history": hist,
    }
    with open(os.path.join(res_dir, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    with open(os.path.join(res_dir, "bakeoff.jsonl"), "a") as f:
        f.write(json.dumps({k: out[k] for k in
                            ("tag", "ternary", "seed", "params_m", "tokens",
                             "best_val_masked_ce", "best_val_ppl", "extra_mb", "zero_frac",
                             "wall_min", "gpu")}, default=str) + "\n")

    print(f"done. [{tag}] best val masked_ce {best:.4f} (ppl {math.exp(best):.1f}) "
          f"| extra {extra:.2f} MB | {out['wall_min']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
