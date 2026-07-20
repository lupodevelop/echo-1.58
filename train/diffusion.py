"""
Masked (absorbing-state) diffusion for language — forward process, loss, sampler.

This is the LLaDA / MDLM objective, which is far simpler than continuous diffusion:
it is essentially BERT-style masking with a *random* masking ratio per example and a
1/t reweighting that makes it a proper (continuous-time) ELBO.

Forward process:  each example draws t ~ U(eps, 1); each token is replaced by [MASK]
independently with probability t.

Loss (LLaDA Eq., Monte-Carlo NELBO):
    L = E_{t, x0, x_t} [ (1/t) * (1/L) * sum_{i masked} -log p_theta(x0_i | x_t) ]
The (1/t)(1/L) scaling keeps the loss O(1) (in expectation it equals mean CE over
masked tokens) while remaining an unbiased ELBO estimator.

VARIANCE NOTE: as t -> 0 the 1/t factor blows up. `t_eps` clamps the low end. If
training is jittery, raise t_eps before touching anything else. (LLaDA 1.5's VRPO
attacks this variance for the RL/alignment stage; for pretraining the eps clamp is
the cheap first lever.)
"""

import math

import torch
import torch.nn.functional as F


def sample_t(B, device, t_eps=1e-3, stratified=False, generator=None):
    """Draw the per-example mask rate t.

    STRATIFIED (MDLM App. D.3, and the low-discrepancy sampler MD4 also uses): partition (0,1]
    into B strata and draw one t per stratum instead of B i.i.d. uniforms. The estimator stays
    unbiased — every t still has the right marginal density — but the batch can no longer draw
    five t's clustered at 0.9 and none near 0.1.

    Be honest about what this buys. Nobody has ever published an isolated perplexity delta for
    it; the literature implements it and never ablates it. What it demonstrably cuts is VARIANCE.
    That matters more for us than for anyone who published it, because ternary QAT already injects
    a large amount of noise into the gradient through the straight-through estimator. Variance
    reduction is worth more when you are already noise-limited.
    """
    if stratified:
        u = (torch.arange(B, device=device, dtype=torch.float32)
             + torch.rand(B, device=device, generator=generator)) / B
        u = u[torch.randperm(B, device=device, generator=generator)]   # decorrelate t from position
    else:
        u = torch.rand(B, device=device, generator=generator)
    return (u * (1.0 - t_eps) + t_eps).unsqueeze(1)                    # (B, 1) in (t_eps, 1]


def forward_mask(x0, mask_token_id: int, t_eps: float = 1e-3, generator=None,
                 stratified=False, antithetic=False):
    """Absorbing-state corruption. Returns (x_t, mask, t).

    Pass a torch.Generator (on x0's device) to make the corruption reproducible — used by
    evaluate() so ternary and FP16-baseline runs see the identical masked val set.

    ANTITHETIC (the MIRROR estimator, arXiv 2511.18159): draw ONE uniform field u per example and
    build two complementary masks from it — {u < t} and {u > 1-t}. The two masks are negatively
    correlated by construction, so averaging their losses cuts the mask-pattern component of the
    variance by at least half:  Var = (sigma^2/2)(1 + rho),  rho <= 0.

    This attacks a source of noise that stratified-t does NOT touch. The variance of this
    objective decomposes into three parts — WHICH t you drew, WHICH mask pattern you drew given t,
    and the data itself. Stratifying t fixes only the first. The mask pattern is the one an
    autoregressive model does not have at all, and it is large.

    Returns doubled tensors (2B, T) when antithetic=True: the caller must handle it, and
    diffusion_loss does.
    """
    B, T = x0.shape
    t = sample_t(B, x0.device, t_eps, stratified, generator)           # (B, 1)

    if antithetic:
        u = torch.rand(B, T, device=x0.device, generator=generator)
        mask = torch.cat([u < t, u > (1.0 - t)], dim=0)                # (2B, T), complementary
        x0 = x0.repeat(2, 1)
        t = t.repeat(2, 1)
    else:
        mask = torch.rand(B, T, device=x0.device, generator=generator) < t

    x_t = torch.where(mask, torch.full_like(x0, mask_token_id), x0)
    return x_t, mask, t


def diffusion_loss(model, x0, mask_token_id: int, attention_mask=None, t_eps: float = 1e-3,
                   stratified=False, antithetic=False, self_cond_p: float = 0.0):
    """Returns (scalar loss, stats dict). Supervises only masked, non-padding tokens.

    self_cond_p: probability of running the Loopholing self-conditioning second pass. The first
    pass is stop-gradded (so the model cannot game it by degrading pass 1 to make pass 2 look
    good) and only pass 2 carries gradient. p in [0.5, 0.9]; 0.9 was best on LM1B from scratch.
    Dropping the second pass with probability 1-p keeps the model able to denoise WITHOUT a
    self-conditioning signal, which is what the first sampling step always faces.
    """
    x_t, mask, t = forward_mask(x0, mask_token_id, t_eps, stratified=stratified,
                                antithetic=antithetic)
    if antithetic:
        x0 = x0.repeat(2, 1)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(2, 1)

    if attention_mask is not None:
        amask = attention_mask.bool()
        mask = mask & amask
        L = amask.sum(dim=1).clamp(min=1).float()                     # (B,)
    else:
        L = torch.full((x0.shape[0],), x0.shape[1], device=x0.device, dtype=torch.float32)

    if self_cond_p > 0.0 and torch.rand(()).item() < self_cond_p:
        with torch.no_grad():
            _, h = model(x_t, attention_mask, return_h=True)
        logits = model(x_t, attention_mask, h_prev=h.detach())        # gradient flows here only
    else:
        logits = model(x_t, attention_mask)                           # (B, T, V)

    logits = zero_mask_logit(logits, mask_token_id)
    B, T, V = logits.shape

    ce = F.cross_entropy(logits.view(B * T, V), x0.view(B * T), reduction="none").view(B, T)
    ce = ce * mask                                                    # zero out non-masked

    # per-example  (1/t) * (1/L) * sum_masked CE  ->  mean over batch. With antithetic masks the
    # batch is the two complementary halves and the mean over it IS the MIRROR estimator.
    per_ex = ce.sum(dim=1) / t.squeeze(1) / L
    loss = per_ex.mean()

    with torch.no_grad():
        denom = mask.sum().clamp(min=1)
        mlm_ce = ce.sum() / denom                                     # unweighted CE on masked (ppl proxy)
        stats = {
            "loss": loss.detach(),
            "mlm_ce": mlm_ce,
            "mlm_ppl": mlm_ce.exp(),
            "mask_frac": mask.float().mean().detach(),
        }
    return loss, stats


def zero_mask_logit(logits, mask_token_id: int):
    """Forbid the model from ever emitting [MASK]. This is the "zero-masking" half of MDLM's SUBS
    parameterization (arXiv 2406.07524, Table 8: zero-masking + carry-over is worth ~1.4 PPL on
    LM1B — 27.04 vs 28.56).

    We already had carry-over (the loss only supervises masked positions, and the sampler copies
    the rest through). We did NOT have this half, and in the sampler it was an outright BUG: [MASK]
    sits in the vocabulary as a real token, so top-k/top-p could select it and write it into the
    generated text as if it were a word.

    p(x0_i = MASK) is zero by construction — the forward process never turns a token INTO a mask in
    the data, the mask is the corruption. Letting the network waste probability mass there is
    strictly wasted capacity."""
    return logits.index_fill(-1, torch.tensor([mask_token_id], device=logits.device),
                             float("-inf"))


def _filter_logits(logits, temperature: float, top_k: int, top_p: float):
    """Temperature + top-k + nucleus (top-p) filtering on the last dim. Excluded
    tokens get -inf so they vanish after softmax. Always keeps at least one token."""
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = logits.topk(k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = (cum - sorted_logits.softmax(dim=-1)) > top_p   # keep through the token that crosses p
        remove_orig = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove_orig, float("-inf"))
    return logits


@torch.no_grad()
def generate(model, prompt_ids, gen_len: int, mask_token_id: int,
             steps: int = 64, temperature: float = 0.9, temp_end=None,
             top_k: int = 0, top_p: float = 0.9, block_size: int = 32,
             remask: bool = False):
    """Multi-stage diffusion sampler. REFERENCE IMPLEMENTATION / SPEC for the Rust
    engine (DESIGN.md: the production parallel decoder lives in a separate Rust crate;
    this mirrors its algorithm 1:1 so it can be ported directly). Usable as the eval
    sampler here too.

    Three stages:

      1. SEMI-AUTOREGRESSIVE BLOCKS. The gen region is split left-to-right into blocks
         of `block_size`; earlier blocks are frozen context while a later block decodes.
         This gives the model a stable left context (helps coherence over distance, the
         main weakness of fully-parallel any-order denoising). block_size >= gen_len =>
         a single fully-parallel block (pure non-AR diffusion, fastest, least coherent).

      2. ITERATIVE REFINEMENT WITH REMASKING. Within a block, each step the WHOLE block
         competes: the model re-predicts every position, the top-`n_keep` by confidence
         are kept and the rest are re-masked. `n_keep` grows on a cosine schedule
         0 -> block_len, so the block converges to fully revealed while early
         low-confidence tokens can still be overwritten on a later step — this is the
         self-correction the old commit-only sampler lacked (it got stuck in loops).
         remask=False (default) is monotonic commit-only (MaskGIT-style, no correction).
         CAVEAT measured here: remask RELIES ON CALIBRATED CONFIDENCE. On an undertrained
         model it amplifies miscalibrated scores and collapses onto high-frequency tokens
         (worse than monotonic). Turn it on only for a well-trained model; default off.

      3. SAMPLING. Per step: temperature (optionally annealed temperature->temp_end
         across the whole run via `temp_end`) + top-k + nucleus(top-p). temperature<=0
         => greedy. Annealing explores early (context mostly holes) and decides late.

    For higher quality at inference cost, wrap with generate_best_of (reranks N samples
    by the model's own pseudo-likelihood).
    """
    model.eval()
    device = prompt_ids.device
    B, P = prompt_ids.shape
    x = torch.cat(
        [prompt_ids, torch.full((B, gen_len), mask_token_id, device=device, dtype=prompt_ids.dtype)],
        dim=1,
    )

    blocks = [(P + s, min(P + s + block_size, P + gen_len)) for s in range(0, gen_len, block_size)]
    total_steps = steps * max(len(blocks), 1)
    gstep = 0

    for lo, hi in blocks:
        blen = hi - lo
        x[:, lo:hi] = mask_token_id
        for s in range(steps):
            temp_now = temperature if temp_end is None else \
                temperature + (temp_end - temperature) * (gstep / max(total_steps - 1, 1))
            gstep += 1

            blk = model(x).float()[:, lo:hi, :]                  # (B, blen, V)
            if temp_now <= 0.0:
                conf, cand = blk.softmax(dim=-1).max(dim=-1)     # (B, blen)
            else:
                probs = _filter_logits(blk, temp_now, top_k, top_p).softmax(dim=-1)
                cand = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).view(B, blen)
                conf = probs.gather(-1, cand.unsqueeze(-1)).squeeze(-1)

            ratio = (s + 1) / steps
            n_keep = blen if s == steps - 1 else blen - int(round(blen * math.cos(0.5 * math.pi * ratio)))
            n_keep = max(1, min(blen, n_keep))

            for b in range(B):
                if remask:
                    keep = conf[b].topk(n_keep).indices
                    newblk = torch.full((blen,), mask_token_id, device=device, dtype=x.dtype)
                    newblk[keep] = cand[b][keep]
                    x[b, lo:hi] = newblk
                else:
                    is_m = x[b, lo:hi] == mask_token_id
                    n_new = max(0, n_keep - int((~is_m).sum()))
                    if n_new > 0:
                        idx = conf[b].masked_fill(~is_m, -1.0).topk(n_new).indices
                        x[b, lo + idx] = cand[b][idx]
    return x


@torch.no_grad()
def _pseudo_nll(model, x, region_lo: int, mask_token_id: int, passes: int = 6, frac: float = 0.3):
    """Monte-Carlo pseudo-NLL of x[:, region_lo:] under the model's own training objective
    (random-mask a fraction, average CE on the masked tokens). Lower = the model finds the
    text more self-consistent. Same metric the model was trained on, so a fair reranker."""
    region = x[:, region_lo:]
    tot, cnt = 0.0, 0
    for _ in range(passes):
        m = torch.rand(region.shape, device=x.device) < frac
        if not m.any():
            continue
        xt = x.clone()
        xt[:, region_lo:] = torch.where(m, torch.full_like(region, mask_token_id), region)
        lg = model(xt).float()[:, region_lo:, :]
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), region.reshape(-1), reduction="none").view(m.shape)
        tot += (ce * m).sum().item()
        cnt += int(m.sum())
    return tot / max(cnt, 1)


@torch.no_grad()
def generate_best_of(model, prompt_ids, gen_len: int, mask_token_id: int,
                     n: int = 4, rerank_passes: int = 6, sampler=None,
                     rep_weight: float = 5.0, **kw):
    """Generate n candidates and return the best by pseudo-NLL + rep_weight * rep4.
    Pseudo-NLL ALONE is a trap, measured on the 341M: the model assigns high likelihood
    to degenerate repetition, so a pure-likelihood reranker actively PREFERS loops
    ("valori valori valori ..."). The rep4 term vetoes them. `sampler` picks the decoder
    (default `generate`; pass `generate_parallel` for the loop-free one)."""
    P = prompt_ids.shape[1]
    sampler = sampler or generate
    best_x, best = None, float("inf")
    for _ in range(n):
        cand = sampler(model, prompt_ids, gen_len, mask_token_id, **kw)
        score = _pseudo_nll(model, cand, P, mask_token_id, passes=rerank_passes) \
            + rep_weight * repetition_rate(cand[0, P:].tolist(), 4)
        if score < best:
            best, best_x = score, cand
    return best_x


@torch.no_grad()
def generate_parallel(model, prompt_ids, gen_len: int, mask_token_id: int,
                      steps: int = 256, temperature: float = 0.9, temp_end: float = 0.4,
                      top_p: float = 0.92, remask: bool = True,
                      rep_penalty: float = 1.15, no_repeat_ngram: int = 3):
    """Fully-parallel confidence decoder with self-correction and repetition control.

    This is the decoder that actually generates coherent text; `generate` above (semi-AR
    blocks, commit-only) is the Rust-engine spec and the loop-prone baseline. Free
    generation on a well-trained model collapses into loops ("uno stato uno stato ...")
    for three decoding reasons, all fixed here:

      1. semi-AR blocks self-poison: commit an early block, later blocks are conditioned
         on it and continue the pattern. -> no blocks; the whole span competes and the
         globally most-confident position is revealed first.
      2. commit-only has no self-correction, a bad token is permanent. -> remask=True.
      3. nothing penalizes repetition, copying context is locally optimal everywhere at
         once. -> per-token repetition penalty + hard no-repeat-ngram.

    The loop is a decoding fault, not undertraining: the same model infills real masked
    sentences at ~50% top-1. Turn remask off only for a genuinely undertrained model
    (miscalibrated confidence)."""
    device = prompt_ids.device
    B, P = prompt_ids.shape
    x = torch.cat([prompt_ids,
                   torch.full((B, gen_len), mask_token_id, device=device, dtype=prompt_ids.dtype)],
                  dim=1)
    lo, hi = P, P + gen_len

    for s in range(steps):
        t = s / max(steps - 1, 1)
        temp = temperature + (temp_end - temperature) * t
        logits = model(x).float()[:, lo:hi, :]                        # (B, gen_len, V)

        # repetition control. penalty is sign-aware (CTRL, Keskar 2019): dividing a
        # negative logit would RAISE its prob, so divide positives / multiply negatives.
        if rep_penalty and rep_penalty != 1.0:
            for b in range(B):
                seen = torch.unique(x[b, :hi][x[b, :hi] != mask_token_id])
                sl = logits[b, :, seen]
                logits[b, :, seen] = torch.where(sl > 0, sl / rep_penalty, sl * rep_penalty)
        if no_repeat_ngram:
            n = no_repeat_ngram
            for b in range(B):
                seq = x[b, :hi].tolist()
                banned = {}
                for i in range(len(seq) - n):
                    if mask_token_id in seq[i:i + n]:
                        continue
                    banned.setdefault(tuple(seq[i:i + n - 1]), set()).add(seq[i + n - 1])
                for pos in range(gen_len):
                    ctx = tuple(x[b, lo + pos - (n - 1):lo + pos].tolist())
                    if len(ctx) == n - 1 and mask_token_id not in ctx and ctx in banned:
                        logits[b, pos, list(banned[ctx])] = -float("inf")

        if temp <= 0:
            probs = logits.softmax(-1)
            cand = probs.argmax(-1)
        else:
            probs = _filter_logits(logits, temp, 0, top_p).softmax(-1)
            cand = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).view(B, gen_len)

        # confidence = top1-top2 MARGIN of the raw distribution, not p(cand). Raw prob on a
        # miscalibrated model rewards high-frequency tokens (the remask collapse mode);
        # margin measures certainty RELATIVE to the alternatives at that position.
        top2 = logits.softmax(-1).topk(2, dim=-1).values                # (B, gen_len, 2)
        conf = top2[..., 0] - top2[..., 1]

        # Gumbel noise on the reveal ORDER (LLaDA-style), annealed to 0: deterministic
        # order repeats the same systematic path every run; early stochastic order explores.
        tau = 1.0 - t
        if tau > 0:
            u = torch.rand_like(conf).clamp_(1e-9, 1 - 1e-9)
            conf = conf + tau * 0.1 * (-torch.log(-torch.log(u)))

        n_keep = gen_len if s == steps - 1 else \
            gen_len - int(round(gen_len * math.cos(0.5 * math.pi * (s + 1) / steps)))
        n_keep = max(1, min(gen_len, n_keep))

        for b in range(B):
            if remask:
                keep = conf[b].topk(n_keep).indices
                row = torch.full((gen_len,), mask_token_id, device=device, dtype=x.dtype)
                row[keep] = cand[b][keep]
                x[b, lo:hi] = row
            else:
                is_m = x[b, lo:hi] == mask_token_id
                n_new = max(0, n_keep - int((~is_m).sum()))
                if n_new:
                    idx = conf[b].masked_fill(~is_m, -1.0).topk(n_new).indices
                    x[b, lo + idx] = cand[b][idx]
    return x


def repetition_rate(ids, n=4):
    """Fraction of n-grams that repeat. 0 = none, ->1 = degenerate loop. The number to
    report alongside samples: it makes "it loops" measurable instead of anecdotal."""
    grams = [tuple(ids[i:i + n]) for i in range(max(0, len(ids) - n + 1))]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def _demo():
    """Smoke check for the sampler: shapes correct, no leftover masks, rerank runs."""
    import torch as _t
    from model.configs import config_smoke
    from model.arch import TernaryDiffusionLM
    cfg = config_smoke()
    m = TernaryDiffusionLM(cfg).eval()
    prompt = _t.randint(0, cfg.mask_token_id, (1, 4))
    for kw in ({"block_size": 8, "remask": True}, {"block_size": 999, "remask": False}, {"temperature": 0.0}):
        out = generate(m, prompt, gen_len=24, mask_token_id=cfg.mask_token_id, steps=12, **kw)
        assert out.shape == (1, 28), out.shape
        assert (out[:, 4:] == cfg.mask_token_id).sum() == 0, "leftover masks"
    bo = generate_best_of(m, prompt, gen_len=24, mask_token_id=cfg.mask_token_id, n=3, steps=12, block_size=8)
    assert bo.shape == (1, 28) and (bo[:, 4:] == cfg.mask_token_id).sum() == 0

    # parallel decoder: shapes, no leftover masks, both rep-control branches run
    pr = generate_parallel(m, prompt, gen_len=24, mask_token_id=cfg.mask_token_id,
                           steps=12, rep_penalty=1.15, no_repeat_ngram=3)
    assert pr.shape == (1, 28) and (pr[:, 4:] == cfg.mask_token_id).sum() == 0
    assert repetition_rate([1, 1, 1, 1], 1) == 0.75 and repetition_rate([1, 2, 3], 1) == 0.0
    print("diffusion sampler demo OK")


if __name__ == "__main__":
    _demo()
