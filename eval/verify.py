"""The checks every arm depends on. Run this BEFORE spending a cent of GPU.

    python verify.py

Two classes of check:

  IDENTITY — with every knob off, the V2 model must be BIT-IDENTICAL to the baseline, and the
  generalized quantizer must reduce EXACTLY to BitNet's round(clip(w/absmean)). Without this, an
  improvement measured later is unattributable: it could be a refactor artifact.

  NOT-A-NO-OP — every knob must actually change something. A flag that silently does nothing would
  report "no improvement" forever and we would conclude the technique does not work, when in fact
  we never ran it. This is the failure mode that quietly wastes a whole bake-off.

Everything here runs on a toy model in seconds. It is not a training run.
"""
import torch

from model import arch as B
from train.diffusion import diffusion_loss, forward_mask, sample_t, zero_mask_logit
from model.mitigations import MitigatedDiffusionLM
from model.quant import MitigationConfig, QATConfig, _quant_ternary


def cfg_small():
    return B.TernaryDiffusionConfig(vocab_size=512, mask_token_id=511, d_model=64,
                                    n_layers=2, n_heads=4, d_ff=128, max_seq_len=32)


def check_quantizer():
    """The generalized ternarizer must reduce to BitNet exactly at thresh=0.5, twn_alpha=False."""
    w = torch.randn(128, 64)
    gamma = w.abs().mean().clamp(min=1e-5)
    bitnet = (w / gamma).round().clamp(-1, 1) * gamma
    ours = _quant_ternary(w, 1e-5, per_channel=False, qat=QATConfig())
    d = (bitnet - ours).abs().max().item()
    print(f"quantizer: max |bitnet - ours(defaults)| = {d:.3e}")
    assert d == 0.0, "the default quantizer is NOT BitNet; the baseline arm is not a baseline"

    # thresh raises the zero bin => strictly more zeros. If it does not, the knob is broken.
    z = {t: (_quant_ternary(w, 1e-5, False, QATConfig(thresh=t)) == 0).float().mean().item()
         for t in (0.4, 0.5, 0.7)}
    print(f"quantizer: zero_frac by threshold {[(t, round(v, 3)) for t, v in z.items()]}")
    assert z[0.4] < z[0.5] < z[0.7], "threshold does not control sparsity"

    # TWN's alpha is the mean over SURVIVORS, so it must exceed the absmean over ALL weights
    # (which is dragged down by the ones about to be zeroed). This is the whole point of it.
    q = _quant_ternary(w, 1e-5, False, QATConfig(thresh=0.5, twn_alpha=True))
    alpha = q.abs().max().item()
    print(f"quantizer: twn alpha {alpha:.4f} vs absmean {gamma.item():.4f}")
    assert alpha > gamma.item(), "twn_alpha is not doing what it claims"


def check_zero_mask_logit():
    mask_id = 511
    lg = torch.randn(2, 8, 512)
    out = zero_mask_logit(lg, mask_id)
    assert torch.isinf(out[..., mask_id]).all() and (out[..., mask_id] < 0).all()
    assert (out[..., :mask_id] == lg[..., :mask_id]).all(), "zero_mask_logit touched other logits"
    p = out.softmax(-1)
    print(f"zero-mask: p(MASK) = {p[..., mask_id].max().item():.1e} (was "
          f"{lg.softmax(-1)[..., mask_id].max().item():.3f})")
    assert p[..., mask_id].max().item() == 0.0, "[MASK] is still sampleable — the sampler bug lives"


def check_t_sampling():
    """Stratified t must stay UNBIASED (uniform marginal) while cutting the spread of the batch
    mean. If it biased the marginal it would silently change the objective."""
    torch.manual_seed(0)
    d = "cpu"
    n = 4000
    means_i, means_s = [], []
    for _ in range(n):
        means_i.append(sample_t(32, d).mean().item())
        means_s.append(sample_t(32, d, stratified=True).mean().item())
    mi, ms = torch.tensor(means_i), torch.tensor(means_s)
    print(f"t-sampling: E[t] iid {mi.mean():.4f} strat {ms.mean():.4f} (both must be ~0.5)")
    print(f"t-sampling: std of batch-mean  iid {mi.std():.4f} -> strat {ms.std():.4f} "
          f"({mi.std()/ms.std():.1f}x tighter)")
    assert abs(ms.mean() - 0.5) < 0.01, "stratified t is BIASED — it changed the objective"
    assert ms.std() < mi.std() / 2, "stratified t did not reduce variance"


def check_antithetic():
    """Complementary masks: mask1 = {u<t}, mask2 = {u>1-t}. Both must have the right marginal rate
    t, and their overlap must be smaller than two independent draws would give (that negative
    correlation IS the variance reduction)."""
    torch.manual_seed(0)
    x0 = torch.randint(0, 500, (256, 64))
    _, m, t = forward_mask(x0, 511, antithetic=True)
    b = x0.shape[0]
    m1, m2 = m[:b], m[b:]
    rate = m.float().mean(dim=1)
    print(f"antithetic: shapes {tuple(m.shape)} | mask rate vs t: corr "
          f"{torch.corrcoef(torch.stack([rate, t.squeeze(1)]))[0,1]:.3f}")

    # The right yardstick is PER-EXAMPLE, not global. Two INDEPENDENT masks at the same rate t
    # overlap by E[t^2] = 1/3. Our complementary pair can only overlap where t > 0.5, giving
    # E[max(2t-1, 0)] = 1/4. Comparing against the global-mean product (0.5*0.5) instead would
    # yield ~0.25 vs ~0.27 — a difference so small the test would pass even with MIRROR broken.
    tt = t[:b].squeeze(1)
    overlap = (m1 & m2).float().mean().item()
    indep = (tt ** 2).mean().item()                       # E[t^2], the independent-pair overlap
    theory = torch.clamp(2 * tt - 1, min=0).mean().item()  # E[(2t-1)+], the antithetic overlap
    print(f"antithetic: overlap {overlap:.4f} | theory {theory:.4f} | independent pair {indep:.4f}")
    assert abs(overlap - theory) < 0.02, "the complementary masks are not what the math says"
    assert overlap < indep - 0.05, "the two masks are not negatively correlated — MIRROR is dead"

    # And the payoff itself: the mask-pattern noise in the per-example mask RATE must drop.
    # This is the component of the variance that stratifying t cannot touch.
    dev_anti = ((m1.float().mean(1) + m2.float().mean(1)) / 2 - tt).abs().mean().item()
    u1 = torch.rand(b, 64) < tt[:, None]
    u2 = torch.rand(b, 64) < tt[:, None]
    dev_iid = ((u1.float().mean(1) + u2.float().mean(1)) / 2 - tt).abs().mean().item()
    print(f"antithetic: |realized rate - t|  antithetic {dev_anti:.4f} vs iid {dev_iid:.4f} "
          f"({dev_iid/max(dev_anti,1e-9):.1f}x tighter)")
    assert dev_anti < dev_iid, "antithetic pairs do not reduce mask-pattern noise"


def main():
    torch.manual_seed(0)
    cfg = cfg_small()

    torch.manual_seed(0)
    base = B.TernaryDiffusionLM(cfg).eval()
    torch.manual_seed(0)
    v2 = MitigatedDiffusionLM(cfg, MitigationConfig(), QATConfig()).eval()   # everything OFF
    v2.load_state_dict(base.state_dict(), strict=True)

    x = torch.randint(0, 500, (2, 32))
    with torch.no_grad():
        a, b = base(x), v2(x)
    d = (a - b).abs().max().item()
    print(f"identity: all knobs off, max |base - v2| = {d:.3e}")
    assert d == 0.0, "V2 is NOT identical to the baseline; every later number is meaningless"
    print("OK: identical.\n")

    check_quantizer()
    print()
    check_zero_mask_logit()
    print()
    check_t_sampling()
    print()
    check_antithetic()
    print()

    # Self-conditioning: the LayerNorm gain is zero-initialized, so at step 0 the conditioned pass
    # must be IDENTICAL to the unconditioned one. The arm starts on the baseline and the technique
    # has to earn its way in — same discipline as the zero-init low-rank factors.
    torch.manual_seed(0)
    sc = MitigatedDiffusionLM(cfg, MitigationConfig(), QATConfig(), self_cond=True).eval()
    sc.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        lg0, h = sc(x, return_h=True)
        lg1 = sc(x, h_prev=h)
    d = (lg0 - lg1).abs().max().item()
    print(f"self-cond: |pass1 - pass2| at init = {d:.3e} (zero-init gain => must be 0)")
    assert d == 0.0, "self-conditioning does not start at the baseline"
    # ...but it must be ABLE to change the output, or the whole path is dead weight.
    with torch.no_grad():
        sc.sc_norm.weight.fill_(1.0)
        lg2 = sc(x, h_prev=h)
    changed = (lg0 - lg2).abs().max().item()
    print(f"self-cond: with gain=1, output moves by {changed:.3e} (must be > 0)")
    assert changed > 0, "the self-conditioning path is dead — h_prev never reaches the model"

    # Every QAT knob must move the output. A silent no-op would report "no gain" forever.
    print()
    for name, q in [
        ("thresh=0.7",  QATConfig(thresh=0.7)),
        ("twn_alpha",   QATConfig(twn_alpha=True)),
        ("absmedian",   QATConfig(absmedian=True)),
        ("chan_scale",  QATConfig(chan_scale=True)),
    ]:
        torch.manual_seed(0)
        m = MitigatedDiffusionLM(cfg, MitigationConfig(), q).eval()
        m.load_state_dict(base.state_dict(), strict=False)
        with torch.no_grad():
            out = m(x)
        ch = (out - a).abs().max().item()
        print(f"  {name:12} changes output by {ch:.3e} | zero_frac {m.zero_frac():.3f}")
        if name == "chan_scale":
            assert ch == 0.0, "chan_scale must start at 1.0 (identity)"
        else:
            assert ch > 0, f"{name} is a NO-OP"

    # Dampening must be exactly zero when off, and positive when on, or the ramp is meaningless.
    torch.manual_seed(0)
    dm = MitigatedDiffusionLM(cfg, MitigationConfig(), QATConfig(dampen=1e-3))
    dl = dm.dampen_loss().item()
    off = MitigatedDiffusionLM(cfg, MitigationConfig(), QATConfig()).dampen_loss().item()
    print(f"\ndampen: loss on {dl:.4e} | off {off:.4e} (off must be exactly 0)")
    assert off == 0.0 and dl > 0, "dampening loss is broken"

    # The loss must run end to end with everything on at once, and stay finite.
    torch.manual_seed(0)
    full = MitigatedDiffusionLM(cfg, MitigationConfig(),
                                QATConfig(thresh=0.6, twn_alpha=True, chan_scale=True, dampen=1e-3),
                                self_cond=True)
    x0 = torch.randint(0, 500, (8, 32))
    loss, st = diffusion_loss(full, x0, cfg.mask_token_id, stratified=True, antithetic=True,
                              self_cond_p=1.0)
    loss = loss + full.dampen_loss()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0).item()
    print(f"end-to-end (everything on): loss {loss.item():.4f} | ppl {st['mlm_ppl'].item():.1f} "
          f"| grad_norm {gn:.4f}")
    assert torch.isfinite(loss) and gn > 0, "the full stack does not train"
    assert full.sc_norm.weight.grad is not None, "no gradient reaches the self-conditioning path"

    print("\nALL CHECKS PASS. The knobs are real, the baseline is a baseline, and the "
          "objective is unbiased.")


if __name__ == "__main__":
    main()
