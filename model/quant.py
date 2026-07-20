"""BitLinear with the mitigations, each one motivated by a measurement from the baseline run
(see ../ECHO-DIFF-TERNARY/data/results/). Nothing here is speculative decoration: every knob
exists because a number in that directory says it might pay, and the numbers also say which ones
probably will not.

WHAT THE BASELINE MEASURED, AND WHAT IT IMPLIES

  1. QAT concentrates the quantization residual. Across all 56 quantized matrices, the first
     singular direction of W - ternary(W) carries 17.5% of the residual energy on QAT weights vs
     8.4% on FP16 weights. In attn.o_proj it reaches 37.4%.
     => a LOW-RANK residual correction should recover a real share of the gap, and rank can be
        allocated unevenly: o_proj deserves more than ffn.up.

  2. The residual is ALSO outlier-heavy: the largest 1% of entries carry 17-31% of the squared
     error (a Gaussian matrix would give ~7.6%).
     => a SPARSE FP16 outlier set is a competing design. Which of the two wins is empirical, and
        this file lets you run both.

  3. Per-channel scales barely help: 4.6% reconstruction-error reduction on FP16 weights, 10.9%
     on QAT weights. The reason is SubLN, which already conditions the channels.
     => included because it is nearly free (one fp16 vector per matrix, ~0.2% memory), but do not
        expect much. This is a prediction the bake-off should falsify or confirm.

  4. The ternary-FP16 gap GROWS with scale (8.6% at 27M; tracking above 12% at 341M at 18% of
     training). If pure ternary does not reach 1B within the gate, mitigations stop being
     optional and become the plan.

  5. We train an FP16 twin anyway, for the gate. It is a free teacher.
     => DISTILLATION costs nothing we are not already paying. In the QAT literature this is the
        single most reliable gap-closer (BitDistiller). It is implemented in train_v2.py.

MEMORY IS THE CONSTRAINT, AND IT IS TIGHTER THAN IT LOOKS. Ternary weights cost 0.2 bytes per
parameter. Anything stored at fp16 costs 2 bytes: 10x more per parameter. A rank-16 correction
across a 341M model is 7.6M parameters = 14.4 MB at fp16, which is +25% on top of 58.8 MB of
ternary weights. So the residual must itself be quantized, and the interesting axis is not
fp16-vs-fp8 but RANK VERSUS PRECISION AT A FIXED BYTE BUDGET: rank-16@fp16, rank-32@int8 and
rank-64@int4 all cost the same. int8 is simulated here (fake-quant, as with the weights).
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.arch import RMSNorm, TernaryDiffusionConfig, strip_compile_prefix  # noqa: F401


@dataclass
class MitigationConfig:
    """Off by default: with everything false, BitLinearV2 must be bit-identical to the baseline.
    That is asserted in bakeoff.py, and it is the only way to trust any measured improvement."""
    per_channel_scale: bool = False   # one scale per output row instead of per tensor
    lowrank: int = 0                  # rank r of an additive residual A@B (0 = off)
    lowrank_bits: int = 16            # 16 = fp16, 8 = int8 fake-quant (halves the cost)
    outliers: float = 0.0             # fraction of weights kept in fp16 (e.g. 0.005 = 0.5%)


@dataclass
class QATConfig:
    """Knobs that change how the ternary grid is REACHED during training, as opposed to
    MitigationConfig, which adds inference-time capacity. Defaults reproduce BitNet b1.58 exactly.

    WHY THESE AND NOT THE OTHERS. "Custom Gradient Estimators are STEs in Disguise"
    (arXiv 2405.05171) proves that a broad class of custom weight-gradient estimators — EWGS,
    ReSTE, DSQ — is EQUIVALENT to plain STE under an adaptive optimizer, because Adam's
    per-coordinate normalization divides out exactly the elementwise multiplicative rescaling
    those methods apply. We train with AdamW. So the entire "better STE" literature is expected
    to buy us nothing, and none of it is implemented here.

    What survives that argument is what Adam CANNOT absorb: things that are ADDITIVE to the
    gradient (dampen), that change the FORWARD (thresh, twn_alpha, absmedian, chan_scale), or
    that change the LOSS (distillation, in train_v2.py). That is the organizing principle of
    this config, and it is the reason it is short.
    """
    thresh: float = 0.5      # zero-bin half-width, in units of the scale. 0.5 == round() == BitNet.
                             # TWN (arXiv 1605.04711) argues 0.7*E|w|. Gaussian w: 0.5 gives ~25%
                             # zeros, 0.7 gives ~42%. Nobody has ablated this on an LM. Free to try,
                             # and more zeros is also free inference speed for a ternary kernel.
    twn_alpha: bool = False  # scale = mean|w| over the SURVIVING (nonzero) weights only, which is
                             # the L2-optimal ternary fit (TWN). BitNet's absmean includes the
                             # weights it is about to send to zero, which biases the scale down.
    absmedian: bool = False  # median|w| instead of mean|w| (arXiv 2407.09527). Less outlier-
                             # sensitive. CAUTION: median < mean for heavy-tailed w => LARGER
                             # effective scale => FEWER zeros. Log zero_frac when you A/B this,
                             # or you will confound the threshold with the estimator.
    chan_scale: bool = False # learnable per-output-channel multiplier applied AFTER the matmul.
                             # This is the safe half of LSQ: it cannot move the zero bin (it is
                             # outside the quantizer), so the optimizer cannot cheat by sparsifying
                             # the layer to reduce loss noise. Init 1.0 => starts at baseline.
    clip_ste: bool = False   # CLIPPED STE: zero the gradient of latent weights outside [-a, +a].
                             # Our default (False) is PURE STE — gradient 1 everywhere — which is
                             # what `w + (wq - w).detach()` actually implements, and what BitNet
                             # ships despite the name RoundClip.
                             #
                             # This flag exists because of a theorem that does NOT transfer. arXiv
                             # 2405.05171 proves custom estimators collapse to plain STE under an
                             # ADAPTIVE optimizer — the proof runs on Adam's invariance to
                             # PER-COORDINATE rescaling. MUON IS NOT PER-COORDINATE. It is invariant
                             # to a global scalar, not to a coordinate mask. So under Muon the
                             # estimator is NOT absorbed, and the clip/no-clip choice becomes a
                             # first-class hyperparameter again. We closed that question on Adam's
                             # authority; Muon reopens it, and bakeoff_muon.sh settles it.
                             #
                             # WARNING: clipped STE + Muon is exactly where the Newton-Schulz hazard
                             # lives (see muon.py). The mask is stashed on the parameter so the
                             # optimizer can re-apply it AFTER orthogonalization.
    dampen: float = 0.0      # lambda on the oscillation-dampening penalty (Nagel et al., ICML 2022,
                             # arXiv 2203.11086). Pulls latent weights toward their bin CENTRE, so
                             # they stop flapping across a decision boundary. Additive => Adam
                             # cannot absorb it. Ramp it 0 -> lambda (constant lambda over-
                             # regularizes; the paper is explicit).
                             # HONESTY: their headline numbers are 3/4-bit CNNs and roughly HALF the
                             # gain comes from BatchNorm statistics being corrupted by oscillation.
                             # We have no BatchNorm. Discount accordingly, and only turn this on
                             # after osc_frac() shows the oscillation is actually there.


def _scale(w, eps, per_channel, absmedian=False):
    """gamma: the absmean (or absmedian) scale. Per-tensor by default, per-output-row optionally."""
    a = w.abs()
    if per_channel:
        s = a.median(dim=1, keepdim=True).values if absmedian else a.mean(dim=1, keepdim=True)
    else:
        s = a.median() if absmedian else a.mean()
    return s.clamp(min=eps)


def _quant_ternary(w, eps, per_channel, qat: "QATConfig" = None):
    """Returns the dequantized ternary weight. With QATConfig defaults this is exactly
    round(clip(w/absmean, -1, 1)) * absmean, i.e. BitNet b1.58, bit for bit."""
    q = qat or QATConfig()
    gamma = _scale(w, eps, per_channel, q.absmedian)

    if q.thresh == 0.5 and not q.twn_alpha:
        return (w / gamma).round().clamp(-1, 1) * gamma        # the baseline path, untouched

    tern = torch.where(w.abs() > q.thresh * gamma, w.sign(), torch.zeros_like(w))  # {-1,0,+1}
    if q.twn_alpha:
        nz = tern.abs()
        if per_channel:
            alpha = (w.abs() * nz).sum(dim=1, keepdim=True) / nz.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            alpha = (w.abs() * nz).sum() / nz.sum().clamp(min=1)
        alpha = alpha.clamp(min=eps)
    else:
        alpha = gamma
    return tern * alpha


def _fake_quant_int(x, bits):
    """Symmetric per-tensor int fake-quant. Used for the low-rank factors, which are small but
    stored at high precision would dominate the memory budget."""
    if bits >= 16:
        return x
    n = 2 ** (bits - 1) - 1
    s = x.abs().amax().clamp(min=1e-8) / n
    return ((x / s).round().clamp(-n, n) * s)


class BitLinearV2(nn.Module):
    """Ternary weights + optional corrections. STE throughout, exactly as the baseline."""

    def __init__(self, in_features, out_features, eps=1e-5, act_levels=127, ternary=True,
                 mit: MitigationConfig = None, qat: QATConfig = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.norm = RMSNorm(in_features, eps)
        self.eps, self.act_levels, self.ternary = eps, act_levels, ternary
        self.mit = mit or MitigationConfig()
        self.qat = qat or QATConfig()

        r = self.mit.lowrank
        if r > 0:
            # init at zero so the model starts exactly at the ternary baseline and the correction
            # has to earn its way in. A random init would make an improvement unattributable.
            self.lr_a = nn.Parameter(torch.zeros(out_features, r))
            self.lr_b = nn.Parameter(torch.randn(r, in_features) * 0.02)

        if self.qat.chan_scale:
            # 1.0 => identity at step 0. The arm starts on the baseline and has to earn the change.
            self.cscale = nn.Parameter(torch.ones(out_features))

        # Oscillation instrumentation. int8 holds {-1,0,+1} exactly; one byte per weight, and it is
        # the cheapest way to answer the question that decides whether `dampen` is worth anything.
        # Buffers, not parameters: no grad, and they ride along in the checkpoint for free.
        self.register_buffer("_q_prev", torch.zeros(out_features, in_features, dtype=torch.int8),
                             persistent=False)
        self.register_buffer("_flips", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("_seen", torch.zeros((), dtype=torch.long), persistent=False)
        self._track = False        # off by default: the int8 compare is cheap but not free

    @torch.no_grad()
    def _track_osc(self, tern):
        """Fraction of ternary weights that CHANGED BIN since the last tracked step. This is the
        number that decides whether oscillation dampening is worth its complexity: if late in
        training <1% of weights are flipping, there is nothing to dampen and we drop the technique.
        (Nagel's stricter definition also requires the flip to REVERSE the previous change; this
        looser one is an upper bound, which is the right side to err on for a go/no-go.)"""
        q = tern.to(torch.int8)
        self._flips += (q != self._q_prev).sum()
        self._seen += q.numel()
        self._q_prev.copy_(q)

    def osc_frac(self, reset=True):
        f = (self._flips.float() / self._seen.clamp(min=1)).item()
        if reset:
            self._flips.zero_(); self._seen.zero_()
        return f

    def zero_frac(self):
        """Sparsity of the ternary grid. Must be logged whenever thresh/absmedian/twn_alpha move —
        those three all shift it, and a change in sparsity confounds every other comparison."""
        with torch.no_grad():
            g = _scale(self.weight, self.eps, self.mit.per_channel_scale, self.qat.absmedian)
            return (self.weight.abs() <= self.qat.thresh * g).float().mean().item()

    def dead_frac(self):
        """Fraction of latent weights that have ESCAPED the representable range |w| > alpha.

        This is the diagnostic for the Muon/Newton-Schulz hazard, and it is the one number that can
        tell you your model is dying while the loss still looks fine. A weight out here contributes
        a saturated +/-1 and, under a clipped STE, receives no gradient — so it can never come back.
        If this climbs monotonically, orthogonalization is deleting the clip mask (see muon.py)."""
        with torch.no_grad():
            a = _scale(self.weight, self.eps, self.mit.per_channel_scale, self.qat.absmedian)
            return (self.weight.abs() > a).float().mean().item()

    def wnorm_over_alpha(self):
        """||W||_F / alpha. The bin pattern depends only on W/alpha, so this ratio IS the state of
        the quantizer. Under Muon with no weight decay it grows like sqrt(t), which means the
        effective learning rate on the bin pattern decays like 1/sqrt(t) and the bins quietly stop
        flipping halfway through training. Nothing in the loss curve will tell you that happened."""
        with torch.no_grad():
            a = _scale(self.weight, self.eps, self.mit.per_channel_scale, self.qat.absmedian)
            return (self.weight.norm() / a.mean().clamp(min=self.eps)).item()

    def dampen_loss(self):
        """Nagel et al. ICML 2022, Eq. 7:  || w_hat - clip(w, -alpha, +alpha) ||^2_F, with the bin
        centres w_hat DETACHED. Gradient is 2*(w - w_hat) inside the grid: a spring pulling each
        latent weight to the centre of the bin it currently occupies, so it stops flapping across
        the decision boundary. Zero outside the grid (those weights are already committed).

        This is ADDITIVE to the gradient, which is exactly why it is here and EWGS is not: Adam's
        per-coordinate normalization cannot divide it out (arXiv 2405.05171)."""
        if not self.ternary or self.qat.dampen <= 0:
            return self.weight.sum() * 0.0        # keeps the graph type-stable under torch.compile
        w_hat = _quant_ternary(self.weight, self.eps, self.mit.per_channel_scale, self.qat).detach()
        alpha = w_hat.abs().amax().clamp(min=self.eps)
        return (w_hat - self.weight.clamp(-alpha, alpha)).pow(2).sum()

    def _outlier_mask(self, w, wq):
        """Keep the k entries whose ternarization error is worst, in full precision."""
        f = self.mit.outliers
        if f <= 0:
            return None
        k = max(1, int(f * w.numel()))
        err = (w - wq).abs().flatten()
        idx = err.topk(k).indices
        m = torch.zeros_like(err, dtype=torch.bool)
        m[idx] = True
        return m.view_as(w)

    def forward(self, x):
        x = self.norm(x)
        if not self.ternary:
            return F.linear(x, self.weight)

        w_full = self.weight
        wq = _quant_ternary(w_full, self.eps, self.mit.per_channel_scale, self.qat)

        if self._track and self.training:
            self._track_osc(torch.sign(wq))

        om = self._outlier_mask(w_full, wq)
        if om is not None:
            wq = torch.where(om, w_full, wq)          # outliers escape the grid entirely

        if self.qat.clip_ste:
            # Gradient only for latent weights still INSIDE the representable range. A weight that
            # has already escaped should not be pushed further out — from out there its bin can
            # never flip back, and it is dead for the rest of training.
            with torch.no_grad():
                a = _scale(w_full, self.eps, self.mit.per_channel_scale, self.qat.absmedian)
                mask = (w_full.abs() <= a).to(w_full.dtype)
                # Stashed for Muon: Newton-Schulz mixes rows and columns, so it hands a full-size
                # update to coordinates whose gradient we just zeroed. The optimizer re-applies this
                # AFTER orthogonalization, or the guard above is decorative.
                self.weight._ste_mask = mask
            w = w_full * mask + (wq - w_full * mask).detach()      # d(w)/d(w_full) = mask
        else:
            w = w_full + (wq - w_full).detach()                    # pure STE: gradient 1 everywhere

        # activations: per-token int8, unchanged from the baseline
        xs = x.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps) / self.act_levels
        x_q = (x / xs).round().clamp(-self.act_levels, self.act_levels) * xs
        x = x + (x_q - x).detach()

        out = F.linear(x, w)

        if self.mit.lowrank > 0:
            a = _fake_quant_int(self.lr_a, self.mit.lowrank_bits)
            b = _fake_quant_int(self.lr_b, self.mit.lowrank_bits)
            a = self.lr_a + (a - self.lr_a).detach()
            b = self.lr_b + (b - self.lr_b).detach()
            out = out + F.linear(F.linear(x, b), a)    # x @ B^T @ A^T

        if self.qat.chan_scale:
            out = out * self.cscale                    # free scale, OUTSIDE the quantizer
        return out

    def extra_memory_bytes(self):
        """What the mitigations cost on top of the ternary weights, in bytes. The ternary weights
        themselves are numel/5 (5 trits per byte)."""
        n = 0
        if self.qat.chan_scale:
            n += self.weight.shape[0] * 2                      # one fp16 multiplier per row
        if self.mit.per_channel_scale:
            n += self.weight.shape[0] * 2                      # one fp16 scale per row
        if self.mit.lowrank > 0:
            p = self.lr_a.numel() + self.lr_b.numel()
            n += p * (self.mit.lowrank_bits / 8)
        if self.mit.outliers > 0:
            k = int(self.mit.outliers * self.weight.numel())
            n += k * (2 + 4)                                   # fp16 value + int32 index
        return n
