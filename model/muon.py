"""Muon (Keller Jordan) — momentum orthogonalized by Newton-Schulz — adapted for ternary QAT.

WHY MUON MIGHT BE THE RIGHT OPTIMIZER FOR A TERNARY NET, AND NOT MERELY A FASTER ONE

In a ternary net the latent weight is NOT a weight. It is a BIN SELECTOR: the forward pass only
ever sees sign(W) and a scale. ("Latent Weights Do Not Exist", arXiv 1906.02107 — the paper that
had to invent a whole new optimizer, Bop, because Adam and SGD are built for real weights and
behave badly on binarized nets.)

Look at what the two optimizers do to a bin selector:

  ADAM divides by sqrt(v) per coordinate. A coordinate with a consistently TINY gradient still gets
  a step of size ~lr. So Adam ERASES the relative magnitudes across coordinates — which is exactly
  the information that says WHICH BINS DESERVE TO FLIP. This is the documented mechanism of the QAT
  oscillation pathology, and it is worst at the lowest bit-widths. We are at 1.58.

  MUON normalizes per MATRIX (all singular values -> ~1) and PRESERVES relative magnitudes inside
  it: a genuinely weak coordinate gets a genuinely small step. For a bin selector that is the right
  inductive bias — flip only on strong, consistent evidence. It is much closer in spirit to Bop
  than Adam is.

Be clear about what does NOT survive: Muon's HEADLINE property is a statement about dW as a linear
operator (bounded spectral norm, no collapse into few directions). That semantics applies to the
matrix used in the forward pass. Ours is Wq = alpha * RoundClip(W/alpha) — a coordinate-wise,
non-linear map that destroys singular-value structure. Adding an orthogonal dW produces a dWq that
is a sparse pattern of +/-alpha bin flips, which is emphatically NOT orthogonal. So Muon's spectral
guarantee is a fiction here. What survives, and what we want, is the per-matrix, magnitude-
preserving, NON-per-coordinate step.

TWO HAZARDS NOBODY HAS REPORTED, BECAUSE NOBODY HAS PUT MUON AND TERNARY QAT IN THE SAME ROOM

  (1) NEWTON-SCHULZ DELETES THE CLIP MASK. RoundClip zeroes the gradient of any latent weight
      outside [-alpha, alpha] — that is the guard that stops a weight from being pushed further out
      of range, forever. But Newton-Schulz MIXES ROWS AND COLUMNS: its output is dense. A
      coordinate whose gradient was masked to zero comes out with a full-size update. The guard is
      silently deleted, latent weights drift out of range, saturate, and their bin can never flip
      back. Progressive weight death, invisible in the loss until it is too late.
      -> mask_after_ns=True re-applies the mask AFTER orthogonalization. Cheap, and it keeps both
         properties. (The other honest option is to drop the clip entirely and use pure STE; the
         2x2 experiment in bakeoff_muon.sh measures which is right rather than guessing.)

  (2) NO WEIGHT DECAY => THE BINS FREEZE. The bin pattern depends only on W/alpha, i.e. on the
      DIRECTION of W, not its norm (absmean is scale-equivariant). Muon applies constant-RMS
      updates, so ||W||_F grows like sqrt(t), so the effective learning rate ON THE BIN PATTERN
      decays like 1/sqrt(t). The bins stop flipping halfway through training and nothing in the LR
      schedule says so.
      -> KEEP WEIGHT DECAY ON with Muon. This also explains BitNet's schedule retroactively: their
         "weight decay -> 0 at 2/3" is not an LR trick, it is a deliberate BIN FREEZER (implicit
         iterative weight freezing, Nagel et al. ICML 2022). Do not copy it blind under Muon: you
         would get the freeze for free, and far too early.

AND ONE THEOREM THAT DOES NOT TRANSFER, WHICH REOPENS A CLOSED QUESTION

  arXiv 2405.05171 proves custom gradient estimators (EWGS, ReSTE, DSQ) are equivalent to plain STE
  — but the proof runs on ADAM'S INVARIANCE TO PER-COORDINATE RESCALING. Muon is invariant to a
  GLOBAL scalar rescale, NOT to a per-coordinate one. So under Muon the estimator is NOT absorbed,
  and the choice of estimator matters MORE than with Adam, not less. We dropped that entire family
  on the strength of that theorem. Under Muon it has to be reopened.

A pleasant coincidence: Muon applies to 2-D hidden weights only; embeddings, lm_head, norms and
scalars go to AdamW. BitNet quantizes exactly the hidden linears and leaves the rest in high
precision. THE TWO PARTITIONS COINCIDE EXACTLY. Muon <-> ternary, AdamW <-> FP, no friction.
"""
import torch


@torch.no_grad()
def newtonschulz5(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz. The coefficients are tuned to maximize the slope at zero and DO NOT
    CONVERGE: the singular values land in ~[0.7, 1.3] rather than exactly 1. Keller reports this
    costs nothing in model quality, and it is what makes 5 steps enough. Do not "fix" it."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if X.size(-2) > X.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for the ternary hidden matrices.

    mask_after_ns: re-apply the STE clip mask AFTER orthogonalization (hazard 1 above). Requires
                   the model to stash the mask on each param as `p._ste_mask` during the forward.
                   Off => the clip mask is destroyed by Newton-Schulz, which is the bug we want to
                   be able to MEASURE, not merely avoid.
    """

    def __init__(self, params, lr=0.025, momentum=0.95, nesterov=True, weight_decay=0.05,
                 ns_steps=5, mask_after_ns=True):
        super().__init__(list(params), dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                            weight_decay=weight_decay, ns_steps=ns_steps,
                                            mask_after_ns=mask_after_ns))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            mu, wd, lr = g["momentum"], g["weight_decay"], g["lr"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(grad)
                m = st["m"]
                m.lerp_(grad, 1 - mu)
                upd = grad.lerp(m, mu) if g["nesterov"] else m

                upd = newtonschulz5(upd, g["ns_steps"])

                if g["mask_after_ns"]:
                    mask = getattr(p, "_ste_mask", None)
                    if mask is not None:
                        upd = upd * mask          # the guard Newton-Schulz would have deleted

                # Keller's shape rule: boost tall matrices (fan-out > fan-in).
                upd = upd * max(1.0, p.size(-2) / p.size(-1)) ** 0.5

                # Decoupled WD. KEEP IT ON for ternary: it is the only thing stopping ||W|| from
                # growing like sqrt(t) and silently freezing the bin pattern (hazard 2).
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(upd, alpha=-lr)
        return loss


def split_params(model):
    """Muon takes the 2-D matrices inside the blocks. Everything else — embeddings, lm_head, norm
    gains, biases, the VRL/skip scalars — goes to AdamW. This partition is Keller's, and it happens
    to be exactly BitNet's quantized/not-quantized partition."""
    muon, embed, head, scalars = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embed" in n:
            embed.append(p)
        elif "lm_head" in n:
            head.append(p)
        elif p.ndim >= 2 and "blocks" in n:
            muon.append(p)
        else:
            scalars.append(p)                     # norms, biases, VRL lambdas, channel scales
    return muon, embed, head, scalars


def build_optimizers(model, muon_lr=0.025, muon_wd=0.05, embed_lr=0.7, head_lr=0.004,
                     scalar_lr=0.015, adam_wd=0.001, mask_after_ns=True, use_muon=True):
    """The asymmetric AdamW learning rates are not a typo: embedding 0.7, scalars 0.015, lm_head
    0.004 — a 175x spread, and the speedrun's track-3 ablation says they are load-bearing. Note
    beta1=0.8 and eps=1e-10, both non-default, both from the same tuned baseline."""
    muon_p, embed_p, head_p, scalar_p = split_params(model)
    groups = [dict(params=embed_p, lr=embed_lr),
              dict(params=head_p, lr=head_lr),
              dict(params=scalar_p, lr=scalar_lr)]
    if not use_muon:
        groups.append(dict(params=muon_p, lr=scalar_lr))     # AdamW arm of the 2x2
    adam = torch.optim.AdamW([g for g in groups if g["params"]],
                             betas=(0.8, 0.95), eps=1e-10, weight_decay=adam_wd, fused=True)
    opt = [adam]
    if use_muon:
        opt.append(Muon(muon_p, lr=muon_lr, weight_decay=muon_wd, mask_after_ns=mask_after_ns))
    return opt
