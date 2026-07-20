"""The same transformer as the baseline, with BitLinearV2 swapped in for BitLinear, plus
self-conditioning.

Everything else — RoPE, SwiGLU, QK-norm, the bidirectional attention, which tensors stay full
precision — is unchanged and imported from model_base. The point of this project is to measure the
changes, so nothing else may vary.

SELF-CONDITIONING (Loopholing, arXiv 2510.19304, ICLR 2026). The denoiser gets a second look at
the same corrupted input, this time conditioned on its own hidden state from the first look.

The detail that makes it worth doing, and that the older Analog-Bits formulation gets wrong: we
feed back the HIDDEN STATE h (d_model dims), not the predicted distribution over the vocabulary
(32001 dims). Feeding the distribution back means a V-by-D projection — for us that is a matrix
the size of the embedding table, on the critical path, and it is the single most
quantization-sensitive tensor in the model. Feeding h back costs one RMSNorm.

Measured, from scratch, at 78-110M — i.e. our regime, not a 7B fine-tune:
    MDLM  test PPL (OpenWebText)  23.82 -> 21.90     (-8.1%)
    MDLM  test PPL (LM1B)         27.60 -> 25.95
Generative PPL falls 108.9 -> 49.1, but ignore that number: gen-PPL is trivially gamed by
low-entropy sampling and half this literature is fooling itself with it. The test NELBO is the
number that means something, and it moved.

Cost: one extra forward, no extra backward (pass 1 is stop-grad). ~+30% step time, and for us
somewhat more, because the extra forward also re-pays the fake-quant. The ELBO is untouched: we
changed the parameterization of p_theta, not the bound.
"""
import torch
import torch.nn as nn

from model import arch as B
from model.quant import BitLinearV2, MitigationConfig, QATConfig


def _swap(module, cfg, mit, qat):
    """Replace every BitLinear with a BitLinearV2 carrying the same shape and the same knobs."""
    for name, child in module.named_children():
        if isinstance(child, B.BitLinear):
            new = BitLinearV2(child.weight.shape[1], child.weight.shape[0],
                              cfg.quant_eps, cfg.act_levels, cfg.ternary, mit, qat)
            new.weight = child.weight        # keep the init, so arms start identical
            new.norm = child.norm
            setattr(module, name, new)
        else:
            _swap(child, cfg, mit, qat)


class MitigatedDiffusionLM(B.TernaryDiffusionLM):
    def __init__(self, cfg: B.TernaryDiffusionConfig, mit: MitigationConfig = None,
                 qat: QATConfig = None, self_cond: bool = False):
        super().__init__(cfg)
        self.mit = mit or MitigationConfig()
        self.qat = qat or QATConfig()
        self.self_cond = self_cond
        if cfg.ternary:
            _swap(self, cfg, self.mit, self.qat)

        if self_cond:
            # LayerNorm on the fed-back state, zero-init gain: at step 0 the injected signal is
            # exactly zero, so the model starts as the plain baseline and self-conditioning has to
            # earn its way in. Same discipline as the zero-init low-rank factors.
            self.sc_norm = nn.LayerNorm(cfg.d_model)
            nn.init.zeros_(self.sc_norm.weight)
            nn.init.zeros_(self.sc_norm.bias)

    def forward(self, input_ids, attention_mask=None, attn_bias=None, position_ids=None,
                h_prev=None, return_h=False):
        """h_prev: (B, T, D) hidden state from a previous pass, or None. Injected additively at
        the embedding. return_h: also return the final hidden state, for the next pass."""
        T = input_ids.shape[1]
        x = self.embed(input_ids)

        if h_prev is not None:
            if not self.self_cond:
                raise ValueError("h_prev passed but the model was built with self_cond=False")
            x = x + self.sc_norm(h_prev)

        if position_ids is None:
            cos = self.rope_cos[:T].to(x.dtype)
            sin = self.rope_sin[:T].to(x.dtype)
        else:
            cos = self.rope_cos[position_ids].to(x.dtype)
            sin = self.rope_sin[position_ids].to(x.dtype)

        attn_mask = attn_bias.to(x.dtype) if attn_bias is not None else None
        if attention_mask is not None:
            keep = attention_mask[:, None, None, :].bool()
            pad = torch.zeros_like(keep, dtype=x.dtype).masked_fill(~keep, float("-inf"))
            attn_mask = pad if attn_mask is None else attn_mask + pad

        for blk in self.blocks:
            x = blk(x, cos, sin, attn_mask)

        h = self.final_norm(x)
        logits = self.lm_head(h)
        return (logits, h) if return_h else logits

    # -- reporting ----------------------------------------------------------- #

    def _bits(self):
        return [m for m in self.modules() if isinstance(m, BitLinearV2)]

    def mitigation_bytes(self):
        return sum(m.extra_memory_bytes() for m in self._bits())

    def ternary_bytes(self):
        return sum(m.weight.numel() / 5 for m in self._bits())      # 5 trits per byte

    def track_oscillation(self, on=True):
        for m in self._bits():
            m._track = on

    def osc_frac(self, reset=True):
        """Weight-average of the per-layer bin-flip rate. The go/no-go number for dampening."""
        bits = self._bits()
        if not bits:
            return 0.0
        tot = sum(m.weight.numel() for m in bits)
        return sum(m.osc_frac(reset) * m.weight.numel() for m in bits) / max(tot, 1)

    def zero_frac(self):
        bits = self._bits()
        if not bits:
            return 0.0
        tot = sum(m.weight.numel() for m in bits)
        return sum(m.zero_frac() * m.weight.numel() for m in bits) / max(tot, 1)

    def dampen_loss(self):
        bits = self._bits()
        if not bits:
            return torch.zeros((), device=self.embed.weight.device)
        return sum(m.dampen_loss() for m in bits) / sum(m.weight.numel() for m in bits)

    def latent_params(self):
        """The BitLinear latent weights, which must NOT be weight-decayed.

        A latent weight's only job is to pick a bin. Decaying it is not capacity regularization —
        it is a constant pull toward zero, i.e. toward the decision boundary, i.e. straight into
        the oscillation it causes. BitNet b1.58 and TriLM both kill weight decay late in training
        for exactly this reason; we go further and never apply it to these tensors at all.
        """
        return [m.weight for m in self._bits()]
