"""A plain autoregressive transformer with ternary (1.58-bit) QAT. Nothing exotic — this is the
same shape as every decoder-only LM in the literature, which is exactly the point.

WHY THIS PROJECT EXISTS

The sibling project (../ECHO-DIFF-TERNARY) trains a ternary MASKED DIFFUSION LM. It works, but it
pays a large quantization penalty: at 27M params / 300M tokens the ternary model is 21% worse in
perplexity than its FP16 twin, and that penalty GROWS with the token budget (8.6% at 150M tokens,
21% at 300M). Meanwhile every published 1.58-bit result — BitNet b1.58, TriLM/Spectra, FBI-LLM,
TernaryLLM — is AUTOREGRESSIVE, and they report much smaller penalties.

Nobody has run the controlled comparison, because nobody has trained a ternary masked-diffusion LM
from scratch to compare against. So the open question is:

    Is the penalty a property of TERNARY WEIGHTS,
    or a property of ternary weights UNDER A MASKED-DIFFUSION OBJECTIVE?

Hypothesis, with a mechanism: an autoregressive model predicts ONE token from a FULLY CLEAN left
context. A masked diffusion model predicts MANY tokens IN PARALLEL from a PERFORATED context, and
must model the dependencies among its own simultaneous predictions. That is a harder function to
represent, and a ternary weight has ~10x less room to represent it in. If so, autoregression is
where extreme quantization should be spent — and this folder is that model.

WHAT IS INHERITED AND WHY

  model_base.py  the transformer: BitLinear (absmean ternary + STE + SubLN), RoPE, SwiGLU, QK-norm.
                 Unchanged, deliberately. If the architecture varied, the AR-vs-diffusion comparison
                 would be confounded and the whole point would be lost.
  quant.py       the QAT knobs (zero-threshold, TWN scale, per-channel scale, oscillation tracking),
                 carried over from the diffusion project's mitigation work.

WHAT CHANGES: the attention mask is causal, and the label is the next token. That is all.
"""
from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import arch as B
from model.quant import BitLinearV2, MitigationConfig, QATConfig


class ReLU2(nn.Module):
    """Squared-ReLU FFN, 4x expansion, TWO matrices instead of SwiGLU's three.

    THIS IS NOT A SPEEDRUN AFFECTATION. It is the one place where the ternary literature and the
    frontier-architecture literature independently arrive at the same answer, for different reasons:

      - BitNet b1.58 2B4T (arXiv 2504.12285) picked squared ReLU over SwiGLU explicitly "for
        improved sparsity within the 1-bit context". BitNet a4.8 shows ReLU^2 drives >80% sparsity
        into the down-projection input, and that sparsity is what makes 4-bit activations possible
        at all.
      - Karpathy tested SwiGLU FLOP- AND param-matched at two depths and found it "worse on all
        measures — step efficiency, wall clock, and FLOPs".

    And the mechanism matters for US specifically, because our activations are quantized with
    ABSMAX: one outlier sets the scale for the whole tensor and crushes everything else's
    resolution. SwiGLU's gate MULTIPLIES two unbounded quantities, silu(a) * b, which manufactures
    exactly those heavy-tailed spikes — costing 2-3 effective bits of the 8 we have. ReLU^2 is
    one-sided and non-multiplicative. Under FP16 this is a wash; under our activation quantizer
    SwiGLU is actively harmful.

    Keep the SubLN inside BitLinear: ReLU^2 SQUARES, which widens the dynamic range of whatever
    survives, and the norm before the down-projection is what contains it.
    """

    def __init__(self, cfg, mit, qat):
        super().__init__()
        self.fc = BitLinearV2(cfg.d_model, cfg.d_ff, cfg.quant_eps, cfg.act_levels, cfg.ternary, mit, qat)
        self.down = BitLinearV2(cfg.d_ff, cfg.d_model, cfg.quant_eps, cfg.act_levels, cfg.ternary, mit, qat)

    def forward(self, x):
        return self.down(F.relu(self.fc(x)).square())


def _swap(module, cfg, mit, qat, relu2=True):
    for name, child in module.named_children():
        if relu2 and isinstance(child, B.SwiGLU):
            setattr(module, name, ReLU2(cfg, mit, qat))
        elif isinstance(child, B.BitLinear):
            new = BitLinearV2(child.weight.shape[1], child.weight.shape[0],
                              cfg.quant_eps, cfg.act_levels, cfg.ternary, mit, qat)
            new.weight = child.weight
            new.norm = child.norm
            setattr(module, name, new)
        else:
            _swap(child, cfg, mit, qat, relu2)


class TernaryAR(B.TernaryDiffusionLM):
    """Causal LM. The causal mask is built INSIDE forward and cached as a buffer — it is not an
    argument a caller can forget to pass. A bidirectional model that silently sees the future would
    post a spectacular validation loss and the bug would look like a result.
    """

    def __init__(self, cfg: B.TernaryDiffusionConfig, mit: MitigationConfig = None,
                 qat: QATConfig = None, relu2: bool = True):
        # PARAM-MATCH THE FFN BEFORE BUILDING IT. SwiGLU has THREE d_model x d_ff matrices; ReLU^2
        # has TWO. At equal d_ff the ReLU^2 model would be a third smaller and would lose the
        # comparison on parameter count, not on the activation — the exact confound that makes most
        # published activation-function comparisons worthless. 1.5x d_ff restores parity.
        if relu2:
            cfg = replace(cfg, d_ff=int(round(cfg.d_ff * 1.5 / 64) * 64))
        super().__init__(cfg)
        self.mit = mit or MitigationConfig()
        self.qat = qat or QATConfig()
        self.relu2 = relu2
        if cfg.ternary or relu2:
            _swap(self, cfg, self.mit, self.qat, relu2)

        m = torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf")).triu(1)
        self.register_buffer("causal", m[None, None], persistent=False)   # (1, 1, S, S)

    def forward(self, input_ids, attention_mask=None):
        T = input_ids.shape[1]
        x = self.embed(input_ids)
        cos = self.rope_cos[:T].to(x.dtype)
        sin = self.rope_sin[:T].to(x.dtype)

        bias = self.causal[:, :, :T, :T].to(x.dtype)
        if attention_mask is not None:
            keep = attention_mask[:, None, None, :].bool()
            bias = bias + torch.zeros_like(keep, dtype=x.dtype).masked_fill(~keep, float("-inf"))

        for blk in self.blocks:
            x = blk(x, cos, sin, bias)
        return self.lm_head(self.final_norm(x))

    def loss(self, x):
        """Next-token cross-entropy. Predict position i+1 from positions <= i."""
        logits = self(x)
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               x[:, 1:].reshape(-1))

    # -- reporting (same contract as the diffusion project, so results are comparable) -------- #

    def _bits(self):
        return [m for m in self.modules() if isinstance(m, BitLinearV2)]

    def latent_params(self):
        """Never weight-decay these: a latent weight's job is to pick a bin, and decaying it just
        drags it onto the decision boundary."""
        return [m.weight for m in self._bits()]

    def ternary_bytes(self):
        return sum(m.weight.numel() / 5 for m in self._bits())     # 5 trits per byte

    def mitigation_bytes(self):
        return sum(m.extra_memory_bytes() for m in self._bits())

    def zero_frac(self):
        b = self._bits()
        if not b:
            return 0.0
        tot = sum(m.weight.numel() for m in b)
        return sum(m.zero_frac() * m.weight.numel() for m in b) / max(tot, 1)

    def track_oscillation(self, on=True):
        for m in self._bits():
            m._track = on

    def osc_frac(self, reset=True):
        b = self._bits()
        if not b:
            return 0.0
        tot = sum(m.weight.numel() for m in b)
        return sum(m.osc_frac(reset) * m.weight.numel() for m in b) / max(tot, 1)


@torch.no_grad()
def generate(model, prompt, max_new, temperature=0.8, top_k=50, eos_id=None):
    """Ordinary AR sampling. No diffusion sampler, no remasking, no block schedule — one token at a
    time, which is the whole ergonomic advantage of going back to autoregression."""
    model.eval()
    x = prompt
    for _ in range(max_new):
        window = x[:, -model.cfg.max_seq_len:]
        logits = model(window)[:, -1, :].float()
        if model.cfg.mask_token_id < logits.shape[-1]:
            logits[:, model.cfg.mask_token_id] = float("-inf")   # no [MASK] token in AR output
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), 1)
        x = torch.cat([x, nxt], dim=1)
        if eos_id is not None and (nxt == eos_id).all():
            break
    return x


def _demo():
    """The check that matters: the model must not see the future. A leak here would look like a
    breakthrough on the validation curve."""
    cfg = B.TernaryDiffusionConfig(vocab_size=256, mask_token_id=255, d_model=64, n_layers=2,
                                   n_heads=4, d_ff=128, max_seq_len=32)
    m = TernaryAR(cfg).eval()
    x = torch.randint(0, 250, (2, 32))
    with torch.no_grad():
        a = m(x)
        x2 = x.clone(); x2[:, 16:] = 1              # rewrite the entire future
        b = m(x2)
    early = (a[:, :15] - b[:, :15]).abs().max().item()
    late = (a[:, 16:] - b[:, 16:]).abs().max().item()
    assert early == 0.0, f"CAUSALITY LEAK: past logits moved by {early} when the future changed"
    assert late > 0.0, "the future changed but nothing moved — the model is not reading its input"
    loss = m.loss(x)
    assert torch.isfinite(loss)
    out = generate(m, x[:, :4], max_new=8)
    assert out.shape == (2, 12)
    print(f"AR demo OK | loss {loss.item():.3f} | causal mask airtight | zero_frac {m.zero_frac():.3f}")


if __name__ == "__main__":
    _demo()
