"""
Ternary Masked-Diffusion Language Model — core architecture.

The novel bet: native ternary *training* (BitNet b1.58-style quantization-aware
training, not post-hoc quantization) of a *masked-diffusion* denoiser. The
literature has ternary diffusion LMs only as post-training quantization, and that
already degrades at 2-bit; native QAT on a diffusion objective is open ground.

Design decisions, each load-bearing (see DESIGN.md for the why):

  - Backbone is a BIDIRECTIONAL transformer (no causal mask). This is the diffusion
    denoiser: it sees a partially-[MASK]ed sequence and predicts the clean tokens.
  - NO timestep/noise embedding. Absorbing-state masked diffusion is "secretly
    time-agnostic" (Zheng et al.; LLaDA in practice does not condition on t). The
    masking ratio is implicit in how many [MASK] tokens appear. One less thing to
    train, one less source of instability.
  - Linears inside attention and FFN are ternary BitLinear. Embeddings and the LM
    head stay BF16 — ternarizing them at ~300M wrecks quality (standard BitNet
    practice). Do NOT "ternarize everything for consistency": that is a footgun.
  - Normalization is folded INTO BitLinear (SubLN-style), so the residual blocks
    look like they have no pre-norm. That is intentional. Do not "fix" it by adding
    an outer pre-norm — you would double-normalize.
  - QK-norm (RMSNorm on per-head q and k) is added on top of stock BitNet. It is
    cheap insurance against attention-logit blowup, which matters more here because
    ternary QAT is already touchy.

Fake-quant is done in BF16 with a straight-through estimator; there are NO integer
kernels in this file. Integer/LUT ternary matmul lives in the Rust inference engine.
Training cost is therefore ~equal to a full-precision model of the same shape — the
ternary win is RAM and energy at *inference*, not compute at *training*.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def strip_compile_prefix(state_dict):
    """torch.compile wraps the module, so a state_dict saved from the compiled model has every
    key prefixed with '_orig_mod.'. Resuming works (it loads back into a compiled model), but
    sample.py/export.py build a bare model and would fail. Strip it on load."""
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


@dataclass
class TernaryDiffusionConfig:
    # Vocabulary. Convention: reserve the LAST id as the [MASK] absorbing token.
    # If your tokenizer has 32000 tokens, set vocab_size=32001 and mask_token_id=32000.
    vocab_size: int = 32001
    mask_token_id: int = 32000

    # ~340M total / ~308M ternary (non-embedding). Tune n_layers/d_ff to hit a target.
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    d_ff: int = 2816            # ~ (8/3)*d_model, rounded to a multiple of 256
    max_seq_len: int = 2048

    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0        # at this scale, with enough data, 0 is usual

    # BitLinear / quantization
    act_levels: int = 127       # 8-bit symmetric activations (BitNet b1.58). a4.8 (4-bit) is future work.
    quant_eps: float = 1e-5

    # The kill-switch baseline. ternary=False keeps BitLinear's structure (the internal
    # SubLN, the shapes, the param count) but skips both fake-quant steps, giving a
    # same-shape FP16 model to compare validation masked-CE against. See DESIGN.md.
    ternary: bool = True

    tie_embeddings: bool = True


# ----------------------------------------------------------------------------- #
# Primitives
# ----------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


class BitLinear(nn.Module):
    """BitNet b1.58 linear layer.

    Weights -> ternary {-1, 0, +1} via absmean scaling.
    Activations -> int8 via per-token absmax scaling.
    Both fake-quantized in BF16, gradients via straight-through estimator (STE).

    The internal RMSNorm (applied to the input before activation quantization) is
    part of the recipe, not decoration — it is what makes per-token absmax
    quantization stable. It also serves as this sublayer's pre-norm.
    """

    def __init__(self, in_features: int, out_features: int, eps: float = 1e-5,
                 act_levels: int = 127, ternary: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.norm = RMSNorm(in_features, eps)
        self.eps = eps
        self.act_levels = act_levels
        self.ternary = ternary

    def forward(self, x):
        x = self.norm(x)

        # Baseline mode: same shape, same SubLN, but full precision — no fake-quant.
        if not self.ternary:
            return F.linear(x, self.weight)

        # --- weights: ternary, dequantized, STE back to the real (BF16) weight ---
        ws = self.weight.abs().mean().clamp(min=self.eps)
        w_tern = (self.weight / ws).round().clamp(-1, 1) * ws
        w = self.weight + (w_tern - self.weight).detach()

        # --- activations: per-token int8, dequantized, STE ---
        xs = x.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps) / self.act_levels
        x_q = (x / xs).round().clamp(-self.act_levels, self.act_levels) * xs
        x = x + (x_q - x).detach()

        return F.linear(x, w)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)            # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)     # (T, head_dim)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); cos/sin: (T, D)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


# ----------------------------------------------------------------------------- #
# Blocks
# ----------------------------------------------------------------------------- #

class Attention(nn.Module):
    def __init__(self, cfg: TernaryDiffusionConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout

        self.q_proj = BitLinear(cfg.d_model, cfg.d_model, cfg.quant_eps, cfg.act_levels, cfg.ternary)
        self.k_proj = BitLinear(cfg.d_model, cfg.d_model, cfg.quant_eps, cfg.act_levels, cfg.ternary)
        self.v_proj = BitLinear(cfg.d_model, cfg.d_model, cfg.quant_eps, cfg.act_levels, cfg.ternary)
        self.o_proj = BitLinear(cfg.d_model, cfg.d_model, cfg.quant_eps, cfg.act_levels, cfg.ternary)

        # QK-norm: stabilizes logits under ternary QAT. Cheap, off-recipe, worth it.
        self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)

    def forward(self, x, cos, sin, attn_mask):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # BIDIRECTIONAL: is_causal=False. attn_mask (if given) masks padding columns.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: TernaryDiffusionConfig):
        super().__init__()
        self.gate = BitLinear(cfg.d_model, cfg.d_ff, cfg.quant_eps, cfg.act_levels, cfg.ternary)
        self.up = BitLinear(cfg.d_model, cfg.d_ff, cfg.quant_eps, cfg.act_levels, cfg.ternary)
        self.down = BitLinear(cfg.d_ff, cfg.d_model, cfg.quant_eps, cfg.act_levels, cfg.ternary)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Residual block. No explicit pre-norm: each BitLinear normalizes its own input
    (SubLN), so adding an outer norm here would double-normalize. This is deliberate."""

    def __init__(self, cfg: TernaryDiffusionConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.attn(x, cos, sin, attn_mask)
        x = x + self.ffn(x)
        return x


# ----------------------------------------------------------------------------- #
# Model
# ----------------------------------------------------------------------------- #

class TernaryDiffusionLM(nn.Module):
    def __init__(self, cfg: TernaryDiffusionConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)            # BF16, NOT ternary
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # BF16, NOT ternary
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = precompute_rope(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None, attn_bias=None, position_ids=None):
        """attn_bias: optional additive attention mask, broadcastable to (B, H, S, S).
        None => fully bidirectional (candidate A). position_ids: optional (S,) RoPE
        positions; None => 0..S-1. Candidate B's concatenated dual-stream passes a
        4-quadrant bias AND position_ids that REPEAT (clean i and noisy i share
        position i) so the two streams align. See arch_b.py."""
        B, T = input_ids.shape
        x = self.embed(input_ids)

        if position_ids is None:
            cos = self.rope_cos[:T].to(x.dtype)
            sin = self.rope_sin[:T].to(x.dtype)
        else:
            cos = self.rope_cos[position_ids].to(x.dtype)
            sin = self.rope_sin[position_ids].to(x.dtype)

        attn_mask = attn_bias.to(x.dtype) if attn_bias is not None else None
        if attention_mask is not None:
            # (B, T) with 1=keep, 0=pad  ->  additive (B, 1, 1, T)
            keep = attention_mask[:, None, None, :].bool()
            pad = torch.zeros_like(keep, dtype=x.dtype).masked_fill(~keep, float("-inf"))
            attn_mask = pad if attn_mask is None else attn_mask + pad

        for blk in self.blocks:
            x = blk(x, cos, sin, attn_mask)

        x = self.final_norm(x)
        return self.lm_head(x)

    # -- helpers ------------------------------------------------------------- #
    def num_parameters(self, only_ternary: bool = False):
        if only_ternary:
            return sum(p.numel() for m in self.modules() if isinstance(m, BitLinear) for p in [m.weight])
        return sum(p.numel() for p in self.parameters())

    def optim_param_groups(self, weight_decay: float):
        """No weight decay on norms / embeddings / 1-D params; decay everything else."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "embed" in name or "norm" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
