"""
Generate text from a trained checkpoint via the diffusion sampler (confidence-based
iterative unmasking). Decodes with the SentencePiece tokenizer used for training.

    python sample.py --ckpt runs/fw_tern/best.pt --tokenizer spm_fw.model

This is the eval/eyeball sampler from diffusion.py — the production parallel decoder
lives in the Rust engine. At small scale / short training the Italian will be rough.
"""

import argparse

import torch
import sentencepiece as spm

from model.arch import TernaryDiffusionLM, TernaryDiffusionConfig, strip_compile_prefix
from train.diffusion import generate, generate_parallel, generate_best_of, repetition_rate


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--gen-len", type=int, default=80)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--prompt", action="append", help="repeatable; defaults to a small Italian set")
    args = ap.parse_args()

    # Bake-off of decoders, worst -> best. Each prints rep4 (4-gram repetition rate) so the
    # loop is measured, not eyeballed. The old semi-AR commit-only decoder is the one that
    # loops ("uno stato uno stato ..."); generate_parallel fixes it (see diffusion.py).
    def old(m, pt, **_):
        return generate(m, pt, gen_len=args.gen_len, mask_token_id=cfg.mask_token_id,
                        steps=args.steps, temperature=0.9, top_p=0.9, block_size=16, remask=False)
    def new(m, pt, rep_penalty, no_repeat_ngram):
        return generate_parallel(m, pt, gen_len=args.gen_len, mask_token_id=cfg.mask_token_id,
                                 steps=max(args.steps, 256), temperature=0.9, temp_end=0.4,
                                 top_p=0.92, remask=True, rep_penalty=rep_penalty,
                                 no_repeat_ngram=no_repeat_ngram)
    CONFIGS = [
        ("old semi-AR commit-only", lambda m, pt: old(m, pt)),
        ("parallel + remask", lambda m, pt: new(m, pt, 1.0, 0)),
        ("+ rep penalty", lambda m, pt: new(m, pt, 1.15, 0)),
        ("+ no-repeat 3-gram", lambda m, pt: new(m, pt, 1.15, 3)),
        ("+ best-of-4 rerank", lambda m, pt: generate_best_of(
            m, pt, gen_len=args.gen_len, mask_token_id=cfg.mask_token_id, n=4,
            sampler=generate_parallel, steps=max(args.steps, 256), temperature=0.9,
            temp_end=0.4, top_p=0.92, remask=True, rep_penalty=1.15, no_repeat_ngram=3)),
    ]

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = TernaryDiffusionConfig(**ck["cfg"])
    model = TernaryDiffusionLM(cfg)
    model.load_state_dict(strip_compile_prefix(ck["model"]))
    model.eval()
    dev = pick_device()
    model.to(dev)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    vocab = sp.get_piece_size()
    prompts = args.prompt or [
        "L'Italia è",
        "Nel cuore della città",
        "La storia racconta che",
        "C'era una volta",
        # math/logic probes: characterize the diffusion weakness
        "Quanto fa 12 piu 7? La risposta e",
        "Se tutti i cani sono animali e Fido e un cane, allora Fido e",
    ]

    def decode(ids):
        return sp.decode([i for i in ids if i < vocab])  # drop any leftover [MASK]

    print(f"# ckpt={args.ckpt} ternary={getattr(cfg, 'ternary', True)} step={ck.get('step','?')}")
    for p in prompts:
        ids = sp.encode(p, out_type=int)
        pt = torch.tensor([ids], device=dev, dtype=torch.long)
        print(f"\n### {p!r}")
        for tag, fn in CONFIGS:
            out = fn(model, pt)
            gen = out[0, len(ids):].tolist()
            print(f"  [{tag:26}] rep4={repetition_rate(gen, 4):.2f}  {decode(gen)}")


if __name__ == "__main__":
    main()
