"""Infilling (cloze) benchmark on the 341M checkpoints. The downstream eval for the paper:
the model is trained to reconstruct masked tokens, so measure exactly that, on held-out text,
at a grid of mask ratios, with IDENTICAL seeded masks for both models.

    python cloze_pod.py --ckpt /workspace/runs/ks_tern/final.pt --out /workspace/runs/cloze_tern.json
    python cloze_pod.py --ckpt /workspace/runs/ks_fp16/final.pt --out /workspace/runs/cloze_fp16.json

Part A (quantitative): top-1 / top-5 masked-token recovery accuracy on 256 held-out windows
(last 1% of fineweb.bin, same split convention as training) at ratios 15/30/50/70%.
Part B (qualitative): hand-picked Italian cloze sentences, one content word masked, top-3
predictions. For the paper's illustrative table.
"""
import argparse
import json

import numpy as np
import torch
import sentencepiece as spm

from model.arch import TernaryDiffusionLM, TernaryDiffusionConfig, strip_compile_prefix

RATIOS = [0.15, 0.30, 0.50, 0.70]
N_SEQ = 256
SEQ_LEN = 512
BATCH = 8
SEED = 1234

# (sentence, target word to mask)
CLOZE = [
    ("La Divina Commedia racconta il viaggio di Dante attraverso Inferno, Purgatorio e Paradiso.", "Commedia"),
    ("La capitale della Francia è Parigi.", "Parigi"),
    ("Roma è la capitale d'Italia.", "Italia"),
    ("L'acqua bolle a cento gradi.", "gradi"),
    ("Il sole sorge a est e tramonta a ovest.", "est"),
    ("La pasta alla carbonara si prepara con uova, guanciale e pecorino.", "pecorino"),
    ("Due più due fa quattro.", "quattro"),
    ("Il cambiamento climatico è causato dalle emissioni di gas serra.", "serra"),
]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=False, help="uint16 token bin; last 1% is the held-out split")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = TernaryDiffusionConfig(**ck["cfg"])
    m = TernaryDiffusionLM(cfg)
    m.load_state_dict(strip_compile_prefix(ck["model"]))
    m = m.cuda().eval()
    MASK = cfg.mask_token_id

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    val = data[int(len(data) * 0.99):]
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(val) - SEQ_LEN, size=N_SEQ)
    seqs = np.stack([val[s:s + SEQ_LEN].astype(np.int64) for s in starts])

    res = {"ckpt": args.ckpt, "ternary": bool(cfg.ternary), "step": int(ck.get("step", -1)),
           "n_seq": N_SEQ, "seq_len": SEQ_LEN, "seed": SEED, "ratios": {}}

    # Part A: identical masks across models via fixed torch generator per ratio
    for ratio in RATIOS:
        g = torch.Generator().manual_seed(SEED + int(ratio * 1000))
        mask_all = torch.rand(seqs.shape, generator=g) < ratio      # CPU, model-independent
        top1 = top5 = tot = 0
        for i in range(0, N_SEQ, BATCH):
            x0 = torch.from_numpy(seqs[i:i + BATCH]).cuda()
            mk = mask_all[i:i + BATCH].cuda()
            xt = torch.where(mk, torch.full_like(x0, MASK), x0)
            lg = m(xt).float()
            t5 = lg.topk(5, dim=-1).indices                          # (B, L, 5)
            hit1 = (t5[..., 0] == x0) & mk
            hit5 = (t5 == x0.unsqueeze(-1)).any(-1) & mk
            top1 += int(hit1.sum()); top5 += int(hit5.sum()); tot += int(mk.sum())
        res["ratios"][str(ratio)] = {"top1": top1 / tot, "top5": top5 / tot, "n_masked": tot}
        print(f"ratio {ratio:.2f}: top1 {top1/tot:.4f}  top5 {top5/tot:.4f}  (n={tot})", flush=True)

    # Part B: qualitative cloze, mask the target word's full token span
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    res["cloze"] = []
    for sent, word in CLOZE:
        ids = sp.encode(sent, out_type=int)
        pieces = [sp.id_to_piece(i) for i in ids]
        # find the token span covering `word`
        span = None
        for a in range(len(ids)):
            for b in range(a + 1, min(a + 6, len(ids) + 1)):
                if sp.decode(ids[a:b]).strip() == word:
                    span = (a, b); break
            if span:
                break
        if not span:
            res["cloze"].append({"sent": sent, "word": word, "error": "span not found",
                                 "pieces": pieces})
            continue
        a, b = span
        xt = torch.tensor([ids], device="cuda")
        xt[0, a:b] = MASK
        lg = m(xt).float()[0, a:b, :]                                # (span, V)
        top3_ids = lg.topk(3, dim=-1).indices                        # (span, 3)
        top1_word = sp.decode(top3_ids[:, 0].tolist()).strip()
        alts = [sp.decode(top3_ids[:, k].tolist()).strip() for k in range(3)]
        res["cloze"].append({"sent": sent, "word": word, "span_tokens": b - a,
                             "top1": top1_word, "top3": alts,
                             "correct": top1_word == word})
        print(f"[{'OK ' if top1_word == word else 'MISS'}] {word!r:14} -> {alts}", flush=True)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
