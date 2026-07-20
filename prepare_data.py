"""
Corpus -> tokenizer + tokens.bin. The piece README.md punts on ("build it however
you like"); here is a sane default so you don't have to.

Trains a SentencePiece tokenizer on your text, then encodes the whole corpus into the
flat uint16 stream train.py expects. Convention matches the rest of the repo: the
tokenizer owns ids [0, vocab), and we reserve ONE id past it as [MASK].

    # train a tokenizer on a folder of .md/.txt and encode it
    python prepare_data.py --input "path/to/corpus" --out tokens.bin \
        --tokenizer spm.model --vocab-size 8000 --train

    # reuse an existing tokenizer to encode more text
    python prepare_data.py --input more/ --out more.bin --tokenizer spm.model

It prints the --vocab-size / --mask-id to pass to train.py (model vocab = spm + 1).
"""

import argparse
import glob
import os

import numpy as np
import sentencepiece as spm


def gather_text(input_path: str) -> list[str]:
    if os.path.isdir(input_path):
        files = []
        for ext in ("*.md", "*.txt"):
            files += glob.glob(os.path.join(input_path, "**", ext), recursive=True)
    else:
        files = glob.glob(input_path)
    files = sorted(f for f in files if os.path.isfile(f))
    assert files, f"no .md/.txt files found under {input_path!r}"
    return files


def _flush(sp, lines: list[str], out, mask_id: int) -> int:
    if not lines:
        return 0
    # One array + one buffered write per batch. np.tofile() writes straight to the fd,
    # so a per-sentence tofile() is a ~100-byte syscall each — murder on a network volume.
    flat = [i for ids in sp.encode(lines, out_type=int, num_threads=os.cpu_count()) for i in ids]
    if not flat:
        return 0
    arr = np.asarray(flat, dtype=np.uint16)
    assert arr.max() < mask_id, "token id collided with reserved [MASK] id"
    out.write(arr.tobytes())
    return arr.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dir (recursed) or a glob of .md/.txt")
    ap.add_argument("--out", default="tokens.bin")
    ap.add_argument("--tokenizer", default="spm.model")
    ap.add_argument("--vocab-size", type=int, default=8000, help="SentencePiece vocab (when --train)")
    ap.add_argument("--train", action="store_true", help="(re)train the tokenizer on --input")
    args = ap.parse_args()

    files = gather_text(args.input)
    print(f"{len(files)} files")

    if args.train or not os.path.exists(args.tokenizer):
        prefix = args.tokenizer[:-6] if args.tokenizer.endswith(".model") else args.tokenizer
        spm.SentencePieceTrainer.train(
            input=files,
            model_prefix=prefix,
            vocab_size=args.vocab_size,
            model_type="unigram",
            character_coverage=1.0,        # full coverage: Italian accents, code, symbols
            bos_id=-1, eos_id=-1,          # diffusion denoiser: no BOS/EOS framing
            pad_id=-1, unk_id=0,
            input_sentence_size=2_000_000, # sample for big corpora: bounds RAM/time
            shuffle_input_sentence=True,
        )
        print(f"trained tokenizer -> {prefix}.model")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    spm_vocab = sp.get_piece_size()
    mask_id = spm_vocab                    # reserve the next id as [MASK]
    model_vocab = spm_vocab + 1

    # Stream the encode: batch lines -> C++ batched encode -> append uint16 chunks.
    # Never holds the corpus (or its ids) in RAM, so a 10GB+ file is fine.
    n_tokens = 0
    with open(args.out, "wb", buffering=8 << 20) as out:
        for f in files:
            buf = []
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.strip():
                        buf.append(line)
                    if len(buf) >= 20_000:
                        n_tokens += _flush(sp, buf, out, mask_id)
                        buf = []
                        if n_tokens % 100_000_000 < 1_000_000:
                            print(f"  {n_tokens/1e9:.2f}B tokens", flush=True)
            n_tokens += _flush(sp, buf, out, mask_id)

    print(f"encoded {n_tokens:,} tokens -> {args.out}")
    print(f"pass to train.py:  --vocab-size {model_vocab} --mask-id {mask_id}")


if __name__ == "__main__":
    main()
