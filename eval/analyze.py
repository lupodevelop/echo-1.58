"""Turn results/*.json into the table the paper needs. Never retype a number by hand.

    python analyze_bakeoff.py            # table to stdout
    python analyze_bakeoff.py --latex    # the LaTeX body, ready to paste

The gap we care about is TERNARY vs FP16, and the question for every arm is what fraction of that
gap it closes. An arm that improves the ternary model but closes 3% of the gap is a footnote; one
that closes 40% is a result.
"""
import argparse
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = {
    "fp16":       "FP16 (ceiling)",
    "ternary":    "Ternary b1.58 (baseline)",
    "t_selfcond": "+ self-conditioning",
    "t_varred":   "+ stratified t + antithetic masks",
    "t_thresh07": "+ zero-threshold 0.7 + TWN scale",
    "t_chanscale": "+ learnable channel scale",
    "t_twostage": "+ two-stage LR, WD to 0",
    "t_distil":   "+ distillation from FP16 twin",
    "t_all":      "+ everything above",
}


def load():
    out = {}
    for f in glob.glob(os.path.join(HERE, "results", "*.json")):
        d = json.load(open(f))
        out[d["tag"]] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    r = load()
    if not r:
        print("no results yet")
        return

    fp = r.get("fp16", {}).get("best_val_masked_ce")
    tern = r.get("ternary", {}).get("best_val_masked_ce")
    gap = (tern - fp) if (fp and tern) else None

    order = [k for k in LABEL if k in r] + [k for k in r if k not in LABEL]
    rows = []
    for tag in order:
        d = r[tag]
        ce = d["best_val_masked_ce"]
        ppl = math.exp(ce)
        vs_t = (tern - ce) if tern else None                 # improvement over ternary baseline
        closed = (100 * vs_t / gap) if (gap and gap > 0 and tag not in ("fp16", "ternary")) else None
        rows.append({
            "tag": tag, "label": LABEL.get(tag, tag), "ce": ce, "ppl": ppl,
            "d_ce": vs_t, "closed": closed,
            "zero": d.get("zero_frac"), "mb": d.get("extra_mb", 0.0),
            "min": d.get("wall_min", 0.0),
        })

    if gap:
        print(f"\nternary-FP16 gap: {gap:+.4f} masked_ce "
              f"({100*(math.exp(tern)/math.exp(fp)-1):.1f}% worse perplexity)\n")

    hdr = f"{'arm':<36} {'masked_ce':>10} {'ppl':>8} {'vs tern':>9} {'gap closed':>11} {'zeros':>7} {'min':>6}"
    print(hdr)
    print("-" * len(hdr))
    for x in rows:
        z = f"{x['zero']:.2f}" if x["zero"] is not None else "  -"
        dc = f"{x['d_ce']:+.4f}" if x["d_ce"] is not None and x["tag"] != "ternary" else "    -"
        cl = f"{x['closed']:+.0f}%" if x["closed"] is not None else "    -"
        print(f"{x['label']:<36} {x['ce']:>10.4f} {x['ppl']:>8.1f} {dc:>9} {cl:>11} {z:>7} {x['min']:>6.0f}")

    # The honest caveat, printed every time so nobody forgets it when copying the table.
    print("\nvs tern: masked_ce improvement over the ternary baseline (positive = better).")
    print("gap closed: share of the ternary-FP16 gap recovered. This is the number that matters.")
    print("zeros: fraction of ternary weights at 0. If an arm moved it, sparsity is a confound")
    print("       and the comparison is NOT clean — say so rather than hiding it.")

    if args.latex:
        print("\n% --- paste into the paper ---")
        for x in rows:
            cl = f"{x['closed']:+.0f}\\%" if x["closed"] is not None else "--"
            print(f"{x['label']} & {x['ce']:.4f} & {x['ppl']:.1f} & {cl} \\\\")


if __name__ == "__main__":
    main()
