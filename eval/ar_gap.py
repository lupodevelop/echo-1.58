"""The one number the control experiment exists to produce: gap_MDM / gap_AR.

Reads results/*.json. Prints the 2x2 and the ratio, with the honest caveat attached, because this
is the number most likely to be over-read.
"""
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NOISE = 0.036   # measured run-to-run std on this setup (data/results/seed_noise_27m.md)


def main():
    r = {}
    for f in glob.glob(os.path.join(HERE, "results", "*.json")):
        d = json.load(open(f))
        r[d["tag"]] = d["best_val_masked_ce"]

    need = {"fp16": "MDM fp16", "ternary": "MDM ternary",
            "ar_fp16": "AR fp16", "ar_ternary": "AR ternary"}
    missing = [k for k in need if k not in r]
    if missing:
        print(f"not ready yet, missing: {', '.join(missing)}")
        return

    gap_mdm = r["ternary"] - r["fp16"]
    gap_ar = r["ar_ternary"] - r["ar_fp16"]

    print("\n            objective |    FP16    | ternary |   gap    | ppl penalty")
    print("-" * 66)
    for obj, (f, t) in {"masked diffusion": ("fp16", "ternary"),
                        "autoregressive  ": ("ar_fp16", "ar_ternary")}.items():
        g = r[t] - r[f]
        print(f"  {obj} | {r[f]:10.4f} | {r[t]:7.4f} | {g:+8.4f} | "
              f"{100*(math.exp(r[t])/math.exp(r[f])-1):+6.1f}%")

    print(f"\nCE values across the two objectives are NOT comparable (different tasks).")
    print(f"The GAPS are. That is the whole design.\n")

    if gap_ar <= 0:
        print(f"gap_AR = {gap_ar:+.4f} <= 0: ternary matched or beat FP16 under AR. Striking, and it")
        print("means the ratio is undefined — report the two gaps, not the ratio.")
        return

    ratio = gap_mdm / gap_ar
    print(f"gap_MDM = {gap_mdm:+.4f}")
    print(f"gap_AR  = {gap_ar:+.4f}")
    print(f"RATIO   = {ratio:.2f}x\n")

    # A ratio of two noisy differences is itself noisy. Say so before anyone quotes it.
    err = NOISE * math.sqrt(2)          # std of a difference of two runs
    print(f"CAVEAT: each gap is a difference of two single-seed runs, so each carries ~{err:.3f}")
    print(f"        of noise (run-to-run std is {NOISE}). A ratio of two noisy differences is worse")
    print(f"        still. Before this goes in the paper it needs PAIRED SEEDS on all four cells.")
    if abs(gap_mdm - gap_ar) < 2 * err:
        print(f"\n  --> the two gaps differ by {abs(gap_mdm-gap_ar):.4f}, which is INSIDE 2x the noise.")
        print("      On this evidence the honest statement is: NO DETECTABLE DIFFERENCE between")
        print("      the objectives. Do not claim a ratio.")
    else:
        print(f"\n  --> the two gaps differ by {abs(gap_mdm-gap_ar):.4f}, which is OUTSIDE 2x the noise.")
        print("      Suggestive. Confirm with 3 paired seeds before writing it down.")


if __name__ == "__main__":
    main()
