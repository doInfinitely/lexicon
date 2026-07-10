"""How many polarity dimensions does English antonymy need? Clean answer.

The earlier k-sweep was run on the corrupted gold+indirect set (which we now
know degrades retrieval monotonically), so it measured the dimensionality of a
relation that is partly `antonym o similar_to`. This runs the sweep on GOLD
DIRECT antonyms only, word-level holdout, leak-fixed positive mask, 3 seeds.

If k=1 ties the best k, then "English antonymy is ~one polarity direction" is
a fact about the relation and not an artifact of having 1,331 examples.

Results are appended to real/english/k_sweep.json as they are computed, so a
killed session does not lose them.
"""
import json, os
import numpy as np

from lexicon.clean_expand import build, run

OUT = "real/english/k_sweep.json"


def main():
    vocab, widx, P, tr_gold, tr_clean, val, vw = build()
    print(f"gold direct training pairs : {len(tr_gold)}")
    print(f"held-out gold pairs        : {len(val)} (word-level, 38k retrieval)")
    print(f"seeds                      : 3\n", flush=True)

    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print(f"{'k':>6}{'held-out R@1':>20}", flush=True)
    print("-" * 27, flush=True)
    for k in (1, 2, 4, 8, 16, 32, 64):
        key = str(k)
        if key not in done:
            rs = [run(tr_gold, k, vocab, widx, P, val, vw, seed=s)
                  for s in (0, 1, 2)]
            done[key] = rs
            json.dump(done, open(OUT, "w"), indent=1)   # checkpoint each k
        rs = done[key]
        print(f"{k:>6}{np.mean(rs):>14.3f} +/- {np.std(rs):.3f}", flush=True)

    ks = sorted(done, key=int)
    means = {k: float(np.mean(done[k])) for k in ks}
    best = max(means, key=means.get)
    sd = float(np.std(done[best]))
    print(f"\nbest k = {best} at {means[best]:.3f} +/- {sd:.3f}")
    k1 = means["1"]
    tie = abs(means[best] - k1) <= 2 * sd
    print(f"k=1 scores {k1:.3f}; difference from best is "
          f"{'WITHIN' if tie else 'OUTSIDE'} 2 sd")
    print("\n=> " + ("English antonymy is ~ONE polarity direction; the earlier "
                     "k-sweep was not data-starved."
                     if tie else
                     "antonymy wants more than one dimension once data is "
                     "clean; the one-direction claim does not hold."), flush=True)


if __name__ == "__main__":
    main()
