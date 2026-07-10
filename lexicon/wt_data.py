"""Reframe (Remy): the payoff is DATA efficiency, not a smaller model.

Morphology is a sample-complexity device. BPE estimates walk/walks/walked/walking/walker
as five independent distributions; the lexeme model pools them into <lex:walk> + reusable
operators. That helps when ESTIMATION binds (data-limited), not when CAPACITY binds.

Which reinterprets everything: the lexeme advantage does not "grow with trunk size"
because big models like morphology. It grows because big models are the ones that have
run out of DATA. At L2 capacity binds -> pooling worthless, 20% token cost is pure loss.

SHARP CONSEQUENCE: the derivation deflation (v6 == v4, derivation worth 0) may be an
artifact of data abundance. `originally` occurs ~500x in 200k paragraphs -- enough to
estimate directly, so pooling it with `original` buys nothing. At 10k paragraphs it
occurs ~25x and the pooling should matter. Derived forms are RARER than inflected ones,
so derivation's value should grow FASTEST as data shrinks.

PREDICTIONS, stated before the run:
  P1  lexeme advantage over bpe GROWS as the corpus shrinks (all lexeme arms)
  P2  v3 (all derivation), worth -0.0010 at 200k, becomes clearly better than v6
      (inflection only) at 10k. If so, the atlas is un-retired -- as a data story.
  P3  if neither holds, morphology's benefit is not sample complexity and I have no
      account of it at all.

CONFOUND CAUGHT BEFORE RUNNING: fixed 4000 steps = 4000*24*512 = 49.2M tokens seen,
regardless of corpus size. At 200k paragraphs that is ~1.4 epochs; at 10k it is ~28.
Worse, it has a DIRECTION that favours the hypothesis: at a given rung bpe has FEWER
tokens (1.63M vs v6 1.96M), so bpe makes MORE passes over its own data and overfits
harder. The lexeme arms would look better at small data purely for having a longer token
stream. That would manufacture P1 out of nothing.

So: EQUAL EPOCHS. steps_arm = EPOCHS * tokens_arm / (bs*CTX). Every arm makes the same
number of passes over the same text. Compute then differs by exactly the token ratio
(1.199x for v6), which is the honest cost and which compute-equivalence already charges.
Cannot reuse the 200k rung from wt_seeds2 -- that was fixed-steps. Same held-out eval
(last 2000 paragraphs) throughout.
"""
import json, collections
import numpy as np, torch
from lexicon.ts_lm import GPT, DEVICE, OUT
from lexicon.ts_eval2 import bits_per_char
from lexicon.wt_scale import BPETok, corpus, train_and_loss, N_PARA
from lexicon.wt_seeds2 import Lex, L, D, H

SEEDS = [0, 1, 2]
RUNGS = [10000, 40000, 160000]
EPOCHS = 2
BS, CTX_ = 24, 512
ARMS = {"bpe": None,
        "lex-v6": "dictionary/forest_v6.json",     # inflection only
        "lex-v3": "dictionary/forest_v3.json"}     # + all 8,319 derivational edges


def main():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts = []
    for r in ds:
        t = r["text"]
        if len(t) > 200 and not t.lstrip().startswith("="): texts.append(t)
        if len(texts) >= N_PARA: break
    full_tr, ev = texts[:-2000], texts[-2000:]
    print(f"full train {len(full_tr):,} paragraphs; eval {len(ev):,} (held fixed)\n", flush=True)

    res = collections.defaultdict(list)
    for rung in RUNGS:
        tr = full_tr[:rung]
        nwords = sum(len(t.split()) for t in tr)
        toks = {}
        for n, path in ARMS.items():
            toks[n] = BPETok(tr) if path is None else Lex(tr, path, True, True, n)
        data = {n: corpus(t, tr) for n, t in toks.items()}
        nb = len(data["bpe"])
        print(f"--- {rung:,} paragraphs ({nwords/1e6:.2f}M words) ---")
        for n in ARMS:
            print(f"  {n:<8} vocab {len(toks[n].itos):>6}  tokens {len(data[n]):>11,}  "
                  f"{len(data[n])/nb:.3f}x", flush=True)
        for seed in SEEDS:
            for n in ARMS:
                tok = toks[n]
                torch.manual_seed(seed); np.random.seed(seed)
                m = GPT(len(tok.itos), d=D, layers=L, heads=H).to(DEVICE)
                steps = max(60, int(EPOCHS * len(data[n]) / (BS * CTX_)))
                train_and_loss(m, data[n], len(tok.itos), steps=steps)
                b = bits_per_char(m, tok, ev); res[(rung, n)].append(b)
                print(f"    seed {seed} {n:<8} steps {steps:>5}  bits/char {b:.4f}", flush=True)
                del m; torch.cuda.empty_cache()
        json.dump({f"{k[0]}_{k[1]}": v for k, v in res.items()},
                  open(f"{OUT}/wt_data.json", "w"), indent=1)

    print("\n" + "=" * 76)
    print(f"{'paragraphs':<14}{'bpe':>10}{'lex-v6 (infl)':>16}{'lex-v3 (+deriv)':>18}"
          f"{'v6-bpe':>10}{'v3-v6':>9}")
    print("-" * 76)
    for rung in RUNGS:
        if (rung, "bpe") not in res: continue
        b = np.array(res[(rung, "bpe")]); v6 = np.array(res[(rung, "lex-v6")])
        v3 = np.array(res[(rung, "lex-v3")])
        print(f"{rung:<14,}{b.mean():>10.3f}{v6.mean():>16.3f}{v3.mean():>18.3f}"
              f"{(v6-b).mean():>+10.4f}{(v3-v6).mean():>+9.4f}")
    print("\nEQUAL EPOCHS (2 passes over each arm's own stream); compute differs by the")
    print("token ratio, which compute-equivalence charges separately.")
    print("\nP1: v6-bpe should become MORE negative as the corpus shrinks")
    print("P2: v3-v6 should become NEGATIVE at 10k (derivation earns its place when data is scarce)")


if __name__ == "__main__":
    main()
