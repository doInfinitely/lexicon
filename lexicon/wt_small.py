"""Is the lexeme language's SMALL-TRUNK penalty partly false decompositions?

Remy's argument: a false edge makes the lexeme's meaning context-dependent.
Reconciled with the entropy bound (0.0034 bits/char, info is recoverable from the
operator token), the harm is not information but CAPACITY: a correctly-lexicalised
word is a one-position lookup; a falsely-decomposed one is a two-position
composition (blended lexeme + a shared operator token carrying no word identity).

Prediction: the cost scales inversely with trunk size. A big trunk has heads to
spare for the extra composition; a small trunk does not. The lexeme language
currently LOSES at small trunks (+0.052 at L4, +0.122 at L2 on TinyStories) and I
have been blaming sequence length alone.

So: v1 (contaminated) vs v3 (MorphyNet, clean) at L2/L4, where the effect should
live. If cleaning recovers a chunk of the small-trunk penalty, Remy is right that
this matters, and it matters exactly where we want it to.
"""
import json, collections
import numpy as np, torch
from lexicon.ts_lm import GPT, DEVICE, OUT
from lexicon.ts_eval2 import bits_per_char
from lexicon.ts_scale import trunk_params
from lexicon.wt_scale import BPETok, corpus, train_and_loss, N_PARA
from lexicon.wt_seeds2 import Lex

SEEDS = [0, 1, 2]
CONFIGS = [(2, 128, 4), (4, 256, 4)]

def main():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts = []
    for r in ds:
        t = r["text"]
        if len(t) > 200 and not t.lstrip().startswith("="): texts.append(t)
        if len(texts) >= N_PARA: break
    tr, ev = texts[:-2000], texts[-2000:]

    toks = {"bpe": BPETok(tr),
            "lex-v1": Lex(tr, "dictionary/forest.json",    False, True, "lex-v1"),
            "lex-v3": Lex(tr, "dictionary/forest_v3.json", True,  True, "lex-v3")}
    data = {n: corpus(t, tr) for n, t in toks.items()}
    for n in toks:
        print(f"{n:<8} vocab {len(toks[n].itos):>6}  tokens {len(data[n]):>12,} "
              f"{len(data[n])/len(data['bpe']):.3f}x", flush=True)

    res = collections.defaultdict(list)
    for layers, dd, hh in CONFIGS:
        for seed in SEEDS:
            for n, tok in toks.items():
                torch.manual_seed(seed); np.random.seed(seed)
                m = GPT(len(tok.itos), d=dd, layers=layers, heads=hh).to(DEVICE)
                train_and_loss(m, data[n], len(tok.itos))
                b = bits_per_char(m, tok, ev)
                res[(layers, n)].append(b)
                print(f"  L{layers} seed {seed} {n:<8} bits/char {b:.4f}", flush=True)
                del m; torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print(f"{'trunk':<10}{'bpe':>16}{'lex-v1':>16}{'lex-v3':>16}{'v3-v1':>12}")
    print("-" * 72)
    for layers, dd, _ in CONFIGS:
        c = {n: np.array(res[(layers, n)]) for n in toks}
        print(f"L{layers} d{dd:<6}" + "".join(f"{c[n].mean():>10.3f}+-{c[n].std(ddof=1):<5.3f}"
              for n in toks) + f"{(c['lex-v3']-c['lex-v1']).mean():>+12.3f}")
    print("\nnegative v3-v1 = cleaning the dictionary helped")
    print("prediction: |v3-v1| grows as the trunk shrinks")
    json.dump({f"L{k[0]}_{k[1]}": v for k, v in res.items()},
              open(f"{OUT}/wt_small.json", "w"), indent=1)

if __name__ == "__main__":
    main()
