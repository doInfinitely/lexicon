"""Two minimal contrasts against lex-v1, one variable each. bpe/v1 reused (deterministic).

  v4  correct morphology, v1's TOKEN COST, but 125 FREQUENT derivations
      (v1 spends the same tokens on 3,688 RARE ones)
      -> isolates the frequency profile of derivational decomposition
  v5  v1 with ONLY its 587 false derivational edges deleted; same rules, same slot
      names, same buggy operator order, everything else identical
      -> isolates false decompositions, the thing Remy called poison

v5 ~ v1  => false edges neutral (entropy bound said <=0.0034 bits/char)
v5 < v1  => removing them helps; the fix was right, I just paid for it in tokens
v5 > v1  => they are load-bearing, and I have no explanation
"""
import json
import numpy as np, torch
from lexicon.ts_lm import GPT, DEVICE, OUT
from lexicon.ts_eval2 import bits_per_char
from lexicon.wt_scale import BPETok, corpus, train_and_loss, N_PARA
from lexicon.wt_seeds2 import Lex, SEEDS, L, D, H

def main():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts = []
    for r in ds:
        t = r["text"]
        if len(t) > 200 and not t.lstrip().startswith("="): texts.append(t)
        if len(texts) >= N_PARA: break
    tr, ev = texts[:-2000], texts[-2000:]
    nb = len(corpus(BPETok(tr), tr))

    arms = {"lex-v4": ("dictionary/forest_v4.json", True),
            "lex-v5": ("dictionary/forest_v5.json", False)}   # v5 keeps v1's buggy order
    prev = json.load(open(f"{OUT}/wt_seeds2.json"))
    base = np.array(prev["bpe"])
    for name, (path, order) in arms.items():
        tok = Lex(tr, path, order, True, name)
        data = corpus(tok, tr)
        print(f"\n{name}  vocab {len(tok.itos)}  tokens {len(data):,}  {len(data)/nb:.3f}x bpe",
              flush=True)
        res = []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            m = GPT(len(tok.itos), d=D, layers=L, heads=H).to(DEVICE)
            train_and_loss(m, data, len(tok.itos))
            b = bits_per_char(m, tok, ev); res.append(b)
            print(f"  seed {seed}  {name}  bits/char {b:.4f}", flush=True)
            del m; torch.cuda.empty_cache()
        prev[name] = res
        prev[name + "_tokens"] = len(data) / nb
        json.dump(prev, open(f"{OUT}/wt_seeds2.json","w"), indent=1)

    print("\n" + "=" * 62)
    print(f"{'arm':<10}{'mean':>9}{'sd':>9}{'vs bpe':>10}{'tokens':>9}")
    print("-" * 62)
    for k in ("bpe","lex-v1","lex-v2","lex-v3","lex-v4","lex-v5"):
        if k in prev:
            x = np.array(prev[k]); t = prev.get(k + "_tokens")
            tk = f"{t:.2f}x" if t else {"bpe":"1.00x","lex-v1":"1.23x","lex-v2":"1.25x",
                                        "lex-v3":"1.28x"}.get(k,"")
            print(f"{k:<10}{x.mean():>9.4f}{x.std(ddof=1):>9.4f}{(x-base).mean():>+10.4f}{tk:>9}")
    v1 = np.array(prev["lex-v1"]); v5 = np.array(prev["lex-v5"])
    d = v5 - v1
    print(f"\nv5 - v1 = {d.mean():+.4f} +- {d.std(ddof=1):.4f}   (bpe seed sd 0.0060)")
    print("  = the isolated cost of removing false decompositions")

if __name__ == "__main__":
    main()
