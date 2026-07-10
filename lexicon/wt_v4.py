"""v4: correct morphology at v1's TOKEN COST. Isolates correctness from volume.

v1->v3 is monotone in edges AND tokens AND correctness; all three confounded.
v4 = MorphyNet edges chosen most-frequent-first until derivational occurrence coverage
matches v1's. Same token cost, correct morphology (no station<-state, no number<-numb),
keeps worker<-work.

If v4 ~ v1: correct morphology is NEUTRAL; the v1->v3 gradient was sequence length.
If v4 > v1 (better): correctness HELPS once volume is controlled -- the fix was right,
                     I just paid for it with tokens.
If v4 < v1 (worse):  false decompositions were genuinely LOAD-BEARING. Would need an
                     explanation, and I do not have one.

bpe and lex-v1 are reused from wt_seeds2 (same harness, same seeds, deterministic).
"""
import json, collections
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
    bpe = BPETok(tr); nb = len(corpus(bpe, tr))
    tok = Lex(tr, "dictionary/forest_v4.json", True, True, "lex-v4")
    data = corpus(tok, tr)
    print(f"lex-v4 vocab {len(tok.itos)}  tokens {len(data):,}  {len(data)/nb:.3f}x bpe")
    print(f"  (v1 1.232x, v2 1.251x, v3 1.283x)\n", flush=True)
    res = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        m = GPT(len(tok.itos), d=D, layers=L, heads=H).to(DEVICE)
        train_and_loss(m, data, len(tok.itos))
        b = bits_per_char(m, tok, ev); res.append(b)
        print(f"  seed {seed}  lex-v4  bits/char {b:.4f}", flush=True)
        del m; torch.cuda.empty_cache()
    prev = json.load(open(f"{OUT}/wt_seeds2.json"))
    base = np.array(prev["bpe"]); a = np.array(res)
    print(f"\n{'arm':<10}{'mean':>9}{'sd':>9}{'vs bpe':>10}{'tokens':>9}")
    for k in ("bpe","lex-v1","lex-v2","lex-v3"):
        if k in prev:
            x = np.array(prev[k])
            print(f"{k:<10}{x.mean():>9.4f}{x.std(ddof=1):>9.4f}{(x-base).mean():>+10.4f}")
    print(f"{'lex-v4':<10}{a.mean():>9.4f}{a.std(ddof=1):>9.4f}{(a-base).mean():>+10.4f}"
          f"{len(data)/nb:>8.2f}x")
    prev["lex-v4"] = res
    json.dump(prev, open(f"{OUT}/wt_seeds2.json","w"), indent=1)

if __name__ == "__main__":
    main()
