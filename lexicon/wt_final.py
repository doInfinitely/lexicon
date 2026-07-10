"""Does inflection-only survive at SMALL trunks? That is where the goal lives.

v6 (inflection only, 7 operators, 1.199x tokens) matches v4 and beats every richer
dictionary at L6: -0.0727, compute-eq 1.92x. Derivation contributes nothing.

But every trunk-scaling result we have used the contaminated, high-volume dictionaries,
where sequence length killed the lexeme language at small trunks (TinyStories L2 +0.122;
wikitext L2 v1 +0.032, v3 +0.108). v6 is 3% cheaper in tokens than v1 and much cheaper
than v3. Does the sign flip?

  L2 d128 / L4 d256   bpe vs lex-v4 vs lex-v6   -- is a SMALL model helped?
  L8 d512             bpe vs lex-v6             -- does the advantage survive scale?

Prediction, stated before the run: v6 still LOSES at L2. The length penalty is ~20% more
tokens and a 0.40M trunk cannot absorb it. If v6 WINS at L2, the 'smaller competent
model' thesis is alive and I was wrong to doubt it.
"""
import json, collections
import numpy as np, torch
from lexicon.ts_lm import GPT, DEVICE, OUT
from lexicon.ts_eval2 import bits_per_char
from lexicon.ts_scale import trunk_params
from lexicon.wt_scale import BPETok, corpus, train_and_loss, N_PARA
from lexicon.wt_seeds2 import Lex

SEEDS = [0, 1, 2]

def main():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="train")
    texts=[]
    for r in ds:
        t=r["text"]
        if len(t)>200 and not t.lstrip().startswith("="): texts.append(t)
        if len(texts)>=N_PARA: break
    tr, ev = texts[:-2000], texts[-2000:]

    toks = {"bpe": BPETok(tr),
            "lex-v4": Lex(tr,"dictionary/forest_v4.json",True,True,"lex-v4"),
            "lex-v6": Lex(tr,"dictionary/forest_v6.json",True,True,"lex-v6")}
    data = {n: corpus(t, tr) for n,t in toks.items()}
    nb = len(data["bpe"])
    for n in toks: print(f"{n:<8} tokens {len(data[n]):>12,}  {len(data[n])/nb:.3f}x", flush=True)

    PLAN = [((2,128,4), ["bpe","lex-v4","lex-v6"]),
            ((4,256,4), ["bpe","lex-v4","lex-v6"]),
            ((8,512,8), ["bpe","lex-v6"])]
    res = collections.defaultdict(list); tp = {}
    for (L,D,H), arms in PLAN:
        for seed in SEEDS:
            for n in arms:
                tok = toks[n]
                torch.manual_seed(seed); np.random.seed(seed)
                m = GPT(len(tok.itos), d=D, layers=L, heads=H).to(DEVICE)
                tp[(L,D)] = trunk_params(m)
                train_and_loss(m, data[n], len(tok.itos))
                b = bits_per_char(m, tok, ev); res[(L,n)].append(b)
                print(f"  L{L} seed {seed} {n:<7} bits/char {b:.4f}", flush=True)
                del m; torch.cuda.empty_cache()
        json.dump({f"L{k[0]}_{k[1]}": v for k,v in res.items()},
                  open(f"{OUT}/wt_final.json","w"), indent=1)

    print("\n" + "="*74)
    print(f"{'trunk':<12}{'params':>9}{'bpe':>10}{'lex-v4':>12}{'lex-v6':>12}{'v6 vs bpe':>12}")
    print("-"*74)
    for (L,D,H), arms in PLAN:
        row = f"L{L} d{D:<7}{tp[(L,D)]/1e6:>8.2f}M"
        b = np.array(res[(L,"bpe")])
        row += f"{b.mean():>10.3f}"
        row += f"{np.mean(res[(L,'lex-v4')]):>12.3f}" if (L,"lex-v4") in res else f"{'—':>12}"
        v6 = np.array(res[(L,"lex-v6")])
        row += f"{v6.mean():>12.3f}{(v6-b).mean():>+12.4f}"
        print(row)
    print("\nnegative = lexeme better. L6 reference: v6 -0.0727 at 1.199x tokens")
    print("prediction was: v6 still LOSES at L2 (length penalty on a 0.40M trunk)")

if __name__=="__main__":
    main()
