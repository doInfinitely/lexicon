"""Third dictionary: MorphyNet with NO cosine filter. Reuses bpe from wt_seeds2
(deterministic given seed + same harness), so only 3 new trainings.

  v1  hand rules            deriv 3,688   has station<-state, number<-numb
  v2  MorphyNet + cosine    deriv 6,526   kills both, but also cuts worker<-work
  v3  MorphyNet only        deriv 8,328   kills both, KEEPS worker<-work
                                          (leaves MorphyNet's own errors: offer<-off)

v2 vs v3 isolates the cosine opacity filter. If v3 >= v2, the filter is cutting
more pooling than poison and should go.
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

    bpe = BPETok(tr)
    tok = Lex(tr, "dictionary/forest_v3.json", True, True, "lex-v3")
    nb, nl = len(corpus(bpe, tr)), None
    data = corpus(tok, tr); nl = len(data)
    print(f"lex-v3 vocab {len(tok.itos)}  tokens {nl:,}  {nl/nb:.3f}x bpe\n", flush=True)

    res = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        m = GPT(len(tok.itos), d=D, layers=L, heads=H).to(DEVICE)
        train_and_loss(m, data, len(tok.itos))
        b = bits_per_char(m, tok, ev)
        res.append(b); print(f"  seed {seed}  lex-v3  bits/char {b:.4f}", flush=True)
        del m; torch.cuda.empty_cache()

    prev = json.load(open(f"{OUT}/wt_seeds2.json"))
    base = np.array(prev["bpe"]); a = np.array(res); d = a - base
    print(f"\nlex-v3  mean {a.mean():.4f}  sd {a.std(ddof=1):.4f}  vs bpe {d.mean():+.4f}"
          f"  tokens {nl/nb:.2f}x")
    for k in ("lex-v1", "lex-v2"):
        if k in prev:
            x = np.array(prev[k])
            print(f"{k:<8} mean {x.mean():.4f}  sd {x.std(ddof=1):.4f}  "
                  f"vs bpe {(x-base).mean():+.4f}")
    prev["lex-v3"] = res
    json.dump(prev, open(f"{OUT}/wt_seeds2.json", "w"), indent=1)

if __name__ == "__main__":
    main()
