"""Does derivation earn its place? v6 = inflection only, no derivational operators.

v4 (best arm, -0.0736) has 12,867 inflectional edges and only 125 derivational ones.
It is nearly inflection-only. If v6 matches or beats it, the 'operator language' is
really just morphological inflection, and every derivational operator we built --
suf.ly, noun.agent, the whole atlas -- contributes nothing to a language model.

I predicted the story 'inflection pools, derivation does not' AFTER seeing v1..v3, then
watched v4 refute the frequency half of it. This is the prediction stated BEFORE the run:
  v6 within 0.01 of v4  => derivation contributes nothing; the language is inflection.
  v6 clearly worse      => the 125 frequent derivations are doing real work.
"""
import json
import numpy as np, torch
from lexicon.ts_lm import GPT, DEVICE, OUT
from lexicon.ts_eval2 import bits_per_char
from lexicon.wt_scale import BPETok, corpus, train_and_loss, N_PARA
from lexicon.wt_seeds2 import Lex, SEEDS, L, D, H

def main():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="train")
    texts=[]
    for r in ds:
        t=r["text"]
        if len(t)>200 and not t.lstrip().startswith("="): texts.append(t)
        if len(texts)>=N_PARA: break
    tr, ev = texts[:-2000], texts[-2000:]
    nb = len(corpus(BPETok(tr), tr))
    tok = Lex(tr, "dictionary/forest_v6.json", True, True, "lex-v6")
    data = corpus(tok, tr)
    print(f"lex-v6 vocab {len(tok.itos)}  tokens {len(data):,}  {len(data)/nb:.3f}x bpe\n", flush=True)
    res=[]
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        m = GPT(len(tok.itos), d=D, layers=L, heads=H).to(DEVICE)
        train_and_loss(m, data, len(tok.itos))
        b = bits_per_char(m, tok, ev); res.append(b)
        print(f"  seed {seed}  lex-v6  bits/char {b:.4f}", flush=True)
        del m; torch.cuda.empty_cache()
    prev = json.load(open(f"{OUT}/wt_seeds2.json"))
    base = np.array(prev["bpe"]); a = np.array(res)
    prev["lex-v6"]=res; prev["lex-v6_tokens"]=len(data)/nb
    json.dump(prev, open(f"{OUT}/wt_seeds2.json","w"), indent=1)
    print("\n" + "="*64)
    print(f"{'arm':<10}{'mean':>9}{'sd':>9}{'vs bpe':>10}{'tokens':>9}{'compute-eq':>12}")
    print("-"*64)
    BPD = 0.075/np.log2(2.37)
    TOK = {"bpe":1.0,"lex-v1":1.232,"lex-v2":1.251,"lex-v3":1.283,
           "lex-v4":1.229,"lex-v5":1.225,"lex-v6":len(data)/nb}
    for k in ("bpe","lex-v1","lex-v5","lex-v4","lex-v6","lex-v2","lex-v3"):
        if k not in prev: continue
        x=np.array(prev[k]); d=(x-base).mean(); t=TOK[k]
        ce = (2**(-d/BPD))/t if k!="bpe" else 1.0
        print(f"{k:<10}{x.mean():>9.4f}{x.std(ddof=1):>9.4f}{d:>+10.4f}{t:>8.2f}x{ce:>11.2f}x")
    v4=np.array(prev["lex-v4"]); dd=a-v4
    print(f"\nv6 - v4 = {dd.mean():+.4f} +- {dd.std(ddof=1):.4f}")
    print("  >=0 (v6 no worse): derivation contributes NOTHING; the language is inflection")
    print("  clearly positive : the 125 frequent derivations do real work")

if __name__=="__main__":
    main()
