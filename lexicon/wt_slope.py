"""Re-measure BPE's params->quality slope under HONEST encoding.

compute-equivalence converts a bits/char gain into "how much bigger would the baseline
have to be", using bpe's own L6->L8 curve. Every such number I have quoted used a slope
measured with the <unk>-contaminated tokenizer (bpe silently dropped 6% of words into one
frequent token). With a byte fallback bpe's stream is 35% longer and its losses differ, so
the slope must be re-measured or the conversion is meaningless.

Also gives lex-v6 at L8 under honest encoding, so the advantage can be read at two trunk
sizes rather than extrapolated.

PROTOCOL MISMATCH CAUGHT: the first version used train_and_loss (OneCycleLR, score at the
final step) and reported bpe L6 = 1.9553 on 160k paragraphs. wt_clean, same tokenizer,
same data, same 5000 steps, reported 1.7396 -- and the DELTA moved too (-0.1309 vs
-0.0748). The lexeme advantage depends on the LR schedule, so a slope measured under one
protocol cannot convert a gain measured under another. Both now use train_early_stop:
warmup + constant LR, evaluate every 250 steps, take each arm's own best checkpoint.
"""
import json, collections
import numpy as np, torch
from lexicon.ts_lm import GPT, DEVICE, OUT
from lexicon.ts_eval2 import bits_per_char
from lexicon.ts_scale import trunk_params
from lexicon.wt_scale import corpus, N_PARA
from lexicon.wt_clean import train_early_stop
from lexicon.bytetok import ByteBPETok, ByteLexTok

SEEDS = [0, 1]
STEPS = 5000
CONFIGS = [(6, 384, 6), (8, 512, 8)]
RUNG = 160000

def main():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="train")
    texts=[]
    for r in ds:
        t=r["text"]
        if len(t)>200 and not t.lstrip().startswith("="): texts.append(t)
        if len(texts)>=N_PARA: break
    tr, ev = texts[:-2000][:RUNG], texts[-2000:]
    toks = {"bpe": ByteBPETok(tr),
            "lex-v6": ByteLexTok(tr,"dictionary/forest_v6.json","lex-v6")}
    data = {n: corpus(t, tr) for n,t in toks.items()}
    nb = len(data["bpe"])
    for n in toks: print(f"{n:<8} tokens {len(data[n]):,}  {len(data[n])/nb:.3f}x", flush=True)

    res, tp = collections.defaultdict(list), {}
    for L,D,H in CONFIGS:
        for seed in SEEDS:
            for n, tok in toks.items():
                torch.manual_seed(seed); np.random.seed(seed)
                m = GPT(len(tok.itos), d=D, layers=L, heads=H).to(DEVICE)
                tp[L] = trunk_params(m)
                b, st = train_early_stop(m, data[n], len(tok.itos), tok, ev, seed)
                res[(L,n)].append(b)
                print(f"  L{L} seed {seed} {n:<8} best bits/char {b:.4f} @ step {st}", flush=True)
                del m; torch.cuda.empty_cache()
    json.dump({f"L{k[0]}_{k[1]}": v for k,v in res.items()},
              open(f"{OUT}/wt_slope.json","w"), indent=1)

    b6, b8 = np.mean(res[(6,"bpe")]), np.mean(res[(8,"bpe")])
    l6, l8 = np.mean(res[(6,"lex-v6")]), np.mean(res[(8,"lex-v6")])
    ratio = tp[8]/tp[6]
    bpd = (b6-b8)/np.log2(ratio)
    tokr = len(data["lex-v6"])/nb
    print("\n" + "="*70)
    print(f"trunk     params      bpe    lex-v6    delta")
    print(f"L6      {tp[6]/1e6:>7.2f}M{b6:>9.4f}{l6:>10.4f}{l6-b6:>+9.4f}")
    print(f"L8      {tp[8]/1e6:>7.2f}M{b8:>9.4f}{l8:>10.4f}{l8-b8:>+9.4f}")
    print(f"\nbpe slope: {b6-b8:.4f} bits/char for {ratio:.2f}x params "
          f"=> {bpd:.4f} bits per DOUBLING of trunk params")
    print(f"  (old, <unk>-contaminated slope was 0.0602)")
    for L,dl in ((6,l6-b6),(8,l8-b8)):
        ce = (2**(-dl/bpd))/tokr
        print(f"  L{L}: gain {-dl:.4f} = {2**(-dl/bpd):.2f}x params, "
              f"token cost {tokr:.3f}x  => compute-equivalent {ce:.2f}x")
    print(f"\nlex-v6 at L6 ({tp[6]/1e6:.1f}M) vs bpe at L8 ({tp[8]/1e6:.1f}M): "
          f"{l6:.4f} vs {b8:.4f}")

if __name__=="__main__":
    main()
