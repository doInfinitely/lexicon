"""The headline comparison, under ONE protocol.

wt_factored measured affine-vs-free with train_and_loss (OneCycleLR, fixed 2-epoch steps).
wt_slope showed the lexeme advantage MOVES with the LR schedule (-0.131 vs -0.075 on
identical configs). So the factored numbers cannot be quoted beside the slope. Redo them
with train_early_stop: warmup + constant LR, eval every 250 steps, each arm's own best
checkpoint, MAXSTEPS 9000.

Arms (all byte fallback, nothing ever <unk>):
    bpe     GPT-2 BPE, 16k vocab
    free    one token per word, 16k vocab, independent embeddings
    affine  same tokens; e(w) = E_root + d_slot + U_slot V_slot^T E_root, identity pinned

free - bpe    : the tokenizer effect (grows as data shrinks?)
affine - free : the morphological prior (flat across data in wt_factored: -0.0108/-0.0109)

DECOMPOSITION worth watching. From wt_factored, shufroot - free is -0.0037 at 10k and
+0.0077 at 40k: the low-rank REGULARISATION helps when data is scarce and hurts when it is
not. And affine - shufroot (the morphology proper) is -0.0071 at 10k and -0.0187 at 40k --
it GROWS with data. So the two components move in opposite directions, and the flat
affine-free total hides that. Add shufroot here to confirm across three rungs.
"""
import json, collections, os
import numpy as np, torch
from lexicon.ts_lm import GPT, DEVICE, OUT
from lexicon.ts_eval2 import bits_per_char
from lexicon.wt_scale import corpus, N_PARA
from lexicon.wt_clean import train_early_stop
from lexicon.bytetok import ByteBPETok
from lexicon.wt_factored import ByteWordTok, AffineGPT, SLOTS, init_scale

SEEDS = [0, 1]
L, D, H = 6, 384, 6
RUNGS = [10000, 40000, 160000]

def main():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="train")
    texts=[]
    for r in ds:
        t=r["text"]
        if len(t)>200 and not t.lstrip().startswith("="): texts.append(t)
        if len(texts)>=N_PARA: break
    full_tr, ev = texts[:-2000], texts[-2000:]

    res, steps_at, ratio = collections.defaultdict(list), collections.defaultdict(list), {}
    for rung in RUNGS:
        tr = full_tr[:rung]
        bpe = ByteBPETok(tr); bd = corpus(bpe, tr)
        tok = ByteWordTok(tr); data = corpus(tok, tr)
        R, S, nR, _ = tok.factorise()
        Rs, Ss, _, _ = tok.factorise(shuffle_seed=0)
        d0 = torch.zeros(len(SLOTS), D)
        ratio[rung] = len(data)/len(bd)
        print(f"\n--- {rung:,} paragraphs ---  bpe {len(bd):,}  word {len(data):,} "
              f"({ratio[rung]:.3f}x)  roots {nR:,}", flush=True)
        for seed in SEEDS:
            for arm in ("bpe","free","affine","shufroot"):
                torch.manual_seed(seed); np.random.seed(seed)
                if arm == "bpe":
                    m = GPT(len(bpe.itos), d=D, layers=L, heads=H); init_scale(m); m=m.to(DEVICE)
                    b, st = train_early_stop(m, bd, len(bpe.itos), bpe, ev, seed)
                elif arm == "free":
                    m = GPT(len(tok.itos), d=D, layers=L, heads=H); init_scale(m); m=m.to(DEVICE)
                    b, st = train_early_stop(m, data, len(tok.itos), tok, ev, seed)
                else:
                    rr, ss = (Rs, Ss) if arm=="shufroot" else (R, S)
                    m = AffineGPT(rr, ss, nR, d0, "affine"); init_scale(m); m=m.to(DEVICE)
                    b, st = train_early_stop(m, data, len(tok.itos), tok, ev, seed)
                res[(rung,arm)].append(b); steps_at[(rung,arm)].append(st)
                print(f"  seed {seed} {arm:<9} best {b:.4f} @ step {st}", flush=True)
                del m; torch.cuda.empty_cache()
        json.dump({f"{k[0]}_{k[1]}": {"bpc":res[k],"step":steps_at[k]} for k in res},
                  open(f"{OUT}/wt_best.json","w"), indent=1)

    A = ("bpe","free","affine","shufroot")
    print("\n" + "="*100)
    print(f"{'paragraphs':<12}" + "".join(f"{a:>11}" for a in A) +
          f"{'free-bpe':>11}{'aff-free':>11}{'aff-shuf':>11}{'shuf-free':>11}{'tokens':>9}")
    print("-"*100)
    for rung in RUNGS:
        r = {a: np.array(res[(rung,a)]) for a in A}
        print(f"{rung:<12,}" + "".join(f"{r[a].mean():>11.4f}" for a in A) +
              f"{(r['free']-r['bpe']).mean():>+11.4f}{(r['affine']-r['free']).mean():>+11.4f}"
              f"{(r['affine']-r['shufroot']).mean():>+11.4f}"
              f"{(r['shufroot']-r['free']).mean():>+11.4f}{ratio[rung]:>8.3f}x")
    print("\nfree-bpe   the tokenizer (does it grow as data shrinks?)")
    print("aff-shuf   morphology proper")
    print("shuf-free  low-rank regularisation alone")
    print("aff-free   = aff-shuf + shuf-free  (the two may move in opposite directions)")

if __name__=="__main__":
    main()
