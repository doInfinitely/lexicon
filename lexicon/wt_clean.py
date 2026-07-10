"""The honest experiment: byte fallback + early stopping.

TWO BUGS FIXED.

1. <unk> CHEAT. BPETok mapped OOV to a single frequent <unk> token while bits_per_char
   credited every character of the swallowed word. bpe unk rate 5.70%, lex-v6 2.87%.
   BPE cheated twice as hard and lost anyway -- so past results were conservative -- but
   both sides were contaminated. With a byte fallback the token ratio collapses from
   1.199x to 1.020x: the "20% length penalty" was mostly BPE's cheat, and length was the
   largest effect in the entire study.

2. STEPS ∝ DATA. The equal-epoch ladder gave steps = 268 / 1069 / 4295 across rungs, so
   "advantage vs corpus size" and "advantage vs training length" were the same axis.
   Equal-STEPS was no better: bpe's shorter stream meant more epochs and more overfitting.
   Both protocols biased toward the lexeme arm. Neither answers P1.

   Fix: warmup + CONSTANT lr (no schedule tied to a step budget), evaluate held-out every
   EVAL_EVERY steps, take each arm's OWN best checkpoint, and report the compute it needed.
   Now corpus size varies while each arm sits at its own optimum; neither undertraining nor
   overfitting decides the winner.

Reports quality at the optimum AND tokens processed to reach it -- the two numbers a
compute-efficiency claim actually needs.
"""
import json, collections, math
import numpy as np, torch, torch.nn.functional as F
from lexicon.ts_lm import GPT, DEVICE, OUT, CTX
from lexicon.ts_eval2 import bits_per_char
from lexicon.wt_scale import corpus, N_PARA
from lexicon.bytetok import ByteBPETok, ByteLexTok

SEEDS = [0, 1]
L, D, H = 6, 384, 6
RUNGS = [10000, 40000, 160000]
import os
BS, WARMUP, EVAL_EVERY, PATIENCE = 24, 200, 250, 5
MAXSTEPS = int(os.environ.get('MAXSTEPS', 5000))
LR = 6e-4


def train_early_stop(m, data, V, tok, ev, seed):
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.1)
    d = torch.from_numpy(data).long()
    g = torch.Generator().manual_seed(seed)
    best, best_step, bad = float("inf"), 0, 0
    for s in range(1, MAXSTEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, s / WARMUP)
        m.train()
        ix = torch.randint(0, len(d) - CTX - 1, (BS,), generator=g)
        x = torch.stack([d[i:i+CTX] for i in ix]).to(DEVICE)
        y = torch.stack([d[i+1:i+1+CTX] for i in ix]).to(DEVICE)
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if s % EVAL_EVERY == 0:
            b = bits_per_char(m, tok, ev)
            if b < best - 1e-4: best, best_step, bad = b, s, 0
            else:
                bad += 1
                if bad >= PATIENCE: break
    return best, best_step


def main():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="train")
    texts=[]
    for r in ds:
        t=r["text"]
        if len(t)>200 and not t.lstrip().startswith("="): texts.append(t)
        if len(texts)>=N_PARA: break
    full_tr, ev = texts[:-2000], texts[-2000:]

    res, steps_at = collections.defaultdict(list), collections.defaultdict(list)
    ratios = {}
    for rung in RUNGS:
        tr = full_tr[:rung]
        toks = {"bpe": ByteBPETok(tr),
                "lex-v6": ByteLexTok(tr,"dictionary/forest_v6.json","lex-v6"),
                "lex-v3": ByteLexTok(tr,"dictionary/forest_v3.json","lex-v3")}
        data = {n: corpus(t, tr) for n,t in toks.items()}
        nb = len(data["bpe"])
        ratios[rung] = {n: len(data[n])/nb for n in toks}
        print(f"\n--- {rung:,} paragraphs ---")
        for n in toks:
            print(f"  {n:<8} vocab {len(toks[n].itos):>6}  tokens {len(data[n]):>11,}  "
                  f"{ratios[rung][n]:.3f}x", flush=True)
        for seed in SEEDS:
            for n, tok in toks.items():
                torch.manual_seed(seed); np.random.seed(seed)
                m = GPT(len(tok.itos), d=D, layers=L, heads=H).to(DEVICE)
                b, st = train_early_stop(m, data[n], len(tok.itos), tok, ev, seed)
                res[(rung,n)].append(b); steps_at[(rung,n)].append(st)
                print(f"  seed {seed} {n:<8} best bits/char {b:.4f} @ step {st}", flush=True)
                del m; torch.cuda.empty_cache()
        json.dump({f"{k[0]}_{k[1]}": {"bpc": res[k], "step": steps_at[k]} for k in res},
                  open(f"{OUT}/wt_clean.json","w"), indent=1)

    print("\n" + "="*88)
    print(f"{'paragraphs':<12}{'arm':<9}{'bits/char':>11}{'sd':>8}{'steps*':>8}"
          f"{'tokens':>9}{'vs bpe':>10}{'compute':>10}")
    print("-"*88)
    for rung in RUNGS:
        nb_steps = np.mean(steps_at[(rung,"bpe")])
        base = np.array(res[(rung,"bpe")]).mean()
        for n in ("bpe","lex-v6","lex-v3"):
            v = np.array(res[(rung,n)]); st = np.mean(steps_at[(rung,n)])
            comp = st / nb_steps                      # steps * bs * CTX, ratio to bpe
            print(f"{rung:<12,}{n:<9}{v.mean():>11.4f}{v.std(ddof=1):>8.4f}{st:>8.0f}"
                  f"{ratios[rung][n]:>8.2f}x{v.mean()-base:>+10.4f}{comp:>9.2f}x")
        print()
    print("steps* = steps to each arm's OWN best held-out checkpoint (early stopping)")
    print("compute = steps ratio vs bpe; token-count no longer inflates it (byte fallback)")
    print("\nP1: does (lex-v6 - bpe) get MORE negative as the corpus shrinks?")
    print("P2 (already robust): lex-v3 - lex-v6 > 0 everywhere -> derivation never pays")

if __name__=="__main__":
    main()
