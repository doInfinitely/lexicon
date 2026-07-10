"""Does the lexeme advantage GROW with trunk size, or was L6 a lucky point?

Known: delta (lexeme - word, bits/char) by trunk, 58k stories, equal steps:
    L1 0.11M +0.096 | L2 0.40M +0.122 | L3 1.33M +0.118 | L4 3.16M +0.052 | L6 10.65M -0.098
One point below zero. A trend needs more.

Two problems with just adding bigger trunks at 58k stories:
  - 12M word tokens against a 25M-param trunk is the OVERFITTING regime; any
    delta up there may be about which model overfits more gracefully. The lexeme
    model has half the vocabulary, which reduces overfitting independently.
  - so: more data (150k stories), and report train loss beside eval bits/char
    so overfitting is visible rather than silently driving the answer.
"""
import collections, math, json
import numpy as np
import torch
import torch.nn.functional as F

from lexicon.ts_lm import WordTok, GPT, build, tokenize_corpus, DEVICE, OUT, CTX
from lexicon.ts_postfix import PostfixTok
from lexicon.ts_eval2 import bits_per_char
from lexicon.ts_scale import trunk_params

CONFIGS = [(4, 256, 4), (6, 384, 6), (8, 512, 8)]
STEPS = 4000
N_STORIES = 150000


def train(m, data, vocab_size, steps=STEPS, bs=24, lr=6e-4, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    d = torch.from_numpy(data).long()
    g = torch.Generator().manual_seed(seed)
    m.train(); last = []
    for s in range(steps):
        ix = torch.randint(0, len(d) - CTX - 1, (bs,), generator=g)
        x = torch.stack([d[i:i+CTX] for i in ix]).to(DEVICE)
        y = torch.stack([d[i+1:i+1+CTX] for i in ix]).to(DEVICE)
        loss = F.cross_entropy(m(x).reshape(-1, vocab_size), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if s >= steps - 50:
            last.append(loss.item())
    return m, float(np.mean(last))


def main():
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    texts = [ds[i]["text"] for i in range(N_STORIES)]
    train_texts, clean_eval = texts[:-3000], texts[-3000:]
    print(f"train {len(train_texts):,} stories, eval {len(clean_eval):,}\n")

    toks = {"word": WordTok(train_texts), "lexeme": PostfixTok(train_texts)}
    data = {n: tokenize_corpus(t, train_texts) for n, t in toks.items()}
    for n, t in toks.items():
        print(f"{n:<10} vocab {len(t.itos):>6}   tokens {len(data[n]):>12,}")
    print()

    rows = {}
    for layers, d, heads in CONFIGS:
        for name, tok in toks.items():
            torch.manual_seed(0)
            m = GPT(len(tok.itos), d=d, layers=layers, heads=heads).to(DEVICE)
            tp = trunk_params(m)
            m, trloss = train(m, data[name], len(tok.itos))
            bpc = bits_per_char(m, tok, clean_eval)
            rows.setdefault((layers, d), {})[name] = (bpc, tp, trloss)
            print(f"  L{layers} d{d} {name:<8} trunk {tp/1e6:6.2f}M  "
                  f"train loss {trloss:.3f}  eval bits/char {bpc:.3f}", flush=True)
            del m; torch.cuda.empty_cache()

    print("\n" + "=" * 84)
    print(f"{'trunk':<14}{'params':>10}{'word bpc':>11}{'lexeme bpc':>13}"
          f"{'delta':>9}{'word trainL':>13}{'lex trainL':>12}")
    print("-" * 84)
    for (l, d), v in rows.items():
        w, lx = v["word"], v["lexeme"]
        print(f"L{l} d{d:<9}{w[1]/1e6:>9.2f}M{w[0]:>11.3f}{lx[0]:>13.3f}"
              f"{lx[0]-w[0]:>+9.3f}{w[2]:>13.3f}{lx[2]:>12.3f}")
    print("\nnegative delta = lexeme better.  If |delta| grows with trunk size, the")
    print("advantage scales. Compare train losses: if the gap tracks train loss, it")
    print("is overfitting, not representation.")
    json.dump({f"L{l}_d{d}": {k: v[0] for k, v in r.items()} for (l, d), r in rows.items()},
              open(f"{OUT}/bigscale.json", "w"), indent=1)


if __name__ == "__main__":
    main()
