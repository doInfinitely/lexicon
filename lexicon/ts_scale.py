"""How small can the TRUNK be and still speak English?

Embedding tables are a lookup: zero FLOPs, no depth, no reasoning. Counting them
as "model size" confuses storage with capacity. TinyStories asks how small the
transformer can be; the embedding table rides along.

The lexeme language's real claim is therefore about the TRUNK. A word-level
model has to learn, inside its attention and MLPs, that walk / walks / walked /
walking / walker behave alike. A lexeme model is handed that by the tokenizer.
So the trunk should be able to shrink further before it stops being competent.

We sweep trunk size and hold everything else fixed. Quality is bits per
character -- the only length measure that does not depend on the tokenizer, so
a 6k-token lexeme vocabulary and a 12k-token word vocabulary are comparable.

    trunk params ~= layers * 12 * d^2      (attention + MLP; embeddings excluded)
"""
import json, math, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lexicon.ts_lm import (WordTok, GPT, build, tokenize_corpus, WORD_RE,
                           DEVICE, OUT, CTX)
from lexicon.ts_postfix import PostfixTok
from lexicon.ts_eval2 import bits_per_char

CONFIGS = [(6, 384, 6), (4, 256, 4), (3, 192, 6), (2, 128, 4), (1, 96, 4)]
STEPS = 3000


def trunk_params(m):
    n = 0
    for name, p in m.named_parameters():
        if name.startswith(("blocks", "ln")):
            n += p.numel()
    return n


def train(m, data, vocab_size, steps=STEPS, bs=24, lr=6e-4, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    d = torch.from_numpy(data).long()
    g = torch.Generator().manual_seed(seed)
    m.train()
    for s in range(steps):
        ix = torch.randint(0, len(d) - CTX - 1, (bs,), generator=g)
        x = torch.stack([d[i:i+CTX] for i in ix]).to(DEVICE)
        y = torch.stack([d[i+1:i+1+CTX] for i in ix]).to(DEVICE)
        loss = F.cross_entropy(m(x).reshape(-1, vocab_size), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    return m


def main():
    train_texts, clean_eval, test_texts, held = build()
    toks = {"word": WordTok(train_texts), "lexeme (postfix)": PostfixTok(train_texts)}
    for n, t in toks.items():
        print(f"{n:<20} vocab {len(t.itos)}")
    data = {n: tokenize_corpus(t, train_texts) for n, t in toks.items()}
    for n in toks:
        print(f"{n:<20} {len(data[n]):,} training tokens")

    rows = collections.defaultdict(dict)
    for layers, d, heads in CONFIGS:
        for name, tok in toks.items():
            torch.manual_seed(0)
            m = GPT(len(tok.itos), d=d, layers=layers, heads=heads).to(DEVICE)
            tp = trunk_params(m)
            emb = sum(p.numel() for n_, p in m.named_parameters()
                      if n_.startswith(("tok", "pos")))
            train(m, data[name], len(tok.itos))
            bpc = bits_per_char(m, tok, clean_eval)
            rows[(layers, d)][name] = (bpc, tp, emb)
            print(f"  L{layers} d{d}  {name:<18} trunk {tp/1e6:.2f}M  "
                  f"emb {emb/1e6:.2f}M  bits/char {bpc:.3f}", flush=True)
            del m; torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print("bits/char by TRUNK size (embeddings excluded from the size)")
    print("=" * 78)
    print(f"{'trunk':<16}{'trunk params':>14}{'word':>12}{'lexeme':>12}{'delta':>10}")
    print("-" * 78)
    for (layers, d), v in rows.items():
        if len(v) < 2:
            continue
        w, lx = v["word"][0], v["lexeme (postfix)"][0]
        tp = v["word"][1]
        print(f"L{layers} d{d:<11}{tp/1e6:>13.2f}M{w:>12.3f}{lx:>12.3f}{lx-w:>+10.3f}")
    print("\nnegative delta = the lexeme language is better at that trunk size.")
    print("If the lexeme curve stays flat as the trunk shrinks while the word")
    print("curve degrades, the language is doing work the trunk no longer has to.")
    json.dump({f"L{l}_d{d}": {k: v[0] for k, v in r.items()}
               for (l, d), r in rows.items()}, open(f"{OUT}/scale.json", "w"), indent=1)


if __name__ == "__main__":
    main()
