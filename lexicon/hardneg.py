"""Attack the antonym operator with the negatives it actually fails on.

Diagnosis, established: the operator localises (R@50 = 0.669) and cannot select
(R@1 = 0.238). Its errors are the gold antonym's own neighbours -- `chilly` when
the answer is `cold`. Yet every antonym operator in this project was trained
against RANDOM negatives: closed-form least squares, or InfoNCE over words drawn
uniformly from 51k. A uniform negative is never the confusion, so the loss never
saw the problem.

Fix: mine negatives from where the model is wrong.

  random         negatives ~ Uniform(vocab).                 (what we did)
  hard-nn        negatives = the gold target's k nearest neighbours.
  hard-syn       negatives = the gold target's WordNet `similar_to` set,
                 plus the SOURCE's own neighbours (a common failure: f(hot)
                 returns something near `hot` rather than near `cold`).
  hard-both      union.

Everything else is held fixed: same word-level split, same space (abtt), same
architecture, same steps, 3 seeds. The only variable is where negatives come
from.

The null that matters: a model trained with hard negatives could simply learn
"never answer anything near the source", which would help on antonyms and be
nonsense. So we also report R@1 on the SOURCE's neighbours being excluded and
the fraction of predictions that are the gold's synonyms.
"""
import json, os, collections, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lexicon.paradigm import abtt_space, DEVICE, D
from lexicon.atlas import split_words

TAU = 0.05
K_NN = 20


class Op(nn.Module):
    def __init__(self, dim=768, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    T = abtt_space(torch.stack([protos[w] for w in vocab]).to(DEVICE))
    rels = json.load(open(f"{D}/relations.json"))

    ant = sorted({tuple(sorted(p)) for p in map(tuple, rels["lex:antonym"])
                  if p[0] in widx and p[1] in widx and p[0] != p[1]})
    tr_p, te_p = split_words(ant, frac=0.25)
    train = [(a, b) for a, b in tr_p] + [(b, a) for a, b in tr_p]
    print(f"antonym pairs {len(ant)}; word-level split: train {len(tr_p)}, "
          f"held-out {len(te_p)}; retrieval over {len(vocab)} words\n")

    # every sanctioned antonym, for the positive mask
    pos = collections.defaultdict(set)
    for a, b in ant:
        pos[a].add(widx[b]); pos[b].add(widx[a])

    # precompute nearest neighbours and similar_to sets
    with torch.no_grad():
        nn_idx = torch.empty(len(T), K_NN, dtype=torch.long, device=DEVICE)
        for i in range(0, len(T), 2048):
            s = T[i:i + 2048] @ T.T
            s.scatter_(1, torch.arange(i, min(i + 2048, len(T)), device=DEVICE).unsqueeze(1), -2)
            nn_idx[i:i + 2048] = s.topk(K_NN, dim=1).indices
    sim = collections.defaultdict(set)
    for a, b in map(tuple, rels.get("lex:similar_to", [])):
        if a in widx and b in widx:
            sim[a].add(widx[b]); sim[b].add(widx[a])

    def negatives(kind, src, tgt, gen, n_rand=2048):
        rnd = torch.randint(0, len(T), (n_rand,), device=DEVICE, generator=gen)
        if kind == "random":
            return rnd
        hard = []
        if kind in ("hard-nn", "hard-both"):
            hard.append(nn_idx[tgt].reshape(-1))
        if kind in ("hard-syn", "hard-both"):
            hard.append(nn_idx[src].reshape(-1))
            ss = [i for w in tgt.tolist() for i in list(sim[vocab[w]])[:5]]
            if ss:
                hard.append(torch.tensor(ss, device=DEVICE))
        return torch.cat([rnd] + hard).unique()

    def run(kind, seed, steps=1500, bs=256):
        torch.manual_seed(seed)
        op = Op().to(DEVICE)
        opt = torch.optim.AdamW(op.parameters(), lr=1e-3, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        rng = random.Random(seed)
        for st in range(steps):
            batch = [train[rng.randrange(len(train))] for _ in range(bs)]
            s = torch.tensor([widx[a] for a, _ in batch], device=DEVICE)
            t = torch.tensor([widx[b] for _, b in batch], device=DEVICE)
            out = F.normalize(op(T[s]), dim=-1)
            neg = negatives(kind, s, t, gen)
            cand = torch.cat([t, neg]).unique()
            inv = {int(c): j for j, c in enumerate(cand.tolist())}
            logits = out @ T[cand].T / TAU
            pm = torch.zeros_like(logits, dtype=torch.bool)
            for i, (a, _) in enumerate(batch):
                for w in pos[a]:
                    j = inv.get(w)
                    if j is not None:
                        pm[i, j] = True
            best = logits.masked_fill(~pm, float("-inf")).max(1).values
            loss = (torch.logsumexp(torch.cat(
                [best.unsqueeze(1), logits.masked_fill(pm, float("-inf"))], 1), 1) - best).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()

        with torch.no_grad():
            s = torch.tensor([widx[a] for a, _ in te_p], device=DEVICE)
            t = torch.tensor([widx[b] for _, b in te_p], device=DEVICE)
            out = F.normalize(op(T[s]), dim=-1)
            sims = out @ T.T
            sims.scatter_(1, s.unsqueeze(1), -2)
            top1 = sims.argmax(1)
            r1 = (top1 == t).float().mean().item()
            r5 = (sims.topk(5, 1).indices == t.unsqueeze(1)).any(1).float().mean().item()
            anyv = np.mean([top1[i].item() in pos[a] for i, (a, _) in enumerate(te_p)])
            # is it just avoiding the source's neighbourhood?
            near_src = (top1.unsqueeze(1) == nn_idx[s]).any(1).float().mean().item()
            near_gold = (top1.unsqueeze(1) == nn_idx[t]).any(1).float().mean().item()
        return r1, r5, float(anyv), near_src, near_gold

    print(f"{'negatives':<14}{'R@1':>16}{'R@5':>9}{'R@1 any':>10}"
          f"{'pred near src':>15}{'pred near gold':>16}")
    print("-" * 82)
    for kind in ("random", "hard-nn", "hard-syn", "hard-both"):
        rs = [run(kind, s) for s in (0, 1, 2)]
        m = np.mean(rs, axis=0); sd = np.std([r[0] for r in rs])
        print(f"{kind:<14}{m[0]:>10.3f} +/-{sd:.3f}{m[1]:>9.3f}{m[2]:>10.3f}"
              f"{m[3]:>15.3f}{m[4]:>16.3f}", flush=True)

    print("\n'pred near gold' should RISE if the model learns to land in the right")
    print("region; 'pred near src' should fall if it stops answering with the")
    print("source's own neighbours. If R@1 rises only because 'near src' collapses,")
    print("the model learned an avoidance rule, not antonymy.")


if __name__ == "__main__":
    main()
