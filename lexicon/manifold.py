"""Antonymy as reflection across a LEARNED CURVED SURFACE.

The flat mirror (Householder, x - 2vv'x) reached R@1 ~0.16. Remy's proposal: the
mirror need not be flat. Fit a nonlinear surface that lies BETWEEN all antonym
pairs -- the decision boundary of a nonlinear classifier separating each word
from its opposite -- and reflect across it.

Make it precise. Let g: R^d -> R be a scalar field. The surface is its zero
level set {g = 0}. Reflection across that surface, for a point x:

    project   x_p <- x - (g(x) / ||grad g(x)||^2) grad g(x)     (Newton step,
                                                                 iterated)
    reflect   x'  =  2 x_p - x

For linear g this is exactly the Householder reflection, so the flat mirror is
the special case. For curved g it is a genuine involution to first order, and
the curvature is what buys us the words the flat mirror missed.

Supervision, three parts:
  polarity   every antonym pair must straddle the surface: sign(g(a)) = -sign(g(b)),
             with margin. This needs a GLOBAL side-assignment, so we 2-colour the
             antonym graph. Whether that graph is bipartite is itself a fact worth
             knowing: if it is not, no consistent "side of the mirror" exists.
  midpoint   g((a+b)/2) = 0 -- the surface passes between the pair.
  retrieval  reflect(a) must retrieve b out of 51k words (InfoNCE). This is the
             objective we are actually scored on; the other two shape the surface.

Baselines: the flat mirror (linear g, identical machinery), an unconstrained MLP
operator, and identity. Held out by WORD.
"""
import json, os, collections, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lexicon.paradigm import abtt_space, DEVICE, D
from lexicon.atlas import split_words

TAU = 0.05


def two_colour(pairs):
    """Is the antonym graph bipartite? Returns colouring and #conflicts."""
    adj = collections.defaultdict(list)
    for a, b in pairs:
        adj[a].append(b); adj[b].append(a)
    colour, conflicts, comps = {}, 0, 0
    for s in adj:
        if s in colour:
            continue
        comps += 1
        colour[s] = 1
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in colour:
                    colour[v] = -colour[u]; stack.append(v)
                elif colour[v] == colour[u]:
                    conflicts += 1
    return colour, conflicts, comps, len(adj)


class Field(nn.Module):
    """g(x). `linear=True` reproduces the flat mirror exactly."""

    def __init__(self, d=768, h=512, linear=False):
        super().__init__()
        self.net = (nn.Linear(d, 1) if linear else
                    nn.Sequential(nn.Linear(d, h), nn.GELU(),
                                  nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1)))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def reflect(g, x, steps=2, create_graph=True):
    """Reflect x across {g = 0}: Newton-project, then mirror."""
    xp = x
    for _ in range(steps):
        xp = xp.detach().requires_grad_(True) if not create_graph else xp
        gv = g(xp)
        (grad,) = torch.autograd.grad(gv.sum(), xp, create_graph=create_graph)
        denom = (grad * grad).sum(-1, keepdim=True).clamp(min=1e-6)
        xp = xp - (gv.unsqueeze(-1) / denom) * grad
    return 2 * xp - x


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    T = abtt_space(torch.stack([protos[w] for w in vocab]).to(DEVICE))
    rels = json.load(open(f"{D}/relations.json"))
    ant = sorted({tuple(sorted(p)) for p in map(tuple, rels["lex:antonym"])
                  if p[0] in widx and p[1] in widx and p[0] != p[1]})

    colour, conflicts, comps, nw = two_colour(ant)
    print(f"antonym graph: {nw} words, {len(ant)} edges, {comps} components")
    print(f"2-colouring conflicts (odd cycles): {conflicts}")
    print(f"   -> the graph is {'BIPARTITE: a global side-of-the-mirror exists' if conflicts==0 else 'NOT bipartite; no consistent global polarity'}\n")

    tr_p, te_p = split_words(ant, frac=0.25)
    pos = collections.defaultdict(set)
    for a, b in ant:
        pos[a].add(widx[b]); pos[b].add(widx[a])
    train = [(a, b) for a, b in tr_p] + [(b, a) for a, b in tr_p]
    print(f"word-level split: train {len(tr_p)} pairs, held-out {len(te_p)}; "
          f"retrieval over {len(vocab)}\n")

    def run(linear, seed, steps=1500, bs=256, w_mid=1.0, w_pol=0.3):
        torch.manual_seed(seed)
        g = Field(linear=linear).to(DEVICE)
        opt = torch.optim.AdamW(g.parameters(), lr=3e-4, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        rng = random.Random(seed)
        for _ in range(steps):
            b = [train[rng.randrange(len(train))] for _ in range(bs)]
            s = torch.tensor([widx[a] for a, _ in b], device=DEVICE)
            t = torch.tensor([widx[c] for _, c in b], device=DEVICE)
            x = T[s].clone().requires_grad_(True)
            out = F.normalize(reflect(g, x), dim=-1)
            neg = torch.randint(0, len(T), (2048,), device=DEVICE, generator=gen)
            cand = torch.cat([t, neg]).unique()
            inv = {int(c): j for j, c in enumerate(cand.tolist())}
            lg = out @ T[cand].T / TAU
            pm = torch.zeros_like(lg, dtype=torch.bool)
            for i, (a, _) in enumerate(b):
                for w in pos[a]:
                    j = inv.get(w)
                    if j is not None:
                        pm[i, j] = True
            best = lg.masked_fill(~pm, float("-inf")).max(1).values
            loss = (torch.logsumexp(torch.cat(
                [best.unsqueeze(1), lg.masked_fill(pm, float("-inf"))], 1), 1) - best).mean()
            # the surface passes between each pair
            mid = (T[s] + T[t]) / 2
            loss = loss + w_mid * g(mid).pow(2).mean()
            # each pair straddles it, with margin, using the 2-colouring
            ya = torch.tensor([colour.get(a, 1) for a, _ in b], device=DEVICE, dtype=torch.float)
            loss = loss + w_pol * F.relu(1 - ya * g(T[s])).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()

        with torch.no_grad():
            pass
        s = torch.tensor([widx[a] for a, _ in te_p], device=DEVICE)
        t = torch.tensor([widx[b] for _, b in te_p], device=DEVICE)
        x = T[s].clone().requires_grad_(True)
        out = F.normalize(reflect(g, x, create_graph=False), dim=-1).detach()
        sims = out @ T.T
        sims.scatter_(1, s.unsqueeze(1), -2)
        top1 = sims.argmax(1)
        r1 = (top1 == t).float().mean().item()
        r5 = (sims.topk(5, 1).indices == t.unsqueeze(1)).any(1).float().mean().item()
        anyv = float(np.mean([top1[i].item() in pos[a] for i, (a, _) in enumerate(te_p)]))
        # involution fidelity: reflect twice
        y = out.clone().requires_grad_(True)
        back = reflect(g, y, create_graph=False).detach()
        rt = F.cosine_similarity(back, T[s], dim=-1).mean().item()
        return r1, r5, anyv, rt

    print(f"{'mirror':<26}{'R@1':>16}{'R@5':>9}{'R@1 any':>10}{'round-trip':>13}")
    print("-" * 76)
    for name, lin in (("flat (linear g)", True), ("curved (MLP g)", False)):
        rs = [run(lin, s) for s in (0, 1, 2)]
        m = np.mean(rs, axis=0); sd = np.std([r[0] for r in rs])
        print(f"{name:<26}{m[0]:>10.3f} +/-{sd:.3f}{m[1]:>9.3f}{m[2]:>10.3f}{m[3]:>13.3f}",
              flush=True)
    # references
    with torch.no_grad():
        s = torch.tensor([widx[a] for a, _ in te_p], device=DEVICE)
        t = torch.tensor([widx[b] for _, b in te_p], device=DEVICE)
        sims = T[s] @ T.T
        sims.scatter_(1, s.unsqueeze(1), -2)
        print(f"{'identity (no operator)':<26}{(sims.argmax(1)==t).float().mean().item():>10.3f}")
    print(f"{'unconstrained MLP (earlier)':<26}{0.155:>10.3f}  (3 seeds, +/-0.004)")
    print("\nThe flat mirror is the linear special case of the same machinery, so")
    print("the two rows differ ONLY in whether the surface may curve.")


if __name__ == "__main__":
    main()
