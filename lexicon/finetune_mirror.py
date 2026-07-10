"""Fine-tune the space so that the mirror is reachable.

Diagnosis: in frozen distilbert, antonym pairs lie at distance 0.633 from the
learned surface while the normal decorrelates completely across that gap
(cos(grad g(x), grad g(x')) = 0.062). Reflection across a curved surface is an
involution only INSIDE its reach. The words are outside it. So the curved mirror
scores 0.258 and round-trips at 0.63; forcing it to round-trip flattens it and
costs accuracy.

Remy: fine-tune the embeddings. Learn h (an adapter) and g (the surface)
together, so that in h-space each antonym pair straddles {g = 0} symmetrically
and close enough that reflection is exact.

    loss = InfoNCE( reflect_g(h(a)) -> h(b) )          retrieval, the objective
         + eikonal ||grad g|| = 1                      makes g a distance field
         + (g(h(a)) + g(h(b)))^2                       the pair straddles it
         + |g(h(a))| - margin                          ... within reach
         + anchor: h(x) stays near x                   no collapse

Nulls, because a free adapter absorbs structure (proven earlier: a random
reflection plane matched a trained one once an adapter could rotate):

    adapter + identity       does the adapter alone do the work?
    adapter + free MLP       is the mirror structure worth anything?
    frozen + curved mirror   the 0.258 we already have

Collateral check: `noun_plural` retrieval in the adapted space. If the adapter
buys antonymy by wrecking the rest of the geometry, that shows up here.
"""
import json, collections, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lexicon.paradigm import abtt_space, DEVICE, D
from lexicon.atlas import split_words
from lexicon.manifold import Field

TAU = 0.05


class Adapter(nn.Module):
    def __init__(self, d=768, h=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return F.normalize(x + self.net(x), dim=-1)


class FreeOp(nn.Module):
    def __init__(self, d=768, h=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, x):
        return x + self.net(x)


def grad_of(g, x):
    x = x.requires_grad_(True)
    v = g(x)
    (gr,) = torch.autograd.grad(v.sum(), x, create_graph=torch.is_grad_enabled())
    return v, gr


def reflect(g, x):
    v, gr = grad_of(g, x)
    return x - 2 * v.unsqueeze(1) * gr


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    T0 = abtt_space(torch.stack([protos[w] for w in vocab]).to(DEVICE))
    rels = json.load(open(f"{D}/relations.json"))
    ant = sorted({tuple(sorted(p)) for p in map(tuple, rels["lex:antonym"])
                  if p[0] in widx and p[1] in widx and p[0] != p[1]})
    tr_p, te_p = split_words(ant, frac=0.25)
    train = [(a, b) for a, b in tr_p] + [(b, a) for a, b in tr_p]
    pos = collections.defaultdict(set)
    for a, b in ant:
        pos[a].add(widx[b]); pos[b].add(widx[a])
    print(f"antonym pairs {len(ant)}; train {len(tr_p)}, held-out {len(te_p)}; "
          f"vocab {len(vocab)}\n")

    # collateral: a morphological relation that must not degrade
    pl = [(a, b) for a, b in map(tuple, rels["infl:noun_plural"])
          if a in widx and b in widx][:1500]

    def evaluate(h, op, kind):
        with torch.no_grad():
            tbl = torch.cat([h(T0[i:i + 4096]) for i in range(0, len(T0), 4096)])
        s = torch.tensor([widx[a] for a, _ in te_p], device=DEVICE)
        t = torch.tensor([widx[b] for _, b in te_p], device=DEVICE)
        x = tbl[s].clone()
        out = (reflect(op, x) if kind == "mirror" else
               (op(x) if kind == "free" else x))
        out = out.detach()
        sims = F.normalize(out, dim=-1) @ tbl.T
        sims.scatter_(1, s.unsqueeze(1), -2)
        r1 = (sims.argmax(1) == t).float().mean().item()
        if kind == "mirror":
            back = reflect(op, out.clone()).detach()
            rt = F.cosine_similarity(back, tbl[s], dim=-1).mean().item()
            v, gx = grad_of(op, tbl[s].clone())
            _, gp = grad_of(op, out.clone())
            cosn = F.cosine_similarity(gx, gp, dim=-1).mean().item()
            dist = v.abs().mean().item()
        else:
            rt = cosn = dist = float("nan")
        # collateral
        ps = torch.tensor([widx[a] for a, _ in pl], device=DEVICE)
        pt = torch.tensor([widx[b] for _, b in pl], device=DEVICE)
        d = (tbl[pt] - tbl[ps]).mean(0, keepdim=True)
        psim = F.normalize(tbl[ps] + d, dim=-1) @ tbl.T
        psim.scatter_(1, ps.unsqueeze(1), -2)
        plr1 = (psim.argmax(1) == pt).float().mean().item()
        return r1, rt, cosn, dist, plr1

    def run(kind, seed, steps=2500, bs=256, w_eik=30.0, w_sym=3.0,
            w_anchor=0.3, tune=True):
        torch.manual_seed(seed)
        h = Adapter().to(DEVICE) if tune else (lambda x: x)
        op = (Field(linear=False) if kind == "mirror" else FreeOp()).to(DEVICE) \
            if kind != "identity" else None
        params = ([] if not tune else list(h.parameters())) + \
                 ([] if op is None else list(op.parameters()))
        if not params:
            return evaluate(h, op, kind)
        opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        rng = random.Random(seed)
        for _ in range(steps):
            b = [train[rng.randrange(len(train))] for _ in range(bs)]
            s = torch.tensor([widx[a] for a, _ in b], device=DEVICE)
            t = torch.tensor([widx[c] for _, c in b], device=DEVICE)
            neg = torch.randint(0, len(T0), (2048,), device=DEVICE, generator=gen)
            cand = torch.cat([t, neg]).unique()
            ha, hc = h(T0[s]), h(T0[cand])
            out = (reflect(op, ha.clone()) if kind == "mirror" else
                   (op(ha) if kind == "free" else ha))
            lg = F.normalize(out, dim=-1) @ hc.T / TAU
            inv = {int(c): j for j, c in enumerate(cand.tolist())}
            pm = torch.zeros_like(lg, dtype=torch.bool)
            for i, (a, _) in enumerate(b):
                for w in pos[a]:
                    j = inv.get(w)
                    if j is not None:
                        pm[i, j] = True
            best = lg.masked_fill(~pm, float("-inf")).max(1).values
            loss = (torch.logsumexp(torch.cat(
                [best.unsqueeze(1), lg.masked_fill(pm, float("-inf"))], 1), 1) - best).mean()
            if kind == "mirror":
                hb = h(T0[t])
                ga, _ = grad_of(op, ha.clone())
                gb, _ = grad_of(op, hb.clone())
                loss = loss + w_sym * (ga + gb).pow(2).mean()          # straddle
                loss = loss + 0.3 * F.relu(ga.abs() - 0.25).mean()     # stay in reach
                rnd = h(T0[torch.randint(0, len(T0), (512,), device=DEVICE, generator=gen)])
                _, gr = grad_of(op, torch.cat([ha, hb, rnd]).clone())
                loss = loss + w_eik * (gr.norm(dim=-1) - 1).pow(2).mean()
            if tune:
                idx = torch.randint(0, len(T0), (1024,), device=DEVICE, generator=gen)
                loss = loss + w_anchor * (1 - F.cosine_similarity(
                    h(T0[idx]), T0[idx], dim=-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        return evaluate(h, op, kind)

    print(f"{'condition':<34}{'R@1':>9}{'round-trip':>12}{'cos(normal)':>13}"
          f"{'|g(word)|':>11}{'plural R@1':>12}")
    print("-" * 92)
    rows = [("frozen space + curved mirror", "mirror", False),
            ("fine-tuned + identity (null)", "identity", True),
            ("fine-tuned + free MLP (null)", "free", True),
            ("fine-tuned + curved mirror", "mirror", True)]
    for name, kind, tune in rows:
        rs = [run(kind, s, tune=tune) for s in (0, 1)]
        m = np.nanmean(np.array(rs), axis=0)
        sd = np.std([r[0] for r in rs])
        print(f"{name:<34}{m[0]:>5.3f}+/-{sd:.3f}{m[1]:>12.3f}{m[2]:>13.3f}"
              f"{m[3]:>11.3f}{m[4]:>12.3f}", flush=True)
    print("\nround-trip -> 1.0 and cos(normal) -> 1.0 mean the words now lie INSIDE")
    print("the mirror's reach. 'plural R@1' is the collateral check: if the adapter")
    print("buys antonymy by wrecking the geometry, it falls.")


if __name__ == "__main__":
    main()
