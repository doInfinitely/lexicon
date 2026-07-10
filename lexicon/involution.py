"""An antonym operator that is an involution BY CONSTRUCTION, trained jointly
with the space, under a round-trip loss.

Why architecture and not just a loss. A constant offset cannot be an
involution (x + 2d = x forces d = 0), and an unconstrained MLP asked to
satisfy f(f(x)) = x must spend capacity learning it. But if g is any
invertible map and s is a trivial involution (negation), then

    f = g^-1 . s . g        satisfies    f(f(x)) = g^-1 s s g(x) = x

identically, for every g. So we take g to be an affine-coupling flow (exactly
invertible, hence information preserving -- it cannot collapse antonyms onto
each other) and let training choose the coordinates in which antonymy IS
negation. The involution is then free, and all capacity goes to finding those
coordinates.

Three operators are compared on the same split of WordNet's 3469 antonym
pairs, all jointly optimised with the embedding space:

    mlp          residual MLP, no constraint            (what we had)
    mlp+rt       same, plus round-trip loss f(f(x))~x   (the loss-only cure)
    involution   f = g^-1 . neg . g                     (the structural cure)

Reported on HELD-OUT pairs: retrieval R@1 against the full 38k vocabulary,
displacement alignment (is it a direction?), and exact round-trip fidelity.
"""
import json, random, sys, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lexicon.model import Adapter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = "real/english"
EMB = 768
TAU = 0.05
N_NEG = 8192


# ------------------------------------------------------------------ operators
class MLPOp(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(EMB, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, EMB))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class Coupling(nn.Module):
    """Affine coupling layer: exactly invertible, closed-form inverse."""

    def __init__(self, dim=EMB, hidden=1024, flip=False):
        super().__init__()
        self.half = dim // 2
        self.flip = flip
        self.net = nn.Sequential(nn.Linear(self.half, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * (dim - self.half)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _split(self, x):
        a, b = x[:, :self.half], x[:, self.half:]
        return (b, a) if self.flip else (a, b)

    def _merge(self, a, b):
        return torch.cat([b, a], 1) if self.flip else torch.cat([a, b], 1)

    def forward(self, x):
        a, b = self._split(x)
        s, t = self.net(a).chunk(2, dim=1)
        s = torch.tanh(s)                      # bound the scale for stability
        return self._merge(a, b * torch.exp(s) + t)

    def inverse(self, y):
        a, b = self._split(y)
        s, t = self.net(a).chunk(2, dim=1)
        s = torch.tanh(s)
        return self._merge(a, (b - t) * torch.exp(-s))


class Flow(nn.Module):
    def __init__(self, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList(
            [Coupling(flip=(i % 2 == 1)) for i in range(n_layers)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def inverse(self, y):
        for l in reversed(self.layers):
            y = l.inverse(y)
        return y


class InvolutionOp(nn.Module):
    """f = g^-1 . sigma_k . g,  sigma_k negates the first k coordinates.

    f(f(x)) == x exactly for any k, because sigma_k is an involution and g is
    invertible.

    !! CORRECTION. This docstring used to claim k measures "how many dimensions
    of polarity English antonymy requires", and that k=768 fails because it
    forces every word to have an antonym. Both are wrong.

      k is uninformative.  Held-out R@1 is flat from k=1 to k=512
      (0.267-0.279, seed sd ~0.005), decaying only mildly to 0.259 at k=704.
      The measurement cannot distinguish 1 polarity dimension from 512.

      k=768 fails for a GEOMETRIC reason, not a linguistic one.  At k=768,
      sigma_k = -I. distilbert embeddings occupy a narrow cone: a word sits at
      cos 0.739 from the corpus centroid, and two random words at cos 0.546.
      So -x lands outside the cone entirely -- its nearest word in the whole
      38k vocabulary is at cos -0.237, and ZERO of 512 sampled words have any
      word at positive cosine opposite them. f = -I maps every word into empty
      space. Nothing to do with antonymy. Any codimension > 0 keeps f inside
      the cone, which is why every k < 768 works equally well.
    """

    def __init__(self, n_layers=6, k=8):
        super().__init__()
        self.g = Flow(n_layers)
        self.k = k
        sign = torch.ones(EMB)
        sign[:k] = -1.0
        self.register_buffer("sign", sign)

    def forward(self, x):
        return self.g.inverse(self.sign * self.g(x))

    def fixed_point_score(self, x):
        """How close is x to being a fixed point (a word with no antonym)?"""
        return F.cosine_similarity(self(x), x, dim=-1)


# ------------------------------------------------------------------ data
def load_antonyms(val_frac=0.15, seed=0):
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    rels = json.load(open(f"{D}/relations.json"))
    pairs = [tuple(p) for p in rels["lex:antonym"]]
    # antonymy is symmetric: canonicalise so (a,b) and (b,a) never straddle
    # the split -- otherwise the test pair is the training pair reversed.
    canon = sorted({tuple(sorted(p)) for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(canon)
    k = int(len(canon) * val_frac)
    val, tr = canon[:k], canon[k:]
    # train sees both directions (it is an involution), test stays canonical
    train = [(a, b) for a, b in tr] + [(b, a) for a, b in tr]
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    # LEAK FIX. `pos` is the multi-positive mask used by infonce() to exclude
    # sanctioned answers from the negatives. Building it from ALL pairs meant a
    # held-out antonym of a training source was shielded from repulsion during
    # training -- test information reaching the training loss. Build it from
    # the TRAINING pairs only. (`pos_eval` keeps every sanctioned answer, which
    # is correct at evaluation time and touches no gradient.)
    pos = collections.defaultdict(set)
    for a, b in train:
        pos[a].add(widx[b])
    pos_eval = collections.defaultdict(set)
    for a, b in pairs:
        pos_eval[a].add(widx[b]); pos_eval[b].add(widx[a])
    return vocab, widx, P, train, val, pos, pos_eval


def infonce(out, adapter, P, tgt, srcs, pos, widx, V, gen):
    negs = torch.randint(0, V, (N_NEG,), device=DEVICE, generator=gen)
    cand = torch.cat([tgt, negs]).unique()
    tbl = F.normalize(adapter(P[cand]), dim=-1)
    logits = out @ tbl.T / TAU
    inv = {int(c): j for j, c in enumerate(cand.tolist())}
    pmask = torch.zeros_like(logits, dtype=torch.bool)
    for i, s in enumerate(srcs):
        for w in pos[s]:
            j = inv.get(w)
            if j is not None:
                pmask[i, j] = True
    best = logits.masked_fill(~pmask, float("-inf")).max(1).values
    neg = logits.masked_fill(pmask, float("-inf"))
    return (torch.logsumexp(torch.cat([best.unsqueeze(1), neg], 1), 1) - best).mean()


@torch.no_grad()
def evaluate(adapter, op, P, val, widx, pos, vocab, chunk=4096):
    tbl = torch.cat([F.normalize(adapter(P[i:i+chunk]), dim=-1)
                     for i in range(0, len(P), chunk)])
    s = torch.tensor([widx[a] for a, _ in val], device=DEVICE)
    t = torch.tensor([widx[b] for _, b in val], device=DEVICE)
    out = F.normalize(op(tbl[s]), dim=-1)
    sims = out @ tbl.T
    sims.scatter_(1, s.unsqueeze(1), -2)
    top1 = sims.argmax(1)
    strict = (top1 == t).float().mean().item()
    anyv = np.mean([top1[i].item() in pos[a] for i, (a, _) in enumerate(val)])
    # is it a direction? alignment of held-out displacements
    Dd = tbl[t] - tbl[s]
    align = F.cosine_similarity(Dd, Dd.mean(0, keepdim=True), dim=-1).mean().item()
    # round-trip fidelity, computed WITHOUT renormalising between applications
    rt = F.cosine_similarity(op(op(tbl[s])), tbl[s], dim=-1).mean().item()
    cos = F.cosine_similarity(out, tbl[t], dim=-1).mean().item()
    return dict(R1=strict, R1_any=float(anyv), align=align, roundtrip=rt, cos=cos)


def run(kind, epochs=60, bs=256, lr=4e-4, w_rt=1.0, k=8, w_fp=1.0):
    vocab, widx, P, train, val, pos, pos_eval = load_antonyms()
    V = len(vocab)
    adapter = Adapter().to(DEVICE)
    op = (InvolutionOp(k=k) if kind.startswith("involution")
          else MLPOp()).to(DEVICE)
    use_rt = kind in ("mlp+rt",)          # exact involutions need no rt loss
    # A partial involution needs its support supervised. The architecture only
    # PERMITS fixed points; without this term f wanders on the 35k words that
    # have no antonym, and (since antonyms sit at cos 0.80) ends up nearer the
    # identity on the very words it was trained on. Measured: d = -0.364.
    use_fp = kind.endswith("+fp")
    has_ant = {a for a, b in train} | {b for a, b in train}
    no_ant = [widx[w] for w in vocab if w not in has_ant]
    no_ant_t = torch.tensor(no_ant, device=DEVICE)

    n_par = sum(p.numel() for p in list(adapter.parameters()) + list(op.parameters()))
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    opt = torch.optim.AdamW(list(adapter.parameters()) + list(op.parameters()),
                            lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    P0n = F.normalize(P, dim=-1)

    for ep in range(epochs):
        random.Random(ep).shuffle(train)
        for i in range(0, len(train), bs):
            b = train[i:i + bs]
            s = torch.tensor([widx[a] for a, _ in b], device=DEVICE)
            t = torch.tensor([widx[c] for _, c in b], device=DEVICE)
            zs = F.normalize(adapter(P[s]), dim=-1)
            fx = op(zs)
            loss = infonce(F.normalize(fx, dim=-1), adapter, P, t,
                           [a for a, _ in b], pos, widx, V, gen)
            if use_rt:
                # round trip: applying the operator twice must return the word
                loss = loss + w_rt * (1 - F.cosine_similarity(
                    op(fx), zs, dim=-1)).mean()
            if use_fp:
                # words with no antonym must be FIXED POINTS of f
                fi = no_ant_t[torch.randint(0, len(no_ant_t), (256,),
                                            device=DEVICE, generator=gen)]
                zf = F.normalize(adapter(P[fi]), dim=-1)
                loss = loss + w_fp * (1 - F.cosine_similarity(
                    op(zf), zf, dim=-1)).mean()
            idx = torch.randint(0, V, (2048,), device=DEVICE, generator=gen)
            loss = loss + 0.25 * (1 - F.cosine_similarity(
                F.normalize(adapter(P[idx]), dim=-1), P0n[idx], dim=-1)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(adapter.parameters()) + list(op.parameters()), 1.0)
            opt.step()
        sched.step()
        if ep % 15 == 0 or ep == epochs - 1:
            m = evaluate(adapter, op, P, val, widx, pos_eval, vocab)
            print(f"  [{kind:<10}] ep {ep:2d}  held-out R@1 {m['R1']:.3f} "
                  f"(any {m['R1_any']:.3f})  align {m['align']:.3f}  "
                  f"round-trip {m['roundtrip']:.3f}")
    m = evaluate(adapter, op, P, val, widx, pos_eval, vocab)
    m["params"] = n_par
    torch.save({"adapter": adapter.state_dict(), "op": op.state_dict(),
                "kind": kind}, f"{D}/antonym_{kind.replace('+','_')}.pt")
    return m


def main():
    vocab, widx, P, train, val, pos, pos_eval = load_antonyms()
    print(f"antonym pairs: {len(train)//2} canonical train, {len(val)} held-out")
    print(f"(BATS had 35 training pairs; this is "
          f"{(len(train)//2)/35:.0f}x more)\n")

    # frozen-space reference: what does the raw space already give?
    with torch.no_grad():
        tbl = F.normalize(P, dim=-1)
    s = torch.tensor([widx[a] for a, _ in val], device=DEVICE)
    t = torch.tensor([widx[b] for _, b in val], device=DEVICE)
    Dd = tbl[t] - tbl[s]
    fa = F.cosine_similarity(Dd, Dd.mean(0, keepdim=True), dim=-1).mean().item()
    sims = tbl[s] @ tbl.T
    sims.scatter_(1, s.unsqueeze(1), -2)
    id_r1 = (sims.argmax(1) == t).float().mean().item()
    print(f"frozen space, identity operator : held-out R@1 {id_r1:.3f}  "
          f"align {fa:.3f}\n")

    results = {}
    for kind in ("mlp", "mlp+rt"):
        results[kind] = run(kind)
        print()
    # how many dimensions of polarity does antonymy need?
    for k in (1, 8, 64, 384, 768):
        results[f"involution k={k}"] = run(f"involution_k{k}", k=k)
        print()

    print("=" * 92)
    print(f"{'operator':<18}{'params':>12}{'held-out R@1':>15}{'R@1 any':>10}"
          f"{'direction':>12}{'round-trip':>13}")
    print("=" * 92)
    print(f"{'frozen/identity':<18}{0:>12}{id_r1:>15.3f}{'-':>10}{fa:>12.3f}"
          f"{1.000:>13.3f}")
    for k, m in results.items():
        print(f"{k:<18}{m['params']:>12,}{m['R1']:>15.3f}{m['R1_any']:>10.3f}"
              f"{m['align']:>12.3f}{m['roundtrip']:>13.3f}")
    print("\nk = 768 forces a TOTAL involution: every word paired with an "
          "antonym.\nk small allows FIXED POINTS: words that have no antonym.")
    json.dump({k: {kk: float(vv) for kk, vv in m.items()}
               for k, m in results.items()},
              open(f"{D}/antonym_study.json", "w"), indent=1)
    print(f"\nwrote {D}/antonym_study.json")


if __name__ == "__main__":
    main()
