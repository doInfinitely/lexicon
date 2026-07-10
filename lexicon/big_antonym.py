"""Does exploding the antonym set change the geometry?

Test set: the ORIGINAL direct WordNet lemma antonyms. These are the gold pairs.
Expansion (indirect + morphological) is training data only, and must EARN its
place by improving retrieval of the gold pairs.

WORD-LEVEL split. Pair-level holdout is not enough once we expand: indirect
antonymy connects a held-out word to its opposite through a satellite, so the
answer leaks in by another route. Here the held-out WORDS appear in no
training pair at all, of any source. This is the strictest test: the model
must place a word it has never seen relative to the mirror.

Questions:
  1. does R@1 keep climbing with data, or saturate?
  2. does the flip subspace need more dimensions once we can afford them?
     (k=32 lost to k=8 at 1476 pairs -- was that data starvation?)
  3. does curvature finally beat the flat mirror?
"""
import json, random, sys
import numpy as np
import torch
import torch.nn.functional as F

from lexicon.mirror import Mirror, LinearAdapter, Identity
from lexicon.model import Adapter
from lexicon.involution import InvolutionOp, infonce, DEVICE, D

EMB = 768


def load(n_train=None, seed=0, expanded=True):
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    rels = json.load(open(f"{D}/relations.json"))
    gold = sorted({tuple(sorted(p)) for p in map(tuple, rels["lex:antonym"])})

    # ---- word-level holdout, defined on the GOLD pairs ----
    rng = random.Random(seed)
    g = list(gold); rng.shuffle(g)
    val, val_words = [], set()
    for a, b in g:
        if len(val) >= 300:
            break
        if a not in val_words and b not in val_words:
            val.append((a, b)); val_words |= {a, b}

    if expanded:
        pool = [tuple(p) for p in json.load(open(f"{D}/antonyms_expanded.json"))]
    else:
        pool = gold
    # strict: drop every pair touching a held-out word
    train_pairs = [(a, b) for a, b in pool
                   if a not in val_words and b not in val_words]
    rng.shuffle(train_pairs)
    if n_train:
        train_pairs = train_pairs[:n_train]

    pos = {}
    for a, b in pool:
        pos.setdefault(a, set()).add(widx[b])
        pos.setdefault(b, set()).add(widx[a])
    for a, b in gold:
        pos.setdefault(a, set()).add(widx[b])
        pos.setdefault(b, set()).add(widx[a])

    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    both = [(a, b) for a, b in train_pairs] + [(b, a) for a, b in train_pairs]
    return vocab, widx, P, both, val, pos, val_words


def train_one(kind, k, n_train, seed=0, epochs=None, bs=512, lr=4e-4, expanded=True):
    vocab, widx, P, train_pairs, val, pos, val_words = load(n_train, seed, expanded)
    V = len(vocab)
    adapter = {"linear": LinearAdapter(), "mlp": Adapter(),
               "raw": Identity()}[kind.split("_")[0]].to(DEVICE)
    op = (InvolutionOp(k=k) if "flow" in kind else Mirror(k)).to(DEVICE)

    has_ant = {a for a, b in train_pairs} | {b for a, b in train_pairs}
    no_ant = torch.tensor([widx[w] for w in vocab
                           if w not in has_ant and w not in val_words],
                          device=DEVICE)
    params = list(adapter.parameters()) + list(op.parameters())
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    # equalise gradient steps across data sizes
    # equal gradient-step budget at every data size, so the curve measures
    # DATA and not optimisation time
    steps = max(1, len(train_pairs) // bs)
    epochs = epochs or max(8, min(400, int(8000 / steps)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    P0n = F.normalize(P, dim=-1)

    for ep in range(epochs):
        random.Random(ep).shuffle(train_pairs)
        for i in range(0, len(train_pairs), bs):
            b = train_pairs[i:i + bs]
            s = torch.tensor([widx[a] for a, _ in b], device=DEVICE)
            t = torch.tensor([widx[c] for _, c in b], device=DEVICE)
            zs = F.normalize(adapter(P[s]), dim=-1)
            fx = op(zs)
            loss = infonce(F.normalize(fx, dim=-1), adapter, P, t,
                           [a for a, _ in b], pos, widx, V, gen)
            fi = no_ant[torch.randint(0, len(no_ant), (256,), device=DEVICE,
                                      generator=gen)]
            zf = F.normalize(adapter(P[fi]), dim=-1)
            loss = loss + (1 - F.cosine_similarity(op(zf), zf, dim=-1)).mean()
            idx = torch.randint(0, V, (2048,), device=DEVICE, generator=gen)
            loss = loss + 0.25 * (1 - F.cosine_similarity(
                F.normalize(adapter(P[idx]), dim=-1), P0n[idx], dim=-1)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()

    with torch.no_grad():
        tbl = torch.cat([F.normalize(adapter(P[i:i+4096]), dim=-1)
                         for i in range(0, len(P), 4096)])
        s = torch.tensor([widx[a] for a, _ in val], device=DEVICE)
        t = torch.tensor([widx[b] for _, b in val], device=DEVICE)
        out = F.normalize(op(tbl[s]), dim=-1)
        sims = out @ tbl.T
        sims.scatter_(1, s.unsqueeze(1), -2)
        top1 = sims.argmax(1)
        r1 = (top1 == t).float().mean().item()
        anyv = float(np.mean([top1[i].item() in pos.get(a, set())
                              for i, (a, _) in enumerate(val)]))
        # how much of a held-out antonym difference lies in the flip subspace?
        if hasattr(op, "basis"):
            Vb = op.basis()
            Dm = tbl[t] - tbl[s]
            inpl = ((((Dm @ Vb) @ Vb.T).norm(dim=1)) /
                    (Dm.norm(dim=1) + 1e-9)).mean().item()
        else:
            inpl = float("nan")
        rt = F.cosine_similarity(op(op(tbl[s])), tbl[s], dim=-1).mean().item()
    return dict(R1=r1, any=anyv, in_plane=inpl, rt=rt,
                n=len(train_pairs) // 2, epochs=epochs)


def main():
    _, _, _, allp, val, _, vw = load()
    print(f"gold test pairs (WORD-level holdout): {len(val)}; "
          f"held-out words: {len(vw)}")
    print(f"expanded training pairs available   : {len(allp)//2}\n")

    print("A) data curve, flat mirror k=8, strict word-level holdout")
    print(f"{'train pairs':>12}{'epochs':>8}{'R@1':>9}{'R@1 any':>10}"
          f"{'in-plane':>11}")
    print("-" * 50)
    for n in (1500, 5000, 15000, 40000, None):
        m = train_one("linear_mirror", 8, n)
        print(f"{m['n']:>12}{m['epochs']:>8}{m['R1']:>9.3f}{m['any']:>10.3f}"
              f"{m['in_plane']:>11.3f}")

    print("\nB) with all the data, how many mirror dimensions does antonymy want?")
    print(f"{'k':>6}{'R@1':>9}{'R@1 any':>10}{'in-plane':>11}")
    print("-" * 36)
    for k in (1, 8, 16, 32, 64, 128):
        m = train_one("linear_mirror", k, None)
        print(f"{k:>6}{m['R1']:>9.3f}{m['any']:>10.3f}{m['in_plane']:>11.3f}")

    print("\nC) flat vs curved, now that data is not the bottleneck (3 seeds)")
    for kind in ("linear_mirror", "mlp_mirror", "mlp_flow"):
        rs = [train_one(kind, 8, None, seed=s)["R1"] for s in (0, 1, 2)]
        print(f"  {kind:<16} R@1 {np.mean(rs):.3f} +/- {np.std(rs):.3f}  "
              f"{[f'{x:.3f}' for x in rs]}")


if __name__ == "__main__":
    main()
