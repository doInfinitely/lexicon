"""Score one-to-many relations as SET PREDICTION, not rank-1 retrieval.

`hypernym(dog)` is not one word. It is {canine, mammal, animal, carnivore}.
Asking an operator to place `canine` first and calling everything else a miss
punishes the model for being right: strict R@1 is 0.044 while the answer's
neighbourhood is reached 30% of the time, and `spearman(fan-in, R@1/R@50) =
-0.891`. So we score what the relation actually asserts.

Two metrics, each with the nulls that could make it vacuous.

MAP  Mean average precision over the FULL 38,142-word vocabulary, with every
     sanctioned target as a positive. Standard IR, nothing to tune. A random
     ranking scores |G| / V ~ 1e-4. This measures whether the operator ranks
     the WHOLE correct set highly, not whether it wins a coin flip among
     equally-correct answers.

HULL Does f(a) land in the convex region spanned by G(a)? In 768 dimensions
     the hull of k points has measure zero, so "inside" is meaningless; we
     measure the cosine to the projection onto the hull. That number is
     worthless on its own, because a nearby point projects onto SOME hull. It
     needs two nulls:
       random null   the hull of |G(a)| words drawn at random
       sibling null  the hull of G(a'), the answer set of a DIFFERENT source
                     of the same relation. This is the hard one: it asks
                     whether the operator found THIS word's answer region or
                     merely the region where this relation's answers live.

If HULL(gold) does not clearly beat HULL(sibling), the operator has learned
the relation's typical output region and nothing about the input.
"""
import json, collections
import numpy as np
import torch
import torch.nn.functional as F

from lexicon.atlas import CLOSED_FORM, split_words, DEVICE, D
from lexicon.geometry_fix import make_spaces

RELS = ["lex:hypernym", "lex:hyponym", "lex:part_meronym", "lex:member_meronym",
        "lex:similar_to", "lex:entailment", "lex:antonym", "infl:noun_plural",
        "deriv:adj_ly"]
MAX_SRC = 400
RNG = np.random.default_rng(0)


def project_simplex(v):
    """Euclidean projection of each row of v onto the probability simplex."""
    n = v.shape[1]
    u, _ = torch.sort(v, dim=1, descending=True)
    css = u.cumsum(1) - 1
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype).unsqueeze(0)
    cond = u - css / ind > 0
    rho = cond.float().cumsum(1).argmax(1)
    theta = css.gather(1, rho.unsqueeze(1)) / (rho + 1).unsqueeze(1).to(v.dtype)
    return (v - theta).clamp(min=0)


def hull_cos(x, G, steps=120, lr=0.5):
    """max over convex combinations of G of cos(x, G^T lambda).
    x: [B,d]   G: [B,k,d]  (rows L2-normalised)"""
    B, k, d = G.shape
    lam = torch.full((B, k), 1.0 / k, device=x.device)
    for _ in range(steps):
        lam = lam.detach().requires_grad_(True)
        y = torch.bmm(lam.unsqueeze(1), G).squeeze(1)
        c = F.cosine_similarity(x, y, dim=-1).sum()
        (g,) = torch.autograd.grad(c, lam)
        lam = project_simplex(lam.detach() + lr * g)
    with torch.no_grad():
        y = torch.bmm(lam.unsqueeze(1), G).squeeze(1)
        return F.cosine_similarity(x, y, dim=-1)


def pad_sets(sets, T, k):
    """[B,k,d] tensor of gold sets, sampling with replacement to size k."""
    B = len(sets)
    idx = torch.zeros(B, k, dtype=torch.long)
    for i, s in enumerate(sets):
        s = list(s)
        pick = (s * ((k // len(s)) + 1))[:k]
        idx[i] = torch.tensor(pick)
    return T[idx.to(T.device)]


@torch.no_grad()
def average_precision(sims, gold_idx):
    """AP for one query. sims: [V] (source already masked out)."""
    order = sims.argsort(descending=True)
    gold = torch.zeros(len(sims), dtype=torch.bool, device=sims.device)
    gold[gold_idx] = True
    hits = gold[order]
    ranks = torch.arange(1, len(sims) + 1, device=sims.device, dtype=torch.float)
    csum = hits.cumsum(0).float()
    prec = csum / ranks
    return (prec[hits].sum() / hits.sum()).item()


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    T = make_spaces(P)["abtt"]                      # the space that won
    rels = json.load(open(f"{D}/relations.json"))
    V = len(vocab)

    print("SET PREDICTION for one-to-many relations (abtt space, held-out words)\n")
    print(f"{'relation':<24}{'fan':>6}{'|G|':>6}{'R@1':>8}{'R@1any':>9}{'MAP':>8}"
          f"{'MAPrnd':>9}   {'hull gold':>10}{'hull sib':>10}{'hull rnd':>10}")
    print("-" * 112)

    for rel in RELS:
        if rel not in rels:
            continue
        pairs = [tuple(p) for p in rels[rel]]
        s_ = set(pairs)
        sym = sum(1 for a, b in s_ if (b, a) in s_) / max(len(pairs), 1)
        cpairs = sorted({tuple(sorted(p)) for p in s_}) if sym > 0.5 else sorted(s_)
        tr, te = split_words(cpairs)
        if len(tr) < 60 or len(te) < 25:
            continue
        if len(tr) > 12000:
            tr = [tr[i] for i in RNG.choice(len(tr), 12000, replace=False)]

        # fit the operator on training pairs
        si = torch.tensor([widx[a] for a, _ in tr], device=DEVICE)
        ti = torch.tensor([widx[b] for _, b in tr], device=DEVICE)
        sie = torch.tensor([widx[a] for a, _ in te], device=DEVICE)
        tie = torch.tensor([widx[b] for _, b in te], device=DEVICE)
        best_f, best_r1 = None, -1
        for shape, fit in CLOSED_FORM.items():
            try:
                f = fit(T[si], T[ti])
                pred = F.normalize(f(T[sie]), dim=-1)
                sims = pred @ T.T
                sims.scatter_(1, sie.unsqueeze(1), -2)
                r1 = (sims.argmax(1) == tie).float().mean().item()
            except Exception:
                continue
            if r1 > best_r1:
                best_f, best_r1 = f, r1

        # gold sets, from ALL pairs of the relation (sanctioned answers)
        gold = collections.defaultdict(set)
        for a, b in pairs:
            gold[a].add(widx[b])
            if sym > 0.5:
                gold[b].add(widx[a])
        srcs = sorted({a for a, _ in te if len(gold[a]) >= 1})
        if len(srcs) > MAX_SRC:
            srcs = [srcs[i] for i in RNG.choice(len(srcs), MAX_SRC, replace=False)]
        sidx = torch.tensor([widx[a] for a in srcs], device=DEVICE)
        with torch.no_grad():
            x = F.normalize(best_f(T[sidx]), dim=-1)

        # R@1 / R@1_any / MAP
        r1 = r1any = 0
        aps = []
        with torch.no_grad():
            for i in range(0, len(srcs), 128):
                sub = slice(i, i + 128)
                sims = x[sub] @ T.T
                sims.scatter_(1, sidx[sub].unsqueeze(1), -2)
                top1 = sims.argmax(1)
                for j, a in enumerate(srcs[sub]):
                    g = gold[a]
                    r1any += int(top1[j].item() in g)
                    aps.append(average_precision(sims[j], torch.tensor(sorted(g), device=DEVICE)))
        r1any /= len(srcs)
        MAP = float(np.mean(aps))
        meanG = float(np.mean([len(gold[a]) for a in srcs]))
        map_rnd = meanG / V

        # hull tests: gold vs sibling vs random, size-matched per source
        k = max(2, int(round(meanG)))
        gsets = [sorted(gold[a]) for a in srcs]
        perm = RNG.permutation(len(srcs))
        sib = [gsets[p] for p in perm]                       # another source's set
        rnd = [list(RNG.choice(V, size=max(2, len(g)), replace=False)) for g in gsets]
        hg = hull_cos(x, pad_sets(gsets, T, k)).mean().item()
        hs = hull_cos(x, pad_sets(sib, T, k)).mean().item()
        hr = hull_cos(x, pad_sets(rnd, T, k)).mean().item()

        fan = len(pairs) / max(len({b for _, b in pairs}), 1)
        print(f"{rel:<24}{fan:>6.2f}{meanG:>6.1f}{best_r1:>8.3f}{r1any:>9.3f}"
              f"{MAP:>8.3f}{map_rnd:>9.1e}   {hg:>10.3f}{hs:>10.3f}{hr:>10.3f}",
              flush=True)

    print("\nMAP: chance is |G|/V (~1e-4). MAP >> R@1 means the operator ranks the")
    print("whole correct set highly and is being punished by rank-1 scoring.")
    print("\nHULL: 'gold' must beat 'sib' (the hull of ANOTHER source's answer set).")
    print("If gold ~= sib, the operator found the relation's output region, not")
    print("this word's answer. If gold ~= rnd, the metric is vacuous.")


if __name__ == "__main__":
    main()
