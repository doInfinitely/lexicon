"""Is the mirror flat, or sinuous? A test that fits no model.

If antonymy is a reflection through ONE fixed plane, then for every antonym
pair the difference a - b = 2 V V^T a lies in the SAME k-dimensional flip
subspace span(V). So the set of antonym differences is globally low-rank.

If the mirror is sinuous -- the flip subspace rotating as you travel through
meaning-space -- then differences are low-rank LOCALLY (among semantically
nearby pairs) but full-rank GLOBALLY, because each neighbourhood flips along
its own axes.

    flat     global rank ~= local rank
    sinuous  local rank << global rank

Differences are sign-ambiguous (a-b vs b-a), so everything is computed from
the second-moment matrix D^T D, which is sign invariant.

Nulls: synonym pairs (a relation that is NOT a reflection) and random word
pairs, both processed identically. A low global rank is only meaningful if
the nulls do not show it too -- any set of differences between nearby words
concentrates somewhat.
"""
import json, collections
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = "real/english"
RNG = np.random.default_rng(0)


def spectrum(Dm):
    """Fraction of second-moment mass captured by the top-k directions."""
    Dm = Dm - 0.0                        # differences: do NOT centre (sign sym)
    C = (Dm.T @ Dm) / len(Dm)
    ev = torch.linalg.eigvalsh(C).flip(0).clamp(min=0)
    frac = (ev.cumsum(0) / ev.sum()).cpu().numpy()
    # participation ratio: an estimate of effective dimensionality
    p = ev / ev.sum()
    pr = float(1.0 / (p ** 2).sum())
    return frac, pr


def local_vs_global(pairs, tbl, widx, n_neighbors=40, n_centers=60, k=8):
    """Global rank of all differences vs. mean rank within neighbourhoods."""
    idx = [(widx[a], widx[b]) for a, b in pairs if a in widx and b in widx]
    if len(idx) < 50:
        return None
    A = torch.stack([tbl[i] for i, _ in idx])
    B = torch.stack([tbl[j] for _, j in idx])
    Dm = A - B
    g_frac, g_pr = spectrum(Dm)

    # neighbourhoods in MEANING space: pairs whose source words are close
    centers = RNG.choice(len(idx), size=min(n_centers, len(idx)), replace=False)
    sims = A @ A.T
    locals_frac, locals_pr = [], []
    for c in centers:
        nb = sims[c].topk(min(n_neighbors, len(idx))).indices
        if len(nb) < k + 4:
            continue
        f, pr = spectrum(Dm[nb])
        locals_frac.append(f[k - 1])
        locals_pr.append(pr)
    return dict(global_topk=float(g_frac[k - 1]), global_pr=g_pr,
                local_topk=float(np.mean(locals_frac)),
                local_pr=float(np.mean(locals_pr)), n=len(idx))


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    rels = json.load(open(f"{D}/relations.json"))

    frozen = F.normalize(P, dim=-1)
    # the space in which the flat mirror was found: one linear map
    try:
        mk = torch.load(f"{D}/mirror_linear_k8.pt", weights_only=False)
        W = mk["W"].to(DEVICE)
        adapted = F.normalize(P @ W.T + mk["b"].to(DEVICE), dim=-1)
        spaces = {"frozen distilbert": frozen, "linear-adapted": adapted}
    except FileNotFoundError:
        spaces = {"frozen distilbert": frozen}

    ant = [tuple(p) for p in rels["lex:antonym"]]
    ant = sorted({tuple(sorted(p)) for p in ant})
    syn = sorted({tuple(sorted(p)) for p in map(tuple, rels["lex:similar_to"])})
    syn = [syn[i] for i in RNG.choice(len(syn), size=min(len(ant), len(syn)),
                                      replace=False)]
    rnd = [(vocab[i], vocab[j]) for i, j in
           RNG.integers(0, len(vocab), (len(ant), 2))]
    hyp = sorted({tuple(p) for p in map(tuple, rels["lex:hypernym"])})
    hyp = [hyp[i] for i in RNG.choice(len(hyp), size=len(ant), replace=False)]

    for sname, tbl in spaces.items():
        print("=" * 92)
        print(f"SPACE: {sname}")
        print("=" * 92)
        print(f"{'relation':<22}{'n':>6}{'global top-8':>15}{'local top-8':>14}"
              f"{'global eff.dim':>16}{'local eff.dim':>15}")
        for name, pl in (("antonym", ant), ("synonym (null)", syn),
                         ("hypernym (null)", hyp), ("random (null)", rnd)):
            r = local_vs_global(pl, tbl, widx)
            if r:
                print(f"{name:<22}{r['n']:>6}{r['global_topk']:>15.3f}"
                      f"{r['local_topk']:>14.3f}{r['global_pr']:>16.1f}"
                      f"{r['local_pr']:>15.1f}")
        print()

    print("READING:")
    print("  flat mirror  -> antonym differences are globally low-rank:")
    print("                  global top-8 is high, and close to local top-8.")
    print("  sinuous      -> local top-8 >> global top-8, and global effective")
    print("                  dimension is large: each region flips on its own axes.")
    print("  The nulls calibrate how much concentration comes for free.")


if __name__ == "__main__":
    main()
