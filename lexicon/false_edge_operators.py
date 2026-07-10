"""Do false decompositions corrupt the OPERATOR, not just the lexeme?

Remy: "the point is not for the model to invert it internally into spelling, it's
that the tokenization is semantically laden... the meaning of the lex token
becomes context dependent instead of context free."

Right, and my entropy bound (0.0034 bits/char) answered a different question --
it priced the next-token ambiguity, not the loss of compositionality.

Sharper form: the operator is SHARED. <op:suf.ion> is one map over hundreds of
roots. It cannot encode "...and when applied to `state`, mean `station`". So a
false edge is not merely an ambiguous lexeme -- it is an off-manifold training
pair that drags the fit for every other word using that operator.

TEST, in the surface prototype space, per slot:
  TRUE  edges = attested in MorphyNet (real morphology)
  FALSE edges = absent from MorphyNet (station<-state, comic<-come, ...)

  A) fit operator on TRUE-train        -> R@1 on TRUE-test        (baseline)
  B) fit on TRUE-train + FALSE         -> R@1 on TRUE-test        (contamination)
  C) operator from (A) applied to FALSE sources -> R@1 on FALSE targets
     If false edges were composable, C would be nonzero. It should be ~0.
"""
import json, collections, sys
import numpy as np, torch

S = "/tmp/claude-1000/-home-remy-Code-lexicon/cccbc9d0-251e-44e9-b324-0d530c3837d0/scratchpad/"
INFL = {"noun.plural","verb.ger","verb.3sg","verb.ptcp","verb.past","adj.comp","adj.sup"}
RIDGE = 1.0


def ridge_fit(X, Y, lam=RIDGE):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


def r_at_k(W, src, tgt, En, ks=(1, 10)):
    P = src @ W
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
    sim = P @ En.T
    for i, t in enumerate(tgt):
        sim[i, t] = sim[i, t]                      # keep gold
    order = np.argsort(-sim, axis=1)
    out = {}
    for k in ks:
        out[k] = float(np.mean([tgt[i] in order[i, :k] for i in range(len(tgt))]))
    return out


def main():
    d = torch.load("real/surface/prototypes.pt", map_location="cpu", weights_only=False)
    words = list(d)
    E = np.stack([np.asarray(d[w], dtype=np.float32) for w in words])
    idx = {w: i for i, w in enumerate(words)}
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    mn = collections.defaultdict(set)
    for ln in open(S + "eng.deriv.tsv"):
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 6: mn[f[1].lower()].add(f[0].lower())

    p = {k: tuple(v) for k, v in json.load(open("dictionary/forest.json"))["parent"].items()}
    slots = collections.defaultdict(lambda: {"T": [], "F": []})
    for w, (s, b) in p.items():
        if s in INFL or w not in idx or b not in idx: continue
        slots[s]["T" if b in mn.get(w, ()) else "F"].append((b, w))

    rng = np.random.default_rng(0)
    rows = []
    print(f"{'slot':<16}{'nT':>5}{'nF':>5}{'A: clean':>11}{'B: +false':>11}"
          f"{'delta':>9}{'C: false':>10}")
    print("-" * 68)
    for s, g in sorted(slots.items(), key=lambda x: -len(x[1]["T"])):
        T, F = g["T"], g["F"]
        if len(T) < 60 or len(F) < 10: continue
        T = [T[i] for i in rng.permutation(len(T))]
        cut = int(0.8 * len(T)); tr, te = T[:cut], T[cut:]
        Xtr = np.stack([En[idx[b]] for b, _ in tr]); Ytr = np.stack([En[idx[w]] for _, w in tr])
        Xte = np.stack([En[idx[b]] for b, _ in te]); yte = [idx[w] for _, w in te]
        Xf = np.stack([En[idx[b]] for b, _ in F]);  Yf = np.stack([En[idx[w]] for _, w in F])
        yf = [idx[w] for _, w in F]

        Wa = ridge_fit(Xtr, Ytr)
        Wb = ridge_fit(np.vstack([Xtr, Xf]), np.vstack([Ytr, Yf]))
        a = r_at_k(Wa, Xte, yte, En)[1]
        b_ = r_at_k(Wb, Xte, yte, En)[1]
        c = r_at_k(Wa, Xf, yf, En)[1]
        rows.append((s, len(T), len(F), a, b_, c))
        print(f"{s:<16}{len(T):>5}{len(F):>5}{a:>11.3f}{b_:>11.3f}{b_-a:>+9.3f}{c:>10.3f}")

    A = np.array([r[3] for r in rows]); B = np.array([r[4] for r in rows])
    C = np.array([r[5] for r in rows])
    nT = np.array([r[1] for r in rows]); nF = np.array([r[2] for r in rows])
    print("-" * 68)
    print(f"{'mean':<16}{nT.sum():>5}{nF.sum():>5}{A.mean():>11.3f}{B.mean():>11.3f}"
          f"{(B-A).mean():>+9.3f}{C.mean():>10.3f}")
    print(f"\nA  operator fit on real morphology only, R@1 on held-out real edges")
    print(f"B  same, with false edges added to the fit   -> contamination cost")
    print(f"C  clean operator applied to false pairs     -> are they composable at all?")
    print(f"\nslots where contamination HURT: {(B<A).sum()}/{len(rows)}   "
          f"helped: {(B>A).sum()}/{len(rows)}")
    print(f"contamination ratio: false/(true+false) = {nF.sum()/(nT.sum()+nF.sum()):.1%}")


if __name__ == "__main__":
    main()
