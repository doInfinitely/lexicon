"""Is generation hard because the operators are weak, or because the SPACE is?

The atlas found: every relation is easy to verify (pair-probe AUC 0.93-0.99)
and hard to generate (best R@1 0.39-0.59). Two properties of distilbert's
space could produce that asymmetry without any operator being weak:

  ANISOTROPY  every word sits at cosine 0.739 from the corpus centroid and
              0.546 from a random other word. The embeddings occupy a narrow
              cone. Verification looks at the pair DIFFERENCE, which cancels
              the shared component; generation must land on a precise point
              inside a crowded cone.

  HUBNESS     in a high-dimensional anisotropic space a few "hub" words are the
              nearest neighbour of enormous numbers of points. Generation
              retrieves the hub instead of the answer; verification never has
              to choose.

Both are known pathologies with known fixes, from bilingual lexicon induction
(which is the same problem: map a source word through a linear operator and
retrieve the target).

Spaces tested:
  raw            L2-normalised prototypes
  centered       subtract the mean, then renormalise
  abtt           "all-but-the-top": remove the top-k principal directions
  whitened       ZCA whitening (isotropic covariance)

Retrieval rules:
  cosine         argmax_t  cos(f(s), t)
  CSLS           argmax_t  2 cos(f(s), t) - r(t),  where r(t) is t's mean
                 similarity to its own k nearest query-neighbours. Penalises
                 hubs. No training, no parameters beyond k.

If a space/retrieval change moves R@1 substantially, the "operator" was never
the bottleneck.
"""
import json, collections
import numpy as np
import torch
import torch.nn.functional as F

from lexicon.atlas import CLOSED_FORM, split_words, DEVICE, D

RELS = ["infl:noun_plural", "infl:verb_Ving", "infl:adj_comparative",
        "deriv:adj_ly", "deriv:verb_er", "cook:suf_ness", "cook:suf_ize",
        "lex:antonym", "lex:member_meronym", "lex:hypernym",
        "lex:derivationally_related", "cook:suf_tion"]
CSLS_K = 10


# ------------------------------------------------------------------ spaces
def make_spaces(P):
    out = {}
    out["raw"] = F.normalize(P, dim=-1)

    mu = P.mean(0, keepdim=True)
    out["centered"] = F.normalize(P - mu, dim=-1)

    # all-but-the-top: drop the leading principal directions of the centred data
    X = P - mu
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    k = 8
    proj = Vt[:k]                                   # [k, d]
    out["abtt"] = F.normalize(X - (X @ proj.T) @ proj, dim=-1)

    # ZCA whitening: make the covariance identity
    C = (X.T @ X) / len(X)
    e, V = torch.linalg.eigh(C.double())
    e = e.clamp(min=1e-6)
    W = (V @ torch.diag(e.rsqrt()) @ V.T).float()
    out["whitened"] = F.normalize(X @ W, dim=-1)
    return out


def anisotropy(T, n=2048, seed=0):
    g = torch.Generator(device=T.device).manual_seed(seed)
    idx = torch.randint(0, len(T), (n,), device=T.device, generator=g)
    mu = F.normalize(T.mean(0), dim=0)
    to_centroid = (T[idx] @ mu).mean().item()
    S = T[idx] @ T[idx].T
    S.fill_diagonal_(0)
    return to_centroid, (S.sum() / (n * (n - 1))).item()


@torch.no_grad()
def hubness(T, sample=4000, seed=0):
    """How concentrated is 'being someone's nearest neighbour'?"""
    g = torch.Generator(device=T.device).manual_seed(seed)
    q = torch.randint(0, len(T), (sample,), device=T.device, generator=g)
    sims = T[q] @ T.T
    sims.scatter_(1, q.unsqueeze(1), -2)
    nn = sims.argmax(1)
    c = collections.Counter(nn.tolist())
    counts = np.array(sorted(c.values(), reverse=True))
    top = counts[:10].sum() / sample
    # skew of the N1 distribution: the standard hubness statistic
    full = np.zeros(len(T)); full[list(c)] = list(c.values())
    skew = float(((full - full.mean()) ** 3).mean() / (full.std() ** 3 + 1e-9))
    return top, skew, [c.most_common(3)]


# ------------------------------------------------------------------ retrieval
@torch.no_grad()
def csls_penalty(T, queries, k=CSLS_K, chunk=2048):
    """r(t) = mean similarity of t to its k nearest QUERY points."""
    r = torch.zeros(len(T), device=T.device)
    for i in range(0, len(T), chunk):
        sims = T[i:i + chunk] @ queries.T
        r[i:i + chunk] = sims.topk(min(k, sims.shape[1]), dim=1).values.mean(1)
    return r


@torch.no_grad()
def retrieve_r1(pred, T, tgt, src, rule="cosine", pen=None, chunk=512):
    hits = 0
    for i in range(0, len(pred), chunk):
        p = F.normalize(pred[i:i + chunk], dim=-1)
        sims = p @ T.T
        if rule == "csls":
            sims = 2 * sims - pen.unsqueeze(0)
        s = src[i:i + chunk]
        sims.scatter_(1, s.unsqueeze(1), -2)
        hits += (sims.argmax(1) == tgt[i:i + chunk]).sum().item()
    return hits / len(pred)


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    rels = json.load(open(f"{D}/relations.json"))
    spaces = make_spaces(P)

    print("SPACE GEOMETRY")
    print(f"{'space':<12}{'cos to centroid':>18}{'cos random pair':>18}"
          f"{'top-10 hub share':>19}{'N1 skew':>10}")
    for name, T in spaces.items():
        a, b = anisotropy(T)
        top, skew, ex = hubness(T)
        print(f"{name:<12}{a:>18.3f}{b:>18.3f}{top:>19.1%}{skew:>10.1f}")
    print("\n(hub share = fraction of 4,000 random queries whose nearest neighbour")
    print(" is one of just 10 words. N1 skew > 0 means hubs exist.)\n")

    print("HELD-OUT R@1 BY SPACE AND RETRIEVAL RULE  (best closed-form shape per cell)")
    hdr = f"{'relation':<28}"
    cells = [(s, r) for s in spaces for r in ("cosine", "csls")]
    for s, r in cells:
        hdr += f"{s[:5]+'/'+r[:4]:>13}"
    print(hdr)
    print("-" * len(hdr))

    totals = collections.defaultdict(list)
    for rel in RELS:
        if rel not in rels:
            continue
        pairs = sorted({tuple(p) for p in rels[rel]})
        pairs = [p for p in pairs if p[0] != p[1]]
        s_ = set(pairs)
        if sum(1 for a, b in s_ if (b, a) in s_) / max(len(pairs), 1) > 0.5:
            pairs = sorted({tuple(sorted(p)) for p in pairs})
        tr, te = split_words(pairs)
        if len(tr) < 40 or len(te) < 20:
            continue
        if len(tr) > 8000:
            tr = tr[:8000]
        if len(te) > 800:
            te = te[:800]
        row = f"{rel:<28}"
        for sname, T in spaces.items():
            si = torch.tensor([widx[a] for a, _ in tr], device=DEVICE)
            ti = torch.tensor([widx[b] for _, b in tr], device=DEVICE)
            sie = torch.tensor([widx[a] for a, _ in te], device=DEVICE)
            tie = torch.tensor([widx[b] for _, b in te], device=DEVICE)
            best = {}
            for shape, fit in CLOSED_FORM.items():
                try:
                    f = fit(T[si], T[ti])
                    pred = f(T[sie])
                except Exception:
                    continue
                pen = csls_penalty(T, F.normalize(pred, dim=-1))
                for rule in ("cosine", "csls"):
                    v = retrieve_r1(pred, T, tie, sie, rule, pen)
                    if v > best.get(rule, -1):
                        best[rule] = v
            for rule in ("cosine", "csls"):
                row += f"{best.get(rule, float('nan')):>13.3f}"
                totals[(sname, rule)].append(best.get(rule, 0.0))
        print(row, flush=True)

    print("\n" + "-" * len(hdr))
    row = f"{'MEAN':<28}"
    for s, r in cells:
        row += f"{np.mean(totals[(s, r)]):>13.3f}"
    print(row)
    base = np.mean(totals[("raw", "cosine")])
    print(f"\nbaseline (raw + cosine, what the whole project used): {base:.3f}")
    for s, r in cells:
        v = np.mean(totals[(s, r)])
        if (s, r) != ("raw", "cosine"):
            print(f"  {s:<10} + {r:<7} : {v:.3f}   ({v-base:+.3f})")
    json.dump({f"{s}|{r}": float(np.mean(v)) for (s, r), v in totals.items()},
              open(f"{D}/geometry_fix.json", "w"), indent=1)


if __name__ == "__main__":
    main()
