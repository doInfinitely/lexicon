"""Is polarity information ABSENT from the space, or merely not GENERATABLE?

The antonymy story rests on: "antonyms sit as close as synonyms (0.742 vs
0.693), therefore no operator on position can separate them." That inference
conflates two very different claims:

  ABSENT      nothing in the embeddings distinguishes an antonym pair from a
              synonym pair. The corpus never encoded polarity.
  UNGENERABLE the distinction is present in the pair (a,b), but an operator
              sees only `a` and must PRODUCE `b` out of 38k candidates. That
              is strictly harder than telling two given words apart.

A probe on the PAIR settles it. Features are symmetric (both relations are),
built from frozen distilbert prototypes:  |a-b|, a*b, (a+b)/2, cos(a,b).

Controls, because the last three claims died of missing ones:
  - synonyms are true WordNet co-synset lemmas, not `similar_to`
  - pairs are matched on part of speech AND on Zipf frequency (antonym-bearing
    words are more frequent: d = +0.634)
  - WORD-LEVEL split: no word in a test pair appears in any training pair
  - baselines: cosine alone, and a frequency-only probe

If AUC ~ 0.5 the information is genuinely absent. If AUC is high, the
distinction exists in the space and the operator's failure is one of
generation, not of representation.
"""
import json, random, collections
import numpy as np
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

D = "real/english"
RNG = np.random.default_rng(0)


def pos_of(w):
    return {s.pos() for s in wn.synsets(w)}


def coarse_pos(w):
    p = pos_of(w)
    if {"a", "s"} & p:
        return "adj"
    if "v" in p:
        return "verb"
    if "n" in p:
        return "noun"
    return "other"


def build_pairs(vocab_set):
    rels = json.load(open(f"{D}/relations.json"))
    ants = sorted({tuple(sorted(p)) for p in map(tuple, rels["lex:antonym"])})
    ants = [p for p in ants if p[0] in vocab_set and p[1] in vocab_set]

    # true synonyms: two lemmas of the SAME synset
    syns = set()
    for s in wn.all_synsets():
        names = [l.name().lower() for l in s.lemmas()
                 if l.name().lower() in vocab_set and l.name().isalpha()]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if names[i] != names[j]:
                    syns.add(tuple(sorted((names[i], names[j]))))
    syns -= set(ants)
    return ants, sorted(syns)


def match(ants, syns):
    """Match each antonym pair to a synonym pair with the same POS pattern and
    a similar mean Zipf frequency. Antonym-bearing words are more frequent, so
    an unmatched probe could win on frequency alone."""
    def key(p):
        return tuple(sorted((coarse_pos(p[0]), coarse_pos(p[1]))))

    def zf(p):
        return (zipf_frequency(p[0], "en") + zipf_frequency(p[1], "en")) / 2

    buckets = collections.defaultdict(list)
    for p in syns:
        buckets[key(p)].append((zf(p), p))
    for k in buckets:
        buckets[k].sort()

    out_a, out_s = [], []
    used = set()
    for p in ants:
        k = key(p)
        cands = buckets.get(k)
        if not cands:
            continue
        target = zf(p)
        # nearest unused synonym pair by frequency
        best, bi = None, None
        lo = np.searchsorted([c[0] for c in cands], target)
        for idx in range(max(0, lo - 12), min(len(cands), lo + 12)):
            z, q = cands[idx]
            if q in used:
                continue
            dd = abs(z - target)
            if best is None or dd < best:
                best, bi = dd, idx
        if bi is None or best > 0.5:      # require a close frequency match
            continue
        used.add(cands[bi][1])
        out_a.append(p)
        out_s.append(cands[bi][1])
    return out_a, out_s


def feats(pairs, P, widx):
    A = P[[widx[a] for a, _ in pairs]]
    B = P[[widx[b] for _, b in pairs]]
    d = (A - B).abs()
    m = (A + B) / 2
    prod = A * B
    cos = F.cosine_similarity(A, B, dim=-1, eps=1e-8).unsqueeze(1)
    return torch.cat([d, prod, m, cos], 1).numpy(), cos.numpy()


def freq_feats(pairs):
    return np.array([[zipf_frequency(a, "en"), zipf_frequency(b, "en"),
                      abs(zipf_frequency(a, "en") - zipf_frequency(b, "en"))]
                     for a, b in pairs])


def word_level_split(pairs_a, pairs_s, frac=0.3, seed=0):
    """Test words appear in NO training pair, of either class."""
    rng = random.Random(seed)
    allw = sorted({w for p in pairs_a + pairs_s for w in p})
    rng.shuffle(allw)
    test_w = set(allw[:int(len(allw) * frac)])

    def sel(pairs, in_test):
        return [i for i, (a, b) in enumerate(pairs)
                if ((a in test_w or b in test_w) if in_test
                    else (a not in test_w and b not in test_w))]
    return (sel(pairs_a, False), sel(pairs_s, False),
            sel(pairs_a, True), sel(pairs_s, True))


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    P = F.normalize(torch.stack([protos[w] for w in vocab]), dim=-1)

    ants, syns = build_pairs(set(vocab))
    print(f"antonym pairs: {len(ants)};  true co-synset synonym pairs: {len(syns)}")
    a_m, s_m = match(ants, syns)
    print(f"after POS + frequency matching: {len(a_m)} of each\n")

    za = np.mean([zipf_frequency(a, "en") for p in a_m for a in p])
    zs = np.mean([zipf_frequency(a, "en") for p in s_m for a in p])
    print(f"mean zipf: antonym pairs {za:.3f}  synonym pairs {zs:.3f}  "
          f"(matched)")
    ca = np.mean([F.cosine_similarity(P[widx[a]], P[widx[b]], dim=0).item() for a, b in a_m])
    cs = np.mean([F.cosine_similarity(P[widx[a]], P[widx[b]], dim=0).item() for a, b in s_m])
    print(f"mean cos : antonym pairs {ca:.3f}  synonym pairs {cs:.3f}\n")

    Xa, cos_a = feats(a_m, P, widx)
    Xs, cos_s = feats(s_m, P, widx)
    Fa, Fs = freq_feats(a_m), freq_feats(s_m)

    aucs = collections.defaultdict(list)
    for seed in range(5):
        tra, trs, tea, tes = word_level_split(a_m, s_m, seed=seed)
        if min(len(tra), len(trs), len(tea), len(tes)) < 20:
            continue
        ytr = np.r_[np.ones(len(tra)), np.zeros(len(trs))]
        yte = np.r_[np.ones(len(tea)), np.zeros(len(tes))]

        def run(Xa_, Xs_, name):
            Xtr = np.r_[Xa_[tra], Xs_[trs]]
            Xte = np.r_[Xa_[tea], Xs_[tes]]
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(Xtr), ytr)
            aucs[name].append(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))

        run(Xa, Xs, "full embedding pair probe")
        run(cos_a, cos_s, "cosine(a,b) alone")
        run(Fa, Fs, "word frequency alone")

    print(f"{'probe':<32}{'AUC (5 word-level splits)':>28}")
    print("-" * 60)
    for k in ("full embedding pair probe", "cosine(a,b) alone",
              "word frequency alone"):
        v = aucs[k]
        print(f"{k:<32}{np.mean(v):>18.3f} +/- {np.std(v):.3f}")

    print("\nREADING")
    print("  AUC ~ 0.5      -> polarity is ABSENT from the space. 'No operator")
    print("                    can separate them' is right, and unimprovable.")
    print("  AUC >> 0.5     -> polarity is PRESENT in the pair. The operator's")
    print("                    failure is GENERATION (produce b from a alone,")
    print("                    against 38k candidates), not representation.")


if __name__ == "__main__":
    main()
