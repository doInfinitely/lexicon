"""What SHAPE is a relation, and does a word have one position?

Two questions, tested in the FROZEN distilbert space -- no learned adapter,
no learned operators -- so that the answers are about the embedding geometry
and about language, not about anything we fitted.

QUESTION 1: what shape is each relation?
  A constant offset cannot be an involution. If f(x) = x + d and f(f(x)) = x
  then d = 0. So antonymy -- which IS an involution (dead->alive->dead) --
  provably cannot be a translation in ANY space. That is algebra, not a BERT
  artifact. But a REFLECTION is an involution and is exactly information
  preserving. So we fit competing shapes to every relation and let held-out
  pairs choose:

    identity     t = s
    translation  t = s + d                       (the word2vec premise)
    point-refl   t = 2c - s                      involution, center c
    householder  t = (I - 2vv')s + b             involution, mirror plane v
    affine       t = Ws + b        (ridge)       general upper bound
    orthogonal   t = Qs,  Q'Q = I  (Procrustes)  rigid, information-preserving

  Prediction if the involution argument is right: antonyms should be badly
  served by translation and well served by the reflections, while inflectional
  morphology should be the other way round.

QUESTION 2: does a word have one position?
  Every word here has a cloud of contextual occurrences. We compare
    single-point : one mean vector per word
    multi-point  : K sense centroids per word, and for each relation pair we
                   take the sense pair that the relation best explains.
  If words genuinely emerge at multiple points, the multi-point fit should
  improve relations that involve polysemy -- and, crucially, must be checked
  against a null, because free choice among K x K sense pairs can improve any
  fit by chance alone.
"""
import json, collections
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAT = {"I": "inflectional", "D": "derivational",
       "E": "encyclopedic", "L": "lexicographic"}
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- shapes
def fit_translation(S, T):
    d = (T - S).mean(0, keepdim=True)
    return lambda X: X + d


def fit_point_reflection(S, T):
    # t = 2c - s  =>  c = (s + t)/2 must be constant; take its mean
    c = ((S + T) / 2).mean(0, keepdim=True)
    return lambda X: 2 * c - X


def fit_householder(S, T):
    """t = (I - 2vv')s + b. Solve for the mirror plane by taking the leading
    direction of the differences (s - t) and the offset that centres it."""
    Dm = S - T
    v = torch.linalg.svd(Dm - Dm.mean(0, keepdim=True), full_matrices=False)[2][0]
    v = v / v.norm()
    b = (T - (S - 2 * (S @ v).unsqueeze(1) * v.unsqueeze(0))).mean(0, keepdim=True)
    return lambda X: X - 2 * (X @ v).unsqueeze(1) * v.unsqueeze(0) + b


def fit_affine(S, T, lam=1.0):
    Sa = torch.cat([S, torch.ones(len(S), 1, device=S.device)], 1)
    A = Sa.T @ Sa + lam * torch.eye(Sa.shape[1], device=S.device)
    W = torch.linalg.solve(A, Sa.T @ T)
    return lambda X: torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1) @ W


def fit_orthogonal(S, T):
    """Procrustes: the best rotation/reflection. Rigid, so fully information
    preserving -- it cannot collapse the space."""
    U, _, Vt = torch.linalg.svd(S.T @ T, full_matrices=False)
    Q = U @ Vt
    return lambda X: X @ Q


SHAPES = {"identity": lambda S, T: (lambda X: X),
          "translation": fit_translation,
          "point_reflection": fit_point_reflection,
          "householder": fit_householder,
          "orthogonal": fit_orthogonal,
          "affine": fit_affine}


def evaluate(fit_fn, S_tr, T_tr, S_te, T_te):
    try:
        f = fit_fn(S_tr, T_tr)
        pred = f(S_te)
    except Exception:
        return float("nan")
    return F.cosine_similarity(pred, T_te, dim=-1).mean().item()


def involution_error(fit_fn, S, T):
    """How close is the fitted map to being its own inverse?"""
    try:
        f = fit_fn(S, T)
        return F.cosine_similarity(f(f(S)), S, dim=-1).mean().item()
    except Exception:
        return float("nan")


# ---------------------------------------------------------------- data
def load_bats():
    vocab = json.load(open("harbor/workspace/data/vocab.json"))
    rels = json.load(open("harbor/workspace/data/relations.json"))
    protos = torch.load("real/embeddings/prototypes.pt", weights_only=False)
    senses = torch.load("real/embeddings/senses.pt", weights_only=False)
    P = {w: F.normalize(protos[w], dim=-1).to(DEVICE) for w in vocab}
    Sn = {w: F.normalize(torch.stack(senses[w]), dim=-1).to(DEVICE) for w in vocab}
    pairs = {r: [(p["source"], p["target"]) for p in m["pairs"]
                 if p["source"] != p["target"]] for r, m in rels.items()}
    return vocab, pairs, P, Sn


def split(pl, frac=0.3, seed=0):
    idx = np.random.default_rng(seed).permutation(len(pl))
    k = int(len(pl) * frac)
    return [pl[i] for i in idx[k:]], [pl[i] for i in idx[:k]]


# ---------------------------------------------------------------- Q1
def question_one(pairs, P):
    print("=" * 112)
    print("Q1: WHAT SHAPE IS EACH RELATION?   held-out cosine of the fitted map")
    print("    (a translation cannot be an involution: x + 2d = x forces d = 0)")
    print("=" * 112)
    hdr = f"{'relation':<30}{'cat':<15}" + "".join(f"{k[:11]:>13}" for k in SHAPES)
    print(hdr)
    rows = {}
    for r, pl in sorted(pairs.items()):
        if len(pl) < 30:
            continue
        tr, te = split(pl)
        S_tr = torch.stack([P[a] for a, _ in tr]); T_tr = torch.stack([P[b] for _, b in tr])
        S_te = torch.stack([P[a] for a, _ in te]); T_te = torch.stack([P[b] for _, b in te])
        row = {k: evaluate(f, S_tr, T_tr, S_te, T_te) for k, f in SHAPES.items()}
        rows[r] = row
        print(f"{r:<30}{CAT[r[0]]:<15}" + "".join(f"{row[k]:>13.3f}" for k in SHAPES))

    print("\nmean by category (which shape wins where):")
    bycat = collections.defaultdict(lambda: collections.defaultdict(list))
    for r, row in rows.items():
        for k, v in row.items():
            bycat[CAT[r[0]]][k].append(v)
    print(f"{'category':<18}" + "".join(f"{k[:11]:>13}" for k in SHAPES) + "   winner")
    for c, d in bycat.items():
        m = {k: float(np.nanmean(v)) for k, v in d.items()}
        best = max((k for k in m if k not in ("affine",)), key=lambda k: m[k])
        print(f"{c:<18}" + "".join(f"{m[k]:>13.3f}" for k in SHAPES) + f"   {best}")

    print("\ninvolution check  cos(f(f(x)), x)  -- 1.0 means the map is its own inverse")
    print(f"{'relation':<30}" + "".join(f"{k[:11]:>13}" for k in
                                        ("translation", "point_reflection",
                                         "householder", "orthogonal")))
    for r in ("L10_antonyms_binary", "L09_antonyms_gradable",
              "I01_noun_plural_reg", "L01_hypernyms_animals"):
        if r not in pairs:
            continue
        pl = pairs[r]
        S = torch.stack([P[a] for a, _ in pl]); T = torch.stack([P[b] for _, b in pl])
        vals = [involution_error(SHAPES[k], S, T) for k in
                ("translation", "point_reflection", "householder", "orthogonal")]
        print(f"{r:<30}" + "".join(f"{v:>13.3f}" for v in vals))
    return rows


# ---------------------------------------------------------------- Q2
def question_two(pairs, P, Sn, rows):
    print("\n" + "=" * 112)
    print("Q2: DOES A WORD HAVE ONE POSITION?")
    print("    single-point = the word's mean vector")
    print("    multi-point  = choose, per pair, the sense pair the relation best explains")
    print("    NULL         = same freedom of choice, but senses shuffled between words")
    print("=" * 112)
    print(f"{'relation':<30}{'senses/word':>13}{'single':>10}{'multi':>10}"
          f"{'null':>10}{'multi-null':>12}")

    for r, pl in sorted(pairs.items()):
        if len(pl) < 30:
            continue
        tr, te = split(pl)
        # fit a translation on the training pairs, single-point
        d = torch.stack([P[b] - P[a] for a, b in tr]).mean(0)

        def score(get_s, get_t, data):
            out = []
            for a, b in data:
                Sa, Tb = get_s(a), get_t(b)
                # best sense pair under the fitted translation
                pred = Sa + d                                  # [Ka, 768]
                sims = F.normalize(pred, dim=-1) @ F.normalize(Tb, dim=-1).T
                out.append(sims.max().item())
            return float(np.mean(out))

        single = score(lambda w: P[w].unsqueeze(0), lambda w: P[w].unsqueeze(0), te)
        multi = score(lambda w: Sn[w], lambda w: Sn[w], te)
        # null: give the same K x K freedom, but with senses from random words
        shuffled = {w: Sn[list(Sn)[RNG.integers(len(Sn))]] for w, _ in te}
        shuffled.update({w: Sn[list(Sn)[RNG.integers(len(Sn))]] for _, w in te})
        null = score(lambda w: shuffled[w], lambda w: shuffled[w], te)
        ks = float(np.mean([len(Sn[a]) for a, _ in te] + [len(Sn[b]) for _, b in te]))
        print(f"{r:<30}{ks:>13.2f}{single:>10.3f}{multi:>10.3f}{null:>10.3f}"
              f"{multi - null:>12.3f}")

    print("\n'multi - null' is the honest gain from words having several positions:")
    print("free choice among K x K sense pairs inflates ANY fit, so the null "
          "subtracts that.")


def main():
    vocab, pairs, P, Sn = load_bats()
    rows = question_one(pairs, P)
    question_two(pairs, P, Sn, rows)
    json.dump({r: {k: (None if np.isnan(v) else v) for k, v in row.items()}
               for r, row in rows.items()},
              open("real/geometry.json", "w"), indent=1)
    print("\nwrote real/geometry.json")


if __name__ == "__main__":
    main()
