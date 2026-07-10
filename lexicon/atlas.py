"""The English Relation Shape Atlas.

For each of WordNet's relations, what SHAPE is it in embedding space, and is
it structure at all? One row per relation, computed with every control this
project learned the hard way:

  word-level split      test words appear in NO training pair. Pair-level
                        splits leak: WordNet stores several relations 100%
                        symmetrically, and a naive split left 88% of held-out
                        antonym pairs with their reverse in training.
  canonicalisation      symmetric relations are deduped to sorted tuples
                        BEFORE splitting, detected automatically.
  matched nulls         detectability is measured against random pairs matched
                        on part of speech and Zipf frequency, because relation-
                        bearing words are systematically more frequent.
  full-vocab retrieval  R@1 ranks against all 38,142 words, source excluded.
  frozen space          shapes are fitted in RAW distilbert. No learned
                        adapter, so nothing can absorb the answer -- an
                        adapter makes the choice of shape unidentifiable
                        (see RESULTS.md: a random reflection plane matched a
                        trained one once an adapter was free to rotate).

Shapes, all fitted closed-form on the training pairs:

  identity      t = s                          (is the target just nearby?)
  translation   t = s + d                      (the word2vec premise)
  reflection    t = (I - 2vv')s + b            (an involution; antonymy's shape)
  orthogonal    t = Qs                         (rigid, information-preserving)
  affine        t = Ws + b, ridge              (general linear)
  mlp           t = s + MLP(s), trained        (the nonlinear ceiling)

Plus, per relation: is it a direction (displacement alignment)? how many
dimensions does the displacement occupy (participation ratio vs null)? is it
one-to-many (fan-in)? and is it even DETECTABLE (pair-probe AUC vs matched
random pairs)?

Results are written per relation to real/english/atlas.json as they complete,
so a machine-check reset costs at most one relation.
"""
import json, os, random, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = os.environ.get("LEXICON_DIR", "real/english")
OUT = f"{D}/atlas.json"
RNG = np.random.default_rng(0)
MIN_TRAIN, MIN_TEST = 60, 25


# ----------------------------------------------------------------- shapes
def fit_identity(S, T):
    return lambda X: X


def fit_translation(S, T):
    d = (T - S).mean(0, keepdim=True)
    return lambda X: X + d


def fit_reflection(S, T):
    """Householder: t = (I - 2vv')s + b. v from the leading difference
    direction; b centres it. Exactly involutive when b = 0."""
    Dm = S - T
    v = torch.linalg.svd(Dm - Dm.mean(0, keepdim=True), full_matrices=False)[2][0]
    v = v / v.norm()
    refl = lambda X: X - 2 * (X @ v).unsqueeze(1) * v.unsqueeze(0)
    b = (T - refl(S)).mean(0, keepdim=True)
    return lambda X: refl(X) + b


def fit_orthogonal(S, T):
    U, _, Vt = torch.linalg.svd(S.T @ T, full_matrices=False)
    Q = U @ Vt
    return lambda X: X @ Q


def fit_affine(S, T, lam=1.0):
    Sa = torch.cat([S, torch.ones(len(S), 1, device=S.device)], 1)
    A = Sa.T @ Sa + lam * torch.eye(Sa.shape[1], device=S.device)
    W = torch.linalg.solve(A, Sa.T @ T)
    return lambda X: torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1) @ W


CLOSED_FORM = {"identity": fit_identity, "translation": fit_translation,
               "reflection": fit_reflection, "orthogonal": fit_orthogonal,
               "affine": fit_affine}


class MLPOp(nn.Module):
    def __init__(self, dim=768, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


def fit_mlp(S, T, table, steps=1200, bs=256, lr=1e-3, tau=0.05, n_neg=4096):
    op = MLPOp().to(S.device)
    opt = torch.optim.AdamW(op.parameters(), lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator(device=S.device).manual_seed(0)
    V = len(table)
    for i in range(steps):
        idx = torch.randint(0, len(S), (min(bs, len(S)),), device=S.device, generator=g)
        out = F.normalize(op(S[idx]), dim=-1)
        negs = table[torch.randint(0, V, (n_neg,), device=S.device, generator=g)]
        pos = (out * F.normalize(T[idx], dim=-1)).sum(-1, keepdim=True) / tau
        neg = out @ negs.T / tau
        loss = (-pos.squeeze(1) + torch.logsumexp(torch.cat([pos, neg], 1), 1)).mean()
        loss = loss + (1 - F.cosine_similarity(out, T[idx], dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    op.eval()
    return lambda X: op(X)


# ----------------------------------------------------------------- metrics
@torch.no_grad()
def retrieval(pred, table, src_i, tgt_i, pos_sets, chunk=512):
    """R@1 over the full vocabulary, source excluded. `any` credits every
    target WordNet sanctions for that (relation, source)."""
    hits = anyh = 0
    for i in range(0, len(pred), chunk):
        p = F.normalize(pred[i:i + chunk], dim=-1)
        sims = p @ table.T
        s = src_i[i:i + chunk]
        sims.scatter_(1, s.unsqueeze(1), -2)
        top1 = sims.argmax(1)
        hits += (top1 == tgt_i[i:i + chunk]).sum().item()
        for j, t1 in enumerate(top1.tolist()):
            if t1 in pos_sets[i + j]:
                anyh += 1
    return hits / len(pred), anyh / len(pred)


def alignment(S, T):
    Dm = T - S
    mu = Dm.mean(0, keepdim=True)
    return F.cosine_similarity(Dm, mu, dim=-1).mean().item()


def eff_dim(Dm):
    C = (Dm.T @ Dm) / len(Dm)
    ev = torch.linalg.eigvalsh(C).flip(0).clamp(min=0)
    p = ev / ev.sum()
    return float(1.0 / (p ** 2).sum())


def coarse_pos(w, cache={}):
    if w not in cache:
        p = {s.pos() for s in wn.synsets(w)}
        cache[w] = ("adj" if {"a", "s"} & p else "verb" if "v" in p
                    else "noun" if "n" in p else "other")
    return cache[w]


def matched_random_pairs(pairs, vocab, n):
    """Random pairs matched to `pairs` on POS pattern and mean Zipf."""
    by_pos = collections.defaultdict(list)
    for w in vocab:
        by_pos[coarse_pos(w)].append(w)
    out = []
    for a, b in pairs[:n]:
        pa, pb = coarse_pos(a), coarse_pos(b)
        za, zb = zipf_frequency(a, "en"), zipf_frequency(b, "en")
        for _ in range(40):
            x = by_pos[pa][RNG.integers(len(by_pos[pa]))]
            y = by_pos[pb][RNG.integers(len(by_pos[pb]))]
            if x != y and abs(zipf_frequency(x, "en") - za) < 0.6 \
                    and abs(zipf_frequency(y, "en") - zb) < 0.6:
                out.append((x, y)); break
    return out


def pair_features(pairs, Pn, widx):
    A = Pn[[widx[a] for a, _ in pairs]]
    B = Pn[[widx[b] for _, b in pairs]]
    cos = F.cosine_similarity(A, B, dim=-1, eps=1e-8).unsqueeze(1)
    return torch.cat([(A - B).abs(), A * B, (A + B) / 2, cos], 1).cpu().numpy()


def detectability(tr_pairs, te_pairs, vocab, Pn, widx):
    """Can a probe tell a real pair of this relation from a matched random
    pair? Word-level split is inherited from the caller."""
    ntr = matched_random_pairs(tr_pairs, vocab, len(tr_pairs))
    nte = matched_random_pairs(te_pairs, vocab, len(te_pairs))
    if len(ntr) < 30 or len(nte) < 15:
        return float("nan")
    Xtr = np.r_[pair_features(tr_pairs, Pn, widx), pair_features(ntr, Pn, widx)]
    ytr = np.r_[np.ones(len(tr_pairs)), np.zeros(len(ntr))]
    Xte = np.r_[pair_features(te_pairs, Pn, widx), pair_features(nte, Pn, widx)]
    yte = np.r_[np.ones(len(te_pairs)), np.zeros(len(nte))]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))


# ----------------------------------------------------------------- driver
def split_words(pairs, frac=0.25, seed=0):
    rng = random.Random(seed)
    words = sorted({w for p in pairs for w in p})
    rng.shuffle(words)
    test_w = set(words[:int(len(words) * frac)])
    tr = [p for p in pairs if p[0] not in test_w and p[1] not in test_w]
    te = [p for p in pairs if (p[0] in test_w) ^ (p[1] in test_w)]
    return tr, te


def analyse(rel, raw_pairs, vocab, widx, Pn, table, do_mlp=True):
    # symmetric? then canonicalise before splitting
    s = set(raw_pairs)
    sym = sum(1 for a, b in s if (b, a) in s) / max(len(s), 1)
    pairs = sorted({tuple(sorted(p)) for p in s}) if sym > 0.5 else sorted(s)
    pairs = [p for p in pairs if p[0] != p[1]]
    tr, te = split_words(pairs)
    if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
        return {"skipped": f"train {len(tr)} test {len(te)}"}

    if len(tr) > 20000:
        tr = [tr[i] for i in RNG.choice(len(tr), 20000, replace=False)]
    if len(te) > 1200:
        te = [te[i] for i in RNG.choice(len(te), 1200, replace=False)]

    si = torch.tensor([widx[a] for a, _ in tr], device=DEVICE)
    ti = torch.tensor([widx[b] for _, b in tr], device=DEVICE)
    S, T = Pn[si], Pn[ti]
    si_e = torch.tensor([widx[a] for a, _ in te], device=DEVICE)
    ti_e = torch.tensor([widx[b] for _, b in te], device=DEVICE)
    Se, Te = Pn[si_e], Pn[ti_e]

    # every sanctioned target for each held-out source (for R@1-any)
    gold = collections.defaultdict(set)
    for a, b in raw_pairs:
        gold[a].add(widx[b])
        if sym > 0.5:
            gold[b].add(widx[a])
    pos_sets = [gold[a] for a, _ in te]

    row = {"n_pairs": len(pairs), "n_train": len(tr), "n_test": len(te),
           "symmetric": round(sym, 3)}

    # fan-in: how many distinct targets do the training sources map to?
    row["fan_in"] = round(len(tr) / max(len({b for _, b in tr}), 1), 2)

    # is it a direction, and how many dims does the displacement occupy?
    row["direction"] = round(alignment(Se, Te), 3)
    row["eff_dim"] = round(eff_dim(T - S), 1)
    rnd_i = torch.tensor(RNG.choice(len(Pn), (len(tr), 2)), device=DEVICE)
    row["eff_dim_null"] = round(eff_dim(Pn[rnd_i[:, 0]] - Pn[rnd_i[:, 1]]), 1)

    shapes = {}
    for name, fit in CLOSED_FORM.items():
        try:
            f = fit(S, T)
            pred = f(Se)
            r1, anyv = retrieval(pred, table, si_e, ti_e, pos_sets)
            shapes[name] = {"cos": round(F.cosine_similarity(pred, Te, dim=-1).mean().item(), 3),
                            "R1": round(r1, 3), "R1_any": round(anyv, 3)}
        except Exception as e:
            shapes[name] = {"error": str(e)[:60]}
    if do_mlp:
        f = fit_mlp(S, T, table)
        with torch.no_grad():
            pred = f(Se)
            r1, anyv = retrieval(pred, table, si_e, ti_e, pos_sets)
            shapes["mlp"] = {"cos": round(F.cosine_similarity(pred, Te, dim=-1).mean().item(), 3),
                             "R1": round(r1, 3), "R1_any": round(anyv, 3)}
    row["shapes"] = shapes
    linear_best = max((k for k in shapes if k != "mlp" and "R1" in shapes[k]),
                      key=lambda k: shapes[k]["R1"])
    row["best_linear"] = linear_best
    row["best_overall"] = max((k for k in shapes if "R1" in shapes[k]),
                              key=lambda k: shapes[k]["R1"])
    row["nonlinear_gain"] = round(
        shapes.get("mlp", {}).get("R1", float("nan")) - shapes[linear_best]["R1"], 3)

    # is the relation even detectable from the pair?
    row["detect_auc"] = round(detectability(tr[:1500], te, vocab, Pn, widx), 3)
    return row


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    Pn = F.normalize(torch.stack([protos[w] for w in vocab]).to(DEVICE), dim=-1)
    table = Pn
    rels = json.load(open(f"{D}/relations.json"))

    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    order = sorted(rels, key=lambda r: len(rels[r]))
    for rel in order:
        if rel in done:
            continue
        pairs = [tuple(p) for p in rels[rel]]
        print(f"[{rel}] {len(pairs)} pairs ...", flush=True)
        try:
            done[rel] = analyse(rel, pairs, vocab, widx, Pn, table)
        except Exception as e:
            done[rel] = {"error": str(e)[:120]}
        json.dump(done, open(OUT, "w"), indent=1)
        r = done[rel]
        if "skipped" in r or "error" in r:
            print(f"   -> {r}", flush=True)
        else:
            print(f"   -> best {r['best_overall']:<12} "
                  f"R@1 {r['shapes'][r['best_overall']]['R1']:.3f}  "
                  f"dir {r['direction']:.3f}  detect {r['detect_auc']:.3f}",
                  flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
