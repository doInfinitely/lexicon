"""Is a paradigm ONE object, or a bag of pairwise operator applications?

Forget storage. The structural claim behind "lemmas are virtual words" is that
a lexeme is a single latent object which its surface forms realise. If true, it
makes a prediction that has nothing to do with compression:

    observing MORE forms of a paradigm should locate the lexeme better,
    and therefore predict a HELD-OUT form better.

Knowing {run, runs, running} should place RUN more precisely than {run} alone,
and so produce `ran` more reliably. If a paradigm is merely a set of pairwise
maps from the lemma, extra forms are irrelevant: `ran` depends only on `run`.

This also defeats the degenerate reading. A free vector z per paradigm can
memorise anything it is shown -- so we never show it the target. z is inferred
from the observed forms only; the target form is decoded from z afterwards.

  base        z = the lemma's embedding, and the target is produced by the
              slot operator. This is the standard scheme.
  lexeme(k)   z inferred from k observed forms (the lemma plus k-1 others),
              operators frozen, target never seen.
  centroid(k) z = mean of the k observed forms. A virtual word with no
              learning at all -- the null that killed my last virtual words.

Reported separately for REGULAR paradigms (the form is what the affix rule
predicts) and IRREGULAR ones (child/children, go/went), because that is where
an abstract lexeme should earn its existence -- or fail to.
"""
import json, os, collections, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = os.environ.get("LEXICON_DIR", "real/surface")
INFL = ["infl:noun_plural", "infl:verb_3pSg", "infl:verb_Ving",
        "infl:verb_Ved", "infl:verb_Ven", "infl:adj_comparative",
        "infl:adj_superlative"]
SLOTS = ["lemma"] + INFL

REGULAR = {
    "infl:noun_plural": lambda b: {b + "s", b + "es", b[:-1] + "ies"},
    "infl:verb_3pSg": lambda b: {b + "s", b + "es", b[:-1] + "ies"},
    "infl:verb_Ving": lambda b: {b + "ing", b[:-1] + "ing", b + b[-1] + "ing"},
    "infl:verb_Ved": lambda b: {b + "ed", b + "d", b[:-1] + "ied", b + b[-1] + "ed"},
    "infl:verb_Ven": lambda b: {b + "ed", b + "d", b[:-1] + "ied", b + b[-1] + "ed"},
    "infl:adj_comparative": lambda b: {b + "er", b + "r", b[:-1] + "ier", b + b[-1] + "er"},
    "infl:adj_superlative": lambda b: {b + "est", b + "st", b[:-1] + "iest", b + b[-1] + "est"},
}


def is_regular(base, slot, form):
    try:
        return form in REGULAR[slot](base)
    except Exception:
        return False


class Ops(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.idx = {s: i for i, s in enumerate(SLOTS)}
        self.W = nn.Parameter(torch.eye(dim).repeat(len(SLOTS), 1, 1))
        self.b = nn.Parameter(torch.zeros(len(SLOTS), dim))

    def forward(self, z, slot):
        i = self.idx[slot]
        return z @ self.W[i].T + self.b[i]


def abtt_space(P, k=8):
    mu = P.mean(0, keepdim=True)
    X = P - mu
    Vt = torch.linalg.svd(X, full_matrices=False)[2][:k]
    return F.normalize(X - (X @ Vt.T) @ Vt, dim=-1)


def infer_z(ops, observed, table, steps=250, lr=0.08, init=None):
    """Fit z to the OBSERVED forms only. Operators frozen."""
    if init is None:
        init = torch.stack([table[i] for _, i in observed]).mean(0)
    z = init.clone().unsqueeze(0).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        loss = 0.0
        for slot, i in observed:
            out = F.normalize(ops(z, slot), dim=-1)
            loss = loss + F.cross_entropy(out @ table.T / 0.05,
                                          torch.tensor([i], device=table.device))
        opt.zero_grad(); loss.backward(); opt.step()
    return z.detach()


@torch.no_grad()
def hits(ops, z, slot, gold, table):
    out = F.normalize(ops(z, slot), dim=-1)
    return int((out @ table.T).argmax(-1).item() == gold)


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    table = abtt_space(P)
    rels = json.load(open(f"{D}/relations.json"))

    par = collections.defaultdict(dict)
    for r in INFL:
        for a, b in rels.get(r, []):
            if a in widx and b in widx:
                par[a][r] = b
    par = {b: f for b, f in par.items() if len(f) >= 3}   # need >=3 to vary k
    bases = sorted(par)
    random.Random(0).shuffle(bases)
    k = int(len(bases) * 0.7)
    tr_b, te_b = bases[:k], bases[k:]
    print(f"paradigms with >=3 inflected forms: {len(par)} "
          f"(train {len(tr_b)}, held-out {len(te_b)})")
    nreg = sum(1 for b in bases for s, f in par[b].items() if is_regular(b, s, f))
    ntot = sum(len(par[b]) for b in bases)
    print(f"inflected forms: {ntot}, of which regular {nreg} ({nreg/ntot:.0%})\n")

    # ---- train operators jointly with train-paradigm lexemes ----
    ops = Ops().to(DEVICE)
    Z = nn.Parameter(torch.stack([table[widx[b]] for b in tr_b]).clone())
    opt = torch.optim.Adam(list(ops.parameters()) + [Z], lr=3e-3)
    print("training operators + train lexemes ...")
    for ep in range(30):
        perm = torch.randperm(len(tr_b))
        tot, nb = 0.0, 0
        for i in range(0, len(tr_b), 128):
            sel = perm[i:i + 128].tolist()
            loss = 0.0
            for j in sel:
                b = tr_b[j]
                z = Z[j:j + 1]
                for slot, gi in [("lemma", widx[b])] + [(s, widx[f]) for s, f in par[b].items()]:
                    out = F.normalize(ops(z, slot), dim=-1)
                    loss = loss + F.cross_entropy(out @ table.T / 0.05,
                                                  torch.tensor([gi], device=DEVICE))
            loss = loss / len(sel)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 10 == 0:
            print(f"  epoch {ep:2d} loss {tot/nb:.3f}", flush=True)

    # ---- paradigm completion on held-out paradigms ----
    print("\nPARADIGM COMPLETION: predict a form never shown to z.\n")
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    rng = random.Random(1)
    for b in te_b[:350]:
        forms = list(par[b].items())
        if len(forms) < 3:
            continue
        rng.shuffle(forms)
        tgt_slot, tgt_form = forms[0]
        gold = widx[tgt_form]
        reg = "regular" if is_regular(b, tgt_slot, tgt_form) else "irregular"
        others = forms[1:]

        # base scheme: z = lemma embedding (k=1, no inference)
        zb = table[widx[b]].unsqueeze(0)
        res[reg]["base (lemma only)"].append(hits(ops, zb, tgt_slot, gold, table))

        for nobs in (1, 2, 3):
            obs = [("lemma", widx[b])] + [(s, widx[f]) for s, f in others[:nobs - 1]]
            if len(obs) < nobs:
                continue
            zc = torch.stack([table[i] for _, i in obs]).mean(0, keepdim=True)
            res[reg][f"centroid({nobs})"].append(hits(ops, zc, tgt_slot, gold, table))
            zl = infer_z(ops, obs, table)
            res[reg][f"lexeme({nobs})"].append(hits(ops, zl, tgt_slot, gold, table))

    order = ["base (lemma only)", "centroid(1)", "lexeme(1)", "centroid(2)",
             "lexeme(2)", "centroid(3)", "lexeme(3)"]
    print(f"{'scheme':<22}{'regular':>12}{'irregular':>12}{'all':>10}")
    print("-" * 56)
    for k_ in order:
        r = res["regular"][k_]; ir = res["irregular"][k_]
        if not r and not ir:
            continue
        allv = r + ir
        print(f"{k_:<22}{np.mean(r) if r else float('nan'):>12.3f}"
              f"{np.mean(ir) if ir else float('nan'):>12.3f}{np.mean(allv):>10.3f}")
    print(f"\nn: regular {len(res['regular']['lexeme(3)'])}, "
          f"irregular {len(res['irregular']['lexeme(3)'])}")
    print("\nIf lexeme(3) > lexeme(1), observing more forms LOCATES the lexeme:")
    print("the paradigm is one object, not a bag of pairwise maps.")
    print("If lexeme(k) ~= centroid(k), the 'virtual word' is just an average.")
    json.dump({r: {k_: float(np.mean(v)) for k_, v in d.items() if v}
               for r, d in res.items()}, open(f"{D}/paradigm.json", "w"), indent=1)


if __name__ == "__main__":
    main()
