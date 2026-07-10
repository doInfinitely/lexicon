"""Lemmas as virtual words: is the base form the best launchpad for a paradigm?

A paradigm is a lexeme and its surface forms: CAT -> {cat, cats};
RUN -> {run, runs, running, ran}. Two ways to store it:

  base scheme    store the lemma's embedding; derive the other forms with the
                 inflection operators. Cost: 1 vector + 1 per form that fails
                 to decode.
  lexeme scheme  store a VIRTUAL word z -- a vector that is not any surface
                 form -- and derive EVERY form from it, lemma included. Same
                 cost structure. The lexeme is what linguists mean by a
                 lexeme: abstract, realised by its forms, named by none.

The accounting is identical (one stored vector per paradigm), so the lexeme
wins only if it decodes forms the lemma cannot. Suppletion is the sharp case:
nothing about `go` predicts `went`, but a point between them might reach both.

Nulls, because "a learned vector helps" is exactly the claim I got wrong before:
  centroid   z = mean of the paradigm's forms. A virtual word requiring NO
             learning. If the learned z does not beat this, learning is noise.
  random     z ~ a random vocabulary vector. Should decode nothing.

Protocol. Operators are fitted on TRAIN paradigms only. For held-out
paradigms, z is optimised at encode time (that is legitimate: z IS the stored
code, like a JPEG's coefficients) while the operators stay frozen. Decoding a
form means rank-1 retrieval against the whole 51k vocabulary.
"""
import json, os, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = os.environ.get("LEXICON_DIR", "real/surface")
RNG = np.random.default_rng(0)
INFL = ["infl:noun_plural", "infl:verb_3pSg", "infl:verb_Ving",
        "infl:verb_Ved", "infl:verb_Ven", "infl:adj_comparative",
        "infl:adj_superlative"]


def build_paradigms(rels, vocab_set):
    """base -> {tag: form}"""
    par = collections.defaultdict(dict)
    for r in INFL:
        for a, b in rels.get(r, []):
            if a in vocab_set and b in vocab_set:
                par[a][r] = b
    return {b: f for b, f in par.items() if len(f) >= 2}


class Ops(nn.Module):
    """One linear map per slot, plus a 'lemma' slot for realising the base."""

    def __init__(self, slots, dim=768):
        super().__init__()
        self.slots = list(slots)
        self.idx = {s: i for i, s in enumerate(self.slots)}
        self.W = nn.Parameter(torch.eye(dim).repeat(len(self.slots), 1, 1))
        self.b = nn.Parameter(torch.zeros(len(self.slots), dim))

    def forward(self, z, slot):
        i = self.idx[slot]
        return z @ self.W[i].T + self.b[i]


def decode_ok(vec, table, gold_i, exclude=None):
    v = F.normalize(vec, dim=-1)
    sims = v @ table.T
    if exclude is not None:
        sims[:, exclude] = -2
    return (sims.argmax(-1) == gold_i)


def encode_z(ops, slots_forms, table, steps=300, lr=0.1, init=None):
    """Optimise the stored code z so that every form decodes. Operators frozen."""
    z = (init.clone() if init is not None
         else torch.stack([table[i] for _, i in slots_forms]).mean(0)).detach()
    z = z.unsqueeze(0).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    idx = torch.tensor([i for _, i in slots_forms], device=table.device)
    for _ in range(steps):
        loss = 0.0
        for (slot, i) in slots_forms:
            out = F.normalize(ops(z, slot), dim=-1)
            logits = out @ table.T / 0.05
            loss = loss + F.cross_entropy(logits, torch.tensor([i], device=table.device))
        opt.zero_grad(); loss.backward(); opt.step()
    return z.detach()


def count_ok(ops, slots_forms, z, table):
    n = 0
    with torch.no_grad():
        for slot, i in slots_forms:
            out = ops(z, slot)
            if decode_ok(out, table, torch.tensor([i], device=table.device)).item():
                n += 1
    return n


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    # all-but-the-top: the space that won the geometry test
    mu = P.mean(0, keepdim=True)
    X = P - mu
    Vt = torch.linalg.svd(X, full_matrices=False)[2][:8]
    table = F.normalize(X - (X @ Vt.T) @ Vt, dim=-1)

    rels = json.load(open(f"{D}/relations.json"))
    par = build_paradigms(rels, set(vocab))
    bases = sorted(par)
    RNG.shuffle(bases)
    k = int(len(bases) * 0.7)
    tr_b, te_b = bases[:k], bases[k:]
    print(f"paradigms: {len(par)}  (train {len(tr_b)}, held-out {len(te_b)})")
    sizes = collections.Counter(len(par[b]) + 1 for b in bases)
    print(f"forms per paradigm (incl. lemma): "
          f"{dict(sorted(sizes.items()))}")
    n_forms = sum(len(par[b]) + 1 for b in bases)
    print(f"total surface forms covered: {n_forms}\n")

    slots = ["lemma"] + INFL
    ops = Ops(slots).to(DEVICE)
    opt = torch.optim.Adam(ops.parameters(), lr=1e-3)
    Z = nn.Parameter(torch.stack([table[widx[b]] for b in tr_b]).clone())
    optz = torch.optim.Adam([Z], lr=1e-2)

    print("training operators + train-paradigm lexemes (alternating)...")
    for ep in range(40):
        perm = torch.randperm(len(tr_b))
        tot = 0.0
        for i in range(0, len(tr_b), 256):
            sel = perm[i:i + 256]
            loss = 0.0
            for j in sel.tolist():
                b = tr_b[j]
                pairs = [("lemma", widx[b])] + [(s, widx[f]) for s, f in par[b].items()]
                z = Z[j:j + 1]
                for slot, gi in pairs:
                    out = F.normalize(ops(z, slot), dim=-1)
                    loss = loss + F.cross_entropy(out @ table.T / 0.05,
                                                  torch.tensor([gi], device=DEVICE))
            loss = loss / len(sel)
            opt.zero_grad(); optz.zero_grad(); loss.backward(); opt.step(); optz.step()
            tot += loss.item()
        if ep % 10 == 0:
            print(f"  epoch {ep:2d}  loss {tot/max(1,len(tr_b)//256):.3f}", flush=True)

    print("\nHELD-OUT PARADIGMS: how many forms decode, per scheme?")
    print("(operators frozen; z optimised at encode time where applicable)\n")
    res = collections.defaultdict(list)
    for b in te_b[:400]:
        sf = [("lemma", widx[b])] + [(s, widx[f]) for s, f in par[b].items()]
        total = len(sf)
        # base scheme: lemma stored (free), derive the rest
        with torch.no_grad():
            zb = table[widx[b]].unsqueeze(0)
        ok_base = 1 + count_ok(ops, sf[1:], zb, table)
        # centroid null
        zc = torch.stack([table[i] for _, i in sf]).mean(0, keepdim=True)
        ok_cent = count_ok(ops, sf, zc, table)
        # learned lexeme (encode-time optimisation)
        zl = encode_z(ops, sf, table)
        ok_lex = count_ok(ops, sf, zl, table)
        # random null
        zr = table[RNG.integers(len(table))].unsqueeze(0)
        ok_rnd = count_ok(ops, sf, zr, table)
        for k_, v in (("base", ok_base), ("centroid", ok_cent),
                      ("lexeme", ok_lex), ("random", ok_rnd)):
            res[k_].append(v / total)
        res["total"].append(total)

    print(f"{'scheme':<28}{'forms decoded':>16}{'stored vectors / form':>24}")
    n = np.mean(res["total"])
    for k_ in ("base", "centroid", "lexeme", "random"):
        frac = float(np.mean(res[k_]))
        # cost: 1 stored code + one vector per failed form
        cost = (1 + (1 - frac) * n) / n
        print(f"{k_:<28}{frac:>16.3f}{cost:>24.3f}")
    print(f"\nmean forms per paradigm: {n:.2f}")
    print("\n'stored vectors / form' < 1 means compression. The lexeme scheme beats")
    print("the base scheme only if a virtual word decodes forms the lemma cannot.")
    json.dump({k_: float(np.mean(v)) for k_, v in res.items()},
              open(f"{D}/lexeme.json", "w"), indent=1)


if __name__ == "__main__":
    main()
