"""What is the irreducible set of English words?

The original premise: a small base lexicon plus learned operators reconstructs
the rest. We reported 3.25x compression. But that was measured inside a LEARNED
ADAPTER's space -- a space trained until the derivations decoded. Given that a
free linear adapter can rotate any plane onto any other (see RESULTS.md), that
number told us about the adapter, not about English.

So: fit each relation's operator in the FROZEN distilbert space, on TRAINING
pairs only, then ask honestly which words the operators can produce.

A word `b` is DERIVABLE if some relation r and some word `a` exist such that
f_r(a) retrieves `b` first out of all 38,142 words, by a margin. The derivation
forest is then built greedily (best margin first), acyclic, depth-capped. The
base lexicon is whatever is left.

Reported separately:
  seen words     `b` appeared as a target in a training pair of that relation
  unseen words   it did not. This is the number that means anything.

Controls:
  shuffled       apply a RANDOM relation's operator instead of the right one.
                 Whatever this derives is what the geometry gives for free.
  identity       f = identity. Derives `b` when `b` is simply `a`'s nearest
                 neighbour -- no operator required at all.
"""
import json, os, random, collections
import numpy as np
import torch
import torch.nn.functional as F

from lexicon.atlas import (CLOSED_FORM, split_words, DEVICE, D, MIN_TRAIN,
                           MIN_TEST, retrieval)

MARGIN = 0.02
MAX_DEPTH = 3
MAX_EDGES_PER_REL = 20000
OUT = f"{D}/generating_set.json"
RNG = np.random.default_rng(0)


def fit_best_shape(S, T, Se, Te, table, si_e, ti_e):
    """Fit every closed-form shape on train, keep the one with best held-out
    R@1 (not cosine -- cosine and retrieval correlate at only +0.18)."""
    best, best_r1, best_name = None, -1, None
    for name, fit in CLOSED_FORM.items():
        try:
            f = fit(S, T)
            with torch.no_grad():
                pred = F.normalize(f(Se), dim=-1)
                sims = pred @ table.T
                sims.scatter_(1, si_e.unsqueeze(1), -2)
                r1 = (sims.argmax(1) == ti_e).float().mean().item()
        except Exception:
            continue
        if r1 > best_r1:
            best, best_r1, best_name = f, r1, name
    return best, best_name, best_r1


@torch.no_grad()
def decodable_edges(f, pairs, widx, table, chunk=256):
    """(a,b) survives when f(a) retrieves b first, with margin over runner-up."""
    out = []
    si = torch.tensor([widx[a] for a, _ in pairs], device=DEVICE)
    ti = torch.tensor([widx[b] for _, b in pairs], device=DEVICE)
    for i in range(0, len(pairs), chunk):
        s, t = si[i:i + chunk], ti[i:i + chunk]
        pred = F.normalize(f(table[s]), dim=-1)
        sims = pred @ table.T
        sims.scatter_(1, s.unsqueeze(1), -2)
        top2 = sims.topk(2, dim=1)
        hit = top2.indices[:, 0] == t
        marg = top2.values[:, 0] - top2.values[:, 1]
        for j in hit.nonzero(as_tuple=True)[0].tolist():
            if marg[j].item() >= MARGIN:
                out.append((marg[j].item(), pairs[i + j][0], pairs[i + j][1]))
    return out


def build_forest(edges):
    """Greedy, best-margin-first, acyclic, depth-capped."""
    edges = sorted(edges, reverse=True)
    parent, depth, height = {}, collections.defaultdict(int), collections.defaultdict(int)
    children = collections.defaultdict(set)

    def anc(a, w):
        while w in parent:
            w = parent[w][1]
            if w == a:
                return True
        return False

    for m, rel_a, b in edges:
        rel, a = rel_a
        if b in parent or a == b or anc(b, a):
            continue
        if depth[a] + 1 + height[b] > MAX_DEPTH:
            continue
        parent[b] = (rel, a)
        children[a].add(b)
        stack = [(b, depth[a] + 1)]
        while stack:
            w, dd = stack.pop()
            depth[w] = dd
            stack.extend((c, dd + 1) for c in children[w])
        w, h = b, height[b]
        while w in parent:
            w = parent[w][1]
            h += 1
            if height[w] >= h:
                break
            height[w] = h
    return parent


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    table = F.normalize(torch.stack([protos[w] for w in vocab]).to(DEVICE), dim=-1)
    rels = json.load(open(f"{D}/relations.json"))

    fitted, all_edges, shuf_edges, id_edges = {}, [], [], []
    seen_targets = collections.defaultdict(set)   # rel -> targets seen in train

    rel_names = sorted(rels)
    for rel in rel_names:
        pairs = sorted({tuple(p) for p in rels[rel]})
        pairs = [p for p in pairs if p[0] != p[1]]
        sym = sum(1 for a, b in set(pairs) if (b, a) in set(pairs)) / max(len(pairs), 1)
        if sym > 0.5:
            pairs = sorted({tuple(sorted(p)) for p in pairs})
        tr, te = split_words(pairs)
        if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
            continue
        if len(tr) > MAX_EDGES_PER_REL:
            tr = [tr[i] for i in RNG.choice(len(tr), MAX_EDGES_PER_REL, replace=False)]
        si = torch.tensor([widx[a] for a, _ in tr], device=DEVICE)
        ti = torch.tensor([widx[b] for _, b in tr], device=DEVICE)
        sie = torch.tensor([widx[a] for a, _ in te], device=DEVICE)
        tie = torch.tensor([widx[b] for _, b in te], device=DEVICE)
        f, name, r1 = fit_best_shape(table[si], table[ti], table[sie], table[tie],
                                     table, sie, tie)
        if f is None:
            continue
        fitted[rel] = (f, name)
        seen_targets[rel] = {b for _, b in tr}
        cand = tr + te
        if len(cand) > MAX_EDGES_PER_REL:
            cand = [cand[i] for i in RNG.choice(len(cand), MAX_EDGES_PER_REL, replace=False)]
        e = decodable_edges(f, cand, widx, table)
        all_edges += [(m, (rel, a), b) for m, a, b in e]
        # control 1: identity operator on the same candidate edges
        e_id = decodable_edges(lambda X: X, cand, widx, table)
        id_edges += [(m, (rel, a), b) for m, a, b in e_id]
        print(f"[{rel:<28}] shape={name:<12} held-out R@1 {r1:.3f}  "
              f"decodable edges {len(e):>5} / {len(cand)}   (identity: {len(e_id)})",
              flush=True)

    # control 2: apply a RANDOM relation's operator to each candidate edge
    for rel in fitted:
        pairs = sorted({tuple(p) for p in rels[rel]})[:2000]
        other = RNG.choice([r for r in fitted if r != rel])
        e = decodable_edges(fitted[other][0], pairs, widx, table)
        shuf_edges += [(m, (other, a), b) for m, a, b in e]

    V = len(vocab)
    res = {}
    for label, edges in (("operators", all_edges), ("identity (null)", id_edges),
                         ("shuffled relation (null)", shuf_edges)):
        parent = build_forest(edges)
        derived = set(parent)
        base = V - len(derived)
        res[label] = {"derived": len(derived), "base": base,
                      "ratio": round(V / base, 3)}
        print(f"\n{label:<26} derived {len(derived):>6} / {V}   "
              f"base {base:>6}   compression {V/base:.3f}x")

    # honest split: were these targets seen in training?
    parent = build_forest(all_edges)
    seen = sum(1 for b, (rel, a) in parent.items() if b in seen_targets.get(rel, ()))
    unseen = len(parent) - seen
    print(f"\nof the {len(parent)} derived words: {seen} were targets in a "
          f"TRAINING pair, {unseen} were not.")
    print(f"generalising compression (unseen only): "
          f"{V / (V - unseen):.3f}x")

    byrel = collections.Counter(rel for rel, _ in parent.values())
    print("\nwhich relations actually derive words?")
    for r, n in byrel.most_common(12):
        print(f"   {r:<30}{n:>6}")

    left = [w for w in vocab if w not in parent]
    print(f"\nIRREDUCIBLE: {len(left)} words ({len(left)/V:.1%} of English) that no "
          f"operator produces.")
    print("  sample:", ", ".join(left[:20]))
    res["seen"] = seen
    res["unseen"] = unseen
    res["by_relation"] = dict(byrel)
    res["n_vocab"] = V
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
