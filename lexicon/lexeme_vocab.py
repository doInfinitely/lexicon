"""A vocabulary of LEXEMES, spelled out by a small set of slot operators.

Distributed Morphology, done in embedding space. A lexeme is a category-neutral
root -- DECIDE -- and the surface words are its exponents:

    decide, decides, deciding, decided, decision, decisive, decisively

The lexeme is the centroid of its exponents (no learning needed -- the learned
version barely beat the mean). What must generalise is the small, SHARED set of
slot operators:

    lexeme -> verb.inf      lexeme -> noun.action    lexeme -> adj
    lexeme -> verb.past     lexeme -> noun.agent     lexeme -> adv
    lexeme -> noun.plural   ...

One linear map per slot, fitted across ALL training lexemes, applied to a
held-out lexeme's centroid. If a held-out word is retrieved at rank 1 out of
51k, the lexeme + slot operator generated it.

The comparison that matters, and the nulls:

  base anchor      the standard scheme: operators map the BASE WORD to each
                   exponent. The base is one exponent privileged over the rest.
  lexeme anchor    operators map the CENTROID to each exponent. No exponent is
                   privileged; the root is abstract, as linguists have it.
  identity         is the exponent simply the anchor's nearest neighbour?
  foreign lexeme   apply the slot operator to a DIFFERENT lexeme's centroid.
                   Does the operator need THIS root?

A held-out lexeme's centroid is computed from its exponents -- that is the
encoding step, and it is legitimate (the centroid is the stored code). What may
never be seen is the operator's behaviour on this root: the maps are fitted on
disjoint lexemes.
"""
import json, os, collections, random
import numpy as np
import torch
import torch.nn.functional as F

from lexicon.paradigm import abtt_space, DEVICE, D
from lexicon.atlas import CLOSED_FORM

# slot name -> the relation that realises it from the root
SLOTS = {
    "infl:noun_plural": "noun.plural",
    "infl:verb_3pSg": "verb.3sg",
    "infl:verb_Ving": "verb.ger",
    "infl:verb_Ved": "verb.past",
    "infl:verb_Ven": "verb.ptcp",
    "infl:adj_comparative": "adj.comp",
    "infl:adj_superlative": "adj.sup",
    "cook:suf_er_agent": "noun.agent",
    "cook:suf_tion": "noun.action",
    "cook:suf_ment": "noun.action2",
    "cook:suf_ness": "noun.quality",
    "cook:suf_ity": "noun.quality2",
    "cook:suf_ly": "adv",
    "cook:suf_ic": "adj.ic",
    "cook:suf_al": "adj.al",
    "cook:suf_ous": "adj.ous",
    "cook:suf_able": "adj.able",
    "cook:suf_ist": "noun.ist",
    "cook:suf_ism": "noun.ism",
    "cook:suf_ize": "verb.ize",
    "cook:suf_y_adj": "adj.y",
    "cook:suf_ful": "adj.ful",
    "cook:suf_less": "adj.less",
}
MIN_EXP = 3          # a lexeme needs at least this many exponents (incl. root)
MIN_SLOT_FIT = 40


def build_lexemes(rels, widx):
    """root word -> {slot: exponent}, from depth-1 morphological edges."""
    lex = collections.defaultdict(dict)
    for rel, slot in SLOTS.items():
        for a, b in rels.get(rel, []):
            if a in widx and b in widx and a != b:
                lex[a][slot] = b
    return {r: e for r, e in lex.items() if len(e) + 1 >= MIN_EXP}


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    table = abtt_space(torch.stack([protos[w] for w in vocab]).to(DEVICE))
    rels = json.load(open(f"{D}/relations.json"))

    lex = build_lexemes(rels, widx)
    roots = sorted(lex)
    random.Random(0).shuffle(roots)
    k = int(len(roots) * 0.7)
    tr, te = roots[:k], roots[k:]
    n_forms = sum(len(lex[r]) + 1 for r in roots)
    print(f"lexemes: {len(roots)}  (train {len(tr)}, held-out {len(te)})")
    print(f"surface forms they span: {n_forms}   "
          f"({n_forms/len(roots):.2f} exponents per lexeme)")
    used = collections.Counter(s for r in roots for s in lex[r])
    print(f"slots in use: {len(used)}")
    print("  " + ", ".join(f"{s}({n})" for s, n in used.most_common(8)) + " ...\n")

    def anchor(root, exps, kind):
        idxs = [widx[root]] + [widx[f] for f in exps.values()]
        if kind == "lexeme":
            return F.normalize(table[idxs].mean(0), dim=-1)
        return table[widx[root]]                       # base anchor

    results = {}
    for kind in ("base", "lexeme"):
        # fit one map per slot on TRAIN lexemes
        ops = {}
        for slot in used:
            X, Y = [], []
            for r in tr:
                if slot in lex[r]:
                    X.append(anchor(r, lex[r], kind))
                    Y.append(table[widx[lex[r][slot]]])
            if len(X) < MIN_SLOT_FIT:
                continue
            X, Y = torch.stack(X), torch.stack(Y)
            best, bname, bv = None, None, -1
            for name, fit in CLOSED_FORM.items():
                try:
                    f = fit(X, Y)
                    v = F.cosine_similarity(f(X), Y, dim=-1).mean().item()
                except Exception:
                    continue
                if v > bv:
                    best, bname, bv = f, name, v
            ops[slot] = (best, bname)

        # evaluate on HELD-OUT lexemes
        hit = tot = 0
        idhit = fhit = 0
        by_slot = collections.defaultdict(lambda: [0, 0])
        A = {r: anchor(r, lex[r], kind) for r in te}
        foreign = {r: A[te[(i + 7) % len(te)]] for i, r in enumerate(te)}
        with torch.no_grad():
            for r in te:
                for slot, form in lex[r].items():
                    if slot not in ops:
                        continue
                    f, _ = ops[slot]
                    gold = widx[form]
                    for vec, counter in ((f(A[r].unsqueeze(0)), "op"),
                                         (A[r].unsqueeze(0), "id"),
                                         (f(foreign[r].unsqueeze(0)), "fg")):
                        sims = F.normalize(vec, dim=-1) @ table.T
                        sims[0, widx[r]] = -2
                        ok = int(sims.argmax(-1).item() == gold)
                        if counter == "op":
                            hit += ok; by_slot[slot][0] += ok
                        elif counter == "id":
                            idhit += ok
                        else:
                            fhit += ok
                    tot += 1
                    by_slot[slot][1] += 1
        results[kind] = dict(op=hit / tot, ident=idhit / tot, foreign=fhit / tot,
                             n=tot, by_slot={s: v[0] / v[1] for s, v in by_slot.items()
                                             if v[1] >= 10})

    print(f"{'anchor':<14}{'exponents recovered':>22}{'identity null':>16}"
          f"{'foreign-lexeme null':>22}")
    print("-" * 76)
    for kind in ("base", "lexeme"):
        r = results[kind]
        print(f"{kind:<14}{r['op']:>22.3f}{r['ident']:>16.3f}{r['foreign']:>22.3f}")
    print(f"\n(n = {results['base']['n']} held-out exponents)")

    print("\nper-slot recovery from the LEXEME centroid:")
    bs = results["lexeme"]["by_slot"]
    for s, v in sorted(bs.items(), key=lambda kv: -kv[1]):
        b = results["base"]["by_slot"].get(s, float("nan"))
        print(f"   {s:<16}{v:>7.3f}   (base anchor {b:.3f})")

    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "by_slot"}
               for k, v in results.items()},
              open(f"{D}/lexeme_vocab.json", "w"), indent=1)
    print(f"\nwrote {D}/lexeme_vocab.json")


if __name__ == "__main__":
    main()
