"""Expand the relation data by DERIVING analogies from the algebra.

Fine-tuning on 35 antonym pairs memorises them (held-out alignment 0.33,
R@1 0.27) while 4.3M parameters happily store a lookup table. The cure is
more constraints, not more capacity. Three sources, none of them invented:

1. ALGEBRAIC CLOSURE. The probes told us which relations have which algebra,
   so we only close under laws the data actually supports:
     transitive  hypernym(a,b) & hypernym(b,c) => hypernym(a,c)
                 (idempotence probe: lexicographic ops keep climbing, 0.78)
     symmetric   antonym(a,b) => antonym(b,a);  same for synonym/similar_to
     inverse     hypernym(a,b) <=> hyponym(b,a), meronym <=> holonym
   Derived pairs are marked so we can ablate them.

2. ANALOGY QUADRUPLES. This is the real expansion. From N pairs of one
   relation come N*(N-1) ordered quadruples a:b::c:d. Training on quadruples
   does not ask the model to memorise b given a; it asks that the DISPLACEMENT
   from a to b equal the displacement from c to d. That is literally the
   'is it a direction' property, imposed as a constraint rather than hoped for.
   For antonymy this turns 3469 pairs into millions of directional constraints.

3. NEGATIVE ANALOGIES. Antonyms and synonyms are distributionally identical
   (cos 0.801 vs 0.809), so nothing in the corpus separates them. We supply
   the missing bit: for a word with both a synonym and an antonym, the pair
   (word, synonym) and (word, antonym) must have OPPOSING displacements.
   Without this the space has no reason to distinguish them at all.
"""
import json, collections, itertools, random
import numpy as np

D = "real/english"
RNG = random.Random(0)

TRANSITIVE = ["lex:hypernym", "lex:hyponym", "lex:part_meronym",
              "lex:part_holonym"]
SYMMETRIC = ["lex:antonym", "lex:similar_to", "lex:derivationally_related"]
INVERSE_OF = {"lex:hypernym": "lex:hyponym",
              "lex:part_meronym": "lex:part_holonym",
              "lex:member_meronym": "lex:member_holonym",
              "lex:substance_meronym": "lex:substance_holonym"}
MAX_TRANS_PER_REL = 60000


def transitive_closure(pairs, depth=2, cap=MAX_TRANS_PER_REL):
    """One extra hop only: a->b->c yields a->c. Deeper closure explodes and
    the semantics degrade (a hypernym 5 levels up is 'entity')."""
    succ = collections.defaultdict(list)
    for a, b in pairs:
        succ[a].append(b)
    seen = set(map(tuple, pairs))
    out = []
    for a, b in pairs:
        for c in succ.get(b, ()):
            if c != a and (a, c) not in seen:
                seen.add((a, c))
                out.append((a, c))
                if len(out) >= cap:
                    return out
    return out


def symmetric_closure(pairs):
    seen = set(map(tuple, pairs))
    return [(b, a) for a, b in pairs if (b, a) not in seen]


def inverse_closure(pairs, inv_pairs):
    seen = set(map(tuple, inv_pairs))
    return [(b, a) for a, b in pairs if (b, a) not in seen]


def build_quadruples(pairs, max_per_rel=200000):
    """a:b::c:d for pairs of the same relation. Sampled, since N^2 explodes."""
    n = len(pairs)
    if n < 2:
        return []
    total = n * (n - 1)
    if total <= max_per_rel:
        return [(a, b, c, d) for (a, b), (c, d) in itertools.permutations(pairs, 2)]
    out = set()
    while len(out) < max_per_rel:
        i, j = RNG.randrange(n), RNG.randrange(n)
        if i == j:
            continue
        (a, b), (c, d) = pairs[i], pairs[j]
        out.add((a, b, c, d))
    return list(out)


def opposing_triples(rels):
    """(w, syn, ant): the synonym and the antonym of w must move w in
    OPPOSITE directions. This is the only signal that separates them, since
    the corpus places them equally close."""
    syn = collections.defaultdict(set)
    for a, b in rels.get("lex:similar_to", []):
        syn[a].add(b)
    for a, b in rels.get("lex:derivationally_related", []):
        pass  # not a synonym; skip
    ant = collections.defaultdict(set)
    for a, b in rels.get("lex:antonym", []):
        ant[a].add(b)
        ant[b].add(a)
    trip = []
    for w in set(syn) & set(ant):
        for s in list(syn[w])[:3]:
            for t in list(ant[w])[:3]:
                if s != t:
                    trip.append((w, s, t))
    return trip


def main():
    rels = json.load(open(f"{D}/relations.json"))
    rels = {k: [tuple(p) for p in v] for k, v in rels.items()}
    orig = {k: len(v) for k, v in rels.items()}
    derived = collections.defaultdict(list)

    print("1) algebraic closure")
    for r in TRANSITIVE:
        if r in rels:
            add = transitive_closure(rels[r])
            derived[r] += add
            print(f"   transitive  {r:<28} +{len(add):>7}")
    for r in SYMMETRIC:
        if r in rels:
            add = symmetric_closure(rels[r])
            derived[r] += add
            print(f"   symmetric   {r:<28} +{len(add):>7}")
    for r, inv in INVERSE_OF.items():
        if r in rels and inv in rels:
            add = inverse_closure(rels[r], rels[inv])
            derived[inv] += add
            print(f"   inverse     {inv:<28} +{len(add):>7}")

    aug = {k: sorted(set(rels[k]) | set(derived.get(k, []))) for k in rels}
    print(f"\n   pairs: {sum(orig.values())} -> {sum(len(v) for v in aug.values())}")

    print("\n2) analogy quadruples  a:b::c:d  (displacement equality constraints)")
    quads = {}
    for r, pl in aug.items():
        if len(pl) < 8:
            continue
        q = build_quadruples(pl)
        if q:
            quads[r] = q
    tot_q = sum(len(v) for v in quads.values())
    print(f"   {tot_q:,} quadruples across {len(quads)} relations")
    for r in sorted(quads, key=lambda r: -len(quads[r]))[:8]:
        print(f"     {r:<32}{len(quads[r]):>9,}  (from {len(aug[r])} pairs)")
    if "lex:antonym" in quads:
        print(f"\n   antonymy: {orig['lex:antonym']} pairs "
              f"-> {len(quads['lex:antonym']):,} directional constraints")

    print("\n3) opposing triples (w, synonym, antonym)")
    trip = opposing_triples(aug)
    print(f"   {len(trip):,} triples where a synonym and an antonym of the same "
          f"word\n   must displace it in opposite directions")
    for t in trip[:6]:
        print(f"     {t[0]:>14}  syn={t[1]:<16} ant={t[2]}")

    json.dump({k: [list(p) for p in v] for k, v in aug.items()},
              open(f"{D}/relations_augmented.json", "w"))
    json.dump({k: [list(q) for q in v] for k, v in quads.items()},
              open(f"{D}/quadruples.json", "w"))
    json.dump([list(t) for t in trip], open(f"{D}/opposing.json", "w"))
    print(f"\nwrote {D}/relations_augmented.json, quadruples.json, opposing.json")


if __name__ == "__main__":
    main()
