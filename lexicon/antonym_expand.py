"""Explode the antonym set, without inventing anything.

We had 3469 direct WordNet lemma antonyms. Four honest sources of more:

1. DIRECT, unfiltered. The original set was cut to zipf >= 2.0. Keep the cut
   (we need embeddings) but stop capping lemmas per synset.

2. INDIRECT ANTONYMY. This is how WordNet actually encodes adjective
   opposition. Only "head" adjectives carry antonym links; satellite
   adjectives attach to a head by `similar_to`. So if s ~ h and h <-> h',
   then every lemma of s is an antonym of every lemma of h'. Standard
   WordNet practice, not an invention. ('arid' ~ 'dry' <-> 'wet')

3. MORPHOLOGICAL NEGATION. If X and its negation are BOTH in the vocabulary
   and both are attested words, they are opposites: un-, in-, im-, il-, ir-,
   non-, dis-, a(n)-, plus the -ful/-less alternation and over-/under-.
   Every pair is checked against the vocabulary; nothing is generated.

4. DERIVATIONAL TRANSPORT. If a <-> b are antonyms and a -> a', b -> b' are
   derivationally related forms of the SAME part of speech pattern, then
   a' <-> b'. (happy/unhappy => happiness/unhappiness)

Every derived pair is tagged with its source so any of them can be ablated.
"""
import json, collections, re
from nltk.corpus import wordnet as wn

D = "real/english"

NEG_PREFIX = ["un", "in", "im", "il", "ir", "non", "dis", "mis", "anti"]


def direct(V):
    pairs = set()
    for s in wn.all_synsets():
        for l in s.lemmas():
            a = l.name().lower()
            if a not in V:
                continue
            for ant in l.antonyms():
                b = ant.name().lower()
                if b in V and a != b:
                    pairs.add(tuple(sorted((a, b))))
    return pairs


def indirect(V):
    """satellite ~ head <-> head' : satellite lemmas oppose head' lemmas."""
    pairs = set()
    for s in wn.all_synsets(pos=wn.ADJ):
        heads = s.similar_tos()
        if not heads:
            continue
        sat_lemmas = [l.name().lower() for l in s.lemmas()
                      if l.name().lower() in V]
        if not sat_lemmas:
            continue
        for h in heads:
            for hl in h.lemmas():
                for ant in hl.antonyms():
                    b = ant.name().lower()
                    if b not in V:
                        continue
                    # also carry to the antonym's own satellites
                    opp = [b] + [l.name().lower()
                                 for os in ant.synset().similar_tos()
                                 for l in os.lemmas()
                                 if l.name().lower() in V]
                    for a in sat_lemmas:
                        for c in opp[:6]:
                            if a != c:
                                pairs.add(tuple(sorted((a, c))))
    return pairs


def _pos(w):
    return {s.pos() for s in wn.synsets(w)}


def morphological(V):
    """A negated form is only an antonym if it is the SAME part of speech.
    Without this: 'able'(adj)/'disable'(verb), 'abuse'(n)/'disabuse'(v)."""
    pairs = set()
    for w in V:
        cands = []
        for p in NEG_PREFIX:
            if w.startswith(p) and len(w) > len(p) + 2:
                base = w[len(p):]
                if base in V:
                    cands.append((base, w))
        if w.endswith("ful"):
            less = w[:-3] + "less"
            if less in V:
                cands.append((w, less))
        if w.startswith("over") and len(w) > 6:
            u = "under" + w[4:]
            if u in V:
                cands.append((w, u))
        for a, b in cands:
            if _pos(a) & _pos(b):          # must share a part of speech
                pairs.add(tuple(sorted((a, b))))
    return pairs


def derivational(V, base_pairs):
    """transport antonymy across derivationally related forms."""
    der = collections.defaultdict(set)
    for s in wn.all_synsets():
        for l in s.lemmas():
            a = l.name().lower()
            if a not in V:
                continue
            for d in l.derivationally_related_forms():
                b = d.name().lower()
                if b in V and a != b:
                    der[a].add(b)
    pairs = set()
    for a, b in base_pairs:
        for a2 in list(der.get(a, ()))[:4]:
            for b2 in list(der.get(b, ()))[:4]:
                if a2 != b2:
                    # only if the two derived forms share the same suffix change
                    if a2[:3] != b2[:3] or len(a2) > 3:
                        pairs.add(tuple(sorted((a2, b2))))
    return pairs


def main():
    V = set(json.load(open(f"{D}/vocab.json")))
    print(f"vocabulary: {len(V)} words (all have embeddings)\n")

    src = {}
    src["direct"] = direct(V)
    print(f"1. direct WordNet lemma antonyms      : {len(src['direct']):>7}")
    src["indirect"] = indirect(V) - src["direct"]
    print(f"2. indirect (satellite ~ head <-> ant) : {len(src['indirect']):>7}")
    src["morphological"] = morphological(V) - src["direct"] - src["indirect"]
    print(f"3. morphological negation (POS-checked): {len(src['morphological']):>7}")
    # 4. DERIVATIONAL TRANSPORT: DROPPED. The rule produced 'ability/flab',
    #    'ability/gamey', 'ability/gaming'. Transporting antonymy across
    #    derivational links needs the two derivations to be the same
    #    morphological process, which WordNet does not record. Rather than
    #    ship 9021 pairs of noise into the training set, it is cut.
    print(f"4. derivational transport             :  DROPPED (produced noise)")

    allp = set().union(*src.values())
    print(f"\nTOTAL canonical antonym pairs         : {len(allp):>7}")
    print(f"(was 3469 direct, freq-capped -> "
          f"{len(allp)/1735:.1f}x the canonical count)")

    words = {w for p in allp for w in p}
    print(f"words participating in antonymy       : {len(words):>7} "
          f"({len(words)/len(V):.0%} of the vocabulary)")

    print("\nsamples per source:")
    for k, v in src.items():
        s = sorted(v)[:5]
        print(f"  {k:<15} " + ", ".join(f"{a}/{b}" for a, b in s))

    tagged = {f"{a}|{b}": [k for k, v in src.items() if (a, b) in v]
              for a, b in allp}
    json.dump(sorted(map(list, allp)), open(f"{D}/antonyms_expanded.json", "w"))
    json.dump(tagged, open(f"{D}/antonyms_sources.json", "w"))
    print(f"\nwrote {D}/antonyms_expanded.json")


if __name__ == "__main__":
    main()
