"""Rebuild the dictionary from MorphyNet (Wiktionary-derived) instead of hand rules.

Remy: "this is a wild bug, why are you doing it this way, can't you just grab
wiktionary?" -- correct. The hand-written INV_SUFFIX rules only see suffix
strings, so they invent `number = numb + er`, `busy = bus + y`, `final = fine + al`,
`early = earl + y`. A lexicographic source knows better.

Division of labour, by where each source is actually reliable:

  derivation   MorphyNet eng.derivational  (225k pairs, POS on both sides + affix)
               It has tropical<-tropic, contraction<-contract, and NO number<-numb.
  inflection   lemminflect (kept from the old build)
               MorphyNet's inflectional file is 65% self-rows and back-forms bogus
               lemmas (`abilities -> abilitie`); its `numb -> number ADJ|CMPR` is a
               real but unwanted reading.
  opacity      cosine in the surface prototype space
               Wiktionary decomposition is ETYMOLOGICAL. It will happily give
               `business <- busy` and `tropical <- trope`. For an LM, a decomposition
               earns its keep only if <lex:X> pools statistics across forms that are
               distributionally alike -- so cosine, not etymology, is the criterion.
               Cut derivation edges below the 95th pct of random word-pair cosine.

Also fixes the operator ORDER bug: `chain` runs outermost->innermost, so prefix
must emit `chain` and postfix must emit `reversed(chain)`. ts_encode did
`reversed(chain)` for prefix and ts_postfix reversed it AGAIN, yielding
outermost-first postfix -- `settlements = settle,plural,ment` (pluralize, then
nominalize). Affects 1.04% of wikitext occurrences.
"""
import json, collections, os
import numpy as np, torch

S = "/tmp/claude-1000/-home-remy-Code-lexicon/cccbc9d0-251e-44e9-b324-0d530c3837d0/scratchpad/"
OUT = "dictionary"
INFL_SLOTS = {"noun.plural", "verb.ger", "verb.3sg", "verb.ptcp", "verb.past",
              "adj.comp", "adj.sup"}
POS = {"N": "noun", "V": "verb", "J": "adj", "R": "adv"}


def main():
    from lexicon.ts_encode import load_forest
    old_parent, old_roots = load_forest()
    vocab = set(old_parent) | set(old_roots)

    d = torch.load("real/surface/prototypes.pt", map_location="cpu", weights_only=False)
    words = list(d)
    E = np.stack([np.asarray(d[w], dtype=np.float32) for w in words])
    idx = {w: i for i, w in enumerate(words)}
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, len(words), 40000), rng.integers(0, len(words), 40000)
    THR = float(np.percentile((En[a] * En[b]).sum(1), 95))

    def cos(x, y):
        return float(En[idx[x]] @ En[idx[y]]) if x in idx and y in idx else -1.0

    # ---- keep inflection from the old build -------------------------------
    infl = {w: v for w, v in old_parent.items() if v[0] in INFL_SLOTS}

    # ---- derivation from MorphyNet ----------------------------------------
    cand = collections.defaultdict(list)
    for ln in open(S + "eng.derivational.v1.tsv" if os.path.exists(S + "eng.derivational.v1.tsv")
                   else S + "eng.deriv.tsv"):
        f = ln.rstrip("\n").split("\t")
        if len(f) < 6: continue
        base, der, pb, pd, affix, kind = f[0].lower(), f[1].lower(), f[2], f[3], f[4], f[5]
        if base == der or base not in vocab or der not in vocab: continue
        if der in infl: continue                       # inflection wins
        slot = f"{'suf' if kind == 'suffix' else 'pre'}.{affix}"
        cand[der].append((cos(der, base), slot, base, pb, pd))

    deriv, cut_opaque = {}, []
    for der, cs in cand.items():
        c, slot, base, pb, pd = max(cs)                # best-pooling parent
        (deriv.__setitem__(der, (slot, base)) if c >= THR
         else cut_opaque.append((der, base, slot, c)))

    # ---- prune rare operators ---------------------------------------------
    # An operator used by k words costs one vocab slot and turns those k words
    # from 1 token into 2. Below k=MIN_OP that is pure loss: no pooling benefit,
    # strictly more tokens. `suf.wick` (gatwick) helps nobody.
    MIN_OP = 20
    use = collections.Counter(s for s, _ in deriv.values())
    rare = {w for w, (s, _) in deriv.items() if use[s] < MIN_OP}
    n_rare_ops = sum(1 for s, c in use.items() if c < MIN_OP)
    for w in rare: del deriv[w]

    parent = {**infl, **deriv}

    # ---- break cycles ------------------------------------------------------
    def root_of(w):
        seen, cur = set(), w
        while cur in parent:
            if cur in seen: return None
            seen.add(cur); cur = parent[cur][1]
        return cur
    bad = [w for w in list(parent) if root_of(w) is None]
    for w in bad: del parent[w]
    # every non-derived word is a root
    roots = sorted(vocab - set(parent))

    json.dump({"roots": roots, "parent": {k: list(v) for k, v in parent.items()}},
              open(f"{OUT}/forest_v2.json", "w"))

    print(f"cosine threshold (random-pair p95): {THR:.3f}\n")
    print(f"inflection edges kept        {len(infl):>7,}")
    print(f"derivation edges (morphynet) {len(deriv):>7,}   (old hand-rule: 3,688)")
    print(f"  cut as opaque (cos<thr)    {len(cut_opaque):>7,}")
    print(f"  cut, operator used <{MIN_OP}    {len(rare):>7,}  ({n_rare_ops} operators)")
    print(f"cycles broken                {len(bad):>7,}")
    print(f"\nroots {len(roots):,}   derived {len(parent):,}   total {len(roots)+len(parent):,}")

    print("\nthe words my rules got wrong:")
    for w in ["number", "numbers", "busy", "business", "early", "earlier",
              "final", "finally", "tropical", "nasal", "contraction"]:
        if w in parent:
            s, b = parent[w]; print(f"   {w:<13} <op:{s}>(<lex:{b}>)   cos {cos(w,b):+.3f}")
        else:
            print(f"   {w:<13} ROOT")

    print("\ncut as opaque (morphology real, meaning drifted) -- 12 of "
          f"{len(cut_opaque)}:")
    for w, b, s, c in sorted(cut_opaque, key=lambda x: x[3])[:12]:
        print(f"   {w:<15} <op:{s}>(<lex:{b}>)  cos {c:+.3f}")

    ops = collections.Counter(s for s, _ in parent.values())
    print(f"\noperator inventory: {len(ops)} (was 27)")
    for s, c in ops.most_common(18):
        print(f"   <op:{s}>".ljust(24) + f"{c:>6,}")


if __name__ == "__main__":
    main()
