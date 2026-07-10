"""Expand dictionary coverage so the escape hatch stops dominating the cost.

On wikitext, 7.1% of word occurrences fall through to `<wp> ... </wp>`, and each
escape costs 2 + n tokens. The lexeme stream is 1.23x BPE tokens (NOT the 1.655x
this docstring used to claim -- that was the prefix encoder, which emits `</op>`;
the sweep runs postfix). The cost looks like a DICTIONARY problem: our surface
vocabulary is 51k, and wikitext has 72k types in a 2.3M-token sample.

So: take every wikitext word type occurring >= MIN_FREQ, and give it a home.

  OUTCOME (see wt_scale2.py): this barely helps. Only 9,938 of 65,213 missing
  types are decomposable; escapes go 7.1% -> 6.7%. And step 3 below is a TRAP --
  under the LM's 16k vocab cap those 55,273 bare lexemes never make the cut and
  collapse to <unk>, which discards the word and flatters bits/char. Use the
  decomposition entries only; keep the escape hatch for the tail.

  1. inflection      lemminflect's lemmatiser: `resigned` -> (verb.past, resign)
  2. derivation      invert the suffix/prefix rules (allomorphy handled), root
                     must itself be attested
  3. bare lexeme     anything else frequent enough gets its own token (1 token,
                     the same cost as a word-level model pays)
  4. escape          only the genuine tail

Fairness note: this fits the tokenizer to the corpus, which is exactly what BPE
does (GPT-2's merges were learned on WebText). The comparison stays honest as
long as we do not also fit the LM's vocabulary cap differently.
"""
import json, collections, os
import numpy as np
from nltk.corpus import wordnet as wn

from lexicon.cook import SUFFIX, PREFIX, _e_drop, _y_to_i, _double
from lexicon.ts_encode import load_forest, WORD_RE

MIN_FREQ = 3
N_PARA = 200000
OUT = "dictionary"

# derived -> base candidates (inverse of the cook rules)
INV_SUFFIX = {
    "noun.quality":  lambda w: [w[:-4], w[:-4] + "y"] if w.endswith("ness") else [],
    "adv":           lambda w: [w[:-2], w[:-3] + "y", w[:-2] + "e"] if w.endswith("ly") else [],
    "noun.agent":    lambda w: [w[:-2], w[:-1], w[:-3]] if w.endswith("er") else [],
    "noun.action2":  lambda w: [w[:-4]] if w.endswith("ment") else [],
    "adj.able":      lambda w: [w[:-4], w[:-4] + "e"] if w.endswith("able") else [],
    "noun.action":   lambda w: [w[:-5], w[:-5] + "e", w[:-3]] if w.endswith("ation") else
                               ([w[:-3], w[:-3] + "e"] if w.endswith("ion") else []),
    "noun.ist":      lambda w: [w[:-3], w[:-3] + "e"] if w.endswith("ist") else [],
    "noun.ism":      lambda w: [w[:-3], w[:-3] + "e"] if w.endswith("ism") else [],
    "adj.ic":        lambda w: [w[:-2], w[:-2] + "e"] if w.endswith("ic") else [],
    "adj.al":        lambda w: [w[:-2], w[:-2] + "e"] if w.endswith("al") else [],
    "adj.ous":       lambda w: [w[:-3], w[:-3] + "e"] if w.endswith("ous") else [],
    "adj.ful":       lambda w: [w[:-3], w[:-3] + "y"] if w.endswith("ful") else [],
    "adj.less":      lambda w: [w[:-4], w[:-4] + "y"] if w.endswith("less") else [],
    "adj.y":         lambda w: [w[:-1], w[:-2]] if w.endswith("y") else [],
    "verb.ize":      lambda w: [w[:-3], w[:-3] + "e"] if w.endswith("ize") else [],
    "noun.quality2": lambda w: [w[:-3], w[:-3] + "e"] if w.endswith("ity") else [],
}
INV_PREFIX = {f"pre.{p}": (lambda w, p=p: [w[len(p):]] if w.startswith(p) and
                           len(w) > len(p) + 2 else []) for p in
              ("un", "in", "im", "non", "dis", "mis", "re", "pre", "over",
               "under", "sub", "super", "anti", "inter", "counter", "semi")}

INFL_TAG = {"NNS": "noun.plural", "VBZ": "verb.3sg", "VBG": "verb.ger",
            "VBD": "verb.past", "VBN": "verb.ptcp", "JJR": "adj.comp",
            "JJS": "adj.sup"}


def main():
    from datasets import load_dataset
    from lemminflect import getAllLemmas

    parent, roots = load_forest()
    known = set(roots) | set(parent)
    print(f"current dictionary: {len(roots):,} roots, {len(parent):,} derived, "
          f"{len(known):,} words total")

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts, n = [], 0
    for r in ds:
        t = r["text"]
        if len(t) > 200 and not t.lstrip().startswith("="):
            texts.append(t); n += 1
        if n >= N_PARA:
            break
    cnt = collections.Counter()
    for t in texts:
        cnt.update(m.lower() for m in WORD_RE.findall(t) if m.isalpha())
    print(f"wikitext: {sum(cnt.values()):,} word tokens, {len(cnt):,} types")

    missing = [w for w, c in cnt.items() if c >= MIN_FREQ and w not in known]
    print(f"missing types with freq >= {MIN_FREQ}: {len(missing):,} "
          f"({sum(cnt[w] for w in missing):,} occurrences)\n")

    new_parent, new_roots = {}, set()
    stats = collections.Counter()
    attested = known | {w for w, c in cnt.items() if c >= MIN_FREQ}

    for w in sorted(missing, key=lambda x: -cnt[x]):
        # 1) inflection
        placed = False
        try:
            lem = getAllLemmas(w)
        except Exception:
            lem = {}
        for tag, slot in INFL_TAG.items():
            pos = {"NNS": "NOUN", "VBZ": "VERB", "VBG": "VERB", "VBD": "VERB",
                   "VBN": "VERB", "JJR": "ADJ", "JJS": "ADJ"}[tag]
            for base in lem.get(pos, ()):
                b = base.lower()
                if b != w and b in attested:
                    new_parent[w] = (slot, b); stats["inflection"] += 1
                    placed = True; break
            if placed:
                break
        if placed:
            continue
        # 2) derivation (invert the rules)
        for slot, fn in list(INV_SUFFIX.items()) + list(INV_PREFIX.items()):
            for b in fn(w):
                if len(b) > 2 and b != w and b in attested:
                    new_parent[w] = (slot.replace("pre.", "pre_"), b)
                    stats["derivation"] += 1; placed = True; break
            if placed:
                break
        if placed:
            continue
        # 3) bare lexeme
        new_roots.add(w); stats["bare lexeme"] += 1

    # cycles: a new derived word whose root is itself newly derived is fine,
    # but never let a chain loop
    def root_of(w, seen=None):
        seen = seen or set()
        cur = w
        while cur in new_parent or cur in parent:
            if cur in seen:
                return None
            seen.add(cur)
            cur = (new_parent.get(cur) or parent.get(cur))[1]
        return cur
    bad = [w for w in list(new_parent) if root_of(w) is None]
    for w in bad:
        del new_parent[w]; new_roots.add(w)
    print(f"placed: " + ", ".join(f"{k} {v:,}" for k, v in stats.most_common()))
    print(f"cycles broken: {len(bad)}")

    parent.update(new_parent)
    roots = set(roots) | new_roots
    json.dump({"roots": sorted(roots),
               "parent": {k: list(v) for k, v in parent.items()}},
              open(f"{OUT}/forest_expanded.json", "w"))
    print(f"\nexpanded dictionary: {len(roots):,} roots, {len(parent):,} derived, "
          f"{len(roots)+len(parent):,} words")
    print(f"wrote {OUT}/forest_expanded.json")

    # what does it cost now?
    covered_occ = sum(cnt[w] for w in cnt if w in roots or w in parent)
    total_occ = sum(cnt.values())
    print(f"\nwikitext occurrences now expressible without escape: "
          f"{covered_occ/total_occ:.1%}  (was 92.9%)")


if __name__ == "__main__":
    main()
