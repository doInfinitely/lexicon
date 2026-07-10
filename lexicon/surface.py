"""Rebuild on a SURFACE-FORM vocabulary. The lemma vocabulary was the bug.

WordNet's lemma list is what a lemmatiser leaves behind: `cats`, `dogs`,
`books`, `children`, `women` are all absent, though their Zipf frequencies are
4.5-5.6. Measuring "is English compressible" on that list deletes the
compressible forms before the measurement. `infl:noun_plural` had 639 pairs
against 38,142 words.

Here the vocabulary is the top surface forms of English by frequency, so a
noun and its plural, a verb and its four inflections, and an adjective and its
comparative all appear. Relations:

  inflectional   lemminflect, applied to every vocabulary word treated as a
                 base; a pair is kept only when BOTH forms are in the vocab.
  derivational   the 46 cooked rules (suffix/prefix with allomorphy).
  lexicographic  WordNet, restricted to vocabulary words.

Nothing invented; every pair requires both surface forms to be attested.
"""
import json, os, collections
from wordfreq import top_n_list, zipf_frequency

OUT = "real/surface"
N_TOP = 60000
MIN_ZIPF = 2.5


def build_vocab():
    words = top_n_list("en", N_TOP)
    v = [w for w in words if w.isalpha() and 2 <= len(w) <= 24
         and zipf_frequency(w, "en") >= MIN_ZIPF]
    return sorted(set(v))


def inflectional(vocab):
    from lemminflect import getInflection
    V = set(vocab)
    rels = collections.defaultdict(set)
    TAGS = {"noun_plural": "NNS", "verb_3pSg": "VBZ", "verb_Ving": "VBG",
            "verb_Ved": "VBD", "verb_Ven": "VBN",
            "adj_comparative": "JJR", "adj_superlative": "JJS"}
    for w in vocab:
        for rel, tag in TAGS.items():
            try:
                forms = getInflection(w, tag=tag)
            except Exception:
                continue
            for f in (forms or []):
                f = f.lower()
                if f in V and f != w:
                    rels[f"infl:{rel}"].add((w, f))
    return rels


def lexicographic(vocab):
    from nltk.corpus import wordnet as wn
    V = set(vocab)
    WN = {"hypernym": lambda s: s.hypernyms(),
          "hyponym": lambda s: s.hyponyms(),
          "instance_hypernym": lambda s: s.instance_hypernyms(),
          "part_meronym": lambda s: s.part_meronyms(),
          "part_holonym": lambda s: s.part_holonyms(),
          "member_meronym": lambda s: s.member_meronyms(),
          "substance_meronym": lambda s: s.substance_meronyms(),
          "similar_to": lambda s: s.similar_tos(),
          "attribute": lambda s: s.attributes(),
          "entailment": lambda s: s.entailments(),
          "cause": lambda s: s.causes()}
    rels = collections.defaultdict(set)
    for s in wn.all_synsets():
        names = [l.name().lower() for l in s.lemmas() if l.name().lower() in V]
        if not names:
            continue
        for rel, fn in WN.items():
            for t in fn(s):
                tn = [l.name().lower() for l in t.lemmas() if l.name().lower() in V]
                for a in names[:2]:
                    for b in tn[:2]:
                        if a != b:
                            rels[f"lex:{rel}"].add((a, b))
        for l in s.lemmas():
            a = l.name().lower()
            if a not in V:
                continue
            for ant in l.antonyms():
                b = ant.name().lower()
                if b in V and a != b:
                    rels["lex:antonym"].add((a, b))
            for d in l.derivationally_related_forms():
                b = d.name().lower()
                if b in V and a != b:
                    rels["lex:derivationally_related"].add((a, b))
    return rels


def derivational(vocab):
    from lexicon.cook import SUFFIX, PREFIX
    V = set(vocab)
    rels = collections.defaultdict(set)
    for w in vocab:
        if len(w) < 3:
            continue
        for name, fn in SUFFIX.items():
            for cand in fn(w):
                if cand in V and cand != w and len(cand) > len(w):
                    rels[f"cook:suf_{name}"].add((w, cand)); break
        for name, fn in PREFIX.items():
            for cand in fn(w):
                if cand in V and cand != w:
                    rels[f"cook:pre_{name}"].add((w, cand)); break
    return rels


def main():
    os.makedirs(OUT, exist_ok=True)
    vocab = build_vocab()
    print(f"surface vocabulary: {len(vocab)} words (zipf >= {MIN_ZIPF})")
    for probe in ("cats", "dogs", "children", "women", "running", "ran", "bigger"):
        print(f"   {probe:<10} present: {probe in set(vocab)}")

    print("\nmining relations...")
    rels = {}
    rels.update(inflectional(vocab))
    rels.update(derivational(vocab))
    rels.update(lexicographic(vocab))
    rels = {k: sorted(v) for k, v in rels.items() if len(v) >= 12}

    print(f"\n{'relation':<30}{'pairs':>8}{'fan-in':>9}")
    tot = 0
    for r in sorted(rels, key=lambda r: -len(rels[r]))[:18]:
        tg = collections.Counter(b for _, b in rels[r])
        print(f"{r:<30}{len(rels[r]):>8}{len(rels[r])/len(tg):>9.2f}")
    tot = sum(len(v) for v in rels.values())
    print(f"\n{len(rels)} relations, {tot} pairs")

    det = set()
    for r, pl in rels.items():
        tg = collections.Counter(b for _, b in pl)
        if len(pl) / max(len(tg), 1) <= 1.2:
            det |= set(tg)
    V = len(vocab)
    print(f"\nwords determined by a fan-in<=1.2 relation: {len(det)} ({len(det)/V:.1%})")
    print(f"CEILING compression: {V/(V-len(det)):.3f}x")
    print(f"  (lemma vocabulary gave 28.9% and 1.617x)")

    json.dump(vocab, open(f"{OUT}/vocab.json", "w"))
    json.dump({k: [list(p) for p in v] for k, v in rels.items()},
              open(f"{OUT}/relations.json", "w"))
    print(f"\nwrote {OUT}/vocab.json, {OUT}/relations.json")


if __name__ == "__main__":
    main()
