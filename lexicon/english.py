"""Build a real English lexicon and its relation graph.

BATS gives 2768 words and 2000 pairs. That is far too small to ask whether a
region of meaning-space is "unlexicalized": with 2% of English present, empty
space is guaranteed. So we take the vocabulary to be as much of English as we
can defend, and mine the relations at the same scale.

Vocabulary : WordNet single-word lemmas, restricted to alphabetic forms that
             actually occur in the language (Zipf frequency >= MIN_ZIPF).
Relations  : WordNet gives the lexicographic and encyclopedic ones at scale
             (hypernym, hyponym, meronym x3, holonym x3, antonym, similar_to,
             entailment, cause, derivationally_related).
             lemminflect gives the inflectional morphology (plural, past,
             gerund, 3rd person singular, comparative, superlative).
             Prefix/suffix rules give derivational morphology, but only when
             BOTH forms are attested in the vocabulary -- no invented words.
"""
import json, os, re, collections
from wordfreq import zipf_frequency

MIN_ZIPF = 2.0          # ~ occurs at least once per 100M words
OUT = "real/english"

WN_RELATIONS = {
    "hypernym": lambda s: s.hypernyms(),
    "hyponym": lambda s: s.hyponyms(),
    # NLTK keeps named-entity links in a SEPARATE method from hypernyms().
    # Omitting these made 1,628 words (34% of the apparent "orphans") look
    # unrelated: bernini->sculptor, truman->president, salk->virologist.
    # Note the fan-in is 10.07, so it derives almost nothing -- but the words
    # are not orphans.
    "instance_hypernym": lambda s: s.instance_hypernyms(),
    "instance_hyponym": lambda s: s.instance_hyponyms(),
    "member_meronym": lambda s: s.member_meronyms(),
    "part_meronym": lambda s: s.part_meronyms(),
    "substance_meronym": lambda s: s.substance_meronyms(),
    "member_holonym": lambda s: s.member_holonyms(),
    "part_holonym": lambda s: s.part_holonyms(),
    "substance_holonym": lambda s: s.substance_holonyms(),
    "entailment": lambda s: s.entailments(),
    "cause": lambda s: s.causes(),
    "similar_to": lambda s: s.similar_tos(),
    "attribute": lambda s: s.attributes(),
}

# derivational rules: (name, transform, inverse-check). Applied only when both
# the base and the derived form are attested in the vocabulary.
DERIV_RULES = [
    ("un_adj",     lambda w: "un" + w),
    ("re_verb",    lambda w: "re" + w),
    ("over_adj",   lambda w: "over" + w),
    ("adj_ly",     lambda w: w + "ly"),
    ("adj_ness",   lambda w: (w[:-1] + "iness") if w.endswith("y") else w + "ness"),
    ("noun_less",  lambda w: w + "less"),
    ("verb_able",  lambda w: (w[:-1] + "able") if w.endswith("e") else w + "able"),
    ("verb_er",    lambda w: (w + "r") if w.endswith("e") else w + "er"),
    ("verb_tion",  lambda w: (w[:-1] + "ion") if w.endswith("e") else w + "tion"),
    ("verb_ment",  lambda w: w + "ment"),
    ("noun_ful",   lambda w: w + "ful"),
    ("noun_ish",   lambda w: w + "ish"),
]


def build_vocab():
    from nltk.corpus import wordnet as wn
    lemmas = set()
    for s in wn.all_synsets():
        for l in s.lemmas():
            n = l.name().lower()
            if n.isalpha() and 2 <= len(n) <= 24:
                lemmas.add(n)
    print(f"wordnet alphabetic single-word lemmas: {len(lemmas)}")
    vocab = sorted(w for w in lemmas if zipf_frequency(w, "en") >= MIN_ZIPF)
    print(f"after frequency filter (zipf >= {MIN_ZIPF}): {len(vocab)}")
    return vocab


def wordnet_pairs(vocab):
    from nltk.corpus import wordnet as wn
    V = set(vocab)
    pairs = collections.defaultdict(set)

    def names(syn):
        return [l.name().lower() for l in syn.lemmas()
                if l.name().lower() in V and l.name().isalpha()]

    for s in wn.all_synsets():
        srcs = names(s)
        if not srcs:
            continue
        for rel, fn in WN_RELATIONS.items():
            for tgt_syn in fn(s):
                tgts = names(tgt_syn)
                for a in srcs[:2]:            # cap to limit combinatorics
                    for b in tgts[:2]:
                        if a != b:
                            pairs[rel].add((a, b))
        # antonym and derivational forms live on lemmas, not synsets
        for l in s.lemmas():
            a = l.name().lower()
            if a not in V:
                continue
            for ant in l.antonyms():
                b = ant.name().lower()
                if b in V and a != b:
                    pairs["antonym"].add((a, b))
            for d in l.derivationally_related_forms():
                b = d.name().lower()
                if b in V and a != b:
                    pairs["derivationally_related"].add((a, b))
    return {k: sorted(v) for k, v in pairs.items()}


def morphology_pairs(vocab):
    from lemminflect import getInflection
    V = set(vocab)
    pairs = collections.defaultdict(set)
    TAGS = {"noun_plural": ("NNS", "NOUN"), "verb_3pSg": ("VBZ", "VERB"),
            "verb_Ving": ("VBG", "VERB"), "verb_Ved": ("VBD", "VERB"),
            "adj_comparative": ("JJR", "ADJ"), "adj_superlative": ("JJS", "ADJ")}
    for w in vocab:
        for rel, (tag, _) in TAGS.items():
            try:
                infl = getInflection(w, tag=tag)
            except Exception:
                continue
            for f in (infl or []):
                f = f.lower()
                if f in V and f != w:
                    pairs[rel].add((w, f))
    return {k: sorted(v) for k, v in pairs.items()}


def derivational_pairs(vocab):
    V = set(vocab)
    pairs = collections.defaultdict(set)
    for w in vocab:
        for name, fn in DERIV_RULES:
            try:
                d = fn(w)
            except Exception:
                continue
            if d in V and d != w:
                pairs[name].add((w, d))
    return {k: sorted(v) for k, v in pairs.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    vocab = build_vocab()

    print("mining wordnet relations...")
    wnp = wordnet_pairs(vocab)
    print("mining inflectional morphology...")
    mor = morphology_pairs(vocab)
    print("mining derivational morphology...")
    der = derivational_pairs(vocab)

    rels = {}
    for src, d in (("lex", wnp), ("infl", mor), ("deriv", der)):
        for k, v in d.items():
            rels[f"{src}:{k}"] = v

    # keep only words that participate in at least one relation, plus keep the
    # full vocabulary list separately (we still want to ask what is unreachable)
    used = {w for pl in rels.values() for p in pl for w in p}
    print(f"\nvocabulary          : {len(vocab)}")
    print(f"words in >=1 relation: {len(used)}")
    print(f"relations           : {len(rels)}")
    tot = sum(len(v) for v in rels.values())
    print(f"total pairs         : {tot}")
    print("\nlargest relations:")
    for k, v in sorted(rels.items(), key=lambda kv: -len(kv[1]))[:14]:
        print(f"  {k:<34}{len(v):>8}")

    json.dump(vocab, open(f"{OUT}/vocab.json", "w"))
    json.dump({k: v for k, v in rels.items()}, open(f"{OUT}/relations.json", "w"))
    print(f"\nwrote {OUT}/vocab.json, {OUT}/relations.json")


if __name__ == "__main__":
    main()
