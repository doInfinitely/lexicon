"""Encode TinyStories into the lexeme language, and count what it costs.

Three tokenizations of the same text, so that anything we compare later is
comparable:

  word      one token per surface word (+ punctuation). The ceiling on
            compactness, and the baseline that CANNOT generalise: a word form
            it never saw is simply <unk>.
  bpe       GPT-2 byte-pair encoding. What TinyStories actually used.
  lexeme    our language: <lex:root>, bracketed operators, wordpiece escape.

The lexeme stream is LONGER (operators cost tokens). That is the price. The
thing it buys, and the only reason to pay it, is that a surface form the model
never saw is still expressible from tokens it has seen:

    walkers  ->  <op:noun.plural> <op:noun.agent> <lex:walk> </op> </op>

Reported: tokens per character for each scheme (the only length measure that
does not depend on the tokenizer), and the fraction of word occurrences that
fall back to wordpieces.
"""
import json, collections, re, os
import numpy as np

DICT = "dictionary"
PUNCT = list(".,!?\"';:-()")
WORD_RE = re.compile(r"[a-zA-Z']+|[^\sa-zA-Z']")


def load_forest():
    f = json.load(open(f"{DICT}/forest.json"))
    parent = {k: tuple(v) for k, v in f["parent"].items()}
    roots = set(f["roots"])
    return parent, roots


class LexemeTokenizer:
    def __init__(self):
        self.parent, self.roots = load_forest()
        from transformers import AutoTokenizer
        self.wp = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self._cache = {}

    def encode_word(self, w):
        if w in self._cache:
            return self._cache[w]
        lw = w.lower()
        if lw in self.roots:
            out = [f"<lex:{lw}>"]
        elif lw in self.parent:
            out = []
            cur = lw
            chain = []
            while cur in self.parent:
                slot, cur = self.parent[cur]
                chain.append(slot)
            # chain runs outermost -> innermost (settlements: plural, then ment).
            # Prefix nests f(g(x)), so the OUTERMOST operator comes first.
            # This used to emit reversed(chain), and PostfixTok then reversed it
            # again -- yielding outermost-first postfix, i.e. `settle, plural,
            # ment` = "pluralize settle, then nominalize". Backwards.
            out = [f"<op:{s}>" for s in chain] + [f"<lex:{cur}>"] \
                + ["</op>"] * len(chain)
        else:
            out = ["<wp>"] + [f"<wp:{t}>" for t in self.wp.tokenize(lw)] + ["</wp>"]
        self._cache[w] = out
        return out

    def encode(self, text):
        toks = []
        for m in WORD_RE.findall(text):
            if m.isalpha() or "'" in m:
                toks += self.encode_word(m)
            else:
                toks.append(f"<p:{m}>")
        return toks


def main():
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    ds = load_dataset("roneneldan/TinyStories", split="train")
    n = 20000
    texts = [ds[i]["text"] for i in range(n)]
    chars = sum(len(t) for t in texts)

    lex = LexemeTokenizer()
    bpe = GPT2TokenizerFast.from_pretrained("gpt2")

    n_lex_tok = 0
    kinds = collections.Counter()
    word_occ = 0
    oov_occ = 0
    vocab_used = set()
    for t in texts:
        for m in WORD_RE.findall(t):
            if not (m.isalpha() or "'" in m):
                continue
            word_occ += 1
            e = lex.encode_word(m)
            vocab_used.update(e)
            if e[0] == "<wp>":
                oov_occ += 1
                kinds["wordpiece escape"] += 1
            elif e[0].startswith("<lex:"):
                kinds["bare lexeme"] += 1
            else:
                kinds[f"operator depth {sum(1 for x in e if x.startswith('<op:'))}"] += 1
        n_lex_tok += len(lex.encode(t))

    n_word_tok = sum(len(WORD_RE.findall(t)) for t in texts)
    n_bpe_tok = sum(len(bpe.encode(t)) for t in texts)

    print(f"{n} TinyStories, {chars:,} characters, {word_occ:,} word occurrences\n")
    print(f"{'scheme':<18}{'tokens':>12}{'tokens/char':>14}{'vs word':>10}")
    print("-" * 56)
    for name, k in (("word", n_word_tok), ("bpe (gpt2)", n_bpe_tok),
                    ("lexeme language", n_lex_tok)):
        print(f"{name:<18}{k:>12,}{k/chars:>14.4f}{k/n_word_tok:>10.2f}x")

    print(f"\nhow every word occurrence is expressed:")
    for k, v in kinds.most_common():
        print(f"   {k:<24}{v:>10,}  ({v/word_occ:.1%})")
    print(f"\nOOV rate (needs wordpiece escape): {oov_occ/word_occ:.2%} of tokens")

    types = collections.Counter()
    seen = set()
    for t in texts:
        for m in WORD_RE.findall(t):
            if (m.isalpha() or "'" in m) and m.lower() not in seen:
                seen.add(m.lower())
                e = lex.encode_word(m)
                types["oov" if e[0] == "<wp>" else "covered"] += 1
    print(f"distinct word types: {sum(types.values()):,}   "
          f"covered {types['covered']:,}  oov {types['oov']:,} "
          f"({types['oov']/sum(types.values()):.1%})")

    lex_vocab = {t for t in vocab_used}
    print(f"\nlexeme-language tokens actually used on this sample: {len(lex_vocab):,}")
    json.dump(sorted(lex_vocab), open(f"{DICT}/ts_tokens_used.json", "w"))


if __name__ == "__main__":
    main()
