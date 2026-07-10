"""Rerun with the two dictionary fixes.

1. Wordpiece tokens are self-delimiting (`##` marks continuation), so the
   `<wp>` / `</wp>` brackets cost 2 tokens per escape for nothing. Removing them
   takes the lexeme stream from 1.23x BPE tokens to 1.09x.
2. 9,938 wikitext types that were escaping are decomposable after all
   (4,370 inflections via lemminflect, 5,570 by inverting the affix rules).
   Small: escapes 7.1% -> 6.7%.

NOT done: adding the 55,273 rare types as bare lexemes. With a 16k vocab cap
those tokens never make the cut and collapse to <unk>, which discards the word
and flatters bits/char. The escape hatch exists so rare words are spelled, not
thrown away.

Correction to the record: the earlier "1.655 tokens/word" came from the PREFIX
encoder (a `</op>` per operator); the sweep used POSTFIX and actually paid 1.23x.
"""
import json, collections
import numpy as np
import torch

from lexicon.ts_lm import GPT, DEVICE, OUT, CTX, WORD_RE
from lexicon.ts_postfix import PostfixTok
from lexicon.ts_eval2 import bits_per_char
from lexicon.ts_scale import trunk_params
from lexicon.wt_scale import BPETok, corpus, train_and_loss, N_PARA

CONFIGS = [(6, 384, 6), (8, 512, 8)]


class LeanTok(PostfixTok):
    """postfix, no escape brackets, expanded decomposition."""
    name = "lexeme-lean"

    def __init__(self, texts, max_vocab=16000):
        from lexicon.ts_encode import LexemeTokenizer, load_forest
        self.lex = LexemeTokenizer()
        base_parent, base_roots = load_forest()
        exp = json.load(open("dictionary/forest_expanded.json"))
        ep = {k: tuple(v) for k, v in exp["parent"].items()}
        new = {w: ep[w] for w in ep if w not in base_parent and w not in base_roots}
        self.lex.parent = {**base_parent, **new}      # decomposition only
        self.lex.roots = set(base_roots)
        self.lex._cache = {}
        self._pc = {}
        c = collections.Counter()
        for t in texts[:20000]:
            for m in WORD_RE.findall(t):
                c.update(self.enc_word_str(m) if (m.isalpha() or "'" in m) else [f"<p:{m}>"])
        self.itos = ["<pad>", "<unk>"] + [w for w, _ in c.most_common(max_vocab)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def enc_word_str(self, w):
        if w in self._pc: return self._pc[w]
        lw = w.lower()
        if lw in self.lex.roots:
            out = [f"<lex:{lw}>"]
        elif lw in self.lex.parent:
            ops, cur = [], lw
            while cur in self.lex.parent:
                s, cur = self.lex.parent[cur]; ops.append(s)
            out = [f"<lex:{cur}>"] + [f"<op:{s}>" for s in ops]   # postfix
        else:
            out = [f"<wp:{t}>" for t in self.lex.wp.tokenize(lw)] or ["<unk>"]
        self._pc[w] = out
        return out


def main():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts = []
    for r in ds:
        t = r["text"]
        if len(t) > 200 and not t.lstrip().startswith("="): texts.append(t)
        if len(texts) >= N_PARA: break
    train_texts, eval_texts = texts[:-2000], texts[-2000:]

    toks = {"bpe": BPETok(train_texts), "lexeme-lean": LeanTok(train_texts)}
    data = {n: corpus(t, train_texts) for n, t in toks.items()}
    for n, t in toks.items():
        print(f"{n:<13} vocab {len(t.itos):>6}  tokens {len(data[n]):>12,}  "
              f"{len(data[n])/len(data['bpe']):.3f}x bpe")
    print("  (previous lexeme: 1.230x bpe)\n")

    rows = {}
    for layers, d, heads in CONFIGS:
        for name, tok in toks.items():
            torch.manual_seed(0)
            m = GPT(len(tok.itos), d=d, layers=layers, heads=heads).to(DEVICE)
            tp = trunk_params(m)
            tl = train_and_loss(m, data[name], len(tok.itos))
            bpc = bits_per_char(m, tok, eval_texts)
            rows.setdefault((layers, d), {})[name] = (bpc, tp)
            print(f"  L{layers} d{d} {name:<13} trunk {tp/1e6:6.2f}M  "
                  f"eval bits/char {bpc:.3f}", flush=True)
            del m; torch.cuda.empty_cache()

    ratio = len(data["lexeme-lean"]) / len(data["bpe"])
    print("\n" + "=" * 78)
    print(f"{'trunk':<13}{'params':>10}{'bpe':>9}{'lexeme-lean':>14}{'delta':>9}"
          f"{'token cost':>13}")
    print("-" * 78)
    for (l, d), v in rows.items():
        b, x = v["bpe"], v["lexeme-lean"]
        print(f"L{l} d{d:<8}{b[1]/1e6:>9.2f}M{b[0]:>9.3f}{x[0]:>14.3f}"
              f"{x[0]-b[0]:>+9.3f}{ratio:>12.2f}x")
    print(f"\nreference (bracketed, 1.23x tokens): L6 -0.072, L8 -0.047")
    print(f"BPE alone L6->L8: -0.078 bits/char for 2.37x params (+137% compute)")
    json.dump({f"L{l}_d{d}": {k: v[0] for k, v in r.items()} for (l, d), r in rows.items()},
              open(f"{OUT}/wt_scale2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
