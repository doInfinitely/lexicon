"""Decompose only the TAIL.

Full decomposition lengthens every sentence (+28% tokens) to buy sharing on
every word. But sharing only helps words whose own embedding row cannot be
estimated. `walked` occurs thousands of times; its row is fine. `unpretentious`
occurs three times; its row is noise.

Measured: the lexeme language wins at a 10.65M trunk (-0.194 bits/char, equal
text) and LOSES at 0.40M (+0.079). The benefit scales with trunk size, opposite
to the usual prior-helps-small-models intuition.

Hybrid: a derived form stays a whole token if it occurs >= T times in training;
below T it is expanded into `<lex:root> <op:slot>`. Sequence length stays near
word-level, the vocabulary stays small, and the factorisation lands only where
a free row would have been noise.
"""
import collections, torch, numpy as np
from lexicon.ts_lm import WordTok, GPT, build, tokenize_corpus, WORD_RE, DEVICE, CTX
from lexicon.ts_postfix import PostfixTok
from lexicon.ts_eval2 import bits_per_char
from lexicon.ts_scale import train, trunk_params


class HybridTok(PostfixTok):
    name = "hybrid"

    def __init__(self, texts, T=100, max_vocab=12000):
        from lexicon.ts_encode import LexemeTokenizer
        self.lex = LexemeTokenizer(); self._pc = {}
        self.T = T
        freq = collections.Counter()
        for t in texts:
            freq.update(m.lower() for m in WORD_RE.findall(t) if m.isalpha())
        self.freq = freq
        c = collections.Counter()
        for t in texts[:8000]:
            for m in WORD_RE.findall(t):
                c.update(self.enc_word_str(m) if (m.isalpha() or "'" in m) else [f"<p:{m}>"])
        self.itos = ["<pad>", "<unk>"] + [w for w, _ in c.most_common(max_vocab)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def enc_word_str(self, w):
        if w in self._pc: return self._pc[w]
        lw = w.lower()
        if self.freq.get(lw, 0) >= self.T:          # frequent: keep whole
            out = [f"<w:{lw}>"]
        else:
            out = super().enc_word_str(w)           # rare: decompose
        self._pc[w] = out
        return out


def main():
    train_texts, clean_eval, test_texts, held = build()
    word = WordTok(train_texts)
    full = PostfixTok(train_texts)
    print(f"\n{'tokenizer':<22}{'vocab':>8}{'tokens':>13}{'vs word':>9}")
    toks = {"word": word, "lexeme (full)": full}
    for T in (30, 100, 300):
        toks[f"hybrid T={T}"] = HybridTok(train_texts, T=T)
    data = {}
    for n, t in toks.items():
        data[n] = tokenize_corpus(t, train_texts)
        print(f"{n:<22}{len(t.itos):>8}{len(data[n]):>13,}"
              f"{len(data[n])/len(data['word']):>8.2f}x")

    print(f"\n{'trunk':<12}" + "".join(f"{n[:14]:>16}" for n in toks))
    print("-" * (12 + 16 * len(toks)))
    for layers, d, heads in ((6, 384, 6), (2, 128, 4)):
        row = f"L{layers} d{d:<7}"
        for n, tok in toks.items():
            torch.manual_seed(0)
            m = GPT(len(tok.itos), d=d, layers=layers, heads=heads).to(DEVICE)
            train(m, data[n], len(tok.itos), steps=3000)
            row += f"{bits_per_char(m, tok, clean_eval):>16.3f}"
            del m; torch.cuda.empty_cache()
        print(row, flush=True)
    print("\nequal STEPS (equal compute). word/lexeme-full reference: "
          "L6 1.160/1.062, L2 1.414/1.536")


if __name__ == "__main__":
    main()
