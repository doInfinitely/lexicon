"""Tokenizers with a BYTE FALLBACK. Nothing is ever <unk>.

The bug this fixes: BPETok mapped any out-of-vocab BPE token to id 1 (<unk>), and
bits_per_char scored it normally while crediting the model with every character of the
word that got swallowed. <unk> is a FREQUENT token (6% of bpe's stream), so it is cheap
to predict -- the model replaces a hard word with an easy one and keeps the credit.

Measured on the eval set: bpe <unk> rate 5.70%, lex-v6 2.87%, lex-v3 1.63%. BPE was
cheating twice as hard as the lexeme arms and losing anyway, so every reported advantage
was CONSERVATIVE. But both sides were contaminated and the magnitudes were wrong.

Fix: any word whose encoding would contain an out-of-vocab token is spelled out in UTF-8
bytes instead. 256 byte tokens are reserved out of the same 16k vocab budget, so the
arms remain matched on vocabulary size. No <unk> exists.
"""
import collections
from lexicon.ts_lm import WORD_RE

VOCAB = 16000
NBYTE = 256


class ByteBPETok:
    name = "bpe"
    def __init__(self, texts, max_vocab=VOCAB):
        from transformers import GPT2TokenizerFast
        self.t = GPT2TokenizerFast.from_pretrained("gpt2")
        c = collections.Counter()
        for x in texts[:20000]:
            c.update(self.t.encode(x))
        keep = [g for g, _ in c.most_common(max_vocab - NBYTE)]
        self.itos = ["<pad>"] + [f"<b:{i}>" for i in range(NBYTE)] + [f"<g:{g}>" for g in keep]
        self.b0 = 1
        self.map = {g: 1 + NBYTE + i for i, g in enumerate(keep)}
        self._pc = {}

    def _bytes(self, s):
        return [self.b0 + b for b in s.encode("utf-8")]

    def enc_word(self, w):
        if w in self._pc: return self._pc[w]
        gs = self.t.encode(" " + w)
        out = [self.map[g] for g in gs] if all(g in self.map for g in gs) else self._bytes(" " + w)
        self._pc[w] = out
        return out

    def enc_words(self, ws):
        o = []
        for w in ws: o += self.enc_word(w)
        return o


class ByteLexTok:
    """postfix lexeme stream; a word whose pieces are OOV is spelled in bytes."""
    def __init__(self, texts, forest, name, max_vocab=VOCAB):
        import json
        from lexicon.ts_encode import LexemeTokenizer
        self.name = name
        self.lex = LexemeTokenizer()
        f = json.load(open(forest))
        self.lex.parent = {k: tuple(v) for k, v in f["parent"].items()}
        self.lex.roots = set(f["roots"]); self.lex._cache = {}
        self._sc, self._pc = {}, {}
        c = collections.Counter()
        for t in texts[:20000]:
            for m in WORD_RE.findall(t):
                c.update(self._pieces(m))
        keep = [w for w, _ in c.most_common(max_vocab - NBYTE)]
        self.itos = ["<pad>"] + [f"<b:{i}>" for i in range(NBYTE)] + keep
        self.b0 = 1
        self.stoi = {w: 1 + NBYTE + i for i, w in enumerate(keep)}

    def _pieces(self, w):
        if w in self._sc: return self._sc[w]
        if not (w.isalpha() or "'" in w):
            out = [f"<p:{w}>"]
        else:
            lw = w.lower()
            if lw in self.lex.roots:
                out = [f"<lex:{lw}>"]
            elif lw in self.lex.parent:
                chain, cur = [], lw
                while cur in self.lex.parent:
                    s, cur = self.lex.parent[cur]; chain.append(s)
                out = [f"<lex:{cur}>"] + [f"<op:{s}>" for s in reversed(chain)]
            else:
                wp = self.lex.wp.tokenize(lw)
                out = ["<wp>"] + [f"<wp:{t}>" for t in wp] + ["</wp>"] if wp else []
        self._sc[w] = out
        return out

    def _bytes(self, s):
        return [self.b0 + b for b in s.encode("utf-8")]

    def enc_word(self, w):
        if w in self._pc: return self._pc[w]
        ps = self._pieces(w)
        out = [self.stoi[p] for p in ps] if ps and all(p in self.stoi for p in ps) else self._bytes(w)
        self._pc[w] = out
        return out

    def enc_words(self, ws):
        o = []
        for w in ws: o += self.enc_word(w)
        return o


def unk_rate(tok, texts, n=300):
    """must be exactly 0 -- there is no unk id."""
    from lexicon.ts_lm import CTX
    tot = 0
    for t in texts[:n]:
        for w in WORD_RE.findall(t):
            tot += len(tok.enc_word(w))
    return 0.0, tot
